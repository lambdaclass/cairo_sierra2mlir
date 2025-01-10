//! # Compilation process
//!
//! A Sierra program is compiled one function at a time. Each function has a pre-entry block that
//! will be ran only once, even in tail-recursive functions. All libfuncs are intended to place
//! their stack-allocating operations there so as to not grow the stack when recursing.
//!
//! After the pre-entry block, there is an entry block, which is in charge of preparing the first
//! statement's arguments and jumping into it. From here on, all the statements's
//! [builders](crate::libfuncs::LibfuncBuilder) are invoked. Every libfunc builder must end its
//! execution calling a branch function from the helper, which will generate the operations required
//! to jump to next statements. This simplifies the branching design, especially when a libfunc has
//! multiple target branches.
//!
//! > Note: Libfunc builders must have a branching operation out into each possible branch, even if
//! >  it's unreachable. This is required to keep the state consistent. More on that later.
//!
//! Some statements do require a special landing block. Those are the ones which are the branching
//! target of more than a single statement. In other words, if a statement can be reached (directly)
//! from more than a single place, it needs a landing block.
//!
//! The landing blocks are in charge of synchronizing the Sierra state. The state is just a
//! dictionary mapping variable ids to their values. Since the values can come from a single branch,
//! this landing block is required.
//!
//! In order to generate the libfuncs's blocks, all the libfunc's entry blocks are required. That is
//! why they are generated all beforehand. The order in which they are generated follows a
//! breadth-first ordering; that is, the compiler uses a [BFS algorithm]. This algorithm should
//! generate the libfuncs in the same order as they appear in Sierra. As expected, the algorithm
//! forks the path each time a branching libfunc is found, which dies once a return statement is
//! detected.
//!
//! ## Function nomenclature transforms
//!
//! When compiling from Cairo, or from a Sierra source with debug information (the `-r` flag on
//! `cairo-compile`), those identifiers are the function's exported symbol. However, Sierra programs
//! are not required to contain that information. In those cases, the
//! (`generate_function_name`)[generate_function_name] will generate a new symbol name based on its
//! function id.
//!
//! ## Tail-recursive functions
//!
//! Part of the tail-recursion handling algorithm is implemented here, but tail-recursive functions
//! are better explained in [their metadata section](crate::metadata::tail_recursion).
//!
//! [BFS algorithm]: https://en.wikipedia.org/wiki/Breadth-first_search

use crate::{
    debug::libfunc_to_name,
    error::{panic::ToNativeAssertError, Error},
    libfuncs::{BranchArg, LibfuncBuilder, LibfuncHelper},
    metadata::{
        gas::{GasCost, GasMetadata},
        tail_recursion::TailRecursionMeta,
        MetadataStorage,
    },
    native_panic,
    types::TypeBuilder,
    utils::{generate_function_name, BlockExt},
};
use bumpalo::Bump;
use cairo_lang_sierra::{
    edit_state,
    extensions::{
        core::{CoreConcreteLibfunc, CoreLibfunc, CoreType},
        ConcreteLibfunc,
    },
    ids::{ConcreteTypeId, VarId},
    program::{Function, Invocation, Program, Statement, StatementIdx},
    program_registry::ProgramRegistry,
};
use cairo_lang_utils::ordered_hash_map::OrderedHashMap;
use itertools::Itertools;
use melior::{
    dialect::{
        arith::CmpiPredicate,
        cf, func, index,
        llvm::{self, LoadStoreOptions},
        memref,
    },
    ir::{
        attribute::{
            DenseI64ArrayAttribute, FlatSymbolRefAttribute, IntegerAttribute, StringAttribute,
            TypeAttribute,
        },
        operation::OperationBuilder,
        r#type::{FunctionType, IntegerType, MemRefType},
        Attribute, AttributeLike, Block, BlockRef, Identifier, Location, Module, Region, Type,
        Value,
    },
    Context,
};
use mlir_sys::{
    mlirDisctinctAttrCreate, mlirLLVMDICompileUnitAttrGet, mlirLLVMDIFileAttrGet,
    mlirLLVMDIModuleAttrGet, mlirLLVMDIModuleAttrGetScope, mlirLLVMDISubprogramAttrGet,
    mlirLLVMDISubroutineTypeAttrGet, MlirLLVMDIEmissionKind_MlirLLVMDIEmissionKindFull,
    MlirLLVMDINameTableKind_MlirLLVMDINameTableKindDefault,
};
use std::{
    cell::Cell,
    collections::{hash_map::Entry, BTreeMap, HashMap, HashSet},
    ops::Deref,
};

/// The [BlockStorage] type is used to map each statement into its own entry block (on the right),
/// and its landing block (on the left) if required.
///
/// The landing block contains also the variable ids that must be present when jumping into it,
/// otherwise it's a compiler error due to an inconsistent variable state.
type BlockStorage<'c, 'a> =
    HashMap<StatementIdx, (Option<(BlockRef<'c, 'a>, Vec<VarId>)>, BlockRef<'c, 'a>)>;

/// Run the compiler on a program. The compiled program is stored in the MLIR module.
///
/// The generics `TType` and `TLibfunc` contain the information required to generate the MLIR types
/// and statement operations. Most of the time you'll want to use the default ones, which are
/// [CoreType](cairo_lang_sierra::extensions::core::CoreType) and
/// [CoreLibfunc](cairo_lang_sierra::extensions::core::CoreLibfunc) respectively.
///
/// This function needs the program and the program's registry, which doesn't need to have AP
/// tracking information.
///
/// Additionally, it needs a reference to the MLIR context, the output module and the metadata
/// storage. The last one is passed externally so that stuff can be initialized if necessary.
pub fn compile(
    context: &Context,
    module: &Module,
    program: &Program,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    di_compile_unit_id: Attribute,
    ignore_debug_names: bool,
) -> Result<(), Error> {
    if let Ok(x) = std::env::var("NATIVE_DEBUG_DUMP") {
        if x == "1" || x == "true" {
            std::fs::write("program.sierra", program.to_string())?;
        }
    }

    // Sierra programs have the following structure:
    //   1. Type declarations, one per line.
    //   2. Libfunc declarations, one per line.
    //   3. All the program statements, one per line.
    //   4. Function declarations, one per line.
    // The four sections are separated by a single blank line.
    let num_types = program.type_declarations.len() + 1;
    let n_libfuncs = program.libfunc_declarations.len() + 1;
    let sierra_stmt_start_offset = num_types + n_libfuncs + 1;

    for function in &program.funcs {
        tracing::info!("Compiling function `{}`.", function.id);
        compile_func(
            context,
            module,
            registry,
            function,
            &program.statements,
            metadata,
            di_compile_unit_id,
            sierra_stmt_start_offset,
            ignore_debug_names,
        )?;
    }

    tracing::info!("The program was compiled successfully.");
    Ok(())
}

/// Compile a single Sierra function.
///
/// The function accepts a `Function` argument, which provides the function's entry point, signature
/// and name. Check out [compile](self::compile) for a description of the other arguments.
///
/// The [module docs](self) contain more information about the compilation process.
///
/// If [`ignore_debug_names`] is true, then the function name will always be `f{id}` without any debug name info if it ever exists.
#[allow(clippy::too_many_arguments)]
fn compile_func(
    context: &Context,
    module: &Module,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    function: &Function,
    statements: &[Statement],
    metadata: &mut MetadataStorage,
    di_compile_unit_id: Attribute,
    sierra_stmt_start_offset: usize,
    ignore_debug_names: bool,
) -> Result<(), Error> {
    let fn_location = Location::new(
        context,
        "program.sierra",
        sierra_stmt_start_offset + function.entry_point.0,
        0,
    );

    let region = Region::new();
    let blocks_arena = Bump::new();

    let mut arg_types = extract_types(
        context,
        module,
        &function.signature.param_types,
        registry,
        metadata,
    )
    .collect::<Result<Vec<_>, _>>()?;
    let mut return_types = extract_types(
        context,
        module,
        &function.signature.ret_types,
        registry,
        metadata,
    )
    .collect::<Result<Vec<_>, _>>()?;

    // Replace memory-allocated arguments with pointers.
    for (ty, type_info) in
        arg_types
            .iter_mut()
            .zip(function.signature.param_types.iter().filter_map(|type_id| {
                let type_info = match registry.get_type(type_id) {
                    Ok(x) => x,
                    Err(e) => return Some(Err(e.into())),
                };
                let is_zst = match type_info.is_zst(registry) {
                    Ok(x) => x,
                    Err(e) => return Some(Err(e)),
                };

                if type_info.is_builtin() && is_zst {
                    None
                } else {
                    Some(Ok(type_info))
                }
            }))
    {
        let type_info = type_info?;
        if type_info.is_memory_allocated(registry)? {
            *ty = llvm::r#type::pointer(context, 0);
        }
    }

    // Extract memory-allocated return types from return_types and insert them in arg_types as a
    // pointer.
    let return_type_infos = function
        .signature
        .ret_types
        .iter()
        .filter_map(|type_id| {
            let type_info = match registry.get_type(type_id) {
                Ok(x) => x,
                Err(e) => return Some(Err(e.into())),
            };
            let is_zst = match type_info.is_zst(registry) {
                Ok(x) => x,
                Err(e) => return Some(Err(e)),
            };

            if type_info.is_builtin() && is_zst {
                None
            } else {
                Some(Ok((type_id, type_info)))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Possible values:
    //   None        => Doesn't return anything.
    //   Some(false) => Has a complex return type.
    //   Some(true)  => Has a manual return type which is in `arg_types[0]`.
    let has_return_ptr = if return_type_infos.len() > 1 {
        Some(false)
    } else if return_type_infos
        .first()
        .map(|(_, type_info)| type_info.is_memory_allocated(registry))
        .transpose()?
        == Some(true)
    {
        assert_eq!(return_types.len(), 1);

        return_types.remove(0);
        arg_types.insert(0, llvm::r#type::pointer(context, 0));

        Some(true)
    } else {
        None
    };

    let function_name = generate_function_name(&function.id, ignore_debug_names);
    // Don't care about whether it is for the contract executor for inner impls
    // so we don't have to pass the boolean to the function call libfunc.
    let function_name_for_inner = generate_function_name(&function.id, false);

    let di_subprogram = unsafe {
        // Various DWARF debug attributes for this function.
        // The unsafe is because this is a method not yet found in upstream LLVM nor melior, so
        // we are using our own bindings to the C++ API.
        let file_attr = Attribute::from_raw(mlirLLVMDIFileAttrGet(
            context.to_raw(),
            StringAttribute::new(context, "program.sierra").to_raw(),
            StringAttribute::new(context, ".").to_raw(),
        ));
        let compile_unit = {
            Attribute::from_raw(mlirLLVMDICompileUnitAttrGet(
                context.to_raw(),
                di_compile_unit_id.to_raw(),
                0x0002, // lang C (there is no language sierra in DWARF)
                file_attr.to_raw(),
                StringAttribute::new(context, "cairo-native").to_raw(),
                false,
                MlirLLVMDIEmissionKind_MlirLLVMDIEmissionKindFull,
                MlirLLVMDINameTableKind_MlirLLVMDINameTableKindDefault,
            ))
        };

        let di_module = mlirLLVMDIModuleAttrGet(
            context.to_raw(),
            file_attr.to_raw(),
            compile_unit.to_raw(),
            StringAttribute::new(context, "LLVMDialectModule").to_raw(),
            StringAttribute::new(context, "").to_raw(),
            StringAttribute::new(context, "").to_raw(),
            StringAttribute::new(context, "").to_raw(),
            0,
            false,
        );

        let module_scope = mlirLLVMDIModuleAttrGetScope(di_module);

        Attribute::from_raw({
            let id = mlirDisctinctAttrCreate(
                StringAttribute::new(context, &format!("fn_{}", function.id.id)).to_raw(),
            );

            // Don't add argument types since its not useful, we only use the debugger for source locations.
            let ty = mlirLLVMDISubroutineTypeAttrGet(
                context.to_raw(),
                0x0, // call conv: C
                0,
                std::ptr::null(),
            );

            mlirLLVMDISubprogramAttrGet(
                context.to_raw(),
                id,
                module_scope,
                file_attr.to_raw(),
                StringAttribute::new(context, &function_name).to_raw(),
                StringAttribute::new(context, &function_name).to_raw(),
                file_attr.to_raw(),
                (sierra_stmt_start_offset + function.entry_point.0) as u32,
                (sierra_stmt_start_offset + function.entry_point.0) as u32,
                0x8, // dwarf subprogram flag: definition
                ty,
            )
        })
    };

    tracing::debug!("Generating function structure (region with blocks).");
    let (entry_block, blocks, is_recursive) = generate_function_structure(
        context,
        module,
        &region,
        registry,
        function,
        statements,
        metadata,
        sierra_stmt_start_offset,
    )?;

    tracing::debug!("Generating the function implementation.");
    // Workaround for the `entry block of region may not have predecessors` error:
    let pre_entry_block_args = arg_types
        .iter()
        .map(|ty| {
            (
                *ty,
                Location::new(
                    context,
                    "program.sierra",
                    sierra_stmt_start_offset + function.entry_point.0,
                    0,
                ),
            )
        })
        .collect::<Vec<_>>();
    let pre_entry_block =
        region.insert_block_before(entry_block, Block::new(&pre_entry_block_args));

    let initial_state = edit_state::put_results(OrderedHashMap::<_, Value>::default(), {
        let mut values = Vec::new();

        let mut count = 0;
        for param in &function.params {
            let type_info = registry.get_type(&param.ty)?;
            let location = Location::new(
                context,
                "program.sierra",
                sierra_stmt_start_offset + function.entry_point.0,
                0,
            );

            values.push((
                &param.id,
                if type_info.is_builtin() && type_info.is_zst(registry)? {
                    pre_entry_block
                        .append_operation(llvm::undef(
                            type_info.build(context, module, registry, metadata, &param.ty)?,
                            location,
                        ))
                        .result(0)?
                        .into()
                } else {
                    let value = entry_block.argument(count)?.into();
                    count += 1;

                    value
                },
            ));
        }

        values.into_iter()
    })?;

    tracing::trace!("Implementing the entry block.");
    entry_block.append_operation(cf::br(
        &blocks[&function.entry_point].1,
        &match &statements[function.entry_point.0] {
            Statement::Invocation(x) => &x.args,
            Statement::Return(x) => x,
        }
        .iter()
        .map(|x| initial_state[x])
        .collect::<Vec<_>>(),
        {
            Location::new(
                context,
                "program.sierra",
                sierra_stmt_start_offset + function.entry_point.0,
                0,
            )
        },
    ));

    let mut tailrec_state = Option::<(Value, BlockRef)>::None;
    foreach_statement_in_function::<_, Error>(
        statements,
        function.entry_point,
        initial_state,
        |statement_idx, mut state| {
            if let Some(gas_metadata) = metadata.get::<GasMetadata>() {
                let gas_cost = gas_metadata.get_gas_costs_for_statement(statement_idx);
                metadata.insert(GasCost(gas_cost));
            }

            let (landing_block, block) = &blocks[&statement_idx];

            if let Some((landing_block, _)) = landing_block {
                tracing::trace!("Implementing the statement {statement_idx}'s landing block.");

                state = edit_state::put_results(
                    OrderedHashMap::default(),
                    state
                        .keys()
                        .sorted_by_key(|x| x.id)
                        .enumerate()
                        .map(|(idx, var_id)| Ok((var_id, landing_block.argument(idx)?.into())))
                        .collect::<Result<Vec<_>, Error>>()?
                        .into_iter(),
                )?;

                landing_block.append_operation(cf::br(
                    block,
                    &edit_state::take_args(
                        state.clone(),
                        match &statements[statement_idx.0] {
                            Statement::Invocation(x) => &x.args,
                            Statement::Return(x) => x,
                        }
                        .iter(),
                    )?
                    .1,
                    Location::name(
                        context,
                        &format!("landing_block(stmt_idx={})", statement_idx),
                        fn_location,
                    ),
                ));
            }

            let statement_compilation_result = match &statements[statement_idx.0] {
                Statement::Invocation(invocation) => {
                    tracing::trace!(
                        "Implementing the invocation statement at {statement_idx}: {}.",
                        invocation.libfunc_id
                    );

                    let location = Location::new(
                        context,
                        "program.sierra",
                        sierra_stmt_start_offset + statement_idx.0,
                        0,
                    );

                    #[cfg(feature = "with-debug-utils")]
                    {
                        // If this env var exists and is a valid statement, insert a debug trap before the libfunc call.
                        // Only on when using with-debug-utils feature.
                        if let Ok(x) = std::env::var("NATIVE_DEBUG_TRAP_AT_STMT") {
                            if x.eq_ignore_ascii_case(&statement_idx.0.to_string()) {
                                block.append_operation(
                                    melior::dialect::ods::llvm::intr_debugtrap(context, location)
                                        .into(),
                                );
                            }
                        }
                    }

                    let libfunc_name = if invocation.libfunc_id.debug_name.is_some() {
                        format!("{}(stmt_idx={})", invocation.libfunc_id, statement_idx)
                    } else {
                        let libf = registry.get_libfunc(&invocation.libfunc_id)?;
                        format!("{}(stmt_idx={})", libfunc_to_name(libf), statement_idx)
                    };

                    let (state, _) = edit_state::take_args(state, invocation.args.iter())?;

                    let helper = LibfuncHelper {
                        module,
                        init_block: &pre_entry_block,
                        region: &region,
                        blocks_arena: &blocks_arena,
                        last_block: Cell::new(block),
                        branches: generate_branching_targets(
                            &blocks,
                            statements,
                            statement_idx,
                            invocation,
                            &state,
                        ),
                        results: invocation
                            .branches
                            .iter()
                            .map(|x| vec![Cell::new(None); x.results.len()])
                            .collect::<Vec<_>>(),
                    };

                    let libfunc = registry.get_libfunc(&invocation.libfunc_id)?;
                    if is_recursive {
                        if let Some(target) = libfunc.is_function_call() {
                            if target == &function.id && state.is_empty() {
                                let location = Location::name(
                                    context,
                                    &format!("recursion_counter({})", libfunc_name),
                                    location,
                                );
                                let op0 = pre_entry_block.insert_operation(
                                    0,
                                    memref::alloca(
                                        context,
                                        MemRefType::new(Type::index(context), &[], None, None),
                                        &[],
                                        &[],
                                        None,
                                        location,
                                    ),
                                );
                                let op1 = pre_entry_block.insert_operation_after(
                                    op0,
                                    index::constant(
                                        context,
                                        IntegerAttribute::new(Type::index(context), 0),
                                        location,
                                    ),
                                );
                                pre_entry_block.insert_operation_after(
                                    op1,
                                    memref::store(
                                        op1.result(0)?.into(),
                                        op0.result(0)?.into(),
                                        &[],
                                        location,
                                    ),
                                );

                                metadata
                                    .insert(TailRecursionMeta::new(
                                        op0.result(0)?.into(),
                                        &entry_block,
                                    ))
                                    .to_native_assert_error(
                                        "tail recursion metadata shouldn't be inserted",
                                    )?;
                            }
                        }
                    }

                    libfunc.build(
                        context,
                        registry,
                        block,
                        Location::name(context, &libfunc_name, location),
                        &helper,
                        metadata,
                    )?;
                    assert!(block.terminator().is_some());

                    if let Some(tailrec_meta) = metadata.remove::<TailRecursionMeta>() {
                        if let Some(return_block) = tailrec_meta.return_target() {
                            tailrec_state = Some((tailrec_meta.depth_counter(), return_block));
                        }
                    }

                    StatementCompilationResult::Processed(
                        invocation
                            .branches
                            .iter()
                            .zip(helper.results()?)
                            .map(|(branch_info, result_values)| {
                                assert_eq!(
                                    branch_info.results.len(),
                                    result_values.len(),
                                    "Mismatched number of returned values from branch."
                                );

                                Ok(edit_state::put_results(
                                    state.clone(),
                                    branch_info.results.iter().zip(result_values.into_iter()),
                                )?)
                            })
                            .collect::<Result<_, Error>>()?,
                    )
                }
                Statement::Return(var_ids) => {
                    tracing::trace!("Implementing the return statement at {statement_idx}");

                    let location = Location::name(
                        context,
                        &format!("return(stmt_idx={})", statement_idx),
                        Location::new(
                            context,
                            "program.sierra",
                            sierra_stmt_start_offset + statement_idx.0,
                            0,
                        ),
                    );

                    let (_, mut values) = edit_state::take_args(state, var_ids.iter())?;

                    let mut block = *block;
                    if is_recursive {
                        match tailrec_state {
                            None => {
                                // If this block is reached it means that a return has been detected
                                // within a tail-recursive function before the recursive call has
                                // been generated. Since we don't have the return target block at
                                // this point we need to defer this return statement's generation.
                                return Ok(StatementCompilationResult::Deferred);
                            }
                            Some((depth_counter, recursion_target)) => {
                                let location = Location::name(
                                    context,
                                    &format!("return(stmt_idx={}, tail_recursion)", statement_idx),
                                    Location::new(
                                        context,
                                        "program.sierra",
                                        sierra_stmt_start_offset + statement_idx.0,
                                        0,
                                    ),
                                );

                                // Perform tail recursion.
                                let cont_block = region.insert_block_after(block, Block::new(&[]));

                                let depth_counter_value = block.append_op_result(memref::load(
                                    depth_counter,
                                    &[],
                                    location,
                                ))?;
                                let k0 = block.const_int_from_type(
                                    context,
                                    location,
                                    0,
                                    Type::index(context),
                                )?;
                                let is_zero_depth = block.append_op_result(index::cmp(
                                    context,
                                    CmpiPredicate::Eq,
                                    depth_counter_value,
                                    k0,
                                    location,
                                ))?;

                                let k1 = block.const_int_from_type(
                                    context,
                                    location,
                                    1,
                                    Type::index(context),
                                )?;
                                let depth_counter_value = block.append_op_result(index::sub(
                                    depth_counter_value,
                                    k1,
                                    location,
                                ))?;
                                block.append_operation(memref::store(
                                    depth_counter_value,
                                    depth_counter,
                                    &[],
                                    location,
                                ));

                                let recursive_values = match has_return_ptr {
                                    Some(true) => function
                                        .signature
                                        .ret_types
                                        .iter()
                                        .zip(&values)
                                        .filter_map(|(type_id, value)| {
                                            let type_info = match registry.get_type(type_id) {
                                                Ok(x) => x,
                                                Err(e) => return Some(Err(e.into())),
                                            };
                                            let is_zst = match type_info.is_zst(registry) {
                                                Ok(x) => x,
                                                Err(e) => return Some(Err(e)),
                                            };
                                            let is_memory_allocated =
                                                match type_info.is_memory_allocated(registry) {
                                                    Ok(x) => x,
                                                    Err(e) => return Some(Err(e)),
                                                };

                                            if is_zst || is_memory_allocated {
                                                None
                                            } else {
                                                Some(Ok(*value))
                                            }
                                        })
                                        .collect::<Result<Vec<_>, _>>()?,
                                    Some(false) => function
                                        .signature
                                        .ret_types
                                        .iter()
                                        .zip(&values)
                                        .filter_map(|(type_id, value)| {
                                            let type_info = match registry.get_type(type_id) {
                                                Ok(x) => x,
                                                Err(e) => return Some(Err(e.into())),
                                            };
                                            let is_zst = match type_info.is_zst(registry) {
                                                Ok(x) => x,
                                                Err(e) => return Some(Err(e)),
                                            };

                                            if is_zst {
                                                None
                                            } else {
                                                Some(Ok(*value))
                                            }
                                        })
                                        .collect::<Result<Vec<_>, _>>()?,
                                    None => todo!(),
                                };

                                block.append_operation(cf::cond_br(
                                    context,
                                    is_zero_depth,
                                    &cont_block,
                                    &recursion_target,
                                    &[],
                                    &recursive_values,
                                    location,
                                ));

                                block = cont_block;
                            }
                        }
                    }

                    // Remove ZST builtins from the return values.
                    for (idx, type_id) in function.signature.ret_types.iter().enumerate().rev() {
                        let type_info = registry.get_type(type_id)?;
                        if type_info.is_builtin() && type_info.is_zst(registry)? {
                            values.remove(idx);
                        }
                    }

                    // Store the return value in the return pointer, if there's one.
                    if Some(true) == has_return_ptr {
                        let (_ret_type_id, ret_type_info) = return_type_infos[0];
                        let ret_layout = ret_type_info.layout(registry)?;

                        let ptr = values.remove(0);
                        block.append_operation(llvm::store(
                            context,
                            ptr,
                            pre_entry_block.arg(0)?,
                            location,
                            LoadStoreOptions::new().align(Some(IntegerAttribute::new(
                                IntegerType::new(context, 64).into(),
                                ret_layout.align() as i64,
                            ))),
                        ));
                    }

                    block.append_operation(llvm::r#return(
                        Some({
                            let res_ty = llvm::r#type::r#struct(context, &return_types, false);
                            values.iter().enumerate().try_fold(
                                block.append_op_result(llvm::undef(res_ty, location))?,
                                |acc, (idx, x)| {
                                    block.append_op_result(llvm::insert_value(
                                        context,
                                        acc,
                                        DenseI64ArrayAttribute::new(context, &[idx as i64]),
                                        *x,
                                        location,
                                    ))
                                },
                            )?
                        }),
                        location,
                    ));

                    StatementCompilationResult::Processed(Vec::new())
                }
            };

            metadata.remove::<GasCost>();

            Ok(statement_compilation_result)
        },
    )?;

    // Load arguments and jump to the entry block.
    {
        let mut arg_values = Vec::with_capacity(function.signature.param_types.len());
        for (i, type_id_and_info) in function
            .signature
            .param_types
            .iter()
            .filter_map(|type_id| {
                registry
                    .get_type(type_id)
                    .map(|type_info| {
                        let is_zst = match type_info.is_zst(registry) {
                            Ok(x) => x,
                            Err(e) => return Some(Err(e)),
                        };

                        if type_info.is_builtin() && is_zst {
                            None
                        } else {
                            Some(Ok((type_id, type_info)))
                        }
                    })
                    .map_err(Error::from)
                    .transpose()
                    .map(|x| match x {
                        Ok(Ok(x)) => Ok(x),
                        Ok(Err(e)) | Err(e) => Err(e),
                    })
            })
            .enumerate()
        {
            let (type_id, type_info) = type_id_and_info?;

            let mut value = pre_entry_block
                .argument((has_return_ptr == Some(true)) as usize + i)?
                .into();
            if type_info.is_memory_allocated(registry)? {
                value = pre_entry_block
                    .append_operation(llvm::load(
                        context,
                        value,
                        type_info.build(context, module, registry, metadata, type_id)?,
                        fn_location,
                        LoadStoreOptions::new().align(Some(IntegerAttribute::new(
                            IntegerType::new(context, 64).into(),
                            type_info.layout(registry)?.align() as i64,
                        ))),
                    ))
                    .result(0)?
                    .into();
            }

            arg_values.push(value);
        }

        pre_entry_block.append_operation(cf::br(&entry_block, &arg_values, fn_location));
    }

    let inner_function_name = format!("impl${function_name_for_inner}");
    module.body().append_operation(llvm::func(
        context,
        StringAttribute::new(context, &inner_function_name),
        TypeAttribute::new(llvm::r#type::function(
            llvm::r#type::r#struct(context, &return_types, false),
            &arg_types,
            false,
        )),
        region,
        &[
            (
                Identifier::new(context, "sym_visibility"),
                StringAttribute::new(context, "private").into(),
            ),
            (
                Identifier::new(context, "linkage"),
                Attribute::parse(context, "#llvm.linkage<private>")
                    .ok_or(Error::ParseAttributeError)?,
            ),
            (
                Identifier::new(context, "CConv"),
                Attribute::parse(context, "#llvm.cconv<fastcc>")
                    .ok_or(Error::ParseAttributeError)?,
            ),
        ],
        Location::fused(
            context,
            &[Location::new(
                context,
                "program.sierra",
                sierra_stmt_start_offset + function.entry_point.0,
                0,
            )],
            di_subprogram,
        ),
    ));

    generate_entry_point_wrapper(
        context,
        module,
        function_name.as_ref(),
        &inner_function_name,
        &pre_entry_block_args,
        &return_types,
        Location::new(
            context,
            "program.sierra",
            sierra_stmt_start_offset + function.entry_point.0,
            0,
        ),
    )?;

    tracing::debug!("Done generating function {}.", function.id);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn generate_function_structure<'c, 'a>(
    context: &'c Context,
    module: &'a Module<'c>,
    region: &'a Region<'c>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    function: &Function,
    statements: &[Statement],
    metadata_storage: &mut MetadataStorage,
    sierra_stmt_start_offset: usize,
) -> Result<(BlockRef<'c, 'a>, BlockStorage<'c, 'a>, bool), Error> {
    let initial_state = edit_state::put_results::<Type>(
        OrderedHashMap::default(),
        function
            .params
            .iter()
            .zip(&function.signature.param_types)
            .map(|(param, ty)| {
                let type_info = registry.get_type(ty)?;
                Ok((
                    &param.id,
                    type_info.build(context, module, registry, metadata_storage, ty)?,
                ))
            })
            .collect::<Result<Vec<_>, Error>>()?
            .into_iter(),
    )?;

    let mut blocks = BTreeMap::new();
    let mut predecessors = HashMap::from([(function.entry_point, (initial_state.clone(), 0))]);

    let mut num_tail_recursions = 0usize;
    foreach_statement_in_function::<_, Error>(
        statements,
        function.entry_point,
        initial_state,
        |statement_idx, state| {
            let block = {
                if let std::collections::btree_map::Entry::Vacant(e) = blocks.entry(statement_idx.0)
                {
                    e.insert(Block::new(&[]));
                    blocks
                        .get_mut(&statement_idx.0)
                        .to_native_assert_error("block should exist")?
                } else {
                    native_panic!("statement index already present in block")
                }
            };

            Ok(match &statements[statement_idx.0] {
                Statement::Invocation(invocation) => {
                    tracing::trace!(
                        "Creating block for invocation statement at index {statement_idx}: {}",
                        invocation.libfunc_id
                    );

                    let (state, types) =
                        edit_state::take_args(state.clone(), invocation.args.iter())?;

                    let location = Location::new(
                        context,
                        "program.sierra",
                        sierra_stmt_start_offset + statement_idx.0,
                        0,
                    );

                    for ty in types {
                        block.add_argument(ty, location);
                    }

                    let libfunc = registry.get_libfunc(&invocation.libfunc_id)?;
                    if let CoreConcreteLibfunc::FunctionCall(info) = libfunc {
                        if info.function.id == function.id && state.is_empty() {
                            num_tail_recursions += 1;
                        }
                    }

                    StatementCompilationResult::Processed(
                        invocation
                            .branches
                            .iter()
                            .zip(libfunc.branch_signatures())
                            .map(|(branch, branch_signature)| {
                                let state = edit_state::put_results(
                                    state.clone(),
                                    branch.results.iter().zip(
                                        branch_signature
                                            .vars
                                            .iter()
                                            .map(|var_info| -> Result<_, Error> {
                                                registry.get_type(&var_info.ty)?.build(
                                                    context,
                                                    module,
                                                    registry,
                                                    metadata_storage,
                                                    &var_info.ty,
                                                )
                                            })
                                            .collect::<Result<Vec<_>, _>>()?,
                                    ),
                                )?;

                                let (prev_state, pred_count) =
                                    match predecessors.entry(statement_idx.next(&branch.target)) {
                                        Entry::Occupied(entry) => entry.into_mut(),
                                        Entry::Vacant(entry) => entry.insert((state.clone(), 0)),
                                    };
                                assert!(
                                    prev_state.eq_unordered(&state),
                                    "Branch target states do not match."
                                );
                                *pred_count += 1;

                                Ok(state)
                            })
                            .collect::<Result<_, Error>>()?,
                    )
                }
                Statement::Return(var_ids) => {
                    tracing::trace!(
                        "Creating block for return statement at index {statement_idx}."
                    );

                    let (state, types) = edit_state::take_args(state.clone(), var_ids.iter())?;
                    assert!(
                        state.is_empty(),
                        "State must be empty after a return statement."
                    );

                    let location = Location::new(
                        context,
                        "program.sierra",
                        sierra_stmt_start_offset + statement_idx.0,
                        0,
                    );

                    for ty in types {
                        block.add_argument(ty, location);
                    }

                    StatementCompilationResult::Processed(Vec::new())
                }
            })
        },
    )?;

    tracing::trace!("Generating function entry block.");
    let entry_block = region.append_block(Block::new(&{
        extract_types(
            context,
            module,
            &function.signature.param_types,
            registry,
            metadata_storage,
        )
        .map(|ty| {
            Ok((
                ty?,
                Location::new(
                    context,
                    "program.sierra",
                    sierra_stmt_start_offset + function.entry_point.0,
                    0,
                ),
            ))
        })
        .collect::<Result<Vec<_>, Error>>()?
    }));

    let blocks = blocks
        .into_iter()
        .map(|(i, block)| {
            let statement_idx = StatementIdx(i);

            tracing::trace!("Inserting block for statement at index {statement_idx}.");
            let libfunc_block = region.append_block(block);
            let landing_block = (predecessors[&statement_idx].1 > 1).then(|| {
                tracing::trace!(
                    "Generating a landing block for the statement at index {statement_idx}."
                );

                (
                    region.insert_block_before(
                        libfunc_block,
                        Block::new(
                            &predecessors[&statement_idx]
                                .0
                                .iter()
                                .map(|(var_id, ty)| (var_id.id, *ty))
                                .collect::<BTreeMap<_, _>>()
                                .into_values()
                                .map(|ty| {
                                    (
                                        ty,
                                        Location::new(
                                            context,
                                            "program.sierra",
                                            sierra_stmt_start_offset + statement_idx.0,
                                            0,
                                        ),
                                    )
                                })
                                .collect::<Vec<_>>(),
                        ),
                    ),
                    predecessors[&statement_idx]
                        .0
                        .clone()
                        .into_iter()
                        .sorted_by_key(|(k, _)| k.id)
                        .collect::<Vec<_>>(),
                )
            });

            (statement_idx, (landing_block, libfunc_block))
        })
        .collect::<HashMap<_, _>>();

    Ok((
        entry_block,
        blocks
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    (
                        v.0.map(|x| (x.0, x.1.into_iter().map(|x| x.0).collect::<Vec<_>>())),
                        v.1,
                    ),
                )
            })
            .collect(),
        num_tail_recursions == 1,
    ))
}

fn extract_types<'c: 'a, 'a>(
    context: &'c Context,
    module: &'a Module<'c>,
    type_ids: &'a [ConcreteTypeId],
    registry: &'a ProgramRegistry<CoreType, CoreLibfunc>,
    metadata_storage: &'a mut MetadataStorage,
) -> impl 'a + Iterator<Item = Result<Type<'c>, Error>> {
    type_ids.iter().filter_map(|id| {
        let type_info = match registry.get_type(id) {
            Ok(x) => x,
            Err(e) => return Some(Err(e.into())),
        };
        let is_zst = match type_info.is_zst(registry) {
            Ok(x) => x,
            Err(e) => return Some(Err(e)),
        };

        if type_info.is_builtin() && is_zst {
            None
        } else {
            Some(type_info.build(context, module, registry, metadata_storage, id))
        }
    })
}

fn foreach_statement_in_function<S, E>(
    statements: &[Statement],
    entry_point: StatementIdx,
    initial_state: S,
    mut closure: impl FnMut(StatementIdx, S) -> Result<StatementCompilationResult<Vec<S>>, E>,
) -> Result<(), E>
where
    S: Clone,
{
    let mut queue = vec![(entry_point, initial_state)];
    let mut visited = HashSet::new();

    while let Some((statement_idx, state)) = queue.pop() {
        if !visited.insert(statement_idx) {
            continue;
        }

        match closure(statement_idx, state.clone())? {
            StatementCompilationResult::Processed(branch_states) => {
                let branches = match &statements[statement_idx.0] {
                    Statement::Invocation(x) => x.branches.as_slice(),
                    Statement::Return(_) => &[],
                };
                assert_eq!(
                    branches.len(),
                    branch_states.len(),
                    "Returned number of states must match the number of branches."
                );

                queue.extend(
                    branches
                        .iter()
                        .map(|branch| statement_idx.next(&branch.target))
                        .zip(branch_states),
                );
            }
            StatementCompilationResult::Deferred => {
                tracing::trace!("Statement {statement_idx}'s compilation has been deferred.");

                visited.remove(&statement_idx);
                queue.insert(0, (statement_idx, state));
            }
        }
    }

    Ok(())
}

fn generate_branching_targets<'ctx, 'this, 'a>(
    blocks: &'this BlockStorage<'ctx, 'this>,
    statements: &'this [Statement],
    statement_idx: StatementIdx,
    invocation: &'this Invocation,
    state: &OrderedHashMap<VarId, Value<'ctx, 'this>>,
) -> Vec<(&'this Block<'ctx>, Vec<BranchArg<'ctx, 'this>>)>
where
    'this: 'ctx,
{
    invocation
        .branches
        .iter()
        .map(move |branch| {
            let target_idx = statement_idx.next(&branch.target);
            let (landing_block, block) = &blocks[&target_idx];

            match landing_block {
                Some((landing_block, state_vars)) => {
                    let target_vars = state_vars
                        .iter()
                        .map(|var_id| {
                            match branch.results.iter().find_position(|id| *id == var_id) {
                                Some((i, _)) => BranchArg::Returned(i),
                                None => BranchArg::External(state[var_id]),
                            }
                        })
                        .collect::<Vec<_>>();

                    (landing_block.deref(), target_vars)
                }
                None => {
                    let target_vars = match &statements[target_idx.0] {
                        Statement::Invocation(x) => &x.args,
                        Statement::Return(x) => x,
                    }
                    .iter()
                    .map(|var_id| {
                        match branch
                            .results
                            .iter()
                            .enumerate()
                            .find_map(|(i, id)| (id == var_id).then_some(i))
                        {
                            Some(i) => BranchArg::Returned(i),
                            None => BranchArg::External(state[var_id]),
                        }
                    })
                    .collect::<Vec<_>>();

                    (block.deref(), target_vars)
                }
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn generate_entry_point_wrapper<'c>(
    context: &'c Context,
    module: &Module<'c>,
    public_symbol: &str,
    private_symbol: &str,
    arg_types: &[(Type<'c>, Location<'c>)],
    ret_types: &[Type<'c>],
    location: Location<'c>,
) -> Result<(), Error> {
    let region = Region::new();
    let block = region.append_block(Block::new(arg_types));

    let mut args = Vec::with_capacity(arg_types.len());
    for i in 0..arg_types.len() {
        args.push(block.argument(i)?.into());
    }

    let result = block.append_op_result(
        OperationBuilder::new("llvm.call", location)
            .add_attributes(&[
                (
                    Identifier::new(context, "callee"),
                    FlatSymbolRefAttribute::new(context, private_symbol).into(),
                ),
                (
                    Identifier::new(context, "CConv"),
                    Attribute::parse(context, "#llvm.cconv<fastcc>")
                        .ok_or(Error::ParseAttributeError)?,
                ),
            ])
            .add_operands(&args)
            .add_results(&[llvm::r#type::r#struct(context, ret_types, false)])
            .build()?,
    )?;

    let mut returns = Vec::with_capacity(ret_types.len());
    for (i, ty) in ret_types.iter().enumerate() {
        returns.push(block.extract_value(context, location, result, *ty, i)?);
    }

    block.append_operation(func::r#return(&returns, location));

    module.body().append_operation(func::func(
        context,
        StringAttribute::new(context, public_symbol),
        TypeAttribute::new(
            FunctionType::new(
                context,
                &arg_types.iter().map(|x| x.0).collect::<Vec<_>>(),
                ret_types,
            )
            .into(),
        ),
        region,
        &[
            (
                Identifier::new(context, "sym_visibility"),
                StringAttribute::new(context, "public").into(),
            ),
            (
                Identifier::new(context, "llvm.linkage"),
                Attribute::parse(context, "#llvm.linkage<private>")
                    .ok_or(Error::ParseAttributeError)?,
            ),
            (
                Identifier::new(context, "llvm.CConv"),
                Attribute::parse(context, "#llvm.cconv<fastcc>")
                    .ok_or(Error::ParseAttributeError)?,
            ),
            (
                Identifier::new(context, "llvm.emit_c_interface"),
                Attribute::unit(context),
            ),
        ],
        location,
    ));
    Ok(())
}

/// Return type for the closure in [`foreach_statement_in_function`] that determines whether the
/// statement was processed successfully or needs to be processed again at the end.
#[derive(Clone, Debug)]
enum StatementCompilationResult<T> {
    /// The statement was processed successfully.
    Processed(T),
    /// The statement's processing has to be deferred until the end.
    Deferred,
}
