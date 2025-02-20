//! # Function call libfuncs
//!
//! Includes logic for handling direct tail recursive function calls. More information on this topic
//! at the [tail recursive metadata](crate::metadata::tail_recursion).

use super::LibfuncHelper;
use crate::{
    error::{Error, Result},
    metadata::{tail_recursion::TailRecursionMeta, MetadataStorage},
    types::TypeBuilder,
    utils::{generate_function_name, BlockExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        function_call::SignatureAndFunctionConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{cf, index, llvm, memref},
    ir::{
        attribute::{DenseI32ArrayAttribute, FlatSymbolRefAttribute},
        operation::OperationBuilder,
        r#type::IntegerType,
        Attribute, Block, BlockLike, Identifier, Location, Type, Value,
    },
    Context,
};
use std::alloc::Layout;

/// Generate MLIR operations for the `function_call` libfunc.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndFunctionConcreteLibfunc,
) -> Result<()> {
    let mut tailrec_meta = metadata.remove::<TailRecursionMeta>();

    let mut arguments = Vec::new();
    for (idx, type_id) in info.function.signature.param_types.iter().enumerate() {
        let type_info = registry.get_type(type_id)?;

        if !(type_info.is_builtin() && type_info.is_zst(registry)?) {
            arguments.push(
                if tailrec_meta.is_none() && type_info.is_memory_allocated(registry)? {
                    let elem_ty = type_info.build(context, helper, registry, metadata, type_id)?;
                    let stack_ptr = helper.init_block().alloca1(
                        context,
                        location,
                        elem_ty,
                        type_info.layout(registry)?.align(),
                    )?;

                    entry.store(context, location, stack_ptr, entry.argument(idx)?.into())?;

                    stack_ptr
                } else {
                    entry.argument(idx)?.into()
                },
            );
        }
    }

    if let Some(tailrec_meta) = &mut tailrec_meta {
        let depth_counter =
            entry.append_op_result(memref::load(tailrec_meta.depth_counter(), &[], location))?;

        let k1 = entry.const_int_from_type(context, location, 1, Type::index(context))?;
        let new_depth_counter = entry.append_op_result(index::add(depth_counter, k1, location))?;
        entry.append_operation(memref::store(
            new_depth_counter,
            tailrec_meta.depth_counter(),
            &[],
            location,
        ));

        entry.append_operation(cf::br(
            &tailrec_meta.recursion_target(),
            &arguments,
            location,
        ));

        let cont_block = helper.append_block(Block::new(
            &info
                .function
                .signature
                .ret_types
                .iter()
                .map(|ty| {
                    Ok((
                        registry
                            .get_type(ty)?
                            .build(context, helper, registry, metadata, ty)?,
                        location,
                    ))
                })
                .collect::<Result<Vec<_>>>()?,
        ));
        tailrec_meta.set_return_target(cont_block);

        let mut results = Vec::<Value>::new();
        let mut count = 0;
        for var_info in &info.signature.branch_signatures[0].vars {
            let type_info = registry.get_type(&var_info.ty)?;

            if type_info.is_builtin() && type_info.is_zst(registry)? {
                results.push(cont_block.append_op_result(llvm::undef(
                    type_info.build(context, helper, registry, metadata, &var_info.ty)?,
                    location,
                ))?);
            } else {
                let val = cont_block.argument(count)?.into();
                count += 1;

                results.push(val);
            }
        }

        helper.br(cont_block, 0, &results, location)?;
    } else {
        let mut result_types = Vec::new();
        let return_types = info
            .function
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

                (!(type_info.is_builtin() && is_zst)).then_some(Ok((type_id, type_info)))
            })
            .collect::<Result<Vec<_>>>()?;
        // A function has a return pointer if either:
        //   - There are multiple return values.
        //   - The return value is memory allocated.
        let has_return_ptr = if return_types.len() > 1 {
            result_types.extend(
                return_types
                    .iter()
                    .map(|(type_id, type_info)| {
                        type_info.build(context, helper, registry, metadata, type_id)
                    })
                    .collect::<std::result::Result<Vec<_>, _>>()?,
            );

            Some(false)
        } else if return_types
            .first()
            .map(|(_, type_info)| type_info.is_memory_allocated(registry))
            .transpose()?
            == Some(true)
        {
            let (type_id, type_info) = return_types[0];
            let layout = type_info.layout(registry)?;

            let stack_ptr = helper.init_block().alloca1(
                context,
                location,
                type_info.build(context, helper, registry, metadata, type_id)?,
                layout.align(),
            )?;

            arguments.insert(0, stack_ptr);

            Some(true)
        } else if !return_types.is_empty() {
            let (type_id, type_info) = return_types[0];
            result_types.push(type_info.build(context, helper, registry, metadata, type_id)?);

            None
        } else {
            None
        };

        let function_call_result = entry.append_op_result(
            OperationBuilder::new("llvm.call", location)
                .add_attributes(&[
                    (
                        Identifier::new(context, "callee"),
                        FlatSymbolRefAttribute::new(
                            context,
                            &format!("impl${}", generate_function_name(&info.function.id, false)),
                        )
                        .into(),
                    ),
                    (
                        Identifier::new(context, "CConv"),
                        Attribute::parse(context, "#llvm.cconv<fastcc>")
                            .ok_or(Error::ParseAttributeError)?,
                    ),
                ])
                .add_operands(&arguments)
                .add_results(&[llvm::r#type::r#struct(context, &result_types, false)])
                .build()?,
        )?;

        let mut results = Vec::new();
        match has_return_ptr {
            Some(true) => {
                // Manual return type.

                let mut layout = Layout::new::<()>();
                for (idx, type_id) in info.function.signature.ret_types.iter().enumerate() {
                    let type_info = registry.get_type(type_id)?;

                    if type_info.is_builtin() && type_info.is_zst(registry)? {
                        results.push(entry.argument(idx)?.into());
                    } else {
                        let val = arguments[0];

                        let offset;
                        let ret_layout = type_info.layout(registry)?;
                        (layout, offset) = layout.extend(ret_layout)?;

                        let pointer_val = entry.append_op_result(llvm::get_element_ptr(
                            context,
                            val,
                            DenseI32ArrayAttribute::new(context, &[offset as i32]),
                            IntegerType::new(context, 8).into(),
                            llvm::r#type::pointer(context, 0),
                            location,
                        ))?;

                        results.push(entry.load(
                            context,
                            location,
                            pointer_val,
                            type_info.build(context, helper, registry, metadata, type_id)?,
                        )?);
                    }
                }
            }
            Some(false) => {
                // Complex return type. Just extract the values from the struct, since LLVM will
                // handle the rest.

                for (idx, type_id) in info.function.signature.ret_types.iter().enumerate() {
                    let type_info = registry.get_type(type_id)?;

                    if type_info.is_builtin() && type_info.is_zst(registry)? {
                        results.push(entry.argument(idx)?.into());
                    } else {
                        let val = entry.extract_value(
                            context,
                            location,
                            function_call_result,
                            result_types[idx],
                            idx,
                        )?;

                        results.push(val);
                    }
                }
            }
            None => {
                // Returned data is simple.

                let mut count = 0;
                for (idx, type_id) in info.function.signature.ret_types.iter().enumerate() {
                    let type_info = registry.get_type(type_id)?;
                    assert!(!type_info.is_memory_allocated(registry)?);

                    if type_info.is_builtin() && type_info.is_zst(registry)? {
                        results.push(entry.argument(idx)?.into());
                    } else {
                        let value = entry.extract_value(
                            context,
                            location,
                            function_call_result,
                            result_types[count],
                            count,
                        )?;
                        count += 1;

                        results.push(value);
                    }
                }
            }
        }

        helper.br(entry, 0, &results, location)?;
    }

    if let Some(tailrec_meta) = tailrec_meta {
        metadata.insert(tailrec_meta);
    }

    Ok(())
}
