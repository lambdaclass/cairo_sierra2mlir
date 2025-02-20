//! # `Felt` dictionary entry libfuncs

use super::LibfuncHelper;
use crate::{
    error::{Error, Result},
    metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
    types::TypeBuilder,
    utils::{BlockExt, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        felt252_dict::Felt252DictEntryConcreteLibfunc,
        lib_func::SignatureAndTypeConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{llvm, scf},
    ir::{r#type::IntegerType, Block, BlockLike, Location, Region},
    Context,
};
use std::cell::Cell;

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &Felt252DictEntryConcreteLibfunc,
) -> Result<()> {
    match selector {
        Felt252DictEntryConcreteLibfunc::Get(info) => {
            build_get(context, registry, entry, location, helper, metadata, info)
        }
        Felt252DictEntryConcreteLibfunc::Finalize(info) => {
            build_finalize(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// The felt252_dict_entry_get libfunc receives the dictionary and the key and
/// returns the associated dict entry, along with it's value.
///
/// The dict entry also contains a pointer to the dictionary.
///
/// If the key doesn't yet exist, it is created and the type's default value is returned.
///
/// # Cairo Signature
///
/// ```cairo
/// fn felt252_dict_entry_get<T>(dict: Felt252Dict<T>, key: felt252) -> (Felt252DictEntry<T>, T) nopanic;
/// ```
pub fn build_get<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let (key_ty, key_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.param_signatures()[1].ty,
    )?;
    let entry_ty = registry.build_type(
        context,
        helper,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;
    let concrete_value_type = registry.get_type(&info.ty)?;
    let value_ty = concrete_value_type.build(context, helper, registry, metadata, &info.ty)?;

    let dict_ptr = entry.arg(0)?;
    let entry_key = entry.arg(1)?;

    let entry_key_ptr =
        helper
            .init_block()
            .alloca1(context, location, key_ty, key_layout.align())?;
    entry.store(context, location, entry_key_ptr, entry_key)?;

    let (is_present, value_ptr) = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .ok_or(Error::MissingMetadata)?
        .dict_get(context, helper, entry, dict_ptr, entry_key_ptr, location)?;
    let is_present = entry.trunci(is_present, IntegerType::new(context, 1).into(), location)?;

    let value = entry.append_op_result(scf::r#if(
        is_present,
        &[value_ty],
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            // If the entry is present we can load the current value.
            let value = block.load(context, location, value_ptr, value_ty)?;

            block.append_operation(scf::r#yield(&[value], location));
            region
        },
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let helper = LibfuncHelper {
                module: helper.module,
                init_block: helper.init_block,
                region: &region,
                blocks_arena: helper.blocks_arena,
                last_block: Cell::new(&block),
                branches: Vec::new(),
                results: Vec::new(),

                #[cfg(feature = "with-profiler")]
                profiler: helper.profiler.clone(),
            };

            // When the entry is vacant we need to create the default value.
            let value = concrete_value_type.build_default(
                context, registry, &block, location, &helper, metadata, &info.ty,
            )?;

            block.append_operation(scf::r#yield(&[value], location));
            region
        },
        location,
    ))?;

    let dict_entry = entry.append_op_result(llvm::undef(entry_ty, location))?;
    let dict_entry = entry.insert_values(context, location, dict_entry, &[dict_ptr, value_ptr])?;

    // The `Felt252DictEntry<T>` holds both the `Felt252Dict<T>` and the pointer to the space where
    // the new value will be written when the entry is finalized. If the entry were to be dropped
    // (without being consumed by the finalizer), which shouldn't be possible under normal
    // conditions, and the type `T` requires a custom drop implementation (ex. arrays, dicts...),
    // it'll cause undefined behavior because when the value is moved out of the dictionary (on
    // `get`), the memory it occupied is not modified because we're expecting it to be overwritten
    // by the finalizer (in other words, the extracted element will be dropped twice).

    helper.br(entry, 0, &[dict_entry, value], location)
}

/// The felt252_dict_entry_finalize libfunc receives the dict entry and a new value,
/// inserts the new value in the entry, and returns the full dictionary.
///
/// # Cairo Signature
///
/// ```cairo
/// fn felt252_dict_entry_finalize<T>(dict_entry: Felt252DictEntry<T>, new_value: T) -> Felt252Dict<T> nopanic;
/// ```
pub fn build_finalize<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    // Get the dict entry struct: `crate::types::felt252_dict_entry`.
    let dict_entry = entry.arg(0)?;
    let new_value = entry.arg(1)?;

    let dict_ptr = entry.extract_value(
        context,
        location,
        dict_entry,
        llvm::r#type::pointer(context, 0),
        0,
    )?;
    let value_ptr = entry.extract_value(
        context,
        location,
        dict_entry,
        llvm::r#type::pointer(context, 0),
        1,
    )?;

    entry.store(context, location, value_ptr, new_value)?;

    helper.br(entry, 0, &[dict_ptr], location)
}

#[cfg(test)]
mod test {
    use crate::utils::test::{jit_dict, load_cairo, run_program_assert_output};

    #[test]
    fn run_dict_insert() {
        let program = load_cairo!(
            use traits::Default;
            use dict::Felt252DictTrait;

            fn run_test() -> u32 {
                let mut dict: Felt252Dict<u32> = Default::default();
                dict.insert(2, 1_u32);
                dict.get(2)
            }
        );

        run_program_assert_output(&program, "run_test", &[], 1u32.into());
    }

    #[test]
    fn run_dict_insert_big() {
        let program = load_cairo!(
            use traits::Default;
            use dict::Felt252DictTrait;

            fn run_test() -> u64 {
                let mut dict: Felt252Dict<u64> = Default::default();
                dict.insert(200000000, 4_u64);
                dict.get(200000000)
            }
        );

        run_program_assert_output(&program, "run_test", &[], 4u64.into());
    }

    #[test]
    fn run_dict_insert_ret_dict() {
        let program = load_cairo!(
            use traits::Default;
            use dict::Felt252DictTrait;

            fn run_test() -> Felt252Dict<u32> {
                let mut dict: Felt252Dict<u32> = Default::default();
                dict.insert(2, 1_u32);
                dict
            }
        );

        run_program_assert_output(
            &program,
            "run_test",
            &[],
            jit_dict!(
                2 => 1u32
            ),
        );
    }

    #[test]
    fn run_dict_insert_multiple() {
        let program = load_cairo!(
            use traits::Default;
            use dict::Felt252DictTrait;

            fn run_test() -> u32 {
                let mut dict: Felt252Dict<u32> = Default::default();
                dict.insert(2, 1_u32);
                dict.insert(3, 1_u32);
                dict.insert(4, 1_u32);
                dict.insert(5, 1_u32);
                dict.insert(6, 1_u32);
                dict.insert(7, 1_u32);
                dict.insert(8, 1_u32);
                dict.insert(9, 1_u32);
                dict.insert(10, 1_u32);
                dict.insert(11, 1_u32);
                dict.insert(12, 1_u32);
                dict.insert(13, 1_u32);
                dict.insert(14, 1_u32);
                dict.insert(15, 1_u32);
                dict.insert(16, 1_u32);
                dict.insert(17, 1_u32);
                dict.insert(18, 1345432_u32);
                dict.get(18)
            }
        );

        run_program_assert_output(&program, "run_test", &[], 1345432_u32.into());
    }
}
