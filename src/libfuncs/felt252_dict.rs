//! # `Felt` dictionary libfuncs

use super::LibfuncHelper;
use crate::{
    error::Result,
    metadata::{
        felt252_dict::Felt252DictOverrides, runtime_bindings::RuntimeBindingsMeta, MetadataStorage,
    },
    native_panic,
    types::TypeBuilder,
    utils::BlockExt,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        felt252_dict::Felt252DictConcreteLibfunc,
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{llvm, ods},
    ir::{Block, BlockLike, Location},
    Context,
};

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &Felt252DictConcreteLibfunc,
) -> Result<()> {
    match selector {
        Felt252DictConcreteLibfunc::New(info) => {
            build_new(context, registry, entry, location, helper, metadata, info)
        }
        Felt252DictConcreteLibfunc::Squash(info) => {
            build_squash(context, registry, entry, location, helper, metadata, info)
        }
    }
}

pub fn build_new<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let segment_arena = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let value_type_id = match registry.get_type(&info.signature.branch_signatures[0].vars[1].ty)? {
        CoreTypeConcrete::Felt252Dict(info) => &info.ty,
        _ => native_panic!("entered unreachable code"),
    };

    let (dup_fn, drop_fn) = {
        let mut dict_overrides = metadata
            .remove::<Felt252DictOverrides>()
            .unwrap_or_default();

        let dup_fn = match dict_overrides.build_dup_fn(
            context,
            helper,
            registry,
            metadata,
            value_type_id,
        )? {
            Some(dup_fn) => Some(
                entry.append_op_result(
                    ods::llvm::mlir_addressof(
                        context,
                        llvm::r#type::pointer(context, 0),
                        dup_fn,
                        location,
                    )
                    .into(),
                )?,
            ),
            None => None,
        };
        let drop_fn = match dict_overrides.build_drop_fn(
            context,
            helper,
            registry,
            metadata,
            value_type_id,
        )? {
            Some(drop_fn_symbol) => Some(
                entry.append_op_result(
                    ods::llvm::mlir_addressof(
                        context,
                        llvm::r#type::pointer(context, 0),
                        drop_fn_symbol,
                        location,
                    )
                    .into(),
                )?,
            ),
            None => None,
        };

        metadata.insert(dict_overrides);
        (dup_fn, drop_fn)
    };

    let runtime_bindings = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.");
    let dict_ptr = runtime_bindings.dict_new(
        context,
        helper,
        entry,
        location,
        dup_fn,
        drop_fn,
        registry.get_type(value_type_id)?.layout(registry)?,
    )?;

    entry.append_operation(helper.br(0, &[segment_arena, dict_ptr], location));
    Ok(())
}

pub fn build_squash<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;
    let gas_builtin = entry.arg(1)?;
    let segment_arena = super::increment_builtin_counter(context, entry, location, entry.arg(2)?)?;
    let dict_ptr = entry.arg(3)?;

    let runtime_bindings = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.");

    let gas_refund = runtime_bindings
        .dict_gas_refund(context, helper, entry, dict_ptr, location)?
        .result(0)?
        .into();

    let new_gas_builtin = entry.addi(gas_builtin, gas_refund, location)?;

    entry.append_operation(helper.br(
        0,
        &[range_check, new_gas_builtin, segment_arena, entry.arg(3)?],
        location,
    ));

    Ok(())
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use cairo_lang_sierra::{
        extensions::{
            felt252_dict::{Felt252DictNewLibfunc, Felt252DictSquashLibfunc},
            int::unsigned::{Uint32Type, Uint64Type},
        },
        program::GenericArg,
    };

    use crate::{
        utils::{
            sierra_gen::SierraGenerator,
            test::{
                jit_dict, jit_enum, jit_struct, load_cairo, run_program_assert_output,
                run_sierra_program,
            },
        },
        values::Value,
    };

    #[test]
    fn run_dict_new() {
        let program = {
            let mut generator = SierraGenerator::<Felt252DictNewLibfunc>::default();

            let u32_ty = generator.push_type_declaration::<Uint32Type>(&[]).clone();

            generator.build(&[GenericArg::Type(u32_ty)])
        };

        let result = run_sierra_program(&program, &[]).return_value;

        let dict = Value::Felt252Dict {
            value: HashMap::new(),
            debug_name: None,
        };

        assert_eq!(result, dict);
    }

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
    fn run_dict_insert_ret_dict() {
        let program = load_cairo!(
            use traits::Default;
            use dict::Felt252DictTrait;

            fn run_test() -> Felt252Dict<u32> {
                let mut dict: Felt252Dict<u32> = Default::default();
                dict.insert(1, 2_u32);
                dict.insert(2, 3_u32);
                dict.insert(3, 4_u32);
                dict.insert(4, 5_u32);
                dict.insert(5, 6_u32);
                dict
            }
        );

        run_program_assert_output(
            &program,
            "run_test",
            &[],
            jit_dict!(
                1 => 2u32,
                2 => 3u32,
                3 => 4u32,
                4 => 5u32,
                5 => 6u32,
            ),
        );
    }

    #[test]
    fn run_dict_squash() {
        let program = {
            let mut generator = SierraGenerator::<Felt252DictSquashLibfunc>::default();

            let u64_ty = generator.push_type_declaration::<Uint64Type>(&[]).clone();

            generator.build(&[GenericArg::Type(u64_ty)])
        };

        let dict = Value::Felt252Dict {
            value: HashMap::new(),
            debug_name: None,
        };

        run_sierra_program(&program, &[dict]);
    }

    #[test]
    fn run_dict_deserialize() {
        let program = load_cairo!(
            use traits::Default;
            use dict::Felt252DictTrait;

            fn run_test(mut dict: Felt252Dict<u32>) -> Felt252Dict<u32> {
                dict
            }
        );

        run_program_assert_output(
            &program,
            "run_test",
            &[jit_dict!(
                1 => 2u32,
                2 => 3u32,
                3 => 4u32,
                4 => 5u32,
                5 => 6u32,
            )],
            jit_dict!(
                1 => 2u32,
                2 => 3u32,
                3 => 4u32,
                4 => 5u32,
                5 => 6u32,
            ),
        );
    }

    #[test]
    fn run_dict_deserialize2() {
        let program = load_cairo!(
            use traits::Default;
            use dict::Felt252DictTrait;

            fn run_test(mut dict: Felt252Dict<u32>) -> (felt252, Felt252Dict<u32>) {
                (0, dict)
            }
        );

        run_program_assert_output(
            &program,
            "run_test",
            &[jit_dict!(
                1 => 2u32,
                2 => 3u32,
                3 => 4u32,
                4 => 5u32,
                5 => 6u32,
            )],
            jit_struct!(
                Value::Felt252(0.into()),
                jit_dict!(
                    1 => 2u32,
                    2 => 3u32,
                    3 => 4u32,
                    4 => 5u32,
                    5 => 6u32,
                )
            ),
        );
    }

    #[test]
    fn run_dict_deserialize_struct() {
        let program = load_cairo! {
            use core::{dict::Felt252DictTrait, nullable::Nullable};

            fn run_test() -> Felt252Dict<Nullable<(u32, u64, u128)>> {
                let mut x: Felt252Dict<Nullable<(u32, u64, u128)>> = Default::default();
                x.insert(0, NullableTrait::new((1_u32, 2_u64, 3_u128)));
                x.insert(1, NullableTrait::new((2_u32, 3_u64, 4_u128)));
                x.insert(2, NullableTrait::new((3_u32, 4_u64, 5_u128)));
                x
            }
        };

        run_program_assert_output(
            &program,
            "run_test",
            &[],
            jit_dict!(
                0 => jit_struct!(1u32.into(), 2u64.into(), 3u128.into()),
                1 => jit_struct!(2u32.into(), 3u64.into(), 4u128.into()),
                2 => jit_struct!(3u32.into(), 4u64.into(), 5u128.into()),
            ),
        );
    }

    #[test]
    fn run_dict_deserialize_enum() {
        let program = load_cairo! {
            use core::{dict::Felt252DictTrait, nullable::Nullable};

            #[derive(Drop)]
            enum MyEnum {
                A: u32,
                B: u64,
                C: u128,
            }

            fn run_test() -> Felt252Dict<Nullable<MyEnum>> {
                let mut x: Felt252Dict<Nullable<MyEnum>> = Default::default();
                x.insert(0, NullableTrait::new(MyEnum::A(1)));
                x.insert(1, NullableTrait::new(MyEnum::B(2)));
                x.insert(2, NullableTrait::new(MyEnum::C(3)));
                x
            }
        };

        run_program_assert_output(
            &program,
            "run_test",
            &[],
            jit_dict!(
                0 => jit_enum!(0, 1u32.into()),
                1 => jit_enum!(1, 2u64.into()),
                2 => jit_enum!(2, 3u128.into()),
            ),
        );
    }
}
