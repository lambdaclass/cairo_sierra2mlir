//! # Const libfuncs

use super::LibfuncHelper;
use crate::{
    error::{Error, Result},
    libfuncs::{r#enum::build_enum_value, r#struct::build_struct_value},
    metadata::{realloc_bindings::ReallocBindingsMeta, MetadataStorage},
    native_panic,
    types::TypeBuilder,
    utils::{BlockExt, ProgramRegistryExt, RangeExt, PRIME},
};
use cairo_lang_sierra::{
    extensions::{
        bounded_int::BoundedIntConcreteType,
        const_type::{
            ConstAsBoxConcreteLibfunc, ConstAsImmediateConcreteLibfunc, ConstConcreteLibfunc,
            ConstConcreteType,
        },
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
    },
    program::GenericArg,
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm::{self, r#type::pointer},
    ir::{Block, Location, Value},
    Context,
};
use num_bigint::Sign;

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &ConstConcreteLibfunc,
) -> Result<()> {
    match selector {
        ConstConcreteLibfunc::AsBox(info) => {
            build_const_as_box(context, registry, entry, location, helper, metadata, info)
        }
        ConstConcreteLibfunc::AsImmediate(info) => {
            build_const_as_immediate(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `const_as_box` libfunc.
pub fn build_const_as_box<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &ConstAsBoxConcreteLibfunc,
) -> Result<()> {
    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, helper));
    }

    let const_type_outer = registry.get_type(&info.const_type)?;

    // Create constant
    let const_type = match &const_type_outer {
        CoreTypeConcrete::Const(inner) => inner,
        _ => native_panic!("matched an unexpected CoreTypeConcrete that is not a Const"),
    };

    let value = build_const_type_value(
        context, registry, entry, location, helper, metadata, const_type,
    )?;

    let const_ty = registry.get_type(&const_type.inner_ty)?;
    let inner_layout = const_ty.layout(registry)?;

    // Create box
    let value_len = entry.const_int(context, location, inner_layout.pad_to_align().size(), 64)?;

    let ptr = entry.append_op_result(llvm::zero(pointer(context, 0), location))?;
    let ptr = entry.append_op_result(ReallocBindingsMeta::realloc(
        context, ptr, value_len, location,
    )?)?;

    // Store constant in box
    entry.store(context, location, ptr, value)?;

    entry.append_operation(helper.br(0, &[ptr], location));
    Ok(())
}

/// Generate MLIR operations for the `const_as_immediate` libfunc.
pub fn build_const_as_immediate<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &ConstAsImmediateConcreteLibfunc,
) -> Result<()> {
    let const_ty = registry.get_type(&info.const_type)?;

    let const_type = match &const_ty {
        CoreTypeConcrete::Const(inner) => inner,
        _ => native_panic!("matched an unexpected CoreTypeConcrete that is not a Const"),
    };

    let value = build_const_type_value(
        context, registry, entry, location, helper, metadata, const_type,
    )?;

    entry.append_operation(helper.br(0, &[value], location));
    Ok(())
}

pub fn build_const_type_value<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &ConstConcreteType,
) -> Result<Value<'ctx, 'this>> {
    // const_type.inner_data Should be one of the following:
    // - A single value, if the inner type is a simple numeric type (e.g., `felt252`, `u32`,
    //   etc.).
    // - A list of const types, if the inner type is a struct. The type of each const type must be
    //   the same as the corresponding struct member type.
    // - A selector (a single value) followed by a const type, if the inner type is an enum. The
    //   type of the const type must be the same as the corresponding enum variant type.

    let inner_type = registry.get_type(&info.inner_ty)?;
    let inner_ty = registry.build_type(context, helper, metadata, &info.inner_ty)?;

    match inner_type {
        CoreTypeConcrete::Struct(_) => {
            let mut fields = Vec::new();

            for field in &info.inner_data {
                match field {
                    GenericArg::Type(const_field_ty) => {
                        let field_type = registry.get_type(const_field_ty)?;

                        let const_field_type = match &field_type {
                            CoreTypeConcrete::Const(inner) => inner,
                            _ => native_panic!(
                                "matched an unexpected CoreTypeConcrete that is not a Const"
                            ),
                        };

                        let field_value = build_const_type_value(
                            context,
                            registry,
                            entry,
                            location,
                            helper,
                            metadata,
                            const_field_type,
                        )?;
                        fields.push(field_value);
                    }
                    _ => return Err(Error::ConstDataMismatch),
                }
            }

            build_struct_value(
                context,
                registry,
                entry,
                location,
                helper,
                metadata,
                &info.inner_ty,
                &fields,
            )
        }
        CoreTypeConcrete::Enum(_enum_info) => match &info.inner_data[..] {
            [GenericArg::Value(variant_index), GenericArg::Type(payload_ty)] => {
                let payload_type = registry.get_type(payload_ty)?;
                let const_payload_type = match payload_type {
                    CoreTypeConcrete::Const(inner) => inner,
                    _ => {
                        native_panic!("matched an unexpected CoreTypeConcrete that is not a Const")
                    }
                };

                let payload_value = build_const_type_value(
                    context,
                    registry,
                    entry,
                    location,
                    helper,
                    metadata,
                    const_payload_type,
                )?;

                build_enum_value(
                    context,
                    registry,
                    entry,
                    location,
                    helper,
                    metadata,
                    payload_value,
                    &info.inner_ty,
                    payload_ty,
                    variant_index
                        .try_into()
                        .map_err(|_| Error::IntegerConversion)?,
                )
            }
            _ => Err(Error::ConstDataMismatch),
        },
        CoreTypeConcrete::NonZero(_) => match &info.inner_data[..] {
            // Copied from the sierra to casm lowering
            // NonZero is the same type as the inner type in native.
            [GenericArg::Type(inner)] => {
                let inner_type = registry.get_type(inner)?;
                let const_inner_type = match inner_type {
                    CoreTypeConcrete::Const(inner) => inner,
                    _ => native_panic!("unreachable: unexpected CoreTypeConcrete found"),
                };

                build_const_type_value(
                    context,
                    registry,
                    entry,
                    location,
                    helper,
                    metadata,
                    const_inner_type,
                )
            }
            _ => Err(Error::ConstDataMismatch),
        },
        CoreTypeConcrete::BoundedInt(BoundedIntConcreteType { range, .. }) => {
            let value = match &info.inner_data.as_slice() {
                [GenericArg::Value(value)] => value.clone(),
                _ => return Err(Error::ConstDataMismatch),
            };

            // Offset the value so that 0 matches with lower.
            let value = &value - &range.lower;

            entry.const_int(
                context,
                location,
                value,
                inner_type.integer_range(registry)?.offset_bit_width(),
            )
        }
        CoreTypeConcrete::Felt252(_) => {
            let value = match &info.inner_data.as_slice() {
                [GenericArg::Value(value)] => value.clone(),
                _ => return Err(Error::ConstDataMismatch),
            };

            let (sign, value) = value.into_parts();
            let value = match sign {
                Sign::Minus => PRIME.clone() - value,
                _ => value,
            };

            entry.const_int_from_type(context, location, value, inner_ty)
        }
        CoreTypeConcrete::Uint8(_)
        | CoreTypeConcrete::Uint16(_)
        | CoreTypeConcrete::Uint32(_)
        | CoreTypeConcrete::Uint64(_)
        | CoreTypeConcrete::Uint128(_)
        | CoreTypeConcrete::Sint8(_)
        | CoreTypeConcrete::Sint16(_)
        | CoreTypeConcrete::Sint32(_)
        | CoreTypeConcrete::Sint64(_)
        | CoreTypeConcrete::Sint128(_)
        | CoreTypeConcrete::Bytes31(_) => match &info.inner_data.as_slice() {
            [GenericArg::Value(value)] => {
                entry.const_int_from_type(context, location, value.clone(), inner_ty)
            }
            _ => Err(Error::ConstDataMismatch),
        },
        _ => todo!("const for type {}", info.inner_ty),
    }
}

#[cfg(test)]
pub mod test {
    use cairo_lang_sierra::ProgramParser;

    use crate::{
        utils::test::{jit_struct, run_sierra_program},
        values::Value,
    };

    #[test]
    fn run_const_as_box() {
        // use core::box::BoxTrait;
        // struct Hello {
        //     x: i32,
        // }
        // fn run_test() -> Hello {
        //     let x = BoxTrait::new(Hello {
        //         x: -2
        //     });
        //     x.unbox()
        // }
        let program = ProgramParser::new().parse(r#"
            type [2] = Box<[1]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [0] = i32 [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = Struct<ut@program::program::Hello, [0]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [4] = Const<[1], [3]> [storable: false, drop: false, dup: false, zero_sized: false];
            type [3] = Const<[0], -2> [storable: false, drop: false, dup: false, zero_sized: false];

            libfunc [1] = const_as_box<[4], 0>;
            libfunc [0] = unbox<[1]>;
            libfunc [2] = store_temp<[1]>;

            [1]() -> ([0]); // 0
            [0]([0]) -> ([1]); // 1
            [2]([1]) -> ([1]); // 2
            return([1]); // 3

            [0]@0() -> ([1]);
        "#).map_err(|e| e.to_string()).unwrap();

        let result = run_sierra_program(&program, &[]).return_value;

        assert_eq!(result, jit_struct!(Value::Sint32(-2)));
    }
}
