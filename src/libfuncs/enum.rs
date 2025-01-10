//! # Enum-related libfuncs
//!
//! Check out [the enum type](crate::types::enum) for more information on enum layouts.

use super::LibfuncHelper;
use crate::{
    error::{panic::ToNativeAssertError, Error, Result},
    metadata::{enum_snapshot_variants::EnumSnapshotVariantsMeta, MetadataStorage},
    native_assert, native_panic,
    types::TypeBuilder,
    utils::BlockExt,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        enm::{EnumConcreteLibfunc, EnumFromBoundedIntConcreteLibfunc, EnumInitConcreteLibfunc},
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc,
    },
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{arith, cf, llvm, ods},
    ir::{
        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
        r#type::IntegerType,
        Block, Location, Value,
    },
    Context,
};
use std::num::TryFromIntError;

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &EnumConcreteLibfunc,
) -> Result<()> {
    match selector {
        EnumConcreteLibfunc::Init(info) => {
            build_init(context, registry, entry, location, helper, metadata, info)
        }
        EnumConcreteLibfunc::Match(info) => {
            build_match(context, registry, entry, location, helper, metadata, info)
        }
        EnumConcreteLibfunc::SnapshotMatch(info) => {
            build_snapshot_match(context, registry, entry, location, helper, metadata, info)
        }
        EnumConcreteLibfunc::FromBoundedInt(info) => {
            build_from_bounded_int(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `enum_init` libfunc.
pub fn build_init<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &EnumInitConcreteLibfunc,
) -> Result<()> {
    #[cfg(feature = "with-debug-utils")]
    if let Some(auto_breakpoint) =
        metadata.get::<crate::metadata::auto_breakpoint::AutoBreakpoint>()
    {
        auto_breakpoint.maybe_breakpoint(
            entry,
            location,
            metadata,
            &crate::metadata::auto_breakpoint::BreakpointEvent::EnumInit {
                type_id: info.signature.branch_signatures[0].vars[0].ty.clone(),
                variant_idx: info.index,
            },
        )?;
    }

    let val = build_enum_value(
        context,
        registry,
        entry,
        location,
        helper,
        metadata,
        entry.arg(0)?,
        &info.branch_signatures()[0].vars[0].ty,
        &info.signature.param_signatures[0].ty,
        info.index,
    )?;
    entry.append_operation(helper.br(0, &[val], location));

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn build_enum_value<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    payload_value: Value<'ctx, 'this>,
    enum_type: &ConcreteTypeId,
    variant_type: &ConcreteTypeId,
    variant_index: usize,
) -> Result<Value<'ctx, 'this>> {
    let type_info = registry.get_type(enum_type)?;
    let payload_type_info = registry.get_type(variant_type)?;

    let (layout, (tag_ty, _), variant_tys) = crate::types::r#enum::get_type_for_variants(
        context,
        helper,
        registry,
        metadata,
        type_info
            .variants()
            .to_native_assert_error("found non-enum type where an enum is required")?,
    )?;

    Ok(match variant_tys.len() {
        0 => native_panic!("attempt to initialize a zero-variant enum"),
        1 => payload_value,
        _ => {
            let enum_ty = llvm::r#type::r#struct(
                context,
                &[
                    tag_ty,
                    if payload_type_info.is_zst(registry)? {
                        llvm::r#type::array(IntegerType::new(context, 8).into(), 0)
                    } else {
                        variant_tys[variant_index].0
                    },
                ],
                false,
            );

            let tag_val = entry
                .append_operation(arith::constant(
                    context,
                    IntegerAttribute::new(
                        tag_ty,
                        variant_index
                            .try_into()
                            .expect("couldnt convert index to i64"),
                    )
                    .into(),
                    location,
                ))
                .result(0)?
                .into();

            let val = entry.append_op_result(llvm::undef(enum_ty, location))?;
            let val = entry.insert_value(context, location, val, tag_val, 0)?;

            let mut val = if payload_type_info.is_zst(registry)? {
                val
            } else {
                entry.insert_value(context, location, val, payload_value, 1)?
            };

            if type_info.is_memory_allocated(registry)? {
                let stack_ptr = helper.init_block().alloca1(
                    context,
                    location,
                    type_info.build(context, helper, registry, metadata, enum_type)?,
                    layout.align(),
                )?;

                // Convert the enum from the concrete variant to the internal representation.
                entry.store(context, location, stack_ptr, val)?;
                val = entry.load(
                    context,
                    location,
                    stack_ptr,
                    type_info.build(context, helper, registry, metadata, enum_type)?,
                )?;
            };

            val
        }
    })
}

/// Generate MLIR operations for the `enum_from_bounded_int` libfunc.
pub fn build_from_bounded_int<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &EnumFromBoundedIntConcreteLibfunc,
) -> Result<()> {
    let inp_ty = registry.get_type(&info.param_signatures()[0].ty)?;
    let varaint_selector_type: IntegerType = inp_ty
        .build(
            context,
            helper,
            registry,
            metadata,
            &info.param_signatures()[0].ty,
        )?
        .try_into()?;
    let enum_type = registry.get_type(&info.branch_signatures()[0].vars[0].ty)?;
    // we assume its never memory allocated since its always an enum with only a tag
    native_assert!(
        !enum_type.is_memory_allocated(registry)?,
        "an enum with only a tag should not be allocated in memory"
    );

    let enum_ty = enum_type.build(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;

    let tag_bits = info.n_variants.next_power_of_two().trailing_zeros();
    let tag_type = IntegerType::new(context, tag_bits);

    let mut tag_value: Value = entry.arg(0)?;

    match tag_type.width().cmp(&varaint_selector_type.width()) {
        std::cmp::Ordering::Less => {
            tag_value = entry.append_op_result(
                ods::llvm::trunc(context, tag_type.into(), tag_value, location).into(),
            )?;
        }
        std::cmp::Ordering::Equal => {}
        std::cmp::Ordering::Greater => {
            tag_value = entry.append_op_result(
                ods::llvm::zext(context, tag_type.into(), tag_value, location).into(),
            )?;
        }
    };

    let value = entry.append_op_result(llvm::undef(enum_ty, location))?;
    let value = entry.insert_value(context, location, value, tag_value, 0)?;

    entry.append_operation(helper.br(0, &[value], location));

    Ok(())
}

/// Generate MLIR operations for the `enum_match` libfunc.
pub fn build_match<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let type_info = registry.get_type(&info.param_signatures()[0].ty)?;

    let variant_ids = type_info
        .variants()
        .to_native_assert_error("found non-enum type where an enum is required")?;
    match variant_ids.len() {
        0 => {
            // The Cairo compiler will generate an enum match for enums without variants, so this
            // case cannot be a compile-time error. We're assuming that even though it's been
            // generated, it's just dead code and can be made into an assertion that always fails.

            let k0 = entry.const_int(context, location, 0, 1)?;
            entry.append_operation(cf::assert(
                context,
                k0,
                "attempt to match a zero-variant enum",
                location,
            ));

            entry.append_operation(llvm::unreachable(location));
        }
        1 => {
            entry.append_operation(helper.br(0, &[entry.arg(0)?], location));
        }
        _ => {
            let (layout, (tag_ty, _), variant_tys) = crate::types::r#enum::get_type_for_variants(
                context,
                helper,
                registry,
                metadata,
                variant_ids,
            )?;

            let (stack_ptr, tag_val) = if type_info.is_memory_allocated(registry)? {
                let stack_ptr = helper.init_block().alloca1(
                    context,
                    location,
                    type_info.build(
                        context,
                        helper,
                        registry,
                        metadata,
                        &info.param_signatures()[0].ty,
                    )?,
                    layout.align(),
                )?;
                entry.store(context, location, stack_ptr, entry.arg(0)?)?;
                let tag_val = entry.load(context, location, stack_ptr, tag_ty)?;

                (Some(stack_ptr), tag_val)
            } else {
                let tag_val = entry
                    .append_operation(llvm::extract_value(
                        context,
                        entry.arg(0)?,
                        DenseI64ArrayAttribute::new(context, &[0]),
                        tag_ty,
                        location,
                    ))
                    .result(0)?
                    .into();

                (None, tag_val)
            };

            let default_block = helper.append_block(Block::new(&[]));
            let variant_blocks = variant_tys
                .iter()
                .map(|_| helper.append_block(Block::new(&[])))
                .collect::<Vec<_>>();

            let case_values = (0..variant_tys.len())
                .map(i64::try_from)
                .collect::<std::result::Result<Vec<_>, TryFromIntError>>()?;

            entry.append_operation(cf::switch(
                context,
                &case_values,
                tag_val,
                tag_ty,
                (default_block, &[]),
                &variant_blocks
                    .iter()
                    .copied()
                    .map(|block| (block, [].as_slice()))
                    .collect::<Vec<_>>(),
                location,
            )?);

            // Default block.
            {
                let val = default_block
                    .append_operation(arith::constant(
                        context,
                        IntegerAttribute::new(IntegerType::new(context, 1).into(), 0).into(),
                        location,
                    ))
                    .result(0)?
                    .into();

                default_block.append_operation(cf::assert(
                    context,
                    val,
                    "Invalid enum tag.",
                    location,
                ));
                default_block.append_operation(llvm::unreachable(location));
            }

            // Enum variants.
            for (i, (block, (payload_ty, _))) in
                variant_blocks.into_iter().zip(variant_tys).enumerate()
            {
                let enum_ty = llvm::r#type::r#struct(context, &[tag_ty, payload_ty], false);

                let payload_val = match stack_ptr {
                    Some(stack_ptr) => {
                        let val = block.load(context, location, stack_ptr, enum_ty)?;
                        block.extract_value(context, location, val, payload_ty, 1)?
                    }
                    None => {
                        // If the enum is not memory-allocated it means that:
                        //   - Either it's a C-style enum and all payloads have the same type.
                        //   - Or the enum only has a single non-memory-allocated variant.
                        if variant_ids.len() == 1 {
                            entry.arg(0)?
                        } else {
                            native_assert!(
                                registry.get_type(&variant_ids[i])?.is_zst(registry)?,
                                "should be zero sized"
                            );
                            block
                                .append_operation(llvm::undef(payload_ty, location))
                                .result(0)?
                                .into()
                        }
                    }
                };

                block.append_operation(helper.br(i, &[payload_val], location));
            }
        }
    }

    Ok(())
}

/// Generate MLIR operations for the `enum_snapshot_match` libfunc.
pub fn build_snapshot_match<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let type_info = registry.get_type(&info.param_signatures()[0].ty)?;

    // This libfunc's implementation is identical to `enum_match` aside from fetching the snapshotted enum's variants from the metadata:
    let variant_ids = metadata
        .get::<EnumSnapshotVariantsMeta>()
        .ok_or(Error::MissingMetadata)?
        .get_variants(&info.param_signatures()[0].ty)
        .expect("enum should always have variants")
        .clone();
    match variant_ids.len() {
        0 => {
            // The Cairo compiler will generate an enum match for enums without variants, so this
            // case cannot be a compile-time error. We're assuming that even though it's been
            // generated, it's just dead code and can be made into an assertion that always fails.

            let k0 = entry.const_int(context, location, 0, 1)?;
            entry.append_operation(cf::assert(
                context,
                k0,
                "attempt to match a zero-variant enum",
                location,
            ));

            entry.append_operation(llvm::unreachable(location));
        }
        1 => {
            entry.append_operation(helper.br(0, &[entry.arg(0)?], location));
        }
        _ => {
            let (layout, (tag_ty, _), variant_tys) = crate::types::r#enum::get_type_for_variants(
                context,
                helper,
                registry,
                metadata,
                &variant_ids,
            )?;

            let (stack_ptr, tag_val) = if type_info.is_memory_allocated(registry)? {
                let stack_ptr = helper.init_block().alloca1(
                    context,
                    location,
                    type_info.build(
                        context,
                        helper,
                        registry,
                        metadata,
                        &info.param_signatures()[0].ty,
                    )?,
                    layout.align(),
                )?;
                entry.store(context, location, stack_ptr, entry.arg(0)?)?;
                let tag_val = entry.load(context, location, stack_ptr, tag_ty)?;

                (Some(stack_ptr), tag_val)
            } else {
                let tag_val = entry.extract_value(context, location, entry.arg(0)?, tag_ty, 0)?;

                (None, tag_val)
            };

            let default_block = helper.append_block(Block::new(&[]));
            let variant_blocks = variant_tys
                .iter()
                .map(|_| helper.append_block(Block::new(&[])))
                .collect::<Vec<_>>();

            let case_values = (0..variant_tys.len())
                .map(i64::try_from)
                .collect::<std::result::Result<Vec<_>, TryFromIntError>>()?;

            entry.append_operation(cf::switch(
                context,
                &case_values,
                tag_val,
                tag_ty,
                (default_block, &[]),
                &variant_blocks
                    .iter()
                    .copied()
                    .map(|block| (block, [].as_slice()))
                    .collect::<Vec<_>>(),
                location,
            )?);

            // Default block.
            {
                let val = default_block.const_int(context, location, 0, 1)?;

                default_block.append_operation(cf::assert(
                    context,
                    val,
                    "Invalid enum tag.",
                    location,
                ));
                default_block.append_operation(llvm::unreachable(location));
            }

            // Enum variants.
            for (i, (block, (payload_ty, _))) in
                variant_blocks.into_iter().zip(variant_tys).enumerate()
            {
                let enum_ty = llvm::r#type::r#struct(context, &[tag_ty, payload_ty], false);

                let payload_val = match stack_ptr {
                    Some(stack_ptr) => {
                        let val = block.load(context, location, stack_ptr, enum_ty)?;
                        block.extract_value(context, location, val, payload_ty, 1)?
                    }
                    None => {
                        // If the enum is not memory-allocated it means that:
                        //   - Either it's a C-style enum and all payloads have the same type.
                        //   - Or the enum only has a single non-memory-allocated variant.
                        if variant_ids.len() == 1 {
                            entry.arg(0)?
                        } else {
                            native_assert!(
                                registry.get_type(&variant_ids[i])?.is_zst(registry)?,
                                "should be zero sized"
                            );
                            block.append_op_result(llvm::undef(payload_ty, location))?
                        }
                    }
                };

                block.append_operation(helper.br(i, &[payload_val], location));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{
        context::NativeContext,
        utils::test::{jit_enum, jit_struct, run_sierra_program},
        Value,
    };
    use cairo_lang_sierra::{program::Program, ProgramParser};
    use lazy_static::lazy_static;
    use starknet_types_core::felt::Felt;

    lazy_static! {
        // enum MySmallEnum {
        //     A: felt252,
        // }
        // enum MyEnum {
        //     A: felt252,
        //     B: u8,
        //     C: u16,
        //     D: u32,
        //     E: u64,
        // }
        // fn run_test() -> (MySmallEnum, MyEnum, MyEnum, MyEnum, MyEnum, MyEnum) {
        //     (
        //         MySmallEnum::A(-1),
        //         MyEnum::A(5678),
        //         MyEnum::B(90),
        //         MyEnum::C(9012),
        //         MyEnum::D(34567890),
        //         MyEnum::E(1234567890123456),
        //     )
        // }
        static ref ENUM_INIT: Program = ProgramParser::new().parse(r#"
            type [0] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = Enum<ut@program::program::MySmallEnum, [0]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [2] = u8 [storable: true, drop: true, dup: true, zero_sized: false];
            type [3] = u16 [storable: true, drop: true, dup: true, zero_sized: false];
            type [4] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
            type [5] = u64 [storable: true, drop: true, dup: true, zero_sized: false];
            type [6] = Enum<ut@program::program::MyEnum, [0], [2], [3], [4], [5]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [7] = Struct<ut@Tuple, [1], [6], [6], [6], [6], [6]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [13] = Const<[5], 1234567890123456> [storable: false, drop: false, dup: false, zero_sized: false];
            type [12] = Const<[4], 34567890> [storable: false, drop: false, dup: false, zero_sized: false];
            type [11] = Const<[3], 9012> [storable: false, drop: false, dup: false, zero_sized: false];
            type [10] = Const<[2], 90> [storable: false, drop: false, dup: false, zero_sized: false];
            type [9] = Const<[0], 5678> [storable: false, drop: false, dup: false, zero_sized: false];
            type [8] = Const<[0], -1> [storable: false, drop: false, dup: false, zero_sized: false];

            libfunc [7] = const_as_immediate<[8]>;
            libfunc [6] = enum_init<[1], 0>;
            libfunc [8] = const_as_immediate<[9]>;
            libfunc [5] = enum_init<[6], 0>;
            libfunc [9] = const_as_immediate<[10]>;
            libfunc [4] = enum_init<[6], 1>;
            libfunc [10] = const_as_immediate<[11]>;
            libfunc [3] = enum_init<[6], 2>;
            libfunc [11] = const_as_immediate<[12]>;
            libfunc [2] = enum_init<[6], 3>;
            libfunc [12] = const_as_immediate<[13]>;
            libfunc [1] = enum_init<[6], 4>;
            libfunc [0] = struct_construct<[7]>;
            libfunc [13] = store_temp<[7]>;

            [7]() -> ([0]); // 0
            [6]([0]) -> ([1]); // 1
            [8]() -> ([2]); // 2
            [5]([2]) -> ([3]); // 3
            [9]() -> ([4]); // 4
            [4]([4]) -> ([5]); // 5
            [10]() -> ([6]); // 6
            [3]([6]) -> ([7]); // 7
            [11]() -> ([8]); // 8
            [2]([8]) -> ([9]); // 9
            [12]() -> ([10]); // 10
            [1]([10]) -> ([11]); // 11
            [0]([1], [3], [5], [7], [9], [11]) -> ([12]); // 12
            [13]([12]) -> ([12]); // 13
            return([12]); // 14

            [0]@0() -> ([7]);
        "#).map_err(|e| e.to_string()).unwrap();
        // enum MyEnum {
        //     A: felt252,
        //     B: u8,
        //     C: u16,
        //     D: u32,
        //     E: u64,
        // }
        // fn match_a() -> felt252 {
        //     let x = MyEnum::A(5);
        //     match x {
        //         MyEnum::A(x) => x,
        //         MyEnum::B(_) => 0,
        //         MyEnum::C(_) => 1,
        //         MyEnum::D(_) => 2,
        //         MyEnum::E(_) => 3,
        //     }
        // }
        static ref ENUM_MATCH_FELT: Program = ProgramParser::new().parse(r#"
            type [0] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = Const<[0], 5> [storable: false, drop: false, dup: false, zero_sized: false];

            libfunc [0] = const_as_immediate<[1]>;
            libfunc [1] = store_temp<[0]>;

            [0]() -> ([0]); // 0
            [1]([0]) -> ([0]); // 1
            return([0]); // 2

            [0]@0() -> ([0]);
        "#).map_err(|e| e.to_string()).unwrap();
        // enum MyEnum {
        //     A: felt252,
        //     B: u8,
        //     C: u16,
        //     D: u32,
        //     E: u64,
        // }
        // fn match_b() -> u8 {
        //     let x = MyEnum::B(5_u8);
        //     match x {
        //         MyEnum::A(_) => 0_u8,
        //         MyEnum::B(x) => x,
        //         MyEnum::C(_) => 1_u8,
        //         MyEnum::D(_) => 2_u8,
        //         MyEnum::E(_) => 3_u8,
        //     }
        // }
        static ref ENUM_MATCH_U8: Program = ProgramParser::new().parse(r#"
            type [0] = u8 [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = Const<[0], 5> [storable: false, drop: false, dup: false, zero_sized: false];

            libfunc [0] = const_as_immediate<[1]>;
            libfunc [1] = store_temp<[0]>;

            [0]() -> ([0]); // 0
            [1]([0]) -> ([0]); // 1
            return([0]); // 2

            [0]@0() -> ([0]);
        "#).map_err(|e| e.to_string()).unwrap();
    }

    #[test]
    fn enum_init() {
        let return_value = run_sierra_program(&ENUM_INIT, &[]).return_value;

        assert_eq!(
            jit_struct!(
                jit_enum!(0, Felt::from(-1).into()),
                jit_enum!(0, Felt::from(5678).into()),
                jit_enum!(1, 90u8.into()),
                jit_enum!(2, 9012u16.into()),
                jit_enum!(3, 34567890u32.into()),
                jit_enum!(4, 1234567890123456u64.into()),
            ),
            return_value
        )
    }

    #[test]
    fn enum_match() {
        let return_value1 = run_sierra_program(&ENUM_MATCH_FELT, &[]).return_value;
        assert_eq!(Value::from(Felt::from(5)), return_value1);
        let return_value2 = run_sierra_program(&ENUM_MATCH_U8, &[]).return_value;
        assert_eq!(Value::from(5u8), return_value2)
    }

    #[test]
    fn compile_enum_match_without_variants() {
        // enum MyEnum {}
        // fn main(value: MyEnum) {
        //     match value {}
        // }
        let program = ProgramParser::new().parse(r#"
            type [0] = Enum<ut@program::program::MyEnum> [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];

            libfunc [0] = enum_match<[0]>;

            [0]([0]) { }; // 0

            [0]@0([0]: [0]) -> ();
        "#).map_err(|e| e.to_string()).unwrap();

        let native_context = NativeContext::new();
        native_context
            .compile(&program, false, Some(Default::default()))
            .unwrap();
    }
}
