//! # Nullable libfuncs
//!
//! Like a Box but it can be null.

use super::LibfuncHelper;
use crate::{error::Result, metadata::MetadataStorage, utils::BlockExt};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        lib_func::{SignatureAndTypeConcreteLibfunc, SignatureOnlyConcreteLibfunc},
        nullable::NullableConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{cf, llvm::r#type::pointer, ods},
    ir::{
        attribute::IntegerAttribute, operation::OperationBuilder, r#type::IntegerType, Block,
        Identifier, Location,
    },
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
    selector: &NullableConcreteLibfunc,
) -> Result<()> {
    match selector {
        NullableConcreteLibfunc::ForwardSnapshot(info)
        | NullableConcreteLibfunc::NullableFromBox(info) => super::build_noop::<1, true>(
            context,
            registry,
            entry,
            location,
            helper,
            metadata,
            &info.signature.param_signatures,
        ),
        NullableConcreteLibfunc::MatchNullable(info) => {
            build_match_nullable(context, registry, entry, location, helper, metadata, info)
        }
        NullableConcreteLibfunc::Null(info) => {
            build_null(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `null` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_null<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let value = entry
        .append_op_result(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())?;

    entry.append_operation(helper.br(0, &[value], location));
    Ok(())
}

/// Generate MLIR operations for the `match_nullable` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_match_nullable<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let arg = entry.arg(0)?;

    let nullptr = entry
        .append_op_result(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())?;

    let is_null_ptr = entry.append_op_result(
        OperationBuilder::new("llvm.icmp", location)
            .add_operands(&[arg, nullptr])
            .add_attributes(&[(
                Identifier::new(context, "predicate"),
                IntegerAttribute::new(IntegerType::new(context, 64).into(), 0).into(),
            )])
            .add_results(&[IntegerType::new(context, 1).into()])
            .build()?,
    )?;

    let block_is_null = helper.append_block(Block::new(&[]));
    let block_is_not_null = helper.append_block(Block::new(&[]));

    entry.append_operation(cf::cond_br(
        context,
        is_null_ptr,
        block_is_null,
        block_is_not_null,
        &[],
        &[],
        location,
    ));

    block_is_null.append_operation(helper.br(0, &[], location));
    block_is_not_null.append_operation(helper.br(1, &[arg], location));

    Ok(())
}

#[cfg(test)]
mod test {
    use cairo_lang_sierra::ProgramParser;

    use crate::{
        utils::test::{jit_enum, jit_struct, run_sierra_program},
        values::Value,
    };

    #[test]
    fn run_null() {
        // use nullable::null;
        // use nullable::match_nullable;
        // use nullable::FromNullableResult;
        // use nullable::nullable_from_box;
        // use box::BoxTrait;
        // fn run_test() {
        //     let _a: Nullable<u8> = null();
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
            type [1] = Nullable<[0]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [2] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
            type [0] = u8 [storable: true, drop: true, dup: true, zero_sized: false];

            libfunc [0] = null<[0]>;
            libfunc [2] = drop<[1]>;

            [0]() -> ([0]); // 0
            [2]([0]) -> (); // 1
            return(); // 2

            [0]@0() -> ();
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let return_value = run_sierra_program(&program, &[]).return_value;

        assert_eq!(jit_struct!(), return_value)
    }

    #[test]
    fn run_null_jit() {
        // use nullable::null;
        // use nullable::match_nullable;
        // use nullable::FromNullableResult;
        // use nullable::nullable_from_box;
        // use box::BoxTrait;
        // fn run_test() -> Nullable<u8> {
        //     let a: Nullable<u8> = null();
        //     a
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [1] = Nullable<[0]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [0] = u8 [storable: true, drop: true, dup: true, zero_sized: false];

                libfunc [0] = null<[0]>;
                libfunc [2] = store_temp<[1]>;

                [0]() -> ([0]); // 0
                [2]([0]) -> ([0]); // 1
                return([0]); // 2

                [0]@0() -> ([1]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let return_value = run_sierra_program(&program, &[]).return_value;

        assert_eq!(Value::Null, return_value);
    }

    #[test]
    fn run_not_null() {
        // use nullable::null;
        // use nullable::match_nullable;
        // use nullable::FromNullableResult;
        // use nullable::nullable_from_box;
        // use box::BoxTrait;
        // fn run_test(x: u8) -> u8 {
        //     let b: Box<u8> = BoxTrait::new(x);
        //     let c = if x == 0 {
        //         null()
        //     } else {
        //         nullable_from_box(b)
        //     };
        //     let d = match match_nullable(c) {
        //         FromNullableResult::Null(_) => 99_u8,
        //         FromNullableResult::NotNull(value) => value.unbox()
        //     };
        //     d
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
            type [0] = u8 [storable: true, drop: true, dup: true, zero_sized: false];
            type [4] = Const<[0], 99> [storable: false, drop: false, dup: false, zero_sized: false];
            type [3] = Nullable<[0]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [2] = NonZero<[0]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = Box<[0]> [storable: true, drop: true, dup: true, zero_sized: false];

            libfunc [7] = dup<[0]>;
            libfunc [5] = into_box<[0]>;
            libfunc [4] = u8_is_zero;
            libfunc [8] = branch_align;
            libfunc [9] = drop<[1]>;
            libfunc [3] = null<[0]>;
            libfunc [13] = store_temp<[3]>;
            libfunc [10] = jump;
            libfunc [11] = drop<[2]>;
            libfunc [2] = nullable_from_box<[0]>;
            libfunc [1] = match_nullable<[0]>;
            libfunc [12] = const_as_immediate<[4]>;
            libfunc [14] = store_temp<[0]>;
            libfunc [0] = unbox<[0]>;

            [7]([0]) -> ([0], [1]); // 0
            [5]([1]) -> ([2]); // 1
            [4]([0]) { fallthrough() 8([3]) }; // 2
            [8]() -> (); // 3
            [9]([2]) -> (); // 4
            [3]() -> ([4]); // 5
            [13]([4]) -> ([5]); // 6
            [10]() { 12() }; // 7
            [8]() -> (); // 8
            [11]([3]) -> (); // 9
            [2]([2]) -> ([6]); // 10
            [13]([6]) -> ([5]); // 11
            [1]([5]) { fallthrough() 17([7]) }; // 12
            [8]() -> (); // 13
            [12]() -> ([8]); // 14
            [14]([8]) -> ([8]); // 15
            return([8]); // 16
            [8]() -> (); // 17
            [0]([7]) -> ([9]); // 18
            [14]([9]) -> ([9]); // 19
            return([9]); // 20

            [0]@0([0]: [0]) -> ([0]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let return_value1 = run_sierra_program(&program, &[4u8.into()]).return_value;
        let return_value2 = run_sierra_program(&program, &[0u8.into()]).return_value;

        assert_eq!(Value::from(4u8), return_value1);
        assert_eq!(Value::from(99u8), return_value2);
    }

    #[test]
    fn match_snapshot_nullable_clone_bug() {
        // use core::{NullableTrait, match_nullable, null, nullable::FromNullableResult};
        // fn run_test(x: Option<u8>) -> Option<u8> {
        //     let a = match x {
        //         Option::Some(x) => @NullableTrait::new(x),
        //         Option::None(_) => @null::<u8>(),
        //     };
        //     let b = *a;
        //     match match_nullable(b) {
        //         FromNullableResult::Null(_) => Option::None(()),
        //         FromNullableResult::NotNull(x) => Option::Some(x.unbox()),
        //     }
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [0] = u8 [storable: true, drop: true, dup: true, zero_sized: false];
                type [1] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
                type [2] = Enum<ut@core::option::Option::<core::integer::u8>, [0], [1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [4] = Nullable<[0]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [3] = Box<[0]> [storable: true, drop: true, dup: true, zero_sized: false];

                libfunc [8] = enum_match<[2]>;
                libfunc [9] = branch_align;
                libfunc [7] = into_box<[0]>;
                libfunc [6] = nullable_from_box<[0]>;
                libfunc [10] = snapshot_take<[4]>;
                libfunc [11] = drop<[4]>;
                libfunc [14] = rename<[4]>;
                libfunc [12] = jump;
                libfunc [13] = drop<[1]>;
                libfunc [5] = null<[0]>;
                libfunc [15] = store_temp<[4]>;
                libfunc [4] = match_nullable<[0]>;
                libfunc [3] = struct_construct<[1]>;
                libfunc [2] = enum_init<[2], 1>;
                libfunc [16] = store_temp<[2]>;
                libfunc [1] = unbox<[0]>;
                libfunc [0] = enum_init<[2], 0>;

                [8]([0]) { fallthrough([1]) 8([2]) }; // 0
                [9]() -> (); // 1
                [7]([1]) -> ([3]); // 2
                [6]([3]) -> ([4]); // 3
                [10]([4]) -> ([5], [6]); // 4
                [11]([5]) -> (); // 5
                [14]([6]) -> ([7]); // 6
                [12]() { 14() }; // 7
                [9]() -> (); // 8
                [13]([2]) -> (); // 9
                [5]() -> ([8]); // 10
                [10]([8]) -> ([9], [10]); // 11
                [11]([9]) -> (); // 12
                [15]([10]) -> ([7]); // 13
                [14]([7]) -> ([11]); // 14
                [4]([11]) { fallthrough() 21([12]) }; // 15
                [9]() -> (); // 16
                [3]() -> ([13]); // 17
                [2]([13]) -> ([14]); // 18
                [16]([14]) -> ([14]); // 19
                return([14]); // 20
                [9]() -> (); // 21
                [1]([12]) -> ([15]); // 22
                [0]([15]) -> ([16]); // 23
                [16]([16]) -> ([16]); // 24
                return([16]); // 25

                [0]@0([0]: [2]) -> ([2]);
            "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let return_value1 = run_sierra_program(&program, &[jit_enum!(0, 42u8.into())]).return_value;

        assert_eq!(jit_enum!(0, 42u8.into()), return_value1);

        let return_value2 = run_sierra_program(
            &program,
            &[jit_enum!(
                1,
                Value::Struct {
                    fields: Vec::new(),
                    debug_name: None
                }
            )],
        )
        .return_value;

        assert_eq!(
            jit_enum!(
                1,
                Value::Struct {
                    fields: Vec::new(),
                    debug_name: None
                }
            ),
            return_value2
        )
    }
}
