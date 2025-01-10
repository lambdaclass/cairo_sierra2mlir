//! # Boolean libfuncs

use super::LibfuncHelper;
use crate::{
    error::Result,
    metadata::MetadataStorage,
    types::TypeBuilder,
    utils::{BlockExt, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        boolean::BoolConcreteLibfunc,
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{arith, llvm},
    ir::{r#type::IntegerType, Block, Location},
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
    selector: &BoolConcreteLibfunc,
) -> Result<()> {
    match selector {
        BoolConcreteLibfunc::And(info) => build_bool_binary(
            context,
            registry,
            entry,
            location,
            helper,
            metadata,
            info,
            BoolOp::And,
        ),
        BoolConcreteLibfunc::Not(info) => {
            build_bool_not(context, registry, entry, location, helper, metadata, info)
        }
        BoolConcreteLibfunc::Xor(info) => build_bool_binary(
            context,
            registry,
            entry,
            location,
            helper,
            metadata,
            info,
            BoolOp::Xor,
        ),
        BoolConcreteLibfunc::Or(info) => build_bool_binary(
            context,
            registry,
            entry,
            location,
            helper,
            metadata,
            info,
            BoolOp::Or,
        ),
        BoolConcreteLibfunc::ToFelt252(info) => {
            build_bool_to_felt252(context, registry, entry, location, helper, metadata, info)
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum BoolOp {
    And,
    Xor,
    Or,
}

/// Generate MLIR operations for the `bool_not_impl` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_bool_binary<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
    bin_op: BoolOp,
) -> Result<()> {
    let enum_ty = registry.get_type(&info.param_signatures()[0].ty)?;
    let tag_bits = enum_ty
        .variants()
        .expect("bool is a enum and has variants")
        .len()
        .next_power_of_two()
        .trailing_zeros();
    let tag_ty = IntegerType::new(context, tag_bits).into();

    let lhs = entry.arg(0)?;
    let rhs = entry.arg(1)?;

    let lhs_tag = entry.extract_value(context, location, lhs, tag_ty, 0)?;

    let rhs_tag = entry.extract_value(context, location, rhs, tag_ty, 0)?;

    let new_tag_value = match bin_op {
        BoolOp::And => entry.append_op_result(arith::andi(lhs_tag, rhs_tag, location))?,
        BoolOp::Xor => entry.append_op_result(arith::xori(lhs_tag, rhs_tag, location))?,
        BoolOp::Or => entry.append_op_result(arith::ori(lhs_tag, rhs_tag, location))?,
    };

    let res = entry.append_op_result(llvm::undef(
        enum_ty.build(
            context,
            helper,
            registry,
            metadata,
            &info.param_signatures()[0].ty,
        )?,
        location,
    ))?;

    let res = entry.insert_value(context, location, res, new_tag_value, 0)?;

    entry.append_operation(helper.br(0, &[res], location));
    Ok(())
}

/// Generate MLIR operations for the `bool_not_impl` libfunc.
pub fn build_bool_not<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let enum_ty = registry.get_type(&info.param_signatures()[0].ty)?;
    let tag_bits = enum_ty
        .variants()
        .expect("bool is a enum and has variants")
        .len()
        .next_power_of_two()
        .trailing_zeros();
    let tag_ty = IntegerType::new(context, tag_bits).into();

    let value = entry.arg(0)?;
    let tag_value = entry.extract_value(context, location, value, tag_ty, 0)?;

    let const_1 = entry.const_int_from_type(context, location, 1, tag_ty)?;

    let new_tag_value = entry.append_op_result(arith::xori(tag_value, const_1, location))?;

    let res = entry.append_op_result(llvm::undef(
        enum_ty.build(
            context,
            helper,
            registry,
            metadata,
            &info.param_signatures()[0].ty,
        )?,
        location,
    ))?;
    let res = entry.insert_value(context, location, res, new_tag_value, 0)?;

    entry.append_operation(helper.br(0, &[res], location));
    Ok(())
}

/// Generate MLIR operations for the `unbox` libfunc.
pub fn build_bool_to_felt252<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let enum_ty = registry.get_type(&info.param_signatures()[0].ty)?;
    let felt252_ty = registry.build_type(
        context,
        helper,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;

    let tag_bits = enum_ty
        .variants()
        .expect("bool is a enum and has variants")
        .len()
        .next_power_of_two()
        .trailing_zeros();
    let tag_ty = IntegerType::new(context, tag_bits).into();

    let value = entry.arg(0)?;
    let tag_value = entry.extract_value(context, location, value, tag_ty, 0)?;

    let result = entry.extui(tag_value, felt252_ty, location)?;

    entry.append_operation(helper.br(0, &[result], location));
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
    fn run_not() {
        // use array::ArrayTrait;
        //
        // fn run_test(a: bool) -> bool {
        //   !a
        // }
        let program = ProgramParser::new().parse(r#"
            type [0] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
            type [1] = Enum<ut@core::bool, [0], [0]> [storable: true, drop: true, dup: true, zero_sized: false];

            libfunc [0] = bool_not_impl;
            libfunc [2] = store_temp<[1]>;

            [0]([0]) -> ([1]); // 0
            [2]([1]) -> ([1]); // 1
            return([1]); // 2

            [0]@0([0]: [1]) -> ([1]);
            "#).map_err(|e| e.to_string()).unwrap();

        let result = run_sierra_program(&program, &[jit_enum!(0, jit_struct!())]).return_value;
        assert_eq!(result, jit_enum!(1, jit_struct!()));

        let result = run_sierra_program(&program, &[jit_enum!(1, jit_struct!())]).return_value;
        assert_eq!(result, jit_enum!(0, jit_struct!()));
    }

    #[test]
    fn run_and() {
        // use array::ArrayTrait;

        // fn run_test(a: bool, b: bool) -> bool {
        //     a && b
        // }
        let program = ProgramParser::new().parse(r#"
            type [0] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
            type [1] = Enum<ut@core::bool, [0], [0]> [storable: true, drop: true, dup: true, zero_sized: false];

            libfunc [2] = enum_match<[1]>;
            libfunc [3] = branch_align;
            libfunc [4] = drop<[0]>;
            libfunc [5] = drop<[1]>;
            libfunc [1] = struct_construct<[0]>;
            libfunc [0] = enum_init<[1], 0>;
            libfunc [6] = store_temp<[1]>;

            [2]([0]) { fallthrough([2]) 8([3]) }; // 0
            [3]() -> (); // 1
            [4]([2]) -> (); // 2
            [5]([1]) -> (); // 3
            [1]() -> ([4]); // 4
            [0]([4]) -> ([5]); // 5
            [6]([5]) -> ([5]); // 6
            return([5]); // 7
            [3]() -> (); // 8
            [4]([3]) -> (); // 9
            [6]([1]) -> ([1]); // 10
            return([1]); // 11

            [0]@0([0]: [1], [1]: [1]) -> ([1]);
            "#).map_err(|e| e.to_string()).unwrap();

        let result = run_sierra_program(
            &program,
            &[jit_enum!(1, jit_struct!()), jit_enum!(1, jit_struct!())],
        )
        .return_value;
        assert_eq!(result, jit_enum!(1, jit_struct!()));

        let result = run_sierra_program(
            &program,
            &[jit_enum!(1, jit_struct!()), jit_enum!(0, jit_struct!())],
        )
        .return_value;
        assert_eq!(result, jit_enum!(0, jit_struct!()));

        let result = run_sierra_program(
            &program,
            &[jit_enum!(0, jit_struct!()), jit_enum!(1, jit_struct!())],
        )
        .return_value;
        assert_eq!(result, jit_enum!(0, jit_struct!()));

        let result = run_sierra_program(
            &program,
            &[jit_enum!(0, jit_struct!()), jit_enum!(0, jit_struct!())],
        )
        .return_value;
        assert_eq!(result, jit_enum!(0, jit_struct!()));
    }

    #[test]
    fn run_xor() {
        let program = ProgramParser::new().parse(r#"
            type [0] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
            type [1] = Enum<ut@core::bool, [0], [0]> [storable: true, drop: true, dup: true, zero_sized: false];

            libfunc [0] = bool_xor_impl;
            libfunc [2] = store_temp<[1]>;

            [0]([0], [1]) -> ([2]); // 0
            [2]([2]) -> ([2]); // 1
            return([2]); // 2

            [0]@0([0]: [1], [1]: [1]) -> ([1]);
            "#).map_err(|e| e.to_string()).unwrap();

        let result = run_sierra_program(
            &program,
            &[jit_enum!(1, jit_struct!()), jit_enum!(1, jit_struct!())],
        )
        .return_value;
        assert_eq!(result, jit_enum!(0, jit_struct!()));

        let result = run_sierra_program(
            &program,
            &[jit_enum!(1, jit_struct!()), jit_enum!(0, jit_struct!())],
        )
        .return_value;
        assert_eq!(result, jit_enum!(1, jit_struct!()));

        let result = run_sierra_program(
            &program,
            &[jit_enum!(0, jit_struct!()), jit_enum!(1, jit_struct!())],
        )
        .return_value;
        assert_eq!(result, jit_enum!(1, jit_struct!()));

        let result = run_sierra_program(
            &program,
            &[jit_enum!(0, jit_struct!()), jit_enum!(0, jit_struct!())],
        )
        .return_value;
        assert_eq!(result, jit_enum!(0, jit_struct!()));
    }

    #[test]
    fn run_or() {
        let program = ProgramParser::new().parse(r#"
            type [0] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
            type [1] = Enum<ut@core::bool, [0], [0]> [storable: true, drop: true, dup: true, zero_sized: false];

            libfunc [2] = enum_match<[1]>;
            libfunc [3] = branch_align;
            libfunc [4] = drop<[0]>;
            libfunc [6] = store_temp<[1]>;
            libfunc [5] = drop<[1]>;
            libfunc [1] = struct_construct<[0]>;
            libfunc [0] = enum_init<[1], 1>;

            [2]([0]) { fallthrough([2]) 5([3]) }; // 0
            [3]() -> (); // 1
            [4]([2]) -> (); // 2
            [6]([1]) -> ([1]); // 3
            return([1]); // 4
            [3]() -> (); // 5
            [4]([3]) -> (); // 6
            [5]([1]) -> (); // 7
            [1]() -> ([4]); // 8
            [0]([4]) -> ([5]); // 9
            [6]([5]) -> ([5]); // 10
            return([5]); // 11

            [0]@0([0]: [1], [1]: [1]) -> ([1]);
            "#).map_err(|e| e.to_string()).unwrap();

        let result = run_sierra_program(
            &program,
            &[jit_enum!(1, jit_struct!()), jit_enum!(1, jit_struct!())],
        )
        .return_value;
        assert_eq!(result, jit_enum!(1, jit_struct!()));

        let result = run_sierra_program(
            &program,
            &[jit_enum!(1, jit_struct!()), jit_enum!(0, jit_struct!())],
        )
        .return_value;
        assert_eq!(result, jit_enum!(1, jit_struct!()));

        let result = run_sierra_program(
            &program,
            &[jit_enum!(0, jit_struct!()), jit_enum!(1, jit_struct!())],
        )
        .return_value;
        assert_eq!(result, jit_enum!(1, jit_struct!()));

        let result = run_sierra_program(
            &program,
            &[jit_enum!(0, jit_struct!()), jit_enum!(0, jit_struct!())],
        )
        .return_value;
        assert_eq!(result, jit_enum!(0, jit_struct!()));
    }

    #[test]
    fn bool_to_felt252() {
        let program = ProgramParser::new().parse(r#"
            type [0] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
            type [1] = Enum<ut@core::bool, [0], [0]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [2] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];

            libfunc [0] = bool_to_felt252;
            libfunc [2] = store_temp<[2]>;

            [0]([0]) -> ([1]); // 0
            [2]([1]) -> ([1]); // 1
            return([1]); // 2

            [0]@0([0]: [1]) -> ([2]);
            "#).map_err(|e| e.to_string()).unwrap();

        let result = run_sierra_program(&program, &[jit_enum!(1, jit_struct!())]).return_value;
        assert_eq!(result, Value::Felt252(1.into()));

        let result = run_sierra_program(&program, &[jit_enum!(0, jit_struct!())]).return_value;
        assert_eq!(result, Value::Felt252(0.into()));
    }
}
