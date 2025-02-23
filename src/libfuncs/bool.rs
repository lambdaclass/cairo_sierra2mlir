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
    ir::{r#type::IntegerType, Block, BlockLike, Location},
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
    use cairo_lang_sierra::extensions::boolean::{
        BoolAndLibfunc, BoolNotLibfunc, BoolOrLibfunc, BoolToFelt252Libfunc, BoolXorLibfunc,
    };

    use crate::{
        utils::{
            sierra_gen::SierraGenerator,
            test::{jit_enum, jit_struct, run_sierra_program},
        },
        values::Value,
    };

    #[test]
    fn run_not() {
        let program = {
            let sierra_generator = SierraGenerator::<BoolNotLibfunc>::default();

            sierra_generator.build(&[])
        };

        let result = run_sierra_program(&program, &[jit_enum!(0, jit_struct!())]).return_value;
        assert_eq!(result, jit_enum!(1, jit_struct!()));

        let result = run_sierra_program(&program, &[jit_enum!(1, jit_struct!())]).return_value;
        assert_eq!(result, jit_enum!(0, jit_struct!()));
    }

    #[test]
    fn run_and() {
        let program = {
            let sierra_generator = SierraGenerator::<BoolAndLibfunc>::default();

            sierra_generator.build(&[])
        };

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
        let program = {
            let sierra_generator = SierraGenerator::<BoolXorLibfunc>::default();

            sierra_generator.build(&[])
        };

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
        let program = {
            let sierra_generator = SierraGenerator::<BoolOrLibfunc>::default();

            sierra_generator.build(&[])
        };

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
        let program = {
            let sierra_generator = SierraGenerator::<BoolToFelt252Libfunc>::default();

            sierra_generator.build(&[])
        };

        let result = run_sierra_program(&program, &[jit_enum!(1, jit_struct!())]).return_value;
        assert_eq!(result, Value::Felt252(1.into()));

        let result = run_sierra_program(&program, &[jit_enum!(0, jit_struct!())]).return_value;
        assert_eq!(result, Value::Felt252(0.into()));
    }
}
