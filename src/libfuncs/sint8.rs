//! # `i8`-related libfuncs
use super::{LibfuncBuilder, LibfuncHelper};
use crate::libfuncs::BranchTarget;
use crate::values::JITValue;
use crate::{
    error::{
        libfuncs::{Error, Result},
        CoreTypeBuilderError,
    },
    metadata::MetadataStorage,
    types::TypeBuilder,
    utils::ProgramRegistryExt,
};
use cairo_lang_sierra::{
    extensions::{
        int::{
            signed::{Sint8Concrete, Sint8Traits, SintConcrete},
            IntConstConcreteLibfunc, IntOperationConcreteLibfunc, IntOperator,
        },
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc, GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        cf, llvm, scf,
    },
    ir::{
        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
        operation::OperationBuilder,
        r#type::IntegerType,
        Attribute, Block, Identifier, Location, Region, Value, ValueLike,
    },
    Context,
};

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &Sint8Concrete,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    match selector {
        SintConcrete::Const(info) => {
            build_const(context, registry, entry, location, helper, metadata, info)
        }
        SintConcrete::Operation(info) => {
            build_operation(context, registry, entry, location, helper, info)
        }
        SintConcrete::Equal(info) => build_equal(context, registry, entry, location, helper, info),
        SintConcrete::ToFelt252(info) => {
            build_to_felt252(context, registry, entry, location, helper, metadata, info)
        }
        SintConcrete::FromFelt252(info) => {
            build_from_felt252(context, registry, entry, location, helper, metadata, info)
        }
        SintConcrete::IsZero(info) => {
            build_is_zero(context, registry, entry, location, helper, info)
        }
        SintConcrete::WideMul(info) => {
            build_widemul(context, registry, entry, location, helper, metadata, info)
        }
        SintConcrete::Diff(info) => todo!(),
    }
}

/// Generate MLIR operations for the `i8_const` libfunc.
pub fn build_const<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &IntConstConcreteLibfunc<Sint8Traits>,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let value = info.c;
    let value_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.signature.branch_signatures[0].vars[0].ty,
    )?;

    let op0 = entry.append_operation(arith::constant(
        context,
        Attribute::parse(context, &format!("{value} : {value_ty}")).unwrap(),
        location,
    ));
    entry.append_operation(helper.br(0, &[op0.result(0)?.into()], location));

    Ok(())
}

/// Generate MLIR operations for the i8 operation libfunc.
pub fn build_operation<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    info: &IntOperationConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let range_check: Value = entry.argument(0)?.into();
    let lhs: Value = entry.argument(1)?.into();
    let rhs: Value = entry.argument(2)?.into();

    let op_name = match info.operator {
        IntOperator::OverflowingAdd => "llvm.intr.uadd.with.overflow",
        IntOperator::OverflowingSub => "llvm.intr.usub.with.overflow",
    };

    let values_type = lhs.r#type();

    let result_type = llvm::r#type::r#struct(
        context,
        &[values_type, IntegerType::new(context, 1).into()],
        false,
    );

    let op = entry.append_operation(
        OperationBuilder::new(op_name, location)
            .add_operands(&[lhs, rhs])
            .add_results(&[result_type])
            .build()?,
    );
    let result = op.result(0)?.into();

    let op = entry.append_operation(llvm::extract_value(
        context,
        result,
        DenseI64ArrayAttribute::new(context, &[0]),
        values_type,
        location,
    ));

    let op_result = op.result(0)?.into();

    let op = entry.append_operation(llvm::extract_value(
        context,
        result,
        DenseI64ArrayAttribute::new(context, &[1]),
        IntegerType::new(context, 1).into(),
        location,
    ));
    // let op_overflow = op.result(0)?.into();
    // Create a const operation to get the i8::MAX value to compare against
    let max_op = entry.append_operation(arith::constant(
        context,
        Attribute::parse(context, &format!("{} : {}", i8::MAX, values_type)).unwrap(),
        location,
    ));
    let const_max = max_op.result(0)?.into();
    // Compare result against i8::MAX to check if an oveflow has happened
    let op_compare_max = entry.append_operation(arith::cmpi(
        context,
        CmpiPredicate::Sle,
        op_result,
        const_max,
        location,
    ));
    let not_overflow = op_compare_max.result(0)?.into();

    // Create a const operation to get the i8::MIN value to compare against
    let min_op = entry.append_operation(arith::constant(
        context,
        Attribute::parse(context, &format!("{} : {}", i8::MIN, values_type)).unwrap(),
        location,
    ));
    let const_min = min_op.result(0)?.into();
    // Compare result against i8::MIN to check if an underflow has happened
    let op_compare_min = entry.append_operation(arith::cmpi(
        context,
        CmpiPredicate::Sge,
        op_result,
        const_min,
        location,
    ));
    let not_underflow = op_compare_min.result(0)?.into();
    let block_not_oveflow = helper.append_block(Block::new(&[]));
    let block_overflow = helper.append_block(Block::new(&[]));

    entry.append_operation(cf::cond_br(
        context,
        not_overflow,
        block_not_oveflow,
        block_overflow,
        &[],
        &[],
        location,
    ));

    block_not_oveflow.append_operation(helper.cond_br(
        context,
        not_underflow,
        [0, 1],
        [&[range_check, op_result], &[range_check, op_result]],
        location,
    ));

    block_overflow.append_operation(helper.br(2, &[range_check, op_result], location));
    Ok(())
}

/// Generate MLIR operations for the `i8_eq` libfunc.
pub fn build_equal<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let arg0: Value = entry.argument(0)?.into();
    let arg1: Value = entry.argument(1)?.into();

    let op0 = entry.append_operation(arith::cmpi(
        context,
        CmpiPredicate::Eq,
        arg0,
        arg1,
        location,
    ));

    entry.append_operation(helper.cond_br(
        context,
        op0.result(0)?.into(),
        [1, 0],
        [&[]; 2],
        location,
    ));

    Ok(())
}

/// Generate MLIR operations for the `i8_is_zero` libfunc.
pub fn build_is_zero<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let arg0: Value = entry.argument(0)?.into();

    let op = entry.append_operation(arith::constant(
        context,
        IntegerAttribute::new(0, arg0.r#type()).into(),
        location,
    ));
    let const_0 = op.result(0)?.into();

    let op = entry.append_operation(arith::cmpi(
        context,
        CmpiPredicate::Eq,
        arg0,
        const_0,
        location,
    ));
    let condition = op.result(0)?.into();

    entry.append_operation(helper.cond_br(context, condition, [0, 1], [&[], &[arg0]], location));

    Ok(())
}

/// Generate MLIR operations for the `i8_widemul` libfunc.
pub fn build_widemul<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let target_type = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.output_types()[0][0],
    )?;
    let lhs: Value = entry.argument(0)?.into();
    let rhs: Value = entry.argument(1)?.into();

    let op = entry.append_operation(arith::extui(lhs, target_type, location));
    let lhs = op.result(0)?.into();

    let op = entry.append_operation(arith::extui(rhs, target_type, location));
    let rhs = op.result(0)?.into();

    let op = entry.append_operation(arith::muli(lhs, rhs, location));
    let result = op.result(0)?.into();

    entry.append_operation(helper.br(0, &[result], location));
    Ok(())
}

/// Generate MLIR operations for the `i8_to_felt252` libfunc.
pub fn build_to_felt252<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let felt252_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;
    let value: Value = entry.argument(0)?.into();

    let op = entry.append_operation(arith::extui(value, felt252_ty, location));

    let result = op.result(0)?.into();

    entry.append_operation(helper.br(0, &[result], location));

    Ok(())
}

/// Generate MLIR operations for the `i8_from_felt252` libfunc.
pub fn build_from_felt252<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder<TType, TLibfunc, Error = CoreTypeBuilderError>,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder<TType, TLibfunc, Error = Error>,
{
    let range_check: Value = entry.argument(0)?.into();
    let value: Value = entry.argument(1)?.into();

    let felt252_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.param_signatures()[1].ty,
    )?;
    let result_ty = registry.build_type(
        context,
        helper,
        registry,
        metadata,
        &info.branch_signatures()[0].vars[1].ty,
    )?;

    let op = entry.append_operation(arith::constant(
        context,
        Attribute::parse(context, &format!("{} : {}", u8::MAX, felt252_ty)).unwrap(),
        location,
    ));
    let const_max = op.result(0)?.into();

    let op = entry.append_operation(arith::cmpi(
        context,
        CmpiPredicate::Ule,
        value,
        const_max,
        location,
    ));
    let is_ule = op.result(0)?.into();

    let block_success = helper.append_block(Block::new(&[]));
    let block_failure = helper.append_block(Block::new(&[]));

    entry.append_operation(cf::cond_br(
        context,
        is_ule,
        block_success,
        block_failure,
        &[],
        &[],
        location,
    ));

    let op = block_success.append_operation(arith::trunci(value, result_ty, location));
    let value = op.result(0)?.into();
    block_success.append_operation(helper.br(0, &[range_check, value], location));

    block_failure.append_operation(helper.br(1, &[range_check], location));

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{
        utils::test::{jit_enum, jit_panic, jit_struct, load_cairo, run_program_assert_output},
        values::JITValue,
    };
    use cairo_felt::Felt252;
    use cairo_lang_sierra::program::Program;
    use lazy_static::lazy_static;
    use num_bigint::ToBigUint;

    lazy_static! {
        static ref I8_OVERFLOWING_ADD: (String, Program) = load_cairo! {
            fn run_test(lhs: i8, rhs: i8) -> i8 {
                lhs + rhs
            }
        };
        static ref I8_OVERFLOWING_SUB: (String, Program) = load_cairo! {
            fn run_test(lhs: i8, rhs: i8) -> i8 {
                lhs - rhs
            }
        };
        static ref I8_EQUAL: (String, Program) = load_cairo! {
            fn run_test(lhs: i8, rhs: i8) -> bool {
                lhs == rhs
            }
        };
        static ref I8_IS_ZERO: (String, Program) = load_cairo! {
            use zeroable::IsZeroResult;

            extern fn i8_is_zero(a: i8) -> IsZeroResult<i8> implicits() nopanic;

            fn run_test(value: i8) -> bool {
                match i8_is_zero(value) {
                    IsZeroResult::Zero(_) => true,
                    IsZeroResult::NonZero(_) => false,
                }
            }
        };
        static ref I8_WIDEMUL: (String, Program) = load_cairo! {
            use integer::i8_wide_mul;
            fn run_test(lhs: i8, rhs: i8) -> i16 {
                u8_wide_mul(lhs, rhs)
            }
        };
    }

    #[test]
    fn i8_const_min() {
        let program = load_cairo!(
            fn run_test() -> i8 {
                0_i8
            }
        );

        run_program_assert_output(&program, "run_test", &[], &[0u8.into()]);
    }

    #[test]
    fn i8_const_max() {
        let program = load_cairo!(
            fn run_test() -> i8 {
                127_i8
            }
        );

        run_program_assert_output(&program, "run_test", &[], &[(i8::MAX as u8).into()]);
    }

    #[test]
    fn i8_to_felt252() {
        let program = load_cairo!(
            use traits::Into;

            fn run_test() -> felt252 {
                2_i8.into()
            }
        );

        run_program_assert_output(&program, "run_test", &[], &[Felt252::new(2).into()]);
    }

    #[test]
    fn i8_from_felt252() {
        let program = load_cairo!(
            use traits::TryInto;

            fn run_test() -> (Option<i8>, Option<i8>) {
                (255.try_into(), 256.try_into())
            }
        );

        run_program_assert_output(
            &program,
            "run_test",
            &[],
            &[jit_struct!(
                jit_enum!(0, 255u8.into()),
                jit_enum!(1, jit_struct!()),
            )],
        );
    }

    #[test]
    fn i8_overflowing_add() {
        #[track_caller]
        fn run(lhs: u8, rhs: u8) {
            let program = &I8_OVERFLOWING_ADD;
            let error = Felt252::from_bytes_be(b"i8_add Overflow");

            let add = lhs.checked_add(rhs);

            match add {
                Some(result) => {
                    run_program_assert_output(
                        program,
                        "run_test",
                        &[lhs.into(), rhs.into()],
                        &[jit_enum!(0, jit_struct!(result.into()))],
                    );
                }
                None => {
                    run_program_assert_output(
                        program,
                        "run_test",
                        &[lhs.into(), rhs.into()],
                        &[jit_panic!(JITValue::Felt252(error))],
                    );
                }
            }
        }

        const MAX: u8 = u8::MAX;

        run(0, 0);
        run(0, 1);
        run(0, MAX - 1);
        run(0, MAX);

        run(1, 0);
        run(1, 1);
        run(1, MAX - 1);
        run(1, MAX);

        run(MAX - 1, 0);
        run(MAX - 1, 1);
        run(MAX - 1, MAX - 1);
        run(MAX - 1, MAX);

        run(MAX, 0);
        run(MAX, 1);
        run(MAX, MAX - 1);
        run(MAX, MAX);
    }

    #[test]
    fn i8_overflowing_sub() {
        #[track_caller]
        fn run(lhs: u8, rhs: u8) {
            let program = &I8_OVERFLOWING_SUB;
            let error = Felt252::from_bytes_be(b"i8_sub Overflow");

            let add = lhs.checked_sub(rhs);

            match add {
                Some(result) => {
                    run_program_assert_output(
                        program,
                        "run_test",
                        &[lhs.into(), rhs.into()],
                        &[jit_enum!(0, jit_struct!(result.into()))],
                    );
                }
                None => {
                    run_program_assert_output(
                        program,
                        "run_test",
                        &[lhs.into(), rhs.into()],
                        &[jit_panic!(JITValue::Felt252(error))],
                    );
                }
            }
        }

        const MAX: u8 = u8::MAX;

        run(0, 0);
        run(0, 1);
        run(0, MAX - 1);
        run(0, MAX);

        run(1, 0);
        run(1, 1);
        run(1, MAX - 1);
        run(1, MAX);

        run(MAX - 1, 0);
        run(MAX - 1, 1);
        run(MAX - 1, MAX - 1);
        run(MAX - 1, MAX);

        run(MAX, 0);
        run(MAX, 1);
        run(MAX, MAX - 1);
        run(MAX, MAX);
    }

    #[test]
    fn i8_equal() {
        let program = &I8_EQUAL;

        run_program_assert_output(
            program,
            "run_test",
            &[0u8.into(), 0u8.into()],
            &[jit_enum!(1, jit_struct!())],
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u8.into(), 0u8.into()],
            &[jit_enum!(0, jit_struct!())],
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0u8.into(), 1u8.into()],
            &[jit_enum!(0, jit_struct!())],
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u8.into(), 1u8.into()],
            &[jit_enum!(1, jit_struct!())],
        );
    }

    #[test]
    fn i8_is_zero() {
        let program = &I8_IS_ZERO;

        run_program_assert_output(
            program,
            "run_test",
            &[0u8.into()],
            &[jit_enum!(1, jit_struct!())],
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u8.into()],
            &[jit_enum!(0, jit_struct!())],
        );
    }

    #[test]
    fn i8_safe_divmod() {
        let program = &I8_IS_ZERO;

        run_program_assert_output(
            program,
            "run_test",
            &[0u8.into(), 0u8.into()],
            &[jit_enum!(1, jit_struct!())],
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0u8.into(), 1u8.into()],
            &[jit_enum!(1, jit_struct!())],
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0u8.into(), 0xFFu8.into()],
            &[jit_enum!(1, jit_struct!())],
        );

        run_program_assert_output(
            program,
            "run_test",
            &[1u8.into(), 0u8.into()],
            &[jit_enum!(0, jit_struct!())],
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u8.into(), 1u8.into()],
            &[jit_enum!(0, jit_struct!())],
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u8.into(), 0xFFu8.into()],
            &[jit_enum!(0, jit_struct!())],
        );

        run_program_assert_output(
            program,
            "run_test",
            &[0xFFu8.into(), 0u8.into()],
            &[jit_enum!(0, jit_struct!())],
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0xFFu8.into(), 1u8.into()],
            &[jit_enum!(0, jit_struct!())],
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0xFFu8.into(), 0xFFu8.into()],
            &[jit_enum!(0, jit_struct!())],
        );
    }

    #[test]
    fn i8_widemul() {
        let program = &I8_WIDEMUL;

        run_program_assert_output(
            program,
            "run_test",
            &[0u8.into(), 0u8.into()],
            &[0u16.into()],
        );
        run_program_assert_output(
            program,
            "run_test",
            &[0u8.into(), 1u8.into()],
            &[0u16.into()],
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u8.into(), 0u8.into()],
            &[0u16.into()],
        );
        run_program_assert_output(
            program,
            "run_test",
            &[1u8.into(), 1u8.into()],
            &[1u16.into()],
        );
        run_program_assert_output(
            program,
            "run_test",
            &[u8::MAX.into(), u8::MAX.into()],
            &[(u8::MAX as u16 * u8::MAX as u16).into()],
        );
    }
}
