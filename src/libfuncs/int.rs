use super::{BlockExt, LibfuncHelper};
use crate::{
    error::Result,
    metadata::MetadataStorage,
    native_panic,
    types::TypeBuilder,
    utils::{ProgramRegistryExt, PRIME},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        int::{
            signed::{SintConcrete, SintTraits},
            signed128::Sint128Concrete,
            unsigned::{UintConcrete, UintTraits},
            unsigned128::Uint128Concrete,
            IntConstConcreteLibfunc, IntMulTraits, IntOperationConcreteLibfunc, IntOperator,
            IntTraits,
        },
        is_zero::IsZeroTraits,
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        cf, llvm,
        ods::{self, math},
        scf,
    },
    ir::{
        attribute::IntegerAttribute, operation::OperationBuilder, r#type::IntegerType, Block,
        BlockLike, Location, Region, ValueLike,
    },
    Context,
};
use num_bigint::{BigInt, Sign};
use num_traits::Zero;

pub fn build_unsigned<'ctx, 'this, T>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &UintConcrete<T>,
) -> Result<()>
where
    T: IntMulTraits + IsZeroTraits + UintTraits,
{
    match selector {
        UintConcrete::Bitwise(info) => {
            build_bitwise(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::Const(info) => {
            build_const(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::Divmod(info) => {
            build_divmod(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::Equal(info) => {
            build_equal(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::FromFelt252(info) => {
            build_from_felt252(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::IsZero(info) => {
            build_is_zero(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::Operation(info) => {
            build_operation(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::SquareRoot(info) => {
            build_square_root(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::ToFelt252(info) => {
            build_to_felt252(context, registry, entry, location, helper, metadata, info)
        }
        UintConcrete::WideMul(info) => {
            build_wide_mul(context, registry, entry, location, helper, metadata, info)
        }
    }
}

pub fn build_signed<'ctx, 'this, T>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &SintConcrete<T>,
) -> Result<()>
where
    T: IntMulTraits + IsZeroTraits + SintTraits,
{
    match selector {
        SintConcrete::Const(info) => {
            build_const(context, registry, entry, location, helper, metadata, info)
        }
        SintConcrete::Diff(info) => {
            build_diff(context, registry, entry, location, helper, metadata, info)
        }
        SintConcrete::Equal(info) => {
            build_equal(context, registry, entry, location, helper, metadata, info)
        }
        SintConcrete::FromFelt252(info) => {
            build_from_felt252(context, registry, entry, location, helper, metadata, info)
        }
        SintConcrete::IsZero(info) => {
            build_is_zero(context, registry, entry, location, helper, metadata, info)
        }
        SintConcrete::Operation(info) => {
            build_operation(context, registry, entry, location, helper, metadata, info)
        }
        SintConcrete::ToFelt252(info) => {
            build_to_felt252(context, registry, entry, location, helper, metadata, info)
        }
        SintConcrete::WideMul(info) => {
            build_wide_mul(context, registry, entry, location, helper, metadata, info)
        }
    }
}

pub fn build_u128<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &Uint128Concrete,
) -> Result<()> {
    match selector {
        Uint128Concrete::Bitwise(info) => {
            build_bitwise(context, registry, entry, location, helper, metadata, info)
        }
        Uint128Concrete::ByteReverse(info) => {
            build_byte_reverse(context, registry, entry, location, helper, metadata, info)
        }
        Uint128Concrete::Const(info) => {
            build_const(context, registry, entry, location, helper, metadata, info)
        }
        Uint128Concrete::Divmod(info) => {
            build_divmod(context, registry, entry, location, helper, metadata, info)
        }
        Uint128Concrete::Equal(info) => {
            build_equal(context, registry, entry, location, helper, metadata, info)
        }
        Uint128Concrete::FromFelt252(info) => {
            build_u128s_from_felt252(context, registry, entry, location, helper, metadata, info)
        }
        Uint128Concrete::GuaranteeMul(info) => {
            build_guarantee_mul(context, registry, entry, location, helper, metadata, info)
        }
        Uint128Concrete::IsZero(info) => {
            build_is_zero(context, registry, entry, location, helper, metadata, info)
        }
        Uint128Concrete::MulGuaranteeVerify(info) => {
            build_mul_guarantee_verify(context, registry, entry, location, helper, metadata, info)
        }
        Uint128Concrete::Operation(info) => {
            build_operation(context, registry, entry, location, helper, metadata, info)
        }
        Uint128Concrete::SquareRoot(info) => {
            build_square_root(context, registry, entry, location, helper, metadata, info)
        }
        Uint128Concrete::ToFelt252(info) => {
            build_to_felt252(context, registry, entry, location, helper, metadata, info)
        }
    }
}

pub fn build_i128<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &Sint128Concrete,
) -> Result<()> {
    match selector {
        Sint128Concrete::Const(info) => {
            build_const(context, registry, entry, location, helper, metadata, info)
        }
        Sint128Concrete::Diff(info) => {
            build_diff(context, registry, entry, location, helper, metadata, info)
        }
        Sint128Concrete::Equal(info) => {
            build_equal(context, registry, entry, location, helper, metadata, info)
        }
        Sint128Concrete::FromFelt252(info) => {
            build_from_felt252(context, registry, entry, location, helper, metadata, info)
        }
        Sint128Concrete::IsZero(info) => {
            build_is_zero(context, registry, entry, location, helper, metadata, info)
        }
        Sint128Concrete::Operation(info) => {
            build_operation(context, registry, entry, location, helper, metadata, info)
        }
        Sint128Concrete::ToFelt252(info) => {
            build_to_felt252(context, registry, entry, location, helper, metadata, info)
        }
    }
}

fn build_bitwise<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let bitwise = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let lhs = entry.arg(1)?;
    let rhs = entry.arg(2)?;

    let logical_and = entry.append_op_result(arith::andi(lhs, rhs, location))?;
    let logical_xor = entry.append_op_result(arith::xori(lhs, rhs, location))?;
    let logical_or = entry.append_op_result(arith::ori(lhs, rhs, location))?;

    helper.br(
        entry,
        0,
        &[bitwise, logical_and, logical_xor, logical_or],
        location,
    )
}

fn build_byte_reverse<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let bitwise = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let value =
        entry.append_op_result(ods::llvm::intr_bswap(context, entry.arg(1)?, location).into())?;

    helper.br(entry, 0, &[bitwise, value], location)
}

fn build_const<'ctx, 'this, T>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &IntConstConcreteLibfunc<T>,
) -> Result<()>
where
    T: IntTraits,
{
    let value_ty = registry.build_type(
        context,
        helper,
        metadata,
        &info.signature.branch_signatures[0].vars[0].ty,
    )?;

    let value = entry.const_int_from_type(context, location, info.c, value_ty)?;

    helper.br(entry, 0, &[value], location)
}

fn build_diff<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let lhs = entry.arg(1)?;
    let rhs = entry.arg(2)?;

    let is_greater_equal = entry.cmpi(context, CmpiPredicate::Sge, lhs, rhs, location)?;
    let value_difference = entry.append_op_result(arith::subi(lhs, rhs, location))?;

    helper.cond_br(
        context,
        entry,
        is_greater_equal,
        [0, 1],
        [&[range_check, value_difference]; 2],
        location,
    )
}

fn build_divmod<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let lhs = entry.arg(1)?;
    let rhs = entry.arg(2)?;

    let result_div = entry.append_op_result(arith::divui(lhs, rhs, location))?;
    let result_rem = entry.append_op_result(arith::remui(lhs, rhs, location))?;

    helper.br(entry, 0, &[range_check, result_div, result_rem], location)
}

fn build_equal<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let are_equal = entry.cmpi(
        context,
        CmpiPredicate::Eq,
        entry.arg(0)?,
        entry.arg(1)?,
        location,
    )?;

    helper.cond_br(context, entry, are_equal, [1, 0], [&[]; 2], location)
}

fn build_from_felt252<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let value_ty = registry.get_type(&info.signature.branch_signatures[0].vars[1].ty)?;
    let threshold = value_ty.integer_range(registry)?;

    let value_ty = value_ty.build(
        context,
        helper,
        registry,
        metadata,
        &info.signature.branch_signatures[0].vars[1].ty,
    )?;

    let input = entry.arg(1)?;

    // Handle signedness separately.
    let (is_in_range, value) = if threshold.lower.is_zero() {
        let upper_threshold =
            entry.const_int_from_type(context, location, threshold.upper, input.r#type())?;
        let is_in_range = entry.cmpi(
            context,
            CmpiPredicate::Ult,
            input,
            upper_threshold,
            location,
        )?;

        (is_in_range, input)
    } else {
        let lower_threshold = entry.const_int_from_type(
            context,
            location,
            if threshold.lower.sign() == Sign::Minus {
                &*PRIME - threshold.lower.magnitude()
            } else {
                threshold.lower.magnitude().clone()
            },
            input.r#type(),
        )?;
        let upper_threshold = entry.const_int_from_type(
            context,
            location,
            if threshold.upper.sign() == Sign::Minus {
                &*PRIME - threshold.upper.magnitude()
            } else {
                threshold.upper.magnitude().clone()
            },
            input.r#type(),
        )?;

        let lower_check = entry.cmpi(
            context,
            CmpiPredicate::Sge,
            input,
            lower_threshold,
            location,
        )?;
        let upper_check = entry.cmpi(
            context,
            CmpiPredicate::Slt,
            input,
            upper_threshold,
            location,
        )?;

        let is_in_range =
            entry.append_op_result(arith::andi(lower_check, upper_check, location))?;

        let k0 = entry.const_int_from_type(context, location, 0, input.r#type())?;
        let is_negative = entry.cmpi(context, CmpiPredicate::Slt, input, k0, location)?;
        let value = entry.append_op_result(scf::r#if(
            is_negative,
            &[input.r#type()],
            {
                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                let prime = block.const_int_from_type(
                    context,
                    location,
                    BigInt::from_biguint(Sign::Plus, PRIME.clone()),
                    input.r#type(),
                )?;
                let value = block.append_op_result(arith::subi(input, prime, location))?;

                block.append_operation(scf::r#yield(&[value], location));
                region
            },
            {
                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                block.append_operation(scf::r#yield(&[input], location));
                region
            },
            location,
        ))?;

        (is_in_range, value)
    };

    let value = entry.trunci(value, value_ty, location)?;

    helper.cond_br(
        context,
        entry,
        is_in_range,
        [0, 1],
        [&[range_check, value], &[range_check]],
        location,
    )
}

fn build_guarantee_mul<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let guarantee_ty = registry.build_type(
        context,
        helper,
        metadata,
        &info.signature.branch_signatures[0].vars[2].ty,
    )?;

    let mul_op = entry.append_operation(arith::mului_extended(
        entry.arg(0)?,
        entry.arg(1)?,
        location,
    ));

    let lo = mul_op.result(0)?.into();
    let hi = mul_op.result(1)?.into();

    let guarantee = entry.append_op_result(llvm::undef(guarantee_ty, location))?;
    helper.br(entry, 0, &[hi, lo, guarantee], location)
}

fn build_is_zero<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let input = entry.arg(0)?;

    let k0 = entry.const_int_from_type(context, location, 0, input.r#type())?;
    let is_zero = entry.cmpi(context, CmpiPredicate::Eq, input, k0, location)?;

    helper.cond_br(context, entry, is_zero, [0, 1], [&[], &[input]], location)
}

fn build_mul_guarantee_verify<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    helper.br(entry, 0, &[range_check], location)
}

fn build_operation<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &IntOperationConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let value_ty = registry.get_type(&info.signature.param_signatures[1].ty)?;
    let is_signed = !value_ty.integer_range(registry)?.lower.is_zero();
    let value_ty = value_ty.build(
        context,
        helper,
        registry,
        metadata,
        &info.signature.param_signatures[1].ty,
    )?;

    let op_name = match (is_signed, info.operator) {
        (false, IntOperator::OverflowingAdd) => "llvm.intr.uadd.with.overflow",
        (false, IntOperator::OverflowingSub) => "llvm.intr.usub.with.overflow",
        (true, IntOperator::OverflowingAdd) => "llvm.intr.sadd.with.overflow",
        (true, IntOperator::OverflowingSub) => "llvm.intr.ssub.with.overflow",
    };
    let result_with_overflow = entry.append_op_result(
        OperationBuilder::new(op_name, location)
            .add_operands(&[entry.arg(1)?, entry.arg(2)?])
            .add_results(&[llvm::r#type::r#struct(
                context,
                &[value_ty, IntegerType::new(context, 1).into()],
                false,
            )])
            .build()?,
    )?;

    let result = entry.extract_value(context, location, result_with_overflow, value_ty, 0)?;
    let overflow = entry.extract_value(
        context,
        location,
        result_with_overflow,
        IntegerType::new(context, 1).into(),
        1,
    )?;

    if is_signed {
        let block_in_range = helper.append_block(Block::new(&[]));
        let block_overflow = helper.append_block(Block::new(&[]));

        entry.append_operation(cf::cond_br(
            context,
            overflow,
            block_overflow,
            block_in_range,
            &[],
            &[],
            location,
        ));

        helper.br(block_in_range, 0, &[range_check, result], location)?;

        {
            let k0 = block_overflow.const_int_from_type(context, location, 0, result.r#type())?;
            let is_positive =
                block_overflow.cmpi(context, CmpiPredicate::Sge, result, k0, location)?;
            helper.cond_br(
                context,
                block_overflow,
                is_positive,
                [1, 2],
                [&[range_check, result]; 2],
                location,
            )
        }
    } else {
        helper.cond_br(
            context,
            entry,
            overflow,
            [1, 0],
            [&[range_check, result]; 2],
            location,
        )
    }
}

fn build_square_root<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let input = entry.arg(1)?;
    let (input_bits, value_bits) =
        match registry.get_type(&info.signature.param_signatures[1].ty)? {
            CoreTypeConcrete::Uint8(_) => (8, 8),
            CoreTypeConcrete::Uint16(_) => (16, 8),
            CoreTypeConcrete::Uint32(_) => (32, 16),
            CoreTypeConcrete::Uint64(_) => (64, 32),
            CoreTypeConcrete::Uint128(_) => (128, 64),
            _ => native_panic!("invalid value type in int square root"),
        };

    let k1 = entry.const_int(context, location, 1, input_bits)?;
    let is_small = entry.cmpi(context, CmpiPredicate::Ule, input, k1, location)?;

    let value = entry.append_op_result(scf::r#if(
        is_small,
        &[IntegerType::new(context, value_bits).into()],
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let value = block.trunci(
                input,
                IntegerType::new(context, value_bits).into(),
                location,
            )?;

            block.append_operation(scf::r#yield(&[value], location));
            region
        },
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let leading_zeros = block.append_op_result(
                ods::llvm::intr_ctlz(
                    context,
                    IntegerType::new(context, input_bits).into(),
                    input,
                    IntegerAttribute::new(IntegerType::new(context, 1).into(), 1),
                    location,
                )
                .into(),
            )?;

            let k_bits = block.const_int(context, location, input_bits, input_bits)?;
            let num_bits = block.append_op_result(arith::subi(k_bits, leading_zeros, location))?;
            let shift_amount = block.addi(num_bits, k1, location)?;

            let parity_mask = block.const_int(context, location, -2, input_bits)?;
            let shift_amount =
                block.append_op_result(arith::andi(shift_amount, parity_mask, location))?;

            let k0 = block.const_int(context, location, 0, input_bits)?;
            let value = block.append_op_result(scf::r#while(
                &[k0, shift_amount],
                &[
                    IntegerType::new(context, input_bits).into(),
                    IntegerType::new(context, input_bits).into(),
                ],
                {
                    let region = Region::new();
                    let block = region.append_block(Block::new(&[
                        (IntegerType::new(context, input_bits).into(), location),
                        (IntegerType::new(context, input_bits).into(), location),
                    ]));

                    let value = block.shli(block.arg(0)?, k1, location)?;
                    let large_candidate =
                        block.append_op_result(arith::xori(value, k1, location))?;
                    let large_candidate_squared =
                        block.muli(large_candidate, large_candidate, location)?;

                    let threshold = block.shrui(input, block.arg(1)?, location)?;
                    let threshold_is_poison =
                        block.cmpi(context, CmpiPredicate::Eq, block.arg(1)?, k_bits, location)?;
                    let threshold = block.append_op_result(arith::select(
                        threshold_is_poison,
                        k0,
                        threshold,
                        location,
                    ))?;

                    let is_in_range = block.cmpi(
                        context,
                        CmpiPredicate::Ule,
                        large_candidate_squared,
                        threshold,
                        location,
                    )?;
                    let value = block.append_op_result(arith::select(
                        is_in_range,
                        large_candidate,
                        value,
                        location,
                    ))?;

                    let k2 = block.const_int(context, location, 2, input_bits)?;
                    let shift_amount =
                        block.append_op_result(arith::subi(block.arg(1)?, k2, location))?;

                    let should_continue =
                        block.cmpi(context, CmpiPredicate::Sge, shift_amount, k0, location)?;
                    block.append_operation(scf::condition(
                        should_continue,
                        &[value, shift_amount],
                        location,
                    ));

                    region
                },
                {
                    let region = Region::new();
                    let block = region.append_block(Block::new(&[
                        (IntegerType::new(context, input_bits).into(), location),
                        (IntegerType::new(context, input_bits).into(), location),
                    ]));

                    block.append_operation(scf::r#yield(&[block.arg(0)?, block.arg(1)?], location));
                    region
                },
                location,
            ))?;

            let value = if input_bits == value_bits {
                value
            } else {
                block.trunci(
                    value,
                    IntegerType::new(context, value_bits).into(),
                    location,
                )?
            };

            block.append_operation(scf::r#yield(&[value], location));
            region
        },
        location,
    ))?;

    helper.br(entry, 0, &[range_check, value], location)
}

fn build_to_felt252<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let value_ty = registry.get_type(&info.signature.param_signatures[0].ty)?;
    let is_signed = !value_ty.integer_range(registry)?.lower.is_zero();

    let felt252_ty = registry.build_type(
        context,
        helper,
        metadata,
        &info.signature.branch_signatures[0].vars[0].ty,
    )?;

    let value = if is_signed {
        let prime = entry.const_int_from_type(
            context,
            location,
            BigInt::from_biguint(Sign::Plus, PRIME.clone()),
            felt252_ty,
        )?;

        let k0 = entry.const_int_from_type(
            context,
            location,
            0,
            value_ty.build(
                context,
                helper,
                registry,
                metadata,
                &info.signature.param_signatures[0].ty,
            )?,
        )?;
        let is_negative = entry.cmpi(context, CmpiPredicate::Slt, entry.arg(0)?, k0, location)?;

        let value = entry.extui(entry.arg(0)?, felt252_ty, location)?;

        let neg_value =
            entry.append_op_result(math::absi(context, entry.arg(0)?, location).into())?;
        let neg_value = entry.extui(neg_value, felt252_ty, location)?;
        let neg_value = entry.append_op_result(arith::subi(prime, neg_value, location))?;

        entry.append_op_result(arith::select(is_negative, neg_value, value, location))?
    } else {
        entry.extui(entry.arg(0)?, felt252_ty, location)?
    };

    helper.br(entry, 0, &[value], location)
}

fn build_u128s_from_felt252<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let target_ty = IntegerType::new(context, 128).into();

    let lo = entry.trunci(entry.arg(1)?, target_ty, location)?;

    let k128 = entry.const_int_from_type(context, location, 128, entry.arg(1)?.r#type())?;
    let hi = entry.shrui(entry.arg(1)?, k128, location)?;
    let hi = entry.trunci(hi, target_ty, location)?;

    let k0 = entry.const_int_from_type(context, location, 0, target_ty)?;
    let is_wide = entry.cmpi(context, CmpiPredicate::Ne, hi, k0, location)?;

    helper.cond_br(
        context,
        entry,
        is_wide,
        [1, 0],
        [&[range_check, hi, lo], &[range_check, lo]],
        location,
    )
}

fn build_wide_mul<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let result_ty = registry.build_type(
        context,
        helper,
        metadata,
        &info.signature.branch_signatures[0].vars[0].ty,
    )?;

    let ext_fn = if registry
        .get_type(&info.signature.param_signatures[0].ty)?
        .integer_range(registry)
        .unwrap()
        .lower
        .is_zero()
    {
        BlockExt::extui
    } else {
        BlockExt::extsi
    };

    let lhs = ext_fn(entry, entry.arg(0)?, result_ty, location)?;
    let rhs = ext_fn(entry, entry.arg(1)?, result_ty, location)?;
    let result = entry.muli(lhs, rhs, location)?;

    helper.br(entry, 0, &[result], location)
}

#[cfg(test)]
mod test {
    use crate::{
        context::NativeContext, executor::JitNativeExecutor, utils::HALF_PRIME, OptLevel, Value,
    };
    use cairo_lang_sierra::ProgramParser;
    use itertools::Itertools;
    use num_bigint::{BigInt, BigUint, Sign};
    use num_integer::Roots;
    use num_traits::{
        ops::overflowing::{OverflowingAdd, OverflowingSub},
        Bounded, Num,
    };
    use starknet_types_core::felt::Felt;
    use std::{
        fmt::Display,
        mem,
        ops::{BitAnd, BitOr, BitXor},
    };

    fn test_bitwise<T>() -> Result<(), Box<dyn std::error::Error>>
    where
        T: Bounded + Copy + Num + BitAnd<Output = T> + BitOr<Output = T> + BitXor<Output = T>,
        Value: From<T>,
    {
        let n_bits = 8 * mem::size_of::<T>();
        let type_id = format!(
            "{}{n_bits}",
            if T::min_value().is_zero() { 'u' } else { 'i' }
        );

        let program = ProgramParser::new()
            .parse(&format!(
                r#"
                    type Bitwise = Bitwise;
                    type {type_id} = {type_id};
                    type Tuple<{type_id}, {type_id}, {type_id}> = Struct<ut@Tuple, {type_id}, {type_id}, {type_id}>;

                    libfunc {0} = {0};
                    libfunc struct_construct<Tuple<{type_id}, {type_id}, {type_id}>> = struct_construct<Tuple<{type_id}, {type_id}, {type_id}>>;

                    {0}([0], [1], [2]) -> ([3], [4], [5], [6]);
                    struct_construct<Tuple<{type_id}, {type_id}, {type_id}>>([4], [5], [6]) -> ([7]);
                    return([3], [7]);

                    [0]@0([0]: Bitwise, [1]: {type_id}, [2]: {type_id}) -> (Bitwise, Tuple<{type_id}, {type_id}, {type_id}>);
                "#,
                if n_bits == 128 {
                    "bitwise".to_string()
                } else {
                    format!("{type_id}_bitwise")
                }
            ))
            .map_err(|e| e.to_string())?;

        let context = NativeContext::new();
        let module = context.compile(&program, false, None)?;
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default())?;

        let data = [T::min_value(), T::zero(), T::one(), T::max_value()];
        for perm in Itertools::permutations(data.into_iter(), 2) {
            let result = executor.invoke_dynamic(
                &program.funcs[0].id,
                &[perm[0].into(), perm[1].into()],
                None,
            )?;

            assert_eq!(result.builtin_stats.bitwise, 1);
            assert_eq!(
                result.return_value,
                Value::Struct {
                    fields: vec![
                        (perm[0] & perm[1]).into(),
                        (perm[0] ^ perm[1]).into(),
                        (perm[0] | perm[1]).into(),
                    ],
                    debug_name: None,
                },
            );
        }

        Ok(())
    }

    fn test_byte_reverse() -> Result<(), Box<dyn std::error::Error>> {
        let program = ProgramParser::new()
            .parse(
                r#"
                    type Bitwise = Bitwise;
                    type u128 = u128;

                    libfunc u128_byte_reverse = u128_byte_reverse;

                    u128_byte_reverse([0], [1]) -> ([2], [3]);
                    return([2], [3]);

                    [0]@0([0]: Bitwise, [1]: u128) -> (Bitwise, u128);
                "#,
            )
            .map_err(|e| e.to_string())?;

        let context = NativeContext::new();
        let module = context.compile(&program, false, None)?;
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default())?;

        let data = [0u128, 1u128, u128::MAX];
        for value in data.into_iter() {
            let result = executor.invoke_dynamic(&program.funcs[0].id, &[value.into()], None)?;

            assert_eq!(result.builtin_stats.bitwise, 1);
            assert_eq!(result.return_value, Value::Uint128(value.swap_bytes()));
        }

        Ok(())
    }

    fn test_const<T>() -> Result<(), Box<dyn std::error::Error>>
    where
        T: Bounded + Display + Num,
        Value: From<T>,
    {
        let n_bits = 8 * mem::size_of::<T>();
        let type_id = format!(
            "{}{n_bits}",
            if T::min_value().is_zero() { 'u' } else { 'i' }
        );

        let min = T::min_value();
        let max = T::max_value();
        let program = if min.is_zero() {
            ProgramParser::new()
                .parse(&format!(
                    r#"
                        type {type_id} = {type_id};

                        libfunc {type_id}_const<0> = {type_id}_const<0>;
                        libfunc {type_id}_const<1> = {type_id}_const<1>;
                        libfunc {type_id}_const<{max}> = {type_id}_const<{max}>;

                        {type_id}_const<0>() -> ([0]);
                        return([0]);
                        {type_id}_const<1>() -> ([0]);
                        return([0]);
                        {type_id}_const<{max}>() -> ([0]);
                        return([0]);

                        test_zero@0() -> ({type_id});
                        test_one@2() -> ({type_id});
                        test_max@4() -> ({type_id});
                    "#,
                ))
                .map_err(|e| e.to_string())?
        } else {
            ProgramParser::new()
                .parse(&format!(
                    r#"
                        type {type_id} = {type_id};

                        libfunc {type_id}_const<{min}> = {type_id}_const<{min}>;
                        libfunc {type_id}_const<0> = {type_id}_const<0>;
                        libfunc {type_id}_const<1> = {type_id}_const<1>;
                        libfunc {type_id}_const<{max}> = {type_id}_const<{max}>;

                        {type_id}_const<{min}>() -> ([0]);
                        return([0]);
                        {type_id}_const<0>() -> ([0]);
                        return([0]);
                        {type_id}_const<1>() -> ([0]);
                        return([0]);
                        {type_id}_const<{max}>() -> ([0]);
                        return([0]);

                        test_min@0() -> ({type_id});
                        test_zero@2() -> ({type_id});
                        test_one@4() -> ({type_id});
                        test_max@6() -> ({type_id});
                    "#,
                ))
                .map_err(|e| e.to_string())?
        };

        let context = NativeContext::new();
        let module = context.compile(&program, false, None)?;
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default())?;

        if min.is_zero() {
            assert_eq!(
                executor
                    .invoke_dynamic(&program.funcs[0].id, &[], None)?
                    .return_value,
                T::zero().into(),
            );
            assert_eq!(
                executor
                    .invoke_dynamic(&program.funcs[1].id, &[], None)?
                    .return_value,
                T::one().into(),
            );
            assert_eq!(
                executor
                    .invoke_dynamic(&program.funcs[2].id, &[], None)?
                    .return_value,
                max.into(),
            );
        } else {
            assert_eq!(
                executor
                    .invoke_dynamic(&program.funcs[0].id, &[], None)?
                    .return_value,
                min.into(),
            );
            assert_eq!(
                executor
                    .invoke_dynamic(&program.funcs[1].id, &[], None)?
                    .return_value,
                T::zero().into(),
            );
            assert_eq!(
                executor
                    .invoke_dynamic(&program.funcs[2].id, &[], None)?
                    .return_value,
                T::one().into(),
            );
            assert_eq!(
                executor
                    .invoke_dynamic(&program.funcs[3].id, &[], None)?
                    .return_value,
                max.into(),
            );
        }

        Ok(())
    }

    fn test_diff<T>() -> Result<(), Box<dyn std::error::Error>>
    where
        T: Bounded + Copy + Num + Ord,
        Value: From<T>,
    {
        let n_bits = 8 * mem::size_of::<T>();
        let type_id = format!("i{n_bits}");
        let target_type_id = format!("u{n_bits}");

        let program = ProgramParser::new()
            .parse(&format!(
                r#"
                    type RangeCheck = RangeCheck;
                    type {type_id} = {type_id};
                    type {target_type_id} = {target_type_id};
                    type Result<{target_type_id}, {target_type_id}> = Enum<ut@core::result::Result::<core::integer::{target_type_id}, core::integer::{target_type_id}>, {target_type_id}, {target_type_id}>;

                    libfunc {type_id}_diff = {type_id}_diff;
                    libfunc branch_align = branch_align;
                    libfunc enum_init<Result<{target_type_id}, {target_type_id}>, 0> = enum_init<Result<{target_type_id}, {target_type_id}>, 0>;
                    libfunc enum_init<Result<{target_type_id}, {target_type_id}>, 1> = enum_init<Result<{target_type_id}, {target_type_id}>, 1>;

                    {type_id}_diff([0], [1], [2]) {{ fallthrough([3], [4]) 4([3], [4]) }};
                    branch_align() -> ();
                    enum_init<Result<{target_type_id}, {target_type_id}>, 0>([4]) -> ([5]);
                    return([3], [5]);
                    branch_align() -> ();
                    enum_init<Result<{target_type_id}, {target_type_id}>, 1>([4]) -> ([5]);
                    return([3], [5]);

                    [0]@0([0]: RangeCheck, [1]: {type_id}, [2]: {type_id}) -> (RangeCheck, Result<{target_type_id}, {target_type_id}>);
                "#
            ))
            .map_err(|e| e.to_string())?;

        let context = NativeContext::new();
        let module = context.compile(&program, false, None)?;
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default())?;

        let data = [T::min_value(), T::zero(), T::one(), T::max_value()];
        for perm in Itertools::permutations(data.into_iter(), 2) {
            let lhs = Value::from(perm[0]);
            let rhs = Value::from(perm[1]);

            let result =
                executor.invoke_dynamic(&program.funcs[0].id, &[lhs.clone(), rhs.clone()], None)?;

            let is_greater_equal = perm[0] >= perm[1];
            let value_difference = match (lhs, rhs) {
                (Value::Sint8(lhs), Value::Sint8(rhs)) => {
                    Value::Uint8((lhs.wrapping_sub(rhs)) as _)
                }
                (Value::Sint16(lhs), Value::Sint16(rhs)) => {
                    Value::Uint16((lhs.wrapping_sub(rhs)) as _)
                }
                (Value::Sint32(lhs), Value::Sint32(rhs)) => {
                    Value::Uint32((lhs.wrapping_sub(rhs)) as _)
                }
                (Value::Sint64(lhs), Value::Sint64(rhs)) => {
                    Value::Uint64((lhs.wrapping_sub(rhs)) as _)
                }
                (Value::Sint128(lhs), Value::Sint128(rhs)) => {
                    Value::Uint128((lhs.wrapping_sub(rhs)) as _)
                }
                _ => unreachable!(),
            };

            assert_eq!(result.builtin_stats.range_check, 1);
            assert_eq!(
                result.return_value,
                Value::Enum {
                    tag: (!is_greater_equal) as usize,
                    value: Box::new(value_difference),
                    debug_name: None,
                },
            );
        }

        Ok(())
    }

    fn test_divmod<T>() -> Result<(), Box<dyn std::error::Error>>
    where
        T: Bounded + Copy + Num,
        Value: From<T>,
    {
        let n_bits = 8 * mem::size_of::<T>();
        let type_id = format!(
            "{}{n_bits}",
            if T::min_value().is_zero() { 'u' } else { 'i' }
        );

        let program = ProgramParser::new()
            .parse(&format!(
                r#"
                    type RangeCheck = RangeCheck;
                    type {type_id} = {type_id};
                    type NonZero<{type_id}> = NonZero<{type_id}>;
                    type Tuple<{type_id}, {type_id}> = Struct<ut@Tuple, {type_id}, {type_id}>;

                    libfunc {type_id}_safe_divmod = {type_id}_safe_divmod;
                    libfunc struct_construct<Tuple<{type_id}, {type_id}>> = struct_construct<Tuple<{type_id}, {type_id}>>;

                    {type_id}_safe_divmod([0], [1], [2]) -> ([3], [4], [5]);
                    struct_construct<Tuple<{type_id}, {type_id}>>([4], [5]) -> ([6]);
                    return([3], [6]);

                    [0]@0([0]: RangeCheck, [1]: {type_id}, [2]: NonZero<{type_id}>) -> (RangeCheck, Tuple<{type_id}, {type_id}>);
                "#,
            ))
            .map_err(|e| e.to_string())?;

        let context = NativeContext::new();
        let module = context.compile(&program, false, None)?;
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default())?;

        let data = [T::min_value(), T::zero(), T::one(), T::max_value()];
        for perm in Itertools::permutations(data.into_iter(), 2) {
            if perm[1].is_zero() {
                continue;
            }

            let result = executor.invoke_dynamic(
                &program.funcs[0].id,
                &[perm[0].into(), perm[1].into()],
                None,
            )?;

            assert_eq!(result.builtin_stats.range_check, 1);
            assert_eq!(
                result.return_value,
                Value::Struct {
                    fields: vec![(perm[0] / perm[1]).into(), (perm[0] % perm[1]).into()],
                    debug_name: None,
                },
            );
        }

        Ok(())
    }

    fn test_equal<T>() -> Result<(), Box<dyn std::error::Error>>
    where
        T: Bounded + Copy + Num,
        Value: From<T>,
    {
        let n_bits = 8 * mem::size_of::<T>();
        let type_id = format!(
            "{}{n_bits}",
            if T::min_value().is_zero() { 'u' } else { 'i' }
        );

        let program = ProgramParser::new()
            .parse(&format!(
                r#"
                    type Unit = Struct<ut@Tuple>;
                    type {type_id} = {type_id};
                    type core::bool = Enum<ut@core::bool, Unit, Unit>;

                    libfunc struct_construct<Unit> = struct_construct<Unit>;
                    libfunc {type_id}_eq = {type_id}_eq;
                    libfunc branch_align = branch_align;
                    libfunc enum_init<core::bool, 0> = enum_init<core::bool, 0>;
                    libfunc enum_init<core::bool, 1> = enum_init<core::bool, 1>;

                    struct_construct<Unit>() -> ([2]);
                    {type_id}_eq([0], [1]) {{ fallthrough() 5() }};
                    branch_align() -> ();
                    enum_init<core::bool, 0>([2]) -> ([3]);
                    return([3]);
                    branch_align() -> ();
                    enum_init<core::bool, 1>([2]) -> ([3]);
                    return([3]);

                    [0]@0([0]: {type_id}, [1]: {type_id}) -> (core::bool);
                "#,
            ))
            .map_err(|e| e.to_string())?;

        let context = NativeContext::new();
        let module = context.compile(&program, false, None)?;
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default())?;

        let data = [T::min_value(), T::zero(), T::one(), T::max_value()];
        for perm in Itertools::permutations(data.into_iter(), 2) {
            let result = executor.invoke_dynamic(
                &program.funcs[0].id,
                &[perm[0].into(), perm[1].into()],
                None,
            )?;

            assert_eq!(
                result.return_value,
                Value::Enum {
                    tag: (perm[0] == perm[1]) as usize,
                    value: Box::new(Value::Struct {
                        fields: Vec::new(),
                        debug_name: None,
                    }),
                    debug_name: None,
                },
            );
        }

        Ok(())
    }

    fn test_from_felt252<T>() -> Result<(), Box<dyn std::error::Error>>
    where
        T: Bounded + Copy + Num + TryFrom<Value>,
        Felt: From<T>,
        Value: From<T>,
    {
        let n_bits = 8 * mem::size_of::<T>();
        let type_id = format!(
            "{}{n_bits}",
            if T::min_value().is_zero() { 'u' } else { 'i' }
        );

        let program = ProgramParser::new()
            .parse(&format!(
                r#"
                    type RangeCheck = RangeCheck;
                    type felt252 = felt252;
                    type {type_id} = {type_id};
                    type Unit = Struct<ut@Tuple>;
                    type core::option::Option::<core::integer::{type_id}> = Enum<ut@core::option::Option::<core::integer::{type_id}>, {type_id}, Unit>;

                    libfunc {type_id}_try_from_felt252 = {type_id}_try_from_felt252;
                    libfunc branch_align = branch_align;
                    libfunc enum_init<core::option::Option::<core::integer::{type_id}>, 0> = enum_init<core::option::Option::<core::integer::{type_id}>, 0>;
                    libfunc struct_construct<Unit> = struct_construct<Unit>;
                    libfunc enum_init<core::option::Option::<core::integer::{type_id}>, 1> = enum_init<core::option::Option::<core::integer::{type_id}>, 1>;

                    {type_id}_try_from_felt252([0], [1]) {{ fallthrough([2], [3]) 4([2]) }};
                    branch_align() -> ();
                    enum_init<core::option::Option::<core::integer::{type_id}>, 0>([3]) -> ([4]);
                    return([2], [4]);
                    branch_align() -> ();
                    struct_construct<Unit>() -> ([3]);
                    enum_init<core::option::Option::<core::integer::{type_id}>, 1>([3]) -> ([4]);
                    return([2], [4]);

                    [0]@0([0]: RangeCheck, [1]: felt252) -> (RangeCheck, core::option::Option::<core::integer::{type_id}>);
                "#,
            ))
            .map_err(|e| e.to_string())?;

        let context = NativeContext::new();
        let module = context.compile(&program, false, None)?;
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default())?;

        let data = [
            (Felt::from(T::min_value()), Some(T::min_value())),
            (Felt::from(T::zero()), Some(T::zero())),
            (Felt::from(T::one()), Some(T::one())),
            (Felt::from(T::max_value()), Some(T::max_value())),
            (Felt::ZERO, Some(T::zero())),
            (
                Felt::MAX,
                (T::min_value() != T::zero()).then(|| T::zero() - T::one()),
            ),
            (
                BigInt::from_biguint(Sign::Plus, HALF_PRIME.clone()).into(),
                None,
            ),
            (
                BigInt::from_biguint(Sign::Minus, HALF_PRIME.clone()).into(),
                None,
            ),
        ];
        for (value, target) in data {
            let result = executor.invoke_dynamic(&program.funcs[0].id, &[value.into()], None)?;

            assert_eq!(result.builtin_stats.range_check, 1);
            assert_eq!(
                result.return_value,
                match target {
                    Some(x) => Value::Enum {
                        tag: 0,
                        value: Box::new(x.into()),
                        debug_name: None,
                    },
                    None => Value::Enum {
                        tag: 1,
                        value: Box::new(Value::Struct {
                            fields: Vec::new(),
                            debug_name: None,
                        }),
                        debug_name: None,
                    },
                },
            );
        }

        Ok(())
    }

    fn test_guarantee_mul() -> Result<(), Box<dyn std::error::Error>> {
        let program = ProgramParser::new()
            .parse(
                r#"
                    type RangeCheck = RangeCheck;
                    type u128 = u128;
                    type U128MulGuarantee = U128MulGuarantee;
                    type Tuple<u128, u128> = Struct<ut@Tuple, u128, u128>;

                    libfunc u128_guarantee_mul = u128_guarantee_mul;
                    libfunc u128_mul_guarantee_verify = u128_mul_guarantee_verify;
                    libfunc struct_construct<Tuple<u128, u128>> = struct_construct<Tuple<u128, u128>>;

                    u128_guarantee_mul([1], [2]) -> ([3], [4], [5]);
                    u128_mul_guarantee_verify([0], [5]) -> ([0]);
                    struct_construct<Tuple<u128, u128>>([3], [4]) -> ([6]);
                    return([0], [6]);

                    [0]@0([0]: RangeCheck, [1]: u128, [2]: u128) -> (RangeCheck, Tuple<u128, u128>);
                "#,
            )
            .map_err(|e| e.to_string())?;

        let context = NativeContext::new();
        let module = context.compile(&program, false, None)?;
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default())?;

        let data = [0u128, 1u128, u128::MAX];
        for values in data.into_iter().permutations(2) {
            let result = executor.invoke_dynamic(
                &program.funcs[0].id,
                &[values[0].into(), values[1].into()],
                None,
            )?;

            let lhs = BigUint::from(values[0]);
            let rhs = BigUint::from(values[1]);
            let res = lhs * rhs;

            let mut res_bytes = res.to_bytes_le();
            res_bytes.resize(size_of::<u128>() * 2, 0);
            let lo = u128::from_le_bytes(res_bytes[..16].try_into().unwrap());
            let hi = u128::from_le_bytes(res_bytes[16..].try_into().unwrap());

            assert_eq!(result.builtin_stats.range_check, 1);
            assert_eq!(
                result.return_value,
                Value::Struct {
                    fields: vec![Value::Uint128(hi), Value::Uint128(lo)],
                    debug_name: None,
                },
            );
        }

        Ok(())
    }

    fn test_is_zero<T>() -> Result<(), Box<dyn std::error::Error>>
    where
        T: Bounded + Copy + Num,
        Value: From<T>,
    {
        let n_bits = 8 * mem::size_of::<T>();
        let type_id = format!(
            "{}{n_bits}",
            if T::min_value().is_zero() { 'u' } else { 'i' }
        );

        let program = ProgramParser::new()
            .parse(&format!(
                r#"
                    type {type_id} = {type_id};
                    type Unit = Struct<ut@Tuple>;
                    type NonZero<{type_id}> = NonZero<{type_id}>;
                    type IsZeroResult<{type_id}> = Enum<ut@core::zeroable::IsZeroResult::<core::integer::{type_id}>, Unit, NonZero<{type_id}>>;

                    libfunc {type_id}_is_zero = {type_id}_is_zero;
                    libfunc branch_align = branch_align;
                    libfunc struct_construct<Unit> = struct_construct<Unit>;
                    libfunc enum_init<IsZeroResult<{type_id}>, 0> = enum_init<IsZeroResult<{type_id}>, 0>;
                    libfunc enum_init<IsZeroResult<{type_id}>, 1> = enum_init<IsZeroResult<{type_id}>, 1>;

                    {type_id}_is_zero([0]) {{ fallthrough() 5([2]) }};
                    branch_align() -> ();
                    struct_construct<Unit>() -> ([2]);
                    enum_init<IsZeroResult<{type_id}>, 0>([2]) -> ([3]);
                    return([3]);
                    branch_align() -> ();
                    enum_init<IsZeroResult<{type_id}>, 1>([2]) -> ([3]);
                    return([3]);

                    [0]@0([0]: {type_id}) -> (IsZeroResult<{type_id}>);
                "#,
            ))
            .map_err(|e| e.to_string())?;

        let context = NativeContext::new();
        let module = context.compile(&program, false, None)?;
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default())?;

        let data = [T::min_value(), T::zero(), T::one(), T::max_value()];
        for value in data.into_iter() {
            let result = executor.invoke_dynamic(&program.funcs[0].id, &[value.into()], None)?;

            assert_eq!(
                result.return_value,
                Value::Enum {
                    tag: !value.is_zero() as usize,
                    value: Box::new(if value.is_zero() {
                        Value::Struct {
                            fields: Vec::new(),
                            debug_name: None,
                        }
                    } else {
                        value.into()
                    }),
                    debug_name: None,
                },
            );
        }

        Ok(())
    }

    fn test_unsigned_operation<T>() -> Result<(), Box<dyn std::error::Error>>
    where
        T: Bounded + Copy + Num + OverflowingAdd + OverflowingSub,
        Value: From<T>,
    {
        let n_bits = 8 * size_of::<T>();
        let type_id = format!("u{n_bits}");

        let program = ProgramParser::new()
            .parse(&format!(
                r#"
                    type {type_id} = {type_id};
                    type Result<{type_id}, {type_id}> = Enum<ut@core::result::Result::<core::integer::{type_id}, core::integer::{type_id}>, {type_id}, {type_id}>;
                    type Tuple<Result<{type_id}, {type_id}>, Result<{type_id}, {type_id}>> = Struct<ut@Tuple, Result<{type_id}, {type_id}>, Result<{type_id}, {type_id}>>;
                    type RangeCheck = RangeCheck;

                    libfunc dup<{type_id}> = dup<{type_id}>;
                    libfunc {type_id}_overflowing_add = {type_id}_overflowing_add;
                    libfunc branch_align = branch_align;
                    libfunc enum_init<Result<{type_id}, {type_id}>, 0> = enum_init<Result<{type_id}, {type_id}>, 0>;
                    libfunc jump = jump;
                    libfunc enum_init<Result<{type_id}, {type_id}>, 1> = enum_init<Result<{type_id}, {type_id}>, 1>;
                    libfunc {type_id}_overflowing_sub = {type_id}_overflowing_sub;
                    libfunc struct_construct<Tuple<Result<{type_id}, {type_id}>, Result<{type_id}, {type_id}>>> = struct_construct<Tuple<Result<{type_id}, {type_id}>, Result<{type_id}, {type_id}>>>;

                    dup<{type_id}>([1]) -> ([1], [3]);
                    dup<{type_id}>([2]) -> ([2], [4]);
                    {type_id}_overflowing_add([0], [1], [2]) {{ fallthrough([5], [6]) 6([5], [6]) }};
                    branch_align() -> ();
                    enum_init<Result<{type_id}, {type_id}>, 0>([6]) -> ([6]);
                    jump() {{ 8() }};
                    branch_align() -> ();
                    enum_init<Result<{type_id}, {type_id}>, 1>([6]) -> ([6]);
                    {type_id}_overflowing_sub([5], [3], [4]) {{ fallthrough([7], [8]) 12([7], [8]) }};
                    branch_align() -> ();
                    enum_init<Result<{type_id}, {type_id}>, 0>([8]) -> ([8]);
                    jump() {{ 14() }};
                    branch_align() -> ();
                    enum_init<Result<{type_id}, {type_id}>, 1>([8]) -> ([8]);
                    struct_construct<Tuple<Result<{type_id}, {type_id}>, Result<{type_id}, {type_id}>>>([6], [8]) -> ([9]);
                    return([7], [9]);

                    [0]@0([0]: RangeCheck, [1]: {type_id}, [2]: {type_id}) -> (RangeCheck, Tuple<Result<{type_id}, {type_id}>, Result<{type_id}, {type_id}>>);
                "#,
            ))
            .map_err(|e| e.to_string())?;

        let context = NativeContext::new();
        let module = context.compile(&program, false, None)?;
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default())?;

        let data = [T::min_value(), T::zero(), T::one(), T::max_value()];
        for values in data.into_iter().permutations(2) {
            let lhs = values[0];
            let rhs = values[1];

            let result =
                executor.invoke_dynamic(&program.funcs[0].id, &[lhs.into(), rhs.into()], None)?;

            let (add_result, add_overflow) = lhs.overflowing_add(&rhs);
            let (sub_result, sub_overflow) = lhs.overflowing_sub(&rhs);
            assert_eq!(
                result.return_value,
                Value::Struct {
                    fields: vec![
                        Value::Enum {
                            tag: add_overflow as usize,
                            value: Box::new(add_result.into()),
                            debug_name: None,
                        },
                        Value::Enum {
                            tag: sub_overflow as usize,
                            value: Box::new(sub_result.into()),
                            debug_name: None,
                        },
                    ],
                    debug_name: None
                },
            );
        }

        Ok(())
    }

    fn test_signed_operation<T>() -> Result<(), Box<dyn std::error::Error>>
    where
        T: Bounded + Copy + Num + Ord + OverflowingAdd + OverflowingSub,
        Value: From<T>,
    {
        let n_bits = 8 * size_of::<T>();
        let type_id = format!("i{n_bits}");

        let program = ProgramParser::new()
            .parse(&format!(
                r#"
                    type {type_id} = {type_id};
                    type SignedIntegerResult<{type_id}> = Enum<ut@core::integer::SignedIntegerResult::<core::integer::{type_id}>, {type_id}, {type_id}, {type_id}>;
                    type Tuple<SignedIntegerResult<{type_id}>, SignedIntegerResult<{type_id}>> = Struct<ut@Tuple, SignedIntegerResult<{type_id}>, SignedIntegerResult<{type_id}>>;
                    type RangeCheck = RangeCheck;

                    libfunc dup<{type_id}> = dup<{type_id}>;
                    libfunc {type_id}_overflowing_add_impl = {type_id}_overflowing_add_impl;
                    libfunc branch_align = branch_align;
                    libfunc enum_init<SignedIntegerResult<{type_id}>, 0> = enum_init<SignedIntegerResult<{type_id}>, 0>;
                    libfunc jump = jump;
                    libfunc enum_init<SignedIntegerResult<{type_id}>, 1> = enum_init<SignedIntegerResult<{type_id}>, 1>;
                    libfunc enum_init<SignedIntegerResult<{type_id}>, 2> = enum_init<SignedIntegerResult<{type_id}>, 2>;
                    libfunc {type_id}_overflowing_sub_impl = {type_id}_overflowing_sub_impl;
                    libfunc struct_construct<Tuple<SignedIntegerResult<{type_id}>, SignedIntegerResult<{type_id}>>> = struct_construct<Tuple<SignedIntegerResult<{type_id}>, SignedIntegerResult<{type_id}>>>;

                    dup<{type_id}>([1]) -> ([1], [3]);
                    dup<{type_id}>([2]) -> ([2], [4]);
                    {type_id}_overflowing_add_impl([0], [1], [2]) {{ fallthrough([5], [6]) 6([5], [6]) 9([5], [6]) }};
                    branch_align() -> ();
                    enum_init<SignedIntegerResult<{type_id}>, 0>([6]) -> ([6]);
                    jump() {{ 11() }};
                    branch_align() -> ();
                    enum_init<SignedIntegerResult<{type_id}>, 1>([6]) -> ([6]);
                    jump() {{ 11() }};
                    branch_align() -> ();
                    enum_init<SignedIntegerResult<{type_id}>, 2>([6]) -> ([6]);
                    {type_id}_overflowing_sub_impl([5], [3], [4]) {{ fallthrough([7], [8]) 15([7], [8]) 18([7], [8]) }};
                    branch_align() -> ();
                    enum_init<SignedIntegerResult<{type_id}>, 0>([8]) -> ([8]);
                    jump() {{ 20() }};
                    branch_align() -> ();
                    enum_init<SignedIntegerResult<{type_id}>, 1>([8]) -> ([8]);
                    jump() {{ 20() }};
                    branch_align() -> ();
                    enum_init<SignedIntegerResult<{type_id}>, 2>([8]) -> ([8]);
                    struct_construct<Tuple<SignedIntegerResult<{type_id}>, SignedIntegerResult<{type_id}>>>([6], [8]) -> ([9]);
                    return([7], [9]);

                    [0]@0([0]: RangeCheck, [1]: {type_id}, [2]: {type_id}) -> (RangeCheck, Tuple<SignedIntegerResult<{type_id}>, SignedIntegerResult<{type_id}>>);
                "#,
            ))
            .map_err(|e| e.to_string())?;

        let context = NativeContext::new();
        let module = context.compile(&program, false, None)?;
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default())?;

        let data = [T::min_value(), T::zero(), T::one(), T::max_value()];
        for values in data.into_iter().permutations(2) {
            let lhs = values[0];
            let rhs = values[1];

            let result =
                executor.invoke_dynamic(&program.funcs[0].id, &[lhs.into(), rhs.into()], None)?;

            let (add_result, add_overflow) = lhs.overflowing_add(&rhs);
            let (sub_result, sub_overflow) = lhs.overflowing_sub(&rhs);
            assert_eq!(
                result.return_value,
                Value::Struct {
                    fields: vec![
                        Value::Enum {
                            tag: add_overflow
                                .then(|| lhs >= T::zero() || rhs >= T::zero())
                                .map(|x| (x as usize) + 1)
                                .unwrap_or_default(),
                            value: Box::new(add_result.into()),
                            debug_name: None,
                        },
                        Value::Enum {
                            tag: sub_overflow
                                .then(|| lhs > rhs)
                                .map(|x| (x as usize) + 1)
                                .unwrap_or_default(),
                            value: Box::new(sub_result.into()),
                            debug_name: None,
                        },
                    ],
                    debug_name: None
                },
            );
        }

        Ok(())
    }

    fn test_square_root<T>() -> Result<(), Box<dyn std::error::Error>>
    where
        T: Bounded + Copy + Num + Eq,
        Value: From<T>,
    {
        let n_bits = 8 * size_of::<T>();
        let type_id = format!("u{n_bits}");
        let target_type_id = format!("u{}", (n_bits >> 1).max(8));

        let program = ProgramParser::new()
            .parse(&format!(
                r#"
                    type RangeCheck = RangeCheck;
                    type {type_id} = {type_id};{}

                    libfunc {type_id}_sqrt = {type_id}_sqrt;

                    {type_id}_sqrt([0], [1]) -> ([2], [3]);
                    return([2], [3]);

                    [0]@0([0]: RangeCheck, [1]: {type_id}) -> (RangeCheck, {target_type_id});
                "#,
                match n_bits {
                    8 => "".to_string(),
                    _ => format!("\ntype {target_type_id} = {target_type_id};"),
                }
            ))
            .map_err(|e| e.to_string())?;

        let context = NativeContext::new();
        let module = context.compile(&program, false, None)?;
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default())?;

        let data = [T::min_value(), T::zero(), T::one(), T::max_value()];
        for value in data.into_iter() {
            let result = executor.invoke_dynamic(&program.funcs[0].id, &[value.into()], None)?;

            match (Value::from(value), result.return_value) {
                (Value::Uint8(target), Value::Uint8(result)) => {
                    assert_eq!(result, target.sqrt());
                }
                (Value::Uint16(target), Value::Uint8(result)) => {
                    assert_eq!(result as u16, target.sqrt());
                }
                (Value::Uint32(target), Value::Uint16(result)) => {
                    assert_eq!(result as u32, target.sqrt());
                }
                (Value::Uint64(target), Value::Uint32(result)) => {
                    assert_eq!(result as u64, target.sqrt());
                }
                (Value::Uint128(target), Value::Uint64(result)) => {
                    assert_eq!(result as u128, target.sqrt());
                }
                _ => unreachable!(),
            }
        }

        Ok(())
    }

    fn test_to_felt252<T>() -> Result<(), Box<dyn std::error::Error>>
    where
        T: Bounded + Copy + Num,
        Felt: From<T>,
        Value: From<T>,
    {
        let n_bits = 8 * size_of::<T>();
        let type_id = format!(
            "{}{n_bits}",
            if T::min_value().is_zero() { 'u' } else { 'i' }
        );

        let program = ProgramParser::new()
            .parse(&format!(
                r#"
                    type felt252 = felt252;
                    type {type_id} = {type_id};

                    libfunc {type_id}_to_felt252 = {type_id}_to_felt252;

                    {type_id}_to_felt252([0]) -> ([1]);
                    return([1]);

                    [0]@0([0]: {type_id}) -> (felt252);
                "#
            ))
            .map_err(|e| e.to_string())?;

        let context = NativeContext::new();
        let module = context.compile(&program, false, None)?;
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default())?;

        let data = [T::min_value(), T::zero(), T::one(), T::max_value()];
        for value in data.into_iter() {
            let result = executor.invoke_dynamic(&program.funcs[0].id, &[value.into()], None)?;

            assert_eq!(result.return_value, Value::Felt252(value.into()));
        }

        Ok(())
    }

    fn test_u128s_from_felt252() -> Result<(), Box<dyn std::error::Error>> {
        let program = ProgramParser::new()
            .parse(
                r#"
                    type RangeCheck = RangeCheck;
                    type felt252 = felt252;
                    type u128 = u128;
                    type Tuple<u128, u128> = Struct<ut@Tuple, u128, u128>;
                    type U128sFromFelt252Result = Enum<ut@sample::sample::U128sFromFelt252Result, u128, Tuple<u128, u128>>;

                    libfunc u128s_from_felt252 = u128s_from_felt252;
                    libfunc branch_align = branch_align;
                    libfunc enum_init<U128sFromFelt252Result, 0> = enum_init<U128sFromFelt252Result, 0>;
                    libfunc enum_init<U128sFromFelt252Result, 1> = enum_init<U128sFromFelt252Result, 1>;
                    libfunc struct_construct<Tuple<u128, u128>> = struct_construct<Tuple<u128, u128>>;

                    u128s_from_felt252([0], [1]) { fallthrough([2], [3]) 4([2], [3], [4]) };
                    branch_align() -> ();
                    enum_init<U128sFromFelt252Result, 0>([3]) -> ([4]);
                    return([2], [4]);
                    branch_align() -> ();
                    struct_construct<Tuple<u128, u128>>([3], [4]) -> ([5]);
                    enum_init<U128sFromFelt252Result, 1>([5]) -> ([6]);
                    return([2], [6]);

                    [0]@0([0]: RangeCheck, [1]: felt252) -> (RangeCheck, U128sFromFelt252Result);
                "#,
            )
            .map_err(|e| e.to_string())?;

        let context = NativeContext::new();
        let module = context.compile(&program, false, None)?;
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default())?;

        let data = [
            Felt::from(BigInt::from_biguint(Sign::Minus, HALF_PRIME.clone())),
            Felt::from(BigInt::ZERO),
            Felt::from(BigInt::from(1)),
            Felt::from(BigInt::from_biguint(Sign::Plus, HALF_PRIME.clone())),
        ];
        for value in data.into_iter() {
            let result = executor.invoke_dynamic(&program.funcs[0].id, &[value.into()], None)?;

            let value_bytes = value.to_bytes_le();
            let lo = u128::from_le_bytes(value_bytes[..16].try_into().unwrap());
            let hi = u128::from_le_bytes(value_bytes[16..].try_into().unwrap());

            assert_eq!(result.builtin_stats.range_check, 1);
            assert_eq!(
                result.return_value,
                Value::Enum {
                    tag: (hi != 0) as usize,
                    value: Box::new(if hi == 0 {
                        Value::Uint128(lo)
                    } else {
                        Value::Struct {
                            fields: vec![Value::Uint128(hi), Value::Uint128(lo)],
                            debug_name: None,
                        }
                    }),
                    debug_name: None,
                },
            );
        }

        Ok(())
    }

    fn test_wide_mul<T>() -> Result<(), Box<dyn std::error::Error>>
    where
        T: Bounded + Copy + Num,
        Value: From<T>,
    {
        let n_bits = 8 * size_of::<T>();
        let type_id = format!(
            "{}{n_bits}",
            if T::min_value().is_zero() { 'u' } else { 'i' }
        );
        let target_type_id = format!(
            "{}{}",
            if T::min_value().is_zero() { 'u' } else { 'i' },
            n_bits << 1,
        );

        let program = ProgramParser::new()
            .parse(&format!(
                r#"
                    type {type_id} = {type_id};
                    type {target_type_id} = {target_type_id};

                    libfunc {type_id}_wide_mul = {type_id}_wide_mul;

                    {type_id}_wide_mul([0], [1]) -> ([2]);
                    return([2]);

                    [0]@0([0]: {type_id}, [1]: {type_id}) -> ({target_type_id});
                "#
            ))
            .map_err(|e| e.to_string())?;

        let context = NativeContext::new();
        let module = context.compile(&program, false, None)?;
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::default())?;

        let data = [T::min_value(), T::zero(), T::one(), T::max_value()];
        for values in data.into_iter().permutations(2) {
            let result = executor.invoke_dynamic(
                &program.funcs[0].id,
                &[values[0].into(), values[1].into()],
                None,
            )?;

            match (
                Value::from(values[0]),
                Value::from(values[1]),
                result.return_value,
            ) {
                (Value::Uint8(lhs), Value::Uint8(rhs), Value::Uint16(result)) => {
                    assert_eq!(result, (lhs as u16) * (rhs as u16));
                }
                (Value::Uint16(lhs), Value::Uint16(rhs), Value::Uint32(result)) => {
                    assert_eq!(result, (lhs as u32) * (rhs as u32));
                }
                (Value::Uint32(lhs), Value::Uint32(rhs), Value::Uint64(result)) => {
                    assert_eq!(result, (lhs as u64) * (rhs as u64));
                }
                (Value::Uint64(lhs), Value::Uint64(rhs), Value::Uint128(result)) => {
                    assert_eq!(result, (lhs as u128) * (rhs as u128));
                }
                (Value::Sint8(lhs), Value::Sint8(rhs), Value::Sint16(result)) => {
                    assert_eq!(result, (lhs as i16) * (rhs as i16));
                }
                (Value::Sint16(lhs), Value::Sint16(rhs), Value::Sint32(result)) => {
                    assert_eq!(result, (lhs as i32) * (rhs as i32));
                }
                (Value::Sint32(lhs), Value::Sint32(rhs), Value::Sint64(result)) => {
                    assert_eq!(result, (lhs as i64) * (rhs as i64));
                }
                (Value::Sint64(lhs), Value::Sint64(rhs), Value::Sint128(result)) => {
                    assert_eq!(result, (lhs as i128) * (rhs as i128));
                }
                _ => unreachable!(),
            }
        }

        Ok(())
    }

    macro_rules! impl_tests {
        ( $( $target:ident for $( $ty:ty as $name:ident ),+ ; )+ ) => {
            $( $(
                #[test]
                fn $name() {
                    $target::<$ty>().unwrap();
                }
            )+ )+
        };
    }

    impl_tests! {
        test_bitwise for
            u8 as u8_bitwise,
            u16 as u16_bitwise,
            u32 as u32_bitwise,
            u64 as u64_bitwise,
            u128 as u128_bitwise;

        test_const for
            u8 as u8_const,
            u16 as u16_const,
            u32 as u32_const,
            u64 as u64_const,
            u128 as u128_const,
            i8 as i8_const,
            i16 as i16_const,
            i32 as i32_const,
            i64 as i64_const,
            i128 as i128_const;

        test_diff for
            i8 as i8_diff,
            i16 as i16_diff,
            i32 as i32_diff,
            i64 as i64_diff,
            i128 as i128_diff;

        test_divmod for
            u8 as u8_divmod,
            u16 as u16_divmod,
            u32 as u32_divmod,
            u64 as u64_divmod,
            u128 as u128_divmod;

        test_equal for
            u8 as u8_equal,
            u16 as u16_equal,
            u32 as u32_equal,
            u64 as u64_equal,
            u128 as u128_equal,
            i8 as i8_equal,
            i16 as i16_equal,
            i32 as i32_equal,
            i64 as i64_equal,
            i128 as i128_equal;

        test_from_felt252 for
            u8 as u8_from_felt252,
            u16 as u16_from_felt252,
            u32 as u32_from_felt252,
            u64 as u64_from_felt252,
            i8 as i8_from_felt252,
            i16 as i16_from_felt252,
            i32 as i32_from_felt252,
            i64 as i64_from_felt252,
            i128 as i128_from_felt252;

        test_is_zero for
            u8 as u8_is_zero,
            u16 as u16_is_zero,
            u32 as u32_is_zero,
            u64 as u64_is_zero,
            u128 as u128_is_zero,
            i8 as i8_is_zero,
            i16 as i16_is_zero,
            i32 as i32_is_zero,
            i64 as i64_is_zero,
            i128 as i128_is_zero;

        test_unsigned_operation for
            u8 as u8_operation,
            u16 as u16_operation,
            u32 as u32_operation,
            u64 as u64_operation,
            u128 as u128_operation;
        test_signed_operation for
            i8 as i8_operation,
            i16 as i16_operation,
            i32 as i32_operation,
            i64 as i64_operation,
            i128 as i128_operation;

        test_square_root for
            u8 as u8_square_root,
            u16 as u16_square_root,
            u32 as u32_square_root,
            u64 as u64_square_root,
            u128 as u128_square_root;

        test_to_felt252 for
            u8 as u8_to_felt252,
            u16 as u16_to_felt252,
            u32 as u32_to_felt252,
            u64 as u64_to_felt252,
            u128 as u128_to_felt252,
            i8 as i8_to_felt252,
            i16 as i16_to_felt252,
            i32 as i32_to_felt252,
            i64 as i64_to_felt252,
            i128 as i128_to_felt252;

        test_wide_mul for
            u8 as u8_wide_mul,
            u16 as u16_wide_mul,
            u32 as u32_wide_mul,
            u64 as u64_wide_mul,
            i8 as i8_wide_mul,
            i16 as i16_wide_mul,
            i32 as i32_wide_mul,
            i64 as i64_wide_mul;
    }

    #[test]
    fn u128_byte_reverse() {
        test_byte_reverse().unwrap();
    }

    #[test]
    fn u128s_from_felt252() {
        test_u128s_from_felt252().unwrap();
    }

    #[test]
    fn u128_guarantee_mul() {
        test_guarantee_mul().unwrap();
    }
}
