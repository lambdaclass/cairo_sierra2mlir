//! # Bounded int libfuncs

use super::LibfuncHelper;
use crate::{
    error::Result,
    metadata::MetadataStorage,
    native_assert,
    types::TypeBuilder,
    utils::{BlockExt, RangeExt},
};
use cairo_lang_sierra::{
    extensions::{
        bounded_int::{
            BoundedIntConcreteLibfunc, BoundedIntConstrainConcreteLibfunc,
            BoundedIntDivRemConcreteLibfunc, BoundedIntTrimConcreteLibfunc,
        },
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
        utils::Range,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        cf,
    },
    ir::{r#type::IntegerType, Block, BlockLike, Location, Value, ValueLike},
    Context,
};
use num_bigint::{BigInt, Sign};

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &BoundedIntConcreteLibfunc,
) -> Result<()> {
    match selector {
        BoundedIntConcreteLibfunc::Add(info) => {
            build_add(context, registry, entry, location, helper, metadata, info)
        }
        BoundedIntConcreteLibfunc::Sub(info) => {
            build_sub(context, registry, entry, location, helper, metadata, info)
        }
        BoundedIntConcreteLibfunc::Mul(info) => {
            build_mul(context, registry, entry, location, helper, metadata, info)
        }
        BoundedIntConcreteLibfunc::DivRem(info) => {
            build_divrem(context, registry, entry, location, helper, metadata, info)
        }
        BoundedIntConcreteLibfunc::Constrain(info) => {
            build_constrain(context, registry, entry, location, helper, metadata, info)
        }
        BoundedIntConcreteLibfunc::TrimMin(info) | BoundedIntConcreteLibfunc::TrimMax(info) => {
            build_trim(context, registry, entry, location, helper, metadata, info)
        }
        BoundedIntConcreteLibfunc::IsZero(info) => {
            build_is_zero(context, registry, entry, location, helper, metadata, info)
        }
        BoundedIntConcreteLibfunc::WrapNonZero(info) => {
            build_wrap_non_zero(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `bounded_int_add` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_add<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let lhs_value = entry.arg(0)?;
    let rhs_value = entry.arg(1)?;

    // Extract the ranges for the operands and the result type.
    let lhs_ty = registry.get_type(&info.signature.param_signatures[0].ty)?;
    let rhs_ty = registry.get_type(&info.signature.param_signatures[1].ty)?;

    let lhs_range = lhs_ty.integer_range(registry)?;
    let rhs_range = rhs_ty.integer_range(registry)?;
    let dst_range = registry
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)?
        .integer_range(registry)?;

    let lhs_width = if lhs_ty.is_bounded_int(registry)? {
        lhs_range.offset_bit_width()
    } else {
        lhs_range.zero_based_bit_width()
    };
    let rhs_width = if rhs_ty.is_bounded_int(registry)? {
        rhs_range.offset_bit_width()
    } else {
        rhs_range.zero_based_bit_width()
    };

    // Calculate the computation range.
    let compute_range = Range {
        lower: (&lhs_range.lower)
            .min(&rhs_range.lower)
            .min(&dst_range.lower)
            .clone(),
        upper: (&lhs_range.upper)
            .max(&rhs_range.upper)
            .max(&dst_range.upper)
            .clone(),
    };
    let compute_ty = IntegerType::new(context, compute_range.offset_bit_width()).into();

    // Zero-extend operands into the computation range.
    native_assert!(
        compute_range.offset_bit_width() >= lhs_width,
        "the lhs_range bit_width must be less or equal than the compute_range"
    );
    native_assert!(
        compute_range.offset_bit_width() >= rhs_width,
        "the rhs_range bit_width must be less or equal than the compute_range"
    );

    let lhs_value = if compute_range.offset_bit_width() > lhs_width {
        if lhs_range.lower.sign() != Sign::Minus || lhs_ty.is_bounded_int(registry)? {
            entry.extui(lhs_value, compute_ty, location)?
        } else {
            entry.extsi(lhs_value, compute_ty, location)?
        }
    } else {
        lhs_value
    };
    let rhs_value = if compute_range.offset_bit_width() > rhs_width {
        if rhs_range.lower.sign() != Sign::Minus || rhs_ty.is_bounded_int(registry)? {
            entry.extui(rhs_value, compute_ty, location)?
        } else {
            entry.extsi(rhs_value, compute_ty, location)?
        }
    } else {
        rhs_value
    };

    // Offset the operands so that they are compatible.
    let lhs_offset = if lhs_ty.is_bounded_int(registry)? {
        &lhs_range.lower - &compute_range.lower
    } else {
        lhs_range.lower
    };
    let lhs_value = if lhs_offset != BigInt::ZERO {
        let lhs_offset = entry.const_int_from_type(context, location, lhs_offset, compute_ty)?;
        entry.addi(lhs_value, lhs_offset, location)?
    } else {
        lhs_value
    };

    let rhs_offset = if rhs_ty.is_bounded_int(registry)? {
        &rhs_range.lower - &compute_range.lower
    } else {
        rhs_range.lower
    };
    let rhs_value = if rhs_offset != BigInt::ZERO {
        let rhs_offset = entry.const_int_from_type(context, location, rhs_offset, compute_ty)?;
        entry.addi(rhs_value, rhs_offset, location)?
    } else {
        rhs_value
    };

    // Compute the operation.
    let res_value = entry.addi(lhs_value, rhs_value, location)?;

    // Offset and truncate the result to the output type.
    let res_offset = &dst_range.lower - &compute_range.lower;
    let res_value = if res_offset != BigInt::ZERO {
        let res_offset = entry.const_int_from_type(context, location, res_offset, compute_ty)?;
        entry.append_op_result(arith::subi(res_value, res_offset, location))?
    } else {
        res_value
    };

    let res_value = if dst_range.offset_bit_width() < compute_range.offset_bit_width() {
        entry.trunci(
            res_value,
            IntegerType::new(context, dst_range.offset_bit_width()).into(),
            location,
        )?
    } else {
        res_value
    };

    helper.br(entry, 0, &[res_value], location)
}

/// Generate MLIR operations for the `bounded_int_sub` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_sub<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let lhs_value = entry.arg(0)?;
    let rhs_value = entry.arg(1)?;

    // Extract the ranges for the operands and the result type.
    let lhs_ty = registry.get_type(&info.signature.param_signatures[0].ty)?;
    let rhs_ty = registry.get_type(&info.signature.param_signatures[1].ty)?;

    let lhs_range = lhs_ty.integer_range(registry)?;
    let rhs_range = rhs_ty.integer_range(registry)?;
    let dst_range = registry
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)?
        .integer_range(registry)?;

    let lhs_width = if lhs_ty.is_bounded_int(registry)? {
        lhs_range.offset_bit_width()
    } else {
        lhs_range.zero_based_bit_width()
    };
    let rhs_width = if rhs_ty.is_bounded_int(registry)? {
        rhs_range.offset_bit_width()
    } else {
        rhs_range.zero_based_bit_width()
    };

    // Calculate the computation range.
    let compute_range = Range {
        lower: (&lhs_range.lower)
            .min(&rhs_range.lower)
            .min(&dst_range.lower)
            .clone(),
        upper: (&lhs_range.upper)
            .max(&rhs_range.upper)
            .max(&dst_range.upper)
            .clone(),
    };
    let compute_ty = IntegerType::new(context, compute_range.offset_bit_width()).into();

    // Zero-extend operands into the computation range.
    native_assert!(
        compute_range.offset_bit_width() >= lhs_width,
        "the lhs_range bit_width must be less or equal than the compute_range"
    );
    native_assert!(
        compute_range.offset_bit_width() >= rhs_width,
        "the rhs_range bit_width must be less or equal than the compute_range"
    );

    let lhs_value = if compute_range.offset_bit_width() > lhs_width {
        if lhs_range.lower.sign() != Sign::Minus || lhs_ty.is_bounded_int(registry)? {
            entry.extui(lhs_value, compute_ty, location)?
        } else {
            entry.extsi(lhs_value, compute_ty, location)?
        }
    } else {
        lhs_value
    };
    let rhs_value = if compute_range.offset_bit_width() > rhs_width {
        if rhs_range.lower.sign() != Sign::Minus || rhs_ty.is_bounded_int(registry)? {
            entry.extui(rhs_value, compute_ty, location)?
        } else {
            entry.extsi(rhs_value, compute_ty, location)?
        }
    } else {
        rhs_value
    };

    // Offset the operands so that they are compatible.
    let lhs_offset = if lhs_ty.is_bounded_int(registry)? {
        &lhs_range.lower - &compute_range.lower
    } else {
        lhs_range.lower
    };
    let lhs_value = if lhs_offset != BigInt::ZERO {
        let lhs_offset = entry.const_int_from_type(context, location, lhs_offset, compute_ty)?;
        entry.addi(lhs_value, lhs_offset, location)?
    } else {
        lhs_value
    };

    let rhs_offset = if rhs_ty.is_bounded_int(registry)? {
        &rhs_range.lower - &compute_range.lower
    } else {
        rhs_range.lower
    };
    let rhs_value = if rhs_offset != BigInt::ZERO {
        let rhs_offset = entry.const_int_from_type(context, location, rhs_offset, compute_ty)?;
        entry.addi(rhs_value, rhs_offset, location)?
    } else {
        rhs_value
    };

    // Compute the operation.
    let res_value = entry.append_op_result(arith::subi(lhs_value, rhs_value, location))?;

    // Offset and truncate the result to the output type.
    let res_offset = dst_range.lower.clone();
    let res_value = if res_offset != BigInt::ZERO {
        let res_offset = entry.const_int_from_type(context, location, res_offset, compute_ty)?;
        entry.append_op_result(arith::subi(res_value, res_offset, location))?
    } else {
        res_value
    };

    let res_value = if dst_range.offset_bit_width() < compute_range.offset_bit_width() {
        entry.trunci(
            res_value,
            IntegerType::new(context, dst_range.offset_bit_width()).into(),
            location,
        )?
    } else {
        res_value
    };

    helper.br(entry, 0, &[res_value], location)
}

/// Generate MLIR operations for the `bounded_int_mul` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_mul<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let lhs_value = entry.arg(0)?;
    let rhs_value = entry.arg(1)?;

    // Extract the ranges for the operands and the result type.
    let lhs_ty = registry.get_type(&info.signature.param_signatures[0].ty)?;
    let rhs_ty = registry.get_type(&info.signature.param_signatures[1].ty)?;

    let lhs_range = lhs_ty.integer_range(registry)?;
    let rhs_range = rhs_ty.integer_range(registry)?;
    let dst_range = registry
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)?
        .integer_range(registry)?;

    let lhs_width = if lhs_ty.is_bounded_int(registry)? {
        lhs_range.offset_bit_width()
    } else {
        lhs_range.zero_based_bit_width()
    };
    let rhs_width = if rhs_ty.is_bounded_int(registry)? {
        rhs_range.offset_bit_width()
    } else {
        rhs_range.zero_based_bit_width()
    };

    // Calculate the computation range.
    let compute_range = Range {
        lower: (&lhs_range.lower)
            .min(&rhs_range.lower)
            .min(&dst_range.lower)
            .min(&BigInt::ZERO)
            .clone(),
        upper: (&lhs_range.upper)
            .max(&rhs_range.upper)
            .max(&dst_range.upper)
            .clone(),
    };
    let compute_ty = IntegerType::new(context, compute_range.zero_based_bit_width()).into();

    // Zero-extend operands into the computation range.
    native_assert!(
        compute_range.offset_bit_width() >= lhs_width,
        "the lhs_range bit_width must be less or equal than the compute_range"
    );
    native_assert!(
        compute_range.offset_bit_width() >= rhs_width,
        "the rhs_range bit_width must be less or equal than the compute_range"
    );

    let lhs_value = if compute_range.zero_based_bit_width() > lhs_width {
        if lhs_range.lower.sign() != Sign::Minus || lhs_ty.is_bounded_int(registry)? {
            entry.extui(lhs_value, compute_ty, location)?
        } else {
            entry.extsi(lhs_value, compute_ty, location)?
        }
    } else {
        lhs_value
    };
    let rhs_value = if compute_range.zero_based_bit_width() > rhs_width {
        if rhs_range.lower.sign() != Sign::Minus || rhs_ty.is_bounded_int(registry)? {
            entry.extui(rhs_value, compute_ty, location)?
        } else {
            entry.extsi(rhs_value, compute_ty, location)?
        }
    } else {
        rhs_value
    };

    // Offset the operands so that they are compatible with the operation.
    let lhs_value = if lhs_ty.is_bounded_int(registry)? && lhs_range.lower != BigInt::ZERO {
        let lhs_offset =
            entry.const_int_from_type(context, location, lhs_range.lower, compute_ty)?;
        entry.addi(lhs_value, lhs_offset, location)?
    } else {
        lhs_value
    };
    let rhs_value = if rhs_ty.is_bounded_int(registry)? && rhs_range.lower != BigInt::ZERO {
        let rhs_offset =
            entry.const_int_from_type(context, location, rhs_range.lower, compute_ty)?;
        entry.addi(rhs_value, rhs_offset, location)?
    } else {
        rhs_value
    };

    // Compute the operation.
    let res_value = entry.muli(lhs_value, rhs_value, location)?;

    // Offset and truncate the result to the output type.
    let res_offset = (&dst_range.lower).max(&compute_range.lower).clone();
    let res_value = if res_offset != BigInt::ZERO {
        let res_offset = entry.const_int_from_type(context, location, res_offset, compute_ty)?;
        entry.append_op_result(arith::subi(res_value, res_offset, location))?
    } else {
        res_value
    };

    let res_value = if dst_range.offset_bit_width() < compute_range.zero_based_bit_width() {
        entry.trunci(
            res_value,
            IntegerType::new(context, dst_range.offset_bit_width()).into(),
            location,
        )?
    } else {
        res_value
    };

    helper.br(entry, 0, &[res_value], location)
}

/// Generate MLIR operations for the `bounded_int_divrem` libfunc.
/// Libfunc for dividing two non negative BoundedInts and getting the quotient and remainder.
fn build_divrem<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &BoundedIntDivRemConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let lhs_value = entry.arg(1)?;
    let rhs_value = entry.arg(2)?;

    // Extract the ranges for the operands and the result type.
    let lhs_ty = registry.get_type(&info.param_signatures()[1].ty)?;
    let rhs_ty = registry.get_type(&info.param_signatures()[2].ty)?;

    let lhs_range = lhs_ty.integer_range(registry)?;
    let rhs_range = rhs_ty.integer_range(registry)?;
    let div_range = registry
        .get_type(&info.branch_signatures()[0].vars[1].ty)?
        .integer_range(registry)?;
    let rem_range = registry
        .get_type(&info.branch_signatures()[0].vars[2].ty)?
        .integer_range(registry)?;

    let lhs_width = if lhs_ty.is_bounded_int(registry)? {
        lhs_range.offset_bit_width()
    } else {
        lhs_range.zero_based_bit_width()
    };
    let rhs_width = if rhs_ty.is_bounded_int(registry)? {
        rhs_range.offset_bit_width()
    } else {
        rhs_range.zero_based_bit_width()
    };

    // Calculate the computation range.
    let compute_range = Range {
        lower: (&lhs_range.lower)
            .min(&rhs_range.lower)
            .min(&div_range.lower)
            .min(&rem_range.lower)
            .min(&BigInt::ZERO)
            .clone(),
        upper: (&lhs_range.upper)
            .max(&rhs_range.upper)
            .max(&div_range.upper)
            .max(&rem_range.upper)
            .clone(),
    };
    let compute_ty = IntegerType::new(context, compute_range.zero_based_bit_width()).into();

    // Zero-extend operands into the computation range.
    native_assert!(
        compute_range.offset_bit_width() >= lhs_width,
        "the lhs_range bit_width must be less or equal than the compute_range"
    );
    native_assert!(
        compute_range.offset_bit_width() >= rhs_width,
        "the rhs_range bit_width must be less or equal than the compute_range"
    );

    let lhs_value = if compute_range.zero_based_bit_width() > lhs_width {
        if lhs_range.lower.sign() != Sign::Minus || lhs_ty.is_bounded_int(registry)? {
            entry.extui(lhs_value, compute_ty, location)?
        } else {
            entry.extsi(lhs_value, compute_ty, location)?
        }
    } else {
        lhs_value
    };
    let rhs_value = if compute_range.zero_based_bit_width() > rhs_width {
        if rhs_range.lower.sign() != Sign::Minus || rhs_ty.is_bounded_int(registry)? {
            entry.extui(rhs_value, compute_ty, location)?
        } else {
            entry.extsi(rhs_value, compute_ty, location)?
        }
    } else {
        rhs_value
    };

    // Offset the operands so that they are compatible with the operation.
    let lhs_value = if lhs_ty.is_bounded_int(registry)? && lhs_range.lower != BigInt::ZERO {
        let lhs_offset =
            entry.const_int_from_type(context, location, lhs_range.lower, compute_ty)?;
        entry.addi(lhs_value, lhs_offset, location)?
    } else {
        lhs_value
    };
    let rhs_value = if rhs_ty.is_bounded_int(registry)? && rhs_range.lower != BigInt::ZERO {
        let rhs_offset =
            entry.const_int_from_type(context, location, rhs_range.lower, compute_ty)?;
        entry.addi(rhs_value, rhs_offset, location)?
    } else {
        rhs_value
    };

    // Compute the operation.
    let div_value = entry.append_op_result(arith::divui(lhs_value, rhs_value, location))?;
    let rem_value = entry.append_op_result(arith::remui(lhs_value, rhs_value, location))?;

    // Offset and truncate the result to the output type.
    let div_offset = (&div_range.lower).max(&compute_range.lower).clone();
    let rem_offset = (&rem_range.lower).max(&compute_range.lower).clone();

    let div_value = if div_offset != BigInt::ZERO {
        let div_offset = entry.const_int_from_type(context, location, div_offset, compute_ty)?;
        entry.append_op_result(arith::subi(div_value, div_offset, location))?
    } else {
        div_value
    };
    let rem_value = if rem_offset != BigInt::ZERO {
        let rem_offset = entry.const_int_from_type(context, location, rem_offset, compute_ty)?;
        entry.append_op_result(arith::subi(rem_value, rem_offset, location))?
    } else {
        rem_value
    };

    let div_value = if div_range.offset_bit_width() < compute_range.zero_based_bit_width() {
        entry.trunci(
            div_value,
            IntegerType::new(context, div_range.offset_bit_width()).into(),
            location,
        )?
    } else {
        div_value
    };
    let rem_value = if rem_range.offset_bit_width() < compute_range.zero_based_bit_width() {
        entry.trunci(
            rem_value,
            IntegerType::new(context, rem_range.offset_bit_width()).into(),
            location,
        )?
    } else {
        rem_value
    };

    helper.br(entry, 0, &[range_check, div_value, rem_value], location)
}

/// Generate MLIR operations for the `bounded_int_constrain` libfunc.
fn build_constrain<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &BoundedIntConstrainConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;
    let src_value: Value = entry.arg(1)?;

    let src_ty = registry.get_type(&info.param_signatures()[1].ty)?;
    let src_range = src_ty.integer_range(registry)?;

    let src_width = if src_ty.is_bounded_int(registry)? {
        src_range.offset_bit_width()
    } else {
        src_range.zero_based_bit_width()
    };

    let lower_range = registry
        .get_type(&info.branch_signatures()[0].vars[1].ty)?
        .integer_range(registry)?;
    let upper_range = registry
        .get_type(&info.branch_signatures()[1].vars[1].ty)?
        .integer_range(registry)?;

    let boundary =
        entry.const_int_from_type(context, location, info.boundary.clone(), src_value.r#type())?;
    let is_lower = entry.cmpi(
        context,
        if src_range.lower.sign() == Sign::Minus {
            CmpiPredicate::Slt
        } else {
            CmpiPredicate::Ult
        },
        src_value,
        boundary,
        location,
    )?;

    let lower_block = helper.append_block(Block::new(&[]));
    let upper_block = helper.append_block(Block::new(&[]));
    entry.append_operation(cf::cond_br(
        context,
        is_lower,
        lower_block,
        upper_block,
        &[],
        &[],
        location,
    ));

    {
        let res_value = if src_range.lower != lower_range.lower {
            let lower_offset = &lower_range.lower - &src_range.lower;
            let lower_offset = lower_block.const_int_from_type(
                context,
                location,
                lower_offset,
                src_value.r#type(),
            )?;
            lower_block.append_op_result(arith::subi(src_value, lower_offset, location))?
        } else {
            src_value
        };

        let res_value = if src_width > lower_range.offset_bit_width() {
            lower_block.trunci(
                res_value,
                IntegerType::new(context, lower_range.offset_bit_width()).into(),
                location,
            )?
        } else {
            res_value
        };

        helper.br(lower_block, 0, &[range_check, res_value], location)?;
    }

    {
        let res_value = if src_range.lower != upper_range.lower {
            let upper_offset = &upper_range.lower - &src_range.lower;
            let upper_offset = upper_block.const_int_from_type(
                context,
                location,
                upper_offset,
                src_value.r#type(),
            )?;
            upper_block.append_op_result(arith::subi(src_value, upper_offset, location))?
        } else {
            src_value
        };

        let res_value = if src_width > upper_range.offset_bit_width() {
            upper_block.trunci(
                res_value,
                IntegerType::new(context, upper_range.offset_bit_width()).into(),
                location,
            )?
        } else {
            res_value
        };

        helper.br(upper_block, 1, &[range_check, res_value], location)?;
    }

    Ok(())
}

/// Makes a downcast of a type `T` to `BoundedInt<T::MIN, T::MAX - 1>`
/// or `BoundedInt<T::MIN + 1, T::MAX>` where `T` can be any type of signed
/// or unsigned integer.
///
/// ```cairo
/// extern fn bounded_int_trim<T, const TRIMMED_VALUE: felt252, impl H: TrimHelper<T, TRIMMED_VALUE>>(
///     value: T,
/// ) -> core::internal::OptionRev<H::Target> nopanic;
/// ```
fn build_trim<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &BoundedIntTrimConcreteLibfunc,
) -> Result<()> {
    let value: Value = entry.arg(0)?;
    let trimmed_value = entry.const_int_from_type(
        context,
        location,
        info.trimmed_value.clone(),
        value.r#type(),
    )?;
    let trim_type = registry.get_type(&info.param_signatures()[0].ty)?;
    let is_invalid = entry.cmpi(context, CmpiPredicate::Eq, value, trimmed_value, location)?;
    let int_range = trim_type.integer_range(registry)?;

    // There is no need to truncate the value type since we're only receiving power-of-two integers
    // and constraining their range a single value from either the lower or upper limit. However,
    // since we're returning a `BoundedInt` we need to offset its internal representation
    // accordingly.
    let value = if info.trimmed_value == BigInt::ZERO || int_range.lower < BigInt::ZERO {
        let offset = entry.const_int_from_type(
            context,
            location,
            &info.trimmed_value + 1,
            value.r#type(),
        )?;
        entry.append_op_result(arith::subi(value, offset, location))?
    } else {
        value
    };

    helper.cond_br(
        context,
        entry,
        is_invalid,
        [0, 1],
        [&[], &[value]],
        location,
    )
}

/// Generate MLIR operations for the `bounded_int_is_zero` libfunc.
fn build_is_zero<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let src_value: Value = entry.arg(0)?;

    let src_ty = registry.get_type(&info.signature.param_signatures[0].ty)?;
    let src_range = src_ty.integer_range(registry)?;

    native_assert!(
        src_range.lower <= BigInt::ZERO && BigInt::ZERO < src_range.upper,
        "value can never be zero"
    );

    let k0 = entry.const_int_from_type(context, location, 0, src_value.r#type())?;
    let src_is_zero = entry.cmpi(context, CmpiPredicate::Eq, src_value, k0, location)?;

    helper.cond_br(
        context,
        entry,
        src_is_zero,
        [0, 1],
        [&[], &[src_value]],
        location,
    )
}

/// Generate MLIR operations for the `bounded_int_wrap_non_zero` libfunc.
fn build_wrap_non_zero<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let src_range = registry
        .get_type(&info.signature.param_signatures[0].ty)?
        .integer_range(registry)?;

    native_assert!(
        src_range.lower > BigInt::ZERO || BigInt::ZERO >= src_range.upper,
        "value must not be zero"
    );

    super::build_noop::<1, true>(
        context,
        registry,
        entry,
        location,
        helper,
        metadata,
        &info.signature.param_signatures,
    )
}

#[cfg(test)]
mod test {
    use cairo_vm::Felt252;

    use crate::{
        context::NativeContext, execution_result::ExecutionResult, executor::JitNativeExecutor,
        utils::test::load_cairo, OptLevel, Value,
    };

    #[test]
    fn test_trim_some_pos_i8() {
        let (_, program) = load_cairo!(
            use core::internal::{OptionRev, bounded_int::BoundedInt};
            use core::internal::bounded_int;
            fn main() -> BoundedInt<-128, 126> {
                let num = match bounded_int::trim_max::<i8>(1) {
                    OptionRev::Some(n) => n,
                    OptionRev::None => 0,
                };

                num
            }
        );
        let ctx = NativeContext::new();
        let module = ctx.compile(&program, false, None).unwrap();
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::Default).unwrap();
        let ExecutionResult {
            remaining_gas: _,
            return_value,
            builtin_stats: _,
        } = executor
            .invoke_dynamic(&program.funcs[0].id, &[], None)
            .unwrap();

        let Value::BoundedInt { value, range: _ } = return_value else {
            panic!();
        };
        assert_eq!(value, Felt252::from(1_u8));
    }

    #[test]
    fn test_trim_some_neg_i8() {
        let (_, program) = load_cairo!(
            use core::internal::{OptionRev, bounded_int::BoundedInt};
            use core::internal::bounded_int;
            fn main() -> BoundedInt<-127, 127> {
                let num = match bounded_int::trim_min::<i8>(1) {
                    OptionRev::Some(n) => n,
                    OptionRev::None => 1,
                };

                num
            }
        );
        let ctx = NativeContext::new();
        let module = ctx.compile(&program, false, None).unwrap();
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::Default).unwrap();
        let ExecutionResult {
            remaining_gas: _,
            return_value,
            builtin_stats: _,
        } = executor
            .invoke_dynamic(&program.funcs[0].id, &[], None)
            .unwrap();

        let Value::BoundedInt { value, range: _ } = return_value else {
            panic!();
        };
        assert_eq!(value, Felt252::from(1_u8));
    }

    #[test]
    fn test_trim_some_u32() {
        let (_, program) = load_cairo!(
            use core::internal::{OptionRev, bounded_int::BoundedInt};
            use core::internal::bounded_int;
            fn main() -> BoundedInt<0, 4294967294> {
                let num = match bounded_int::trim_max::<u32>(0xfffffffe) {
                    OptionRev::Some(n) => n,
                    OptionRev::None => 0,
                };

                num
            }
        );
        let ctx = NativeContext::new();
        let module = ctx.compile(&program, false, None).unwrap();
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::Default).unwrap();
        let ExecutionResult {
            remaining_gas: _,
            return_value,
            builtin_stats: _,
        } = executor
            .invoke_dynamic(&program.funcs[0].id, &[], None)
            .unwrap();

        let Value::BoundedInt { value, range: _ } = return_value else {
            panic!();
        };
        assert_eq!(value, Felt252::from(0xfffffffe_u32));
    }

    #[test]
    fn test_trim_none() {
        let (_, program) = load_cairo!(
            use core::internal::{OptionRev, bounded_int::BoundedInt};
            use core::internal::bounded_int;
            fn main() -> BoundedInt<-32767, 32767> {
                let num = match bounded_int::trim_min::<i16>(-0x8000) {
                    OptionRev::Some(n) => n,
                    OptionRev::None => 0,
                };

                num
            }
        );
        let ctx = NativeContext::new();
        let module = ctx.compile(&program, false, None).unwrap();
        let executor = JitNativeExecutor::from_native_module(module, OptLevel::Default).unwrap();
        let ExecutionResult {
            remaining_gas: _,
            return_value,
            builtin_stats: _,
        } = executor
            .invoke_dynamic(&program.funcs[0].id, &[], None)
            .unwrap();

        let Value::BoundedInt { value, range: _ } = return_value else {
            panic!();
        };
        assert_eq!(value, Felt252::from(0));
    }
}
