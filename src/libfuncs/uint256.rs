//! # `u256`-related libfuncs

use super::{BlockExt, LibfuncHelper};
use crate::{error::Result, metadata::MetadataStorage, utils::ProgramRegistryExt};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        int::unsigned256::Uint256Concrete,
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        llvm, ods, scf,
    },
    ir::{
        attribute::{DenseI64ArrayAttribute, IntegerAttribute},
        operation::OperationBuilder,
        r#type::IntegerType,
        Block, Location, Region, Value,
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
    selector: &Uint256Concrete,
) -> Result<()> {
    match selector {
        Uint256Concrete::Divmod(info) => {
            build_divmod(context, registry, entry, location, helper, metadata, info)
        }
        Uint256Concrete::IsZero(info) => {
            build_is_zero(context, registry, entry, location, helper, metadata, info)
        }
        Uint256Concrete::SquareRoot(info) => {
            build_square_root(context, registry, entry, location, helper, metadata, info)
        }
        Uint256Concrete::InvModN(info) => build_u256_guarantee_inv_mod_n(
            context, registry, entry, location, helper, metadata, info,
        ),
    }
}

/// Generate MLIR operations for the `u256_safe_divmod` libfunc.
pub fn build_divmod<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let i128_ty = IntegerType::new(context, 128).into();
    let i256_ty = IntegerType::new(context, 256).into();

    let guarantee_type =
        registry.build_type(context, helper, metadata, &info.output_types()[0][3])?;

    let lhs_struct: Value = entry.arg(1)?;
    let rhs_struct: Value = entry.arg(2)?;

    let lhs_lo = entry
        .append_operation(llvm::extract_value(
            context,
            lhs_struct,
            DenseI64ArrayAttribute::new(context, &[0]),
            i128_ty,
            location,
        ))
        .result(0)?
        .into();
    let lhs_hi = entry
        .append_operation(llvm::extract_value(
            context,
            lhs_struct,
            DenseI64ArrayAttribute::new(context, &[1]),
            i128_ty,
            location,
        ))
        .result(0)?
        .into();
    let rhs_lo = entry
        .append_operation(llvm::extract_value(
            context,
            rhs_struct,
            DenseI64ArrayAttribute::new(context, &[0]),
            i128_ty,
            location,
        ))
        .result(0)?
        .into();
    let rhs_hi = entry
        .append_operation(llvm::extract_value(
            context,
            rhs_struct,
            DenseI64ArrayAttribute::new(context, &[1]),
            i128_ty,
            location,
        ))
        .result(0)?
        .into();

    let lhs_lo = entry
        .append_operation(arith::extui(lhs_lo, i256_ty, location))
        .result(0)?
        .into();
    let lhs_hi = entry
        .append_operation(arith::extui(lhs_hi, i256_ty, location))
        .result(0)?
        .into();
    let rhs_lo = entry
        .append_operation(arith::extui(rhs_lo, i256_ty, location))
        .result(0)?
        .into();
    let rhs_hi = entry
        .append_operation(arith::extui(rhs_hi, i256_ty, location))
        .result(0)?
        .into();

    let k128 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(i256_ty, 128).into(),
            location,
        ))
        .result(0)?
        .into();
    let lhs_hi = entry
        .append_operation(arith::shli(lhs_hi, k128, location))
        .result(0)?
        .into();
    let rhs_hi = entry
        .append_operation(arith::shli(rhs_hi, k128, location))
        .result(0)?
        .into();

    let lhs = entry
        .append_operation(arith::ori(lhs_hi, lhs_lo, location))
        .result(0)?
        .into();
    let rhs = entry
        .append_operation(arith::ori(rhs_hi, rhs_lo, location))
        .result(0)?
        .into();

    let result_div = entry
        .append_operation(arith::divui(lhs, rhs, location))
        .result(0)?
        .into();
    let result_rem = entry
        .append_operation(arith::remui(lhs, rhs, location))
        .result(0)?
        .into();

    let result_div_lo = entry
        .append_operation(arith::trunci(result_div, i128_ty, location))
        .result(0)?
        .into();
    let result_div_hi = entry
        .append_operation(arith::shrui(result_div, k128, location))
        .result(0)?
        .into();
    let result_div_hi = entry
        .append_operation(arith::trunci(result_div_hi, i128_ty, location))
        .result(0)?
        .into();

    let result_rem_lo = entry
        .append_operation(arith::trunci(result_rem, i128_ty, location))
        .result(0)?
        .into();
    let result_rem_hi = entry
        .append_operation(arith::shrui(result_rem, k128, location))
        .result(0)?
        .into();
    let result_rem_hi = entry
        .append_operation(arith::trunci(result_rem_hi, i128_ty, location))
        .result(0)?
        .into();

    let result_div = entry
        .append_operation(llvm::undef(
            llvm::r#type::r#struct(context, &[i128_ty, i128_ty], false),
            location,
        ))
        .result(0)?
        .into();
    let result_div = entry
        .append_operation(llvm::insert_value(
            context,
            result_div,
            DenseI64ArrayAttribute::new(context, &[0]),
            result_div_lo,
            location,
        ))
        .result(0)?
        .into();
    let result_div = entry
        .append_operation(llvm::insert_value(
            context,
            result_div,
            DenseI64ArrayAttribute::new(context, &[1]),
            result_div_hi,
            location,
        ))
        .result(0)?
        .into();

    let result_rem = entry
        .append_operation(llvm::undef(
            llvm::r#type::r#struct(context, &[i128_ty, i128_ty], false),
            location,
        ))
        .result(0)?
        .into();
    let result_rem = entry
        .append_operation(llvm::insert_value(
            context,
            result_rem,
            DenseI64ArrayAttribute::new(context, &[0]),
            result_rem_lo,
            location,
        ))
        .result(0)?
        .into();
    let result_rem = entry
        .append_operation(llvm::insert_value(
            context,
            result_rem,
            DenseI64ArrayAttribute::new(context, &[1]),
            result_rem_hi,
            location,
        ))
        .result(0)?
        .into();

    let op = entry.append_operation(llvm::undef(guarantee_type, location));
    let guarantee = op.result(0)?.into();

    entry.append_operation(helper.br(
        0,
        &[range_check, result_div, result_rem, guarantee],
        location,
    ));
    Ok(())
}

/// Generate MLIR operations for the `u256_is_zero` libfunc.
pub fn build_is_zero<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let i128_ty = IntegerType::new(context, 128).into();

    let val_struct = entry.arg(0)?;
    let val_lo = entry
        .append_operation(llvm::extract_value(
            context,
            val_struct,
            DenseI64ArrayAttribute::new(context, &[0]),
            i128_ty,
            location,
        ))
        .result(0)?
        .into();
    let val_hi = entry
        .append_operation(llvm::extract_value(
            context,
            val_struct,
            DenseI64ArrayAttribute::new(context, &[1]),
            i128_ty,
            location,
        ))
        .result(0)?
        .into();

    let k0 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(i128_ty, 0).into(),
            location,
        ))
        .result(0)?
        .into();
    let val_lo_is_zero = entry
        .append_operation(arith::cmpi(
            context,
            CmpiPredicate::Eq,
            val_lo,
            k0,
            location,
        ))
        .result(0)?
        .into();
    let val_hi_is_zero = entry
        .append_operation(arith::cmpi(
            context,
            CmpiPredicate::Eq,
            val_hi,
            k0,
            location,
        ))
        .result(0)?
        .into();

    let val_is_zero = entry
        .append_operation(arith::andi(val_lo_is_zero, val_hi_is_zero, location))
        .result(0)?
        .into();

    entry.append_operation(helper.cond_br(
        context,
        val_is_zero,
        [0, 1],
        [&[], &[val_struct]],
        location,
    ));
    Ok(())
}

/// Generate MLIR operations for the `u256_sqrt` libfunc.
pub fn build_square_root<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let i128_ty = IntegerType::new(context, 128).into();
    let i256_ty = IntegerType::new(context, 256).into();

    let arg_struct = entry.arg(1)?;
    let arg_lo = entry
        .append_operation(llvm::extract_value(
            context,
            arg_struct,
            DenseI64ArrayAttribute::new(context, &[0]),
            i128_ty,
            location,
        ))
        .result(0)?
        .into();
    let arg_hi = entry
        .append_operation(llvm::extract_value(
            context,
            arg_struct,
            DenseI64ArrayAttribute::new(context, &[1]),
            i128_ty,
            location,
        ))
        .result(0)?
        .into();

    let arg_lo = entry
        .append_operation(arith::extui(arg_lo, i256_ty, location))
        .result(0)?
        .into();
    let arg_hi = entry
        .append_operation(arith::extui(arg_hi, i256_ty, location))
        .result(0)?
        .into();

    let k128 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(i256_ty, 128).into(),
            location,
        ))
        .result(0)?
        .into();
    let arg_hi = entry
        .append_operation(arith::shli(arg_hi, k128, location))
        .result(0)?
        .into();

    let arg_value = entry
        .append_operation(arith::ori(arg_hi, arg_lo, location))
        .result(0)?
        .into();

    let k1 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(i256_ty, 1).into(),
            location,
        ))
        .result(0)?
        .into();

    let is_small = entry
        .append_operation(arith::cmpi(
            context,
            CmpiPredicate::Ule,
            arg_value,
            k1,
            location,
        ))
        .result(0)?
        .into();

    let result = entry
        .append_operation(scf::r#if(
            is_small,
            &[i256_ty],
            {
                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                block.append_operation(scf::r#yield(&[arg_value], location));

                region
            },
            {
                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                let k128 = entry
                    .append_operation(arith::constant(
                        context,
                        IntegerAttribute::new(i256_ty, 256).into(),
                        location,
                    ))
                    .result(0)?
                    .into();

                let leading_zeros = block
                    .append_operation(
                        ods::llvm::intr_ctlz(
                            context,
                            i256_ty,
                            arg_value,
                            IntegerAttribute::new(IntegerType::new(context, 1).into(), 1),
                            location,
                        )
                        .into(),
                    )
                    .result(0)?
                    .into();

                let num_bits = block
                    .append_operation(arith::subi(k128, leading_zeros, location))
                    .result(0)?
                    .into();

                let shift_amount = block
                    .append_operation(arith::addi(num_bits, k1, location))
                    .result(0)?
                    .into();

                let parity_mask = block
                    .append_operation(arith::constant(
                        context,
                        IntegerAttribute::new(i256_ty, -2).into(),
                        location,
                    ))
                    .result(0)?
                    .into();
                let shift_amount = block
                    .append_operation(arith::andi(shift_amount, parity_mask, location))
                    .result(0)?
                    .into();

                let k0 = block
                    .append_operation(arith::constant(
                        context,
                        IntegerAttribute::new(i256_ty, 0).into(),
                        location,
                    ))
                    .result(0)?
                    .into();
                let result = block
                    .append_operation(scf::r#while(
                        &[k0, shift_amount],
                        &[i256_ty, i256_ty],
                        {
                            let region = Region::new();
                            let block = region.append_block(Block::new(&[
                                (i256_ty, location),
                                (i256_ty, location),
                            ]));

                            let result = block
                                .append_operation(arith::shli(block.arg(0)?, k1, location))
                                .result(0)?
                                .into();
                            let large_candidate = block
                                .append_operation(arith::xori(result, k1, location))
                                .result(0)?
                                .into();

                            let large_candidate_squared = block
                                .append_operation(arith::muli(
                                    large_candidate,
                                    large_candidate,
                                    location,
                                ))
                                .result(0)?
                                .into();

                            let threshold = block
                                .append_operation(arith::shrui(arg_value, block.arg(1)?, location))
                                .result(0)?
                                .into();
                            let threshold_is_poison = block
                                .append_operation(arith::cmpi(
                                    context,
                                    CmpiPredicate::Eq,
                                    block.arg(1)?,
                                    k128,
                                    location,
                                ))
                                .result(0)?
                                .into();
                            let threshold = block
                                .append_operation(
                                    OperationBuilder::new("arith.select", location)
                                        .add_operands(&[threshold_is_poison, k0, threshold])
                                        .add_results(&[i256_ty])
                                        .build()?,
                                )
                                .result(0)?
                                .into();

                            let is_in_range = block
                                .append_operation(arith::cmpi(
                                    context,
                                    CmpiPredicate::Ule,
                                    large_candidate_squared,
                                    threshold,
                                    location,
                                ))
                                .result(0)?
                                .into();

                            let result = block
                                .append_operation(
                                    OperationBuilder::new("arith.select", location)
                                        .add_operands(&[is_in_range, large_candidate, result])
                                        .add_results(&[i256_ty])
                                        .build()?,
                                )
                                .result(0)?
                                .into();

                            let k2 = block
                                .append_operation(arith::constant(
                                    context,
                                    IntegerAttribute::new(i256_ty, 2).into(),
                                    location,
                                ))
                                .result(0)?
                                .into();

                            let shift_amount = block
                                .append_operation(arith::subi(block.arg(1)?, k2, location))
                                .result(0)?
                                .into();

                            let should_continue = block
                                .append_operation(arith::cmpi(
                                    context,
                                    CmpiPredicate::Sge,
                                    shift_amount,
                                    k0,
                                    location,
                                ))
                                .result(0)?
                                .into();
                            block.append_operation(scf::condition(
                                should_continue,
                                &[result, shift_amount],
                                location,
                            ));

                            region
                        },
                        {
                            let region = Region::new();
                            let block = region.append_block(Block::new(&[
                                (i256_ty, location),
                                (i256_ty, location),
                            ]));

                            block.append_operation(scf::r#yield(
                                &[block.arg(0)?, block.argument(1)?.into()],
                                location,
                            ));

                            region
                        },
                        location,
                    ))
                    .result(0)?
                    .into();

                block.append_operation(scf::r#yield(&[result], location));

                region
            },
            location,
        ))
        .result(0)?
        .into();

    let result = entry
        .append_operation(arith::trunci(result, i128_ty, location))
        .result(0)?
        .into();

    entry.append_operation(helper.br(0, &[range_check, result], location));
    Ok(())
}

/// Generate MLIR operations for the `u256_guarantee_inv_mod_n` libfunc.
pub fn build_u256_guarantee_inv_mod_n<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let i128_ty = IntegerType::new(context, 128).into();
    let i256_ty = IntegerType::new(context, 256).into();

    let lhs_struct = entry.arg(1)?;
    let lhs_lo = entry
        .append_operation(llvm::extract_value(
            context,
            lhs_struct,
            DenseI64ArrayAttribute::new(context, &[0]),
            i128_ty,
            location,
        ))
        .result(0)?
        .into();
    let lhs_hi = entry
        .append_operation(llvm::extract_value(
            context,
            lhs_struct,
            DenseI64ArrayAttribute::new(context, &[1]),
            i128_ty,
            location,
        ))
        .result(0)?
        .into();

    let rhs_struct = entry.arg(2)?;
    let rhs_lo = entry
        .append_operation(llvm::extract_value(
            context,
            rhs_struct,
            DenseI64ArrayAttribute::new(context, &[0]),
            i128_ty,
            location,
        ))
        .result(0)?
        .into();
    let rhs_hi = entry
        .append_operation(llvm::extract_value(
            context,
            rhs_struct,
            DenseI64ArrayAttribute::new(context, &[1]),
            i128_ty,
            location,
        ))
        .result(0)?
        .into();

    let lhs_lo = entry
        .append_operation(arith::extui(lhs_lo, i256_ty, location))
        .result(0)?
        .into();
    let lhs_hi = entry
        .append_operation(arith::extui(lhs_hi, i256_ty, location))
        .result(0)?
        .into();

    let rhs_lo = entry
        .append_operation(arith::extui(rhs_lo, i256_ty, location))
        .result(0)?
        .into();
    let rhs_hi = entry
        .append_operation(arith::extui(rhs_hi, i256_ty, location))
        .result(0)?
        .into();

    let k128 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(i256_ty, 128).into(),
            location,
        ))
        .result(0)?
        .into();
    let lhs_hi = entry
        .append_operation(arith::shli(lhs_hi, k128, location))
        .result(0)?
        .into();
    let rhs_hi = entry
        .append_operation(arith::shli(rhs_hi, k128, location))
        .result(0)?
        .into();

    let lhs = entry
        .append_operation(arith::ori(lhs_hi, lhs_lo, location))
        .result(0)?
        .into();
    let rhs = entry
        .append_operation(arith::ori(rhs_hi, rhs_lo, location))
        .result(0)?
        .into();

    let k0 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(i256_ty, 0).into(),
            location,
        ))
        .result(0)?
        .into();
    let k1 = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(i256_ty, 1).into(),
            location,
        ))
        .result(0)?
        .into();

    let result = entry.append_operation(scf::r#while(
        &[lhs, rhs, k1, k0],
        &[i256_ty, i256_ty, i256_ty, i256_ty],
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[
                (i256_ty, location),
                (i256_ty, location),
                (i256_ty, location),
                (i256_ty, location),
            ]));

            let q = block
                .append_operation(arith::divui(block.arg(1)?, block.arg(0)?, location))
                .result(0)?
                .into();

            let q_c = block
                .append_operation(arith::muli(q, block.arg(0)?, location))
                .result(0)?
                .into();
            let c = block
                .append_operation(arith::subi(block.arg(1)?, q_c, location))
                .result(0)?
                .into();

            let q_uc = block
                .append_operation(arith::muli(q, block.arg(2)?, location))
                .result(0)?
                .into();
            let u_c = block
                .append_operation(arith::subi(block.arg(3)?, q_uc, location))
                .result(0)?
                .into();

            let should_continue = block
                .append_operation(arith::cmpi(context, CmpiPredicate::Ne, c, k0, location))
                .result(0)?
                .into();
            block.append_operation(scf::condition(
                should_continue,
                &[c, block.arg(0)?, u_c, block.argument(2)?.into()],
                location,
            ));

            region
        },
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[
                (i256_ty, location),
                (i256_ty, location),
                (i256_ty, location),
                (i256_ty, location),
            ]));

            block.append_operation(scf::r#yield(
                &[block.arg(0)?, block.arg(1)?, block.arg(2)?, block.arg(3)?],
                location,
            ));

            region
        },
        location,
    ));

    let inv = result.result(3)?.into();

    let inv = entry
        .append_operation(scf::r#if(
            entry
                .append_operation(arith::cmpi(context, CmpiPredicate::Slt, inv, k0, location))
                .result(0)?
                .into(),
            &[i256_ty],
            {
                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                block.append_operation(scf::r#yield(
                    &[entry
                        .append_operation(arith::addi(inv, rhs, location))
                        .result(0)?
                        .into()],
                    location,
                ));

                region
            },
            {
                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                block.append_operation(scf::r#yield(
                    &[entry
                        .append_operation(arith::remui(inv, rhs, location))
                        .result(0)?
                        .into()],
                    location,
                ));

                region
            },
            location,
        ))
        .result(0)?
        .into();

    let inv_lo = entry
        .append_operation(arith::trunci(inv, i128_ty, location))
        .result(0)?
        .into();
    let inv_hi = entry
        .append_operation(arith::shrui(inv, k128, location))
        .result(0)?
        .into();
    let inv_hi = entry
        .append_operation(arith::trunci(inv_hi, i128_ty, location))
        .result(0)?
        .into();

    let return_ty = registry.build_type(context, helper, metadata, &info.output_types()[0][1])?;
    let result_inv = entry
        .append_operation(llvm::undef(return_ty, location))
        .result(0)?
        .into();
    let result_inv = entry
        .append_operation(llvm::insert_value(
            context,
            result_inv,
            DenseI64ArrayAttribute::new(context, &[0]),
            inv_lo,
            location,
        ))
        .result(0)?
        .into();
    let result_inv = entry
        .append_operation(llvm::insert_value(
            context,
            result_inv,
            DenseI64ArrayAttribute::new(context, &[1]),
            inv_hi,
            location,
        ))
        .result(0)?
        .into();

    let lhs_is_invertible = entry
        .append_operation(arith::cmpi(
            context,
            CmpiPredicate::Eq,
            result.result(1)?.into(),
            k1,
            location,
        ))
        .result(0)?
        .into();
    let inv_not_zero = entry
        .append_operation(arith::cmpi(context, CmpiPredicate::Ne, inv, k0, location))
        .result(0)?
        .into();
    let condition = entry
        .append_operation(arith::andi(lhs_is_invertible, inv_not_zero, location))
        .result(0)?
        .into();

    let guarantee_type =
        registry.build_type(context, helper, metadata, &info.output_types()[0][2])?;
    let op = entry.append_operation(llvm::undef(guarantee_type, location));
    let guarantee = op.result(0)?.into();

    entry.append_operation(helper.cond_br(
        context,
        condition,
        [0, 1],
        [
            &[
                entry.arg(0)?,
                result_inv,
                guarantee,
                guarantee,
                guarantee,
                guarantee,
                guarantee,
                guarantee,
                guarantee,
                guarantee,
            ],
            &[entry.arg(0)?, guarantee, guarantee],
        ],
        location,
    ));

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{
        utils::test::{jit_enum, jit_panic, jit_struct, run_sierra_program},
        values::Value,
    };
    use cairo_lang_sierra::{program::Program, ProgramParser};
    use lazy_static::lazy_static;
    use num_bigint::BigUint;
    use num_traits::One;
    use starknet_types_core::felt::Felt;
    use std::ops::Shl;

    lazy_static! {
        // use zeroable::IsZeroResult;
        // extern fn u256_is_zero(a: u256) -> IsZeroResult<u256> implicits() nopanic;
        // fn run_test(value: u256) -> bool {
        //     match u256_is_zero(value) {
        //         IsZeroResult::Zero(_) => true,
        //         IsZeroResult::NonZero(_) => false,
        //     }
        // }
        static ref U256_IS_ZERO: Program = ProgramParser::new().parse(r#"
            type [0] = u128 [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = Struct<ut@core::integer::u256, [0], [0]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [3] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
            type [4] = Enum<ut@core::bool, [3], [3]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [2] = NonZero<[1]> [storable: true, drop: true, dup: true, zero_sized: false];

            libfunc [3] = u256_is_zero;
            libfunc [4] = branch_align;
            libfunc [1] = struct_construct<[3]>;
            libfunc [2] = enum_init<[4], 1>;
            libfunc [6] = store_temp<[4]>;
            libfunc [5] = drop<[2]>;
            libfunc [0] = enum_init<[4], 0>;

            [3]([0]) { fallthrough() 6([1]) }; // 0
            [4]() -> (); // 1
            [1]() -> ([2]); // 2
            [2]([2]) -> ([3]); // 3
            [6]([3]) -> ([3]); // 4
            return([3]); // 5
            [4]() -> (); // 6
            [5]([1]) -> (); // 7
            [1]() -> ([4]); // 8
            [0]([4]) -> ([5]); // 9
            [6]([5]) -> ([5]); // 10
            return([5]); // 11

            [0]@0([0]: [1]) -> ([4]);
        "#).map_err(|e| e.to_string()).unwrap();
        // fn run_test(lhs: u256, rhs: u256) -> (u256, u256) {
        //     let q = lhs / rhs;
        //     let r = lhs % rhs;
        //     (q, r)
        // }
        static ref U256_SAFE_DIVMOD: Program = ProgramParser::new().parse(r#"
            type [1] = u128 [storable: true, drop: true, dup: true, zero_sized: false];
            type [2] = Struct<ut@core::integer::u256, [1], [1]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [8] = Struct<ut@Tuple, [2], [2]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [9] = Struct<ut@Tuple, [8]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [11] = U128MulGuarantee [storable: true, drop: false, dup: false, zero_sized: false];
            type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
            type [6] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
            type [5] = Array<[4]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [7] = Struct<ut@Tuple, [6], [5]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [10] = Enum<ut@core::panics::PanicResult::<((core::integer::u256, core::integer::u256),)>, [9], [7]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [12] = Const<[4], 5420154128225384396790819266608> [storable: false, drop: false, dup: false, zero_sized: false];
            type [4] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [3] = NonZero<[2]> [storable: true, drop: true, dup: true, zero_sized: false];

            libfunc [11] = dup<[2]>;
            libfunc [10] = u256_is_zero;
            libfunc [12] = branch_align;
            libfunc [13] = drop<[2]>;
            libfunc [9] = array_new<[4]>;
            libfunc [14] = const_as_immediate<[12]>;
            libfunc [15] = store_temp<[4]>;
            libfunc [8] = array_append<[4]>;
            libfunc [7] = struct_construct<[6]>;
            libfunc [6] = struct_construct<[7]>;
            libfunc [5] = enum_init<[10], 1>;
            libfunc [16] = store_temp<[0]>;
            libfunc [17] = store_temp<[10]>;
            libfunc [4] = u256_safe_divmod;
            libfunc [3] = u128_mul_guarantee_verify;
            libfunc [2] = struct_construct<[8]>;
            libfunc [1] = struct_construct<[9]>;
            libfunc [0] = enum_init<[10], 0>;

            [11]([2]) -> ([2], [3]); // 0
            [10]([3]) { fallthrough() 15([4]) }; // 1
            [12]() -> (); // 2
            [13]([2]) -> (); // 3
            [13]([1]) -> (); // 4
            [9]() -> ([5]); // 5
            [14]() -> ([6]); // 6
            [15]([6]) -> ([6]); // 7
            [8]([5], [6]) -> ([7]); // 8
            [7]() -> ([8]); // 9
            [6]([8], [7]) -> ([9]); // 10
            [5]([9]) -> ([10]); // 11
            [16]([0]) -> ([0]); // 12
            [17]([10]) -> ([10]); // 13
            return([0], [10]); // 14
            [12]() -> (); // 15
            [11]([1]) -> ([1], [11]); // 16
            [4]([0], [11], [4]) -> ([12], [13], [14], [15]); // 17
            [13]([14]) -> (); // 18
            [3]([12], [15]) -> ([16]); // 19
            [16]([16]) -> ([16]); // 20
            [10]([2]) { fallthrough() 35([17]) }; // 21
            [12]() -> (); // 22
            [13]([13]) -> (); // 23
            [13]([1]) -> (); // 24
            [9]() -> ([18]); // 25
            [14]() -> ([19]); // 26
            [15]([19]) -> ([19]); // 27
            [8]([18], [19]) -> ([20]); // 28
            [7]() -> ([21]); // 29
            [6]([21], [20]) -> ([22]); // 30
            [5]([22]) -> ([23]); // 31
            [16]([16]) -> ([16]); // 32
            [17]([23]) -> ([23]); // 33
            return([16], [23]); // 34
            [12]() -> (); // 35
            [4]([16], [1], [17]) -> ([24], [25], [26], [27]); // 36
            [13]([25]) -> (); // 37
            [3]([24], [27]) -> ([28]); // 38
            [2]([13], [26]) -> ([29]); // 39
            [1]([29]) -> ([30]); // 40
            [0]([30]) -> ([31]); // 41
            [16]([28]) -> ([28]); // 42
            [17]([31]) -> ([31]); // 43
            return([28], [31]); // 44

            [0]@0([0]: [0], [1]: [2], [2]: [2]) -> ([0], [10]);
        "#).map_err(|e| e.to_string()).unwrap();
        // use core::num::traits::Sqrt;
        // fn run_test(value: u256) -> u128 {
        //     value.sqrt()
        // }
        static ref U256_SQRT: Program = ProgramParser::new().parse(r#"
            type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
            type [1] = u128 [storable: true, drop: true, dup: true, zero_sized: false];
            type [2] = Struct<ut@core::integer::u256, [1], [1]> [storable: true, drop: true, dup: true, zero_sized: false];

            libfunc [0] = u256_sqrt;
            libfunc [2] = store_temp<[0]>;
            libfunc [3] = store_temp<[1]>;

            [0]([0], [1]) -> ([2], [3]); // 0
            [2]([2]) -> ([2]); // 1
            [3]([3]) -> ([3]); // 2
            return([2], [3]); // 3

            [0]@0([0]: [0], [1]: [2]) -> ([0], [1]);
        "#).map_err(|e| e.to_string()).unwrap();
        // use core::math::u256_inv_mod;
        // fn run_test(a: u256, n: NonZero<u256>) -> Option<NonZero<u256>> {
        //     u256_inv_mod(a, n)
        // }
        static ref U256_INV_MOD_N: Program = ProgramParser::new().parse(r#"
            type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
            type [5] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
            type [1] = u128 [storable: true, drop: true, dup: true, zero_sized: false];
            type [2] = Struct<ut@core::integer::u256, [1], [1]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [3] = NonZero<[2]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [6] = Enum<ut@core::option::Option::<core::zeroable::NonZero::<core::integer::u256>>, [3], [5]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [4] = U128MulGuarantee [storable: true, drop: false, dup: false, zero_sized: false];

            libfunc [4] = u256_guarantee_inv_mod_n;
            libfunc [5] = branch_align;
            libfunc [2] = u128_mul_guarantee_verify;
            libfunc [3] = enum_init<[6], 0>;
            libfunc [6] = store_temp<[0]>;
            libfunc [7] = store_temp<[6]>;
            libfunc [1] = struct_construct<[5]>;
            libfunc [0] = enum_init<[6], 1>;

            [4]([0], [1], [2]) { fallthrough([3], [4], [5], [6], [7], [8], [9], [10], [11], [12]) 14([13], [14], [15]) }; // 0
            [5]() -> (); // 1
            [2]([3], [12]) -> ([16]); // 2
            [2]([16], [11]) -> ([17]); // 3
            [2]([17], [10]) -> ([18]); // 4
            [2]([18], [9]) -> ([19]); // 5
            [2]([19], [8]) -> ([20]); // 6
            [2]([20], [7]) -> ([21]); // 7
            [2]([21], [6]) -> ([22]); // 8
            [2]([22], [5]) -> ([23]); // 9
            [3]([4]) -> ([24]); // 10
            [6]([23]) -> ([23]); // 11
            [7]([24]) -> ([24]); // 12
            return([23], [24]); // 13
            [5]() -> (); // 14
            [2]([13], [15]) -> ([25]); // 15
            [2]([25], [14]) -> ([26]); // 16
            [1]() -> ([27]); // 17
            [0]([27]) -> ([28]); // 18
            [6]([26]) -> ([26]); // 19
            [7]([28]) -> ([28]); // 20
            return([26], [28]); // 21

            [0]@0([0]: [0], [1]: [2], [2]: [3]) -> ([0], [6]);
        "#).map_err(|e| e.to_string()).unwrap();
    }

    fn u256(value: BigUint) -> Value {
        assert!(value.bits() <= 256);
        jit_struct!(
            Value::Uint128((&value & &u128::MAX.into()).try_into().unwrap()),
            Value::Uint128(((&value >> 128u32) & &u128::MAX.into()).try_into().unwrap()),
        )
    }

    #[test]
    fn u256_is_zero() {
        let return_value = run_sierra_program(&U256_IS_ZERO, &[u256(0u32.into())]).return_value;

        assert_eq!(jit_enum!(1, jit_struct!()), return_value);

        let return_value = run_sierra_program(&U256_IS_ZERO, &[u256(1u32.into())]).return_value;

        assert_eq!(jit_enum!(0, jit_struct!()), return_value);

        let return_value =
            run_sierra_program(&U256_IS_ZERO, &[u256(BigUint::one() << 128u32)]).return_value;

        assert_eq!(jit_enum!(0, jit_struct!()), return_value);

        let return_value =
            run_sierra_program(&U256_IS_ZERO, &[u256((BigUint::one() << 128u32) + 1u32)])
                .return_value;

        assert_eq!(jit_enum!(0, jit_struct!()), return_value);
    }

    #[test]
    fn u256_safe_divmod() {
        #[track_caller]
        fn run(lhs: (u128, u128), rhs: (u128, u128), result: Value) {
            let return_value = run_sierra_program(
                &U256_SAFE_DIVMOD,
                &[
                    jit_struct!(lhs.1.into(), lhs.0.into()),
                    jit_struct!(rhs.1.into(), rhs.0.into()),
                ],
            )
            .return_value;

            assert_eq!(result, return_value);
        }

        let u256_is_zero = Felt::from_bytes_be_slice(b"Division by 0");
        let max_value = 0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFFu128;

        run((0, 0), (0, 0), jit_panic!(u256_is_zero));
        run(
            (0, 0),
            (0, 1),
            jit_enum!(
                0,
                jit_struct!(jit_struct!(
                    jit_struct!(0u128.into(), 0u128.into()),
                    jit_struct!(0u128.into(), 0u128.into()),
                ))
            ),
        );
        run(
            (0, 0),
            (max_value, max_value),
            jit_enum!(
                0,
                jit_struct!(jit_struct!(
                    jit_struct!(0u128.into(), 0u128.into()),
                    jit_struct!(0u128.into(), 0u128.into()),
                ))
            ),
        );

        run((0, 1), (0, 0), jit_panic!(u256_is_zero));
        run(
            (0, 1),
            (0, 1),
            jit_enum!(
                0,
                jit_struct!(jit_struct!(
                    jit_struct!(1u128.into(), 0u128.into()),
                    jit_struct!(0u128.into(), 0u128.into()),
                ))
            ),
        );
        run(
            (0, 1),
            (max_value, max_value),
            jit_enum!(
                0,
                jit_struct!(jit_struct!(
                    jit_struct!(0u128.into(), 0u128.into()),
                    jit_struct!(1u128.into(), 0u128.into()),
                ))
            ),
        );
        run((max_value, max_value), (0, 0), jit_panic!(u256_is_zero));

        run(
            (max_value, max_value),
            (0, 1),
            jit_enum!(
                0,
                jit_struct!(jit_struct!(
                    jit_struct!(max_value.into(), max_value.into()),
                    jit_struct!(0u128.into(), 0u128.into()),
                ))
            ),
        );
        run(
            (max_value, max_value),
            (max_value, max_value),
            jit_enum!(
                0,
                jit_struct!(jit_struct!(
                    jit_struct!(1u128.into(), 0u128.into()),
                    jit_struct!(0u128.into(), 0u128.into()),
                ))
            ),
        );
    }

    #[test]
    fn u256_sqrt() {
        #[track_caller]
        fn run(value: (u128, u128), result: Value) {
            let return_value =
                run_sierra_program(&U256_SQRT, &[jit_struct!(value.1.into(), value.0.into())])
                    .return_value;

            assert_eq!(result, return_value);
        }

        run((0u128, 0u128), 0u128.into());
        run((0u128, 1u128), 1u128.into());
        run((u128::MAX, u128::MAX), u128::MAX.into());

        for i in 0..u128::BITS {
            let x = 1u128 << i;
            let y: u128 = BigUint::from(x)
                .sqrt()
                .try_into()
                .expect("should always fit into a u128");

            run((0, x), y.into());
        }

        for i in 0..u128::BITS {
            let x = 1u128 << i;
            let y: u128 = BigUint::from(x)
                .shl(128usize)
                .sqrt()
                .try_into()
                .expect("should always fit into a u128");

            run((x, 0), y.into());
        }
    }

    #[test]
    fn u256_inv_mod_n() {
        #[track_caller]
        fn run(a: (u128, u128), n: (u128, u128), result: Value) {
            let return_value = run_sierra_program(
                &U256_INV_MOD_N,
                &[
                    jit_struct!(a.0.into(), a.1.into()),
                    jit_struct!(n.0.into(), n.1.into()),
                ],
            )
            .return_value;

            assert_eq!(result, return_value);
        }

        let none = jit_enum!(1, jit_struct!());

        // Not invertible.
        run((0, 0), (0, 0), none.clone());
        run((1, 0), (1, 0), none.clone());
        run((0, 0), (1, 0), none.clone());
        run((0, 0), (7, 0), none.clone());
        run((3, 0), (6, 0), none.clone());
        run((4, 0), (6, 0), none.clone());
        run((8, 0), (4, 0), none.clone());
        run((8, 0), (24, 0), none.clone());
        run(
            (
                112713230461650448610759614893138283713,
                311795268193434200766998031144865279193,
            ),
            (
                214442144331145623175443765631916854552,
                85683151001472364977354294776284843870,
            ),
            none.clone(),
        );
        run(
            (
                138560372230216185616572678448146427468,
                178030013799389090502578959553486954963,
            ),
            (
                299456334380503763038201670272353657683,
                285941620966047830312853638602560712796,
            ),
            none,
        );

        // Invertible.
        run(
            (5, 0),
            (24, 0),
            jit_enum!(0, jit_struct!(5u128.into(), 0u128.into())),
        );
        run(
            (29, 0),
            (24, 0),
            jit_enum!(0, jit_struct!(5u128.into(), 0u128.into())),
        );
        run(
            (1, 0),
            (24, 0),
            jit_enum!(0, jit_struct!(1u128.into(), 0u128.into())),
        );
        run(
            (1, 0),
            (5, 0),
            jit_enum!(0, jit_struct!(1u128.into(), 0u128.into())),
        );
        run(
            (2, 0),
            (5, 0),
            jit_enum!(0, jit_struct!(3u128.into(), 0u128.into())),
        );
    }
}
