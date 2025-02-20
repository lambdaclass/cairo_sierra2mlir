//! # Casting libfuncs

use super::LibfuncHelper;
use crate::{
    error::Result,
    metadata::MetadataStorage,
    native_panic,
    types::TypeBuilder,
    utils::{BlockExt, RangeExt, HALF_PRIME, PRIME},
};
use cairo_lang_sierra::{
    extensions::{
        casts::{CastConcreteLibfunc, DowncastConcreteLibfunc},
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
        utils::Range,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::arith::{self, CmpiPredicate},
    ir::{r#type::IntegerType, Block, Location, Value, ValueLike},
    Context,
};
use num_bigint::{BigInt, Sign};
use num_traits::One;

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &CastConcreteLibfunc,
) -> Result<()> {
    match selector {
        CastConcreteLibfunc::Downcast(info) => {
            build_downcast(context, registry, entry, location, helper, metadata, info)
        }
        CastConcreteLibfunc::Upcast(info) => {
            build_upcast(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `downcast` libfunc.
pub fn build_downcast<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &DowncastConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;
    let src_value: Value = entry.arg(1)?;

    if info.signature.param_signatures[1].ty == info.signature.branch_signatures[0].vars[1].ty {
        let k0 = entry.const_int(context, location, 0, 1)?;
        return helper.cond_br(
            context,
            entry,
            k0,
            [0, 1],
            [&[range_check, src_value], &[range_check]],
            location,
        );
    }

    let src_ty = registry.get_type(&info.signature.param_signatures[1].ty)?;
    let dst_ty = registry.get_type(&info.signature.branch_signatures[0].vars[1].ty)?;

    let dst_range = dst_ty.integer_range(registry)?;
    let src_range = if src_ty.is_felt252(registry)? && dst_range.lower.sign() == Sign::Minus {
        if dst_range.upper.sign() != Sign::Plus {
            Range {
                lower: BigInt::from_biguint(Sign::Minus, PRIME.clone()) + 1,
                upper: BigInt::one(),
            }
        } else {
            Range {
                lower: BigInt::from_biguint(Sign::Minus, HALF_PRIME.clone()),
                upper: BigInt::from_biguint(Sign::Plus, HALF_PRIME.clone()) + BigInt::one(),
            }
        }
    } else {
        src_ty.integer_range(registry)?
    };

    let src_width = if src_ty.is_bounded_int(registry)? {
        src_range.offset_bit_width()
    } else {
        src_ty.integer_range(registry)?.zero_based_bit_width()
    };
    let dst_width = if dst_ty.is_bounded_int(registry)? {
        dst_range.offset_bit_width()
    } else {
        dst_range.zero_based_bit_width()
    };

    let compute_width = src_range
        .zero_based_bit_width()
        .max(dst_range.zero_based_bit_width());

    let is_signed = src_range.lower.sign() == Sign::Minus;

    let src_value = if compute_width > src_width {
        if is_signed && !src_ty.is_bounded_int(registry)? && !src_ty.is_felt252(registry)? {
            entry.extsi(
                src_value,
                IntegerType::new(context, compute_width).into(),
                location,
            )?
        } else {
            entry.extui(
                src_value,
                IntegerType::new(context, compute_width).into(),
                location,
            )?
        }
    } else {
        src_value
    };

    let src_value = if is_signed && src_ty.is_felt252(registry)? {
        if src_range.upper.is_one() {
            let adj_offset =
                entry.const_int_from_type(context, location, PRIME.clone(), src_value.r#type())?;
            entry.append_op_result(arith::subi(src_value, adj_offset, location))?
        } else {
            let adj_offset = entry.const_int_from_type(
                context,
                location,
                HALF_PRIME.clone(),
                src_value.r#type(),
            )?;
            let is_negative =
                entry.cmpi(context, CmpiPredicate::Ugt, src_value, adj_offset, location)?;

            let k_prime =
                entry.const_int_from_type(context, location, PRIME.clone(), src_value.r#type())?;
            let adj_value = entry.append_op_result(arith::subi(src_value, k_prime, location))?;

            entry.append_op_result(arith::select(is_negative, adj_value, src_value, location))?
        }
    } else if src_ty.is_bounded_int(registry)? && src_range.lower != BigInt::ZERO {
        let dst_offset = entry.const_int_from_type(
            context,
            location,
            src_range.lower.clone(),
            src_value.r#type(),
        )?;
        entry.addi(src_value, dst_offset, location)?
    } else {
        src_value
    };

    if !(dst_range.lower > src_range.lower || dst_range.upper < src_range.upper) {
        let dst_value = if dst_ty.is_bounded_int(registry)? && dst_range.lower != BigInt::ZERO {
            let dst_offset = entry.const_int_from_type(
                context,
                location,
                dst_range.lower,
                src_value.r#type(),
            )?;
            entry.append_op_result(arith::subi(src_value, dst_offset, location))?
        } else {
            src_value
        };

        let dst_value = if dst_width < compute_width {
            entry.trunci(
                dst_value,
                IntegerType::new(context, dst_width).into(),
                location,
            )?
        } else {
            dst_value
        };

        let is_in_bounds = entry.const_int(context, location, 1, 1)?;

        helper.cond_br(
            context,
            entry,
            is_in_bounds,
            [0, 1],
            [&[range_check, dst_value], &[range_check]],
            location,
        )?;
    } else {
        let lower_check = if dst_range.lower > src_range.lower {
            let dst_lower = entry.const_int_from_type(
                context,
                location,
                dst_range.lower.clone(),
                src_value.r#type(),
            )?;
            Some(entry.cmpi(
                context,
                if !is_signed {
                    CmpiPredicate::Uge
                } else {
                    CmpiPredicate::Sge
                },
                src_value,
                dst_lower,
                location,
            )?)
        } else {
            None
        };
        let upper_check = if dst_range.upper < src_range.upper {
            let dst_upper = entry.const_int_from_type(
                context,
                location,
                dst_range.upper.clone(),
                src_value.r#type(),
            )?;
            Some(entry.cmpi(
                context,
                if !is_signed {
                    CmpiPredicate::Ult
                } else {
                    CmpiPredicate::Slt
                },
                src_value,
                dst_upper,
                location,
            )?)
        } else {
            None
        };

        let is_in_bounds = match (lower_check, upper_check) {
            (Some(lower_check), Some(upper_check)) => {
                entry.append_op_result(arith::andi(lower_check, upper_check, location))?
            }
            (Some(lower_check), None) => lower_check,
            (None, Some(upper_check)) => upper_check,
            // its always in bounds since dst is larger than src (i.e no bounds checks needed)
            (None, None) => {
                native_panic!("matched an unreachable: no bounds checks are being performed")
            }
        };

        let dst_value = if dst_ty.is_bounded_int(registry)? && dst_range.lower != BigInt::ZERO {
            let dst_offset = entry.const_int_from_type(
                context,
                location,
                dst_range.lower,
                src_value.r#type(),
            )?;
            entry.append_op_result(arith::subi(src_value, dst_offset, location))?
        } else {
            src_value
        };

        let dst_value = if dst_width < compute_width {
            entry.trunci(
                dst_value,
                IntegerType::new(context, dst_width).into(),
                location,
            )?
        } else {
            dst_value
        };
        helper.cond_br(
            context,
            entry,
            is_in_bounds,
            [0, 1],
            [&[range_check, dst_value], &[range_check]],
            location,
        )?;
    }

    Ok(())
}

/// Generate MLIR operations for the `upcast` libfunc.
pub fn build_upcast<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let src_value = entry.arg(0)?;

    if info.signature.param_signatures[0].ty == info.signature.branch_signatures[0].vars[0].ty {
        return helper.br(entry, 0, &[src_value], location);
    }

    let src_ty = registry.get_type(&info.signature.param_signatures[0].ty)?;
    let dst_ty = registry.get_type(&info.signature.branch_signatures[0].vars[0].ty)?;

    let src_range = src_ty.integer_range(registry)?;
    let dst_range = dst_ty.integer_range(registry)?;
    assert!(
        if dst_ty.is_felt252(registry)? {
            let alt_range = Range {
                lower: BigInt::from_biguint(Sign::Minus, HALF_PRIME.clone()),
                upper: BigInt::from_biguint(Sign::Plus, HALF_PRIME.clone()) + BigInt::one(),
            };

            (dst_range.lower <= src_range.lower && dst_range.upper >= src_range.upper)
                || (alt_range.lower <= src_range.lower && alt_range.upper >= src_range.upper)
        } else {
            dst_range.lower <= src_range.lower && dst_range.upper >= src_range.upper
        },
        "invalid upcast `{:?}` into `{:?}`: target range doesn't contain the source range",
        info.signature.param_signatures[0].ty,
        info.signature.branch_signatures[0].vars[0].ty
    );

    let src_width = if src_ty.is_bounded_int(registry)? {
        src_range.offset_bit_width()
    } else {
        src_range.zero_based_bit_width()
    };
    let dst_width = if dst_ty.is_bounded_int(registry)? {
        dst_range.offset_bit_width()
    } else {
        dst_range.zero_based_bit_width()
    };

    // If the source can be negative, the target type must also contain negatives when upcasting.
    assert!(
        src_range.lower.sign() != Sign::Minus
            || dst_ty.is_felt252(registry)?
            || dst_range.lower.sign() == Sign::Minus
    );
    let is_signed = src_range.lower.sign() == Sign::Minus;

    let dst_value = if dst_width > src_width {
        if is_signed && !src_ty.is_bounded_int(registry)? {
            entry.extsi(
                src_value,
                IntegerType::new(context, dst_width).into(),
                location,
            )?
        } else {
            entry.extui(
                src_value,
                IntegerType::new(context, dst_width).into(),
                location,
            )?
        }
    } else {
        src_value
    };

    let dst_value = if src_ty.is_bounded_int(registry)? && src_range.lower != BigInt::ZERO {
        let dst_offset = entry.const_int_from_type(
            context,
            location,
            if dst_ty.is_bounded_int(registry)? {
                &src_range.lower - &dst_range.lower
            } else {
                src_range.lower.clone()
            },
            dst_value.r#type(),
        )?;
        entry.addi(dst_value, dst_offset, location)?
    } else {
        dst_value
    };

    let dst_value = if dst_ty.is_felt252(registry)? && src_range.lower.sign() == Sign::Minus {
        let k0 = entry.const_int(context, location, 0, 252)?;
        let is_negative = entry.cmpi(context, CmpiPredicate::Slt, dst_value, k0, location)?;

        let k_prime = entry.const_int(context, location, PRIME.clone(), 252)?;
        let adj_value = entry.addi(dst_value, k_prime, location)?;

        entry.append_op_result(arith::select(is_negative, adj_value, dst_value, location))?
    } else {
        dst_value
    };

    helper.br(entry, 0, &[dst_value], location)
}

#[cfg(test)]
mod test {
    use crate::{
        utils::test::{jit_enum, jit_struct, load_cairo, run_program_assert_output},
        values::Value,
    };
    use cairo_lang_sierra::program::Program;
    use lazy_static::lazy_static;

    lazy_static! {
        static ref DOWNCAST: (String, Program) = load_cairo! {
            use core::integer::downcast;

            fn run_test(
                v8: u8, v16: u16, v32: u32, v64: u64, v128: u128
            ) -> (
                (Option<u8>, Option<u8>, Option<u8>, Option<u8>, Option<u8>),
                (Option<u16>, Option<u16>, Option<u16>, Option<u16>),
                (Option<u32>, Option<u32>, Option<u32>),
                (Option<u64>, Option<u64>),
                (Option<u128>,),
            ) {
                (
                    (downcast(v128), downcast(v64), downcast(v32), downcast(v16), downcast(v8)),
                    (downcast(v128), downcast(v64), downcast(v32), downcast(v16)),
                    (downcast(v128), downcast(v64), downcast(v32)),
                    (downcast(v128), downcast(v64)),
                    (downcast(v128),),
                )
            }
        };
        static ref UPCAST: (String, Program) = load_cairo! {
            use core::integer::upcast;

            fn run_test(
                v8: u8, v16: u16, v32: u32, v64: u64, v128: u128, v248: bytes31
            ) -> (
                (u8,),
                (u16, u16),
                (u32, u32, u32),
                (u64, u64, u64, u64),
                (u128, u128, u128, u128, u128),
                (bytes31, bytes31, bytes31, bytes31, bytes31, bytes31)
            ) {
                (
                    (upcast(v8),),
                    (upcast(v8), upcast(v16)),
                    (upcast(v8), upcast(v16), upcast(v32)),
                    (upcast(v8), upcast(v16), upcast(v32), upcast(v64)),
                    (upcast(v8), upcast(v16), upcast(v32), upcast(v64), upcast(v128)),
                    (upcast(v8), upcast(v16), upcast(v32), upcast(v64), upcast(v128), upcast(v248)),
                )
            }
        };
    }

    #[test]
    fn downcast() {
        run_program_assert_output(
            &DOWNCAST,
            "run_test",
            &[
                u8::MAX.into(),
                u16::MAX.into(),
                u32::MAX.into(),
                u64::MAX.into(),
                u128::MAX.into(),
            ],
            jit_struct!(
                jit_struct!(
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                ),
                jit_struct!(
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                ),
                jit_struct!(
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                ),
                jit_struct!(jit_enum!(1, jit_struct!()), jit_enum!(1, jit_struct!())),
                jit_struct!(jit_enum!(1, jit_struct!())),
            ),
        );
    }

    #[test]
    fn upcast() {
        run_program_assert_output(
            &UPCAST,
            "run_test",
            &[
                u8::MAX.into(),
                u16::MAX.into(),
                u32::MAX.into(),
                u64::MAX.into(),
                u128::MAX.into(),
                Value::Bytes31([0xFF; 31]),
            ],
            jit_struct!(
                jit_struct!(u8::MAX.into()),
                jit_struct!((u8::MAX as u16).into(), u16::MAX.into()),
                jit_struct!(
                    (u8::MAX as u32).into(),
                    (u16::MAX as u32).into(),
                    u32::MAX.into()
                ),
                jit_struct!(
                    (u8::MAX as u64).into(),
                    (u16::MAX as u64).into(),
                    (u32::MAX as u64).into(),
                    u64::MAX.into()
                ),
                jit_struct!(
                    (u8::MAX as u128).into(),
                    (u16::MAX as u128).into(),
                    (u32::MAX as u128).into(),
                    (u64::MAX as u128).into(),
                    u128::MAX.into()
                ),
                jit_struct!(
                    Value::Bytes31([
                        u8::MAX,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]),
                    Value::Bytes31([
                        u8::MAX,
                        u8::MAX,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]),
                    Value::Bytes31([
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]),
                    Value::Bytes31([
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]),
                    Value::Bytes31([
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]),
                    Value::Bytes31([u8::MAX; 31]),
                ),
            ),
        );
    }
}
