//! # Int range libfuncs

use super::LibfuncHelper;
use crate::{
    error::Result,
    metadata::MetadataStorage,
    types::TypeBuilder,
    utils::{BlockExt, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
        range::IntRangeConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        ods,
    },
    ir::{Block, Location},
    Context,
};
use num_bigint::BigInt;

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &IntRangeConcreteLibfunc,
) -> Result<()> {
    match selector {
        IntRangeConcreteLibfunc::TryNew(info) => {
            build_int_range_try_new(context, registry, entry, location, helper, metadata, info)
        }
        IntRangeConcreteLibfunc::PopFront(info) => {
            build_int_range_pop_front(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `int_range_try_new` libfunc.
pub fn build_int_range_try_new<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check = entry.arg(0)?;
    let x = entry.arg(1)?;
    let y = entry.arg(2)?;
    let range_ty = registry.build_type(
        context,
        helper,
        metadata,
        &info.branch_signatures()[0].vars[1].ty,
    )?;
    let inner = registry.get_type(&info.param_signatures()[1].ty)?;
    // to know if it is signed
    let inner_range = inner.integer_range(registry)?;

    let is_valid = if inner_range.lower < BigInt::ZERO {
        entry.cmpi(context, CmpiPredicate::Sle, x, y, location)?
    } else {
        entry.cmpi(context, CmpiPredicate::Ule, x, y, location)?
    };

    let range =
        entry.append_op_result(ods::llvm::mlir_undef(context, range_ty, location).into())?;

    // if the range is not valid, return the empty range [y, y)
    let x_val = entry.append_op_result(arith::select(is_valid, x, y, location))?;
    let range = entry.insert_values(context, location, range, &[x_val, y])?;

    helper.cond_br(
        context,
        entry,
        is_valid,
        [0, 1],
        [&[range_check, range], &[range_check, range]],
        location,
    )
}

/// Generate MLIR operations for the `int_range_pop_front` libfunc.
pub fn build_int_range_pop_front<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range = entry.arg(0)?;

    let inner_ty = registry.build_type(
        context,
        helper,
        metadata,
        &info.branch_signatures()[1].vars[1].ty,
    )?;

    let inner = registry.get_type(&info.branch_signatures()[1].vars[1].ty)?;

    let x = entry.extract_value(context, location, range, inner_ty, 0)?;
    let k1 = entry.const_int_from_type(context, location, 1, inner_ty)?;
    let x_p_1 = entry.addi(x, k1, location)?;
    let y = entry.extract_value(context, location, range, inner_ty, 1)?;

    // to know if it is signed
    let inner_range = inner.integer_range(registry)?;

    let is_valid = if inner_range.lower < BigInt::ZERO {
        entry.cmpi(context, CmpiPredicate::Slt, x, y, location)?
    } else {
        entry.cmpi(context, CmpiPredicate::Ult, x, y, location)?
    };
    let range = entry.insert_value(context, location, range, x_p_1, 0)?;

    helper.cond_br(
        context,
        entry,
        is_valid,
        [1, 0], // failure, success
        [&[range, x], &[]],
        location,
    )
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
        static ref INT_RANGE_TRY_NEW: (String, Program) = load_cairo! {
            #[derive(Copy, Drop)]
            pub extern type IntRange<T>;
            pub extern fn int_range_try_new<T>(
                x: T, y: T
            ) -> Result<IntRange<T>, IntRange<T>> implicits(core::RangeCheck) nopanic;

            fn run_test(lhs: u64, rhs: u64) -> IntRange<u64> {
                int_range_try_new(lhs, rhs).unwrap()
            }
        };
    }

    #[test]
    fn int_range_try_new() {
        run_program_assert_output(
            &INT_RANGE_TRY_NEW,
            "run_test",
            &[2u64.into(), 4u64.into()],
            jit_enum!(
                0,
                jit_struct!(Value::IntRange {
                    x: Box::new(2u64.into()),
                    y: Box::new(4u64.into()),
                })
            ),
        );
    }
}
