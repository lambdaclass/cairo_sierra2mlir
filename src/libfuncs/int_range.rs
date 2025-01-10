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

    entry.append_operation(helper.cond_br(
        context,
        is_valid,
        [0, 1],
        [&[range_check, range], &[range_check, range]],
        location,
    ));
    Ok(())
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

    entry.append_operation(helper.cond_br(
        context,
        is_valid,
        [1, 0], // failure, success
        [&[range, x], &[]],
        location,
    ));
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{
        utils::test::{jit_enum, jit_struct, run_sierra_program},
        values::Value,
    };
    use cairo_lang_sierra::{program::Program, ProgramParser};
    use lazy_static::lazy_static;

    lazy_static! {
        // #[derive(Copy, Drop)]
        // pub extern type IntRange<T>;
        // pub extern fn int_range_try_new<T>(
        //     x: T, y: T
        // ) -> Result<IntRange<T>, IntRange<T>> implicits(core::RangeCheck) nopanic;
        // fn run_test(lhs: u64, rhs: u64) -> IntRange<u64> {
        //     int_range_try_new(lhs, rhs).unwrap()
        // }
        static ref INT_RANGE_TRY_NEW: Program = ProgramParser::new().parse(r#"
            type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
            type [4] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
            type [6] = Array<[5]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [7] = Struct<ut@Tuple, [4], [6]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [9] = Const<[5], 30828113188794245257250221355944970489240709081949230> [storable: false, drop: false, dup: false, zero_sized: false];
            type [5] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [2] = IntRange<[1]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [3] = Struct<ut@Tuple, [2]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [8] = Enum<ut@core::panics::PanicResult::<(program::program::IntRange::<core::integer::u64>,)>, [3], [7]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [1] = u64 [storable: true, drop: true, dup: true, zero_sized: false];

            libfunc [7] = int_range_try_new<[1]>;
            libfunc [8] = branch_align;
            libfunc [6] = struct_construct<[3]>;
            libfunc [5] = enum_init<[8], 0>;
            libfunc [11] = store_temp<[0]>;
            libfunc [12] = store_temp<[8]>;
            libfunc [9] = drop<[2]>;
            libfunc [4] = array_new<[5]>;
            libfunc [10] = const_as_immediate<[9]>;
            libfunc [13] = store_temp<[5]>;
            libfunc [3] = array_append<[5]>;
            libfunc [2] = struct_construct<[4]>;
            libfunc [1] = struct_construct<[7]>;
            libfunc [0] = enum_init<[8], 1>;

            [7]([0], [1], [2]) { fallthrough([3], [4]) 7([5], [6]) }; // 0
            [8]() -> (); // 1
            [6]([4]) -> ([7]); // 2
            [5]([7]) -> ([8]); // 3
            [11]([3]) -> ([3]); // 4
            [12]([8]) -> ([8]); // 5
            return([3], [8]); // 6
            [8]() -> (); // 7
            [9]([6]) -> (); // 8
            [4]() -> ([9]); // 9
            [10]() -> ([10]); // 10
            [13]([10]) -> ([10]); // 11
            [3]([9], [10]) -> ([11]); // 12
            [2]() -> ([12]); // 13
            [1]([12], [11]) -> ([13]); // 14
            [0]([13]) -> ([14]); // 15
            [11]([5]) -> ([5]); // 16
            [12]([14]) -> ([14]); // 17
            return([5], [14]); // 18

            [0]@0([0]: [0], [1]: [1], [2]: [1]) -> ([0], [8]);
        "#).map_err(|e| e.to_string()).unwrap();
    }

    #[test]
    fn int_range_try_new() {
        let return_value =
            run_sierra_program(&INT_RANGE_TRY_NEW, &[2u64.into(), 4u64.into()]).return_value;

        assert_eq!(
            jit_enum!(
                0,
                jit_struct!(Value::IntRange {
                    x: Box::new(2u64.into()),
                    y: Box::new(4u64.into()),
                })
            ),
            return_value
        );
    }
}
