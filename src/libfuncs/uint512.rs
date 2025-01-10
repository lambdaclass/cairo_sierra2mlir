//! # `u512`-related libfuncs

use super::LibfuncHelper;
use crate::{
    error::Result,
    metadata::MetadataStorage,
    utils::{BlockExt, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        int::unsigned512::Uint512Concrete,
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{arith, llvm},
    ir::{r#type::IntegerType, Block, Location, Value},
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
    selector: &Uint512Concrete,
) -> Result<()> {
    match selector {
        Uint512Concrete::DivModU256(info) => {
            build_divmod_u256(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `u512_safe_divmod_by_u256` libfunc.
pub fn build_divmod_u256<'ctx, 'this>(
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
    let i512_ty = IntegerType::new(context, 512).into();

    let guarantee_type =
        registry.build_type(context, helper, metadata, &info.output_types()[0][3])?;

    let lhs_struct: Value = entry.arg(1)?;
    let rhs_struct: Value = entry.arg(2)?;

    let dividend = (
        entry.extract_value(context, location, lhs_struct, i128_ty, 0)?,
        entry.extract_value(context, location, lhs_struct, i128_ty, 1)?,
        entry.extract_value(context, location, lhs_struct, i128_ty, 2)?,
        entry.extract_value(context, location, lhs_struct, i128_ty, 3)?,
    );
    let divisor = (
        entry.extract_value(context, location, rhs_struct, i128_ty, 0)?,
        entry.extract_value(context, location, rhs_struct, i128_ty, 1)?,
    );

    let dividend = (
        entry.extui(dividend.0, i512_ty, location)?,
        entry.extui(dividend.1, i512_ty, location)?,
        entry.extui(dividend.2, i512_ty, location)?,
        entry.extui(dividend.3, i512_ty, location)?,
    );
    let divisor = (
        entry.extui(divisor.0, i512_ty, location)?,
        entry.extui(divisor.1, i512_ty, location)?,
    );

    let k128 = entry.const_int_from_type(context, location, 128, i512_ty)?;
    let k256 = entry.const_int_from_type(context, location, 256, i512_ty)?;
    let k384 = entry.const_int_from_type(context, location, 384, i512_ty)?;

    let dividend = (
        dividend.0,
        entry.shli(dividend.1, k128, location)?,
        entry.shli(dividend.2, k256, location)?,
        entry.shli(dividend.3, k384, location)?,
    );
    let divisor = (divisor.0, entry.shli(divisor.1, k128, location)?);

    let dividend = {
        let lhs_01 = entry.append_op_result(arith::ori(dividend.0, dividend.1, location))?;
        let lhs_23 = entry.append_op_result(arith::ori(dividend.2, dividend.3, location))?;

        entry.append_op_result(arith::ori(lhs_01, lhs_23, location))?
    };
    let divisor = entry.append_op_result(arith::ori(divisor.0, divisor.1, location))?;

    let result_div = entry.append_op_result(arith::divui(dividend, divisor, location))?;
    let result_rem = entry.append_op_result(arith::remui(dividend, divisor, location))?;

    let result_div = (
        entry.trunci(result_div, i128_ty, location)?,
        entry.trunci(entry.shrui(result_div, k128, location)?, i128_ty, location)?,
        entry.trunci(entry.shrui(result_div, k256, location)?, i128_ty, location)?,
        entry.trunci(entry.shrui(result_div, k384, location)?, i128_ty, location)?,
    );

    let result_rem = (
        entry.trunci(result_rem, i128_ty, location)?,
        entry.trunci(entry.shrui(result_rem, k128, location)?, i128_ty, location)?,
    );

    let result_div_val = entry.append_op_result(llvm::undef(
        llvm::r#type::r#struct(context, &[i128_ty, i128_ty, i128_ty, i128_ty], false),
        location,
    ))?;
    let result_div_val = entry.insert_values(
        context,
        location,
        result_div_val,
        &[result_div.0, result_div.1, result_div.2, result_div.3],
    )?;

    let result_rem_val = entry.append_op_result(llvm::undef(
        llvm::r#type::r#struct(context, &[i128_ty, i128_ty], false),
        location,
    ))?;
    let result_rem_val = entry.insert_values(
        context,
        location,
        result_rem_val,
        &[result_rem.0, result_rem.1],
    )?;

    let guarantee = entry.append_op_result(llvm::undef(guarantee_type, location))?;

    entry.append_operation(helper.br(
        0,
        &[
            range_check,
            result_div_val,
            result_rem_val,
            guarantee,
            guarantee,
            guarantee,
            guarantee,
            guarantee,
        ],
        location,
    ));
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{
        utils::test::{jit_struct, run_sierra_program},
        values::Value,
    };
    use cairo_lang_sierra::{program::Program, ProgramParser};
    use lazy_static::lazy_static;
    use num_bigint::BigUint;
    use num_traits::One;

    lazy_static! {
        // use core::integer::{u512, u512_safe_divmod_by_u256};
        // fn run_test(lhs: u512, rhs: NonZero<u256>) -> (u512, u256) {
        //     let (lhs, rhs, _, _, _, _, _) = u512_safe_divmod_by_u256(lhs, rhs);
        //     (lhs, rhs)
        // }
        static ref UINT512_DIVMOD_U256: Program = ProgramParser::new().parse(r#"
            type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
            type [1] = u128 [storable: true, drop: true, dup: true, zero_sized: false];
            type [2] = Struct<ut@core::integer::u512, [1], [1], [1], [1]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [3] = Struct<ut@core::integer::u256, [1], [1]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [6] = Struct<ut@Tuple, [2], [3]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [5] = U128MulGuarantee [storable: true, drop: false, dup: false, zero_sized: false];
            type [4] = NonZero<[3]> [storable: true, drop: true, dup: true, zero_sized: false];

            libfunc [2] = u512_safe_divmod_by_u256;
            libfunc [1] = u128_mul_guarantee_verify;
            libfunc [0] = struct_construct<[6]>;
            libfunc [4] = store_temp<[0]>;
            libfunc [5] = store_temp<[6]>;

            [2]([0], [1], [2]) -> ([3], [4], [5], [6], [7], [8], [9], [10]); // 0
            [1]([3], [10]) -> ([11]); // 1
            [1]([11], [9]) -> ([12]); // 2
            [1]([12], [8]) -> ([13]); // 3
            [1]([13], [7]) -> ([14]); // 4
            [1]([14], [6]) -> ([15]); // 5
            [0]([4], [5]) -> ([16]); // 6
            [4]([15]) -> ([15]); // 7
            [5]([16]) -> ([16]); // 8
            return([15], [16]); // 9

            [0]@0([0]: [0], [1]: [2], [2]: [4]) -> ([0], [6]);
        "#).map_err(|e| e.to_string()).unwrap();
    }

    #[test]
    fn u512_safe_divmod_by_u256() {
        fn u512(value: BigUint) -> Value {
            assert!(value.bits() <= 512);
            jit_struct!(
                Value::Uint128((&value & &u128::MAX.into()).try_into().unwrap()),
                Value::Uint128(((&value >> 128u32) & &u128::MAX.into()).try_into().unwrap()),
                Value::Uint128(((&value >> 256u32) & &u128::MAX.into()).try_into().unwrap()),
                Value::Uint128(((&value >> 384u32) & &u128::MAX.into()).try_into().unwrap()),
            )
        }

        fn u256(value: BigUint) -> Value {
            assert!(value.bits() <= 256);
            jit_struct!(
                Value::Uint128((&value & &u128::MAX.into()).try_into().unwrap()),
                Value::Uint128(((&value >> 128u32) & &u128::MAX.into()).try_into().unwrap()),
            )
        }

        #[track_caller]
        fn r2(lhs: BigUint, rhs: BigUint, output_u512: BigUint, output_u256: BigUint) {
            let lhs = u512(lhs);
            let rhs = u256(rhs);
            let output_u512 = u512(output_u512);
            let output_u256 = u256(output_u256);
            let return_value = run_sierra_program(&UINT512_DIVMOD_U256, &[lhs, rhs]).return_value;
            assert_eq!(jit_struct!(output_u512, output_u256), return_value);
        }

        r2(0u32.into(), 1u32.into(), 0u32.into(), 0u32.into());
        r2(
            0u32.into(),
            (BigUint::one() << 256u32) - 2u32,
            0u32.into(),
            0u32.into(),
        );
        r2(
            0u32.into(),
            (BigUint::one() << 256u32) - 1u32,
            0u32.into(),
            0u32.into(),
        );

        r2(1u32.into(), 1u32.into(), 1u32.into(), 0u32.into());
        r2(
            1u32.into(),
            (BigUint::one() << 256u32) - 2u32,
            0u32.into(),
            1u32.into(),
        );
        r2(
            1u32.into(),
            (BigUint::one() << 256u32) - 1u32,
            0u32.into(),
            1u32.into(),
        );

        r2(
            (BigUint::one() << 512u32) - 2u32,
            (BigUint::one() << 256u32) - 2u32,
            (BigUint::one() << 256) + 2u32,
            2u32.into(),
        );
        r2(
            (BigUint::one() << 512u32) - 2u32,
            1u32.into(),
            (BigUint::one() << 512u32) - 2u32,
            0u32.into(),
        );
        r2(
            (BigUint::one() << 512u32) - 2u32,
            (BigUint::one() << 256u32) - 2u32,
            (BigUint::one() << 256) + 2u32,
            2u32.into(),
        );
        r2(
            (BigUint::one() << 512u32) - 2u32,
            (BigUint::one() << 256u32) - 1u32,
            BigUint::one() << 256u32,
            (BigUint::one() << 256u32) - 2u32,
        );

        r2(
            (BigUint::one() << 512u32) - 1u32,
            1u32.into(),
            (BigUint::one() << 512u32) - 1u32,
            0u32.into(),
        );
        r2(
            (BigUint::one() << 512u32) - 1u32,
            (BigUint::one() << 256u32) - 2u32,
            (BigUint::one() << 256) + 2u32,
            3u32.into(),
        );
        r2(
            (BigUint::one() << 512u32) - 1u32,
            (BigUint::one() << 256u32) - 1u32,
            (BigUint::one() << 256) + 1u32,
            0u32.into(),
        );
    }
}
