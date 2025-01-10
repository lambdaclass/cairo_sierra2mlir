//! # Bytes31-related libfuncs

use super::LibfuncHelper;
use crate::{
    error::{Error, Result},
    metadata::MetadataStorage,
    utils::{BlockExt, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        bytes31::Bytes31ConcreteLibfunc,
        consts::SignatureAndConstConcreteLibfunc,
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        cf,
    },
    ir::{Attribute, Block, Location, Value},
    Context,
};
use num_bigint::BigUint;

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &Bytes31ConcreteLibfunc,
) -> Result<()> {
    match selector {
        Bytes31ConcreteLibfunc::Const(info) => {
            build_const(context, registry, entry, location, helper, metadata, info)
        }
        Bytes31ConcreteLibfunc::ToFelt252(info) => {
            build_to_felt252(context, registry, entry, location, helper, metadata, info)
        }
        Bytes31ConcreteLibfunc::TryFromFelt252(info) => {
            build_from_felt252(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `bytes31_const` libfunc.
pub fn build_const<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndConstConcreteLibfunc,
) -> Result<()> {
    let value = &info.c;
    let value_ty = registry.build_type(
        context,
        helper,
        metadata,
        &info.signature.branch_signatures[0].vars[0].ty,
    )?;

    let op0 = entry.append_operation(arith::constant(
        context,
        Attribute::parse(context, &format!("{value} : {value_ty}"))
            .ok_or(Error::ParseAttributeError)?,
        location,
    ));

    entry.append_operation(helper.br(0, &[op0.result(0)?.into()], location));

    Ok(())
}

/// Generate MLIR operations for the `bytes31_to_felt252` libfunc.
pub fn build_to_felt252<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let felt252_ty = registry.build_type(
        context,
        helper,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;
    let value: Value = entry.arg(0)?;

    let result = entry.extui(value, felt252_ty, location)?;

    entry.append_operation(helper.br(0, &[result], location));

    Ok(())
}

/// Generate MLIR operations for the `u8_from_felt252` libfunc.
pub fn build_from_felt252<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check: Value =
        super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let value: Value = entry.arg(1)?;

    let felt252_ty =
        registry.build_type(context, helper, metadata, &info.param_signatures()[1].ty)?;
    let result_ty = registry.build_type(
        context,
        helper,
        metadata,
        &info.branch_signatures()[0].vars[1].ty,
    )?;

    let max_value = BigUint::from(2u32).pow(248) - 1u32;

    let const_max = entry.append_op_result(arith::constant(
        context,
        Attribute::parse(context, &format!("{} : {}", max_value, felt252_ty))
            .ok_or(Error::ParseAttributeError)?,
        location,
    ))?;

    let is_ule = entry.cmpi(context, CmpiPredicate::Ule, value, const_max, location)?;

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

    let value = block_success.trunci(value, result_ty, location)?;
    block_success.append_operation(helper.br(0, &[range_check, value], location));

    block_failure.append_operation(helper.br(1, &[range_check], location));

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::utils::test::{jit_enum, jit_panic, jit_struct, run_sierra_program};
    use cairo_lang_sierra::{program::Program, ProgramParser};
    use lazy_static::lazy_static;
    use starknet_types_core::felt::Felt;

    lazy_static! {
        // TODO: Test `bytes31_const` once the compiler supports it.

        // use core::bytes_31::{bytes31_try_from_felt252, bytes31_to_felt252};
        // fn run_test(value: felt252) -> felt252 {
        //     let a: bytes31 = bytes31_try_from_felt252(value).unwrap();
        //     bytes31_to_felt252(a)
        // }
        static ref BYTES31_ROUNDTRIP: Program = ProgramParser::new().parse(r#"
            type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
            type [4] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
            type [5] = Array<[1]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [6] = Struct<ut@Tuple, [4], [5]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [8] = Const<[1], 29721761890975875353235833581453094220424382983267374> [storable: false, drop: false, dup: false, zero_sized: false];
            type [1] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [3] = Struct<ut@Tuple, [1]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [7] = Enum<ut@core::panics::PanicResult::<(core::felt252,)>, [3], [6]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [2] = bytes31 [storable: true, drop: true, dup: true, zero_sized: false];

            libfunc [8] = bytes31_try_from_felt252;
            libfunc [9] = branch_align;
            libfunc [7] = bytes31_to_felt252;
            libfunc [6] = struct_construct<[3]>;
            libfunc [5] = enum_init<[7], 0>;
            libfunc [11] = store_temp<[0]>;
            libfunc [12] = store_temp<[7]>;
            libfunc [4] = array_new<[1]>;
            libfunc [10] = const_as_immediate<[8]>;
            libfunc [13] = store_temp<[1]>;
            libfunc [3] = array_append<[1]>;
            libfunc [2] = struct_construct<[4]>;
            libfunc [1] = struct_construct<[6]>;
            libfunc [0] = enum_init<[7], 1>;

            [8]([0], [1]) { fallthrough([2], [3]) 8([4]) }; // 0
            [9]() -> (); // 1
            [7]([3]) -> ([5]); // 2
            [6]([5]) -> ([6]); // 3
            [5]([6]) -> ([7]); // 4
            [11]([2]) -> ([2]); // 5
            [12]([7]) -> ([7]); // 6
            return([2], [7]); // 7
            [9]() -> (); // 8
            [4]() -> ([8]); // 9
            [10]() -> ([9]); // 10
            [13]([9]) -> ([9]); // 11
            [3]([8], [9]) -> ([10]); // 12
            [2]() -> ([11]); // 13
            [1]([11], [10]) -> ([12]); // 14
            [0]([12]) -> ([13]); // 15
            [11]([4]) -> ([4]); // 16
            [12]([13]) -> ([13]); // 17
            return([4], [13]); // 18

            [0]@0([0]: [0], [1]: [1]) -> ([0], [7]);
            "#).map_err(|e| e.to_string()).unwrap();
    }

    #[test]
    fn bytes31_roundtrip() {
        let return_value1 =
            run_sierra_program(&BYTES31_ROUNDTRIP, &[Felt::from(2).into()]).return_value;

        assert_eq!(
            jit_enum!(0, jit_struct!(Felt::from(2).into())),
            return_value1
        );

        let return_value2 =
            run_sierra_program(&BYTES31_ROUNDTRIP, &[Felt::MAX.into()]).return_value;

        assert_eq!(
            return_value2,
            jit_panic!(Felt::from_bytes_be_slice(b"Option::unwrap failed."))
        );
    }
}
