//! # Gas management libfuncs

use super::LibfuncHelper;
use crate::{
    error::{Error, Result},
    metadata::{gas::GasCost, runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
    native_panic,
    utils::{BlockExt, GepIndex},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        gas::{CostTokenType, GasConcreteLibfunc},
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{arith::CmpiPredicate, ods},
    ir::{r#type::IntegerType, Block, Location},
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
    selector: &GasConcreteLibfunc,
) -> Result<()> {
    match selector {
        GasConcreteLibfunc::WithdrawGas(info) => {
            build_withdraw_gas(context, registry, entry, location, helper, metadata, info)
        }
        GasConcreteLibfunc::RedepositGas(info) => {
            build_redeposit_gas(context, registry, entry, location, helper, metadata, info)
        }
        GasConcreteLibfunc::GetAvailableGas(info) => {
            build_get_available_gas(context, registry, entry, location, helper, metadata, info)
        }
        GasConcreteLibfunc::BuiltinWithdrawGas(info) => {
            build_builtin_withdraw_gas(context, registry, entry, location, helper, metadata, info)
        }
        GasConcreteLibfunc::GetBuiltinCosts(info) => {
            build_get_builtin_costs(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `get_available_gas` libfunc.
pub fn build_get_available_gas<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let gas = entry.arg(0)?;
    let gas_u128 = entry.extui(gas, IntegerType::new(context, 128).into(), location)?;
    // The gas is returned as u128 on the second arg.
    entry.append_operation(helper.br(0, &[entry.arg(0)?, gas_u128], location));
    Ok(())
}

/// Generate MLIR operations for the `withdraw_gas` libfunc.
pub fn build_withdraw_gas<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;
    let current_gas = entry.arg(1)?;

    let gas_cost = metadata
        .get::<GasCost>()
        .expect("withdraw_gas should always have a gas cost")
        .clone();

    let u64_type: melior::ir::Type = IntegerType::new(context, 64).into();

    let builtin_ptr = {
        let runtime = metadata
            .get_mut::<RuntimeBindingsMeta>()
            .ok_or(Error::MissingMetadata)?;
        runtime
            .get_gas_builtin(context, helper, entry, location)?
            .result(0)?
            .into()
    };

    let mut total_gas_cost_value = entry.const_int_from_type(context, location, 0, u64_type)?;

    for (cost_count, token_type) in &gas_cost.0 {
        if *cost_count == 0 {
            continue;
        }

        let builtin_costs_index = match token_type {
            CostTokenType::Const => 0,
            CostTokenType::Pedersen => 1,
            CostTokenType::Bitwise => 2,
            CostTokenType::EcOp => 3,
            CostTokenType::Poseidon => 4,
            CostTokenType::AddMod => 5,
            CostTokenType::MulMod => 6,
            _ => native_panic!("matched an unexpected CostTokenType which is not being used"),
        };

        let cost_count_value =
            entry.const_int_from_type(context, location, *cost_count, u64_type)?;
        let builtin_costs_index_value =
            entry.const_int_from_type(context, location, builtin_costs_index, u64_type)?;

        let builtin_cost_value_ptr = entry.gep(
            context,
            location,
            builtin_ptr,
            &[GepIndex::Value(builtin_costs_index_value)],
            u64_type,
        )?;
        let cost_value = entry.load(context, location, builtin_cost_value_ptr, u64_type)?;
        let gas_cost_value = entry.muli(cost_count_value, cost_value, location)?;
        total_gas_cost_value = entry.addi(total_gas_cost_value, gas_cost_value, location)?;
    }

    let is_enough = entry.cmpi(
        context,
        CmpiPredicate::Uge,
        current_gas,
        total_gas_cost_value,
        location,
    )?;

    let resulting_gas = entry.append_op_result(
        ods::llvm::intr_usub_sat(context, current_gas, total_gas_cost_value, location).into(),
    )?;

    entry.append_operation(helper.cond_br(
        context,
        is_enough,
        [0, 1],
        [&[range_check, resulting_gas], &[range_check, current_gas]],
        location,
    ));

    Ok(())
}

/// Returns the unused gas to the remaining
///
/// ```cairo
/// extern fn redeposit_gas() implicits(GasBuiltin) nopanic;
/// ```
pub fn build_redeposit_gas<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let current_gas = entry.arg(0)?;

    let gas_cost = metadata
        .get::<GasCost>()
        .expect("redeposit_gas should always have a gas cost")
        .clone();

    let u64_type: melior::ir::Type = IntegerType::new(context, 64).into();

    let builtin_ptr = {
        let runtime = metadata
            .get_mut::<RuntimeBindingsMeta>()
            .ok_or(Error::MissingMetadata)?;
        runtime
            .get_gas_builtin(context, helper, entry, location)?
            .result(0)?
            .into()
    };

    let mut total_gas_cost_value = entry.const_int_from_type(context, location, 0, u64_type)?;

    for (cost_count, token_type) in &gas_cost.0 {
        if *cost_count == 0 {
            continue;
        }

        let builtin_costs_index = match token_type {
            CostTokenType::Const => 0,
            CostTokenType::Pedersen => 1,
            CostTokenType::Bitwise => 2,
            CostTokenType::EcOp => 3,
            CostTokenType::Poseidon => 4,
            CostTokenType::AddMod => 5,
            CostTokenType::MulMod => 6,
            _ => native_panic!("matched an unexpected CostTokenType which is not being used"),
        };

        let cost_count_value =
            entry.const_int_from_type(context, location, *cost_count, u64_type)?;
        let builtin_costs_index_value =
            entry.const_int_from_type(context, location, builtin_costs_index, u64_type)?;

        let builtin_cost_value_ptr = entry.gep(
            context,
            location,
            builtin_ptr,
            &[GepIndex::Value(builtin_costs_index_value)],
            u64_type,
        )?;
        let cost_value = entry.load(context, location, builtin_cost_value_ptr, u64_type)?;
        let gas_cost_value = entry.muli(cost_count_value, cost_value, location)?;
        total_gas_cost_value = entry.addi(total_gas_cost_value, gas_cost_value, location)?;
    }

    let resulting_gas = entry.append_op_result(
        ods::llvm::intr_uadd_sat(context, current_gas, total_gas_cost_value, location).into(),
    )?;

    entry.append_operation(helper.br(0, &[resulting_gas], location));

    Ok(())
}

/// Generate MLIR operations for the `withdraw_gas_all` libfunc.
pub fn build_builtin_withdraw_gas<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;
    let current_gas = entry.arg(1)?;
    let builtin_ptr = entry.arg(2)?;

    let gas_cost = metadata
        .get::<GasCost>()
        .expect("builtin_withdraw_gas should always have a gas cost");

    let u64_type: melior::ir::Type = IntegerType::new(context, 64).into();

    let mut total_gas_cost_value = entry.const_int_from_type(context, location, 0, u64_type)?;

    for (cost_count, token_type) in &gas_cost.0 {
        if *cost_count == 0 {
            continue;
        }

        let builtin_costs_index = match token_type {
            CostTokenType::Const => 0,
            CostTokenType::Pedersen => 1,
            CostTokenType::Bitwise => 2,
            CostTokenType::EcOp => 3,
            CostTokenType::Poseidon => 4,
            CostTokenType::AddMod => 5,
            CostTokenType::MulMod => 6,
            _ => native_panic!("matched an unexpected CostTokenType which is not being used"),
        };

        let cost_count_value =
            entry.const_int_from_type(context, location, *cost_count, u64_type)?;
        let builtin_costs_index_value =
            entry.const_int_from_type(context, location, builtin_costs_index, u64_type)?;

        let builtin_cost_value_ptr = entry.gep(
            context,
            location,
            builtin_ptr,
            &[GepIndex::Value(builtin_costs_index_value)],
            u64_type,
        )?;
        let cost_value = entry.load(context, location, builtin_cost_value_ptr, u64_type)?;
        let gas_cost_value = entry.muli(cost_count_value, cost_value, location)?;
        total_gas_cost_value = entry.addi(total_gas_cost_value, gas_cost_value, location)?;
    }

    let is_enough = entry.cmpi(
        context,
        CmpiPredicate::Uge,
        current_gas,
        total_gas_cost_value,
        location,
    )?;

    let resulting_gas = entry.append_op_result(
        ods::llvm::intr_usub_sat(context, current_gas, total_gas_cost_value, location).into(),
    )?;

    entry.append_operation(helper.cond_br(
        context,
        is_enough,
        [0, 1],
        [&[range_check, resulting_gas], &[range_check, current_gas]],
        location,
    ));

    Ok(())
}

/// Generate MLIR operations for the `get_builtin_costs` libfunc.
pub fn build_get_builtin_costs<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Get the ptr to the global, holding a ptr to the list.
    let builtin_ptr = {
        let runtime = metadata
            .get_mut::<RuntimeBindingsMeta>()
            .ok_or(Error::MissingMetadata)?;
        runtime
            .get_gas_builtin(context, helper, entry, location)?
            .result(0)?
            .into()
    };

    entry.append_operation(helper.br(0, &[builtin_ptr], location));

    Ok(())
}

#[cfg(test)]
mod test {
    use cairo_lang_sierra::ProgramParser;

    use crate::utils::test::run_sierra_program;

    #[test]
    fn run_withdraw_gas() {
        // use gas::withdraw_gas;
        // fn run_test() {
        //     let mut i = 10;
        //     loop {
        //         if i == 0 {
        //             break;
        //         }
        //         match withdraw_gas() {
        //             Option::Some(()) => {
        //                 i = i - 1;
        //             },
        //             Option::None(()) => {
        //                 break;
        //             }
        //         };
        //         i = i - 1;
        //     }
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
            type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
            type [3] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
            type [2] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [6] = NonZero<[2]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [5] = Const<[2], 1> [storable: false, drop: false, dup: false, zero_sized: false];
            type [4] = Const<[2], 10> [storable: false, drop: false, dup: false, zero_sized: false];
            type [1] = GasBuiltin [storable: true, drop: false, dup: false, zero_sized: false];

            libfunc [4] = disable_ap_tracking;
            libfunc [3] = withdraw_gas;
            libfunc [5] = branch_align;
            libfunc [0] = redeposit_gas;
            libfunc [6] = const_as_immediate<[4]>;
            libfunc [7] = const_as_immediate<[5]>;
            libfunc [9] = store_temp<[2]>;
            libfunc [2] = felt252_sub;
            libfunc [10] = store_temp<[0]>;
            libfunc [11] = store_temp<[1]>;
            libfunc [1] = function_call<user@[0]>;
            libfunc [8] = drop<[2]>;
            libfunc [13] = dup<[2]>;
            libfunc [12] = felt252_is_zero;
            libfunc [14] = drop<[6]>;

            [4]() -> (); // 0
            [3]([0], [1]) { fallthrough([2], [3]) 19([4], [5]) }; // 1
            [5]() -> (); // 2
            [0]([3]) -> ([6]); // 3
            [6]() -> ([7]); // 4
            [7]() -> ([8]); // 5
            [9]([7]) -> ([7]); // 6
            [2]([7], [8]) -> ([9]); // 7
            [7]() -> ([10]); // 8
            [9]([9]) -> ([9]); // 9
            [2]([9], [10]) -> ([11]); // 10
            [10]([2]) -> ([2]); // 11
            [11]([6]) -> ([6]); // 12
            [9]([11]) -> ([11]); // 13
            [1]([2], [6], [11]) -> ([12], [13], [14]); // 14
            [8]([14]) -> (); // 15
            [10]([12]) -> ([12]); // 16
            [11]([13]) -> ([13]); // 17
            return([12], [13]); // 18
            [5]() -> (); // 19
            [0]([5]) -> ([15]); // 20
            [10]([4]) -> ([4]); // 21
            [11]([15]) -> ([15]); // 22
            return([4], [15]); // 23
            [4]() -> (); // 24
            [13]([2]) -> ([2], [3]); // 25
            [12]([3]) { fallthrough() 33([4]) }; // 26
            [5]() -> (); // 27
            [0]([1]) -> ([5]); // 28
            [10]([0]) -> ([0]); // 29
            [11]([5]) -> ([5]); // 30
            [9]([2]) -> ([2]); // 31
            return([0], [5], [2]); // 32
            [5]() -> (); // 33
            [14]([4]) -> (); // 34
            [0]([1]) -> ([6]); // 35
            [11]([6]) -> ([6]); // 36
            [3]([0], [6]) { fallthrough([7], [8]) 50([9], [10]) }; // 37
            [5]() -> (); // 38
            [0]([8]) -> ([11]); // 39
            [7]() -> ([12]); // 40
            [2]([2], [12]) -> ([13]); // 41
            [7]() -> ([14]); // 42
            [9]([13]) -> ([13]); // 43
            [2]([13], [14]) -> ([15]); // 44
            [10]([7]) -> ([7]); // 45
            [11]([11]) -> ([11]); // 46
            [9]([15]) -> ([15]); // 47
            [1]([7], [11], [15]) -> ([16], [17], [18]); // 48
            return([16], [17], [18]); // 49
            [5]() -> (); // 50
            [0]([10]) -> ([19]); // 51
            [10]([9]) -> ([9]); // 52
            [11]([19]) -> ([19]); // 53
            [9]([2]) -> ([2]); // 54
            return([9], [19], [2]); // 55

            [1]@0([0]: [0], [1]: [1]) -> ([0], [1]);
            [0]@24([0]: [0], [1]: [1], [2]: [2]) -> ([0], [1], [2]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let result = run_sierra_program(&program, &[]);
        assert_eq!(result.remaining_gas, Some(18446744073709545265));
    }
}
