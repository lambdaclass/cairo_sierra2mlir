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
    let i128_ty = IntegerType::new(context, 128).into();

    let gas_u128 = entry.extui(entry.arg(0)?, i128_ty, location)?;

    // The gas is returned as u128 on the second arg.
    helper.br(entry, 0, &[entry.arg(0)?, gas_u128], location)
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
    let builtin_ptr = {
        let runtime = metadata
            .get_mut::<RuntimeBindingsMeta>()
            .ok_or(Error::MissingMetadata)?;
        runtime
            .get_gas_builtin(context, helper, entry, location)?
            .result(0)?
            .into()
    };

    let gas_cost = metadata
        .get::<GasCost>()
        .expect("withdraw_gas should always have a gas cost")
        .clone();

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

    helper.cond_br(
        context,
        entry,
        is_enough,
        [0, 1],
        [&[range_check, resulting_gas], &[range_check, current_gas]],
        location,
    )
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

    helper.br(entry, 0, &[resulting_gas], location)
}

/// Generate MLIR operations for the `withdraw_gas_all` libfunc.
pub fn build_builtin_withdraw_gas<'ctx, 'this>(
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
    let builtin_ptr = entry.arg(2)?;

    let gas_cost = metadata
        .get::<GasCost>()
        .expect("builtin_withdraw_gas should always have a gas cost")
        .clone();

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

    helper.cond_br(
        context,
        entry,
        is_enough,
        [0, 1],
        [&[range_check, resulting_gas], &[range_check, current_gas]],
        location,
    )
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

    helper.br(entry, 0, &[builtin_ptr], location)
}

#[cfg(test)]
mod test {
    use crate::utils::test::{load_cairo, run_program};

    #[test]
    fn run_withdraw_gas() {
        #[rustfmt::skip]
        let program = load_cairo!(
            use gas::withdraw_gas;

            fn run_test() {
                let mut i = 10;

                loop {
                    if i == 0 {
                        break;
                    }

                    match withdraw_gas() {
                        Option::Some(()) => {
                            i = i - 1;
                        },
                        Option::None(()) => {
                            break;
                        }
                    };
                    i = i - 1;
                }
            }
        );

        let result = run_program(&program, "run_test", &[]);
        assert_eq!(result.remaining_gas, Some(18446744073709545265));
    }
}
