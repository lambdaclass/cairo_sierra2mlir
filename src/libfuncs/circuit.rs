//! # Circuit libfuncs

use super::{increment_builtin_counter_by, LibfuncHelper};
use crate::{
    error::{Result, SierraAssertError},
    libfuncs::r#struct::build_struct_value,
    metadata::MetadataStorage,
    types::TypeBuilder,
    utils::{get_integer_layout, layout_repeat, BlockExt, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        circuit::{
            self, CircuitConcreteLibfunc, CircuitTypeConcrete, ConcreteGetOutputLibFunc,
            ConcreteU96LimbsLessThanGuaranteeVerifyLibfunc,
        },
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        lib_func::{SignatureAndTypeConcreteLibfunc, SignatureOnlyConcreteLibfunc},
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        cf, llvm,
    },
    ir::{
        attribute::DenseI32ArrayAttribute, r#type::IntegerType, Block, Location, Value, ValueLike,
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
    selector: &CircuitConcreteLibfunc,
) -> Result<()> {
    match selector {
        CircuitConcreteLibfunc::AddInput(info) => {
            build_add_input(context, registry, entry, location, helper, metadata, info)
        }
        CircuitConcreteLibfunc::Eval(info) => {
            build_eval(context, registry, entry, location, helper, metadata, info)
        }
        CircuitConcreteLibfunc::GetDescriptor(info) => {
            build_get_descriptor(context, registry, entry, location, helper, metadata, info)
        }
        CircuitConcreteLibfunc::InitCircuitData(info) => {
            build_init_circuit_data(context, registry, entry, location, helper, metadata, info)
        }
        CircuitConcreteLibfunc::GetOutput(info) => {
            build_get_output(context, registry, entry, location, helper, metadata, info)
        }
        CircuitConcreteLibfunc::TryIntoCircuitModulus(info) => build_try_into_circuit_modulus(
            context, registry, entry, location, helper, metadata, info,
        ),
        CircuitConcreteLibfunc::FailureGuaranteeVerify(info) => build_failure_guarantee_verify(
            context, registry, entry, location, helper, metadata, info,
        ),
        CircuitConcreteLibfunc::IntoU96Guarantee(SignatureAndTypeConcreteLibfunc {
            signature,
            ..
        })
        | CircuitConcreteLibfunc::U96GuaranteeVerify(SignatureOnlyConcreteLibfunc { signature }) => {
            super::build_noop::<1, true>(
                context,
                registry,
                entry,
                location,
                helper,
                metadata,
                &signature.param_signatures,
            )
        }
        CircuitConcreteLibfunc::U96LimbsLessThanGuaranteeVerify(info) => {
            build_u96_limbs_less_than_guarantee_verify(
                context, registry, entry, location, helper, metadata, info,
            )
        }
        CircuitConcreteLibfunc::U96SingleLimbLessThanGuaranteeVerify(info) => {
            build_u96_single_limb_less_than_guarantee_verify(
                context, registry, entry, location, helper, metadata, info,
            )
        }
    }
}

/// Generate MLIR operations for the `init_circuit_data` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_init_circuit_data<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let rc_usage = match registry.get_type(&info.ty)? {
        CoreTypeConcrete::Circuit(CircuitTypeConcrete::Circuit(info)) => {
            info.circuit_info.rc96_usage()
        }
        _ => return Err(SierraAssertError::BadTypeInfo.into()),
    };
    let rc = increment_builtin_counter_by(context, entry, location, entry.arg(0)?, rc_usage)?;

    let k0 = entry.const_int(context, location, 0, 64)?;
    let accumulator_ty = &info.branch_signatures()[0].vars[1].ty;
    let accumulator = build_struct_value(
        context,
        registry,
        entry,
        location,
        helper,
        metadata,
        accumulator_ty,
        &[k0],
    )?;

    entry.append_operation(helper.br(0, &[rc, accumulator], location));

    Ok(())
}

/// Generate MLIR operations for the `add_circuit_input` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_add_input<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let n_inputs = match registry.get_type(&info.ty)? {
        CoreTypeConcrete::Circuit(CircuitTypeConcrete::Circuit(info)) => info.circuit_info.n_inputs,
        _ => return Err(SierraAssertError::BadTypeInfo.into()),
    };
    let accumulator_type_id = &info.param_signatures()[0].ty;
    let accumulator_ctype = registry.get_type(accumulator_type_id)?;
    let accumulator_layout = accumulator_ctype.layout(registry)?;

    let accumulator: Value = entry.arg(0)?;

    // Get accumulator current length
    let current_length = entry.extract_value(
        context,
        location,
        accumulator,
        IntegerType::new(context, 64).into(),
        0,
    )?;

    // Check if last_insert: current_length == number_of_inputs - 1
    let n_inputs_minus_1 = entry.const_int(context, location, n_inputs - 1, 64)?;
    let last_insert = entry.cmpi(
        context,
        arith::CmpiPredicate::Eq,
        current_length,
        n_inputs_minus_1,
        location,
    )?;

    let middle_insert_block = helper.append_block(Block::new(&[]));
    let last_insert_block = helper.append_block(Block::new(&[]));
    entry.append_operation(cf::cond_br(
        context,
        last_insert,
        last_insert_block,
        middle_insert_block,
        &[],
        &[],
        location,
    ));

    // If not last insert, then:
    {
        // Calculate next length: next_length = current_length + 1
        let k1 = middle_insert_block.const_int(context, location, 1, 64)?;
        let next_length = middle_insert_block.addi(current_length, k1, location)?;

        // Insert next_length into accumulator
        let accumulator =
            middle_insert_block.insert_value(context, location, accumulator, next_length, 0)?;

        // Get pointer to accumulator with alloc and store
        let accumulator_ptr = helper.init_block().alloca1(
            context,
            location,
            accumulator.r#type(),
            accumulator_layout.align(),
        )?;
        middle_insert_block.store(context, location, accumulator_ptr, accumulator)?;

        // Get pointer to next input to insert
        let k0 = middle_insert_block.const_int(context, location, 0, 64)?;
        let next_input_ptr =
            middle_insert_block.append_op_result(llvm::get_element_ptr_dynamic(
                context,
                accumulator_ptr,
                &[k0, k1, current_length],
                accumulator.r#type(),
                llvm::r#type::pointer(context, 0),
                location,
            ))?;

        // Interpret u384 struct (input) as u384 integer
        let u384_struct = entry.arg(1)?;
        let new_input =
            u384_struct_to_integer(context, middle_insert_block, location, u384_struct)?;

        // Store the u384 into next input pointer
        middle_insert_block.store(context, location, next_input_ptr, new_input)?;

        // Load accumulator from pointer
        let accumulator =
            middle_insert_block.load(context, location, accumulator_ptr, accumulator.r#type())?;

        middle_insert_block.append_operation(helper.br(1, &[accumulator], location));
    }

    // If is last insert, then:
    {
        let data_type_id = &info.branch_signatures()[0].vars[0].ty;
        let (data_type, data_layout) =
            registry.build_type_with_layout(context, helper, metadata, data_type_id)?;

        // Alloc return data
        let data_ptr =
            helper
                .init_block()
                .alloca1(context, location, data_type, data_layout.align())?;

        // Get pointer to accumulator with alloc and store
        let accumulator_ptr = helper.init_block().alloca1(
            context,
            location,
            accumulator.r#type(),
            accumulator_layout.align(),
        )?;
        last_insert_block.store(context, location, accumulator_ptr, accumulator)?;

        // Get pointer to accumulator input
        let k0 = last_insert_block.const_int(context, location, 0, 64)?;
        let k1 = last_insert_block.const_int(context, location, 1, 64)?;
        let accumulator_input_ptr =
            last_insert_block.append_op_result(llvm::get_element_ptr_dynamic(
                context,
                accumulator_ptr,
                &[k0, k1],
                accumulator.r#type(),
                llvm::r#type::pointer(context, 0),
                location,
            ))?;

        // Copy accumulator input into return data
        let accumulator_input_length = last_insert_block.const_int(
            context,
            location,
            layout_repeat(&get_integer_layout(384), n_inputs - 1)?
                .0
                .size(),
            64,
        )?;
        last_insert_block.memcpy(
            context,
            location,
            accumulator_input_ptr,
            data_ptr,
            accumulator_input_length,
        );

        // Interpret u384 struct (input) as u384 integer
        let u384_struct = entry.arg(1)?;
        let new_input = u384_struct_to_integer(context, last_insert_block, location, u384_struct)?;

        // Get pointer to data end
        let data_end_ptr = last_insert_block.append_op_result(llvm::get_element_ptr(
            context,
            data_ptr,
            DenseI32ArrayAttribute::new(context, &[0, n_inputs as i32 - 1]),
            data_type,
            llvm::r#type::pointer(context, 0),
            location,
        ))?;

        // Store the u384 into next input pointer
        last_insert_block.store(context, location, data_end_ptr, new_input)?;

        // Load data from pointer
        let data = last_insert_block.load(context, location, data_ptr, data_type)?;

        last_insert_block.append_operation(helper.br(0, &[data], location));
    }

    Ok(())
}

/// Generate MLIR operations for the `try_into_circuit_modulus` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_try_into_circuit_modulus<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let modulus = u384_struct_to_integer(context, entry, location, entry.arg(0)?)?;
    let k1 = entry.const_int(context, location, 1, 384)?;

    let is_valid = entry.cmpi(context, arith::CmpiPredicate::Ugt, modulus, k1, location)?;

    entry.append_operation(helper.cond_br(context, is_valid, [0, 1], [&[modulus], &[]], location));

    Ok(())
}

/// Generate MLIR operations for the `get_circuit_descriptor` libfunc.
/// NOOP
#[allow(clippy::too_many_arguments)]
fn build_get_descriptor<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let descriptor_type_id = &info.branch_signatures()[0].vars[0].ty;
    let descriptor_type = registry.build_type(context, helper, metadata, descriptor_type_id)?;

    let unit = entry.append_op_result(llvm::undef(descriptor_type, location))?;

    entry.append_operation(helper.br(0, &[unit], location));

    Ok(())
}

/// Generate MLIR operations for the `eval_circuit` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_eval<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let circuit_info = match registry.get_type(&info.ty)? {
        CoreTypeConcrete::Circuit(CircuitTypeConcrete::Circuit(info)) => &info.circuit_info,
        _ => return Err(SierraAssertError::BadTypeInfo.into()),
    };
    let add_mod = entry.arg(0)?;
    let mul_mod = entry.arg(1)?;
    let circuit_data = entry.arg(3)?;
    let circuit_modulus = entry.arg(4)?;

    // arguments 5 and 6 are used to build the gate 0 (with constant value 1)
    // let zero = entry.argument(5)?;
    // let one = entry.argument(6)?;

    // We multiply the amount of gates evaluated by 4 (the amount of u96s in each gate)
    let add_mod = increment_builtin_counter_by(
        context,
        entry,
        location,
        add_mod,
        circuit_info.add_offsets.len() * 4,
    )?;

    let ([ok_block, err_block], gates) = build_gate_evaluation(
        context,
        entry,
        location,
        helper,
        circuit_info,
        circuit_data,
        circuit_modulus,
    )?;

    // Ok case
    {
        let mul_mod = increment_builtin_counter_by(
            context,
            ok_block,
            location,
            mul_mod,
            circuit_info.mul_offsets.len() * 4,
        )?;

        // Build output struct
        let outputs_type_id = &info.branch_signatures()[0].vars[2].ty;
        let outputs = build_struct_value(
            context,
            registry,
            ok_block,
            location,
            helper,
            metadata,
            outputs_type_id,
            &gates,
        )?;

        ok_block.append_operation(helper.br(0, &[add_mod, mul_mod, outputs], location));
    }

    // Error case
    {
        // We only consider mul gates evaluated before failure
        let mul_mod = {
            let mul_mod_usage = err_block.muli(
                err_block.arg(0)?,
                err_block.const_int(context, location, 4, 64)?,
                location,
            )?;
            err_block.addi(mul_mod, mul_mod_usage, location)
        }?;

        let partial_type_id = &info.branch_signatures()[1].vars[2].ty;
        let partial = err_block.append_op_result(llvm::undef(
            registry.build_type(context, helper, metadata, partial_type_id)?,
            location,
        ))?;
        let failure_type_id = &info.branch_signatures()[1].vars[3].ty;
        let failure = err_block.append_op_result(llvm::undef(
            registry.build_type(context, helper, metadata, failure_type_id)?,
            location,
        ))?;
        err_block.append_operation(helper.br(1, &[add_mod, mul_mod, partial, failure], location));
    }

    Ok(())
}

/// Builds the evaluation of all circuit gates, returning:
/// - An array of two branches, the success block and the error block respectively.
///   - The error block contains the index of the first failure as argument.
/// - A vector of the gate values. In case of failure, not all values are guaranteed to be computed.
///
/// The original Cairo hint evaluates all gates, even in case of failure. This implementation exits on first error, as there is no need for the partial outputs yet.
fn build_gate_evaluation<'ctx, 'this>(
    context: &'this Context,
    mut block: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    circuit_info: &circuit::CircuitInfo,
    circuit_data: Value<'ctx, 'ctx>,
    circuit_modulus: Value<'ctx, 'ctx>,
) -> Result<([&'this Block<'ctx>; 2], Vec<Value<'ctx, 'ctx>>)> {
    // Throughout the evaluation of the circuit we maintain an array of known gate values
    // Initially, it only contains the inputs of the circuit.
    // Unknown values are represented as None

    let mut values = vec![None; 1 + circuit_info.n_inputs + circuit_info.values.len()];
    values[0] = Some(block.const_int(context, location, 1, 384)?);
    for i in 0..circuit_info.n_inputs {
        values[i + 1] = Some(block.extract_value(
            context,
            location,
            circuit_data,
            IntegerType::new(context, 384).into(),
            i,
        )?);
    }

    let err_block = helper.append_block(Block::new(&[(
        IntegerType::new(context, 64).into(),
        location,
    )]));

    let mut add_offsets = circuit_info.add_offsets.iter().peekable();
    let mut mul_offsets = circuit_info.mul_offsets.iter().enumerate();

    // We loop until all gates have been solved
    loop {
        // We iterate the add gate offsets as long as we can
        while let Some(&add_gate_offset) = add_offsets.peek() {
            let lhs_value = values[add_gate_offset.lhs].to_owned();
            let rhs_value = values[add_gate_offset.rhs].to_owned();
            let output_value = values[add_gate_offset.output].to_owned();

            // Depending on the values known at the time, we can deduce if we are dealing with an ADD gate or a SUB gate.
            match (lhs_value, rhs_value, output_value) {
                // ADD: lhs + rhs = out
                (Some(lhs_value), Some(rhs_value), None) => {
                    // Extend to avoid overflow
                    let lhs_value = block.extui(
                        lhs_value,
                        IntegerType::new(context, 384 + 1).into(),
                        location,
                    )?;
                    let rhs_value = block.extui(
                        rhs_value,
                        IntegerType::new(context, 384 + 1).into(),
                        location,
                    )?;
                    let circuit_modulus = block.extui(
                        circuit_modulus,
                        IntegerType::new(context, 384 + 1).into(),
                        location,
                    )?;
                    // value = (lhs_value + rhs_value) % circuit_modulus
                    let value = block.addi(lhs_value, rhs_value, location)?;
                    let value =
                        block.append_op_result(arith::remui(value, circuit_modulus, location))?;
                    // Truncate back
                    let value =
                        block.trunci(value, IntegerType::new(context, 384).into(), location)?;
                    values[add_gate_offset.output] = Some(value);
                }
                // SUB: lhs = out - rhs
                (None, Some(rhs_value), Some(output_value)) => {
                    // Extend to avoid overflow
                    let rhs_value = block.extui(
                        rhs_value,
                        IntegerType::new(context, 384 + 1).into(),
                        location,
                    )?;
                    let output_value = block.extui(
                        output_value,
                        IntegerType::new(context, 384 + 1).into(),
                        location,
                    )?;
                    let circuit_modulus = block.extui(
                        circuit_modulus,
                        IntegerType::new(context, 384 + 1).into(),
                        location,
                    )?;
                    // value = (output_value + circuit_modulus - rhs_value) % circuit_modulus
                    let value = block.addi(output_value, circuit_modulus, location)?;
                    let value = block.append_op_result(arith::subi(value, rhs_value, location))?;
                    let value =
                        block.append_op_result(arith::remui(value, circuit_modulus, location))?;
                    // Truncate back
                    let value =
                        block.trunci(value, IntegerType::new(context, 384).into(), location)?;
                    values[add_gate_offset.lhs] = Some(value);
                }
                // We can't solve this add gate yet, so we break from the loop
                _ => break,
            }

            add_offsets.next();
        }

        // If we can't advance any more with add gate offsets, then we solve the next mul gate offset and go back to the start of the loop (solving add gate offsets).
        if let Some((gate_offset_idx, &circuit::GateOffsets { lhs, rhs, output })) =
            mul_offsets.next()
        {
            let lhs_value = values[lhs].to_owned();
            let rhs_value = values[rhs].to_owned();
            let output_value = values[output].to_owned();

            // Depending on the values known at the time, we can deduce if we are dealing with an MUL gate or a INV gate.
            match (lhs_value, rhs_value, output_value) {
                // MUL: lhs * rhs = out
                (Some(lhs_value), Some(rhs_value), None) => {
                    // Extend to avoid overflow
                    let lhs_value = block.extui(
                        lhs_value,
                        IntegerType::new(context, 384 * 2).into(),
                        location,
                    )?;
                    let rhs_value = block.extui(
                        rhs_value,
                        IntegerType::new(context, 384 * 2).into(),
                        location,
                    )?;
                    let circuit_modulus = block.extui(
                        circuit_modulus,
                        IntegerType::new(context, 384 * 2).into(),
                        location,
                    )?;
                    // value = (lhs_value * rhs_value) % circuit_modulus
                    let value = block.muli(lhs_value, rhs_value, location)?;
                    let value =
                        block.append_op_result(arith::remui(value, circuit_modulus, location))?;
                    // Truncate back
                    let value =
                        block.trunci(value, IntegerType::new(context, 384).into(), location)?;
                    values[output] = Some(value)
                }
                // INV: lhs = 1 / rhs
                (None, Some(rhs_value), Some(_)) => {
                    // Extend to avoid overflow
                    let rhs_value = block.extui(
                        rhs_value,
                        IntegerType::new(context, 384 * 2).into(),
                        location,
                    )?;
                    let circuit_modulus = block.extui(
                        circuit_modulus,
                        IntegerType::new(context, 384 * 2).into(),
                        location,
                    )?;
                    let integer_type = rhs_value.r#type();

                    // Apply egcd to find gcd and inverse
                    let egcd_result_block = build_euclidean_algorithm(
                        context,
                        block,
                        location,
                        helper,
                        rhs_value,
                        circuit_modulus,
                    )?;
                    let gcd = egcd_result_block.arg(0)?;
                    let inverse = egcd_result_block.arg(1)?;
                    block = egcd_result_block;

                    // if the gcd is not 1, then fail (a and b are not coprimes)
                    let one = block.const_int_from_type(context, location, 1, integer_type)?;
                    let gate_offset_idx_value = block.const_int_from_type(
                        context,
                        location,
                        gate_offset_idx,
                        IntegerType::new(context, 64).into(),
                    )?;
                    let has_inverse = block.cmpi(context, CmpiPredicate::Eq, gcd, one, location)?;
                    let has_inverse_block = helper.append_block(Block::new(&[]));
                    block.append_operation(cf::cond_br(
                        context,
                        has_inverse,
                        has_inverse_block,
                        err_block,
                        &[],
                        &[gate_offset_idx_value],
                        location,
                    ));
                    block = has_inverse_block;

                    // if the inverse is negative, then add modulus
                    let zero = block.const_int_from_type(context, location, 0, integer_type)?;
                    let is_negative = block
                        .append_operation(arith::cmpi(
                            context,
                            CmpiPredicate::Slt,
                            inverse,
                            zero,
                            location,
                        ))
                        .result(0)?
                        .into();
                    let wrapped_inverse = block.addi(inverse, circuit_modulus, location)?;
                    let inverse = block.append_op_result(arith::select(
                        is_negative,
                        wrapped_inverse,
                        inverse,
                        location,
                    ))?;

                    // Truncate back
                    let inverse =
                        block.trunci(inverse, IntegerType::new(context, 384).into(), location)?;

                    values[lhs] = Some(inverse);
                }
                // The imposibility to solve this mul gate offset would render the circuit unsolvable
                _ => return Err(SierraAssertError::ImpossibleCircuit.into()),
            }
        } else {
            // If there are no mul gate offsets left, then we have the finished evaluation.
            break;
        }
    }

    // Validate all values have been calculated
    // Should only fail if the circuit is not solvable (bad form)
    let values = values
        .into_iter()
        .skip(1 + circuit_info.n_inputs)
        .collect::<Option<Vec<Value>>>()
        .ok_or(SierraAssertError::ImpossibleCircuit)?;

    Ok(([block, err_block], values))
}

/// Generate MLIR operations for the `circuit_failure_guarantee_verify` libfunc.
/// NOOP
#[allow(clippy::too_many_arguments)]
fn build_failure_guarantee_verify<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let rc = entry.arg(0)?;
    let mul_mod = entry.arg(1)?;
    let rc = increment_builtin_counter_by(context, entry, location, rc, 4)?;

    let mul_mod = increment_builtin_counter_by(context, entry, location, mul_mod, 4)?;

    let guarantee_type_id = &info.branch_signatures()[0].vars[2].ty;
    let guarantee_type = registry.build_type(context, helper, metadata, guarantee_type_id)?;

    let guarantee = entry.append_op_result(llvm::undef(guarantee_type, location))?;

    entry.append_operation(helper.br(0, &[rc, mul_mod, guarantee], location));

    Ok(())
}

/// Generate MLIR operations for the `u96_limbs_less_than_guarantee_verify` libfunc.
/// NOOP
#[allow(clippy::too_many_arguments)]
fn build_u96_limbs_less_than_guarantee_verify<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &ConcreteU96LimbsLessThanGuaranteeVerifyLibfunc,
) -> Result<()> {
    let guarantee_type_id = &info.branch_signatures()[0].vars[0].ty;
    let guarantee_type = registry.build_type(context, helper, metadata, guarantee_type_id)?;

    let guarantee = entry.append_op_result(llvm::undef(guarantee_type, location))?;

    let u96_type_id = &info.branch_signatures()[1].vars[0].ty;
    let u96_type = registry.build_type(context, helper, metadata, u96_type_id)?;

    let u96 = entry.append_op_result(llvm::undef(u96_type, location))?;

    let kfalse = entry.const_int(context, location, 0, 64)?;
    entry.append_operation(helper.cond_br(
        context,
        kfalse,
        [0, 1],
        [&[guarantee], &[u96]],
        location,
    ));

    Ok(())
}

/// Generate MLIR operations for the `u96_single_limb_less_than_guarantee_verify` libfunc.
/// NOOP
#[allow(clippy::too_many_arguments)]
fn build_u96_single_limb_less_than_guarantee_verify<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let u96_type_id = &info.branch_signatures()[0].vars[0].ty;
    let u96_type = registry.build_type(context, helper, metadata, u96_type_id)?;
    let u96 = entry.append_op_result(llvm::undef(u96_type, location))?;

    entry.append_operation(helper.br(0, &[u96], location));

    Ok(())
}

/// Generate MLIR operations for the `get_circuit_output` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_get_output<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &ConcreteGetOutputLibFunc,
) -> Result<()> {
    let circuit_info = match registry.get_type(&info.circuit_ty)? {
        CoreTypeConcrete::Circuit(CircuitTypeConcrete::Circuit(info)) => &info.circuit_info,
        _ => return Err(SierraAssertError::BadTypeInfo.into()),
    };
    let output_type_id = &info.output_ty;

    let output_offset_idx = *circuit_info
        .values
        .get(output_type_id)
        .ok_or(SierraAssertError::BadTypeInfo)?;

    let output_idx = output_offset_idx - circuit_info.n_inputs - 1;

    let outputs = entry.arg(0)?;
    let output_integer = entry.extract_value(
        context,
        location,
        outputs,
        IntegerType::new(context, 384).into(),
        output_idx,
    )?;
    let output_struct = u384_integer_to_struct(context, entry, location, output_integer)?;

    let guarantee_type_id = &info.branch_signatures()[0].vars[1].ty;
    let guarantee_type = registry.build_type(context, helper, metadata, guarantee_type_id)?;
    let guarantee = entry.append_op_result(llvm::undef(guarantee_type, location))?;

    entry.append_operation(helper.br(0, &[output_struct, guarantee], location));

    Ok(())
}

fn u384_struct_to_integer<'a>(
    context: &'a Context,
    block: &'a Block<'a>,
    location: Location<'a>,
    u384_struct: Value<'a, 'a>,
) -> Result<Value<'a, 'a>> {
    let u96_type = IntegerType::new(context, 96).into();

    let limb1 = block.extui(
        block.extract_value(context, location, u384_struct, u96_type, 0)?,
        IntegerType::new(context, 384).into(),
        location,
    )?;

    let limb2 = {
        let limb = block.extui(
            block.extract_value(context, location, u384_struct, u96_type, 1)?,
            IntegerType::new(context, 384).into(),
            location,
        )?;
        let k96 = block.const_int(context, location, 96, 384)?;
        block.shli(limb, k96, location)?
    };

    let limb3 = {
        let limb = block.extui(
            block.extract_value(context, location, u384_struct, u96_type, 2)?,
            IntegerType::new(context, 384).into(),
            location,
        )?;
        let k192 = block.const_int(context, location, 96 * 2, 384)?;
        block.shli(limb, k192, location)?
    };

    let limb4 = {
        let limb = block.extui(
            block.extract_value(context, location, u384_struct, u96_type, 3)?,
            IntegerType::new(context, 384).into(),
            location,
        )?;
        let k288 = block.const_int(context, location, 96 * 3, 384)?;
        block.shli(limb, k288, location)?
    };

    let value = block.append_op_result(arith::ori(limb1, limb2, location))?;
    let value = block.append_op_result(arith::ori(value, limb3, location))?;
    let value = block.append_op_result(arith::ori(value, limb4, location))?;

    Ok(value)
}

fn u384_integer_to_struct<'a>(
    context: &'a Context,
    block: &'a Block<'a>,
    location: Location<'a>,
    integer: Value<'a, 'a>,
) -> Result<Value<'a, 'a>> {
    let u96_type = IntegerType::new(context, 96).into();

    let limb1 = block.trunci(integer, IntegerType::new(context, 96).into(), location)?;
    let limb2 = {
        let k96 = block.const_int(context, location, 96, 384)?;
        let limb = block.shrui(integer, k96, location)?;
        block.trunci(limb, u96_type, location)?
    };
    let limb3 = {
        let k192 = block.const_int(context, location, 96 * 2, 384)?;
        let limb = block.shrui(integer, k192, location)?;
        block.trunci(limb, u96_type, location)?
    };
    let limb4 = {
        let k288 = block.const_int(context, location, 96 * 3, 384)?;
        let limb = block.shrui(integer, k288, location)?;
        block.trunci(limb, u96_type, location)?
    };

    let struct_type = llvm::r#type::r#struct(
        context,
        &[
            IntegerType::new(context, 96).into(),
            IntegerType::new(context, 96).into(),
            IntegerType::new(context, 96).into(),
            IntegerType::new(context, 96).into(),
        ],
        false,
    );
    let struct_value = block.append_op_result(llvm::undef(struct_type, location))?;

    block.insert_values(
        context,
        location,
        struct_value,
        &[limb1, limb2, limb3, limb4],
    )
}

/// The extended euclidean algorithm calculates the greatest common divisor (gcd) of two integers a and b,
/// as well as the bezout coefficients x and y such that ax+by=gcd(a,b)
/// if gcd(a,b) = 1, then x is the modular multiplicative inverse of a modulo b.
/// See https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
///
/// Given two numbers a, b. It returns a block with gcd(a, b) and the bezout coefficient x.
fn build_euclidean_algorithm<'ctx, 'this>(
    context: &'ctx Context,
    block: &'ctx Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    a: Value<'ctx, 'ctx>,
    b: Value<'ctx, 'ctx>,
) -> Result<&'this Block<'ctx>> {
    let integer_type = a.r#type();

    let loop_block = helper.append_block(Block::new(&[
        (integer_type, location),
        (integer_type, location),
        (integer_type, location),
        (integer_type, location),
    ]));
    let end_block = helper.append_block(Block::new(&[
        (integer_type, location),
        (integer_type, location),
    ]));

    // The algorithm egcd works by calculating a series of remainders, each the remainder of dividing the previous two
    // For the initial setup, r0 = b, r1 = a
    // This order is chosen because if we reverse them, then the first iteration will just swap them
    let prev_remainder = b;
    let remainder = a;
    // Similarly we'll calculate another series which starts 0,1,... and from which we will retrieve the modular inverse of a
    let prev_inverse = block.const_int_from_type(context, location, 0, integer_type)?;
    let inverse = block.const_int_from_type(context, location, 1, integer_type)?;
    block.append_operation(cf::br(
        loop_block,
        &[prev_remainder, remainder, prev_inverse, inverse],
        location,
    ));

    // -- Loop body --
    // Arguments are rem_(i-1), rem, inv_(i-1), inv
    let prev_remainder = loop_block.arg(0)?;
    let remainder = loop_block.arg(1)?;
    let prev_inverse = loop_block.arg(2)?;
    let inverse = loop_block.arg(3)?;

    // First calculate q = rem_(i-1)/rem_i, rounded down
    let quotient =
        loop_block.append_op_result(arith::divui(prev_remainder, remainder, location))?;

    // Then r_(i+1) = r_(i-1) - q * r_i, and inv_(i+1) = inv_(i-1) - q * inv_i
    let rem_times_quo = loop_block.muli(remainder, quotient, location)?;
    let inv_times_quo = loop_block.muli(inverse, quotient, location)?;
    let next_remainder =
        loop_block.append_op_result(arith::subi(prev_remainder, rem_times_quo, location))?;
    let next_inverse =
        loop_block.append_op_result(arith::subi(prev_inverse, inv_times_quo, location))?;

    // Check if r_(i+1) is 0
    // If true, then:
    // - r_i is the gcd of a and b
    // - inv_i is the bezout coefficient x

    let zero = loop_block.const_int_from_type(context, location, 0, integer_type)?;
    let next_remainder_eq_zero =
        loop_block.cmpi(context, CmpiPredicate::Eq, next_remainder, zero, location)?;
    loop_block.append_operation(cf::cond_br(
        context,
        next_remainder_eq_zero,
        end_block,
        loop_block,
        &[remainder, inverse],
        &[remainder, next_remainder, inverse, next_inverse],
        location,
    ));

    Ok(end_block)
}

#[cfg(test)]
mod test {

    use crate::{
        utils::{
            felt252_str,
            test::{jit_enum, jit_panic, jit_struct, run_sierra_program},
        },
        values::Value,
    };
    use cairo_lang_sierra::{extensions::utils::Range, ProgramParser};
    use num_bigint::BigUint;
    use num_traits::Num;
    use starknet_types_core::felt::Felt;

    fn u384(limbs: [&str; 4]) -> Value {
        fn u96_range() -> Range {
            Range {
                lower: BigUint::from_str_radix("0", 16).unwrap().into(),
                upper: BigUint::from_str_radix("79228162514264337593543950336", 10)
                    .unwrap()
                    .into(),
            }
        }

        Value::Struct {
            fields: vec![
                Value::BoundedInt {
                    value: Felt::from_hex_unchecked(limbs[0]),
                    range: u96_range(),
                },
                Value::BoundedInt {
                    value: Felt::from_hex_unchecked(limbs[1]),
                    range: u96_range(),
                },
                Value::BoundedInt {
                    value: Felt::from_hex_unchecked(limbs[2]),
                    range: u96_range(),
                },
                Value::BoundedInt {
                    value: Felt::from_hex_unchecked(limbs[3]),
                    range: u96_range(),
                },
            ],
            debug_name: None,
        }
    }

    #[test]
    fn run_add_circuit() {
        // use core::circuit::{
        //     RangeCheck96, AddMod, MulMod, u96, CircuitElement, CircuitInput, circuit_add,
        //     circuit_sub, circuit_mul, circuit_inverse, EvalCircuitTrait, u384,
        //     CircuitOutputsTrait, CircuitModulus, AddInputResultTrait, CircuitInputs,
        // };
        // fn main() -> u384 {
        //     let in1 = CircuitElement::<CircuitInput<0>> {};
        //     let in2 = CircuitElement::<CircuitInput<1>> {};
        //     let add = circuit_add(in1, in2);
        //     let modulus = TryInto::<_, CircuitModulus>::try_into([12, 12, 12, 12]).unwrap();
        //     let outputs = (add,)
        //         .new_inputs()
        //         .next([3, 3, 3, 3])
        //         .next([6, 6, 6, 6])
        //         .done()
        //         .eval(modulus)
        //         .unwrap();
        //     outputs.get_output(add)
        // }
        let program = ProgramParser::new().parse(r#"
            type [3] = BoundedInt<0, 79228162514264337593543950335> [storable: true, drop: true, dup: true, zero_sized: false];
            type [40] = Const<[15], 29721761890975875353235833581453094220424382983267374> [storable: false, drop: false, dup: false, zero_sized: false];
            type [39] = Const<[15], 138583295661092166701491297054433349032460315956105119041111996301516236132> [storable: false, drop: false, dup: false, zero_sized: false];
            type [38] = Const<[15], 30828113188794245257250221355944970489240709081949230> [storable: false, drop: false, dup: false, zero_sized: false];
            type [19] = Struct<ut@core::circuit::u384, [3], [3], [3], [3]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [20] = Struct<ut@Tuple, [19]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [31] = U96LimbsLtGuarantee<1> [storable: true, drop: false, dup: false, zero_sized: false];
            type [30] = U96LimbsLtGuarantee<2> [storable: true, drop: false, dup: false, zero_sized: false];
            type [29] = U96LimbsLtGuarantee<3> [storable: true, drop: false, dup: false, zero_sized: false];
            type [9] = AddModGate<[10], [11]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [28] = U96LimbsLtGuarantee<4> [storable: true, drop: false, dup: false, zero_sized: false];
            type [11] = CircuitInput<1> [storable: false, drop: false, dup: false, zero_sized: true];
            type [10] = CircuitInput<0> [storable: false, drop: false, dup: false, zero_sized: true];
            type [27] = CircuitFailureGuarantee [storable: true, drop: false, dup: false, zero_sized: false];
            type [26] = CircuitPartialOutputs<[7]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [25] = CircuitOutputs<[7]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [37] = Const<[24], 1> [storable: false, drop: false, dup: false, zero_sized: false];
            type [24] = BoundedInt<1, 1> [storable: true, drop: true, dup: true, zero_sized: false];
            type [36] = Const<[23], 0> [storable: false, drop: false, dup: false, zero_sized: false];
            type [23] = BoundedInt<0, 0> [storable: true, drop: true, dup: true, zero_sized: false];
            type [22] = CircuitDescriptor<[7]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [35] = Const<[3], 6> [storable: false, drop: false, dup: false, zero_sized: false];
            type [1] = MulMod [storable: true, drop: false, dup: false, zero_sized: false];
            type [0] = AddMod [storable: true, drop: false, dup: false, zero_sized: false];
            type [17] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
            type [16] = Array<[15]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [18] = Struct<ut@Tuple, [17], [16]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [21] = Enum<ut@core::panics::PanicResult::<(core::circuit::u384,)>, [20], [18]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [34] = Const<[15], 26913677086973030051406221357623718750637972950955665348321109348> [storable: false, drop: false, dup: false, zero_sized: false];
            type [15] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [14] = CircuitData<[7]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [12] = U96Guarantee [storable: true, drop: false, dup: false, zero_sized: false];
            type [13] = Struct<ut@Tuple, [12], [12], [12], [12]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [33] = Const<[3], 3> [storable: false, drop: false, dup: false, zero_sized: false];
            type [7] = Circuit<[6]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [6] = Struct<ut@Tuple, [9]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [8] = CircuitInputAccumulator<[7]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [2] = RangeCheck96 [storable: true, drop: false, dup: false, zero_sized: false];
            type [5] = CircuitModulus [storable: true, drop: true, dup: true, zero_sized: false];
            type [4] = Struct<ut@Tuple, [3], [3], [3], [3]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [32] = Const<[3], 12> [storable: false, drop: false, dup: false, zero_sized: false];

            libfunc [22] = const_as_immediate<[32]>;
            libfunc [21] = struct_construct<[4]>;
            libfunc [37] = store_temp<[4]>;
            libfunc [20] = try_into_circuit_modulus;
            libfunc [23] = branch_align;
            libfunc [19] = init_circuit_data<[7]>;
            libfunc [24] = const_as_immediate<[33]>;
            libfunc [18] = into_u96_guarantee<[3]>;
            libfunc [17] = struct_construct<[13]>;
            libfunc [38] = store_temp<[8]>;
            libfunc [39] = store_temp<[13]>;
            libfunc [40] = store_temp<[2]>;
            libfunc [16] = add_circuit_input<[7]>;
            libfunc [25] = drop<[14]>;
            libfunc [26] = drop<[5]>;
            libfunc [4] = array_new<[15]>;
            libfunc [27] = const_as_immediate<[34]>;
            libfunc [41] = store_temp<[15]>;
            libfunc [3] = array_append<[15]>;
            libfunc [2] = struct_construct<[17]>;
            libfunc [1] = struct_construct<[18]>;
            libfunc [0] = enum_init<[21], 1>;
            libfunc [42] = store_temp<[0]>;
            libfunc [43] = store_temp<[1]>;
            libfunc [44] = store_temp<[21]>;
            libfunc [28] = const_as_immediate<[35]>;
            libfunc [15] = get_circuit_descriptor<[7]>;
            libfunc [29] = const_as_immediate<[36]>;
            libfunc [30] = const_as_immediate<[37]>;
            libfunc [45] = store_temp<[23]>;
            libfunc [46] = store_temp<[24]>;
            libfunc [14] = eval_circuit<[7]>;
            libfunc [13] = get_circuit_output<[7], [9]>;
            libfunc [9] = u96_limbs_less_than_guarantee_verify<4>;
            libfunc [8] = u96_limbs_less_than_guarantee_verify<3>;
            libfunc [7] = u96_limbs_less_than_guarantee_verify<2>;
            libfunc [6] = u96_single_limb_less_than_guarantee_verify;
            libfunc [47] = store_temp<[12]>;
            libfunc [31] = jump;
            libfunc [5] = u96_guarantee_verify;
            libfunc [12] = struct_construct<[20]>;
            libfunc [11] = enum_init<[21], 0>;
            libfunc [32] = drop<[26]>;
            libfunc [33] = const_as_immediate<[38]>;
            libfunc [10] = circuit_failure_guarantee_verify;
            libfunc [48] = store_temp<[16]>;
            libfunc [34] = drop<[8]>;
            libfunc [35] = const_as_immediate<[39]>;
            libfunc [36] = const_as_immediate<[40]>;

            [22]() -> ([3]); // 0
            [22]() -> ([4]); // 1
            [22]() -> ([5]); // 2
            [22]() -> ([6]); // 3
            [21]([3], [4], [5], [6]) -> ([7]); // 4
            [37]([7]) -> ([7]); // 5
            [20]([7]) { fallthrough([8]) 142() }; // 6
            [23]() -> (); // 7
            [19]([2]) -> ([9], [10]); // 8
            [24]() -> ([11]); // 9
            [18]([11]) -> ([12]); // 10
            [24]() -> ([13]); // 11
            [18]([13]) -> ([14]); // 12
            [24]() -> ([15]); // 13
            [18]([15]) -> ([16]); // 14
            [24]() -> ([17]); // 15
            [18]([17]) -> ([18]); // 16
            [17]([12], [14], [16], [18]) -> ([19]); // 17
            [38]([10]) -> ([10]); // 18
            [39]([19]) -> ([19]); // 19
            [40]([9]) -> ([9]); // 20
            [16]([10], [19]) { fallthrough([20]) 37([21]) }; // 21
            [23]() -> (); // 22
            [25]([20]) -> (); // 23
            [26]([8]) -> (); // 24
            [4]() -> ([22]); // 25
            [27]() -> ([23]); // 26
            [41]([23]) -> ([23]); // 27
            [3]([22], [23]) -> ([24]); // 28
            [2]() -> ([25]); // 29
            [1]([25], [24]) -> ([26]); // 30
            [0]([26]) -> ([27]); // 31
            [42]([0]) -> ([0]); // 32
            [43]([1]) -> ([1]); // 33
            [40]([9]) -> ([9]); // 34
            [44]([27]) -> ([27]); // 35
            return([0], [1], [9], [27]); // 36
            [23]() -> (); // 37
            [28]() -> ([28]); // 38
            [18]([28]) -> ([29]); // 39
            [28]() -> ([30]); // 40
            [18]([30]) -> ([31]); // 41
            [28]() -> ([32]); // 42
            [18]([32]) -> ([33]); // 43
            [28]() -> ([34]); // 44
            [18]([34]) -> ([35]); // 45
            [17]([29], [31], [33], [35]) -> ([36]); // 46
            [39]([36]) -> ([36]); // 47
            [16]([21], [36]) { fallthrough([37]) 127([38]) }; // 48
            [23]() -> (); // 49
            [15]() -> ([39]); // 50
            [29]() -> ([40]); // 51
            [30]() -> ([41]); // 52
            [45]([40]) -> ([40]); // 53
            [46]([41]) -> ([41]); // 54
            [14]([0], [1], [39], [37], [8], [40], [41]) { fallthrough([42], [43], [44]) 85([45], [46], [47], [48]) }; // 55
            [23]() -> (); // 56
            [13]([44]) -> ([49], [50]); // 57
            [42]([42]) -> ([42]); // 58
            [43]([43]) -> ([43]); // 59
            [9]([50]) { fallthrough([51]) 75([52]) }; // 60
            [23]() -> (); // 61
            [8]([51]) { fallthrough([53]) 72([54]) }; // 62
            [23]() -> (); // 63
            [7]([53]) { fallthrough([55]) 69([56]) }; // 64
            [23]() -> (); // 65
            [6]([55]) -> ([57]); // 66
            [47]([57]) -> ([58]); // 67
            [31]() { 77() }; // 68
            [23]() -> (); // 69
            [47]([56]) -> ([58]); // 70
            [31]() { 77() }; // 71
            [23]() -> (); // 72
            [47]([54]) -> ([58]); // 73
            [31]() { 77() }; // 74
            [23]() -> (); // 75
            [47]([52]) -> ([58]); // 76
            [5]([9], [58]) -> ([59]); // 77
            [12]([49]) -> ([60]); // 78
            [11]([60]) -> ([61]); // 79
            [42]([42]) -> ([42]); // 80
            [43]([43]) -> ([43]); // 81
            [40]([59]) -> ([59]); // 82
            [44]([61]) -> ([61]); // 83
            return([42], [43], [59], [61]); // 84
            [23]() -> (); // 85
            [32]([47]) -> (); // 86
            [4]() -> ([62]); // 87
            [33]() -> ([63]); // 88
            [41]([63]) -> ([63]); // 89
            [3]([62], [63]) -> ([64]); // 90
            [29]() -> ([65]); // 91
            [30]() -> ([66]); // 92
            [43]([46]) -> ([46]); // 93
            [45]([65]) -> ([65]); // 94
            [46]([66]) -> ([66]); // 95
            [10]([9], [46], [48], [65], [66]) -> ([67], [68], [69]); // 96
            [42]([45]) -> ([45]); // 97
            [48]([64]) -> ([64]); // 98
            [40]([67]) -> ([67]); // 99
            [43]([68]) -> ([68]); // 100
            [9]([69]) { fallthrough([70]) 116([71]) }; // 101
            [23]() -> (); // 102
            [8]([70]) { fallthrough([72]) 113([73]) }; // 103
            [23]() -> (); // 104
            [7]([72]) { fallthrough([74]) 110([75]) }; // 105
            [23]() -> (); // 106
            [6]([74]) -> ([76]); // 107
            [47]([76]) -> ([77]); // 108
            [31]() { 118() }; // 109
            [23]() -> (); // 110
            [47]([75]) -> ([77]); // 111
            [31]() { 118() }; // 112
            [23]() -> (); // 113
            [47]([73]) -> ([77]); // 114
            [31]() { 118() }; // 115
            [23]() -> (); // 116
            [47]([71]) -> ([77]); // 117
            [5]([67], [77]) -> ([78]); // 118
            [2]() -> ([79]); // 119
            [1]([79], [64]) -> ([80]); // 120
            [0]([80]) -> ([81]); // 121
            [42]([45]) -> ([45]); // 122
            [43]([68]) -> ([68]); // 123
            [40]([78]) -> ([78]); // 124
            [44]([81]) -> ([81]); // 125
            return([45], [68], [78], [81]); // 126
            [23]() -> (); // 127
            [34]([38]) -> (); // 128
            [26]([8]) -> (); // 129
            [4]() -> ([82]); // 130
            [35]() -> ([83]); // 131
            [41]([83]) -> ([83]); // 132
            [3]([82], [83]) -> ([84]); // 133
            [2]() -> ([85]); // 134
            [1]([85], [84]) -> ([86]); // 135
            [0]([86]) -> ([87]); // 136
            [42]([0]) -> ([0]); // 137
            [43]([1]) -> ([1]); // 138
            [40]([9]) -> ([9]); // 139
            [44]([87]) -> ([87]); // 140
            return([0], [1], [9], [87]); // 141
            [23]() -> (); // 142
            [4]() -> ([88]); // 143
            [36]() -> ([89]); // 144
            [41]([89]) -> ([89]); // 145
            [3]([88], [89]) -> ([90]); // 146
            [2]() -> ([91]); // 147
            [1]([91], [90]) -> ([92]); // 148
            [0]([92]) -> ([93]); // 149
            [42]([0]) -> ([0]); // 150
            [43]([1]) -> ([1]); // 151
            [40]([2]) -> ([2]); // 152
            [44]([93]) -> ([93]); // 153
            return([0], [1], [2], [93]); // 154

            [0]@0([0]: [0], [1]: [1], [2]: [2]) -> ([0], [1], [2], [21]);
        "#).map_err(|e| e.to_string()).unwrap();

        let return_value = run_sierra_program(&program, &[]).return_value;

        assert_eq!(
            jit_enum!(0, jit_struct!(u384(["0x9", "0x9", "0x9", "0x9"]))),
            return_value
        );
    }

    #[test]
    fn run_sub_circuit() {
        // use core::circuit::{
        //     RangeCheck96, AddMod, MulMod, u96, CircuitElement, CircuitInput, circuit_add,
        //     circuit_sub, circuit_mul, circuit_inverse, EvalCircuitTrait, u384,
        //     CircuitOutputsTrait, CircuitModulus, AddInputResultTrait, CircuitInputs,
        // };
        // fn main() -> u384 {
        //     let in1 = CircuitElement::<CircuitInput<0>> {};
        //     let in2 = CircuitElement::<CircuitInput<1>> {};
        //     let mul = circuit_sub(in1, in2);
        //     let modulus = TryInto::<_, CircuitModulus>::try_into([12, 12, 12, 12]).unwrap();
        //     let outputs = (mul,)
        //         .new_inputs()
        //         .next([6, 6, 6, 6])
        //         .next([3, 3, 3, 3])
        //         .done()
        //         .eval(modulus)
        //         .unwrap();
        //     outputs.get_output(mul)
        // }
        let program = ProgramParser::new().parse(r#"
            type [3] = BoundedInt<0, 79228162514264337593543950335> [storable: true, drop: true, dup: true, zero_sized: false];
            type [40] = Const<[15], 29721761890975875353235833581453094220424382983267374> [storable: false, drop: false, dup: false, zero_sized: false];
            type [39] = Const<[15], 138583295661092166701491297054433349032460315956105119041111996301516236132> [storable: false, drop: false, dup: false, zero_sized: false];
            type [38] = Const<[15], 30828113188794245257250221355944970489240709081949230> [storable: false, drop: false, dup: false, zero_sized: false];
            type [19] = Struct<ut@core::circuit::u384, [3], [3], [3], [3]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [20] = Struct<ut@Tuple, [19]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [31] = U96LimbsLtGuarantee<1> [storable: true, drop: false, dup: false, zero_sized: false];
            type [30] = U96LimbsLtGuarantee<2> [storable: true, drop: false, dup: false, zero_sized: false];
            type [29] = U96LimbsLtGuarantee<3> [storable: true, drop: false, dup: false, zero_sized: false];
            type [9] = SubModGate<[10], [11]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [28] = U96LimbsLtGuarantee<4> [storable: true, drop: false, dup: false, zero_sized: false];
            type [11] = CircuitInput<1> [storable: false, drop: false, dup: false, zero_sized: true];
            type [10] = CircuitInput<0> [storable: false, drop: false, dup: false, zero_sized: true];
            type [27] = CircuitFailureGuarantee [storable: true, drop: false, dup: false, zero_sized: false];
            type [26] = CircuitPartialOutputs<[7]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [25] = CircuitOutputs<[7]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [37] = Const<[24], 1> [storable: false, drop: false, dup: false, zero_sized: false];
            type [24] = BoundedInt<1, 1> [storable: true, drop: true, dup: true, zero_sized: false];
            type [36] = Const<[23], 0> [storable: false, drop: false, dup: false, zero_sized: false];
            type [23] = BoundedInt<0, 0> [storable: true, drop: true, dup: true, zero_sized: false];
            type [22] = CircuitDescriptor<[7]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [35] = Const<[3], 3> [storable: false, drop: false, dup: false, zero_sized: false];
            type [1] = MulMod [storable: true, drop: false, dup: false, zero_sized: false];
            type [0] = AddMod [storable: true, drop: false, dup: false, zero_sized: false];
            type [17] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
            type [16] = Array<[15]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [18] = Struct<ut@Tuple, [17], [16]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [21] = Enum<ut@core::panics::PanicResult::<(core::circuit::u384,)>, [20], [18]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [34] = Const<[15], 26913677086973030051406221357623718750637972950955665348321109348> [storable: false, drop: false, dup: false, zero_sized: false];
            type [15] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [14] = CircuitData<[7]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [12] = U96Guarantee [storable: true, drop: false, dup: false, zero_sized: false];
            type [13] = Struct<ut@Tuple, [12], [12], [12], [12]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [33] = Const<[3], 6> [storable: false, drop: false, dup: false, zero_sized: false];
            type [7] = Circuit<[6]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [6] = Struct<ut@Tuple, [9]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [8] = CircuitInputAccumulator<[7]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [2] = RangeCheck96 [storable: true, drop: false, dup: false, zero_sized: false];
            type [5] = CircuitModulus [storable: true, drop: true, dup: true, zero_sized: false];
            type [4] = Struct<ut@Tuple, [3], [3], [3], [3]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [32] = Const<[3], 12> [storable: false, drop: false, dup: false, zero_sized: false];

            libfunc [22] = const_as_immediate<[32]>;
            libfunc [21] = struct_construct<[4]>;
            libfunc [37] = store_temp<[4]>;
            libfunc [20] = try_into_circuit_modulus;
            libfunc [23] = branch_align;
            libfunc [19] = init_circuit_data<[7]>;
            libfunc [24] = const_as_immediate<[33]>;
            libfunc [18] = into_u96_guarantee<[3]>;
            libfunc [17] = struct_construct<[13]>;
            libfunc [38] = store_temp<[8]>;
            libfunc [39] = store_temp<[13]>;
            libfunc [40] = store_temp<[2]>;
            libfunc [16] = add_circuit_input<[7]>;
            libfunc [25] = drop<[14]>;
            libfunc [26] = drop<[5]>;
            libfunc [4] = array_new<[15]>;
            libfunc [27] = const_as_immediate<[34]>;
            libfunc [41] = store_temp<[15]>;
            libfunc [3] = array_append<[15]>;
            libfunc [2] = struct_construct<[17]>;
            libfunc [1] = struct_construct<[18]>;
            libfunc [0] = enum_init<[21], 1>;
            libfunc [42] = store_temp<[0]>;
            libfunc [43] = store_temp<[1]>;
            libfunc [44] = store_temp<[21]>;
            libfunc [28] = const_as_immediate<[35]>;
            libfunc [15] = get_circuit_descriptor<[7]>;
            libfunc [29] = const_as_immediate<[36]>;
            libfunc [30] = const_as_immediate<[37]>;
            libfunc [45] = store_temp<[23]>;
            libfunc [46] = store_temp<[24]>;
            libfunc [14] = eval_circuit<[7]>;
            libfunc [13] = get_circuit_output<[7], [9]>;
            libfunc [9] = u96_limbs_less_than_guarantee_verify<4>;
            libfunc [8] = u96_limbs_less_than_guarantee_verify<3>;
            libfunc [7] = u96_limbs_less_than_guarantee_verify<2>;
            libfunc [6] = u96_single_limb_less_than_guarantee_verify;
            libfunc [47] = store_temp<[12]>;
            libfunc [31] = jump;
            libfunc [5] = u96_guarantee_verify;
            libfunc [12] = struct_construct<[20]>;
            libfunc [11] = enum_init<[21], 0>;
            libfunc [32] = drop<[26]>;
            libfunc [33] = const_as_immediate<[38]>;
            libfunc [10] = circuit_failure_guarantee_verify;
            libfunc [48] = store_temp<[16]>;
            libfunc [34] = drop<[8]>;
            libfunc [35] = const_as_immediate<[39]>;
            libfunc [36] = const_as_immediate<[40]>;

            [22]() -> ([3]); // 0
            [22]() -> ([4]); // 1
            [22]() -> ([5]); // 2
            [22]() -> ([6]); // 3
            [21]([3], [4], [5], [6]) -> ([7]); // 4
            [37]([7]) -> ([7]); // 5
            [20]([7]) { fallthrough([8]) 142() }; // 6
            [23]() -> (); // 7
            [19]([2]) -> ([9], [10]); // 8
            [24]() -> ([11]); // 9
            [18]([11]) -> ([12]); // 10
            [24]() -> ([13]); // 11
            [18]([13]) -> ([14]); // 12
            [24]() -> ([15]); // 13
            [18]([15]) -> ([16]); // 14
            [24]() -> ([17]); // 15
            [18]([17]) -> ([18]); // 16
            [17]([12], [14], [16], [18]) -> ([19]); // 17
            [38]([10]) -> ([10]); // 18
            [39]([19]) -> ([19]); // 19
            [40]([9]) -> ([9]); // 20
            [16]([10], [19]) { fallthrough([20]) 37([21]) }; // 21
            [23]() -> (); // 22
            [25]([20]) -> (); // 23
            [26]([8]) -> (); // 24
            [4]() -> ([22]); // 25
            [27]() -> ([23]); // 26
            [41]([23]) -> ([23]); // 27
            [3]([22], [23]) -> ([24]); // 28
            [2]() -> ([25]); // 29
            [1]([25], [24]) -> ([26]); // 30
            [0]([26]) -> ([27]); // 31
            [42]([0]) -> ([0]); // 32
            [43]([1]) -> ([1]); // 33
            [40]([9]) -> ([9]); // 34
            [44]([27]) -> ([27]); // 35
            return([0], [1], [9], [27]); // 36
            [23]() -> (); // 37
            [28]() -> ([28]); // 38
            [18]([28]) -> ([29]); // 39
            [28]() -> ([30]); // 40
            [18]([30]) -> ([31]); // 41
            [28]() -> ([32]); // 42
            [18]([32]) -> ([33]); // 43
            [28]() -> ([34]); // 44
            [18]([34]) -> ([35]); // 45
            [17]([29], [31], [33], [35]) -> ([36]); // 46
            [39]([36]) -> ([36]); // 47
            [16]([21], [36]) { fallthrough([37]) 127([38]) }; // 48
            [23]() -> (); // 49
            [15]() -> ([39]); // 50
            [29]() -> ([40]); // 51
            [30]() -> ([41]); // 52
            [45]([40]) -> ([40]); // 53
            [46]([41]) -> ([41]); // 54
            [14]([0], [1], [39], [37], [8], [40], [41]) { fallthrough([42], [43], [44]) 85([45], [46], [47], [48]) }; // 55
            [23]() -> (); // 56
            [13]([44]) -> ([49], [50]); // 57
            [42]([42]) -> ([42]); // 58
            [43]([43]) -> ([43]); // 59
            [9]([50]) { fallthrough([51]) 75([52]) }; // 60
            [23]() -> (); // 61
            [8]([51]) { fallthrough([53]) 72([54]) }; // 62
            [23]() -> (); // 63
            [7]([53]) { fallthrough([55]) 69([56]) }; // 64
            [23]() -> (); // 65
            [6]([55]) -> ([57]); // 66
            [47]([57]) -> ([58]); // 67
            [31]() { 77() }; // 68
            [23]() -> (); // 69
            [47]([56]) -> ([58]); // 70
            [31]() { 77() }; // 71
            [23]() -> (); // 72
            [47]([54]) -> ([58]); // 73
            [31]() { 77() }; // 74
            [23]() -> (); // 75
            [47]([52]) -> ([58]); // 76
            [5]([9], [58]) -> ([59]); // 77
            [12]([49]) -> ([60]); // 78
            [11]([60]) -> ([61]); // 79
            [42]([42]) -> ([42]); // 80
            [43]([43]) -> ([43]); // 81
            [40]([59]) -> ([59]); // 82
            [44]([61]) -> ([61]); // 83
            return([42], [43], [59], [61]); // 84
            [23]() -> (); // 85
            [32]([47]) -> (); // 86
            [4]() -> ([62]); // 87
            [33]() -> ([63]); // 88
            [41]([63]) -> ([63]); // 89
            [3]([62], [63]) -> ([64]); // 90
            [29]() -> ([65]); // 91
            [30]() -> ([66]); // 92
            [43]([46]) -> ([46]); // 93
            [45]([65]) -> ([65]); // 94
            [46]([66]) -> ([66]); // 95
            [10]([9], [46], [48], [65], [66]) -> ([67], [68], [69]); // 96
            [42]([45]) -> ([45]); // 97
            [48]([64]) -> ([64]); // 98
            [40]([67]) -> ([67]); // 99
            [43]([68]) -> ([68]); // 100
            [9]([69]) { fallthrough([70]) 116([71]) }; // 101
            [23]() -> (); // 102
            [8]([70]) { fallthrough([72]) 113([73]) }; // 103
            [23]() -> (); // 104
            [7]([72]) { fallthrough([74]) 110([75]) }; // 105
            [23]() -> (); // 106
            [6]([74]) -> ([76]); // 107
            [47]([76]) -> ([77]); // 108
            [31]() { 118() }; // 109
            [23]() -> (); // 110
            [47]([75]) -> ([77]); // 111
            [31]() { 118() }; // 112
            [23]() -> (); // 113
            [47]([73]) -> ([77]); // 114
            [31]() { 118() }; // 115
            [23]() -> (); // 116
            [47]([71]) -> ([77]); // 117
            [5]([67], [77]) -> ([78]); // 118
            [2]() -> ([79]); // 119
            [1]([79], [64]) -> ([80]); // 120
            [0]([80]) -> ([81]); // 121
            [42]([45]) -> ([45]); // 122
            [43]([68]) -> ([68]); // 123
            [40]([78]) -> ([78]); // 124
            [44]([81]) -> ([81]); // 125
            return([45], [68], [78], [81]); // 126
            [23]() -> (); // 127
            [34]([38]) -> (); // 128
            [26]([8]) -> (); // 129
            [4]() -> ([82]); // 130
            [35]() -> ([83]); // 131
            [41]([83]) -> ([83]); // 132
            [3]([82], [83]) -> ([84]); // 133
            [2]() -> ([85]); // 134
            [1]([85], [84]) -> ([86]); // 135
            [0]([86]) -> ([87]); // 136
            [42]([0]) -> ([0]); // 137
            [43]([1]) -> ([1]); // 138
            [40]([9]) -> ([9]); // 139
            [44]([87]) -> ([87]); // 140
            return([0], [1], [9], [87]); // 141
            [23]() -> (); // 142
            [4]() -> ([88]); // 143
            [36]() -> ([89]); // 144
            [41]([89]) -> ([89]); // 145
            [3]([88], [89]) -> ([90]); // 146
            [2]() -> ([91]); // 147
            [1]([91], [90]) -> ([92]); // 148
            [0]([92]) -> ([93]); // 149
            [42]([0]) -> ([0]); // 150
            [43]([1]) -> ([1]); // 151
            [40]([2]) -> ([2]); // 152
            [44]([93]) -> ([93]); // 153
            return([0], [1], [2], [93]); // 154

            [0]@0([0]: [0], [1]: [1], [2]: [2]) -> ([0], [1], [2], [21]);
        "#).map_err(|e| e.to_string()).unwrap();

        let return_value = run_sierra_program(&program, &[]).return_value;

        assert_eq!(
            jit_enum!(0, jit_struct!(u384(["0x3", "0x3", "0x3", "0x3"]))),
            return_value
        );
    }

    #[test]
    fn run_mul_circuit() {
        // use core::circuit::{
        //     RangeCheck96, AddMod, MulMod, u96, CircuitElement, CircuitInput, circuit_add,
        //     circuit_sub, circuit_mul, circuit_inverse, EvalCircuitTrait, u384,
        //     CircuitOutputsTrait, CircuitModulus, AddInputResultTrait, CircuitInputs,
        // };
        // fn main() -> u384 {
        //     let in1 = CircuitElement::<CircuitInput<0>> {};
        //     let in2 = CircuitElement::<CircuitInput<1>> {};
        //     let mul = circuit_mul(in1, in2);
        //     let modulus = TryInto::<_, CircuitModulus>::try_into([12, 12, 12, 12]).unwrap();
        //     let outputs = (mul,)
        //         .new_inputs()
        //         .next([3, 0, 0, 0])
        //         .next([3, 3, 3, 3])
        //         .done()
        //         .eval(modulus)
        //         .unwrap();
        //     outputs.get_output(mul)
        // }
        let program = ProgramParser::new().parse(r#"
            type [3] = BoundedInt<0, 79228162514264337593543950335> [storable: true, drop: true, dup: true, zero_sized: false];
            type [40] = Const<[15], 29721761890975875353235833581453094220424382983267374> [storable: false, drop: false, dup: false, zero_sized: false];
            type [39] = Const<[15], 138583295661092166701491297054433349032460315956105119041111996301516236132> [storable: false, drop: false, dup: false, zero_sized: false];
            type [38] = Const<[15], 30828113188794245257250221355944970489240709081949230> [storable: false, drop: false, dup: false, zero_sized: false];
            type [19] = Struct<ut@core::circuit::u384, [3], [3], [3], [3]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [20] = Struct<ut@Tuple, [19]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [31] = U96LimbsLtGuarantee<1> [storable: true, drop: false, dup: false, zero_sized: false];
            type [30] = U96LimbsLtGuarantee<2> [storable: true, drop: false, dup: false, zero_sized: false];
            type [29] = U96LimbsLtGuarantee<3> [storable: true, drop: false, dup: false, zero_sized: false];
            type [9] = MulModGate<[10], [11]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [28] = U96LimbsLtGuarantee<4> [storable: true, drop: false, dup: false, zero_sized: false];
            type [11] = CircuitInput<1> [storable: false, drop: false, dup: false, zero_sized: true];
            type [10] = CircuitInput<0> [storable: false, drop: false, dup: false, zero_sized: true];
            type [27] = CircuitFailureGuarantee [storable: true, drop: false, dup: false, zero_sized: false];
            type [26] = CircuitPartialOutputs<[7]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [25] = CircuitOutputs<[7]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [37] = Const<[24], 1> [storable: false, drop: false, dup: false, zero_sized: false];
            type [24] = BoundedInt<1, 1> [storable: true, drop: true, dup: true, zero_sized: false];
            type [36] = Const<[23], 0> [storable: false, drop: false, dup: false, zero_sized: false];
            type [23] = BoundedInt<0, 0> [storable: true, drop: true, dup: true, zero_sized: false];
            type [22] = CircuitDescriptor<[7]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = MulMod [storable: true, drop: false, dup: false, zero_sized: false];
            type [0] = AddMod [storable: true, drop: false, dup: false, zero_sized: false];
            type [17] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
            type [16] = Array<[15]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [18] = Struct<ut@Tuple, [17], [16]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [21] = Enum<ut@core::panics::PanicResult::<(core::circuit::u384,)>, [20], [18]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [35] = Const<[15], 26913677086973030051406221357623718750637972950955665348321109348> [storable: false, drop: false, dup: false, zero_sized: false];
            type [15] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [14] = CircuitData<[7]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [12] = U96Guarantee [storable: true, drop: false, dup: false, zero_sized: false];
            type [13] = Struct<ut@Tuple, [12], [12], [12], [12]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [34] = Const<[3], 0> [storable: false, drop: false, dup: false, zero_sized: false];
            type [33] = Const<[3], 3> [storable: false, drop: false, dup: false, zero_sized: false];
            type [7] = Circuit<[6]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [6] = Struct<ut@Tuple, [9]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [8] = CircuitInputAccumulator<[7]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [2] = RangeCheck96 [storable: true, drop: false, dup: false, zero_sized: false];
            type [5] = CircuitModulus [storable: true, drop: true, dup: true, zero_sized: false];
            type [4] = Struct<ut@Tuple, [3], [3], [3], [3]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [32] = Const<[3], 12> [storable: false, drop: false, dup: false, zero_sized: false];

            libfunc [22] = const_as_immediate<[32]>;
            libfunc [21] = struct_construct<[4]>;
            libfunc [37] = store_temp<[4]>;
            libfunc [20] = try_into_circuit_modulus;
            libfunc [23] = branch_align;
            libfunc [19] = init_circuit_data<[7]>;
            libfunc [24] = const_as_immediate<[33]>;
            libfunc [18] = into_u96_guarantee<[3]>;
            libfunc [25] = const_as_immediate<[34]>;
            libfunc [17] = struct_construct<[13]>;
            libfunc [38] = store_temp<[8]>;
            libfunc [39] = store_temp<[13]>;
            libfunc [40] = store_temp<[2]>;
            libfunc [16] = add_circuit_input<[7]>;
            libfunc [26] = drop<[14]>;
            libfunc [27] = drop<[5]>;
            libfunc [4] = array_new<[15]>;
            libfunc [28] = const_as_immediate<[35]>;
            libfunc [41] = store_temp<[15]>;
            libfunc [3] = array_append<[15]>;
            libfunc [2] = struct_construct<[17]>;
            libfunc [1] = struct_construct<[18]>;
            libfunc [0] = enum_init<[21], 1>;
            libfunc [42] = store_temp<[0]>;
            libfunc [43] = store_temp<[1]>;
            libfunc [44] = store_temp<[21]>;
            libfunc [15] = get_circuit_descriptor<[7]>;
            libfunc [29] = const_as_immediate<[36]>;
            libfunc [30] = const_as_immediate<[37]>;
            libfunc [45] = store_temp<[23]>;
            libfunc [46] = store_temp<[24]>;
            libfunc [14] = eval_circuit<[7]>;
            libfunc [13] = get_circuit_output<[7], [9]>;
            libfunc [9] = u96_limbs_less_than_guarantee_verify<4>;
            libfunc [8] = u96_limbs_less_than_guarantee_verify<3>;
            libfunc [7] = u96_limbs_less_than_guarantee_verify<2>;
            libfunc [6] = u96_single_limb_less_than_guarantee_verify;
            libfunc [47] = store_temp<[12]>;
            libfunc [31] = jump;
            libfunc [5] = u96_guarantee_verify;
            libfunc [12] = struct_construct<[20]>;
            libfunc [11] = enum_init<[21], 0>;
            libfunc [32] = drop<[26]>;
            libfunc [33] = const_as_immediate<[38]>;
            libfunc [10] = circuit_failure_guarantee_verify;
            libfunc [48] = store_temp<[16]>;
            libfunc [34] = drop<[8]>;
            libfunc [35] = const_as_immediate<[39]>;
            libfunc [36] = const_as_immediate<[40]>;

            [22]() -> ([3]); // 0
            [22]() -> ([4]); // 1
            [22]() -> ([5]); // 2
            [22]() -> ([6]); // 3
            [21]([3], [4], [5], [6]) -> ([7]); // 4
            [37]([7]) -> ([7]); // 5
            [20]([7]) { fallthrough([8]) 142() }; // 6
            [23]() -> (); // 7
            [19]([2]) -> ([9], [10]); // 8
            [24]() -> ([11]); // 9
            [18]([11]) -> ([12]); // 10
            [25]() -> ([13]); // 11
            [18]([13]) -> ([14]); // 12
            [25]() -> ([15]); // 13
            [18]([15]) -> ([16]); // 14
            [25]() -> ([17]); // 15
            [18]([17]) -> ([18]); // 16
            [17]([12], [14], [16], [18]) -> ([19]); // 17
            [38]([10]) -> ([10]); // 18
            [39]([19]) -> ([19]); // 19
            [40]([9]) -> ([9]); // 20
            [16]([10], [19]) { fallthrough([20]) 37([21]) }; // 21
            [23]() -> (); // 22
            [26]([20]) -> (); // 23
            [27]([8]) -> (); // 24
            [4]() -> ([22]); // 25
            [28]() -> ([23]); // 26
            [41]([23]) -> ([23]); // 27
            [3]([22], [23]) -> ([24]); // 28
            [2]() -> ([25]); // 29
            [1]([25], [24]) -> ([26]); // 30
            [0]([26]) -> ([27]); // 31
            [42]([0]) -> ([0]); // 32
            [43]([1]) -> ([1]); // 33
            [40]([9]) -> ([9]); // 34
            [44]([27]) -> ([27]); // 35
            return([0], [1], [9], [27]); // 36
            [23]() -> (); // 37
            [24]() -> ([28]); // 38
            [18]([28]) -> ([29]); // 39
            [24]() -> ([30]); // 40
            [18]([30]) -> ([31]); // 41
            [24]() -> ([32]); // 42
            [18]([32]) -> ([33]); // 43
            [24]() -> ([34]); // 44
            [18]([34]) -> ([35]); // 45
            [17]([29], [31], [33], [35]) -> ([36]); // 46
            [39]([36]) -> ([36]); // 47
            [16]([21], [36]) { fallthrough([37]) 127([38]) }; // 48
            [23]() -> (); // 49
            [15]() -> ([39]); // 50
            [29]() -> ([40]); // 51
            [30]() -> ([41]); // 52
            [45]([40]) -> ([40]); // 53
            [46]([41]) -> ([41]); // 54
            [14]([0], [1], [39], [37], [8], [40], [41]) { fallthrough([42], [43], [44]) 85([45], [46], [47], [48]) }; // 55
            [23]() -> (); // 56
            [13]([44]) -> ([49], [50]); // 57
            [42]([42]) -> ([42]); // 58
            [43]([43]) -> ([43]); // 59
            [9]([50]) { fallthrough([51]) 75([52]) }; // 60
            [23]() -> (); // 61
            [8]([51]) { fallthrough([53]) 72([54]) }; // 62
            [23]() -> (); // 63
            [7]([53]) { fallthrough([55]) 69([56]) }; // 64
            [23]() -> (); // 65
            [6]([55]) -> ([57]); // 66
            [47]([57]) -> ([58]); // 67
            [31]() { 77() }; // 68
            [23]() -> (); // 69
            [47]([56]) -> ([58]); // 70
            [31]() { 77() }; // 71
            [23]() -> (); // 72
            [47]([54]) -> ([58]); // 73
            [31]() { 77() }; // 74
            [23]() -> (); // 75
            [47]([52]) -> ([58]); // 76
            [5]([9], [58]) -> ([59]); // 77
            [12]([49]) -> ([60]); // 78
            [11]([60]) -> ([61]); // 79
            [42]([42]) -> ([42]); // 80
            [43]([43]) -> ([43]); // 81
            [40]([59]) -> ([59]); // 82
            [44]([61]) -> ([61]); // 83
            return([42], [43], [59], [61]); // 84
            [23]() -> (); // 85
            [32]([47]) -> (); // 86
            [4]() -> ([62]); // 87
            [33]() -> ([63]); // 88
            [41]([63]) -> ([63]); // 89
            [3]([62], [63]) -> ([64]); // 90
            [29]() -> ([65]); // 91
            [30]() -> ([66]); // 92
            [43]([46]) -> ([46]); // 93
            [45]([65]) -> ([65]); // 94
            [46]([66]) -> ([66]); // 95
            [10]([9], [46], [48], [65], [66]) -> ([67], [68], [69]); // 96
            [42]([45]) -> ([45]); // 97
            [48]([64]) -> ([64]); // 98
            [40]([67]) -> ([67]); // 99
            [43]([68]) -> ([68]); // 100
            [9]([69]) { fallthrough([70]) 116([71]) }; // 101
            [23]() -> (); // 102
            [8]([70]) { fallthrough([72]) 113([73]) }; // 103
            [23]() -> (); // 104
            [7]([72]) { fallthrough([74]) 110([75]) }; // 105
            [23]() -> (); // 106
            [6]([74]) -> ([76]); // 107
            [47]([76]) -> ([77]); // 108
            [31]() { 118() }; // 109
            [23]() -> (); // 110
            [47]([75]) -> ([77]); // 111
            [31]() { 118() }; // 112
            [23]() -> (); // 113
            [47]([73]) -> ([77]); // 114
            [31]() { 118() }; // 115
            [23]() -> (); // 116
            [47]([71]) -> ([77]); // 117
            [5]([67], [77]) -> ([78]); // 118
            [2]() -> ([79]); // 119
            [1]([79], [64]) -> ([80]); // 120
            [0]([80]) -> ([81]); // 121
            [42]([45]) -> ([45]); // 122
            [43]([68]) -> ([68]); // 123
            [40]([78]) -> ([78]); // 124
            [44]([81]) -> ([81]); // 125
            return([45], [68], [78], [81]); // 126
            [23]() -> (); // 127
            [34]([38]) -> (); // 128
            [27]([8]) -> (); // 129
            [4]() -> ([82]); // 130
            [35]() -> ([83]); // 131
            [41]([83]) -> ([83]); // 132
            [3]([82], [83]) -> ([84]); // 133
            [2]() -> ([85]); // 134
            [1]([85], [84]) -> ([86]); // 135
            [0]([86]) -> ([87]); // 136
            [42]([0]) -> ([0]); // 137
            [43]([1]) -> ([1]); // 138
            [40]([9]) -> ([9]); // 139
            [44]([87]) -> ([87]); // 140
            return([0], [1], [9], [87]); // 141
            [23]() -> (); // 142
            [4]() -> ([88]); // 143
            [36]() -> ([89]); // 144
            [41]([89]) -> ([89]); // 145
            [3]([88], [89]) -> ([90]); // 146
            [2]() -> ([91]); // 147
            [1]([91], [90]) -> ([92]); // 148
            [0]([92]) -> ([93]); // 149
            [42]([0]) -> ([0]); // 150
            [43]([1]) -> ([1]); // 151
            [40]([2]) -> ([2]); // 152
            [44]([93]) -> ([93]); // 153
            return([0], [1], [2], [93]); // 154

            [0]@0([0]: [0], [1]: [1], [2]: [2]) -> ([0], [1], [2], [21]);
        "#).map_err(|e| e.to_string()).unwrap();

        let return_value = run_sierra_program(&program, &[]).return_value;

        assert_eq!(
            jit_enum!(0, jit_struct!(u384(["0x9", "0x9", "0x9", "0x9"]))),
            return_value
        );
    }

    #[test]
    fn run_inverse_circuit() {
        // use core::circuit::{
        //     RangeCheck96, AddMod, MulMod, u96, CircuitElement, CircuitInput, circuit_add,
        //     circuit_sub, circuit_mul, circuit_inverse, EvalCircuitTrait, u384,
        //     CircuitOutputsTrait, CircuitModulus, AddInputResultTrait, CircuitInputs,
        // };
        // fn main() -> u384 {
        //     let in1 = CircuitElement::<CircuitInput<0>> {};
        //     let inv = circuit_inverse(in1);
        //     let modulus = TryInto::<_, CircuitModulus>::try_into([11, 0, 0, 0]).unwrap();
        //     let outputs = (inv,)
        //         .new_inputs()
        //         .next([2, 0, 0, 0])
        //         .done()
        //         .eval(modulus)
        //         .unwrap();
        //     outputs.get_output(inv)
        // }
        let program = ProgramParser::new().parse(r#"
            type [3] = BoundedInt<0, 79228162514264337593543950335> [storable: true, drop: true, dup: true, zero_sized: false];
            type [38] = Const<[27], 29721761890975875353235833581453094220424382983267374> [storable: false, drop: false, dup: false, zero_sized: false];
            type [37] = Const<[27], 138583295661092166701491297054433349032460315956105119041111996301516236132> [storable: false, drop: false, dup: false, zero_sized: false];
            type [26] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
            type [28] = Array<[27]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [29] = Struct<ut@Tuple, [26], [28]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [36] = Const<[27], 30828113188794245257250221355944970489240709081949230> [storable: false, drop: false, dup: false, zero_sized: false];
            type [27] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [20] = Struct<ut@core::circuit::u384, [3], [3], [3], [3]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [25] = Struct<ut@Tuple, [20]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [30] = Enum<ut@core::panics::PanicResult::<(core::circuit::u384,)>, [25], [29]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [24] = U96LimbsLtGuarantee<1> [storable: true, drop: false, dup: false, zero_sized: false];
            type [23] = U96LimbsLtGuarantee<2> [storable: true, drop: false, dup: false, zero_sized: false];
            type [22] = U96LimbsLtGuarantee<3> [storable: true, drop: false, dup: false, zero_sized: false];
            type [9] = InverseGate<[10]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [21] = U96LimbsLtGuarantee<4> [storable: true, drop: false, dup: false, zero_sized: false];
            type [10] = CircuitInput<0> [storable: false, drop: false, dup: false, zero_sized: true];
            type [19] = CircuitFailureGuarantee [storable: true, drop: false, dup: false, zero_sized: false];
            type [18] = CircuitPartialOutputs<[7]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [17] = CircuitOutputs<[7]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = MulMod [storable: true, drop: false, dup: false, zero_sized: false];
            type [0] = AddMod [storable: true, drop: false, dup: false, zero_sized: false];
            type [35] = Const<[16], 1> [storable: false, drop: false, dup: false, zero_sized: false];
            type [16] = BoundedInt<1, 1> [storable: true, drop: true, dup: true, zero_sized: false];
            type [34] = Const<[15], 0> [storable: false, drop: false, dup: false, zero_sized: false];
            type [15] = BoundedInt<0, 0> [storable: true, drop: true, dup: true, zero_sized: false];
            type [14] = CircuitDescriptor<[7]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [13] = CircuitData<[7]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [11] = U96Guarantee [storable: true, drop: false, dup: false, zero_sized: false];
            type [12] = Struct<ut@Tuple, [11], [11], [11], [11]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [33] = Const<[3], 2> [storable: false, drop: false, dup: false, zero_sized: false];
            type [7] = Circuit<[6]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [6] = Struct<ut@Tuple, [9]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [8] = CircuitInputAccumulator<[7]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [2] = RangeCheck96 [storable: true, drop: false, dup: false, zero_sized: false];
            type [5] = CircuitModulus [storable: true, drop: true, dup: true, zero_sized: false];
            type [4] = Struct<ut@Tuple, [3], [3], [3], [3]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [32] = Const<[3], 0> [storable: false, drop: false, dup: false, zero_sized: false];
            type [31] = Const<[3], 11> [storable: false, drop: false, dup: false, zero_sized: false];

            libfunc [22] = const_as_immediate<[31]>;
            libfunc [23] = const_as_immediate<[32]>;
            libfunc [21] = struct_construct<[4]>;
            libfunc [35] = store_temp<[4]>;
            libfunc [20] = try_into_circuit_modulus;
            libfunc [24] = branch_align;
            libfunc [19] = init_circuit_data<[7]>;
            libfunc [25] = const_as_immediate<[33]>;
            libfunc [18] = into_u96_guarantee<[3]>;
            libfunc [17] = struct_construct<[12]>;
            libfunc [36] = store_temp<[8]>;
            libfunc [37] = store_temp<[12]>;
            libfunc [38] = store_temp<[2]>;
            libfunc [16] = add_circuit_input<[7]>;
            libfunc [15] = get_circuit_descriptor<[7]>;
            libfunc [26] = const_as_immediate<[34]>;
            libfunc [27] = const_as_immediate<[35]>;
            libfunc [39] = store_temp<[15]>;
            libfunc [40] = store_temp<[16]>;
            libfunc [14] = eval_circuit<[7]>;
            libfunc [13] = get_circuit_output<[7], [9]>;
            libfunc [41] = store_temp<[0]>;
            libfunc [42] = store_temp<[1]>;
            libfunc [9] = u96_limbs_less_than_guarantee_verify<4>;
            libfunc [8] = u96_limbs_less_than_guarantee_verify<3>;
            libfunc [7] = u96_limbs_less_than_guarantee_verify<2>;
            libfunc [6] = u96_single_limb_less_than_guarantee_verify;
            libfunc [43] = store_temp<[11]>;
            libfunc [28] = jump;
            libfunc [5] = u96_guarantee_verify;
            libfunc [12] = struct_construct<[25]>;
            libfunc [11] = enum_init<[30], 0>;
            libfunc [44] = store_temp<[30]>;
            libfunc [29] = drop<[18]>;
            libfunc [4] = array_new<[27]>;
            libfunc [30] = const_as_immediate<[36]>;
            libfunc [45] = store_temp<[27]>;
            libfunc [3] = array_append<[27]>;
            libfunc [10] = circuit_failure_guarantee_verify;
            libfunc [46] = store_temp<[28]>;
            libfunc [2] = struct_construct<[26]>;
            libfunc [1] = struct_construct<[29]>;
            libfunc [0] = enum_init<[30], 1>;
            libfunc [31] = drop<[8]>;
            libfunc [32] = drop<[5]>;
            libfunc [33] = const_as_immediate<[37]>;
            libfunc [34] = const_as_immediate<[38]>;

            [22]() -> ([3]); // 0
            [23]() -> ([4]); // 1
            [23]() -> ([5]); // 2
            [23]() -> ([6]); // 3
            [21]([3], [4], [5], [6]) -> ([7]); // 4
            [35]([7]) -> ([7]); // 5
            [20]([7]) { fallthrough([8]) 115() }; // 6
            [24]() -> (); // 7
            [19]([2]) -> ([9], [10]); // 8
            [25]() -> ([11]); // 9
            [18]([11]) -> ([12]); // 10
            [23]() -> ([13]); // 11
            [18]([13]) -> ([14]); // 12
            [23]() -> ([15]); // 13
            [18]([15]) -> ([16]); // 14
            [23]() -> ([17]); // 15
            [18]([17]) -> ([18]); // 16
            [17]([12], [14], [16], [18]) -> ([19]); // 17
            [36]([10]) -> ([10]); // 18
            [37]([19]) -> ([19]); // 19
            [38]([9]) -> ([9]); // 20
            [16]([10], [19]) { fallthrough([20]) 100([21]) }; // 21
            [24]() -> (); // 22
            [15]() -> ([22]); // 23
            [26]() -> ([23]); // 24
            [27]() -> ([24]); // 25
            [39]([23]) -> ([23]); // 26
            [40]([24]) -> ([24]); // 27
            [14]([0], [1], [22], [20], [8], [23], [24]) { fallthrough([25], [26], [27]) 58([28], [29], [30], [31]) }; // 28
            [24]() -> (); // 29
            [13]([27]) -> ([32], [33]); // 30
            [41]([25]) -> ([25]); // 31
            [42]([26]) -> ([26]); // 32
            [9]([33]) { fallthrough([34]) 48([35]) }; // 33
            [24]() -> (); // 34
            [8]([34]) { fallthrough([36]) 45([37]) }; // 35
            [24]() -> (); // 36
            [7]([36]) { fallthrough([38]) 42([39]) }; // 37
            [24]() -> (); // 38
            [6]([38]) -> ([40]); // 39
            [43]([40]) -> ([41]); // 40
            [28]() { 50() }; // 41
            [24]() -> (); // 42
            [43]([39]) -> ([41]); // 43
            [28]() { 50() }; // 44
            [24]() -> (); // 45
            [43]([37]) -> ([41]); // 46
            [28]() { 50() }; // 47
            [24]() -> (); // 48
            [43]([35]) -> ([41]); // 49
            [5]([9], [41]) -> ([42]); // 50
            [12]([32]) -> ([43]); // 51
            [11]([43]) -> ([44]); // 52
            [41]([25]) -> ([25]); // 53
            [42]([26]) -> ([26]); // 54
            [38]([42]) -> ([42]); // 55
            [44]([44]) -> ([44]); // 56
            return([25], [26], [42], [44]); // 57
            [24]() -> (); // 58
            [29]([30]) -> (); // 59
            [4]() -> ([45]); // 60
            [30]() -> ([46]); // 61
            [45]([46]) -> ([46]); // 62
            [3]([45], [46]) -> ([47]); // 63
            [26]() -> ([48]); // 64
            [27]() -> ([49]); // 65
            [42]([29]) -> ([29]); // 66
            [39]([48]) -> ([48]); // 67
            [40]([49]) -> ([49]); // 68
            [10]([9], [29], [31], [48], [49]) -> ([50], [51], [52]); // 69
            [41]([28]) -> ([28]); // 70
            [46]([47]) -> ([47]); // 71
            [38]([50]) -> ([50]); // 72
            [42]([51]) -> ([51]); // 73
            [9]([52]) { fallthrough([53]) 89([54]) }; // 74
            [24]() -> (); // 75
            [8]([53]) { fallthrough([55]) 86([56]) }; // 76
            [24]() -> (); // 77
            [7]([55]) { fallthrough([57]) 83([58]) }; // 78
            [24]() -> (); // 79
            [6]([57]) -> ([59]); // 80
            [43]([59]) -> ([60]); // 81
            [28]() { 91() }; // 82
            [24]() -> (); // 83
            [43]([58]) -> ([60]); // 84
            [28]() { 91() }; // 85
            [24]() -> (); // 86
            [43]([56]) -> ([60]); // 87
            [28]() { 91() }; // 88
            [24]() -> (); // 89
            [43]([54]) -> ([60]); // 90
            [5]([50], [60]) -> ([61]); // 91
            [2]() -> ([62]); // 92
            [1]([62], [47]) -> ([63]); // 93
            [0]([63]) -> ([64]); // 94
            [41]([28]) -> ([28]); // 95
            [42]([51]) -> ([51]); // 96
            [38]([61]) -> ([61]); // 97
            [44]([64]) -> ([64]); // 98
            return([28], [51], [61], [64]); // 99
            [24]() -> (); // 100
            [31]([21]) -> (); // 101
            [32]([8]) -> (); // 102
            [4]() -> ([65]); // 103
            [33]() -> ([66]); // 104
            [45]([66]) -> ([66]); // 105
            [3]([65], [66]) -> ([67]); // 106
            [2]() -> ([68]); // 107
            [1]([68], [67]) -> ([69]); // 108
            [0]([69]) -> ([70]); // 109
            [41]([0]) -> ([0]); // 110
            [42]([1]) -> ([1]); // 111
            [38]([9]) -> ([9]); // 112
            [44]([70]) -> ([70]); // 113
            return([0], [1], [9], [70]); // 114
            [24]() -> (); // 115
            [4]() -> ([71]); // 116
            [34]() -> ([72]); // 117
            [45]([72]) -> ([72]); // 118
            [3]([71], [72]) -> ([73]); // 119
            [2]() -> ([74]); // 120
            [1]([74], [73]) -> ([75]); // 121
            [0]([75]) -> ([76]); // 122
            [41]([0]) -> ([0]); // 123
            [42]([1]) -> ([1]); // 124
            [38]([2]) -> ([2]); // 125
            [44]([76]) -> ([76]); // 126
            return([0], [1], [2], [76]); // 127

            [0]@0([0]: [0], [1]: [1], [2]: [2]) -> ([0], [1], [2], [30]);
        "#).map_err(|e| e.to_string()).unwrap();

        let return_value = run_sierra_program(&program, &[]).return_value;

        assert_eq!(
            jit_enum!(0, jit_struct!(u384(["0x6", "0x0", "0x0", "0x0"]))),
            return_value
        );
    }

    #[test]
    fn run_no_coprime_circuit() {
        // use core::circuit::{
        //     RangeCheck96, AddMod, MulMod, u96, CircuitElement, CircuitInput, circuit_add,
        //     circuit_sub, circuit_mul, circuit_inverse, EvalCircuitTrait, u384,
        //     CircuitOutputsTrait, CircuitModulus, AddInputResultTrait, CircuitInputs,
        // };
        // fn main() -> u384 {
        //     let in1 = CircuitElement::<CircuitInput<0>> {};
        //     let inv = circuit_inverse(in1);
        //     let modulus = TryInto::<_, CircuitModulus>::try_into([12, 0, 0, 0]).unwrap();
        //     let outputs = (inv,)
        //         .new_inputs()
        //         .next([3, 0, 0, 0])
        //         .done()
        //         .eval(modulus)
        //         .unwrap();
        //     outputs.get_output(inv)
        // }
        let program = ProgramParser::new().parse(r#"
            type [3] = BoundedInt<0, 79228162514264337593543950335> [storable: true, drop: true, dup: true, zero_sized: false];
            type [38] = Const<[27], 29721761890975875353235833581453094220424382983267374> [storable: false, drop: false, dup: false, zero_sized: false];
            type [37] = Const<[27], 138583295661092166701491297054433349032460315956105119041111996301516236132> [storable: false, drop: false, dup: false, zero_sized: false];
            type [26] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
            type [28] = Array<[27]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [29] = Struct<ut@Tuple, [26], [28]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [36] = Const<[27], 30828113188794245257250221355944970489240709081949230> [storable: false, drop: false, dup: false, zero_sized: false];
            type [27] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [20] = Struct<ut@core::circuit::u384, [3], [3], [3], [3]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [25] = Struct<ut@Tuple, [20]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [30] = Enum<ut@core::panics::PanicResult::<(core::circuit::u384,)>, [25], [29]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [24] = U96LimbsLtGuarantee<1> [storable: true, drop: false, dup: false, zero_sized: false];
            type [23] = U96LimbsLtGuarantee<2> [storable: true, drop: false, dup: false, zero_sized: false];
            type [22] = U96LimbsLtGuarantee<3> [storable: true, drop: false, dup: false, zero_sized: false];
            type [9] = InverseGate<[10]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [21] = U96LimbsLtGuarantee<4> [storable: true, drop: false, dup: false, zero_sized: false];
            type [10] = CircuitInput<0> [storable: false, drop: false, dup: false, zero_sized: true];
            type [19] = CircuitFailureGuarantee [storable: true, drop: false, dup: false, zero_sized: false];
            type [18] = CircuitPartialOutputs<[7]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [17] = CircuitOutputs<[7]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = MulMod [storable: true, drop: false, dup: false, zero_sized: false];
            type [0] = AddMod [storable: true, drop: false, dup: false, zero_sized: false];
            type [35] = Const<[16], 1> [storable: false, drop: false, dup: false, zero_sized: false];
            type [16] = BoundedInt<1, 1> [storable: true, drop: true, dup: true, zero_sized: false];
            type [34] = Const<[15], 0> [storable: false, drop: false, dup: false, zero_sized: false];
            type [15] = BoundedInt<0, 0> [storable: true, drop: true, dup: true, zero_sized: false];
            type [14] = CircuitDescriptor<[7]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [13] = CircuitData<[7]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [11] = U96Guarantee [storable: true, drop: false, dup: false, zero_sized: false];
            type [12] = Struct<ut@Tuple, [11], [11], [11], [11]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [33] = Const<[3], 3> [storable: false, drop: false, dup: false, zero_sized: false];
            type [7] = Circuit<[6]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [6] = Struct<ut@Tuple, [9]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [8] = CircuitInputAccumulator<[7]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [2] = RangeCheck96 [storable: true, drop: false, dup: false, zero_sized: false];
            type [5] = CircuitModulus [storable: true, drop: true, dup: true, zero_sized: false];
            type [4] = Struct<ut@Tuple, [3], [3], [3], [3]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [32] = Const<[3], 0> [storable: false, drop: false, dup: false, zero_sized: false];
            type [31] = Const<[3], 12> [storable: false, drop: false, dup: false, zero_sized: false];

            libfunc [22] = const_as_immediate<[31]>;
            libfunc [23] = const_as_immediate<[32]>;
            libfunc [21] = struct_construct<[4]>;
            libfunc [35] = store_temp<[4]>;
            libfunc [20] = try_into_circuit_modulus;
            libfunc [24] = branch_align;
            libfunc [19] = init_circuit_data<[7]>;
            libfunc [25] = const_as_immediate<[33]>;
            libfunc [18] = into_u96_guarantee<[3]>;
            libfunc [17] = struct_construct<[12]>;
            libfunc [36] = store_temp<[8]>;
            libfunc [37] = store_temp<[12]>;
            libfunc [38] = store_temp<[2]>;
            libfunc [16] = add_circuit_input<[7]>;
            libfunc [15] = get_circuit_descriptor<[7]>;
            libfunc [26] = const_as_immediate<[34]>;
            libfunc [27] = const_as_immediate<[35]>;
            libfunc [39] = store_temp<[15]>;
            libfunc [40] = store_temp<[16]>;
            libfunc [14] = eval_circuit<[7]>;
            libfunc [13] = get_circuit_output<[7], [9]>;
            libfunc [41] = store_temp<[0]>;
            libfunc [42] = store_temp<[1]>;
            libfunc [9] = u96_limbs_less_than_guarantee_verify<4>;
            libfunc [8] = u96_limbs_less_than_guarantee_verify<3>;
            libfunc [7] = u96_limbs_less_than_guarantee_verify<2>;
            libfunc [6] = u96_single_limb_less_than_guarantee_verify;
            libfunc [43] = store_temp<[11]>;
            libfunc [28] = jump;
            libfunc [5] = u96_guarantee_verify;
            libfunc [12] = struct_construct<[25]>;
            libfunc [11] = enum_init<[30], 0>;
            libfunc [44] = store_temp<[30]>;
            libfunc [29] = drop<[18]>;
            libfunc [4] = array_new<[27]>;
            libfunc [30] = const_as_immediate<[36]>;
            libfunc [45] = store_temp<[27]>;
            libfunc [3] = array_append<[27]>;
            libfunc [10] = circuit_failure_guarantee_verify;
            libfunc [46] = store_temp<[28]>;
            libfunc [2] = struct_construct<[26]>;
            libfunc [1] = struct_construct<[29]>;
            libfunc [0] = enum_init<[30], 1>;
            libfunc [31] = drop<[8]>;
            libfunc [32] = drop<[5]>;
            libfunc [33] = const_as_immediate<[37]>;
            libfunc [34] = const_as_immediate<[38]>;

            [22]() -> ([3]); // 0
            [23]() -> ([4]); // 1
            [23]() -> ([5]); // 2
            [23]() -> ([6]); // 3
            [21]([3], [4], [5], [6]) -> ([7]); // 4
            [35]([7]) -> ([7]); // 5
            [20]([7]) { fallthrough([8]) 115() }; // 6
            [24]() -> (); // 7
            [19]([2]) -> ([9], [10]); // 8
            [25]() -> ([11]); // 9
            [18]([11]) -> ([12]); // 10
            [23]() -> ([13]); // 11
            [18]([13]) -> ([14]); // 12
            [23]() -> ([15]); // 13
            [18]([15]) -> ([16]); // 14
            [23]() -> ([17]); // 15
            [18]([17]) -> ([18]); // 16
            [17]([12], [14], [16], [18]) -> ([19]); // 17
            [36]([10]) -> ([10]); // 18
            [37]([19]) -> ([19]); // 19
            [38]([9]) -> ([9]); // 20
            [16]([10], [19]) { fallthrough([20]) 100([21]) }; // 21
            [24]() -> (); // 22
            [15]() -> ([22]); // 23
            [26]() -> ([23]); // 24
            [27]() -> ([24]); // 25
            [39]([23]) -> ([23]); // 26
            [40]([24]) -> ([24]); // 27
            [14]([0], [1], [22], [20], [8], [23], [24]) { fallthrough([25], [26], [27]) 58([28], [29], [30], [31]) }; // 28
            [24]() -> (); // 29
            [13]([27]) -> ([32], [33]); // 30
            [41]([25]) -> ([25]); // 31
            [42]([26]) -> ([26]); // 32
            [9]([33]) { fallthrough([34]) 48([35]) }; // 33
            [24]() -> (); // 34
            [8]([34]) { fallthrough([36]) 45([37]) }; // 35
            [24]() -> (); // 36
            [7]([36]) { fallthrough([38]) 42([39]) }; // 37
            [24]() -> (); // 38
            [6]([38]) -> ([40]); // 39
            [43]([40]) -> ([41]); // 40
            [28]() { 50() }; // 41
            [24]() -> (); // 42
            [43]([39]) -> ([41]); // 43
            [28]() { 50() }; // 44
            [24]() -> (); // 45
            [43]([37]) -> ([41]); // 46
            [28]() { 50() }; // 47
            [24]() -> (); // 48
            [43]([35]) -> ([41]); // 49
            [5]([9], [41]) -> ([42]); // 50
            [12]([32]) -> ([43]); // 51
            [11]([43]) -> ([44]); // 52
            [41]([25]) -> ([25]); // 53
            [42]([26]) -> ([26]); // 54
            [38]([42]) -> ([42]); // 55
            [44]([44]) -> ([44]); // 56
            return([25], [26], [42], [44]); // 57
            [24]() -> (); // 58
            [29]([30]) -> (); // 59
            [4]() -> ([45]); // 60
            [30]() -> ([46]); // 61
            [45]([46]) -> ([46]); // 62
            [3]([45], [46]) -> ([47]); // 63
            [26]() -> ([48]); // 64
            [27]() -> ([49]); // 65
            [42]([29]) -> ([29]); // 66
            [39]([48]) -> ([48]); // 67
            [40]([49]) -> ([49]); // 68
            [10]([9], [29], [31], [48], [49]) -> ([50], [51], [52]); // 69
            [41]([28]) -> ([28]); // 70
            [46]([47]) -> ([47]); // 71
            [38]([50]) -> ([50]); // 72
            [42]([51]) -> ([51]); // 73
            [9]([52]) { fallthrough([53]) 89([54]) }; // 74
            [24]() -> (); // 75
            [8]([53]) { fallthrough([55]) 86([56]) }; // 76
            [24]() -> (); // 77
            [7]([55]) { fallthrough([57]) 83([58]) }; // 78
            [24]() -> (); // 79
            [6]([57]) -> ([59]); // 80
            [43]([59]) -> ([60]); // 81
            [28]() { 91() }; // 82
            [24]() -> (); // 83
            [43]([58]) -> ([60]); // 84
            [28]() { 91() }; // 85
            [24]() -> (); // 86
            [43]([56]) -> ([60]); // 87
            [28]() { 91() }; // 88
            [24]() -> (); // 89
            [43]([54]) -> ([60]); // 90
            [5]([50], [60]) -> ([61]); // 91
            [2]() -> ([62]); // 92
            [1]([62], [47]) -> ([63]); // 93
            [0]([63]) -> ([64]); // 94
            [41]([28]) -> ([28]); // 95
            [42]([51]) -> ([51]); // 96
            [38]([61]) -> ([61]); // 97
            [44]([64]) -> ([64]); // 98
            return([28], [51], [61], [64]); // 99
            [24]() -> (); // 100
            [31]([21]) -> (); // 101
            [32]([8]) -> (); // 102
            [4]() -> ([65]); // 103
            [33]() -> ([66]); // 104
            [45]([66]) -> ([66]); // 105
            [3]([65], [66]) -> ([67]); // 106
            [2]() -> ([68]); // 107
            [1]([68], [67]) -> ([69]); // 108
            [0]([69]) -> ([70]); // 109
            [41]([0]) -> ([0]); // 110
            [42]([1]) -> ([1]); // 111
            [38]([9]) -> ([9]); // 112
            [44]([70]) -> ([70]); // 113
            return([0], [1], [9], [70]); // 114
            [24]() -> (); // 115
            [4]() -> ([71]); // 116
            [34]() -> ([72]); // 117
            [45]([72]) -> ([72]); // 118
            [3]([71], [72]) -> ([73]); // 119
            [2]() -> ([74]); // 120
            [1]([74], [73]) -> ([75]); // 121
            [0]([75]) -> ([76]); // 122
            [41]([0]) -> ([0]); // 123
            [42]([1]) -> ([1]); // 124
            [38]([2]) -> ([2]); // 125
            [44]([76]) -> ([76]); // 126
            return([0], [1], [2], [76]); // 127

            [0]@0([0]: [0], [1]: [1], [2]: [2]) -> ([0], [1], [2], [30]);
        "#).map_err(|e| e.to_string()).unwrap();

        let return_value = run_sierra_program(&program, &[]).return_value;

        assert_eq!(
            jit_panic!(felt252_str(
                "30828113188794245257250221355944970489240709081949230"
            )),
            return_value
        );
    }

    #[test]
    fn run_mul_overflow_circuit() {
        // use core::circuit::{
        //     RangeCheck96, AddMod, MulMod, u96, CircuitElement, CircuitInput, circuit_add,
        //     circuit_sub, circuit_mul, circuit_inverse, EvalCircuitTrait, u384,
        //     CircuitOutputsTrait, CircuitModulus, AddInputResultTrait, CircuitInputs,
        // };
        // fn main() -> u384 {
        //     let in1 = CircuitElement::<CircuitInput<0>> {};
        //     let in2 = CircuitElement::<CircuitInput<1>> {};
        //     let mul = circuit_mul(in1, in2);
        //     let modulus = TryInto::<_, CircuitModulus>::try_into([
        //         0xffffffffffffffffffffffff,
        //         0xffffffffffffffffffffffff,
        //         0xffffffffffffffffffffffff,
        //         0xffffffffffffffffffffffff,
        //     ])
        //     .unwrap();
        //     let outputs = (mul,)
        //         .new_inputs()
        //         .next([0, 0, 0, 0xffffffffffffffffffffffff])
        //         .next([16, 0, 0, 0])
        //         .done()
        //         .eval(modulus)
        //         .unwrap();
        //     outputs.get_output(mul)
        // }
        let program = ProgramParser::new().parse(r#"
            type [3] = BoundedInt<0, 79228162514264337593543950335> [storable: true, drop: true, dup: true, zero_sized: false];
            type [40] = Const<[15], 29721761890975875353235833581453094220424382983267374> [storable: false, drop: false, dup: false, zero_sized: false];
            type [39] = Const<[15], 138583295661092166701491297054433349032460315956105119041111996301516236132> [storable: false, drop: false, dup: false, zero_sized: false];
            type [38] = Const<[15], 30828113188794245257250221355944970489240709081949230> [storable: false, drop: false, dup: false, zero_sized: false];
            type [19] = Struct<ut@core::circuit::u384, [3], [3], [3], [3]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [20] = Struct<ut@Tuple, [19]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [31] = U96LimbsLtGuarantee<1> [storable: true, drop: false, dup: false, zero_sized: false];
            type [30] = U96LimbsLtGuarantee<2> [storable: true, drop: false, dup: false, zero_sized: false];
            type [29] = U96LimbsLtGuarantee<3> [storable: true, drop: false, dup: false, zero_sized: false];
            type [9] = MulModGate<[10], [11]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [28] = U96LimbsLtGuarantee<4> [storable: true, drop: false, dup: false, zero_sized: false];
            type [11] = CircuitInput<1> [storable: false, drop: false, dup: false, zero_sized: true];
            type [10] = CircuitInput<0> [storable: false, drop: false, dup: false, zero_sized: true];
            type [27] = CircuitFailureGuarantee [storable: true, drop: false, dup: false, zero_sized: false];
            type [26] = CircuitPartialOutputs<[7]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [25] = CircuitOutputs<[7]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [37] = Const<[24], 1> [storable: false, drop: false, dup: false, zero_sized: false];
            type [24] = BoundedInt<1, 1> [storable: true, drop: true, dup: true, zero_sized: false];
            type [36] = Const<[23], 0> [storable: false, drop: false, dup: false, zero_sized: false];
            type [23] = BoundedInt<0, 0> [storable: true, drop: true, dup: true, zero_sized: false];
            type [22] = CircuitDescriptor<[7]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [35] = Const<[3], 16> [storable: false, drop: false, dup: false, zero_sized: false];
            type [1] = MulMod [storable: true, drop: false, dup: false, zero_sized: false];
            type [0] = AddMod [storable: true, drop: false, dup: false, zero_sized: false];
            type [17] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
            type [16] = Array<[15]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [18] = Struct<ut@Tuple, [17], [16]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [21] = Enum<ut@core::panics::PanicResult::<(core::circuit::u384,)>, [20], [18]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [34] = Const<[15], 26913677086973030051406221357623718750637972950955665348321109348> [storable: false, drop: false, dup: false, zero_sized: false];
            type [15] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [14] = CircuitData<[7]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [12] = U96Guarantee [storable: true, drop: false, dup: false, zero_sized: false];
            type [13] = Struct<ut@Tuple, [12], [12], [12], [12]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [33] = Const<[3], 0> [storable: false, drop: false, dup: false, zero_sized: false];
            type [7] = Circuit<[6]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [6] = Struct<ut@Tuple, [9]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [8] = CircuitInputAccumulator<[7]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [2] = RangeCheck96 [storable: true, drop: false, dup: false, zero_sized: false];
            type [5] = CircuitModulus [storable: true, drop: true, dup: true, zero_sized: false];
            type [4] = Struct<ut@Tuple, [3], [3], [3], [3]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [32] = Const<[3], 79228162514264337593543950335> [storable: false, drop: false, dup: false, zero_sized: false];

            libfunc [22] = const_as_immediate<[32]>;
            libfunc [21] = struct_construct<[4]>;
            libfunc [37] = store_temp<[4]>;
            libfunc [20] = try_into_circuit_modulus;
            libfunc [23] = branch_align;
            libfunc [19] = init_circuit_data<[7]>;
            libfunc [24] = const_as_immediate<[33]>;
            libfunc [18] = into_u96_guarantee<[3]>;
            libfunc [17] = struct_construct<[13]>;
            libfunc [38] = store_temp<[8]>;
            libfunc [39] = store_temp<[13]>;
            libfunc [40] = store_temp<[2]>;
            libfunc [16] = add_circuit_input<[7]>;
            libfunc [25] = drop<[14]>;
            libfunc [26] = drop<[5]>;
            libfunc [4] = array_new<[15]>;
            libfunc [27] = const_as_immediate<[34]>;
            libfunc [41] = store_temp<[15]>;
            libfunc [3] = array_append<[15]>;
            libfunc [2] = struct_construct<[17]>;
            libfunc [1] = struct_construct<[18]>;
            libfunc [0] = enum_init<[21], 1>;
            libfunc [42] = store_temp<[0]>;
            libfunc [43] = store_temp<[1]>;
            libfunc [44] = store_temp<[21]>;
            libfunc [28] = const_as_immediate<[35]>;
            libfunc [15] = get_circuit_descriptor<[7]>;
            libfunc [29] = const_as_immediate<[36]>;
            libfunc [30] = const_as_immediate<[37]>;
            libfunc [45] = store_temp<[23]>;
            libfunc [46] = store_temp<[24]>;
            libfunc [14] = eval_circuit<[7]>;
            libfunc [13] = get_circuit_output<[7], [9]>;
            libfunc [9] = u96_limbs_less_than_guarantee_verify<4>;
            libfunc [8] = u96_limbs_less_than_guarantee_verify<3>;
            libfunc [7] = u96_limbs_less_than_guarantee_verify<2>;
            libfunc [6] = u96_single_limb_less_than_guarantee_verify;
            libfunc [47] = store_temp<[12]>;
            libfunc [31] = jump;
            libfunc [5] = u96_guarantee_verify;
            libfunc [12] = struct_construct<[20]>;
            libfunc [11] = enum_init<[21], 0>;
            libfunc [32] = drop<[26]>;
            libfunc [33] = const_as_immediate<[38]>;
            libfunc [10] = circuit_failure_guarantee_verify;
            libfunc [48] = store_temp<[16]>;
            libfunc [34] = drop<[8]>;
            libfunc [35] = const_as_immediate<[39]>;
            libfunc [36] = const_as_immediate<[40]>;

            [22]() -> ([3]); // 0
            [22]() -> ([4]); // 1
            [22]() -> ([5]); // 2
            [22]() -> ([6]); // 3
            [21]([3], [4], [5], [6]) -> ([7]); // 4
            [37]([7]) -> ([7]); // 5
            [20]([7]) { fallthrough([8]) 142() }; // 6
            [23]() -> (); // 7
            [19]([2]) -> ([9], [10]); // 8
            [24]() -> ([11]); // 9
            [18]([11]) -> ([12]); // 10
            [24]() -> ([13]); // 11
            [18]([13]) -> ([14]); // 12
            [24]() -> ([15]); // 13
            [18]([15]) -> ([16]); // 14
            [22]() -> ([17]); // 15
            [18]([17]) -> ([18]); // 16
            [17]([12], [14], [16], [18]) -> ([19]); // 17
            [38]([10]) -> ([10]); // 18
            [39]([19]) -> ([19]); // 19
            [40]([9]) -> ([9]); // 20
            [16]([10], [19]) { fallthrough([20]) 37([21]) }; // 21
            [23]() -> (); // 22
            [25]([20]) -> (); // 23
            [26]([8]) -> (); // 24
            [4]() -> ([22]); // 25
            [27]() -> ([23]); // 26
            [41]([23]) -> ([23]); // 27
            [3]([22], [23]) -> ([24]); // 28
            [2]() -> ([25]); // 29
            [1]([25], [24]) -> ([26]); // 30
            [0]([26]) -> ([27]); // 31
            [42]([0]) -> ([0]); // 32
            [43]([1]) -> ([1]); // 33
            [40]([9]) -> ([9]); // 34
            [44]([27]) -> ([27]); // 35
            return([0], [1], [9], [27]); // 36
            [23]() -> (); // 37
            [28]() -> ([28]); // 38
            [18]([28]) -> ([29]); // 39
            [24]() -> ([30]); // 40
            [18]([30]) -> ([31]); // 41
            [24]() -> ([32]); // 42
            [18]([32]) -> ([33]); // 43
            [24]() -> ([34]); // 44
            [18]([34]) -> ([35]); // 45
            [17]([29], [31], [33], [35]) -> ([36]); // 46
            [39]([36]) -> ([36]); // 47
            [16]([21], [36]) { fallthrough([37]) 127([38]) }; // 48
            [23]() -> (); // 49
            [15]() -> ([39]); // 50
            [29]() -> ([40]); // 51
            [30]() -> ([41]); // 52
            [45]([40]) -> ([40]); // 53
            [46]([41]) -> ([41]); // 54
            [14]([0], [1], [39], [37], [8], [40], [41]) { fallthrough([42], [43], [44]) 85([45], [46], [47], [48]) }; // 55
            [23]() -> (); // 56
            [13]([44]) -> ([49], [50]); // 57
            [42]([42]) -> ([42]); // 58
            [43]([43]) -> ([43]); // 59
            [9]([50]) { fallthrough([51]) 75([52]) }; // 60
            [23]() -> (); // 61
            [8]([51]) { fallthrough([53]) 72([54]) }; // 62
            [23]() -> (); // 63
            [7]([53]) { fallthrough([55]) 69([56]) }; // 64
            [23]() -> (); // 65
            [6]([55]) -> ([57]); // 66
            [47]([57]) -> ([58]); // 67
            [31]() { 77() }; // 68
            [23]() -> (); // 69
            [47]([56]) -> ([58]); // 70
            [31]() { 77() }; // 71
            [23]() -> (); // 72
            [47]([54]) -> ([58]); // 73
            [31]() { 77() }; // 74
            [23]() -> (); // 75
            [47]([52]) -> ([58]); // 76
            [5]([9], [58]) -> ([59]); // 77
            [12]([49]) -> ([60]); // 78
            [11]([60]) -> ([61]); // 79
            [42]([42]) -> ([42]); // 80
            [43]([43]) -> ([43]); // 81
            [40]([59]) -> ([59]); // 82
            [44]([61]) -> ([61]); // 83
            return([42], [43], [59], [61]); // 84
            [23]() -> (); // 85
            [32]([47]) -> (); // 86
            [4]() -> ([62]); // 87
            [33]() -> ([63]); // 88
            [41]([63]) -> ([63]); // 89
            [3]([62], [63]) -> ([64]); // 90
            [29]() -> ([65]); // 91
            [30]() -> ([66]); // 92
            [43]([46]) -> ([46]); // 93
            [45]([65]) -> ([65]); // 94
            [46]([66]) -> ([66]); // 95
            [10]([9], [46], [48], [65], [66]) -> ([67], [68], [69]); // 96
            [42]([45]) -> ([45]); // 97
            [48]([64]) -> ([64]); // 98
            [40]([67]) -> ([67]); // 99
            [43]([68]) -> ([68]); // 100
            [9]([69]) { fallthrough([70]) 116([71]) }; // 101
            [23]() -> (); // 102
            [8]([70]) { fallthrough([72]) 113([73]) }; // 103
            [23]() -> (); // 104
            [7]([72]) { fallthrough([74]) 110([75]) }; // 105
            [23]() -> (); // 106
            [6]([74]) -> ([76]); // 107
            [47]([76]) -> ([77]); // 108
            [31]() { 118() }; // 109
            [23]() -> (); // 110
            [47]([75]) -> ([77]); // 111
            [31]() { 118() }; // 112
            [23]() -> (); // 113
            [47]([73]) -> ([77]); // 114
            [31]() { 118() }; // 115
            [23]() -> (); // 116
            [47]([71]) -> ([77]); // 117
            [5]([67], [77]) -> ([78]); // 118
            [2]() -> ([79]); // 119
            [1]([79], [64]) -> ([80]); // 120
            [0]([80]) -> ([81]); // 121
            [42]([45]) -> ([45]); // 122
            [43]([68]) -> ([68]); // 123
            [40]([78]) -> ([78]); // 124
            [44]([81]) -> ([81]); // 125
            return([45], [68], [78], [81]); // 126
            [23]() -> (); // 127
            [34]([38]) -> (); // 128
            [26]([8]) -> (); // 129
            [4]() -> ([82]); // 130
            [35]() -> ([83]); // 131
            [41]([83]) -> ([83]); // 132
            [3]([82], [83]) -> ([84]); // 133
            [2]() -> ([85]); // 134
            [1]([85], [84]) -> ([86]); // 135
            [0]([86]) -> ([87]); // 136
            [42]([0]) -> ([0]); // 137
            [43]([1]) -> ([1]); // 138
            [40]([9]) -> ([9]); // 139
            [44]([87]) -> ([87]); // 140
            return([0], [1], [9], [87]); // 141
            [23]() -> (); // 142
            [4]() -> ([88]); // 143
            [36]() -> ([89]); // 144
            [41]([89]) -> ([89]); // 145
            [3]([88], [89]) -> ([90]); // 146
            [2]() -> ([91]); // 147
            [1]([91], [90]) -> ([92]); // 148
            [0]([92]) -> ([93]); // 149
            [42]([0]) -> ([0]); // 150
            [43]([1]) -> ([1]); // 151
            [40]([2]) -> ([2]); // 152
            [44]([93]) -> ([93]); // 153
            return([0], [1], [2], [93]); // 154

            [0]@0([0]: [0], [1]: [1], [2]: [2]) -> ([0], [1], [2], [21]);
        "#).map_err(|e| e.to_string()).unwrap();

        let return_value = run_sierra_program(&program, &[]).return_value;

        assert_eq!(
            jit_enum!(
                0,
                jit_struct!(u384(["0xf", "0x0", "0x0", "0xfffffffffffffffffffffff0"]))
            ),
            return_value
        );
    }

    #[test]
    fn run_full_circuit() {
        // use core::circuit::{
        //     RangeCheck96, AddMod, MulMod, u96, CircuitElement, CircuitInput, circuit_add,
        //     circuit_sub, circuit_mul, circuit_inverse, EvalCircuitTrait, u384,
        //     CircuitOutputsTrait, CircuitModulus, AddInputResultTrait, CircuitInputs,
        // };
        // fn main() -> u384 {
        //     let in1 = CircuitElement::<CircuitInput<0>> {};
        //     let in2 = CircuitElement::<CircuitInput<1>> {};
        //     let add1 = circuit_add(in1, in2);
        //     let mul1 = circuit_mul(add1, in1);
        //     let mul2 = circuit_mul(mul1, add1);
        //     let inv1 = circuit_inverse(mul2);
        //     let sub1 = circuit_sub(inv1, in2);
        //     let sub2 = circuit_sub(sub1, mul2);
        //     let inv2 = circuit_inverse(sub2);
        //     let add2 = circuit_add(inv2, inv2);
        //     let modulus = TryInto::<_, CircuitModulus>::try_into([17, 14, 14, 14]).unwrap();
        //     let outputs = (add2,)
        //         .new_inputs()
        //         .next([9, 2, 9, 3])
        //         .next([5, 7, 0, 8])
        //         .done()
        //         .eval(modulus)
        //         .unwrap();
        //     outputs.get_output(add2)
        // }
        let program = ProgramParser::new().parse(r#"
            type [3] = BoundedInt<0, 79228162514264337593543950335> [storable: true, drop: true, dup: true, zero_sized: false];
            type [53] = Const<[22], 29721761890975875353235833581453094220424382983267374> [storable: false, drop: false, dup: false, zero_sized: false];
            type [52] = Const<[22], 138583295661092166701491297054433349032460315956105119041111996301516236132> [storable: false, drop: false, dup: false, zero_sized: false];
            type [51] = Const<[22], 30828113188794245257250221355944970489240709081949230> [storable: false, drop: false, dup: false, zero_sized: false];
            type [26] = Struct<ut@core::circuit::u384, [3], [3], [3], [3]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [27] = Struct<ut@Tuple, [26]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [38] = U96LimbsLtGuarantee<1> [storable: true, drop: false, dup: false, zero_sized: false];
            type [37] = U96LimbsLtGuarantee<2> [storable: true, drop: false, dup: false, zero_sized: false];
            type [36] = U96LimbsLtGuarantee<3> [storable: true, drop: false, dup: false, zero_sized: false];
            type [9] = AddModGate<[10], [10]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [35] = U96LimbsLtGuarantee<4> [storable: true, drop: false, dup: false, zero_sized: false];
            type [10] = InverseGate<[11]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [11] = SubModGate<[12], [13]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [34] = CircuitFailureGuarantee [storable: true, drop: false, dup: false, zero_sized: false];
            type [13] = MulModGate<[14], [15]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [12] = SubModGate<[18], [17]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [15] = AddModGate<[16], [17]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [17] = CircuitInput<1> [storable: false, drop: false, dup: false, zero_sized: true];
            type [16] = CircuitInput<0> [storable: false, drop: false, dup: false, zero_sized: true];
            type [18] = InverseGate<[13]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [14] = MulModGate<[15], [16]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [33] = CircuitPartialOutputs<[7]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [32] = CircuitOutputs<[7]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [50] = Const<[31], 1> [storable: false, drop: false, dup: false, zero_sized: false];
            type [31] = BoundedInt<1, 1> [storable: true, drop: true, dup: true, zero_sized: false];
            type [49] = Const<[30], 0> [storable: false, drop: false, dup: false, zero_sized: false];
            type [30] = BoundedInt<0, 0> [storable: true, drop: true, dup: true, zero_sized: false];
            type [29] = CircuitDescriptor<[7]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [48] = Const<[3], 8> [storable: false, drop: false, dup: false, zero_sized: false];
            type [47] = Const<[3], 0> [storable: false, drop: false, dup: false, zero_sized: false];
            type [46] = Const<[3], 7> [storable: false, drop: false, dup: false, zero_sized: false];
            type [45] = Const<[3], 5> [storable: false, drop: false, dup: false, zero_sized: false];
            type [1] = MulMod [storable: true, drop: false, dup: false, zero_sized: false];
            type [0] = AddMod [storable: true, drop: false, dup: false, zero_sized: false];
            type [24] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
            type [23] = Array<[22]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [25] = Struct<ut@Tuple, [24], [23]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [28] = Enum<ut@core::panics::PanicResult::<(core::circuit::u384,)>, [27], [25]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [44] = Const<[22], 26913677086973030051406221357623718750637972950955665348321109348> [storable: false, drop: false, dup: false, zero_sized: false];
            type [22] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [21] = CircuitData<[7]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [19] = U96Guarantee [storable: true, drop: false, dup: false, zero_sized: false];
            type [20] = Struct<ut@Tuple, [19], [19], [19], [19]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [43] = Const<[3], 3> [storable: false, drop: false, dup: false, zero_sized: false];
            type [42] = Const<[3], 2> [storable: false, drop: false, dup: false, zero_sized: false];
            type [41] = Const<[3], 9> [storable: false, drop: false, dup: false, zero_sized: false];
            type [7] = Circuit<[6]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [6] = Struct<ut@Tuple, [9]> [storable: false, drop: false, dup: false, zero_sized: true];
            type [8] = CircuitInputAccumulator<[7]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [2] = RangeCheck96 [storable: true, drop: false, dup: false, zero_sized: false];
            type [5] = CircuitModulus [storable: true, drop: true, dup: true, zero_sized: false];
            type [4] = Struct<ut@Tuple, [3], [3], [3], [3]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [40] = Const<[3], 14> [storable: false, drop: false, dup: false, zero_sized: false];
            type [39] = Const<[3], 17> [storable: false, drop: false, dup: false, zero_sized: false];

            libfunc [22] = const_as_immediate<[39]>;
            libfunc [23] = const_as_immediate<[40]>;
            libfunc [21] = struct_construct<[4]>;
            libfunc [43] = store_temp<[4]>;
            libfunc [20] = try_into_circuit_modulus;
            libfunc [24] = branch_align;
            libfunc [19] = init_circuit_data<[7]>;
            libfunc [25] = const_as_immediate<[41]>;
            libfunc [18] = into_u96_guarantee<[3]>;
            libfunc [26] = const_as_immediate<[42]>;
            libfunc [27] = const_as_immediate<[43]>;
            libfunc [17] = struct_construct<[20]>;
            libfunc [44] = store_temp<[8]>;
            libfunc [45] = store_temp<[20]>;
            libfunc [46] = store_temp<[2]>;
            libfunc [16] = add_circuit_input<[7]>;
            libfunc [28] = drop<[21]>;
            libfunc [29] = drop<[5]>;
            libfunc [4] = array_new<[22]>;
            libfunc [30] = const_as_immediate<[44]>;
            libfunc [47] = store_temp<[22]>;
            libfunc [3] = array_append<[22]>;
            libfunc [2] = struct_construct<[24]>;
            libfunc [1] = struct_construct<[25]>;
            libfunc [0] = enum_init<[28], 1>;
            libfunc [48] = store_temp<[0]>;
            libfunc [49] = store_temp<[1]>;
            libfunc [50] = store_temp<[28]>;
            libfunc [31] = const_as_immediate<[45]>;
            libfunc [32] = const_as_immediate<[46]>;
            libfunc [33] = const_as_immediate<[47]>;
            libfunc [34] = const_as_immediate<[48]>;
            libfunc [15] = get_circuit_descriptor<[7]>;
            libfunc [35] = const_as_immediate<[49]>;
            libfunc [36] = const_as_immediate<[50]>;
            libfunc [51] = store_temp<[30]>;
            libfunc [52] = store_temp<[31]>;
            libfunc [14] = eval_circuit<[7]>;
            libfunc [13] = get_circuit_output<[7], [9]>;
            libfunc [9] = u96_limbs_less_than_guarantee_verify<4>;
            libfunc [8] = u96_limbs_less_than_guarantee_verify<3>;
            libfunc [7] = u96_limbs_less_than_guarantee_verify<2>;
            libfunc [6] = u96_single_limb_less_than_guarantee_verify;
            libfunc [53] = store_temp<[19]>;
            libfunc [37] = jump;
            libfunc [5] = u96_guarantee_verify;
            libfunc [12] = struct_construct<[27]>;
            libfunc [11] = enum_init<[28], 0>;
            libfunc [38] = drop<[33]>;
            libfunc [39] = const_as_immediate<[51]>;
            libfunc [10] = circuit_failure_guarantee_verify;
            libfunc [54] = store_temp<[23]>;
            libfunc [40] = drop<[8]>;
            libfunc [41] = const_as_immediate<[52]>;
            libfunc [42] = const_as_immediate<[53]>;

            [22]() -> ([3]); // 0
            [23]() -> ([4]); // 1
            [23]() -> ([5]); // 2
            [23]() -> ([6]); // 3
            [21]([3], [4], [5], [6]) -> ([7]); // 4
            [43]([7]) -> ([7]); // 5
            [20]([7]) { fallthrough([8]) 142() }; // 6
            [24]() -> (); // 7
            [19]([2]) -> ([9], [10]); // 8
            [25]() -> ([11]); // 9
            [18]([11]) -> ([12]); // 10
            [26]() -> ([13]); // 11
            [18]([13]) -> ([14]); // 12
            [25]() -> ([15]); // 13
            [18]([15]) -> ([16]); // 14
            [27]() -> ([17]); // 15
            [18]([17]) -> ([18]); // 16
            [17]([12], [14], [16], [18]) -> ([19]); // 17
            [44]([10]) -> ([10]); // 18
            [45]([19]) -> ([19]); // 19
            [46]([9]) -> ([9]); // 20
            [16]([10], [19]) { fallthrough([20]) 37([21]) }; // 21
            [24]() -> (); // 22
            [28]([20]) -> (); // 23
            [29]([8]) -> (); // 24
            [4]() -> ([22]); // 25
            [30]() -> ([23]); // 26
            [47]([23]) -> ([23]); // 27
            [3]([22], [23]) -> ([24]); // 28
            [2]() -> ([25]); // 29
            [1]([25], [24]) -> ([26]); // 30
            [0]([26]) -> ([27]); // 31
            [48]([0]) -> ([0]); // 32
            [49]([1]) -> ([1]); // 33
            [46]([9]) -> ([9]); // 34
            [50]([27]) -> ([27]); // 35
            return([0], [1], [9], [27]); // 36
            [24]() -> (); // 37
            [31]() -> ([28]); // 38
            [18]([28]) -> ([29]); // 39
            [32]() -> ([30]); // 40
            [18]([30]) -> ([31]); // 41
            [33]() -> ([32]); // 42
            [18]([32]) -> ([33]); // 43
            [34]() -> ([34]); // 44
            [18]([34]) -> ([35]); // 45
            [17]([29], [31], [33], [35]) -> ([36]); // 46
            [45]([36]) -> ([36]); // 47
            [16]([21], [36]) { fallthrough([37]) 127([38]) }; // 48
            [24]() -> (); // 49
            [15]() -> ([39]); // 50
            [35]() -> ([40]); // 51
            [36]() -> ([41]); // 52
            [51]([40]) -> ([40]); // 53
            [52]([41]) -> ([41]); // 54
            [14]([0], [1], [39], [37], [8], [40], [41]) { fallthrough([42], [43], [44]) 85([45], [46], [47], [48]) }; // 55
            [24]() -> (); // 56
            [13]([44]) -> ([49], [50]); // 57
            [48]([42]) -> ([42]); // 58
            [49]([43]) -> ([43]); // 59
            [9]([50]) { fallthrough([51]) 75([52]) }; // 60
            [24]() -> (); // 61
            [8]([51]) { fallthrough([53]) 72([54]) }; // 62
            [24]() -> (); // 63
            [7]([53]) { fallthrough([55]) 69([56]) }; // 64
            [24]() -> (); // 65
            [6]([55]) -> ([57]); // 66
            [53]([57]) -> ([58]); // 67
            [37]() { 77() }; // 68
            [24]() -> (); // 69
            [53]([56]) -> ([58]); // 70
            [37]() { 77() }; // 71
            [24]() -> (); // 72
            [53]([54]) -> ([58]); // 73
            [37]() { 77() }; // 74
            [24]() -> (); // 75
            [53]([52]) -> ([58]); // 76
            [5]([9], [58]) -> ([59]); // 77
            [12]([49]) -> ([60]); // 78
            [11]([60]) -> ([61]); // 79
            [48]([42]) -> ([42]); // 80
            [49]([43]) -> ([43]); // 81
            [46]([59]) -> ([59]); // 82
            [50]([61]) -> ([61]); // 83
            return([42], [43], [59], [61]); // 84
            [24]() -> (); // 85
            [38]([47]) -> (); // 86
            [4]() -> ([62]); // 87
            [39]() -> ([63]); // 88
            [47]([63]) -> ([63]); // 89
            [3]([62], [63]) -> ([64]); // 90
            [35]() -> ([65]); // 91
            [36]() -> ([66]); // 92
            [49]([46]) -> ([46]); // 93
            [51]([65]) -> ([65]); // 94
            [52]([66]) -> ([66]); // 95
            [10]([9], [46], [48], [65], [66]) -> ([67], [68], [69]); // 96
            [48]([45]) -> ([45]); // 97
            [54]([64]) -> ([64]); // 98
            [46]([67]) -> ([67]); // 99
            [49]([68]) -> ([68]); // 100
            [9]([69]) { fallthrough([70]) 116([71]) }; // 101
            [24]() -> (); // 102
            [8]([70]) { fallthrough([72]) 113([73]) }; // 103
            [24]() -> (); // 104
            [7]([72]) { fallthrough([74]) 110([75]) }; // 105
            [24]() -> (); // 106
            [6]([74]) -> ([76]); // 107
            [53]([76]) -> ([77]); // 108
            [37]() { 118() }; // 109
            [24]() -> (); // 110
            [53]([75]) -> ([77]); // 111
            [37]() { 118() }; // 112
            [24]() -> (); // 113
            [53]([73]) -> ([77]); // 114
            [37]() { 118() }; // 115
            [24]() -> (); // 116
            [53]([71]) -> ([77]); // 117
            [5]([67], [77]) -> ([78]); // 118
            [2]() -> ([79]); // 119
            [1]([79], [64]) -> ([80]); // 120
            [0]([80]) -> ([81]); // 121
            [48]([45]) -> ([45]); // 122
            [49]([68]) -> ([68]); // 123
            [46]([78]) -> ([78]); // 124
            [50]([81]) -> ([81]); // 125
            return([45], [68], [78], [81]); // 126
            [24]() -> (); // 127
            [40]([38]) -> (); // 128
            [29]([8]) -> (); // 129
            [4]() -> ([82]); // 130
            [41]() -> ([83]); // 131
            [47]([83]) -> ([83]); // 132
            [3]([82], [83]) -> ([84]); // 133
            [2]() -> ([85]); // 134
            [1]([85], [84]) -> ([86]); // 135
            [0]([86]) -> ([87]); // 136
            [48]([0]) -> ([0]); // 137
            [49]([1]) -> ([1]); // 138
            [46]([9]) -> ([9]); // 139
            [50]([87]) -> ([87]); // 140
            return([0], [1], [9], [87]); // 141
            [24]() -> (); // 142
            [4]() -> ([88]); // 143
            [42]() -> ([89]); // 144
            [47]([89]) -> ([89]); // 145
            [3]([88], [89]) -> ([90]); // 146
            [2]() -> ([91]); // 147
            [1]([91], [90]) -> ([92]); // 148
            [0]([92]) -> ([93]); // 149
            [48]([0]) -> ([0]); // 150
            [49]([1]) -> ([1]); // 151
            [46]([2]) -> ([2]); // 152
            [50]([93]) -> ([93]); // 153
            return([0], [1], [2], [93]); // 154

            [0]@0([0]: [0], [1]: [1], [2]: [2]) -> ([0], [1], [2], [28]);
        "#).map_err(|e| e.to_string()).unwrap();

        let return_value = run_sierra_program(&program, &[]).return_value;

        assert_eq!(
            jit_enum!(
                0,
                jit_struct!(u384([
                    "0x76956587ccb74125e760fdf3",
                    "0xe8c82ede90011c6adc4b5cfa",
                    "0xaf4bed7eef975ff1941fdf3d",
                    "0x7"
                ]))
            ),
            return_value
        );
    }
}
