//! # `Circuit` type

use std::alloc::Layout;

use super::WithSelf;
use crate::{
    error::Result,
    metadata::MetadataStorage,
    utils::{get_integer_layout, layout_repeat},
};
use cairo_lang_sierra::{
    extensions::{
        circuit::CircuitTypeConcrete,
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        types::InfoOnlyConcreteType,
    },
    program::GenericArg,
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm,
    ir::{r#type::IntegerType, Module, Type},
    Context,
};

pub const CIRCUIT_INPUT_SIZE: usize = 384;

/// Build the MLIR type.
///
/// Check out [the module](self) for more info.
pub fn build<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    selector: WithSelf<CircuitTypeConcrete>,
) -> Result<Type<'ctx>> {
    match &*selector {
        CircuitTypeConcrete::CircuitModulus(_) => {
            Ok(IntegerType::new(context, CIRCUIT_INPUT_SIZE as u32).into())
        }
        CircuitTypeConcrete::U96Guarantee(_) => Ok(IntegerType::new(context, 96).into()),
        CircuitTypeConcrete::CircuitInputAccumulator(info) => build_circuit_accumulator(
            context,
            module,
            registry,
            metadata,
            WithSelf::new(selector.self_ty(), info),
        ),
        CircuitTypeConcrete::CircuitData(info) => build_circuit_data(
            context,
            module,
            registry,
            metadata,
            WithSelf::new(selector.self_ty(), info),
        ),
        CircuitTypeConcrete::CircuitOutputs(info) => build_circuit_outputs(
            context,
            module,
            registry,
            metadata,
            WithSelf::new(selector.self_ty(), info),
        ),
        // builtins
        CircuitTypeConcrete::AddMod(_)
        | CircuitTypeConcrete::U96LimbsLessThanGuarantee(_)
        | CircuitTypeConcrete::MulMod(_) => Ok(IntegerType::new(context, 64).into()),
        // noops
        CircuitTypeConcrete::CircuitDescriptor(_)
        | CircuitTypeConcrete::CircuitFailureGuarantee(_)
        | CircuitTypeConcrete::CircuitPartialOutputs(_) => {
            Ok(llvm::r#type::array(IntegerType::new(context, 8).into(), 0))
        }
        // phantoms
        // todo! swap unreachable with debug assert and error return
        CircuitTypeConcrete::Circuit(_)
        | CircuitTypeConcrete::AddModGate(_)
        | CircuitTypeConcrete::SubModGate(_)
        | CircuitTypeConcrete::MulModGate(_)
        | CircuitTypeConcrete::InverseGate(_)
        | CircuitTypeConcrete::CircuitInput(_) => unreachable!(),
    }
}

pub fn build_circuit_accumulator<'ctx>(
    context: &'ctx Context,
    _module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _metadata: &mut MetadataStorage,
    info: WithSelf<InfoOnlyConcreteType>,
) -> Result<Type<'ctx>> {
    // todo! swap unreachable with debug assert and error return
    let Some(generic_arg) = info.info.long_id.generic_args.first() else {
        unreachable!();
    };
    let GenericArg::Type(circuit_type_id) = generic_arg else {
        unreachable!();
    };
    let CoreTypeConcrete::Circuit(CircuitTypeConcrete::Circuit(circuit)) =
        registry.get_type(circuit_type_id)?
    else {
        unreachable!()
    };

    let n_inputs = circuit.circuit_info.n_inputs;

    let fields = vec![
        IntegerType::new(context, 64).into(),
        llvm::r#type::array(
            IntegerType::new(context, CIRCUIT_INPUT_SIZE as u32).into(),
            n_inputs as u32 - 1,
        ),
    ];

    Ok(llvm::r#type::r#struct(context, &fields, false))
}

pub fn build_circuit_data<'ctx>(
    context: &'ctx Context,
    _module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _metadata: &mut MetadataStorage,
    info: WithSelf<InfoOnlyConcreteType>,
) -> Result<Type<'ctx>> {
    // todo! swap unreachable with debug assert and error return
    let Some(generic_arg) = info.info.long_id.generic_args.first() else {
        unreachable!();
    };
    let GenericArg::Type(circuit_type_id) = generic_arg else {
        unreachable!();
    };
    let CoreTypeConcrete::Circuit(CircuitTypeConcrete::Circuit(circuit)) =
        registry.get_type(circuit_type_id)?
    else {
        unreachable!()
    };

    let n_inputs = circuit.circuit_info.n_inputs;

    Ok(llvm::r#type::array(
        IntegerType::new(context, CIRCUIT_INPUT_SIZE as u32).into(),
        n_inputs as u32,
    ))
}

pub fn build_circuit_outputs<'ctx>(
    context: &'ctx Context,
    _module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    _metadata: &mut MetadataStorage,
    info: WithSelf<InfoOnlyConcreteType>,
) -> Result<Type<'ctx>> {
    // todo! swap unreachable with debug assert and error return
    let Some(generic_arg) = info.info.long_id.generic_args.first() else {
        unreachable!();
    };
    let GenericArg::Type(circuit_type_id) = generic_arg else {
        unreachable!();
    };
    let CoreTypeConcrete::Circuit(CircuitTypeConcrete::Circuit(circuit)) =
        registry.get_type(circuit_type_id)?
    else {
        unreachable!()
    };

    let n_gates = circuit.circuit_info.values.len();

    Ok(llvm::r#type::array(
        IntegerType::new(context, CIRCUIT_INPUT_SIZE as u32).into(),
        n_gates as u32,
    ))
}

pub fn is_complex(info: &CircuitTypeConcrete) -> bool {
    match *info {
        CircuitTypeConcrete::AddMod(_)
        | CircuitTypeConcrete::MulMod(_)
        | CircuitTypeConcrete::AddModGate(_)
        | CircuitTypeConcrete::SubModGate(_)
        | CircuitTypeConcrete::MulModGate(_)
        | CircuitTypeConcrete::U96Guarantee(_)
        | CircuitTypeConcrete::InverseGate(_)
        | CircuitTypeConcrete::U96LimbsLessThanGuarantee(_)
        | CircuitTypeConcrete::CircuitModulus(_)
        | CircuitTypeConcrete::CircuitInput(_)
        | CircuitTypeConcrete::Circuit(_)
        | CircuitTypeConcrete::CircuitDescriptor(_)
        | CircuitTypeConcrete::CircuitFailureGuarantee(_) => false,

        CircuitTypeConcrete::CircuitInputAccumulator(_)
        | CircuitTypeConcrete::CircuitPartialOutputs(_)
        | CircuitTypeConcrete::CircuitData(_)
        | CircuitTypeConcrete::CircuitOutputs(_) => true,
    }
}

pub fn is_zst(info: &CircuitTypeConcrete) -> bool {
    match *info {
        CircuitTypeConcrete::AddModGate(_)
        | CircuitTypeConcrete::SubModGate(_)
        | CircuitTypeConcrete::MulModGate(_)
        | CircuitTypeConcrete::CircuitInput(_)
        | CircuitTypeConcrete::InverseGate(_)
        | CircuitTypeConcrete::U96LimbsLessThanGuarantee(_)
        | CircuitTypeConcrete::Circuit(_)
        | CircuitTypeConcrete::CircuitDescriptor(_)
        | CircuitTypeConcrete::CircuitFailureGuarantee(_) => true,

        CircuitTypeConcrete::AddMod(_)
        | CircuitTypeConcrete::CircuitModulus(_)
        | CircuitTypeConcrete::U96Guarantee(_)
        | CircuitTypeConcrete::MulMod(_)
        | CircuitTypeConcrete::CircuitInputAccumulator(_)
        | CircuitTypeConcrete::CircuitPartialOutputs(_)
        | CircuitTypeConcrete::CircuitData(_)
        | CircuitTypeConcrete::CircuitOutputs(_) => false,
    }
}

pub fn layout(
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    info: &CircuitTypeConcrete,
) -> Result<Layout> {
    match info {
        CircuitTypeConcrete::AddMod(_) | CircuitTypeConcrete::MulMod(_) => {
            Ok(get_integer_layout(64))
        }
        CircuitTypeConcrete::CircuitModulus(_) => Ok(get_integer_layout(CIRCUIT_INPUT_SIZE as u32)),
        CircuitTypeConcrete::U96Guarantee(_) => Ok(get_integer_layout(96)),

        CircuitTypeConcrete::AddModGate(_)
        | CircuitTypeConcrete::SubModGate(_)
        | CircuitTypeConcrete::MulModGate(_)
        | CircuitTypeConcrete::CircuitInput(_)
        | CircuitTypeConcrete::InverseGate(_)
        | CircuitTypeConcrete::U96LimbsLessThanGuarantee(_)
        | CircuitTypeConcrete::Circuit(_)
        | CircuitTypeConcrete::CircuitDescriptor(_)
        | CircuitTypeConcrete::CircuitFailureGuarantee(_) => Ok(Layout::new::<()>()),

        CircuitTypeConcrete::CircuitData(info) => {
            // todo! swap unreachable with debug assert and error return
            let Some(generic_arg) = info.info.long_id.generic_args.first() else {
                unreachable!();
            };
            let GenericArg::Type(circuit_type_id) = generic_arg else {
                unreachable!();
            };
            let CoreTypeConcrete::Circuit(CircuitTypeConcrete::Circuit(circuit)) =
                registry.get_type(circuit_type_id)?
            else {
                unreachable!()
            };

            let n_inputs = circuit.circuit_info.n_inputs;

            // todo! fix calculation
            let u384_layout = Layout::from_size_align(
                CIRCUIT_INPUT_SIZE >> 3,
                (CIRCUIT_INPUT_SIZE >> 3).min(16),
            )?;

            let layout = layout_repeat(&u384_layout, n_inputs)?.0;

            Ok(layout)
        }
        CircuitTypeConcrete::CircuitOutputs(info) => {
            // todo! swap unreachable with debug assert and error return
            let Some(generic_arg) = info.info.long_id.generic_args.first() else {
                unreachable!();
            };
            let GenericArg::Type(circuit_type_id) = generic_arg else {
                unreachable!();
            };
            let CoreTypeConcrete::Circuit(CircuitTypeConcrete::Circuit(circuit)) =
                registry.get_type(circuit_type_id)?
            else {
                unreachable!()
            };

            let n_gates = circuit.circuit_info.values.len();

            // todo! fix calculation
            let u384_layout = Layout::from_size_align(
                CIRCUIT_INPUT_SIZE >> 3,
                (CIRCUIT_INPUT_SIZE >> 3).min(16),
            )?;

            let layout = layout_repeat(&u384_layout, n_gates)?.0;

            Ok(layout)
        }
        CircuitTypeConcrete::CircuitPartialOutputs(_) => todo!(),
        CircuitTypeConcrete::CircuitInputAccumulator(info) => {
            // todo! swap unreachable with debug assert and error return
            let Some(generic_arg) = info.info.long_id.generic_args.first() else {
                unreachable!();
            };
            let GenericArg::Type(circuit_type_id) = generic_arg else {
                unreachable!();
            };
            let CoreTypeConcrete::Circuit(CircuitTypeConcrete::Circuit(circuit)) =
                registry.get_type(circuit_type_id)?
            else {
                unreachable!()
            };

            let n_inputs = circuit.circuit_info.n_inputs;

            let length_layout = get_integer_layout(64);

            // todo! fix calculation
            let u384_layout = Layout::from_size_align(
                CIRCUIT_INPUT_SIZE >> 3,
                (CIRCUIT_INPUT_SIZE >> 3).min(16),
            )?;
            let inputs_layout = layout_repeat(&u384_layout, n_inputs - 1)?.0;
            let layout = length_layout.extend(inputs_layout)?.0;

            Ok(layout)
        }
    }
}
