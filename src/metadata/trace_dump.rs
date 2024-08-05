#![cfg(feature = "with-trace-dump")]

use crate::{block_ext::BlockExt, error::Result, starknet::ArrayAbi, types::TypeBuilder};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        types::InfoAndTypeConcreteType,
    },
    ids::{ConcreteTypeId, VarId},
    program::StatementIdx,
    program_registry::ProgramRegistry,
};
use cairo_lang_utils::ordered_hash_map::OrderedHashMap;
use melior::{
    dialect::{func, llvm, ods},
    ir::{
        attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        r#type::{FunctionType, IntegerType},
        Block, Identifier, Location, Module, Region, Value,
    },
    Context, ExecutionEngine,
};
use sierra_emu::{ProgramTrace, StateDump};
use starknet_types_core::felt::Felt;
use std::{
    borrow::Borrow,
    cell::RefCell,
    collections::HashSet,
    mem::swap,
    sync::{Arc, Weak},
};

pub struct InternalState {
    trace: RefCell<ProgramTrace>,
    state: RefCell<OrderedHashMap<VarId, sierra_emu::Value>>,
    registry: ProgramRegistry<CoreType, CoreLibfunc>,
}

impl InternalState {
    pub fn new(registry: ProgramRegistry<CoreType, CoreLibfunc>) -> Self {
        Self {
            trace: RefCell::default(),
            state: RefCell::default(),
            registry,
        }
    }

    pub fn extract(&self) -> ProgramTrace {
        self.trace.borrow().clone()
    }
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
enum TraceBinding {
    State,
    Push,
}

pub struct TraceDump {
    trace: Arc<InternalState>,
    bindings: HashSet<TraceBinding>,
}

impl TraceDump {
    pub fn new(registry: ProgramRegistry<CoreType, CoreLibfunc>) -> Self {
        Self {
            trace: Arc::new(InternalState::new(registry)),
            bindings: HashSet::default(),
        }
    }
    pub fn internal_state(&self) -> Arc<InternalState> {
        self.trace.clone()
    }

    pub fn build_state(
        &mut self,
        context: &Context,
        module: &Module,
        block: &Block,
        var_id: &VarId,
        value_ty: &ConcreteTypeId,
        value_ptr: Value,
        location: Location,
    ) -> Result<()> {
        if self.bindings.insert(TraceBinding::State) {
            module.body().append_operation(func::func(
                context,
                StringAttribute::new(context, "__trace__state"),
                TypeAttribute::new(
                    FunctionType::new(
                        context,
                        &[
                            // internal state
                            llvm::r#type::pointer(context, 0),
                            // var id
                            IntegerType::new(context, 64).into(),
                            // value type
                            IntegerType::new(context, 64).into(),
                            // value ptr
                            llvm::r#type::pointer(context, 0),
                        ],
                        &[],
                    )
                    .into(),
                ),
                Region::new(),
                &[(
                    Identifier::new(context, "sym_visibility"),
                    StringAttribute::new(context, "private").into(),
                )],
                Location::unknown(context),
            ));
        }

        let state = {
            let state = block.const_int(
                context,
                location,
                Arc::downgrade(&self.trace).into_raw() as i64,
                64,
            )?;
            block.append_op_result(
                ods::llvm::inttoptr(context, llvm::r#type::pointer(context, 0), state, location)
                    .into(),
            )?
        };
        let var_id = block.const_int(context, location, var_id.id, 64).unwrap();
        let value_id = block.const_int(context, location, value_ty.id, 64).unwrap();

        block.append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, "__trace__state"),
            &[state, var_id, value_id, value_ptr],
            &[],
            location,
        ));

        Ok(())
    }

    pub fn build_push(
        &mut self,
        context: &Context,
        module: &Module,
        block: &Block,
        statement_idx: StatementIdx,
        location: Location,
    ) -> Result<()> {
        if self.bindings.insert(TraceBinding::Push) {
            module.body().append_operation(func::func(
                context,
                StringAttribute::new(context, "__trace__push"),
                TypeAttribute::new(
                    FunctionType::new(
                        context,
                        &[
                            llvm::r#type::pointer(context, 0),
                            IntegerType::new(context, 64).into(),
                        ],
                        &[],
                    )
                    .into(),
                ),
                Region::new(),
                &[(
                    Identifier::new(context, "sym_visibility"),
                    StringAttribute::new(context, "private").into(),
                )],
                Location::unknown(context),
            ));
        }

        let state = {
            let state = block.const_int(
                context,
                location,
                Arc::downgrade(&self.trace).into_raw() as i64,
                64,
            )?;
            block.append_op_result(
                ods::llvm::inttoptr(context, llvm::r#type::pointer(context, 0), state, location)
                    .into(),
            )?
        };
        let statement_idx = block.const_int(context, location, statement_idx.0, 64)?;

        block.append_operation(func::call(
            context,
            FlatSymbolRefAttribute::new(context, "__trace__push"),
            &[state, statement_idx],
            &[],
            location,
        ));

        Ok(())
    }

    pub fn register_impls(&self, engine: &ExecutionEngine) {
        if self.bindings.contains(&TraceBinding::State) {
            unsafe {
                engine.register_symbol("__trace__state", trace_state as *mut ());
            }
        }

        if !self.bindings.is_empty() {
            unsafe {
                engine.register_symbol(
                    "__trace__push",
                    trace_push as *const fn(*const InternalState) -> () as *mut (),
                );
            }
        }
    }
}

extern "C" fn trace_state(
    state: *const InternalState,
    var_id: u64,
    value_type_id: u64,
    value_ptr: *const (),
) {
    let Some(state) = unsafe { Weak::from_raw(state) }.upgrade() else {
        return;
    };

    state.state.borrow_mut().insert(
        VarId::new(var_id),
        value_from_pointer(
            state.borrow(),
            &ConcreteTypeId::new(value_type_id),
            value_ptr,
        ),
    );
}

fn value_from_pointer(
    state: &InternalState,
    value_type_id: &ConcreteTypeId,
    value_ptr: *const (),
) -> sierra_emu::Value {
    let value_type = state.registry.get_type(value_type_id).unwrap();

    match value_type {
        CoreTypeConcrete::Array(InfoAndTypeConcreteType {
            ty: inner_type_id, ..
        }) => {
            let inner_size = state
                .registry
                .get_type(inner_type_id)
                .unwrap()
                .layout(&state.registry)
                .unwrap()
                .pad_to_align()
                .size();

            let array = unsafe { value_ptr.cast::<ArrayAbi<()>>().as_ref().unwrap() };

            let length = (array.until - array.since) as usize;
            let start_ptr = unsafe { array.ptr.byte_add(array.since as usize * inner_size) };
            let mut data = Vec::with_capacity(length);

            for i in 0..length {
                let current_ptr = unsafe { start_ptr.byte_add(i * inner_size) };
                data.push(value_from_pointer(state, inner_type_id, current_ptr))
            }

            sierra_emu::Value::Array {
                ty: inner_type_id.clone(),
                data,
            }
        }
        CoreTypeConcrete::Coupon(_) => todo!(),
        CoreTypeConcrete::Bitwise(_) => todo!(),
        CoreTypeConcrete::Box(_) => todo!(),
        CoreTypeConcrete::Circuit(_) => todo!(),
        CoreTypeConcrete::Const(_) => todo!(),
        CoreTypeConcrete::EcOp(_) => todo!(),
        CoreTypeConcrete::EcPoint(_) => todo!(),
        CoreTypeConcrete::EcState(_) => todo!(),
        CoreTypeConcrete::Felt252(_) => {
            let bytes = unsafe { value_ptr.cast::<[u8; 32]>().as_ref().unwrap() };
            sierra_emu::Value::Felt(Felt::from_bytes_le(bytes))
        }
        CoreTypeConcrete::GasBuiltin(_) => todo!(),
        CoreTypeConcrete::BuiltinCosts(_) => todo!(),
        CoreTypeConcrete::Uint8(_) => {
            let bytes = unsafe { value_ptr.cast::<[u8; 1]>().as_ref().unwrap() };
            sierra_emu::Value::U8(u8::from_le_bytes(*bytes))
        }
        CoreTypeConcrete::Uint16(_) => todo!(),
        CoreTypeConcrete::Uint32(_) => {
            let bytes = unsafe { value_ptr.cast::<[u8; 4]>().as_ref().unwrap() };
            sierra_emu::Value::U32(u32::from_le_bytes(*bytes))
        }
        CoreTypeConcrete::Uint64(_) => todo!(),
        CoreTypeConcrete::Uint128(_) => {
            let bytes = unsafe { value_ptr.cast::<[u8; 16]>().as_ref().unwrap() };
            sierra_emu::Value::U128(u128::from_le_bytes(*bytes))
        }
        CoreTypeConcrete::Uint128MulGuarantee(_) => todo!(),
        CoreTypeConcrete::Sint8(_) => todo!(),
        CoreTypeConcrete::Sint16(_) => todo!(),
        CoreTypeConcrete::Sint32(_) => todo!(),
        CoreTypeConcrete::Sint64(_) => todo!(),
        CoreTypeConcrete::Sint128(_) => todo!(),
        CoreTypeConcrete::NonZero(_) => todo!(),
        CoreTypeConcrete::Nullable(_) => todo!(),
        CoreTypeConcrete::RangeCheck(_) => todo!(),
        CoreTypeConcrete::RangeCheck96(_) => todo!(),
        CoreTypeConcrete::Uninitialized(_) => todo!(),
        CoreTypeConcrete::Enum(_) => todo!(),
        CoreTypeConcrete::Struct(_) => todo!(),
        CoreTypeConcrete::Felt252Dict(_) => todo!(),
        CoreTypeConcrete::Felt252DictEntry(_) => todo!(),
        CoreTypeConcrete::SquashedFelt252Dict(_) => todo!(),
        CoreTypeConcrete::Pedersen(_) => todo!(),
        CoreTypeConcrete::Poseidon(_) => todo!(),
        CoreTypeConcrete::Span(_) => todo!(),
        CoreTypeConcrete::StarkNet(_) => todo!(),
        CoreTypeConcrete::SegmentArena(_) => todo!(),
        CoreTypeConcrete::Snapshot(_) => todo!(),
        CoreTypeConcrete::Bytes31(_) => todo!(),
        CoreTypeConcrete::BoundedInt(_) => todo!(),
    }
}

extern "C" fn trace_push(state: *const InternalState, statement_idx: usize) {
    let state = unsafe { Weak::from_raw(state) };
    if let Some(state) = state.upgrade() {
        let mut items = OrderedHashMap::default();
        swap(&mut items, &mut *state.state.borrow_mut());

        state
            .trace
            .borrow_mut()
            .push(StateDump::new(StatementIdx(statement_idx), items));
    }
}
