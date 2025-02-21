#![cfg(feature = "with-trace-dump")]

use crate::{
    error::{Error, Result},
    runtime,
    utils::BlockExt,
};
use cairo_lang_sierra::{
    ids::{ConcreteTypeId, VarId},
    program::StatementIdx,
};
use melior::{
    dialect::{llvm, memref, ods},
    ir::{
        attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        operation::OperationBuilder,
        r#type::{IntegerType, MemRefType},
        Attribute, Block, BlockLike, Location, Module, Region, Value,
    },
    Context,
};
use std::{collections::HashSet, ffi::c_void, ptr};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TraceBinding {
    State,
    Push,
    TraceId,
}

impl TraceBinding {
    pub const fn symbol(self) -> &'static str {
        match self {
            TraceBinding::State => "cairo_native__trace_dump__state",
            TraceBinding::Push => "cairo_native__trace_dump__push",
            TraceBinding::TraceId => "TRACE_DUMP__TRACE_ID",
        }
    }

    const fn function_ptr(self) -> *const () {
        match self {
            TraceBinding::State => {
                runtime::trace_dump::cairo_native__trace_dump__state as *const ()
            }
            TraceBinding::Push => runtime::trace_dump::cairo_native__trace_dump__push as *const (),
            // it has no function pointer, as its a global constant
            TraceBinding::TraceId => ptr::null(),
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TraceDumpMeta {
    active_map: HashSet<TraceBinding>,
}

impl TraceDumpMeta {
    /// Register the global for the given binding, if not yet registered, and return
    /// a pointer to the stored value.
    ///
    /// For the function to be available, `setup_runtime` must be called before running the module
    fn build_function<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        location: Location<'c>,
        binding: TraceBinding,
    ) -> Result<Value<'c, 'a>> {
        if self.active_map.insert(binding) {
            module.body().append_operation(
                ods::llvm::mlir_global(
                    context,
                    Region::new(),
                    TypeAttribute::new(llvm::r#type::pointer(context, 0)),
                    StringAttribute::new(context, binding.symbol()),
                    Attribute::parse(context, "#llvm.linkage<weak>")
                        .ok_or(Error::ParseAttributeError)?,
                    location,
                )
                .into(),
            );
        }

        let global_address = block.append_op_result(
            ods::llvm::mlir_addressof(
                context,
                llvm::r#type::pointer(context, 0),
                FlatSymbolRefAttribute::new(context, binding.symbol()),
                location,
            )
            .into(),
        )?;

        block.load(
            context,
            location,
            global_address,
            llvm::r#type::pointer(context, 0),
        )
    }

    #[allow(clippy::too_many_arguments)]
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
        let trace_id = self.build_trace_id(context, module, block, location)?;

        let var_id = block.const_int(context, location, var_id.id, 64)?;
        let value_ty = block.const_int(context, location, value_ty.id, 64)?;

        let function =
            self.build_function(context, module, block, location, TraceBinding::State)?;
        block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .add_operands(&[trace_id, var_id, value_ty, value_ptr])
                .build()?,
        );

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
        let trace_id = self.build_trace_id(context, module, block, location)?;
        let statement_idx = block.const_int(context, location, statement_idx.0, 64)?;

        let function = self.build_function(context, module, block, location, TraceBinding::Push)?;

        block.append_operation(
            OperationBuilder::new("llvm.call", location)
                .add_operands(&[function])
                .add_operands(&[trace_id, statement_idx])
                .build()?,
        );

        Ok(())
    }

    pub fn build_trace_id<'c, 'a>(
        &mut self,
        context: &'c Context,
        module: &Module,
        block: &'a Block<'c>,
        location: Location<'c>,
    ) -> Result<Value<'c, 'a>> {
        if self.active_map.insert(TraceBinding::TraceId) {
            module.body().append_operation(memref::global(
                context,
                TraceBinding::TraceId.symbol(),
                None,
                MemRefType::new(IntegerType::new(context, 64).into(), &[], None, None),
                None,
                false,
                None,
                location,
            ));
        }

        let trace_id_ptr = block
            .append_op_result(memref::get_global(
                context,
                TraceBinding::TraceId.symbol(),
                MemRefType::new(IntegerType::new(context, 64).into(), &[], None, None),
                location,
            ))
            .unwrap();

        block.append_op_result(memref::load(trace_id_ptr, &[], location))
    }
}

pub fn setup_runtime(find_symbol_ptr: impl Fn(&str) -> Option<*mut c_void>) {
    let bindings = &[TraceBinding::State, TraceBinding::Push];

    for binding in bindings {
        if let Some(global) = find_symbol_ptr(binding.symbol()) {
            let global = global.cast::<*const ()>();
            unsafe { *global = binding.function_ptr() };
        }
    }
}
