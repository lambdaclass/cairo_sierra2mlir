use crate::{
    error::Error,
    execution_result::{ContractExecutionResult, ExecutionResult},
    metadata::{
        felt252_dict::Felt252DictOverrides, gas::GasMetadata, runtime_bindings::setup_runtime,
    },
    module::NativeModule,
    starknet::{DummySyscallHandler, StarknetSyscallHandler},
    utils::{create_engine, generate_function_name},
    values::Value,
    OptLevel,
};
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    ids::{ConcreteTypeId, FunctionId},
    program::FunctionSignature,
    program_registry::ProgramRegistry,
};
use libc::c_void;
use melior::{ir::Module, ExecutionEngine};
use starknet_types_core::felt::Felt;
use std::mem::transmute;

/// A MLIR JIT execution engine in the context of Cairo Native.
pub struct JitNativeExecutor<'m> {
    engine: ExecutionEngine,

    module: Module<'m>,
    registry: ProgramRegistry<CoreType, CoreLibfunc>,

    gas_metadata: GasMetadata,
    dict_overrides: Felt252DictOverrides,
}

unsafe impl Send for JitNativeExecutor<'_> {}
unsafe impl Sync for JitNativeExecutor<'_> {}

impl std::fmt::Debug for JitNativeExecutor<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitNativeExecutor")
            .field("module", &self.module)
            .field("gas_metadata", &self.gas_metadata)
            .finish()
    }
}

impl<'m> JitNativeExecutor<'m> {
    pub fn from_native_module(
        native_module: NativeModule<'m>,
        opt_level: OptLevel,
    ) -> Result<Self, Error> {
        let NativeModule {
            module,
            registry,
            mut metadata,
        } = native_module;

        let executor = Self {
            engine: create_engine(&module, &metadata, opt_level),
            module,
            registry,
            gas_metadata: metadata.remove().ok_or(Error::MissingMetadata)?,
            dict_overrides: metadata.remove().unwrap_or_default(),
        };

        setup_runtime(|name| executor.find_symbol_ptr(name));

        Ok(executor)
    }

    pub const fn program_registry(&self) -> &ProgramRegistry<CoreType, CoreLibfunc> {
        &self.registry
    }

    pub const fn module(&self) -> &Module<'m> {
        &self.module
    }

    /// Execute a program with the given params.
    pub fn invoke_dynamic(
        &self,
        function_id: &FunctionId,
        args: &[Value],
        gas: Option<u64>,
    ) -> Result<ExecutionResult, Error> {
        let available_gas = self
            .gas_metadata
            .get_initial_available_gas(function_id, gas)
            .map_err(crate::error::Error::GasMetadataError)?;

        #[cfg(feature = "with-profiler")]
        self.setup_profiling();

        super::invoke_dynamic(
            &self.registry,
            self.find_function_ptr(function_id),
            self.extract_signature(function_id)?,
            args,
            available_gas,
            Option::<DummySyscallHandler>::None,
            self.build_find_dict_overrides(),
        )
    }

    /// Execute a program with the given params.
    pub fn invoke_dynamic_with_syscall_handler(
        &self,
        function_id: &FunctionId,
        args: &[Value],
        gas: Option<u64>,
        syscall_handler: impl StarknetSyscallHandler,
    ) -> Result<ExecutionResult, Error> {
        let available_gas = self
            .gas_metadata
            .get_initial_available_gas(function_id, gas)
            .map_err(crate::error::Error::GasMetadataError)?;

        #[cfg(feature = "with-profiler")]
        self.setup_profiling();

        super::invoke_dynamic(
            &self.registry,
            self.find_function_ptr(function_id),
            self.extract_signature(function_id)?,
            args,
            available_gas,
            Some(syscall_handler),
            self.build_find_dict_overrides(),
        )
    }

    pub fn invoke_contract_dynamic(
        &self,
        function_id: &FunctionId,
        args: &[Felt],
        gas: Option<u64>,
        syscall_handler: impl StarknetSyscallHandler,
    ) -> Result<ContractExecutionResult, Error> {
        let available_gas = self
            .gas_metadata
            .get_initial_available_gas(function_id, gas)
            .map_err(crate::error::Error::GasMetadataError)?;

        #[cfg(feature = "with-profiler")]
        self.setup_profiling();

        ContractExecutionResult::from_execution_result(super::invoke_dynamic(
            &self.registry,
            self.find_function_ptr(function_id),
            self.extract_signature(function_id)?,
            &[Value::Struct {
                fields: vec![Value::Array(
                    args.iter().cloned().map(Value::Felt252).collect(),
                )],
                debug_name: None,
            }],
            available_gas,
            Some(syscall_handler),
            self.build_find_dict_overrides(),
        )?)
    }

    pub fn find_function_ptr(&self, function_id: &FunctionId) -> *mut c_void {
        let function_name = generate_function_name(function_id, false);
        let function_name = format!("_mlir_ciface_{function_name}");

        // Arguments and return values are hardcoded since they'll be handled by the trampoline.
        self.engine.lookup(&function_name) as *mut c_void
    }

    pub fn find_symbol_ptr(&self, name: &str) -> Option<*mut c_void> {
        let ptr = self.engine.lookup(name) as *mut c_void;

        if ptr.is_null() {
            None
        } else {
            Some(ptr)
        }
    }

    fn extract_signature(&self, function_id: &FunctionId) -> Result<&FunctionSignature, Error> {
        Ok(self
            .program_registry()
            .get_function(function_id)
            .map(|func| &func.signature)?)
    }

    fn build_find_dict_overrides(
        &self,
    ) -> impl '_
           + Copy
           + Fn(
        &ConcreteTypeId,
    ) -> (
        Option<extern "C" fn(*mut c_void, *mut c_void)>,
        Option<extern "C" fn(*mut c_void)>,
    ) {
        |type_id| {
            (
                self.dict_overrides
                    .get_dup_fn(type_id)
                    .and_then(|symbol| self.find_symbol_ptr(symbol))
                    .map(|ptr| unsafe { transmute(ptr as *const ()) }),
                self.dict_overrides
                    .get_drop_fn(type_id)
                    .and_then(|symbol| self.find_symbol_ptr(symbol))
                    .map(|ptr| unsafe { transmute(ptr as *const ()) }),
            )
        }
    }

    #[cfg(feature = "with-profiler")]
    fn setup_profiling(&self) {
        unsafe {
            let callback_ptr: *mut extern "C" fn(u64, u64) =
                self.engine.lookup("__profiler_callback").cast();

            *callback_ptr = crate::metadata::profiler::ProfilerImpl::callback;
        }
    }
}
