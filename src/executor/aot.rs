use crate::{
    error::Error,
    execution_result::{ContractExecutionResult, ExecutionResult},
    metadata::{
        felt252_dict::Felt252DictOverrides, gas::GasMetadata, runtime_bindings::setup_runtime,
    },
    module::NativeModule,
    starknet::{DummySyscallHandler, StarknetSyscallHandler},
    utils::generate_function_name,
    values::Value,
    OptLevel,
};
use cairo_lang_sierra::{
    extensions::core::{CoreLibfunc, CoreType},
    ids::{ConcreteTypeId, FunctionId},
    program::FunctionSignature,
    program_registry::ProgramRegistry,
};
use educe::Educe;
use libc::c_void;
use libloading::Library;
use starknet_types_core::felt::Felt;
use std::{io, mem::transmute};
use tempfile::NamedTempFile;

#[derive(Educe)]
#[educe(Debug)]
pub struct AotNativeExecutor {
    #[educe(Debug(ignore))]
    library: Library,
    #[educe(Debug(ignore))]
    registry: ProgramRegistry<CoreType, CoreLibfunc>,

    gas_metadata: GasMetadata,
    dict_overrides: Felt252DictOverrides,
}

unsafe impl Send for AotNativeExecutor {}
unsafe impl Sync for AotNativeExecutor {}

impl AotNativeExecutor {
    pub fn new(
        library: Library,
        registry: ProgramRegistry<CoreType, CoreLibfunc>,
        gas_metadata: GasMetadata,
        dict_overrides: Felt252DictOverrides,
    ) -> Self {
        let executor = Self {
            library,
            registry,
            gas_metadata,
            dict_overrides,
        };

        setup_runtime(|name| executor.find_symbol_ptr(name));

        #[cfg(feature = "with-trace-dump")]
        crate::metadata::trace_dump::setup_runtime(|name| executor.find_symbol_ptr(name));

        executor
    }

    /// Utility to convert a [`NativeModule`] into an [`AotNativeExecutor`].
    pub fn from_native_module(module: NativeModule, opt_level: OptLevel) -> Result<Self, Error> {
        let NativeModule {
            module,
            registry,
            mut metadata,
        } = module;

        let library_path = NamedTempFile::new()?
            .into_temp_path()
            .keep()
            .map_err(io::Error::from)?;

        let object_data = crate::module_to_object(&module, opt_level)?;
        crate::object_to_shared_lib(&object_data, &library_path)?;

        Ok(Self::new(
            unsafe { Library::new(&library_path)? },
            registry,
            metadata.remove().ok_or(Error::MissingMetadata)?,
            metadata.remove().unwrap_or_default(),
        ))
    }

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

        super::invoke_dynamic(
            &self.registry,
            self.find_function_ptr(function_id)?,
            self.extract_signature(function_id)?,
            args,
            available_gas,
            Option::<DummySyscallHandler>::None,
            self.build_find_dict_overrides(),
        )
    }

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

        super::invoke_dynamic(
            &self.registry,
            self.find_function_ptr(function_id)?,
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

        ContractExecutionResult::from_execution_result(super::invoke_dynamic(
            &self.registry,
            self.find_function_ptr(function_id)?,
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

    pub fn find_function_ptr(&self, function_id: &FunctionId) -> Result<*mut c_void, Error> {
        let function_name = generate_function_name(function_id, false);
        let function_name = format!("_mlir_ciface_{function_name}");

        // Arguments and return values are hardcoded since they'll be handled by the trampoline.
        unsafe {
            Ok(self
                .library
                .get::<extern "C" fn()>(function_name.as_bytes())?
                .into_raw()
                .into_raw())
        }
    }

    pub fn find_symbol_ptr(&self, name: &str) -> Option<*mut c_void> {
        unsafe {
            self.library
                .get::<*mut ()>(name.as_bytes())
                .ok()
                .map(|x| x.into_raw().into_raw())
        }
    }

    fn extract_signature(&self, function_id: &FunctionId) -> Result<&FunctionSignature, Error> {
        Ok(&self.registry.get_function(function_id)?.signature)
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        context::NativeContext,
        starknet_stub::StubSyscallHandler,
        utils::test::{load_cairo, load_starknet},
    };
    use cairo_lang_sierra::program::Program;
    use rstest::*;

    #[fixture]
    fn program() -> Program {
        let (_, program) = load_cairo! {
            use core::starknet::{SyscallResultTrait, get_block_hash_syscall};

            fn run_test() -> felt252 {
                42
            }

            fn get_block_hash() -> felt252 {
                get_block_hash_syscall(1).unwrap_syscall()
            }
        };
        program
    }

    #[fixture]
    fn starknet_program() -> Program {
        let (_, program) = load_starknet! {
            #[starknet::interface]
            trait ISimpleStorage<TContractState> {
                fn get(self: @TContractState) -> u128;
            }

            #[starknet::contract]
            mod contract {
                #[storage]
                struct Storage {}

                #[abi(embed_v0)]
                impl ISimpleStorageImpl of super::ISimpleStorage<ContractState> {
                    fn get(self: @ContractState) -> u128 {
                        42
                    }
                }
            }
        };
        program
    }

    #[rstest]
    #[case(OptLevel::None)]
    #[case(OptLevel::Default)]
    #[case(OptLevel::Aggressive)]
    fn test_invoke_dynamic(program: Program, #[case] optlevel: OptLevel) {
        let native_context = NativeContext::new();
        let module = native_context
            .compile(&program, false, Some(Default::default()))
            .expect("failed to compile context");
        let executor = AotNativeExecutor::from_native_module(module, optlevel).unwrap();

        // The first function in the program is `run_test`.
        let entrypoint_function_id = &program.funcs.first().expect("should have a function").id;

        let result = executor
            .invoke_dynamic(entrypoint_function_id, &[], Some(u64::MAX))
            .unwrap();

        assert_eq!(result.return_value, Value::Felt252(Felt::from(42)));
    }

    #[rstest]
    #[case(OptLevel::None)]
    #[case(OptLevel::Default)]
    #[case(OptLevel::Aggressive)]
    fn test_invoke_dynamic_with_syscall_handler(program: Program, #[case] optlevel: OptLevel) {
        let native_context = NativeContext::new();
        let module = native_context
            .compile(&program, false, Some(Default::default()))
            .expect("failed to compile context");
        let executor = AotNativeExecutor::from_native_module(module, optlevel).unwrap();

        // The second function in the program is `get_block_hash`.
        let entrypoint_function_id = &program.funcs.get(1).expect("should have a function").id;

        let mut syscall_handler = &mut StubSyscallHandler::default();

        let expected_value = syscall_handler.get_block_hash(1, &mut 0).unwrap();

        let result = executor
            .invoke_dynamic_with_syscall_handler(
                entrypoint_function_id,
                &[],
                Some(u64::MAX),
                syscall_handler,
            )
            .unwrap();

        let expected_value = Value::Enum {
            tag: 0,
            value: Value::Struct {
                fields: vec![Value::Felt252(expected_value)],
                debug_name: Some("Tuple<felt252>".into()),
            }
            .into(),
            debug_name: Some("core::panics::PanicResult::<(core::felt252,)>".into()),
        };
        assert_eq!(result.return_value, expected_value);
    }

    #[rstest]
    #[case(OptLevel::None)]
    #[case(OptLevel::Default)]
    #[case(OptLevel::Aggressive)]
    fn test_invoke_contract_dynamic(starknet_program: Program, #[case] optlevel: OptLevel) {
        let native_context = NativeContext::new();
        let module = native_context
            .compile(&starknet_program, false, Some(Default::default()))
            .expect("failed to compile context");
        let executor = AotNativeExecutor::from_native_module(module, optlevel).unwrap();

        // The last function in the program is the `get` wrapper function.
        let entrypoint_function_id = &starknet_program
            .funcs
            .last()
            .expect("should have a function")
            .id;

        let result = executor
            .invoke_contract_dynamic(
                entrypoint_function_id,
                &[],
                Some(u64::MAX),
                &mut StubSyscallHandler::default(),
            )
            .unwrap();

        assert_eq!(result.return_values, vec![Felt::from(42)]);
    }
}
