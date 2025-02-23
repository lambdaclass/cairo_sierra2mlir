//! # Various utilities

pub(crate) use self::{
    block_ext::{BlockExt, GepIndex},
    program_registry_ext::ProgramRegistryExt,
    range_ext::RangeExt,
};
use crate::{metadata::MetadataStorage, OptLevel};
use cairo_lang_compiler::CompilerConfig;
use cairo_lang_runner::token_gas_cost;
use cairo_lang_sierra::{
    extensions::gas::CostTokenType,
    ids::FunctionId,
    program::{GenFunction, Program, StatementIdx},
};
use melior::{
    ir::Module,
    pass::{self, PassManager},
    Context, Error, ExecutionEngine,
};
use num_bigint::{BigInt, BigUint, Sign};
use serde::{Deserialize, Serialize};
use starknet_types_core::felt::Felt;
use std::sync::{Arc, LazyLock};
use std::{
    alloc::Layout,
    borrow::Cow,
    fmt::{self, Display},
    path::Path,
};
use thiserror::Error;

mod block_ext;
pub mod mem_tracing;
mod program_registry_ext;
mod range_ext;
#[cfg(feature = "with-segfault-catcher")]
pub mod safe_runner;
pub mod sierra_gen;

#[cfg(target_os = "macos")]
pub const SHARED_LIBRARY_EXT: &str = "dylib";
#[cfg(target_os = "linux")]
pub const SHARED_LIBRARY_EXT: &str = "so";

/// The `felt252` prime modulo.
pub static PRIME: LazyLock<BigUint> = LazyLock::new(|| {
    "3618502788666131213697322783095070105623107215331596699973092056135872020481"
        .parse()
        .expect("hardcoded prime constant should be valid")
});
pub static HALF_PRIME: LazyLock<BigUint> = LazyLock::new(|| {
    "1809251394333065606848661391547535052811553607665798349986546028067936010240"
        .parse()
        .expect("hardcoded half prime constant should be valid")
});

// Order matters, for the libfunc impl
// https://github.com/starkware-libs/sequencer/blob/1b7252f8a30244d39614d7666aa113b81291808e/crates/blockifier/src/execution/entry_point_execution.rs#L208
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[repr(C)]
pub struct BuiltinCosts {
    pub r#const: u64,
    pub pedersen: u64,
    pub bitwise: u64,
    pub ecop: u64,
    pub poseidon: u64,
    pub add_mod: u64,
    pub mul_mod: u64,
}

impl Default for BuiltinCosts {
    fn default() -> Self {
        Self {
            r#const: token_gas_cost(CostTokenType::Const) as u64,
            pedersen: token_gas_cost(CostTokenType::Pedersen) as u64,
            bitwise: token_gas_cost(CostTokenType::Bitwise) as u64,
            ecop: token_gas_cost(CostTokenType::EcOp) as u64,
            poseidon: token_gas_cost(CostTokenType::Poseidon) as u64,
            add_mod: token_gas_cost(CostTokenType::AddMod) as u64,
            mul_mod: token_gas_cost(CostTokenType::MulMod) as u64,
        }
    }
}

impl crate::arch::AbiArgument for BuiltinCosts {
    fn to_bytes(
        &self,
        buffer: &mut Vec<u8>,
        find_dict_overrides: impl Copy
            + Fn(
                &cairo_lang_sierra::ids::ConcreteTypeId,
            ) -> (
                Option<extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void)>,
                Option<extern "C" fn(*mut std::ffi::c_void)>,
            ),
    ) -> crate::error::Result<()> {
        self.r#const.to_bytes(buffer, find_dict_overrides)?;
        self.pedersen.to_bytes(buffer, find_dict_overrides)?;
        self.bitwise.to_bytes(buffer, find_dict_overrides)?;
        self.ecop.to_bytes(buffer, find_dict_overrides)?;
        self.poseidon.to_bytes(buffer, find_dict_overrides)?;
        self.add_mod.to_bytes(buffer, find_dict_overrides)?;
        self.mul_mod.to_bytes(buffer, find_dict_overrides)?;

        Ok(())
    }
}

#[cfg(feature = "with-mem-tracing")]
#[allow(unused_imports)]
pub(crate) use self::mem_tracing::{
    _wrapped_free as libc_free, _wrapped_malloc as libc_malloc, _wrapped_realloc as libc_realloc,
};
#[cfg(not(feature = "with-mem-tracing"))]
#[allow(unused_imports)]
pub(crate) use libc::{free as libc_free, malloc as libc_malloc, realloc as libc_realloc};

/// Generate a function name.
///
/// If the program includes function identifiers, return those. Otherwise return `f` followed by the
/// identifier number.
pub fn generate_function_name(
    function_id: &FunctionId,
    is_for_contract_executor: bool,
) -> Cow<str> {
    // Generic functions can omit their type in the debug_name, leading to multiple functions
    // having the same name, we solve this by adding the id number even if the function has a debug_name

    if is_for_contract_executor {
        Cow::Owned(format!("f{}", function_id.id))
    } else if let Some(name) = function_id.debug_name.as_deref() {
        Cow::Owned(format!("{}(f{})", name, function_id.id))
    } else {
        Cow::Owned(format!("f{}", function_id.id))
    }
}

/// Decode an UTF-8 error message replacing invalid bytes with their hexadecimal representation, as
/// done by Python's `x.decode('utf-8', errors='backslashreplace')`.
pub fn decode_error_message(data: &[u8]) -> String {
    let mut pos = 0;
    utf8_iter::ErrorReportingUtf8Chars::new(data).fold(String::new(), |mut acc, ch| {
        match ch {
            Ok(ch) => {
                acc.push(ch);
                pos += ch.len_utf8();
            }
            Err(_) => {
                acc.push_str(&format!("\\x{:02x}", data[pos]));
                pos += 1;
            }
        };

        acc
    })
}

/// Return the layout for an integer of arbitrary width.
///
/// This assumes the platform's maximum (effective) alignment is 16 bytes, and that every integer
/// with a size in bytes of a power of two has the same alignment as its size.
pub fn get_integer_layout(width: u32) -> Layout {
    if width == 0 {
        Layout::new::<()>()
    } else if width <= 8 {
        Layout::new::<u8>()
    } else if width <= 16 {
        Layout::new::<u16>()
    } else if width <= 32 {
        Layout::new::<u32>()
    } else if width <= 64 {
        Layout::new::<u64>()
    } else if width <= 128 {
        Layout::new::<u128>()
    } else {
        // According to the docs this should never return an error.
        Layout::from_size_align((width as usize).next_multiple_of(8) >> 3, 16)
            .expect("only fails if size is greater than ISIZE::MAX")
    }
}

/// Compile a cairo program found at the given path to sierra.
pub fn cairo_to_sierra(program: &Path) -> crate::error::Result<Arc<Program>> {
    if program
        .extension()
        .map(|x| {
            x.to_ascii_lowercase()
                .to_string_lossy()
                .eq_ignore_ascii_case("cairo")
        })
        .unwrap_or(false)
    {
        cairo_lang_compiler::compile_cairo_project_at_path(
            program,
            CompilerConfig {
                replace_ids: true,
                ..Default::default()
            },
        )
        .map_err(|err| crate::error::Error::ProgramParser(err.to_string()))
    } else {
        let source = std::fs::read_to_string(program)?;
        cairo_lang_sierra::ProgramParser::new()
            .parse(&source)
            .map_err(|err| crate::error::Error::ProgramParser(err.to_string()))
    }
    .map(Arc::new)
}

/// Returns the given entry point if present.
pub fn find_entry_point<'a>(
    program: &'a Program,
    entry_point: &str,
) -> Option<&'a GenFunction<StatementIdx>> {
    program
        .funcs
        .iter()
        .find(|x| x.id.debug_name.as_deref() == Some(entry_point))
}

/// Returns the given entry point if present.
pub fn find_entry_point_by_idx(
    program: &Program,
    entry_point_idx: usize,
) -> Option<&GenFunction<StatementIdx>> {
    program
        .funcs
        .iter()
        .find(|x| x.id.id == entry_point_idx as u64)
}

/// Given a string representing a function name, searches in the program for the id corresponding
/// to said function, and returns a reference to it.
#[track_caller]
pub fn find_function_id<'a>(program: &'a Program, function_name: &str) -> Option<&'a FunctionId> {
    program
        .funcs
        .iter()
        .find(|x| x.id.debug_name.as_deref() == Some(function_name))
        .map(|func| &func.id)
}

/// Parse a numeric string into felt, wrapping negatives around the prime modulo.
pub fn felt252_str(value: &str) -> Felt {
    let value = value
        .parse::<BigInt>()
        .expect("value must be a digit number");

    let value = match value.sign() {
        Sign::Minus => &*PRIME - value.magnitude(),
        _ => value.magnitude().clone(),
    };

    value.into()
}

/// Parse any type that can be a bigint to a felt that can be used in the cairo-native input.
pub fn felt252_bigint(value: impl Into<BigInt>) -> Felt {
    let value: BigInt = value.into();
    let value = match value.sign() {
        Sign::Minus => Cow::Owned(&*PRIME - value.magnitude()),
        _ => Cow::Borrowed(value.magnitude()),
    };

    value.as_ref().into()
}

/// Parse a short string into a felt that can be used in the cairo-native input.
pub fn felt252_short_str(value: &str) -> Felt {
    let values: Vec<_> = value
        .chars()
        .filter_map(|c| c.is_ascii().then_some(c as u8))
        .collect();

    assert!(values.len() < 32, "A felt can't longer than 32 bytes");
    Felt::from_bytes_be_slice(&values)
}

/// Creates the execution engine, with all symbols registered.
pub fn create_engine(
    module: &Module,
    _metadata: &MetadataStorage,
    opt_level: OptLevel,
) -> ExecutionEngine {
    // Create the JIT engine.
    let engine = ExecutionEngine::new(module, opt_level.into(), &[], false);
    #[cfg(feature = "with-debug-utils")]
    _metadata
        .get::<crate::metadata::debug_utils::DebugUtils>()
        .unwrap()
        .register_impls(&engine);

    #[cfg(feature = "with-mem-tracing")]
    self::mem_tracing::register_bindings(&engine);

    engine
}

pub fn run_pass_manager(context: &Context, module: &mut Module) -> Result<(), Error> {
    let pass_manager = PassManager::new(context);
    pass_manager.enable_verifier(true);
    pass_manager.add_pass(pass::transform::create_canonicalizer());
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow()); // needed because to_llvm doesn't include it.
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    pass_manager.run(module)
}

/// Return a type that calls a closure when formatted using [Debug](std::fmt::Debug).
pub fn debug_with<F>(fmt: F) -> impl fmt::Debug
where
    F: Fn(&mut fmt::Formatter) -> fmt::Result,
{
    struct FmtWrapper<F>(F)
    where
        F: Fn(&mut fmt::Formatter) -> fmt::Result;

    impl<F> fmt::Debug for FmtWrapper<F>
    where
        F: Fn(&mut fmt::Formatter) -> fmt::Result,
    {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            self.0(f)
        }
    }

    FmtWrapper(fmt)
}

/// Edit: Copied from the std lib.
///
/// Returns the amount of padding we must insert after `layout`
/// to ensure that the following address will satisfy `align`
/// (measured in bytes).
///
/// e.g., if `layout.size()` is 9, then `layout.padding_needed_for(4)`
/// returns 3, because that is the minimum number of bytes of
/// padding required to get a 4-aligned address (assuming that the
/// corresponding memory block starts at a 4-aligned address).
///
/// The return value of this function has no meaning if `align` is
/// not a power-of-two.
///
/// Note that the utility of the returned value requires `align`
/// to be less than or equal to the alignment of the starting
/// address for the whole allocated block of memory. One way to
/// satisfy this constraint is to ensure `align <= layout.align()`.
#[inline]
pub const fn padding_needed_for(layout: &Layout, align: usize) -> usize {
    let len = layout.size();

    // Rounded up value is:
    //   len_rounded_up = (len + align - 1) & !(align - 1);
    // and then we return the padding difference: `len_rounded_up - len`.
    //
    // We use modular arithmetic throughout:
    //
    // 1. align is guaranteed to be > 0, so align - 1 is always
    //    valid.
    //
    // 2. `len + align - 1` can overflow by at most `align - 1`,
    //    so the &-mask with `!(align - 1)` will ensure that in the
    //    case of overflow, `len_rounded_up` will itself be 0.
    //    Thus the returned padding, when added to `len`, yields 0,
    //    which trivially satisfies the alignment `align`.
    //
    // (Of course, attempts to allocate blocks of memory whose
    // size and padding overflow in the above manner should cause
    // the allocator to yield an error anyway.)

    let len_rounded_up = len.wrapping_add(align).wrapping_sub(1) & !align.wrapping_sub(1);
    len_rounded_up.wrapping_sub(len)
}

#[derive(Clone, PartialEq, Eq, Debug, Error)]
pub struct LayoutError;

impl Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("layout error")
    }
}

/// Copied from std.
///
/// Creates a layout describing the record for `n` instances of
/// `self`, with a suitable amount of padding between each to
/// ensure that each instance is given its requested size and
/// alignment. On success, returns `(k, offs)` where `k` is the
/// layout of the array and `offs` is the distance between the start
/// of each element in the array.
///
/// On arithmetic overflow, returns `LayoutError`.
//#[unstable(feature = "alloc_layout_extra", issue = "55724")]
#[inline]
pub fn layout_repeat(layout: &Layout, n: usize) -> Result<(Layout, usize), LayoutError> {
    // This cannot overflow. Quoting from the invariant of Layout:
    // > `size`, when rounded up to the nearest multiple of `align`,
    // > must not overflow isize (i.e., the rounded value must be
    // > less than or equal to `isize::MAX`)
    let padded_size = layout.size() + padding_needed_for(layout, layout.align());
    let alloc_size = padded_size.checked_mul(n).ok_or(LayoutError)?;

    // The safe constructor is called here to enforce the isize size limit.
    let layout = Layout::from_size_align(alloc_size, layout.align()).map_err(|_| LayoutError)?;
    Ok((layout, padded_size))
}

#[cfg(test)]
pub mod test {
    use crate::{
        context::NativeContext, execution_result::ExecutionResult, executor::JitNativeExecutor,
        starknet_stub::StubSyscallHandler, utils::*, values::Value,
    };
    use cairo_lang_compiler::{
        compile_prepared_db, db::RootDatabase, diagnostics::DiagnosticsReporter,
        project::setup_project, CompilerConfig,
    };
    use cairo_lang_filesystem::db::init_dev_corelib;
    use cairo_lang_sierra::{
        ids::FunctionId,
        program::Program,
        program::{FunctionSignature, GenFunction, StatementIdx},
    };
    use cairo_lang_starknet::{compile::compile_contract_in_prepared_db, starknet_plugin_suite};
    use cairo_lang_starknet_classes::contract_class::ContractClass;
    use pretty_assertions_sorted::assert_eq;
    use std::io::Write;
    use std::{env::var, fmt::Formatter, fs, path::Path};

    macro_rules! load_cairo {
        ( $( $program:tt )+ ) => {
            $crate::utils::test::load_cairo_str(stringify!($($program)+))
        };
    }
    macro_rules! load_starknet {
        ( $( $program:tt )+ ) => {
            $crate::utils::test::load_starknet_str(stringify!($($program)+))
        };
    }
    macro_rules! load_starknet_contract {
        ( $( $program:tt )+ ) => {
            $crate::utils::test::load_starknet_contract_str(stringify!($($program)+))
        };
    }
    pub(crate) use load_cairo;
    pub(crate) use load_starknet;
    pub(crate) use load_starknet_contract;

    // Helper macros for faster testing.
    macro_rules! jit_struct {
        ($($y:expr),* $(,)? ) => {
            crate::values::Value::Struct {
                fields: vec![$($y), *],
                debug_name: None
            }
        };
    }
    macro_rules! jit_enum {
        ( $tag:expr, $value:expr ) => {
            crate::values::Value::Enum {
                tag: $tag,
                value: Box::new($value),
                debug_name: None,
            }
        };
    }
    macro_rules! jit_dict {
        ( $($key:expr $(=>)+ $value:expr),* $(,)? ) => {
            crate::values::Value::Felt252Dict {
                value: {
                    let mut map = std::collections::HashMap::new();
                    $(map.insert($key.into(), $value.into());)*
                    map
                },
                debug_name: None,
            }
        };
    }
    macro_rules! jit_panic {
        ( $($value:expr)? ) => {
            crate::utils::test::jit_enum!(1, crate::utils::test::jit_struct!(
                crate::utils::test::jit_struct!(),
                [$($value), *].into()
            ))
        };
    }
    pub(crate) use jit_dict;
    pub(crate) use jit_enum;
    pub(crate) use jit_panic;
    pub(crate) use jit_struct;

    pub(crate) fn load_cairo_str(program_str: &str) -> (String, Program) {
        compile_program(program_str, RootDatabase::default())
    }

    pub(crate) fn load_starknet_str(program_str: &str) -> (String, Program) {
        compile_program(
            program_str,
            RootDatabase::builder()
                .with_plugin_suite(starknet_plugin_suite())
                .build()
                .unwrap(),
        )
    }

    pub(crate) fn load_starknet_contract_str(program_str: &str) -> (String, ContractClass) {
        compile_contract(
            program_str,
            RootDatabase::builder()
                .with_plugin_suite(starknet_plugin_suite())
                .build()
                .unwrap(),
        )
    }

    pub(crate) fn compile_contract(
        program_str: &str,
        mut db: RootDatabase,
    ) -> (String, ContractClass) {
        let mut program_file = tempfile::Builder::new()
            .prefix("test_")
            .suffix(".cairo")
            .tempfile()
            .unwrap();
        fs::write(&mut program_file, program_str).unwrap();

        init_dev_corelib(
            &mut db,
            Path::new(&var("CARGO_MANIFEST_DIR").unwrap()).join("corelib/src"),
        );
        let main_crate_ids = setup_project(&mut db, program_file.path()).unwrap();
        let contract = compile_contract_in_prepared_db(
            &db,
            None,
            main_crate_ids,
            CompilerConfig {
                diagnostics_reporter: DiagnosticsReporter::stderr(),
                replace_ids: true,
                ..Default::default()
            },
        )
        .unwrap();

        let module_name = program_file.path().with_extension("");
        let module_name = module_name.file_name().unwrap().to_str().unwrap();
        (module_name.to_string(), contract)
    }

    pub(crate) fn compile_program(program_str: &str, mut db: RootDatabase) -> (String, Program) {
        let mut program_file = tempfile::Builder::new()
            .prefix("test_")
            .suffix(".cairo")
            .tempfile()
            .unwrap();
        fs::write(&mut program_file, program_str).unwrap();

        init_dev_corelib(
            &mut db,
            Path::new(&var("CARGO_MANIFEST_DIR").unwrap()).join("corelib/src"),
        );
        let main_crate_ids = setup_project(&mut db, program_file.path()).unwrap();
        let sierra_program_with_dbg = compile_prepared_db(
            &db,
            main_crate_ids,
            CompilerConfig {
                diagnostics_reporter: DiagnosticsReporter::stderr(),
                replace_ids: true,
                ..Default::default()
            },
        )
        .unwrap();

        let module_name = program_file.path().with_extension("");
        let module_name = module_name.file_name().unwrap().to_str().unwrap();
        (module_name.to_string(), sierra_program_with_dbg.program)
    }

    pub fn run_program(
        program: &(String, Program),
        entry_point: &str,
        args: &[Value],
    ) -> ExecutionResult {
        let entry_point = format!("{0}::{0}::{1}", program.0, entry_point);
        let program = &program.1;

        let entry_point_id = &program
            .funcs
            .iter()
            .find(|x| x.id.debug_name.as_deref() == Some(&entry_point))
            .expect("Test program entry point not found.")
            .id;

        let context = NativeContext::new();

        let module = context
            .compile(program, false, Some(Default::default()))
            .expect("Could not compile test program to MLIR.");

        let executor = JitNativeExecutor::from_native_module(module, OptLevel::Less).unwrap();
        executor
            .invoke_dynamic_with_syscall_handler(
                entry_point_id,
                args,
                Some(u64::MAX),
                &mut StubSyscallHandler::default(),
            )
            .unwrap()
    }

    pub fn run_sierra_program(program: &Program, args: &[Value]) -> ExecutionResult {
        let entry_point_id = &program
            .funcs
            .first()
            .expect("program entry point not found.")
            .id;

        let context = NativeContext::new();

        let module = context
            .compile(program, false, Some(Default::default()))
            .expect("Could not compile test program to MLIR.");

        let executor = JitNativeExecutor::from_native_module(module, OptLevel::Less).unwrap();
        executor
            .invoke_dynamic_with_syscall_handler(
                entry_point_id,
                args,
                Some(u64::MAX),
                &mut StubSyscallHandler::default(),
            )
            .unwrap()
    }

    #[track_caller]
    pub fn run_program_assert_output(
        program: &(String, Program),
        entry_point: &str,
        args: &[Value],
        output: Value,
    ) {
        let result = run_program(program, entry_point, args);
        assert_eq!(result.return_value, output);
    }

    // ==============================
    // == TESTS: get_integer_layout
    // ==============================
    /// Ensures that the host's `u8` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u8() {
        assert_eq!(get_integer_layout(8).align(), 1);
    }

    /// Ensures that the host's `u16` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u16() {
        assert_eq!(get_integer_layout(16).align(), 2);
    }

    /// Ensures that the host's `u32` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u32() {
        assert_eq!(get_integer_layout(32).align(), 4);
    }

    /// Ensures that the host's `u64` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u64() {
        assert_eq!(get_integer_layout(64).align(), 8);
    }

    /// Ensures that the host's `u128` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u128() {
        assert_eq!(get_integer_layout(128).align(), 16);
    }

    /// Ensures that the host's `u256` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u256() {
        assert_eq!(get_integer_layout(256).align(), 16);
    }

    /// Ensures that the host's `u512` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_u512() {
        assert_eq!(get_integer_layout(512).align(), 16);
    }

    /// Ensures that the host's `Felt` is compatible with its compiled counterpart.
    #[test]
    fn test_alignment_compatibility_felt() {
        assert_eq!(get_integer_layout(252).align(), 16);
    }

    // ==============================
    // == TESTS: find_entry_point
    // ==============================
    #[test]
    fn test_find_entry_point_with_empty_program() {
        let program = Program {
            type_declarations: vec![],
            libfunc_declarations: vec![],
            statements: vec![],
            funcs: vec![],
        };
        let entry_point = find_entry_point(&program, "entry_point");
        assert!(entry_point.is_none());
    }

    #[test]
    fn test_entry_point_not_found() {
        let program = Program {
            type_declarations: vec![],
            libfunc_declarations: vec![],
            statements: vec![],
            funcs: vec![GenFunction {
                id: FunctionId {
                    id: 0,
                    debug_name: Some("not_entry_point".into()),
                },
                signature: FunctionSignature {
                    ret_types: vec![],
                    param_types: vec![],
                },
                params: vec![],
                entry_point: StatementIdx(0),
            }],
        };
        let entry_point = find_entry_point(&program, "entry_point");
        assert!(entry_point.is_none());
    }

    #[test]
    fn test_entry_point_found() {
        let program = Program {
            type_declarations: vec![],
            libfunc_declarations: vec![],
            statements: vec![],
            funcs: vec![GenFunction {
                id: FunctionId {
                    id: 0,
                    debug_name: Some("entry_point".into()),
                },
                signature: FunctionSignature {
                    ret_types: vec![],
                    param_types: vec![],
                },
                params: vec![],
                entry_point: StatementIdx(0),
            }],
        };
        let entry_point = find_entry_point(&program, "entry_point");
        assert!(entry_point.is_some());
        assert_eq!(entry_point.unwrap().id.id, 0);
    }

    // ====================================
    // == TESTS: find_entry_point_by_idx
    // ====================================
    #[test]
    fn test_find_entry_point_by_idx_with_empty_program() {
        let program = Program {
            type_declarations: vec![],
            libfunc_declarations: vec![],
            statements: vec![],
            funcs: vec![],
        };
        let entry_point = find_entry_point_by_idx(&program, 0);
        assert!(entry_point.is_none());
    }

    #[test]
    fn test_entry_point_not_found_by_id() {
        let program = Program {
            type_declarations: vec![],
            libfunc_declarations: vec![],
            statements: vec![],
            funcs: vec![GenFunction {
                id: FunctionId {
                    id: 0,
                    debug_name: Some("some_name".into()),
                },
                signature: FunctionSignature {
                    ret_types: vec![],
                    param_types: vec![],
                },
                params: vec![],
                entry_point: StatementIdx(0),
            }],
        };
        let entry_point = find_entry_point_by_idx(&program, 1);
        assert!(entry_point.is_none());
    }

    #[test]
    fn test_entry_point_found_by_id() {
        let program = Program {
            type_declarations: vec![],
            libfunc_declarations: vec![],
            statements: vec![],
            funcs: vec![GenFunction {
                id: FunctionId {
                    id: 15,
                    debug_name: Some("some_name".into()),
                },
                signature: FunctionSignature {
                    ret_types: vec![],
                    param_types: vec![],
                },
                params: vec![],
                entry_point: StatementIdx(0),
            }],
        };
        let entry_point = find_entry_point_by_idx(&program, 15);
        assert!(entry_point.is_some());
        assert_eq!(entry_point.unwrap().id.id, 15);
    }

    #[test]
    fn decode_error_message() {
        // Checkout [issue 795](https://github.com/lambdaclass/cairo_native/issues/795) for context.
        assert_eq!(
            super::decode_error_message(&[
                97, 114, 103, 101, 110, 116, 47, 109, 117, 108, 116, 105, 99, 97, 108, 108, 45,
                102, 97, 105, 108, 101, 100, 3, 232, 78, 97, 116, 105, 118, 101, 32, 101, 120, 101,
                99, 117, 116, 105, 111, 110, 32, 101, 114, 114, 111, 114, 58, 32, 69, 114, 114,
                111, 114, 32, 97, 116, 32, 112, 99, 61, 48, 58, 49, 48, 52, 58, 10, 71, 111, 116,
                32, 97, 110, 32, 101, 120, 99, 101, 112, 116, 105, 111, 110, 32, 119, 104, 105,
                108, 101, 32, 101, 120, 101, 99, 117, 116, 105, 110, 103, 32, 97, 32, 104, 105,
                110, 116, 58, 32, 69, 114, 114, 111, 114, 32, 97, 116, 32, 112, 99, 61, 48, 58, 49,
                56, 52, 58, 10, 71, 111, 116, 32, 97, 110, 32, 101, 120, 99, 101, 112, 116, 105,
                111, 110, 32, 119, 104, 105, 108, 101, 32, 101, 120, 101, 99, 117, 116, 105, 110,
                103, 32, 97, 32, 104, 105, 110, 116, 58, 32, 69, 120, 99, 101, 101, 100, 101, 100,
                32, 116, 104, 101, 32, 109, 97, 120, 105, 109, 117, 109, 32, 110, 117, 109, 98,
                101, 114, 32, 111, 102, 32, 101, 118, 101, 110, 116, 115, 44, 32, 110, 117, 109,
                98, 101, 114, 32, 101, 118, 101, 110, 116, 115, 58, 32, 49, 48, 48, 49, 44, 32,
                109, 97, 120, 32, 110, 117, 109, 98, 101, 114, 32, 101, 118, 101, 110, 116, 115,
                58, 32, 49, 48, 48, 48, 46, 10, 67, 97, 105, 114, 111, 32, 116, 114, 97, 99, 101,
                98, 97, 99, 107, 32, 40, 109, 111, 115, 116, 32, 114, 101, 99, 101, 110, 116, 32,
                99, 97, 108, 108, 32, 108, 97, 115, 116, 41, 58, 10, 85, 110, 107, 110, 111, 119,
                110, 32, 108, 111, 99, 97, 116, 105, 111, 110, 32, 40, 112, 99, 61, 48, 58, 49, 52,
                51, 52, 41, 10, 85, 110, 107, 110, 111, 119, 110, 32, 108, 111, 99, 97, 116, 105,
                111, 110, 32, 40, 112, 99, 61, 48, 58, 49, 51, 57, 53, 41, 10, 85, 110, 107, 110,
                111, 119, 110, 32, 108, 111, 99, 97, 116, 105, 111, 110, 32, 40, 112, 99, 61, 48,
                58, 57, 53, 51, 41, 10, 85, 110, 107, 110, 111, 119, 110, 32, 108, 111, 99, 97,
                116, 105, 111, 110, 32, 40, 112, 99, 61, 48, 58, 51, 51, 57, 41, 10, 10, 67, 97,
                105, 114, 111, 32, 116, 114, 97, 99, 101, 98, 97, 99, 107, 32, 40, 109, 111, 115,
                116, 32, 114, 101, 99, 101, 110, 116, 32, 99, 97, 108, 108, 32, 108, 97, 115, 116,
                41, 58, 10, 85, 110, 107, 110, 111, 119, 110, 32, 108, 111, 99, 97, 116, 105, 111,
                110, 32, 40, 112, 99, 61, 48, 58, 49, 54, 55, 56, 41, 10, 85, 110, 107, 110, 111,
                119, 110, 32, 108, 111, 99, 97, 116, 105, 111, 110, 32, 40, 112, 99, 61, 48, 58,
                49, 54, 54, 52, 41, 10
            ]),
            "argent/multicall-failed\x03\\xe8Native execution error: Error at pc=0:104:\nGot an exception while executing a hint: Error at pc=0:184:\nGot an exception while executing a hint: Exceeded the maximum number of events, number events: 1001, max number events: 1000.\nCairo traceback (most recent call last):\nUnknown location (pc=0:1434)\nUnknown location (pc=0:1395)\nUnknown location (pc=0:953)\nUnknown location (pc=0:339)\n\nCairo traceback (most recent call last):\nUnknown location (pc=0:1678)\nUnknown location (pc=0:1664)\n",
        );
    }

    // ==============================
    // == TESTS: felt252_str
    // ==============================
    #[test]
    #[should_panic(expected = "value must be a digit number")]
    fn test_felt252_str_invalid_input() {
        let value = "not_a_number";
        felt252_str(value);
    }

    #[test]
    fn test_felt252_str_positive_number() {
        let value = "123";
        let result = felt252_str(value);
        assert_eq!(result, 123.into());
    }

    #[test]
    fn test_felt252_str_negative_number() {
        let value = "-123";
        let result = felt252_str(value);
        assert_eq!(
            result,
            Felt::from_dec_str(
                "3618502788666131213697322783095070105623107215331596699973092056135872020358"
            )
            .unwrap()
        );
    }

    #[test]
    fn test_felt252_str_zero() {
        let value = "0";
        let result = felt252_str(value);
        assert_eq!(result, Felt::ZERO);
    }

    // ==============================
    // == TESTS: felt252_short_str
    // ==============================
    #[test]
    fn test_felt252_short_str_short_numeric_string() {
        let value = "12345";
        let result = felt252_short_str(value);
        assert_eq!(result, 211295614005u64.into());
    }

    #[test]
    fn test_felt252_short_str_short_string_with_non_numeric_characters() {
        let value = "hello";
        let result = felt252_short_str(value);
        assert_eq!(result, 448378203247u64.into());
    }

    #[test]
    #[should_panic]
    fn test_felt252_short_str_long_numeric_string() {
        felt252_short_str("1234567890123456789012345678901234567890");
    }

    #[test]
    fn test_felt252_short_str_empty_string() {
        let value = "";
        let result = felt252_short_str(value);
        assert_eq!(result, Felt::ZERO);
    }

    #[test]
    fn test_felt252_short_str_string_with_non_ascii_characters() {
        let value = "h€llø";
        let result = felt252_short_str(value);
        assert_eq!(result, 6843500.into());
    }

    // ==============================
    // == TESTS: debug_with
    // ==============================
    #[test]
    fn test_debug_with_empty_closure() {
        let closure = |_f: &mut Formatter| -> fmt::Result { Ok(()) };
        let debug_wrapper = debug_with(closure);
        assert_eq!(format!("{:?}", debug_wrapper), "");
    }

    #[test]
    #[should_panic]
    fn test_debug_with_error_closure() {
        let closure = |_f: &mut Formatter| -> Result<(), fmt::Error> { Err(fmt::Error) };
        let debug_wrapper = debug_with(closure);
        let _ = format!("{:?}", debug_wrapper);
    }

    #[test]
    fn test_debug_with_simple_closure() {
        let closure = |f: &mut fmt::Formatter| write!(f, "Hello, world!");
        let debug_wrapper = debug_with(closure);
        assert_eq!(format!("{:?}", debug_wrapper), "Hello, world!");
    }

    #[test]
    fn test_debug_with_complex_closure() {
        let closure = |f: &mut fmt::Formatter| write!(f, "Name: William, Age: {}", 28);
        let debug_wrapper = debug_with(closure);
        assert_eq!(format!("{:?}", debug_wrapper), "Name: William, Age: 28");
    }

    #[test]
    fn test_generate_function_name_debug_name() {
        let function_id = FunctionId {
            id: 123,
            debug_name: Some("function_name".into()),
        };

        assert_eq!(
            generate_function_name(&function_id, false),
            "function_name(f123)"
        );
    }

    #[test]
    fn test_generate_function_name_debug_name_for_contract_executor() {
        let function_id = FunctionId {
            id: 123,
            debug_name: Some("function_name".into()),
        };

        assert_eq!(generate_function_name(&function_id, true), "f123");
    }

    #[test]
    fn test_generate_function_name_without_debug_name() {
        let function_id = FunctionId {
            id: 123,
            debug_name: None,
        };

        assert_eq!(generate_function_name(&function_id, false), "f123");
    }

    #[test]
    fn test_cairo_to_sierra_path() {
        // Define the path to the cairo program.
        let program_path = Path::new("programs/examples/hello.cairo");
        // Compile the cairo program to sierra.
        let sierra_program = cairo_to_sierra(program_path).unwrap();

        // Define the entry point function for comparison.
        let entry_point = "hello::hello::greet";
        // Find the function ID of the entry point function in the sierra program.
        let entry_point_id = find_function_id(&sierra_program, entry_point).unwrap();

        // Assert that the debug name of the entry point function matches the expected value.
        assert_eq!(
            entry_point_id.debug_name,
            Some("hello::hello::greet".into())
        );
    }

    #[test]
    fn test_cairo_to_sierra_source() {
        // Define the content of the cairo program as a string.
        let content = "type u8 = u8;";

        // Create a named temporary file and write the content to it.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        // Get the path of the temporary file.
        let file_path = file.path().to_path_buf();

        // Compile the cairo program to sierra using the path of the temporary file.
        let sierra_program = cairo_to_sierra(&file_path).unwrap();

        // Assert that the sierra program has no library function declarations, statements, or functions.
        assert!(sierra_program.libfunc_declarations.is_empty());
        assert!(sierra_program.statements.is_empty());
        assert!(sierra_program.funcs.is_empty());

        // Assert that the debug name of the first type declaration matches the expected value.
        assert_eq!(sierra_program.type_declarations.len(), 1);
        assert_eq!(
            sierra_program.type_declarations[0].id.debug_name,
            Some("u8".into())
        );
    }
}
