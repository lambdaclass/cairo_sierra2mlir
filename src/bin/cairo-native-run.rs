use anyhow::Context;
use cairo_lang_compiler::{
    compile_prepared_db, db::RootDatabase, project::setup_project, CompilerConfig,
};
use cairo_lang_runner::short_string::as_cairo_short_string;
use cairo_native::{
    context::NativeContext,
    executor::{AotNativeExecutor, JitNativeExecutor},
    metadata::gas::{GasMetadata, MetadataComputationConfig},
    starknet_stub::StubSyscallHandler,
};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use tracing_subscriber::{EnvFilter, FmtSubscriber};
use utils::{find_function, result_to_runresult};

mod utils;

#[derive(Clone, Debug, ValueEnum)]
enum RunMode {
    Aot,
    Jit,
}

/// Command line args parser.
/// Exits with 1 if the compilation or run fails, otherwise 0.
#[derive(Parser, Debug)]
#[clap(version, verbatim_doc_comment)]
struct Args {
    /// The Cairo project path to compile and run its tests.
    path: PathBuf,
    /// Whether path is a single file.
    #[arg(short, long)]
    single_file: bool,
    /// Allows the compilation to succeed with warnings.
    #[arg(long)]
    allow_warnings: bool,
    /// In cases where gas is available, the amount of provided gas.
    #[arg(long)]
    available_gas: Option<u64>,
    /// Run with JIT or AOT (compiled).
    #[arg(long, value_enum, default_value_t = RunMode::Jit)]
    run_mode: RunMode,
    /// Optimization level, Valid: 0, 1, 2, 3. Values higher than 3 are considered as 3.
    #[arg(short = 'O', long, default_value_t = 0)]
    opt_level: u8,

    #[cfg(feature = "with-profiler")]
    #[arg(long)]
    profiler_output: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    // Configure logging and error handling.
    tracing::subscriber::set_global_default(
        FmtSubscriber::builder()
            .with_env_filter(EnvFilter::from_default_env())
            .finish(),
    )?;

    let args = Args::parse();

    let mut db = RootDatabase::builder().detect_corelib().build()?;
    let main_crate_ids = setup_project(&mut db, &args.path)?;

    let sierra_program = compile_prepared_db(
        &db,
        main_crate_ids,
        CompilerConfig {
            replace_ids: true,
            ..Default::default()
        },
    )?
    .program;

    let native_context = NativeContext::new();

    // Compile the sierra program into a MLIR module.
    let native_module = native_context
        .compile(&sierra_program, false, Some(Default::default()))
        .unwrap();

    let native_executor: Box<dyn Fn(_, _, _, &mut StubSyscallHandler) -> _> = match args.run_mode {
        RunMode::Aot => {
            let executor =
                AotNativeExecutor::from_native_module(native_module, args.opt_level.into())?;
            Box::new(move |function_id, args, gas, syscall_handler| {
                executor.invoke_dynamic_with_syscall_handler(
                    function_id,
                    args,
                    gas,
                    syscall_handler,
                )
            })
        }
        RunMode::Jit => {
            let executor =
                JitNativeExecutor::from_native_module(native_module, args.opt_level.into())?;
            Box::new(move |function_id, args, gas, syscall_handler| {
                executor.invoke_dynamic_with_syscall_handler(
                    function_id,
                    args,
                    gas,
                    syscall_handler,
                )
            })
        }
    };

    let gas_metadata =
        GasMetadata::new(&sierra_program, Some(MetadataComputationConfig::default())).unwrap();

    let func = find_function(&sierra_program, "::main")?;

    let initial_gas = gas_metadata
        .get_initial_available_gas(&func.id, args.available_gas)
        .with_context(|| "not enough gas to run")?;

    let mut syscall_handler = StubSyscallHandler::default();

    let result = native_executor(&func.id, &[], Some(initial_gas), &mut syscall_handler)
        .with_context(|| "Failed to run the function.")?;

    let run_result = result_to_runresult(&result)?;

    match run_result {
        cairo_lang_runner::RunResultValue::Success(values) => {
            println!("Run completed successfully, returning {values:?}")
        }
        cairo_lang_runner::RunResultValue::Panic(values) => {
            print!("Run panicked with [");
            for value in &values {
                match as_cairo_short_string(value) {
                    Some(as_string) => print!("{value} ('{as_string}'), "),
                    None => print!("{value}, "),
                }
            }
            println!("].")
        }
    }
    if let Some(gas) = result.remaining_gas {
        println!("Remaining gas: {gas}");
    }

    #[cfg(feature = "with-profiler")]
    if let Some(profiler_output) = args.profiler_output {
        use cairo_lang_sierra::{ids::ConcreteLibfuncId, program::Statement};
        use std::{collections::HashMap, fs::File, io::Write};

        let mut trace = HashMap::<ConcreteLibfuncId, (Vec<u64>, u64)>::new();

        for (statement_idx, tick_delta) in cairo_native::metadata::profiler::ProfilerImpl::take() {
            if let Statement::Invocation(invocation) = &sierra_program.statements[statement_idx.0] {
                let (tick_deltas, extra_count) =
                    trace.entry(invocation.libfunc_id.clone()).or_default();

                if tick_delta != u64::MAX {
                    tick_deltas.push(tick_delta);
                } else {
                    *extra_count += 1;
                }
            }
        }

        let mut trace = trace
            .into_iter()
            .map(|(libfunc_id, (mut tick_deltas, extra_count))| {
                tick_deltas.sort();

                // Drop outliers.
                {
                    let q1 = tick_deltas[tick_deltas.len() / 4];
                    let q3 = tick_deltas[3 * tick_deltas.len() / 4];
                    let iqr = q3 - q1;

                    let q1_thr = q1.saturating_sub(iqr + iqr / 2);
                    let q3_thr = q3 + (iqr + iqr / 2);

                    tick_deltas.retain(|x| *x >= q1_thr && *x <= q3_thr);
                }

                // Compute the quartiles.
                let quartiles = [
                    *tick_deltas.first().unwrap(),
                    tick_deltas[tick_deltas.len() / 4],
                    tick_deltas[tick_deltas.len() / 2],
                    tick_deltas[3 * tick_deltas.len() / 4],
                    *tick_deltas.last().unwrap(),
                ];

                // Compuite the average.
                let average =
                    tick_deltas.iter().copied().sum::<u64>() as f64 / tick_deltas.len() as f64;

                // Compute the standard deviation.
                let std_dev = {
                    let sum = tick_deltas
                        .iter()
                        .copied()
                        .map(|x| x as f64)
                        .map(|x| (x - average))
                        .map(|x| x * x)
                        .sum::<f64>();
                    sum / (tick_deltas.len() as u64 + extra_count) as f64
                };

                (
                    libfunc_id,
                    (
                        tick_deltas.len() as u64 + extra_count,
                        tick_deltas.iter().sum::<u64>()
                            + (extra_count as f64 * average).round() as u64,
                        quartiles,
                        average,
                        std_dev,
                    ),
                )
            })
            .collect::<Vec<_>>();

        // Sort libfuncs by the order in which they are declared.
        trace.sort_by_key(|(libfunc_id, _)| {
            sierra_program
                .libfunc_declarations
                .iter()
                .enumerate()
                .find_map(|(i, x)| (&x.id == libfunc_id).then_some(i))
                .unwrap()
        });

        let mut output = File::create(profiler_output)?;

        for (libfunc_id, (n_samples, sum, quartiles, average, std_dev)) in trace {
            writeln!(output, "{libfunc_id}")?;
            writeln!(output, "    Samples  : {n_samples}")?;
            writeln!(output, "    Sum      : {sum}")?;
            writeln!(output, "    Average  : {average}")?;
            writeln!(output, "    Deviation: {std_dev}")?;
            writeln!(output, "    Quartiles: {quartiles:?}")?;
            writeln!(output)?;
        }
    }

    Ok(())
}
