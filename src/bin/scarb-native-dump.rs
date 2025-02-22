use anyhow::Context;
use cairo_lang_sierra::program::VersionedProgram;
use cairo_native::context::NativeContext;
use melior::ir::operation::OperationPrintingFlags;
use scarb_metadata::{MetadataCommand, ScarbCommand};
use std::{env, fs};

mod utils;

/// Compiles all packages from a Scarb project on the current directory.
fn main() -> anyhow::Result<()> {
    let metadata = MetadataCommand::new().inherit_stderr().exec()?;

    // Build only the filtered packages.
    ScarbCommand::new().arg("build").run()?;

    // Get `target` directory.
    let profile = env::var("SCARB_PROFILE").unwrap_or("dev".into());
    let default_target_dir = metadata.runtime_manifest.join("target");
    let target_dir = metadata
        .target_dir
        .clone()
        .unwrap_or(default_target_dir)
        .join(profile);

    let native_context = NativeContext::new();
    for package in metadata.packages.iter() {
        for target in &package.targets {
            let lib_file_path = target_dir.join(format!("{}.sierra.json", target.name.clone()));
            println!("Compiling {:?}", lib_file_path);

            if lib_file_path.exists() {
                let compiled = serde_json::from_str::<VersionedProgram>(
                    &fs::read_to_string(lib_file_path.clone())
                        .with_context(|| format!("failed to read file: {lib_file_path}"))?,
                )
                .with_context(|| format!("failed to deserialize compiled file: {lib_file_path}"))?;

                // Compile the sierra program into a MLIR module.
                let native_module = native_context
                    .compile(
                        &compiled.into_v1().unwrap().program,
                        false,
                        Some(Default::default()),
                    )
                    .unwrap();

                // Write the output.
                let output_str = native_module.module().as_operation().to_string_with_flags(
                    OperationPrintingFlags::new().enable_debug_info(true, false),
                )?;

                fs::write(
                    target_dir.join(format!("{}.mlir", target.name.clone())),
                    &output_str,
                )?;
            }

            let contract_files = fs::read_dir(&target_dir)
                .with_context(|| format!("failed to read directory: {}", target_dir))?
                .filter_map(Result::ok)
                .filter(|entry| entry.file_type().map(|ft| ft.is_file()).unwrap_or(false))
                .filter_map(|entry| {
                    let path = entry.path();
                    path.extension()
                        .and_then(|ext| ext.to_str())
                        .filter(|&ext| ext == "json")
                        .and_then(|_| path.file_name())
                        .and_then(|name| name.to_str())
                        .filter(|name| {
                            name.starts_with(&target.name) && name.ends_with(".contract_class.json")
                        })
                        .map(|_| path)
                });

            for contract_file_path in contract_files {
                let sierra_contract_class: cairo_lang_starknet_classes::contract_class::ContractClass = serde_json::from_str(
                    &fs::read_to_string(&contract_file_path)
                        .with_context(|| format!("failed to read file: {:?}", contract_file_path))?,
                )
                .with_context(|| format!("failed to deserialize compiled file: {:?}", contract_file_path))?;

                let sierra_program = sierra_contract_class.extract_sierra_program()?;

                // Compile the sierra program into a MLIR module.
                let native_module = native_context.compile(&sierra_program, false).unwrap();

                // Write the output.
                let output_str = native_module.module().as_operation().to_string_with_flags(
                    OperationPrintingFlags::new().enable_debug_info(true, false),
                )?;

                let output_file_name = contract_file_path
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .replace(".json", ".mlir");
                fs::write(target_dir.join(output_file_name), &output_str)?;
            }
        }
    }

    Ok(())
}
