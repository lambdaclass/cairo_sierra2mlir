use cairo_lang_runner::SierraCasmRunner;
use cairo_lang_sierra::program::Program;
use std::sync::Arc;
use walkdir::WalkDir;

pub fn prepare_programs(path: &str) -> Vec<(Arc<Program>, String)> {
    WalkDir::new(path)
        .into_iter()
        .filter_map(|entry| {
            let e = entry.unwrap();
            let path = e.path();
            match path.extension().map(|x| x.to_str().unwrap()) {
                Some("cairo") => Some((
                    cairo_native::utils::cairo_to_sierra(path).unwrap(),
                    path.display().to_string(),
                )),
                _ => None,
            }
        })
        .collect::<Vec<_>>()
}

#[allow(unused)] // Used in `benches/libfuncs.rs`, but not in the others.
pub fn create_vm_runner(program: &Program) -> SierraCasmRunner {
    SierraCasmRunner::new(
        program.clone(),
        Some(Default::default()),
        Default::default(),
        None,
    )
    .unwrap()
}
