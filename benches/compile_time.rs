use cairo_native::{context::NativeContext, module_to_object, OptLevel};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use util::prepare_programs;

mod util;

pub fn bench_compile_time(c: &mut Criterion) {
    let programs = prepare_programs("programs/compile_benches");

    {
        let mut c = c.benchmark_group("Compilation With Independent Context");

        for (program, filename) in &programs {
            c.bench_with_input(BenchmarkId::new(filename, 1), &program, |b, program| {
                b.iter(|| {
                    let native_context = NativeContext::new();
                    native_context
                        .compile(program, false, Some(Default::default()))
                        .unwrap();
                    // pass manager internally verifies the MLIR output is correct.
                })
            });
        }
    }

    {
        let mut c = c.benchmark_group("Compilation With Shared Context");

        let native_context = NativeContext::new();

        for (program, filename) in &programs {
            c.bench_with_input(BenchmarkId::new(filename, 1), &program, |b, program| {
                b.iter(|| {
                    native_context
                        .compile(program, false, Some(Default::default()))
                        .unwrap();
                    // pass manager internally verifies the MLIR output is correct.
                })
            });
        }
    }

    {
        let mut c = c.benchmark_group("Compilation With Independent Context To Object Code");

        for (program, filename) in &programs {
            c.bench_with_input(BenchmarkId::new(filename, 1), &program, |b, program| {
                b.iter(|| {
                    let native_context = NativeContext::new();
                    let module = native_context
                        .compile(black_box(program), false, Some(Default::default()))
                        .unwrap();
                    let object = module_to_object(module.module(), OptLevel::None)
                        .expect("to compile correctly to a object file");
                    black_box(object)
                })
            });
        }
    }

    {
        let mut c = c.benchmark_group("Compilation With Shared Context To Object Code");

        let native_context = NativeContext::new();

        for (program, filename) in &programs {
            c.bench_with_input(BenchmarkId::new(filename, 1), &program, |b, program| {
                b.iter(|| {
                    let module = native_context
                        .compile(black_box(program), false, Some(Default::default()))
                        .unwrap();
                    let object = module_to_object(module.module(), OptLevel::None)
                        .expect("to compile correctly to a object file");
                    black_box(object)
                })
            });
        }
    }

    {
        let mut c = c.benchmark_group("Compilation With Shared Context To Object Code With -O3");

        let native_context = NativeContext::new();

        for (program, filename) in &programs {
            c.bench_with_input(BenchmarkId::new(filename, 1), &program, |b, program| {
                b.iter(|| {
                    let module = native_context
                        .compile(black_box(program), false, Some(Default::default()))
                        .unwrap();
                    let object = module_to_object(module.module(), OptLevel::Aggressive)
                        .expect("to compile correctly to a object file");
                    black_box(object)
                })
            });
        }
    }
}

criterion_group!(benches, bench_compile_time);
criterion_main!(benches);
