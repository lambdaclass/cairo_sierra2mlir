#![cfg(feature = "build-cli")]
#![allow(dead_code)]

pub mod test;

use anyhow::bail;
use cairo_felt::Felt252;
use cairo_lang_runner::{casm_run::format_next_item, RunResultValue};
use cairo_lang_sierra::program::{Function, Program};
use cairo_native::{execution_result::ExecutionResult, values::JitValue};
use clap::ValueEnum;
use itertools::Itertools;
use starknet_types_core::felt::Felt;
use std::vec::IntoIter;

pub(super) struct RunArgs {
    pub run_mode: RunMode,
    pub opt_level: u8,
}

#[derive(Clone, Debug, ValueEnum)]
pub enum RunMode {
    Aot,
    Jit,
}

/// Find the function ending with `name_suffix` in the program.
pub fn find_function<'a>(
    sierra_program: &'a Program,
    name_suffix: &str,
) -> anyhow::Result<&'a Function> {
    if let Some(x) = sierra_program.funcs.iter().find(|f| {
        if let Some(name) = &f.id.debug_name {
            name.ends_with(name_suffix)
        } else {
            false
        }
    }) {
        Ok(x)
    } else {
        bail!("test function not found")
    }
}

/// Formats the given felts as a panic string.
pub fn format_for_panic(mut felts: IntoIter<Felt252>) -> String {
    let mut items = Vec::new();
    while let Some(item) = format_next_item(&mut felts) {
        items.push(item.quote_if_string());
    }
    let panic_values_string = if let [item] = &items[..] {
        item.clone()
    } else {
        format!("({})", items.join(", "))
    };
    format!("Panicked with {panic_values_string}.")
}

/// Convert the execution result to a run result.
pub fn result_to_runresult(result: &ExecutionResult) -> anyhow::Result<RunResultValue> {
    let is_success;
    let mut felts: Vec<Felt> = Vec::new();

    match &result.return_value {
        JitValue::Enum { tag, value, .. } => {
            is_success = *tag == 0;

            if !is_success {
                match &**value {
                    JitValue::Struct { fields, .. } => {
                        for field in fields {
                            let felt = jitvalue_to_felt(field);
                            felts.extend(felt);
                        }
                    }
                    _ => bail!(
                        "unsuported return value in cairo-native (inside enum): {:#?}",
                        value
                    ),
                }
            }
        }
        value => {
            is_success = true;
            let felt = jitvalue_to_felt(value);
            felts.extend(felt);
        }
    }

    let return_values = felts
        .into_iter()
        .map(|x| x.to_bigint().into())
        .collect_vec();

    Ok(match is_success {
        true => RunResultValue::Success(return_values),
        false => RunResultValue::Panic(return_values),
    })
}

/// Convert a JIT value to a felt.
fn jitvalue_to_felt(value: &JitValue) -> Vec<Felt> {
    let mut felts = Vec::new();
    match value {
        JitValue::Felt252(felt) => vec![felt.to_bigint().into()],
        JitValue::Bytes31(_) => todo!(),
        JitValue::Array(values) => {
            for value in values {
                let felt = jitvalue_to_felt(value);
                felts.extend(felt);
            }
            felts
        }
        JitValue::Struct { fields, .. } => {
            for field in fields {
                let felt = jitvalue_to_felt(field);
                felts.extend(felt);
            }
            felts
        }
        JitValue::Enum { .. } => todo!(),
        JitValue::Felt252Dict { value, .. } => {
            for (key, value) in value {
                felts.push(*key);
                let felt = jitvalue_to_felt(value);
                felts.extend(felt);
            }

            felts
        }
        JitValue::Uint8(x) => vec![(*x).into()],
        JitValue::Uint16(x) => vec![(*x).into()],
        JitValue::Uint32(x) => vec![(*x).into()],
        JitValue::Uint64(x) => vec![(*x).into()],
        JitValue::Uint128(x) => vec![(*x).into()],
        JitValue::Sint8(x) => vec![(*x).into()],
        JitValue::Sint16(x) => vec![(*x).into()],
        JitValue::Sint32(x) => vec![(*x).into()],
        JitValue::Sint64(x) => vec![(*x).into()],
        JitValue::Sint128(x) => vec![(*x).into()],
        JitValue::EcPoint(_, _) => todo!(),
        JitValue::EcState(_, _, _, _) => todo!(),
        JitValue::Secp256K1Point { .. } => todo!(),
        JitValue::Secp256R1Point { .. } => todo!(),
        JitValue::Null => vec![0.into()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cairo_felt::Felt252;
    use cairo_lang_sierra::ProgramParser;

    #[test]
    fn test_find_function() {
        // Parse a simple program containing a function named "Func2"
        let program = ProgramParser::new().parse("Func2@6() -> ();").unwrap();

        // Assert that the function "Func2" is found and returned correctly
        assert_eq!(
            find_function(&program, "Func2").unwrap(),
            program.funcs.first().unwrap()
        );

        // Assert that an error is returned when trying to find a non-existing function "Func3"
        assert!(find_function(&program, "Func3").is_err());

        // Assert that an error is returned when trying to find a function in an empty program
        assert!(find_function(&ProgramParser::new().parse("").unwrap(), "Func2").is_err());
    }

    #[test]
    fn test_result_to_runresult_enum_nonpanic() {
        // Tests the conversion of a non-panic enum result to a `RunResultValue::Success`.
        assert_eq!(
            result_to_runresult(&ExecutionResult {
                remaining_gas: None,
                return_value: JitValue::Enum {
                    tag: 34,
                    value: JitValue::Array(vec![
                        JitValue::Felt252(42.into()),
                        JitValue::Uint8(100),
                        JitValue::Uint128(1000),
                    ])
                    .into(),
                    debug_name: Some("debug_name".into()),
                },
                builtin_stats: Default::default(),
            })
            .unwrap(),
            RunResultValue::Success(vec![
                Felt252::from(34),
                Felt252::from(42),
                Felt252::from(100),
                Felt252::from(1000)
            ])
        );
    }

    #[test]
    fn test_result_to_runresult_success() {
        // Tests the conversion of a success enum result to a `RunResultValue::Success`.
        assert_eq!(
            result_to_runresult(&ExecutionResult {
                remaining_gas: None,
                return_value: JitValue::Enum {
                    tag: 0,
                    value: JitValue::Uint64(24).into(),
                    debug_name: Some("core::panics::PanicResult::Test".into()),
                },
                builtin_stats: Default::default(),
            })
            .unwrap(),
            RunResultValue::Success(vec![Felt252::from(24)])
        );
    }

    #[test]
    #[should_panic(expected = "unsuported return value in cairo-native")]
    fn test_result_to_runresult_panic() {
        // Tests the conversion with unsuported return value.
        let _ = result_to_runresult(&ExecutionResult {
            remaining_gas: None,
            return_value: JitValue::Enum {
                tag: 10,
                value: JitValue::Uint64(24).into(),
                debug_name: Some("core::panics::PanicResult::Test".into()),
            },
            builtin_stats: Default::default(),
        })
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "missing debug name")]
    fn test_result_to_runresult_missing_debug_name() {
        // Tests the conversion with no debug name.
        let _ = result_to_runresult(&ExecutionResult {
            remaining_gas: None,
            return_value: JitValue::Enum {
                tag: 10,
                value: JitValue::Uint64(24).into(),
                debug_name: None,
            },
            builtin_stats: Default::default(),
        })
        .unwrap();
    }

    #[test]
    fn test_result_to_runresult_return() {
        // Tests the conversion of a panic enum result with non-zero tag to a `RunResultValue::Panic`.
        assert_eq!(
            result_to_runresult(&ExecutionResult {
                remaining_gas: None,
                return_value: JitValue::Enum {
                    tag: 10,
                    value: JitValue::Struct {
                        fields: vec![
                            JitValue::Felt252(42.into()),
                            JitValue::Uint8(100),
                            JitValue::Uint128(1000),
                        ],
                        debug_name: Some("debug_name".into()),
                    }
                    .into(),
                    debug_name: Some("core::panics::PanicResult::Test".into()),
                },
                builtin_stats: Default::default(),
            })
            .unwrap(),
            RunResultValue::Panic(vec![
                Felt252::from(42),
                Felt252::from(100),
                Felt252::from(1000)
            ])
        );
    }

    #[test]
    fn test_result_to_runresult_non_enum() {
        // Tests the conversion of a non-enum result to a `RunResultValue::Success`.
        assert_eq!(
            result_to_runresult(&ExecutionResult {
                remaining_gas: None,
                return_value: JitValue::Uint8(10),
                builtin_stats: Default::default(),
            })
            .unwrap(),
            RunResultValue::Success(vec![Felt252::from(10)])
        );
    }

    #[test]
    fn test_jitvalue_to_felt_felt252() {
        let felt_value: Felt = 42.into();

        assert_eq!(
            jitvalue_to_felt(&JitValue::Felt252(felt_value)),
            vec![felt_value]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_array() {
        assert_eq!(
            jitvalue_to_felt(&JitValue::Array(vec![
                JitValue::Felt252(42.into()),
                JitValue::Uint8(100),
                JitValue::Uint128(1000),
            ])),
            vec![Felt::from(42), Felt::from(100), Felt::from(1000)]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_struct() {
        assert_eq!(
            jitvalue_to_felt(&JitValue::Struct {
                fields: vec![
                    JitValue::Felt252(42.into()),
                    JitValue::Uint8(100),
                    JitValue::Uint128(1000)
                ],
                debug_name: Some("debug_name".into())
            }),
            vec![Felt::from(42), Felt::from(100), Felt::from(1000)]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_enum() {
        // With debug name
        assert_eq!(
            jitvalue_to_felt(&JitValue::Enum {
                tag: 34,
                value: JitValue::Array(vec![
                    JitValue::Felt252(42.into()),
                    JitValue::Uint8(100),
                    JitValue::Uint128(1000),
                ])
                .into(),
                debug_name: Some("debug_name".into())
            }),
            vec![
                Felt::from(34),
                Felt::from(42),
                Felt::from(100),
                Felt::from(1000)
            ]
        );

        // With core::bool debug name and tag 1
        assert_eq!(
            jitvalue_to_felt(&JitValue::Enum {
                tag: 1,
                value: JitValue::Uint128(1000).into(),
                debug_name: Some("core::bool".into())
            }),
            vec![Felt::ONE]
        );

        // With core::bool debug name and tag not 1
        assert_eq!(
            jitvalue_to_felt(&JitValue::Enum {
                tag: 10,
                value: JitValue::Uint128(1000).into(),
                debug_name: Some("core::bool".into())
            }),
            vec![Felt::ZERO]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_u8() {
        assert_eq!(jitvalue_to_felt(&JitValue::Uint8(10)), vec![Felt::from(10)]);
    }

    #[test]
    fn test_jitvalue_to_felt_u16() {
        assert_eq!(
            jitvalue_to_felt(&JitValue::Uint16(100)),
            vec![Felt::from(100)]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_u32() {
        assert_eq!(
            jitvalue_to_felt(&JitValue::Uint32(1000)),
            vec![Felt::from(1000)]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_u64() {
        assert_eq!(
            jitvalue_to_felt(&JitValue::Uint64(10000)),
            vec![Felt::from(10000)]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_u128() {
        assert_eq!(
            jitvalue_to_felt(&JitValue::Uint128(100000)),
            vec![Felt::from(100000)]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_sint8() {
        assert_eq!(
            jitvalue_to_felt(&JitValue::Sint8(-10)),
            vec![Felt::from(-10)]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_sint16() {
        assert_eq!(
            jitvalue_to_felt(&JitValue::Sint16(-100)),
            vec![Felt::from(-100)]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_sint32() {
        assert_eq!(
            jitvalue_to_felt(&JitValue::Sint32(-1000)),
            vec![Felt::from(-1000)]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_sint64() {
        assert_eq!(
            jitvalue_to_felt(&JitValue::Sint64(-10000)),
            vec![Felt::from(-10000)]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_sint128() {
        assert_eq!(
            jitvalue_to_felt(&JitValue::Sint128(-100000)),
            vec![Felt::from(-100000)]
        );
    }

    #[test]
    fn test_jitvalue_to_felt_null() {
        assert_eq!(jitvalue_to_felt(&JitValue::Null), vec![Felt::ZERO]);
    }
}
