use crate::common::load_cairo;

use cairo_lang_runner::{Arg, SierraCasmRunner};
use cairo_lang_sierra::program::Program;
use cairo_native::utils::run_native_or_vm_program;
use cairo_native::values::JITValue;
use common::compare_outputs;
use lazy_static::lazy_static;
use proptest::prelude::*;

mod common;

const GAS: usize = usize::MAX;

lazy_static! {
    static ref U8_OVERFLOWING_ADD: ((String, Program), SierraCasmRunner) = load_cairo! {
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(lhs: u8, rhs: u8) -> u8 {
            lhs + rhs
        }

        fn run_test(lhs: felt252, rhs: felt252) -> u8 {
            program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
        }
    };
    static ref U8_OVERFLOWING_SUB: ((String, Program), SierraCasmRunner) = load_cairo! {
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(lhs: u8, rhs: u8) -> u8 {
            lhs - rhs
        }

        fn run_test(lhs: felt252, rhs: felt252) -> u8 {
            program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
        }
    };
    static ref U8_SAFE_DIVMOD: ((String, Program), SierraCasmRunner) = load_cairo! {
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(lhs: u8, rhs: u8) -> (u8, u8) {
            let q = lhs / rhs;
            let r = lhs % rhs;

            (q, r)
        }

        fn run_test(lhs: felt252, rhs: felt252) -> (u8, u8) {
            program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
        }
    };
    static ref U8_EQUAL: ((String, Program), SierraCasmRunner) = load_cairo! {
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(lhs: u8, rhs: u8) -> bool {
            lhs == rhs
        }

        fn run_test(lhs: felt252, rhs: felt252) -> bool {
            program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
        }
    };
    static ref U8_IS_ZERO: ((String, Program), SierraCasmRunner) = load_cairo! {
        use zeroable::IsZeroResult;

        extern fn u8_is_zero(a: u8) -> IsZeroResult<u8> implicits() nopanic;

        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(value: u8) -> bool {
            match u8_is_zero(value) {
                IsZeroResult::Zero(_) => true,
                IsZeroResult::NonZero(_) => false,
            }
        }

        fn run_test(value: felt252) -> bool {
            program(value.try_into().unwrap())
        }
    };
    static ref U8_SQRT: ((String, Program), SierraCasmRunner) = load_cairo! {
        use core::integer::u8_sqrt;
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(value: u8) -> u8 {
            u8_sqrt(value)
        }

        fn run_test(value: felt252) -> u8 {
            program(value.try_into().unwrap())
        }
    };

    // U16

    static ref U16_OVERFLOWING_ADD: ((String, Program), SierraCasmRunner) = load_cairo! {
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(lhs: u16, rhs: u16) -> u16 {
            lhs + rhs
        }

        fn run_test(lhs: felt252, rhs: felt252) -> u16 {
            program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
        }
    };
    static ref U16_OVERFLOWING_SUB: ((String, Program), SierraCasmRunner) = load_cairo! {
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(lhs: u16, rhs: u16) -> u16 {
            lhs - rhs
        }

        fn run_test(lhs: felt252, rhs: felt252) -> u16 {
            program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
        }
    };
    static ref U16_SAFE_DIVMOD: ((String, Program), SierraCasmRunner) = load_cairo! {
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(lhs: u16, rhs: u16) -> (u16, u16) {
            let q = lhs / rhs;
            let r = lhs % rhs;

            (q, r)
        }

        fn run_test(lhs: felt252, rhs: felt252) -> (u16, u16) {
            program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
        }
    };
    static ref U16_EQUAL: ((String, Program), SierraCasmRunner) = load_cairo! {
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(lhs: u16, rhs: u16) -> bool {
            lhs == rhs
        }

        fn run_test(lhs: felt252, rhs: felt252) -> bool {
            program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
        }
    };
    static ref U16_IS_ZERO: ((String, Program), SierraCasmRunner) = load_cairo! {
        use zeroable::IsZeroResult;

        extern fn u16_is_zero(a: u16) -> IsZeroResult<u16> implicits() nopanic;

        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(value: u16) -> bool {
            match u16_is_zero(value) {
                IsZeroResult::Zero(_) => true,
                IsZeroResult::NonZero(_) => false,
            }
        }

        fn run_test(value: felt252) -> bool {
            program(value.try_into().unwrap())
        }
    };
    static ref U16_SQRT: ((String, Program), SierraCasmRunner)= load_cairo! {
        use core::integer::u16_sqrt;
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(value: u16) -> u16 {
            u16_sqrt(value)
        }

        fn run_test(value: felt252) -> u16 {
            program(value.try_into().unwrap())
        }
    };

    // U32

    static ref U32_OVERFLOWING_ADD: ((String, Program), SierraCasmRunner) = load_cairo! {
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(lhs: u32, rhs: u32) -> u32 {
            lhs + rhs
        }

        fn run_test(lhs: felt252, rhs: felt252) -> u32 {
            program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
        }
    };
    static ref U32_OVERFLOWING_SUB: ((String, Program), SierraCasmRunner) = load_cairo! {
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(lhs: u32, rhs: u32) -> u32 {
            lhs - rhs
        }

        fn run_test(lhs: felt252, rhs: felt252) -> u32 {
            program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
        }
    };
    static ref U32_SAFE_DIVMOD: ((String, Program), SierraCasmRunner) = load_cairo! {
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(lhs: u32, rhs: u32) -> (u32, u32) {
            let q = lhs / rhs;
            let r = lhs % rhs;

            (q, r)
        }

        fn run_test(lhs: felt252, rhs: felt252) -> (u32, u32) {
            program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
        }
    };
    static ref U32_EQUAL: ((String, Program), SierraCasmRunner) = load_cairo! {
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(lhs: u32, rhs: u32) -> bool {
            lhs == rhs
        }

        fn run_test(lhs: felt252, rhs: felt252) -> bool {
            program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
        }
    };
    static ref U32_IS_ZERO: ((String, Program), SierraCasmRunner) = load_cairo! {
        use zeroable::IsZeroResult;

        extern fn u32_is_zero(a: u32) -> IsZeroResult<u32> implicits() nopanic;

        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(value: u32) -> bool {
            match u32_is_zero(value) {
                IsZeroResult::Zero(_) => true,
                IsZeroResult::NonZero(_) => false,
            }
        }

        fn run_test(value: felt252) -> bool {
            program(value.try_into().unwrap())
        }
    };
    static ref U32_SQRT: ((String, Program), SierraCasmRunner) = load_cairo! {
        use core::integer::u32_sqrt;
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(value: u32) -> u32 {
            u32_sqrt(value)
        }

        fn run_test(value: felt252) -> u32 {
            program(value.try_into().unwrap())
        }
    };

    // U64

    static ref U64_OVERFLOWING_ADD: ((String, Program), SierraCasmRunner) = load_cairo! {
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(lhs: u64, rhs: u64) -> u64 {
            lhs + rhs
        }

        fn run_test(lhs: felt252, rhs: felt252) -> u64 {
            program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
        }
    };
    static ref U64_OVERFLOWING_SUB: ((String, Program), SierraCasmRunner) = load_cairo! {
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(lhs: u64, rhs: u64) -> u64 {
            lhs - rhs
        }

        fn run_test(lhs: felt252, rhs: felt252) -> u64 {
            program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
        }
    };
    static ref U64_SAFE_DIVMOD: ((String, Program), SierraCasmRunner) = load_cairo! {
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(lhs: u64, rhs: u64) -> (u64, u64) {
            let q = lhs / rhs;
            let r = lhs % rhs;

            (q, r)
        }

        fn run_test(lhs: felt252, rhs: felt252) -> (u64, u64) {
            program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
        }
    };
    static ref U64_EQUAL:((String, Program), SierraCasmRunner) = load_cairo! {
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(lhs: u64, rhs: u64) -> bool {
            lhs == rhs
        }

        fn run_test(lhs: felt252, rhs: felt252) -> bool {
            program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
        }
    };
    static ref U64_IS_ZERO: ((String, Program), SierraCasmRunner) = load_cairo! {
        use zeroable::IsZeroResult;

        extern fn u64_is_zero(a: u64) -> IsZeroResult<u64> implicits() nopanic;

        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(value: u64) -> bool {
            match u64_is_zero(value) {
                IsZeroResult::Zero(_) => true,
                IsZeroResult::NonZero(_) => false,
            }
        }

        fn run_test(value: felt252) -> bool {
            program(value.try_into().unwrap())
        }
    };
    static ref U64_SQRT: ((String, Program), SierraCasmRunner) = load_cairo! {
        use core::integer::u64_sqrt;
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(value: u64) -> u64 {
            u64_sqrt(value)
        }

        fn run_test(value: felt252) -> u64 {
            program(value.try_into().unwrap())
        }
    };

    // U128

    static ref U128_OVERFLOWING_ADD: ((String, Program), SierraCasmRunner) = load_cairo! {
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(lhs: u128, rhs: u128) -> u128 {
            lhs + rhs
        }

        fn run_test(lhs: felt252, rhs: felt252) -> u128 {
            program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
        }
    };
    static ref U128_OVERFLOWING_SUB: ((String, Program), SierraCasmRunner) = load_cairo! {
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(lhs: u128, rhs: u128) -> u128 {
            lhs - rhs
        }

        fn run_test(lhs: felt252, rhs: felt252) -> u128 {
            program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
        }
    };
    static ref U128_SAFE_DIVMOD: ((String, Program), SierraCasmRunner) = load_cairo! {
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(lhs: u128, rhs: u128) -> (u128, u128) {
            let q = lhs / rhs;
            let r = lhs % rhs;

            (q, r)
        }

        fn run_test(lhs: felt252, rhs: felt252) -> (u128, u128) {
            program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
        }
    };
    static ref U128_EQUAL: ((String, Program), SierraCasmRunner) = load_cairo! {
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(lhs: u128, rhs: u128) -> bool {
            lhs == rhs
        }

        fn run_test(lhs: felt252, rhs: felt252) -> bool {
            program(lhs.try_into().unwrap(), rhs.try_into().unwrap())
        }
    };
    static ref U128_IS_ZERO: ((String, Program), SierraCasmRunner) = load_cairo! {
        use zeroable::IsZeroResult;

        extern fn u128_is_zero(a: u128) -> IsZeroResult<u128> implicits() nopanic;

        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(value: u128) -> bool {
            match u128_is_zero(value) {
                IsZeroResult::Zero(_) => true,
                IsZeroResult::NonZero(_) => false,
            }
        }

        fn run_test(value: felt252) -> bool {
            program(value.try_into().unwrap())
        }
    };
    static ref U128_SQRT: ((String, Program), SierraCasmRunner) = load_cairo! {
        use core::integer::u128_sqrt;
        use traits::TryInto;
        use core::option::OptionTrait;

        fn program(value: u128) -> u128 {
            u128_sqrt(value)
        }

        fn run_test(value: felt252) -> u128 {
            program(value.try_into().unwrap())
        }
    };
}

proptest! {
    #[test]
    fn u8_overflowing_add_proptest(a in 0..u8::MAX, b in 0..u8::MAX) {
        let (program, sierra_casn_runner) = &U8_OVERFLOWING_ADD;
        let result_vm = run_native_or_vm_program(
            program,
            "run_test",
            None,
            Some(&[Arg::Value(a.into()), Arg::Value(b.into())]),
            sierra_casn_runner,
            Some(GAS),
        ).
        .unwrap();
        let result_native = run_native_or_vm_program(program, "run_test", Some(&[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None);

        compare_outputs(
            &program.0.1,
            &program.1.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn u8_overflowing_sub_proptest(a in 0..u8::MAX, b in 0..u8::MAX) {
        let (program, sierra_casm_runner) = &U8_OVERFLOWING_SUB;

        let result_vm = run_native_or_vm_program(
            program,
            "run_test",
            None,
            Some(&[Arg::Value(a.into()), Arg::Value(b.into())]),
            sierra_casm_runner,
            Some(GAS),
        ).left().unwrap()
        .unwrap();
        let result_native = run_native_or_vm_program(program, "run_test", Some(&[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None).right().unwrap();

        compare_outputs(
            &program.0.1,
            &program.1.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn u8_safe_divmod_proptest(a in 0..u8::MAX, b in 0..u8::MAX) {
        let (program, sierra_casm_runner) = &U8_SAFE_DIVMOD;
        let result_vm = run_native_or_vm_program(
                program,
            "run_test",
            None,
            Some(&[Arg::Value(a.into()), Arg::Value(b.into())]),
            sierra_casm_runner,
            Some(GAS),
        ).left().unwrap()
        .unwrap();
        let result_native = run_native_or_vm_program(program, "run_test", Some(&[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None).right().unwrap();

        compare_outputs(
            &program.0.1,
            &program.1.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn u8_equal_proptest(a in 0..u8::MAX, b in 0..u8::MAX) {
        let (program, sierra_casm_runner) = &U8_EQUAL;
        let result_vm = run_native_or_vm_program(
                program,
            "run_test",
            None,
            Some(&[Arg::Value(a.into()), Arg::Value(b.into())]),
            sierra_casm_runner,
            Some(GAS),
        ).left().unwrap()
        .unwrap();
        let result_native = run_native_or_vm_program(program, "run_test", Some(&[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None).right().unwrap();

        compare_outputs(
            &program.0.1,
            &program.1.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn u8_is_zero_proptest(a in 0..u8::MAX) {
        let (program, sierra_casm_runner) = &U8_IS_ZERO;
        let result_vm = run_native_or_vm_program(
                program,
            "run_test",
            None,
            Some(&[Arg::Value(a.into()), Arg::Value(b.into())]),
            sierra_casm_runner,
            Some(GAS),
        ).left().unwrap()
        .unwrap();
        let result_native = run_native_or_vm_program(program, "run_test", Some(&[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None).right().unwrap();

        compare_outputs(
            &program.0.1,
            &program.1.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    // u16

    #[test]
    fn u16_overflowing_add_proptest(a in 0..u16::MAX, b in 0..u16::MAX) {
        let (program, sierra_casm_runner) = &U16_OVERFLOWING_ADD;
        let result_vm = run_native_or_vm_program(
                program,
            "run_test",
            None,
            Some(&[Arg::Value(a.into()), Arg::Value(b.into())]),
            sierra_casm_runner,
            Some(GAS),
        ).left().unwrap()
        .unwrap();
        let result_native = run_native_or_vm_program(program, "run_test", Some(&[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None).right().unwrap();

        compare_outputs(
            &program.0.1,
            &program.1.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn u16_overflowing_sub_proptest(a in 0..u16::MAX, b in 0..u16::MAX) {
        let (program, sierra_casm_runner) = &U16_OVERFLOWING_SUB;
        let result_vm = run_native_or_vm_program(
                program,
            "run_test",
            None,
            Some(&[Arg::Value(a.into()), Arg::Value(b.into())]),
            sierra_casm_runner,
            Some(GAS),
        ).left().unwrap()
        .unwrap();
        let result_native = run_native_or_vm_program(program, "run_test", Some(&[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None).right().unwrap();

        compare_outputs(
            &program.0.1,
            &program.1.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn u16_safe_divmod_proptest(a in 0..u16::MAX, b in 0..u16::MAX) {
        let (program, sierra_casm_runner) = &U16_SAFE_DIVMOD;
        let result_vm = run_native_or_vm_program(
                program,
            "run_test",
            None,
            Some(&[Arg::Value(a.into()), Arg::Value(b.into())]),
            sierra_casm_runner,
            Some(GAS),
        ).left().unwrap()
        .unwrap();
        let result_native = run_native_or_vm_program(program, "run_test", Some(&[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None).right().unwrap();

        compare_outputs(
            &program.0.1,
            &program.1.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn u16_equal_proptest(a in 0..u16::MAX, b in 0..u16::MAX) {
        let (program, sierra_casm_runner) = &U16_EQUAL;
        let result_vm = run_native_or_vm_program(
                program,
            "run_test",
            None,
            Some(&[Arg::Value(a.into()), Arg::Value(b.into())]),
            sierra_casm_runner,
            Some(GAS),
        ).left().unwrap()
        .unwrap();
        let result_native = run_native_or_vm_program(program, "run_test", Some(&[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None).right().unwrap();

        compare_outputs(
            &program.0.1,
            &program.1.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn u16_is_zero_proptest(a in 0..u16::MAX) {
        let (program, sierra_casm_runner) = &U16_IS_ZERO;

        let result_vm = run_native_or_vm_program(
                program,
            "run_test",
            None,
            Some(&[Arg::Value(a.into())]),
            sierra_casm_runner,
            Some(GAS),
        ).left().unwrap()
        .unwrap();
        let result_native = run_native_or_vm_program(program, "run_test", Some(&[JITValue::Felt252(a.into())]), None, None, None).right().unwrap();

        compare_outputs(
            &program.0.1,
            &program.1.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    // u32

    #[test]
    fn u32_overflowing_add_proptest(a in 0..u32::MAX, b in 0..u32::MAX) {
        let (program, sierra_casm_runner) = &U32_OVERFLOWING_ADD;
       let result_vm = run_native_or_vm_program(
                program,
            "run_test",
            None,
            Some(&[Arg::Value(a.into()), Arg::Value(b.into())]),
            sierra_casm_runner,
            Some(GAS),
        ).left().unwrap()
        .unwrap();
        let result_native = run_native_or_vm_program(program, "run_test", Some(&[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None).right().unwrap();

        compare_outputs(
            &program.0.1,
            &program.1.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn u32_overflowing_sub_proptest(a in 0..u32::MAX, b in 0..u32::MAX) {
        let (program, sierra_casm_runner) = &U32_OVERFLOWING_SUB;
       let result_vm = run_native_or_vm_program(
                program,
            "run_test",
            None,
            Some(&[Arg::Value(a.into()), Arg::Value(b.into())]),
            sierra_casm_runner,
            Some(GAS),
        ).left().unwrap()
        .unwrap();
        let result_native = run_native_or_vm_program(program, "run_test", Some(&[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None).right().unwrap();

        compare_outputs(
            &program.0.1,
            &program.1.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn u32_safe_divmod_proptest(a in 0..u32::MAX, b in 0..u32::MAX) {
        let (program, sierra_casm_runner) = &U32_SAFE_DIVMOD;
       let result_vm = run_native_or_vm_program(
                program,
            "run_test",
            None,
            Some(&[Arg::Value(a.into()), Arg::Value(b.into())]),
            sierra_casm_runner,
            Some(GAS),
        ).left().unwrap()
        .unwrap();
        let result_native = run_native_or_vm_program(program, "run_test", Some(&[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None).right().unwrap();

        compare_outputs(
            &program.0.1,
            &program.1.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn u32_equal_proptest(a in 0..u32::MAX, b in 0..u32::MAX) {
        let (program, sierra_casm_runner) = &U32_EQUAL;
       let result_vm = run_native_or_vm_program(
                program,
            "run_test",
            None,
            Some(&[Arg::Value(a.into()), Arg::Value(b.into())]),
            sierra_casm_runner,
            Some(GAS),
        ).left().unwrap()
        .unwrap();
        let result_native = run_native_or_vm_program(program, "run_test", Some(&[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None).right().unwrap();

        compare_outputs(
            &program.0.1,
            &program.1.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn u32_is_zero_proptest(a in 0..u32::MAX) {
        let (program, sierra_casm_runner) = &U32_IS_ZERO;
       let result_vm = run_native_or_vm_program(
                program,
            "run_test",
            None,
            Some(&[Arg::Value(a.into()), Arg::Value(b.into())]),
            sierra_casm_runner,
            Some(GAS),
        ).left().unwrap()
        .unwrap();
        let result_native = run_native_or_vm_program(program, "run_test", Some(&[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None).right().unwrap();

        compare_outputs(
            &program.0.1,
            &program.1.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    // u64

    #[test]
    fn u64_overflowing_add_proptest(a in 0..u64::MAX, b in 0..u64::MAX) {
        let (program, sierra_casm_runner) = &U64_OVERFLOWING_ADD;
        let result_vm = run_native_or_vm_program(
                program,
            "run_test",
            None,
            Some(&[Arg::Value(a.into()), Arg::Value(b.into())]),
            sierra_casm_runner,
            Some(GAS),
        ).left().unwrap()
        .unwrap();
        let result_native = run_native_or_vm_program(program, "run_test", Some(&[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None).right().unwrap();

        compare_outputs(
            &program.0.1,
            &program.1.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn u64_overflowing_sub_proptest(a in 0..u64::MAX, b in 0..u64::MAX) {
        let (program, sierra_casm_runner) = &U64_OVERFLOWING_SUB;
       let result_vm = run_native_or_vm_program(
                program,
            "run_test",
            None,
            Some(&[Arg::Value(a.into()), Arg::Value(b.into())]),
            sierra_casm_runner,
            Some(GAS),
        ).left().unwrap()
        .unwrap();
        let result_native = run_native_or_vm_program(program, "run_test", Some(&[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None).right().unwrap();

        compare_outputs(
            &program.0.1,
            &program.1.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn u64_safe_divmod_proptest(a in 0..u64::MAX, b in 0..u64::MAX) {
        let (program, sierra_casm_runner) = &U64_SAFE_DIVMOD;
       let result_vm = run_native_or_vm_program(
                program,
            "run_test",
            None,
            Some(&[Arg::Value(a.into()), Arg::Value(b.into())]),
            sierra_casm_runner,
            Some(GAS),
        ).left().unwrap()
        .unwrap();
        let result_native = run_native_or_vm_program(program, "run_test", Some(&[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None).right().unwrap();

        compare_outputs(
            &program.0.1,
            &program.1.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn u64_equal_proptest(a in 0..u64::MAX, b in 0..u64::MAX) {
        let (program, sierra_casm_runner) = &U64_EQUAL;
      let result_vm = run_native_or_vm_program(
                program,
            "run_test",
            None,
            Some(&[Arg::Value(a.into()), Arg::Value(b.into())]),
            sierra_casm_runner,
            Some(GAS),
        ).left().unwrap()
        .unwrap();
        let result_native = run_native_or_vm_program(program, "run_test", Some(&[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None).right().unwrap();

        compare_outputs(
            &program.0.1,
            &program.1.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn u64_is_zero_proptest(a in 0..u64::MAX) {
        let (program, sierra_casm_runner) = &U64_IS_ZERO;
        let result_vm = run_native_or_vm_program(
                program,
            "run_test",
            None,
            Some(&[Arg::Value(a.into()), Arg::Value(b.into())]),
            sierra_casm_runner,
            Some(GAS),
        ).left().unwrap()
        .unwrap();
        let result_native = run_native_or_vm_program(program, "run_test", Some(&[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None).right().unwrap();

        compare_outputs(
            &program.0.1,
            &program.1.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    // u128

    #[test]
    fn u128_overflowing_add_proptest(a in 0..u128::MAX, b in 0..u128::MAX) {
        let (program, sierra_casm_runner) = &U128_OVERFLOWING_ADD;
       let result_vm = run_native_or_vm_program(
                program,
            "run_test",
            None,
            Some(&[Arg::Value(a.into()), Arg::Value(b.into())]),
            sierra_casm_runner,
            Some(GAS),
        ).left().unwrap()
        .unwrap();
        let result_native = run_native_or_vm_program(program, "run_test", Some(&[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None).right().unwrap();

        compare_outputs(
            &program.0.1,
            &program.1.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn u128_overflowing_sub_proptest(a in 0..u128::MAX, b in 0..u128::MAX) {
        let (program, sierra_casm_runner) = &U128_OVERFLOWING_SUB;
       let result_vm = run_native_or_vm_program(
                program,
            "run_test",
            None,
            Some(&[Arg::Value(a.into()), Arg::Value(b.into())]),
            sierra_casm_runner,
            Some(GAS),
        ).left().unwrap()
        .unwrap();
        let result_native = run_native_or_vm_program(program, "run_test", Some(&[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None).right().unwrap();

        compare_outputs(
            &program.0.1,
            &program.1.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn u128_safe_divmod_proptest(a in 0..u128::MAX, b in 0..u128::MAX) {
        let (program, sierra_casm_runner) = &U128_SAFE_DIVMOD;
        let result_vm = run_native_or_vm_program(
                program,
            "run_test",
            None,
            Some(&[Arg::Value(a.into()), Arg::Value(b.into())]),
            sierra_casm_runner,
            Some(GAS),
        ).left().unwrap()
        .unwrap();
        let result_native = run_native_or_vm_program(program, "run_test", Some(&[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None).right().unwrap();

        compare_outputs(
            &program.0.1,
            &program.1.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn u128_equal_proptest(a in 0..u128::MAX, b in 0..u128::MAX) {
        let (program, sierra_casm_runner) = &U128_EQUAL;
let result_vm = run_native_or_vm_program(
                program,
            "run_test",
            None,
            Some(&[Arg::Value(a.into()), Arg::Value(b.into())]),
            sierra_casm_runner,
            Some(GAS),
        ).left().unwrap()
        .unwrap();
        let result_native = run_native_or_vm_program(program, "run_test", Some(&[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None).right().unwrap();

        compare_outputs(
            &program.0.1,
            &program.1.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }

    #[test]
    fn u128_is_zero_proptest(a in 0..u128::MAX) {
        let (program, sierra_casm_runner) = &U128_IS_ZERO;
        let result_vm = run_native_or_vm_program(
                program,
            "run_test",
            None,
            Some(&[Arg::Value(a.into()), Arg::Value(b.into())]),
            sierra_casm_runner,
            Some(GAS),
        ).left().unwrap()
        .unwrap();
        let result_native = run_native_or_vm_program(program, "run_test", Some(&[JITValue::Felt252(a.into()), JITValue::Felt252(b.into())]), None, None, None).right().unwrap();

        compare_outputs(
            &program.0.1,
            &program.1.find_function("run_test").unwrap().id,
            &result_vm,
            &result_native,
        )?;
    }
}
