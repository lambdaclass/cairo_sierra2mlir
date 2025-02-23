//! # `Felt`-related libfuncs

use super::LibfuncHelper;
use crate::{
    error::Result,
    metadata::MetadataStorage,
    utils::{BlockExt, ProgramRegistryExt, PRIME},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        felt252::{
            Felt252BinaryOperationConcrete, Felt252BinaryOperator, Felt252Concrete,
            Felt252ConstConcreteLibfunc,
        },
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        cf,
    },
    ir::{r#type::IntegerType, Block, BlockLike, Location, Value, ValueLike},
    Context,
};
use num_bigint::{BigInt, Sign};

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &Felt252Concrete,
) -> Result<()> {
    match selector {
        Felt252Concrete::BinaryOperation(info) => {
            build_binary_operation(context, registry, entry, location, helper, metadata, info)
        }
        Felt252Concrete::Const(info) => {
            build_const(context, registry, entry, location, helper, metadata, info)
        }
        Felt252Concrete::IsZero(info) => {
            build_is_zero(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the following libfuncs:
///   - `felt252_add` and `felt252_add_const`.
///   - `felt252_sub` and `felt252_sub_const`.
///   - `felt252_mul` and `felt252_mul_const`.
///   - `felt252_div` and `felt252_div_const`.
pub fn build_binary_operation<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &Felt252BinaryOperationConcrete,
) -> Result<()> {
    let felt252_ty = registry.build_type(
        context,
        helper,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;
    let i256 = IntegerType::new(context, 256).into();
    let i512 = IntegerType::new(context, 512).into();

    let (op, lhs, rhs) = match info {
        Felt252BinaryOperationConcrete::WithVar(operation) => {
            (operation.operator, entry.arg(0)?, entry.arg(1)?)
        }
        Felt252BinaryOperationConcrete::WithConst(operation) => {
            let value = match operation.c.sign() {
                Sign::Minus => (&operation.c + BigInt::from_biguint(Sign::Minus, PRIME.clone()))
                    .magnitude()
                    .clone(),
                _ => operation.c.magnitude().clone(),
            };

            // TODO: Ensure that the constant is on the correct side of the operation.
            let rhs = entry.const_int_from_type(context, location, value, felt252_ty)?;

            (operation.operator, entry.arg(0)?, rhs)
        }
    };

    let result = match op {
        Felt252BinaryOperator::Add => {
            let lhs = entry.extui(lhs, i256, location)?;
            let rhs = entry.extui(rhs, i256, location)?;
            let result = entry.addi(lhs, rhs, location)?;

            let prime = entry.const_int_from_type(context, location, PRIME.clone(), i256)?;
            let result_mod = entry.append_op_result(arith::subi(result, prime, location))?;
            let is_out_of_range =
                entry.cmpi(context, CmpiPredicate::Uge, result, prime, location)?;

            let result = entry.append_op_result(arith::select(
                is_out_of_range,
                result_mod,
                result,
                location,
            ))?;
            entry.trunci(result, felt252_ty, location)?
        }
        Felt252BinaryOperator::Sub => {
            let lhs = entry.extui(lhs, i256, location)?;
            let rhs = entry.extui(rhs, i256, location)?;
            let result = entry.append_op_result(arith::subi(lhs, rhs, location))?;

            let prime = entry.const_int_from_type(context, location, PRIME.clone(), i256)?;
            let result_mod = entry.addi(result, prime, location)?;
            let is_out_of_range = entry.cmpi(context, CmpiPredicate::Ult, lhs, rhs, location)?;

            let result = entry.append_op_result(arith::select(
                is_out_of_range,
                result_mod,
                result,
                location,
            ))?;
            entry.trunci(result, felt252_ty, location)?
        }
        Felt252BinaryOperator::Mul => {
            let lhs = entry.extui(lhs, i512, location)?;
            let rhs = entry.extui(rhs, i512, location)?;
            let result = entry.muli(lhs, rhs, location)?;

            let prime = entry.const_int_from_type(context, location, PRIME.clone(), i512)?;
            let result_mod = entry.append_op_result(arith::remui(result, prime, location))?;
            let is_out_of_range =
                entry.cmpi(context, CmpiPredicate::Uge, result, prime, location)?;

            let result = entry.append_op_result(arith::select(
                is_out_of_range,
                result_mod,
                result,
                location,
            ))?;
            entry.trunci(result, felt252_ty, location)?
        }
        Felt252BinaryOperator::Div => {
            // The extended euclidean algorithm calculates the greatest common divisor of two integers,
            // as well as the bezout coefficients x and y such that for inputs a and b, ax+by=gcd(a,b)
            // We use this in felt division to find the modular inverse of a given number
            // If a is the number we're trying to find the inverse of, we can do
            // ax+y*PRIME=gcd(a,PRIME)=1 => ax = 1 (mod PRIME)
            // Hence for input a, we return x
            // The input MUST be non-zero
            // See https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
            let start_block = helper.append_block(Block::new(&[(i512, location)]));
            let loop_block = helper.append_block(Block::new(&[
                (i512, location),
                (i512, location),
                (i512, location),
                (i512, location),
            ]));
            let negative_check_block = helper.append_block(Block::new(&[]));
            // Block containing final result
            let inverse_result_block = helper.append_block(Block::new(&[(i512, location)]));
            // Egcd works by calculating a series of remainders, each the remainder of dividing the previous two
            // For the initial setup, r0 = PRIME, r1 = a
            // This order is chosen because if we reverse them, then the first iteration will just swap them
            let prev_remainder =
                start_block.const_int_from_type(context, location, PRIME.clone(), i512)?;
            let remainder = start_block.arg(0)?;
            // Similarly we'll calculate another series which starts 0,1,... and from which we will retrieve the modular inverse of a
            let prev_inverse = start_block.const_int_from_type(context, location, 0, i512)?;
            let inverse = start_block.const_int_from_type(context, location, 1, i512)?;
            start_block.append_operation(cf::br(
                loop_block,
                &[prev_remainder, remainder, prev_inverse, inverse],
                location,
            ));

            //---Loop body---
            // Arguments are rem_(i-1), rem, inv_(i-1), inv
            let prev_remainder = loop_block.arg(0)?;
            let remainder = loop_block.arg(1)?;
            let prev_inverse = loop_block.arg(2)?;
            let inverse = loop_block.arg(3)?;

            // First calculate q = rem_(i-1)/rem_i, rounded down
            let quotient =
                loop_block.append_op_result(arith::divui(prev_remainder, remainder, location))?;
            // Then r_(i+1) = r_(i-1) - q * r_i, and inv_(i+1) = inv_(i-1) - q * inv_i
            let rem_times_quo = loop_block.muli(remainder, quotient, location)?;
            let inv_times_quo = loop_block.muli(inverse, quotient, location)?;
            let next_remainder = loop_block.append_op_result(arith::subi(
                prev_remainder,
                rem_times_quo,
                location,
            ))?;
            let next_inverse =
                loop_block.append_op_result(arith::subi(prev_inverse, inv_times_quo, location))?;

            // If r_(i+1) is 0, then inv_i is the inverse
            let zero = loop_block.const_int_from_type(context, location, 0, i512)?;
            let next_remainder_eq_zero =
                loop_block.cmpi(context, CmpiPredicate::Eq, next_remainder, zero, location)?;
            loop_block.append_operation(cf::cond_br(
                context,
                next_remainder_eq_zero,
                negative_check_block,
                loop_block,
                &[],
                &[remainder, next_remainder, inverse, next_inverse],
                location,
            ));

            // egcd sometimes returns a negative number for the inverse,
            // in such cases we must simply wrap it around back into [0, PRIME)
            // this suffices because |inv_i| <= divfloor(PRIME,2)
            let zero = negative_check_block.const_int_from_type(context, location, 0, i512)?;

            let is_negative = negative_check_block
                .append_operation(arith::cmpi(
                    context,
                    CmpiPredicate::Slt,
                    inverse,
                    zero,
                    location,
                ))
                .result(0)?
                .into();
            // if the inverse is < 0, add PRIME
            let prime =
                negative_check_block.const_int_from_type(context, location, PRIME.clone(), i512)?;
            let wrapped_inverse = negative_check_block.addi(inverse, prime, location)?;
            let inverse = negative_check_block.append_op_result(arith::select(
                is_negative,
                wrapped_inverse,
                inverse,
                location,
            ))?;
            negative_check_block.append_operation(cf::br(
                inverse_result_block,
                &[inverse],
                location,
            ));

            // Div Logic Start
            // Fetch operands
            let lhs = entry.extui(lhs, i512, location)?;
            let rhs = entry.extui(rhs, i512, location)?;
            // Calculate inverse of rhs, callling the inverse implementation's starting block
            entry.append_operation(cf::br(start_block, &[rhs], location));
            // Fetch the inverse result from the result block
            let inverse = inverse_result_block.arg(0)?;
            // Peform lhs * (1/ rhs)
            let result = inverse_result_block.muli(lhs, inverse, location)?;
            // Apply modulo and convert result to felt252
            let result_mod =
                inverse_result_block.append_op_result(arith::remui(result, prime, location))?;
            let is_out_of_range =
                inverse_result_block.cmpi(context, CmpiPredicate::Uge, result, prime, location)?;

            let result = inverse_result_block.append_op_result(arith::select(
                is_out_of_range,
                result_mod,
                result,
                location,
            ))?;
            let result = inverse_result_block.trunci(result, felt252_ty, location)?;

            inverse_result_block.append_operation(helper.br(0, &[result], location));
            return Ok(());
        }
    };

    entry.append_operation(helper.br(0, &[result], location));

    Ok(())
}

/// Generate MLIR operations for the `felt252_const` libfunc.
pub fn build_const<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &Felt252ConstConcreteLibfunc,
) -> Result<()> {
    let value = match info.c.sign() {
        Sign::Minus => (&info.c + BigInt::from_biguint(Sign::Plus, PRIME.clone()))
            .magnitude()
            .clone(),
        _ => info.c.magnitude().clone(),
    };

    let felt252_ty = registry.build_type(
        context,
        helper,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;

    let value = entry.const_int_from_type(context, location, value, felt252_ty)?;
    entry.append_operation(helper.br(0, &[value], location));
    Ok(())
}

/// Generate MLIR operations for the `felt252_is_zero` libfunc.
pub fn build_is_zero<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let arg0: Value = entry.arg(0)?;

    let k0 = entry.const_int_from_type(context, location, 0, arg0.r#type())?;
    let condition = entry.cmpi(context, CmpiPredicate::Eq, arg0, k0, location)?;

    entry.append_operation(helper.cond_br(context, condition, [0, 1], [&[], &[arg0]], location));
    Ok(())
}

#[cfg(test)]
pub mod test {
    use crate::{
        utils::{
            sierra_gen::SierraGenerator,
            test::{load_cairo, run_program, run_sierra_program},
        },
        values::Value,
    };
    use cairo_lang_sierra::{
        extensions::{
            felt252::{Felt252BinaryOperationWithVarLibfunc, Felt252Traits},
            is_zero::IsZeroLibfunc,
        },
        program::Program,
    };
    use lazy_static::lazy_static;
    use starknet_types_core::felt::Felt;

    lazy_static! {
        static ref FELT252_ADD: Program = {
            let generator = SierraGenerator::<Felt252BinaryOperationWithVarLibfunc>::default();

            generator.build_with_generic_id("felt252_add".into(), &[])
        };

        static ref FELT252_SUB: Program = {
            let generator = SierraGenerator::<Felt252BinaryOperationWithVarLibfunc>::default();

            generator.build_with_generic_id("felt252_sub".into(), &[])
        };

        static ref FELT252_MUL: Program = {
            let generator = SierraGenerator::<Felt252BinaryOperationWithVarLibfunc>::default();

            generator.build_with_generic_id("felt252_mul".into(), &[])
        };

        static ref FELT252_DIV: Program = {
            let generator = SierraGenerator::<Felt252BinaryOperationWithVarLibfunc>::default();

            generator.build_with_generic_id("felt252_div".into(), &[])
        };

        // TODO: Add test program for `felt252_add_const`.
        // TODO: Add test program for `felt252_sub_const`.
        // TODO: Add test program for `felt252_mul_const`.
        // TODO: Add test program for `felt252_div_const`.

        static ref FELT252_CONST: (String, Program) = load_cairo! {
            extern fn felt252_const<const value: felt252>() -> felt252 nopanic;

            fn run_test() -> (felt252, felt252, felt252, felt252) {
                (
                    felt252_const::<0>(),
                    felt252_const::<1>(),
                    felt252_const::<-2>(),
                    felt252_const::<-1>()
                )
            }
        };

        static ref FELT252_IS_ZERO: Program = {
            let generator = SierraGenerator::<IsZeroLibfunc<Felt252Traits>>::default();

            generator.build(&[])
        };
    }

    fn f(val: &str) -> Felt {
        Felt::from_dec_str(val).unwrap()
    }

    #[test]
    fn felt252_add() {
        fn r(lhs: Felt, rhs: Felt) -> Felt {
            match run_sierra_program(&FELT252_ADD, &[Value::Felt252(lhs), Value::Felt252(rhs)])
                .return_value
            {
                Value::Felt252(x) => x,
                _ => panic!("invalid return type"),
            }
        }

        assert_eq!(r(f("0"), f("0")), f("0"));
        assert_eq!(r(f("1"), f("2")), f("3"));

        assert_eq!(r(f("0"), f("1")), f("1"));
        assert_eq!(r(f("0"), f("-2")), f("-2"));
        assert_eq!(r(f("0"), f("-1")), f("-1"));

        assert_eq!(r(f("1"), f("0")), f("1"));
        assert_eq!(r(f("1"), f("1")), f("2"));
        assert_eq!(r(f("1"), f("-2")), f("-1"));
        assert_eq!(r(f("1"), f("-1")), f("0"));

        assert_eq!(r(f("-2"), f("0")), f("-2"));
        assert_eq!(r(f("-2"), f("1")), f("-1"));
        assert_eq!(r(f("-2"), f("-2")), f("-4"));
        assert_eq!(r(f("-2"), f("-1")), f("-3"));

        assert_eq!(r(f("-1"), f("0")), f("-1"));
        assert_eq!(r(f("-1"), f("1")), f("0"));
        assert_eq!(r(f("-1"), f("-2")), f("-3"));
        assert_eq!(r(f("-1"), f("-1")), f("-2"));
    }

    #[test]
    fn felt252_sub() {
        fn r(lhs: Felt, rhs: Felt) -> Felt {
            match run_sierra_program(&FELT252_SUB, &[Value::Felt252(lhs), Value::Felt252(rhs)])
                .return_value
            {
                Value::Felt252(x) => x,
                _ => panic!("invalid return type"),
            }
        }

        assert_eq!(r(f("0"), f("0")), f("0"));
        assert_eq!(r(f("0"), f("1")), f("-1"));
        assert_eq!(r(f("0"), f("-2")), f("2"));
        assert_eq!(r(f("0"), f("-1")), f("1"));

        assert_eq!(r(f("1"), f("0")), f("1"));
        assert_eq!(r(f("1"), f("1")), f("0"));
        assert_eq!(r(f("1"), f("-2")), f("3"));
        assert_eq!(r(f("1"), f("-1")), f("2"));

        assert_eq!(r(f("-2"), f("0")), f("-2"));
        assert_eq!(r(f("-2"), f("1")), f("-3"));
        assert_eq!(r(f("-2"), f("-2")), f("0"));
        assert_eq!(r(f("-2"), f("-1")), f("-1"));

        assert_eq!(r(f("-1"), f("0")), f("-1"));
        assert_eq!(r(f("-1"), f("1")), f("-2"));
        assert_eq!(r(f("-1"), f("-2")), f("1"));
        assert_eq!(r(f("-1"), f("-1")), f("0"));
    }

    #[test]
    fn felt252_mul() {
        fn r(lhs: Felt, rhs: Felt) -> Felt {
            match run_sierra_program(&FELT252_MUL, &[Value::Felt252(lhs), Value::Felt252(rhs)])
                .return_value
            {
                Value::Felt252(x) => x,
                _ => panic!("invalid return type"),
            }
        }

        assert_eq!(r(f("0"), f("0")), f("0"));
        assert_eq!(r(f("0"), f("1")), f("0"));
        assert_eq!(r(f("0"), f("-2")), f("0"));
        assert_eq!(r(f("0"), f("-1")), f("0"));

        assert_eq!(r(f("1"), f("0")), f("0"));
        assert_eq!(r(f("1"), f("1")), f("1"));
        assert_eq!(r(f("1"), f("-2")), f("-2"));
        assert_eq!(r(f("1"), f("-1")), f("-1"));

        assert_eq!(r(f("-2"), f("0")), f("0"));
        assert_eq!(r(f("-2"), f("1")), f("-2"));
        assert_eq!(r(f("-2"), f("-2")), f("4"));
        assert_eq!(r(f("-2"), f("-1")), f("2"));

        assert_eq!(r(f("-1"), f("0")), f("0"));
        assert_eq!(r(f("-1"), f("1")), f("-1"));
        assert_eq!(r(f("-1"), f("-2")), f("2"));
        assert_eq!(r(f("-1"), f("-1")), f("1"));
    }

    #[test]
    fn felt252_div() {
        // Helper function to run the test and extract the return value.
        fn r(lhs: Felt, rhs: Felt) -> Felt {
            let Value::Felt252(x) =
                run_sierra_program(&FELT252_DIV, &[Value::Felt252(lhs), Value::Felt252(rhs)])
                    .return_value
            else {
                panic!("invalid result");
            };

            x
        }

        // Division by zero is expected to panic.
        assert_eq!(r(f("0"), f("0")), f("0"));
        assert_eq!(r(f("1"), f("0")), f("0"));
        assert_eq!(r(f("-2"), f("0")), f("0"));

        // Test cases for valid division results.
        assert_eq!(r(f("0"), f("1")), f("0"));
        assert_eq!(r(f("0"), f("-2")), f("0"));
        assert_eq!(r(f("0"), f("-1")), f("0"));
        assert_eq!(r(f("1"), f("1")), f("1"));
        assert_eq!(
            r(f("1"), f("-2")),
            f("1809251394333065606848661391547535052811553607665798349986546028067936010240")
        );
        assert_eq!(r(f("1"), f("-1")), f("-1"));
        assert_eq!(r(f("-2"), f("1")), f("-2"));
        assert_eq!(r(f("-2"), f("-2")), f("1"));
        assert_eq!(r(f("-2"), f("-1")), f("2"));
        assert_eq!(r(f("-1"), f("1")), f("-1"));
        assert_eq!(
            r(f("-1"), f("-2")),
            f("1809251394333065606848661391547535052811553607665798349986546028067936010241")
        );
        assert_eq!(r(f("-1"), f("-1")), f("1"));
        assert_eq!(r(f("6"), f("2")), f("3"));
        assert_eq!(r(f("1000"), f("2")), f("500"));
    }

    #[test]
    fn felt252_const() {
        assert_eq!(
            run_program(&FELT252_CONST, "run_test", &[]).return_value,
            Value::Struct {
                fields: [f("0"), f("1"), f("-2"), f("-1")]
                    .map(Value::Felt252)
                    .to_vec(),
                debug_name: None
            }
        );
    }

    #[test]
    fn felt252_is_zero() {
        fn r(x: Felt) -> bool {
            match run_sierra_program(&FELT252_IS_ZERO, &[Value::Felt252(x)]).return_value {
                Value::Enum { tag, .. } => tag == 0,
                _ => panic!("invalid return type"),
            }
        }

        assert!(r(f("0")));
        assert!(!r(f("1")));
        assert!(!r(f("-2")));
        assert!(!r(f("-1")));
    }
}
