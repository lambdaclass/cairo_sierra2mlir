//! # Bytes31-related libfuncs

use super::LibfuncHelper;
use crate::{
    error::{Error, Result},
    metadata::MetadataStorage,
    utils::{BlockExt, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        bytes31::Bytes31ConcreteLibfunc,
        consts::SignatureAndConstConcreteLibfunc,
        core::{CoreLibfunc, CoreType},
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
    ir::{Attribute, Block, BlockLike, Location, Value},
    Context,
};
use num_bigint::BigUint;

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &Bytes31ConcreteLibfunc,
) -> Result<()> {
    match selector {
        Bytes31ConcreteLibfunc::Const(info) => {
            build_const(context, registry, entry, location, helper, metadata, info)
        }
        Bytes31ConcreteLibfunc::ToFelt252(info) => {
            build_to_felt252(context, registry, entry, location, helper, metadata, info)
        }
        Bytes31ConcreteLibfunc::TryFromFelt252(info) => {
            build_from_felt252(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `bytes31_const` libfunc.
pub fn build_const<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndConstConcreteLibfunc,
) -> Result<()> {
    let value = &info.c;
    let value_ty = registry.build_type(
        context,
        helper,
        metadata,
        &info.signature.branch_signatures[0].vars[0].ty,
    )?;

    let op0 = entry.append_operation(arith::constant(
        context,
        Attribute::parse(context, &format!("{value} : {value_ty}"))
            .ok_or(Error::ParseAttributeError)?,
        location,
    ));

    entry.append_operation(helper.br(0, &[op0.result(0)?.into()], location));

    Ok(())
}

/// Generate MLIR operations for the `bytes31_to_felt252` libfunc.
pub fn build_to_felt252<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let felt252_ty = registry.build_type(
        context,
        helper,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;
    let value: Value = entry.arg(0)?;

    let result = entry.extui(value, felt252_ty, location)?;

    entry.append_operation(helper.br(0, &[result], location));

    Ok(())
}

/// Generate MLIR operations for the `u8_from_felt252` libfunc.
pub fn build_from_felt252<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check: Value =
        super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let value: Value = entry.arg(1)?;

    let felt252_ty =
        registry.build_type(context, helper, metadata, &info.param_signatures()[1].ty)?;
    let result_ty = registry.build_type(
        context,
        helper,
        metadata,
        &info.branch_signatures()[0].vars[1].ty,
    )?;

    let max_value = BigUint::from(2u32).pow(248) - 1u32;

    let const_max = entry.append_op_result(arith::constant(
        context,
        Attribute::parse(context, &format!("{} : {}", max_value, felt252_ty))
            .ok_or(Error::ParseAttributeError)?,
        location,
    ))?;

    let is_ule = entry.cmpi(context, CmpiPredicate::Ule, value, const_max, location)?;

    let block_success = helper.append_block(Block::new(&[]));
    let block_failure = helper.append_block(Block::new(&[]));

    entry.append_operation(cf::cond_br(
        context,
        is_ule,
        block_success,
        block_failure,
        &[],
        &[],
        location,
    ));

    let value = block_success.trunci(value, result_ty, location)?;
    block_success.append_operation(helper.br(0, &[range_check, value], location));

    block_failure.append_operation(helper.br(1, &[range_check], location));

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{
        utils::{
            sierra_gen::SierraGenerator,
            test::{jit_enum, jit_struct, run_sierra_program},
        },
        Value,
    };
    use cairo_lang_sierra::{
        extensions::{
            bytes31::{Bytes31FromFelt252Trait, Bytes31ToFelt252Libfunc},
            try_from_felt252::TryFromFelt252Libfunc,
        },
        program::Program,
    };
    use lazy_static::lazy_static;
    use starknet_types_core::felt::Felt;

    lazy_static! {
        // TODO: Test `bytes31_const` once the compiler supports it.
        static ref BYTES31_TRY_FROM_FELT: Program = {
            let generator = SierraGenerator::<TryFromFelt252Libfunc<Bytes31FromFelt252Trait>>::default();

            generator.build(&[])
        };
        static ref BYTES31_TO_FELT: Program = {
            let generator = SierraGenerator::<Bytes31ToFelt252Libfunc>::default();

            generator.build(&[])
        };
    }

    #[test]
    fn bytes31_roundtrip() {
        let Value::Enum {
            tag: _,
            value,
            debug_name: _,
        } = run_sierra_program(&BYTES31_TRY_FROM_FELT, &[Felt::from(2).into()]).return_value
        else {
            panic!("Invalid result type");
        };
        let result = run_sierra_program(&BYTES31_TO_FELT, &[*value]).return_value;

        assert_eq!(Value::Felt252(2.into()), result);

        let result = run_sierra_program(&BYTES31_TRY_FROM_FELT, &[Felt::MAX.into()]).return_value;

        assert_eq!(jit_enum!(1, jit_struct!()), result);
    }
}
