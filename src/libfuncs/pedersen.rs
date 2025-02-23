//! # Pedersen hashing libfuncs
//!

use super::LibfuncHelper;
use crate::{
    error::Result,
    metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
    utils::{get_integer_layout, BlockExt, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
        pedersen::PedersenConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    ir::{r#type::IntegerType, Block, BlockLike, Location},
    Context,
};

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &PedersenConcreteLibfunc,
) -> Result<()> {
    match selector {
        PedersenConcreteLibfunc::PedersenHash(info) => {
            build_pedersen(context, registry, entry, location, helper, metadata, info)
        }
    }
}

pub fn build_pedersen<'ctx>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'ctx Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, '_>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.");

    let pedersen_builtin =
        super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let felt252_ty =
        registry.build_type(context, helper, metadata, &info.param_signatures()[1].ty)?;

    let i256_ty = IntegerType::new(context, 256).into();
    let layout_i256 = get_integer_layout(256);

    let lhs = entry.arg(1)?;
    let rhs = entry.arg(2)?;

    // We must extend to i256 because bswap must be an even number of bytes.

    let lhs_ptr = helper
        .init_block()
        .alloca1(context, location, i256_ty, layout_i256.align())?;
    let rhs_ptr = helper
        .init_block()
        .alloca1(context, location, i256_ty, layout_i256.align())?;
    let dst_ptr = helper
        .init_block()
        .alloca1(context, location, i256_ty, layout_i256.align())?;

    let lhs_i256 = entry.extui(lhs, i256_ty, location)?;
    let rhs_i256 = entry.extui(rhs, i256_ty, location)?;

    entry.store(context, location, lhs_ptr, lhs_i256)?;
    entry.store(context, location, rhs_ptr, rhs_i256)?;

    let runtime_bindings = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.");

    runtime_bindings
        .libfunc_pedersen(context, helper, entry, dst_ptr, lhs_ptr, rhs_ptr, location)?;

    let result = entry.load(context, location, dst_ptr, i256_ty)?;
    let result = entry.trunci(result, felt252_ty, location)?;

    entry.append_operation(helper.br(0, &[pedersen_builtin, result], location));
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{
        utils::{sierra_gen::SierraGenerator, test::run_sierra_program},
        Value,
    };

    use cairo_lang_sierra::extensions::pedersen::PedersenHashLibfunc;
    use starknet_types_core::felt::Felt;

    #[test]
    fn run_pedersen() {
        let program = {
            let generator = SierraGenerator::<PedersenHashLibfunc>::default();

            generator.build(&[])
        };

        let result = run_sierra_program(&program, &[Felt::from(2).into(), Felt::from(4).into()])
            .return_value;

        assert_eq!(
            Value::Felt252(
                Felt::from_dec_str(
                    "2178161520066714737684323463974044933282313051386084149915030950231093462467",
                )
                .unwrap()
            ),
            result
        );
    }
}
