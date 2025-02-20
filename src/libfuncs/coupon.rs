//! # Branch alignment libfunc
//!
//! Natively compiled code doesn't need branch alignment because it has no notion of segments.
//! Because of this, this libfunc is a no-op.

use super::LibfuncHelper;
use crate::{
    error::Result,
    metadata::MetadataStorage,
    utils::{BlockExt, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        coupon::CouponConcreteLibfunc,
        function_call::SignatureAndFunctionConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm,
    ir::{Block, Location},
    Context,
};

/// Generate MLIR operations for the `coupon` libfuncs.
/// In native it's mostly a no-op operation.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &CouponConcreteLibfunc,
) -> Result<()> {
    match selector {
        CouponConcreteLibfunc::Buy(info) => {
            // Libfunc for buying a coupon for a function. The cost of the coupon is the cost of running the
            // function (not including the `call` and `ret` instructions).
            // The coupon can be used to pay in advance for running the function, and run it later for
            // free (paying only for the `call` and `ret` instructions) using `coupon_call`.
            build_buy(context, registry, entry, location, helper, metadata, info)
        }
        CouponConcreteLibfunc::Refund(info) => {
            // Libfunc for getting a refund for an unused coupon. The refund is the cost of the function
            // and it is added back to the gas wallet.
            build_refund(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `coupon` libfunc.
pub fn build_buy<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndFunctionConcreteLibfunc,
) -> Result<()> {
    // In the future if the gas cost is required, this is how to get it.
    // let gas = metadata.get::<GasMetadata>().ok_or(Error::MissingMetadata)?;
    // let gas_cost = gas.initial_required_gas(&info.function.id);
    let ty = registry.build_type(
        context,
        helper,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;
    let coupon = entry.append_op_result(llvm::undef(ty, location))?;

    helper.br(entry, 0, &[coupon], location)
}

/// Generate MLIR operations for the `coupon` libfunc.
pub fn build_refund<'ctx, 'this>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureAndFunctionConcreteLibfunc,
) -> Result<()> {
    // In the future if the gas cost is required, this is how to get it.
    // let gas = metadata.get::<GasMetadata>().ok_or(Error::MissingMetadata)?;
    // let gas_cost = gas.initial_required_gas(&info.function.id);

    helper.br(entry, 0, &[], location)
}
