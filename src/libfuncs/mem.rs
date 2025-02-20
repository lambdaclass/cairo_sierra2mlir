//! # Memory-related libfuncs
//!
//! Natively compiled code doesn't need this kind of memory tracking because it has no notion of the
//! segments. Because of this, all of the memory-related libfuncs here are no-ops.

use super::LibfuncHelper;
use crate::{
    error::Result,
    metadata::MetadataStorage,
    utils::{BlockExt, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        lib_func::{SignatureAndTypeConcreteLibfunc, SignatureOnlyConcreteLibfunc},
        mem::MemConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm,
    ir::{Block, Location},
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
    selector: &MemConcreteLibfunc,
) -> Result<()> {
    match selector {
        MemConcreteLibfunc::AllocLocal(info) => {
            build_alloc_local(context, registry, entry, location, helper, metadata, info)
        }
        MemConcreteLibfunc::StoreLocal(info) => {
            build_store_local(context, registry, entry, location, helper, metadata, info)
        }
        MemConcreteLibfunc::FinalizeLocals(info) => super::build_noop::<0, true>(
            context,
            registry,
            entry,
            location,
            helper,
            metadata,
            &info.signature.param_signatures,
        ),
        MemConcreteLibfunc::Rename(SignatureOnlyConcreteLibfunc { signature })
        | MemConcreteLibfunc::StoreTemp(SignatureAndTypeConcreteLibfunc { signature, .. }) => {
            super::build_noop::<1, false>(
                context,
                registry,
                entry,
                location,
                helper,
                metadata,
                &signature.param_signatures,
            )
        }
    }
}

/// Generate MLIR operations for the `alloc_local` libfunc.
pub fn build_alloc_local<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let target_type = registry.build_type(
        context,
        helper,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;

    let value = entry.append_op_result(llvm::undef(target_type, location))?;

    helper.br(entry, 0, &[value], location)
}

/// Generate MLIR operations for the `store_local` libfunc.
pub fn build_store_local<'ctx, 'this>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    helper.br(entry, 0, &[entry.arg(1)?], location)
}
