//! # State value duplication libfunc
//!
//! Most types are trivial and don't need any clone (or rather, they will be cloned automatically by
//! MLIR). For those types, this libfunc is a no-op.
//!
//! However, types like an array need special handling.

use super::{BlockExt, LibfuncHelper};
use crate::{
    error::Result,
    metadata::{dup_overrides::DupOverridesMeta, MetadataStorage},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    ir::{Block, Location},
    Context,
};

/// Generate MLIR operations for the `dup` libfunc.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Note: The Cairo compiler will avoid using `dup` for some non-trivially-copyable types, but
    //   not all of them. For example, it'll not generate a clone implementation for `Box<T>`.
    //   That's why we need to check for clone implementations within the compiler.

    let values = metadata
        .get_or_insert_with(DupOverridesMeta::default)
        .invoke_override(
            context,
            entry,
            location,
            &info.signature.param_signatures[0].ty,
            entry.arg(0)?,
        )?;
    helper.br(entry, 0, &[values.0, values.1], location)
}
