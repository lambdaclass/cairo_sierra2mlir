//! # Struct-related libfuncs

use super::LibfuncHelper;
use crate::{
    error::Result,
    metadata::MetadataStorage,
    types::TypeBuilder,
    utils::{BlockExt, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
        structure::StructConcreteLibfunc,
        ConcreteLibfunc,
    },
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm,
    ir::{Block, BlockLike, Location, Value},
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
    selector: &StructConcreteLibfunc,
) -> Result<()> {
    match selector {
        StructConcreteLibfunc::Construct(info) => {
            build_construct(context, registry, entry, location, helper, metadata, info)
        }
        StructConcreteLibfunc::Deconstruct(info)
        | StructConcreteLibfunc::SnapshotDeconstruct(info) => {
            build_deconstruct(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `struct_construct` libfunc.
pub fn build_construct<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let mut fields = Vec::new();

    for (i, _) in info.param_signatures().iter().enumerate() {
        fields.push(entry.argument(i)?.into());
    }

    let value = build_struct_value(
        context,
        registry,
        entry,
        location,
        helper,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
        &fields,
    )?;

    helper.br(entry, 0, &[value], location)
}

/// Generate MLIR operations for the `struct_construct` libfunc.
#[allow(clippy::too_many_arguments)]
pub fn build_struct_value<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    struct_type: &ConcreteTypeId,
    fields: &[Value<'ctx, 'this>],
) -> Result<Value<'ctx, 'this>> {
    let struct_ty = registry.build_type(context, helper, metadata, struct_type)?;

    let acc = entry.append_operation(llvm::undef(struct_ty, location));

    entry.insert_values(context, location, acc.result(0)?.into(), fields)
}

/// Generate MLIR operations for the `struct_deconstruct` libfunc.
pub fn build_deconstruct<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let container = entry.arg(0)?;

    let mut fields = Vec::<Value>::with_capacity(info.branch_signatures()[0].vars.len());
    for (i, var_info) in info.branch_signatures()[0].vars.iter().enumerate() {
        let type_info = registry.get_type(&var_info.ty)?;
        let field_ty = type_info.build(context, helper, registry, metadata, &var_info.ty)?;

        let value = entry.extract_value(context, location, container, field_ty, i)?;

        fields.push(value);
    }

    helper.br(entry, 0, &fields, location)
}
