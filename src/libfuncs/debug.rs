//! # Debug libfuncs

// Printable: 9-13, 27, 32, 33-126
//     is_ascii_graphic() -> 33-126
//     is_ascii_whitespace():
//         U+0009 HORIZONTAL TAB
//         U+000A LINE FEED
//         U+000C FORM FEED
//         U+000D CARRIAGE RETURN.
//         U+0020 SPACE

use super::LibfuncHelper;
use crate::{
    error::{Error, Result},
    metadata::{
        drop_overrides::DropOverridesMeta, runtime_bindings::RuntimeBindingsMeta, MetadataStorage,
    },
    utils::{BlockExt, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        debug::DebugConcreteLibfunc,
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{arith, cf, llvm},
    ir::{r#type::IntegerType, Block, Location},
    Context,
};

pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &DebugConcreteLibfunc,
) -> Result<()> {
    match selector {
        DebugConcreteLibfunc::Print(info) => {
            build_print(context, registry, entry, location, helper, metadata, info)
        }
    }
}

pub fn build_print<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let stdout_fd = entry.const_int(context, location, 1, 32)?;

    let values_ptr = entry.extract_value(
        context,
        location,
        entry.arg(0)?,
        llvm::r#type::pointer(context, 0),
        0,
    )?;
    let values_start = entry.extract_value(
        context,
        location,
        entry.arg(0)?,
        IntegerType::new(context, 32).into(),
        1,
    )?;
    let values_end = entry.extract_value(
        context,
        location,
        entry.arg(0)?,
        IntegerType::new(context, 32).into(),
        2,
    )?;

    let runtime_bindings = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.");

    let values_len = entry.append_op_result(arith::subi(values_end, values_start, location))?;

    let values_ptr = {
        let values_start =
            entry.extui(values_start, IntegerType::new(context, 64).into(), location)?;

        entry.append_op_result(llvm::get_element_ptr_dynamic(
            context,
            values_ptr,
            &[values_start],
            IntegerType::new(context, 252).into(),
            llvm::r#type::pointer(context, 0),
            location,
        ))?
    };

    let return_code = runtime_bindings.libfunc_debug_print(
        context, helper, entry, stdout_fd, values_ptr, values_len, location,
    )?;

    let input_ty = &info.signature.param_signatures[0].ty;
    registry.build_type(context, helper, metadata, input_ty)?;
    metadata
        .get::<DropOverridesMeta>()
        .ok_or(Error::MissingMetadata)?
        .invoke_override(context, entry, location, input_ty, entry.arg(0)?)?;

    let k0 = entry.const_int(context, location, 0, 32)?;
    let return_code_is_ok =
        entry.cmpi(context, arith::CmpiPredicate::Eq, return_code, k0, location)?;
    cf::assert(
        context,
        return_code_is_ok,
        "Print libfunc invocation failed.",
        location,
    );

    helper.br(entry, 0, &[], location)
}
