#![cfg(feature = "with-cheatcode")]

use crate::{
    error::Result,
    libfuncs::LibfuncHelper,
    metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
    utils::{get_integer_layout, BlockExt, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        starknet::testing::CheatcodeConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm::{self, alloca, AllocaOptions, LoadStoreOptions},
    ir::{
        attribute::{IntegerAttribute, TypeAttribute},
        r#type::IntegerType,
        Block, BlockLike, Location,
    },
    Context,
};

pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &CheatcodeConcreteLibfunc,
) -> Result<()> {
    // Calculate the result layout and type, based on the branch signature
    let (result_type, result_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;

    // Allocate the result pointer with calculated layout and type
    let result_ptr = helper
        .init_block()
        .append_operation(alloca(
            context,
            helper.init_block().const_int(context, location, 1, 64)?,
            llvm::r#type::pointer(context, 0),
            location,
            AllocaOptions::new()
                .align(Some(IntegerAttribute::new(
                    IntegerType::new(context, 64).into(),
                    result_layout.align().try_into()?,
                )))
                .elem_type(Some(TypeAttribute::new(result_type))),
        ))
        .result(0)?
        .into();

    // Allocate and store selector. The type contains 256 bits as its interpreted as a [u8;32] from the runtime
    let selector = helper
        .init_block()
        .const_int(context, location, info.selector.clone(), 256)?;
    let selector_ptr = helper.init_block().alloca_int(context, location, 256)?;

    helper
        .init_block()
        .store(context, location, selector_ptr, selector)?;

    // Allocate and store arguments. The cairo type is a Span<Felt252> (the outer struct),
    // which contains an Array<Felt252> (the inner struct)
    let span_felt252_type = llvm::r#type::r#struct(
        context,
        &[llvm::r#type::r#struct(
            context,
            &[
                llvm::r#type::pointer(context, 0),
                IntegerType::new(context, 32).into(),
                IntegerType::new(context, 32).into(),
                IntegerType::new(context, 32).into(),
            ],
            false,
        )],
        false,
    );
    let args_ptr = helper.init_block().alloca1(
        context,
        location,
        span_felt252_type,
        get_integer_layout(64).align(),
    )?;
    entry.store(context, location, args_ptr, entry.argument(0)?.into())?;

    // Call runtime cheatcode syscall wrapper
    metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.")
        .vtable_cheatcode(
            context,
            helper,
            entry,
            location,
            result_ptr,
            selector_ptr,
            args_ptr,
        )?;

    // Load result from result ptr and branch
    let result = entry.append_op_result(llvm::load(
        context,
        result_ptr,
        result_type,
        location,
        LoadStoreOptions::new(),
    ))?;
    helper.br(entry, 0, &[result], location)
}
