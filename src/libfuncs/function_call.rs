//! # Function call libfuncs

use super::LibfuncHelper;
use crate::{
    block_ext::BlockExt, error::Result, metadata::MetadataStorage, types::TypeBuilder,
    utils::generate_function_name,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        function_call::SignatureAndFunctionConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{func, llvm},
    ir::{
        attribute::{DenseI32ArrayAttribute, FlatSymbolRefAttribute},
        r#type::IntegerType,
        Block, Location,
    },
    Context,
};
use std::alloc::Layout;

/// Generate MLIR operations for the `function_call` libfunc.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndFunctionConcreteLibfunc,
) -> Result<()> {
    let mut arguments = Vec::new();
    for (idx, type_id) in info.function.signature.param_types.iter().enumerate() {
        let type_info = registry.get_type(type_id)?;

        if !(type_info.is_builtin() && type_info.is_zst(registry)) {
            arguments.push(if type_info.is_memory_allocated(registry) {
                let elem_ty = type_info.build(context, helper, registry, metadata, type_id)?;
                let stack_ptr = helper.init_block().alloca1(
                    context,
                    location,
                    elem_ty,
                    type_info.layout(registry)?.align(),
                )?;

                entry.store(context, location, stack_ptr, entry.argument(idx)?.into())?;

                stack_ptr
            } else {
                entry.argument(idx)?.into()
            });
        }
    }

    let mut result_types = Vec::new();
    let return_types = info
        .function
        .signature
        .ret_types
        .iter()
        .filter_map(|type_id| {
            let type_info = registry.get_type(type_id).unwrap();
            if type_info.is_builtin() && type_info.is_zst(registry) {
                None
            } else {
                Some((type_id, type_info))
            }
        })
        .collect::<Vec<_>>();
    // A function has a return pointer if either:
    //   - There are multiple return values.
    //   - The return value is memory allocated.
    let has_return_ptr = if return_types.len() > 1 {
        result_types.extend(
            return_types
                .iter()
                .map(|(type_id, type_info)| {
                    type_info.build(context, helper, registry, metadata, type_id)
                })
                .collect::<std::result::Result<Vec<_>, _>>()?,
        );

        Some(false)
    } else if return_types
        .first()
        .is_some_and(|(_, type_info)| type_info.is_memory_allocated(registry))
    {
        let (type_id, type_info) = return_types[0];
        let layout = type_info.layout(registry)?;

        let stack_ptr = helper.init_block().alloca1(
            context,
            location,
            type_info.build(context, helper, registry, metadata, type_id)?,
            layout.align(),
        )?;

        arguments.insert(0, stack_ptr);

        Some(true)
    } else if return_types.first().is_some() {
        let (type_id, type_info) = return_types[0];
        result_types.push(type_info.build(context, helper, registry, metadata, type_id)?);

        None
    } else {
        None
    };

    let function_call_result = entry.append_operation(func::call(
        context,
        FlatSymbolRefAttribute::new(context, &generate_function_name(&info.function.id)),
        &arguments,
        &result_types,
        location,
    ));

    let mut results = Vec::new();
    match has_return_ptr {
        Some(true) => {
            // Manual return type.

            let mut layout = Layout::new::<()>();
            for (idx, type_id) in info.function.signature.ret_types.iter().enumerate() {
                let type_info = registry.get_type(type_id)?;

                if type_info.is_builtin() && type_info.is_zst(registry) {
                    results.push(entry.argument(idx)?.into());
                } else {
                    let val = arguments[0];

                    let offset;
                    let ret_layout = type_info.layout(registry)?;
                    (layout, offset) = layout.extend(ret_layout)?;

                    let pointer_val = entry.append_op_result(llvm::get_element_ptr(
                        context,
                        val,
                        DenseI32ArrayAttribute::new(context, &[offset as i32]),
                        IntegerType::new(context, 8).into(),
                        llvm::r#type::pointer(context, 0),
                        location,
                    ))?;

                    results.push(entry.load(
                        context,
                        location,
                        pointer_val,
                        type_info.build(context, helper, registry, metadata, type_id)?,
                    )?);
                }
            }
        }
        Some(false) => {
            // Complex return type. Just extract the values from the struct, since LLVM will
            // handle the rest.

            let mut count = 0;
            for (idx, type_id) in info.function.signature.ret_types.iter().enumerate() {
                let type_info = registry.get_type(type_id)?;

                if type_info.is_builtin() && type_info.is_zst(registry) {
                    results.push(entry.argument(idx)?.into());
                } else {
                    let val = function_call_result.result(count)?.into();
                    count += 1;

                    results.push(val);
                }
            }
        }
        None => {
            // Returned data is simple.

            let mut count = 0;
            for (idx, type_id) in info.function.signature.ret_types.iter().enumerate() {
                let type_info = registry.get_type(type_id)?;
                assert!(!type_info.is_memory_allocated(registry));

                if type_info.is_builtin() && type_info.is_zst(registry) {
                    results.push(entry.argument(idx)?.into());
                } else {
                    let value = function_call_result.result(count)?.into();
                    count += 1;

                    results.push(value);
                }
            }
        }
    }

    entry.append_operation(helper.br(0, &results, location));
    Ok(())
}
