//! # Array libfuncs

use super::LibfuncHelper;
use crate::{
    error::{panic::ToNativeAssertError, Error, Result, SierraAssertError},
    metadata::{
        drop_overrides::DropOverridesMeta, dup_overrides::DupOverridesMeta,
        realloc_bindings::ReallocBindingsMeta, MetadataStorage,
    },
    utils::{get_integer_layout, BlockExt, GepIndex, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        array::{ArrayConcreteLibfunc, ConcreteMultiPopLibfunc},
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        lib_func::{SignatureAndTypeConcreteLibfunc, SignatureOnlyConcreteLibfunc},
        types::InfoAndTypeConcreteType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        cf, llvm, ods, scf,
    },
    ir::{
        attribute::IntegerAttribute, r#type::IntegerType, Block, BlockLike, Location, Region,
        Value, ValueLike,
    },
    Context,
};
use std::alloc::Layout;

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &ArrayConcreteLibfunc,
) -> Result<()> {
    match selector {
        ArrayConcreteLibfunc::New(info) => {
            build_new(context, registry, entry, location, helper, metadata, info)
        }
        ArrayConcreteLibfunc::SpanFromTuple(info) => {
            build_span_from_tuple(context, registry, entry, location, helper, metadata, info)
        }
        ArrayConcreteLibfunc::TupleFromSpan(info) => {
            build_tuple_from_span(context, registry, entry, location, helper, metadata, info)
        }
        ArrayConcreteLibfunc::Append(info) => {
            build_append(context, registry, entry, location, helper, metadata, info)
        }
        ArrayConcreteLibfunc::PopFront(info) => build_pop::<false, false>(
            context,
            registry,
            entry,
            location,
            helper,
            metadata,
            PopInfo::Single(info),
        ),
        ArrayConcreteLibfunc::PopFrontConsume(info) => build_pop::<true, false>(
            context,
            registry,
            entry,
            location,
            helper,
            metadata,
            PopInfo::Single(info),
        ),
        ArrayConcreteLibfunc::Get(info) => {
            build_get(context, registry, entry, location, helper, metadata, info)
        }
        ArrayConcreteLibfunc::Slice(info) => {
            build_slice(context, registry, entry, location, helper, metadata, info)
        }
        ArrayConcreteLibfunc::Len(info) => {
            build_len(context, registry, entry, location, helper, metadata, info)
        }
        ArrayConcreteLibfunc::SnapshotPopFront(info) => build_pop::<false, false>(
            context,
            registry,
            entry,
            location,
            helper,
            metadata,
            PopInfo::Single(info),
        ),
        ArrayConcreteLibfunc::SnapshotPopBack(info) => build_pop::<false, true>(
            context,
            registry,
            entry,
            location,
            helper,
            metadata,
            PopInfo::Single(info),
        ),
        ArrayConcreteLibfunc::SnapshotMultiPopFront(info) => build_pop::<false, false>(
            context,
            registry,
            entry,
            location,
            helper,
            metadata,
            PopInfo::Multi(info),
        ),
        ArrayConcreteLibfunc::SnapshotMultiPopBack(info) => build_pop::<false, true>(
            context,
            registry,
            entry,
            location,
            helper,
            metadata,
            PopInfo::Multi(info),
        ),
    }
}

/// Buils a new array with no initial capacity
///
/// # Cairo Signature
///
/// ```cairo
/// extern fn array_new<T>() -> Array<T> nopanic;
/// ```
pub fn build_new<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let ptr_ty = llvm::r#type::pointer(context, 0);
    let len_ty = IntegerType::new(context, 32).into();

    let nullptr = entry.append_op_result(llvm::zero(ptr_ty, location))?;
    let k0 = entry.const_int_from_type(context, location, 0, len_ty)?;

    let value = entry.append_op_result(llvm::undef(
        llvm::r#type::r#struct(context, &[ptr_ty, len_ty, len_ty, len_ty], false),
        location,
    ))?;
    let value = entry.insert_values(context, location, value, &[nullptr, k0, k0, k0])?;

    entry.append_operation(helper.br(0, &[value], location));
    Ok(())
}

/// Buils a span (a cairo native array) from a box of a tuple (struct with elements of the same type)
///
/// Note: The `&info.ty` field has the entire `[T; N]` tuple. It is not the `T` in `Array<T>`.
///
/// # Cairo Signature
///
/// ```cairo
/// extern fn span_from_tuple<T, impl Info: FixedSizedArrayInfo<T>>(struct_like: Box<@T>) -> @Array<Info::Element> nopanic;
/// ```
pub fn build_span_from_tuple<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    metadata.get_or_insert_with(|| ReallocBindingsMeta::new(context, helper));

    let tuple_len = {
        let CoreTypeConcrete::Struct(info) = registry.get_type(&info.ty)? else {
            return Err(Error::SierraAssert(SierraAssertError::BadTypeInfo));
        };

        info.members.len()
    };

    let ptr_ty = llvm::r#type::pointer(context, 0);
    let len_ty = IntegerType::new(context, 32).into();
    let (_, tuple_layout) = registry.build_type_with_layout(context, helper, metadata, &info.ty)?;

    let array_len_bytes = tuple_layout.pad_to_align().size();
    let array_len_bytes_with_offset = entry.const_int(
        context,
        location,
        array_len_bytes + calc_refcount_offset(tuple_layout),
        64,
    )?;
    let array_len_bytes = entry.const_int(context, location, array_len_bytes, 64)?;
    let array_len = entry.const_int_from_type(context, location, tuple_len, len_ty)?;

    let k0 = entry.const_int_from_type(context, location, 0, len_ty)?;
    let k1 = entry.const_int_from_type(context, location, 1, len_ty)?;

    // build the new span (array)
    let allocation_ptr = entry.append_op_result(llvm::zero(ptr_ty, location))?;
    let allocation_ptr = entry.append_op_result(ReallocBindingsMeta::realloc(
        context,
        allocation_ptr,
        array_len_bytes_with_offset,
        location,
    )?)?;
    entry.store(context, location, allocation_ptr, k1)?;
    let array_ptr = entry.gep(
        context,
        location,
        allocation_ptr,
        &[GepIndex::Const(calc_refcount_offset(tuple_layout) as i32)],
        IntegerType::new(context, 8).into(),
    )?;

    // as a tuple has the same representation as the array data,
    // we just memcpy into the new array.
    entry.memcpy(
        context,
        location,
        entry.argument(0)?.into(),
        array_ptr,
        array_len_bytes,
    );
    entry.append_operation(ReallocBindingsMeta::free(
        context,
        entry.argument(0)?.into(),
        location,
    )?);

    let value = entry.append_op_result(llvm::undef(
        llvm::r#type::r#struct(context, &[ptr_ty, len_ty, len_ty, len_ty], false),
        location,
    ))?;
    let value = entry.insert_values(
        context,
        location,
        value,
        &[array_ptr, k0, array_len, array_len],
    )?;

    entry.append_operation(helper.br(0, &[value], location));
    Ok(())
}

/// Buils a tuple (struct) from an span (a cairo native array)
///
/// Note: The `&info.ty` field has the entire `[T; N]` tuple. It is not the `T` in `Array<T>`.
/// The tuple size `N` must match the span length.
///
/// # Cairo Signature
///
/// ```cairo
/// fn tuple_from_span<T, impl Info: FixedSizedArrayInfo<T>>(span: @Array<Info::Element>  -> Option<@Box<T>> nopanic;
/// ```
pub fn build_tuple_from_span<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    metadata.get_or_insert_with(|| ReallocBindingsMeta::new(context, helper));

    let elem_id = {
        let CoreTypeConcrete::Snapshot(info) =
            registry.get_type(&info.signature.param_signatures[0].ty)?
        else {
            return Err(Error::SierraAssert(SierraAssertError::BadTypeInfo));
        };
        let CoreTypeConcrete::Array(info) = registry.get_type(&info.ty)? else {
            return Err(Error::SierraAssert(SierraAssertError::BadTypeInfo));
        };

        &info.ty
    };
    let tuple_len_const = {
        let CoreTypeConcrete::Struct(param) = registry.get_type(&info.ty)? else {
            return Err(Error::SierraAssert(SierraAssertError::BadTypeInfo));
        };

        param.members.len()
    };

    let ptr_ty = llvm::r#type::pointer(context, 0);
    let len_ty = IntegerType::new(context, 32).into();
    let (_, elem_layout) = registry.build_type_with_layout(context, helper, metadata, elem_id)?;
    let (tuple_ty, tuple_layout) =
        registry.build_type_with_layout(context, helper, metadata, &info.ty)?;

    let array_ptr = entry.extract_value(context, location, entry.argument(0)?.into(), ptr_ty, 0)?;
    let array_start =
        entry.extract_value(context, location, entry.argument(0)?.into(), len_ty, 1)?;
    let array_end = entry.extract_value(context, location, entry.argument(0)?.into(), len_ty, 2)?;

    let array_len = entry.append_op_result(arith::subi(array_end, array_start, location))?;
    let tuple_len = entry.const_int_from_type(context, location, tuple_len_const, len_ty)?;
    let len_matches = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Eq,
        array_len,
        tuple_len,
        location,
    ))?;

    // check if the expected tuple matches the array length
    let valid_block = helper.append_block(Block::new(&[]));
    let error_block = helper.append_block(Block::new(&[]));
    entry.append_operation(cf::cond_br(
        context,
        len_matches,
        valid_block,
        error_block,
        &[],
        &[],
        location,
    ));

    // Ensure the type's clone and drop implementations are registered.
    registry.build_type(
        context,
        helper,
        metadata,
        &info.signature.param_signatures[0].ty,
    )?;

    {
        // if the length matches...

        let value_size = valid_block.const_int(context, location, tuple_layout.size(), 64)?;

        let value = valid_block.append_op_result(llvm::zero(ptr_ty, location))?;
        let value = valid_block.append_op_result(ReallocBindingsMeta::realloc(
            context, value, value_size, location,
        )?)?;

        // check if the array is shared
        let is_shared = is_shared(context, valid_block, location, array_ptr, elem_layout)?;

        let array_start_offset = valid_block.append_op_result(arith::extui(
            array_start,
            IntegerType::new(context, 64).into(),
            location,
        ))?;
        let array_start_offset = valid_block.append_op_result(arith::muli(
            array_start_offset,
            valid_block.const_int(context, location, elem_layout.pad_to_align().size(), 64)?,
            location,
        ))?;
        let array_data_start_ptr = valid_block.gep(
            context,
            location,
            array_ptr,
            &[GepIndex::Value(array_start_offset)],
            IntegerType::new(context, 8).into(),
        )?;

        valid_block.append_operation(scf::r#if(
            is_shared,
            &[],
            {
                // if the array is shared we clone the inner data,
                // as a tuple does not contain a reference counter.

                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                match metadata.get::<DupOverridesMeta>() {
                    Some(dup_overrides_meta) if dup_overrides_meta.is_overriden(&info.ty) => {
                        let src_ptr = array_data_start_ptr;
                        let dst_ptr = value;

                        let value = block.load(context, location, src_ptr, tuple_ty)?;

                        // as the array data has the same representation as the tuple,
                        // we can use the tuple override, which is simpler.
                        let values = dup_overrides_meta
                            .invoke_override(context, &block, location, &info.ty, value)?;
                        block.store(context, location, src_ptr, values.0)?;
                        block.store(context, location, dst_ptr, values.1)?;
                    }
                    _ => block.memcpy(context, location, array_data_start_ptr, value, value_size),
                }

                // drop the original array (decreasing its reference counter)
                metadata
                    .get::<DropOverridesMeta>()
                    .to_native_assert_error("array always has a drop implementation")?
                    .invoke_override(
                        context,
                        &block,
                        location,
                        &info.signature.param_signatures[0].ty,
                        entry.argument(0)?.into(),
                    )?;

                block.append_operation(scf::r#yield(&[], location));
                region
            },
            {
                // if the array is not shared, then move the data to the new tuple
                // and manually free the allocation (without calling drop on its elements).

                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                block.memcpy(context, location, array_data_start_ptr, value, value_size);

                // NOTE: If the target tuple has no elements, and the array is not shared,
                // then we will attempt to free 0xfffffffffffffff0. This is not possible and
                // disallowed by the cairo compiler.

                let array_allocation_ptr = block.gep(
                    context,
                    location,
                    array_ptr,
                    &[GepIndex::Const(-(calc_refcount_offset(elem_layout) as i32))],
                    IntegerType::new(context, 8).into(),
                )?;
                block.append_operation(ReallocBindingsMeta::free(
                    context,
                    array_allocation_ptr,
                    location,
                )?);

                block.append_operation(scf::r#yield(&[], location));
                region
            },
            location,
        ));

        valid_block.append_operation(helper.br(0, &[value], location));
    }

    {
        // if the length doesn't match, free the tuple.

        metadata
            .get::<DropOverridesMeta>()
            .ok_or(Error::MissingMetadata)?
            .invoke_override(
                context,
                error_block,
                location,
                &info.signature.param_signatures[0].ty,
                entry.argument(0)?.into(),
            )?;

        error_block.append_operation(helper.br(1, &[], location));
    }

    Ok(())
}

/// Generate MLIR operations for the `array_append` libfunc.
pub fn build_append<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    /*
     * 1. Check if shared.
     * 2. If shared:
     *   1. Deep clone with space for at least 1 extra element.
     * 3. If not shared:
     *   1. Either realloc, move or do nothing.
     * 4. Append element.
     */

    metadata.get_or_insert_with(|| ReallocBindingsMeta::new(context, helper));

    let self_ty = registry.build_type(
        context,
        helper,
        metadata,
        &info.signature.param_signatures[0].ty,
    )?;

    let ptr_ty = llvm::r#type::pointer(context, 0);
    let len_ty = IntegerType::new(context, 32).into();

    let (elem_ty, elem_layout) =
        registry.build_type_with_layout(context, helper, metadata, &info.ty)?;
    let elem_stride = entry.const_int(context, location, elem_layout.pad_to_align().size(), 64)?;

    let k0 = entry.const_int(context, location, 0, 32)?;
    let k1 = entry.const_int(context, location, 1, 32)?;

    let array_ptr = entry.extract_value(context, location, entry.argument(0)?.into(), ptr_ty, 0)?;
    let array_start =
        entry.extract_value(context, location, entry.argument(0)?.into(), len_ty, 1)?;
    let array_end = entry.extract_value(context, location, entry.argument(0)?.into(), len_ty, 2)?;
    let array_capacity =
        entry.extract_value(context, location, entry.argument(0)?.into(), len_ty, 3)?;

    let array_len = entry.append_op_result(arith::subi(array_end, array_start, location))?;
    let array_size = entry.append_op_result(arith::extui(
        array_len,
        IntegerType::new(context, 64).into(),
        location,
    ))?;
    let array_size = entry.append_op_result(arith::muli(array_size, elem_stride, location))?;

    let data_offset = entry.append_op_result(arith::extui(
        array_start,
        IntegerType::new(context, 64).into(),
        location,
    ))?;
    let data_offset = entry.append_op_result(arith::muli(data_offset, elem_stride, location))?;
    let data_ptr = entry.gep(
        context,
        location,
        array_ptr,
        &[GepIndex::Value(data_offset)],
        IntegerType::new(context, 8).into(),
    )?;

    fn compute_next_capacity<'ctx, 'this>(
        context: &'ctx Context,
        block: &'this Block<'ctx>,
        location: Location<'ctx>,
        elem_stride: Value<'ctx, 'this>,
        array_end: Value<'ctx, 'this>,
    ) -> Result<(Value<'ctx, 'this>, Value<'ctx, 'this>)> {
        let len_ty = IntegerType::new(context, 32).into();

        let k1 = block.const_int_from_type(context, location, 1, len_ty)?;
        let k8 = block.const_int_from_type(context, location, 8, len_ty)?;
        let k1024 = block.const_int_from_type(context, location, 1024, len_ty)?;

        let realloc_len = block.append_op_result(arith::shli(array_end, k1, location))?;
        let realloc_len = block.append_op_result(arith::minui(realloc_len, k1024, location))?;
        let realloc_len = block.append_op_result(arith::addi(realloc_len, array_end, location))?;
        let realloc_len = block.append_op_result(arith::maxui(realloc_len, k8, location))?;

        let realloc_size = block.append_op_result(arith::extui(
            realloc_len,
            IntegerType::new(context, 64).into(),
            location,
        ))?;
        let realloc_size =
            block.append_op_result(arith::muli(realloc_size, elem_stride, location))?;

        Result::Ok((realloc_len, realloc_size))
    }

    let is_empty = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Eq,
        array_capacity,
        k0,
        location,
    ))?;

    let is_shared = entry.append_op_result(scf::r#if(
        is_empty,
        &[IntegerType::new(context, 1).into()],
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let k0 = block.const_int(context, location, 0, 1)?;

            block.append_operation(scf::r#yield(&[k0], location));
            region
        },
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let is_shared = is_shared(context, entry, location, array_ptr, elem_layout)?;

            block.append_operation(scf::r#yield(&[is_shared], location));
            region
        },
        location,
    ))?;

    let value = entry.append_op_result(scf::r#if(
        is_shared,
        &[self_ty],
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let has_space = block.append_op_result(arith::cmpi(
                context,
                CmpiPredicate::Ugt,
                array_capacity,
                array_len,
                location,
            ))?;

            let op = block.append_operation(scf::r#if(
                has_space,
                &[len_ty, IntegerType::new(context, 64).into()],
                {
                    let region = Region::new();
                    let block = region.append_block(Block::new(&[]));

                    let clone_size = block.append_op_result(arith::extui(
                        array_capacity,
                        IntegerType::new(context, 64).into(),
                        location,
                    ))?;
                    let clone_size =
                        block.append_op_result(arith::muli(clone_size, elem_stride, location))?;

                    block.append_operation(scf::r#yield(&[array_capacity, clone_size], location));
                    region
                },
                {
                    let region = Region::new();
                    let block = region.append_block(Block::new(&[]));

                    let (realloc_capacity, realloc_size) =
                        compute_next_capacity(context, &block, location, elem_stride, array_end)?;

                    block.append_operation(scf::r#yield(
                        &[realloc_capacity, realloc_size],
                        location,
                    ));
                    region
                },
                location,
            ));
            let clone_capacity = op.result(0)?.into();
            let clone_size = op.result(1)?.into();

            let clone_size_with_refcount = block.append_op_result(arith::addi(
                clone_size,
                block.const_int(context, location, calc_refcount_offset(elem_layout), 64)?,
                location,
            ))?;

            let clone_ptr = block.append_op_result(llvm::zero(ptr_ty, location))?;
            let clone_ptr = block.append_op_result(ReallocBindingsMeta::realloc(
                context,
                clone_ptr,
                clone_size_with_refcount,
                location,
            )?)?;
            block.store(context, location, clone_ptr, k1)?;

            let clone_ptr = block.gep(
                context,
                location,
                clone_ptr,
                &[GepIndex::Const(calc_refcount_offset(elem_layout) as i32)],
                IntegerType::new(context, 8).into(),
            )?;

            match metadata.get::<DupOverridesMeta>() {
                Some(dup_overrides_meta) if dup_overrides_meta.is_overriden(&info.ty) => {
                    let k0 = block.const_int(context, location, 0, 64)?;
                    block.append_operation(scf::r#for(
                        k0,
                        array_size,
                        elem_stride,
                        {
                            let region = Region::new();
                            let block = region.append_block(Block::new(&[(
                                IntegerType::new(context, 64).into(),
                                location,
                            )]));

                            let offset = block.argument(0)?.into();
                            let source_ptr = block.gep(
                                context,
                                location,
                                data_ptr,
                                &[GepIndex::Value(offset)],
                                IntegerType::new(context, 8).into(),
                            )?;
                            let target_ptr = block.gep(
                                context,
                                location,
                                clone_ptr,
                                &[GepIndex::Value(offset)],
                                IntegerType::new(context, 8).into(),
                            )?;

                            let value = block.load(context, location, source_ptr, elem_ty)?;
                            let values = dup_overrides_meta
                                .invoke_override(context, &block, location, &info.ty, value)?;
                            block.store(context, location, source_ptr, values.0)?;
                            block.store(context, location, target_ptr, values.1)?;

                            block.append_operation(scf::r#yield(&[], location));
                            region
                        },
                        location,
                    ));
                }
                _ => block.memcpy(context, location, array_ptr, clone_ptr, clone_size),
            }

            let clone_value = block.append_op_result(llvm::zero(self_ty, location))?;
            let clone_value = block.insert_values(
                context,
                location,
                clone_value,
                &[clone_ptr, k0, array_len, clone_capacity],
            )?;

            metadata
                .get::<DropOverridesMeta>()
                .to_native_assert_error("drop overrides meta should exist")?
                .invoke_override(
                    context,
                    &block,
                    location,
                    &info.signature.param_signatures[0].ty,
                    entry.argument(0)?.into(),
                )?;

            block.append_operation(scf::r#yield(&[clone_value], location));
            region
        },
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let has_tail_space = block.append_op_result(arith::cmpi(
                context,
                CmpiPredicate::Ult,
                array_end,
                array_capacity,
                location,
            ))?;
            let array_value = block.append_op_result(scf::r#if(
                has_tail_space,
                &[self_ty],
                {
                    let region = Region::new();
                    let block = region.append_block(Block::new(&[]));

                    block.append_operation(scf::r#yield(&[entry.argument(0)?.into()], location));
                    region
                },
                {
                    let region = Region::new();
                    let block = region.append_block(Block::new(&[]));

                    let has_head_space = block.append_op_result(arith::cmpi(
                        context,
                        CmpiPredicate::Ugt,
                        array_start,
                        k0,
                        location,
                    ))?;
                    let array_value = block.append_op_result(scf::r#if(
                        has_head_space,
                        &[self_ty],
                        {
                            let region = Region::new();
                            let block = region.append_block(Block::new(&[]));

                            block.append_operation(
                                ods::llvm::intr_memmove(
                                    context,
                                    array_ptr,
                                    data_ptr,
                                    array_size,
                                    IntegerAttribute::new(IntegerType::new(context, 1).into(), 0),
                                    location,
                                )
                                .into(),
                            );

                            let array_value = block.insert_value(
                                context,
                                location,
                                entry.argument(0)?.into(),
                                k0,
                                1,
                            )?;
                            let array_value =
                                block.insert_value(context, location, array_value, array_len, 2)?;

                            block.append_operation(scf::r#yield(&[array_value], location));
                            region
                        },
                        {
                            let region = Region::new();
                            let block = region.append_block(Block::new(&[]));

                            let offset_array_ptr = block.gep(
                                context,
                                location,
                                array_ptr,
                                &[GepIndex::Const(-(calc_refcount_offset(elem_layout) as i32))],
                                IntegerType::new(context, 8).into(),
                            )?;
                            let array_ptr = block.append_op_result(arith::select(
                                is_empty,
                                array_ptr,
                                offset_array_ptr,
                                location,
                            ))?;

                            let (realloc_len, realloc_size) = compute_next_capacity(
                                context,
                                &block,
                                location,
                                elem_stride,
                                array_end,
                            )?;
                            let realloc_size_with_refcount =
                                block.append_op_result(arith::addi(
                                    realloc_size,
                                    block.const_int(
                                        context,
                                        location,
                                        calc_refcount_offset(elem_layout),
                                        64,
                                    )?,
                                    location,
                                ))?;

                            let array_ptr =
                                block.append_op_result(ReallocBindingsMeta::realloc(
                                    context,
                                    array_ptr,
                                    realloc_size_with_refcount,
                                    location,
                                )?)?;

                            let ref_count = block.load(context, location, array_ptr, len_ty)?;
                            let ref_count = block.append_op_result(arith::select(
                                is_empty, k1, ref_count, location,
                            ))?;
                            block.store(context, location, array_ptr, ref_count)?;

                            let array_ptr = block.gep(
                                context,
                                location,
                                array_ptr,
                                &[GepIndex::Const(calc_refcount_offset(elem_layout) as i32)],
                                IntegerType::new(context, 8).into(),
                            )?;

                            let array_value = block.insert_value(
                                context,
                                location,
                                entry.argument(0)?.into(),
                                array_ptr,
                                0,
                            )?;
                            let array_value = block.insert_value(
                                context,
                                location,
                                array_value,
                                realloc_len,
                                3,
                            )?;

                            block.append_operation(scf::r#yield(&[array_value], location));
                            region
                        },
                        location,
                    ))?;

                    block.append_operation(scf::r#yield(&[array_value], location));
                    region
                },
                location,
            ))?;

            block.append_operation(scf::r#yield(&[array_value], location));
            region
        },
        location,
    ))?;

    let array_ptr = entry.extract_value(context, location, value, ptr_ty, 0)?;
    let array_end = entry.extract_value(context, location, value, len_ty, 2)?;

    let data_offset = entry.append_op_result(arith::extui(
        array_end,
        IntegerType::new(context, 64).into(),
        location,
    ))?;
    let data_offset = entry.append_op_result(arith::muli(data_offset, elem_stride, location))?;
    let data_ptr = entry.gep(
        context,
        location,
        array_ptr,
        &[GepIndex::Value(data_offset)],
        IntegerType::new(context, 8).into(),
    )?;
    entry.store(context, location, data_ptr, entry.argument(1)?.into())?;

    let array_end = entry.append_op_result(arith::addi(array_end, k1, location))?;
    let value = entry.insert_value(context, location, value, array_end, 2)?;

    entry.append_operation(helper.br(0, &[value], location));
    Ok(())
}

#[derive(Clone, Copy)]
enum PopInfo<'a> {
    Single(&'a SignatureAndTypeConcreteLibfunc),
    Multi(&'a ConcreteMultiPopLibfunc),
}

/// Generate MLIR operations for the `array_pop_*` libfuncs.
///
/// Template arguments:
///   - Consume: Whether to consume or not the array on failure.
///   - Reverse: False for front-popping, true for back-popping.
///
/// The `info` argument contains how many items to pop.
fn build_pop<'ctx, 'this, const CONSUME: bool, const REVERSE: bool>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: PopInfo,
) -> Result<()> {
    /*
     * 1. Check if there's enough data to pop.
     * 2. If there is not enough data, maybe consume and return.
     * 3. Allocate output.
     * 4. Clone or copy the popped data.
     *    - Clone if shared.
     *    - Copy if not shared.
     */

    metadata.get_or_insert_with(|| ReallocBindingsMeta::new(context, helper));

    let ptr_ty = llvm::r#type::pointer(context, 0);
    let len_ty = IntegerType::new(context, 32).into();

    let (self_ty, elem_ty, array_value, extract_len, mut branch_values) = match info {
        PopInfo::Single(info) => (
            &info.signature.param_signatures[0].ty,
            &info.ty,
            entry.argument(0)?.into(),
            1,
            Vec::new(),
        ),
        PopInfo::Multi(ConcreteMultiPopLibfunc {
            popped_ty,
            signature,
        }) => {
            let range_check = super::increment_builtin_counter(
                context,
                entry,
                location,
                entry.argument(0)?.into(),
            )?;

            let CoreTypeConcrete::Snapshot(InfoAndTypeConcreteType { ty, .. }) =
                registry.get_type(&signature.param_signatures[1].ty)?
            else {
                return Err(Error::SierraAssert(SierraAssertError::BadTypeInfo));
            };

            let CoreTypeConcrete::Array(InfoAndTypeConcreteType { ty, .. }) =
                registry.get_type(ty)?
            else {
                return Err(Error::SierraAssert(SierraAssertError::BadTypeInfo));
            };

            let CoreTypeConcrete::Struct(info) = registry.get_type(popped_ty)? else {
                return Err(Error::SierraAssert(SierraAssertError::BadTypeInfo));
            };
            debug_assert!(info.members.iter().all(|member_ty| member_ty == ty));

            (
                &signature.param_signatures[1].ty,
                ty,
                entry.argument(1)?.into(),
                info.members.len(),
                vec![range_check],
            )
        }
    };

    registry.build_type(context, helper, metadata, self_ty)?;
    let extract_len_value = entry.const_int_from_type(context, location, extract_len, len_ty)?;

    let (elem_type, elem_layout) =
        registry.build_type_with_layout(context, helper, metadata, elem_ty)?;

    let array_start = entry.extract_value(context, location, array_value, len_ty, 1)?;
    let array_end = entry.extract_value(context, location, array_value, len_ty, 2)?;

    let array_len = entry.append_op_result(arith::subi(array_end, array_start, location))?;
    let has_enough_data = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Ule,
        extract_len_value,
        array_len,
        location,
    ))?;

    let valid_block = helper.append_block(Block::new(&[]));
    let error_block = helper.append_block(Block::new(&[]));
    entry.append_operation(cf::cond_br(
        context,
        has_enough_data,
        valid_block,
        error_block,
        &[],
        &[],
        location,
    ));

    {
        // Clone branch_values so that it doesn't interfere with the other branch.
        let mut branch_values = branch_values.clone();

        let value_size = valid_block.const_int(
            context,
            location,
            elem_layout.pad_to_align().size() * extract_len,
            64,
        )?;

        let value_ptr = valid_block.append_op_result(llvm::zero(ptr_ty, location))?;
        let value_ptr = valid_block.append_op_result(ReallocBindingsMeta::realloc(
            context, value_ptr, value_size, location,
        )?)?;

        let array_ptr = valid_block.extract_value(context, location, array_value, ptr_ty, 0)?;
        let is_shared = is_shared(context, valid_block, location, array_ptr, elem_layout)?;

        let data_ptr = {
            let offset_elems = if REVERSE {
                valid_block.append_op_result(arith::subi(array_end, extract_len_value, location))?
            } else {
                array_start
            };

            let offset = valid_block.append_op_result(arith::extui(
                offset_elems,
                IntegerType::new(context, 64).into(),
                location,
            ))?;
            let elem_stride =
                valid_block.const_int(context, location, elem_layout.pad_to_align().size(), 64)?;
            let offset =
                valid_block.append_op_result(arith::muli(offset, elem_stride, location))?;

            valid_block.gep(
                context,
                location,
                array_ptr,
                &[GepIndex::Value(offset)],
                IntegerType::new(context, 8).into(),
            )?
        };

        let new_array_ptr = valid_block.append_op_result(scf::r#if(
            is_shared,
            &[ptr_ty],
            {
                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                let k0 = block.const_int(context, location, 0, 64)?;
                let k1 = block.const_int_from_type(context, location, 1, len_ty)?;

                let elem_stride =
                    block.const_int(context, location, elem_layout.pad_to_align().size(), 64)?;
                match metadata.get::<DupOverridesMeta>() {
                    Some(dup_overrides_meta) if dup_overrides_meta.is_overriden(elem_ty) => {
                        // TODO: If extract_len is 1 there is no need for the for loop.
                        block.append_operation(scf::r#for(
                            k0,
                            value_size,
                            elem_stride,
                            {
                                let region = Region::new();
                                let block = region.append_block(Block::new(&[(
                                    IntegerType::new(context, 64).into(),
                                    location,
                                )]));

                                let offset = block.argument(0)?.into();
                                let source_ptr = block.gep(
                                    context,
                                    location,
                                    data_ptr,
                                    &[GepIndex::Value(offset)],
                                    IntegerType::new(context, 8).into(),
                                )?;
                                let target_ptr = block.gep(
                                    context,
                                    location,
                                    value_ptr,
                                    &[GepIndex::Value(offset)],
                                    IntegerType::new(context, 8).into(),
                                )?;

                                let value = block.load(context, location, source_ptr, elem_type)?;
                                let values = dup_overrides_meta
                                    .invoke_override(context, &block, location, elem_ty, value)?;
                                block.store(context, location, source_ptr, values.0)?;
                                block.store(context, location, target_ptr, values.1)?;

                                block.append_operation(scf::r#yield(&[], location));
                                region
                            },
                            location,
                        ));
                    }
                    _ => block.memcpy(context, location, data_ptr, value_ptr, value_size),
                }

                let array_ptr = {
                    let array_len_bytes =
                        block.append_op_result(arith::subi(array_len, k1, location))?;
                    let array_len_bytes = block.append_op_result(arith::extui(
                        array_len_bytes,
                        IntegerType::new(context, 64).into(),
                        location,
                    ))?;
                    let array_len_bytes = block.append_op_result(arith::muli(
                        array_len_bytes,
                        elem_stride,
                        location,
                    ))?;
                    let array_len_bytes = block.append_op_result(arith::addi(
                        array_len_bytes,
                        block.const_int(
                            context,
                            location,
                            calc_refcount_offset(elem_layout),
                            64,
                        )?,
                        location,
                    ))?;

                    let clone_ptr = block.append_op_result(llvm::zero(ptr_ty, location))?;
                    let clone_ptr = block.append_op_result(ReallocBindingsMeta::realloc(
                        context,
                        clone_ptr,
                        array_len_bytes,
                        location,
                    )?)?;
                    block.store(context, location, clone_ptr, k1)?;

                    let clone_ptr = block.gep(
                        context,
                        location,
                        clone_ptr,
                        &[GepIndex::Const(calc_refcount_offset(elem_layout) as i32)],
                        IntegerType::new(context, 8).into(),
                    )?;

                    let data_ptr = {
                        let offset = if REVERSE {
                            array_start
                        } else {
                            block.addi(array_start, extract_len_value, location)?
                        };

                        let offset = block.append_op_result(arith::extui(
                            offset,
                            IntegerType::new(context, 64).into(),
                            location,
                        ))?;
                        let offset =
                            block.append_op_result(arith::muli(offset, elem_stride, location))?;

                        block.gep(
                            context,
                            location,
                            array_ptr,
                            &[GepIndex::Value(offset)],
                            IntegerType::new(context, 8).into(),
                        )?
                    };

                    let others_len = block.append_op_result(arith::subi(
                        array_len,
                        extract_len_value,
                        location,
                    ))?;
                    let others_len = block.append_op_result(arith::extui(
                        others_len,
                        IntegerType::new(context, 64).into(),
                        location,
                    ))?;
                    let others_size =
                        block.append_op_result(arith::muli(others_len, elem_stride, location))?;

                    match metadata.get::<DupOverridesMeta>() {
                        Some(dup_overrides_meta) => {
                            block.append_operation(scf::r#for(
                                k0,
                                others_size,
                                elem_stride,
                                {
                                    let region = Region::new();
                                    let block = region.append_block(Block::new(&[(
                                        IntegerType::new(context, 64).into(),
                                        location,
                                    )]));

                                    let offset = block.argument(0)?.into();
                                    let source_ptr = block.gep(
                                        context,
                                        location,
                                        data_ptr,
                                        &[GepIndex::Value(offset)],
                                        IntegerType::new(context, 8).into(),
                                    )?;
                                    let target_ptr = block.gep(
                                        context,
                                        location,
                                        clone_ptr,
                                        &[GepIndex::Value(offset)],
                                        IntegerType::new(context, 8).into(),
                                    )?;

                                    let value =
                                        block.load(context, location, source_ptr, elem_type)?;
                                    let values = dup_overrides_meta.invoke_override(
                                        context, &block, location, elem_ty, value,
                                    )?;
                                    block.store(context, location, source_ptr, values.0)?;
                                    block.store(context, location, target_ptr, values.1)?;

                                    block.append_operation(scf::r#yield(&[], location));
                                    region
                                },
                                location,
                            ));
                        }
                        _ => block.memcpy(context, location, data_ptr, clone_ptr, others_size),
                    }

                    metadata
                        .get::<DropOverridesMeta>()
                        .to_native_assert_error("drop overrides meta should exist")?
                        .invoke_override(context, &block, location, self_ty, array_value)?;

                    clone_ptr
                };

                block.append_operation(scf::r#yield(&[array_ptr], location));
                region
            },
            {
                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                block.memcpy(context, location, data_ptr, value_ptr, value_size);

                block.append_operation(scf::r#yield(&[array_ptr], location));
                region
            },
            location,
        ))?;

        let array_value =
            valid_block.insert_value(context, location, array_value, new_array_ptr, 0)?;

        let has_realloc = valid_block.append_op_result(
            ods::llvm::icmp(
                context,
                IntegerType::new(context, 1).into(),
                array_ptr,
                new_array_ptr,
                IntegerAttribute::new(IntegerType::new(context, 64).into(), 1).into(),
                location,
            )
            .into(),
        )?;
        let array_value = valid_block.append_op_result(scf::r#if(
            has_realloc,
            &[array_value.r#type()],
            {
                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                let k0 = block.const_int_from_type(context, location, 0, len_ty)?;
                let array_len =
                    block.append_op_result(arith::subi(array_len, extract_len_value, location))?;

                let array_value = block.insert_value(context, location, array_value, k0, 1)?;
                let array_value =
                    block.insert_value(context, location, array_value, array_len, 2)?;
                let array_value =
                    block.insert_value(context, location, array_value, array_len, 3)?;

                block.append_operation(scf::r#yield(&[array_value], location));
                region
            },
            {
                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                let array_value = if REVERSE {
                    let array_end = block.append_op_result(arith::subi(
                        array_end,
                        extract_len_value,
                        location,
                    ))?;
                    block.insert_value(context, location, array_value, array_end, 2)?
                } else {
                    let array_start = block.append_op_result(arith::addi(
                        array_start,
                        extract_len_value,
                        location,
                    ))?;
                    block.insert_value(context, location, array_value, array_start, 1)?
                };

                block.append_operation(scf::r#yield(&[array_value], location));
                region
            },
            location,
        ))?;

        branch_values.push(array_value);
        branch_values.push(value_ptr);

        valid_block.append_operation(helper.br(0, &branch_values, location));
    }

    {
        if CONSUME {
            let self_ty = match info {
                PopInfo::Single(info) => &info.signature.param_signatures[0].ty,
                PopInfo::Multi(info) => &info.signature.param_signatures[1].ty,
            };

            metadata
                .get::<DropOverridesMeta>()
                .to_native_assert_error("drop overrides meta should exist")?
                .invoke_override(context, error_block, location, self_ty, array_value)?;
        } else {
            branch_values.push(array_value);
        }

        error_block.append_operation(helper.br(1, &branch_values, location));
    }

    Ok(())
}

/// Generate MLIR operations for the `array_get` libfunc.
pub fn build_get<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let ptr_ty = llvm::r#type::pointer(context, 0);
    let len_ty = IntegerType::new(context, 32).into();

    // Build the type so that the drop impl invocation works properly.
    registry.build_type(
        context,
        helper,
        metadata,
        &info.signature.param_signatures[1].ty,
    )?;

    let range_check =
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;

    let array_start =
        entry.extract_value(context, location, entry.argument(1)?.into(), len_ty, 1)?;
    let array_end = entry.extract_value(context, location, entry.argument(1)?.into(), len_ty, 2)?;

    let array_len = entry.append_op_result(arith::subi(array_end, array_start, location))?;
    let is_valid = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Ult,
        entry.argument(2)?.into(),
        array_len,
        location,
    ))?;

    let valid_block = helper.append_block(Block::new(&[]));
    let error_block = helper.append_block(Block::new(&[]));
    entry.append_operation(cf::cond_br(
        context,
        is_valid,
        valid_block,
        error_block,
        &[],
        &[],
        location,
    ));

    {
        let (elem_ty, elem_layout) =
            registry.build_type_with_layout(context, helper, metadata, &info.ty)?;
        let elem_stride =
            valid_block.const_int(context, location, elem_layout.pad_to_align().size(), 64)?;

        let value_ptr = valid_block.append_op_result(llvm::zero(ptr_ty, location))?;
        let value_ptr = valid_block.append_op_result(ReallocBindingsMeta::realloc(
            context,
            value_ptr,
            elem_stride,
            location,
        )?)?;

        let array_ptr =
            valid_block.extract_value(context, location, entry.argument(1)?.into(), ptr_ty, 0)?;
        let is_shared = is_shared(context, valid_block, location, array_ptr, elem_layout)?;

        let offset = valid_block.append_op_result(arith::addi(
            array_start,
            entry.argument(2)?.into(),
            location,
        ))?;
        let offset = valid_block.append_op_result(arith::extui(
            offset,
            IntegerType::new(context, 64).into(),
            location,
        ))?;
        let offset = valid_block.append_op_result(arith::muli(offset, elem_stride, location))?;

        let source_ptr = valid_block.gep(
            context,
            location,
            array_ptr,
            &[GepIndex::Value(offset)],
            IntegerType::new(context, 8).into(),
        )?;

        valid_block.append_operation(scf::r#if(
            is_shared,
            &[],
            {
                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                match metadata.get::<DupOverridesMeta>() {
                    Some(dup_overrides_meta) if dup_overrides_meta.is_overriden(&info.ty) => {
                        let value = block.load(context, location, source_ptr, elem_ty)?;
                        let values = dup_overrides_meta
                            .invoke_override(context, &block, location, &info.ty, value)?;
                        block.store(context, location, source_ptr, values.0)?;
                        block.store(context, location, value_ptr, values.1)?;
                    }
                    _ => block.memcpy(context, location, source_ptr, value_ptr, elem_stride),
                }

                metadata
                    .get::<DropOverridesMeta>()
                    .to_native_assert_error("drop overrides meta should exist")?
                    .invoke_override(
                        context,
                        &block,
                        location,
                        &info.signature.param_signatures[1].ty,
                        entry.argument(1)?.into(),
                    )?;

                block.append_operation(scf::r#yield(&[], location));
                region
            },
            {
                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                block.memcpy(context, location, source_ptr, value_ptr, elem_stride);

                match metadata.get::<DropOverridesMeta>() {
                    Some(drop_overrides_meta) if drop_overrides_meta.is_overriden(&info.ty) => {
                        let drop_loop = |o0, o1| {
                            block.append_operation(scf::r#for(
                                o0,
                                o1,
                                elem_stride,
                                {
                                    let region = Region::new();
                                    let block = region.append_block(Block::new(&[(
                                        IntegerType::new(context, 64).into(),
                                        location,
                                    )]));

                                    let value_ptr = block.gep(
                                        context,
                                        location,
                                        array_ptr,
                                        &[GepIndex::Value(block.argument(0)?.into())],
                                        IntegerType::new(context, 8).into(),
                                    )?;
                                    let value =
                                        block.load(context, location, value_ptr, elem_ty)?;
                                    drop_overrides_meta.invoke_override(
                                        context, &block, location, &info.ty, value,
                                    )?;

                                    block.append_operation(scf::r#yield(&[], location));
                                    region
                                },
                                location,
                            ));

                            Result::Ok(())
                        };

                        let o0 = block.append_op_result(arith::extui(
                            array_start,
                            IntegerType::new(context, 64).into(),
                            location,
                        ))?;
                        let o1 = block.append_op_result(arith::addi(
                            array_start,
                            entry.argument(2)?.into(),
                            location,
                        ))?;
                        let o1 = block.append_op_result(arith::extui(
                            o1,
                            IntegerType::new(context, 64).into(),
                            location,
                        ))?;
                        let o0 = block.append_op_result(arith::muli(o0, elem_stride, location))?;
                        let o1 = block.append_op_result(arith::muli(o1, elem_stride, location))?;
                        drop_loop(o0, o1)?;

                        let o0 = block.append_op_result(arith::addi(o1, elem_stride, location))?;
                        let o1 = block.append_op_result(arith::extui(
                            array_end,
                            IntegerType::new(context, 64).into(),
                            location,
                        ))?;
                        let o1 = block.append_op_result(arith::muli(o1, elem_stride, location))?;
                        drop_loop(o0, o1)?;
                    }
                    _ => {}
                }

                let array_ptr = block.gep(
                    context,
                    location,
                    array_ptr,
                    &[GepIndex::Const(-(calc_refcount_offset(elem_layout) as i32))],
                    IntegerType::new(context, 8).into(),
                )?;
                block.append_operation(ReallocBindingsMeta::free(context, array_ptr, location)?);

                block.append_operation(scf::r#yield(&[], location));
                region
            },
            location,
        ));

        valid_block.append_operation(helper.br(0, &[range_check, value_ptr], location));
    }

    {
        metadata
            .get::<DropOverridesMeta>()
            .to_native_assert_error("drop overrides meta should exist")?
            .invoke_override(
                context,
                error_block,
                location,
                &info.signature.param_signatures[1].ty,
                entry.argument(1)?.into(),
            )?;

        error_block.append_operation(helper.br(1, &[range_check], location));
    }

    Ok(())
}

/// Generate MLIR operations for the `array_slice` libfunc.
pub fn build_slice<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    // Signature:
    //   Params: RangeCheck, Snapshot<Array<felt252>>, u32, u32
    //   Branches:
    //     0: RangeCheck, Snapshot<Array<felt252>>
    //     1: RangeCheck

    let ptr_ty = llvm::r#type::pointer(context, 0);
    let len_ty = IntegerType::new(context, 32).into();

    let self_ty = registry.build_type(
        context,
        helper,
        metadata,
        &info.signature.param_signatures[1].ty,
    )?;

    let range_check =
        super::increment_builtin_counter(context, entry, location, entry.argument(0)?.into())?;

    let k0 = entry.const_int_from_type(context, location, 0, len_ty)?;
    let k1 = entry.const_int_from_type(context, location, 1, len_ty)?;

    let array_start =
        entry.extract_value(context, location, entry.argument(1)?.into(), len_ty, 1)?;
    let array_end = entry.extract_value(context, location, entry.argument(1)?.into(), len_ty, 2)?;
    let array_len = entry.append_op_result(arith::subi(array_end, array_start, location))?;

    let slice_start = entry.argument(2)?.into();
    let slice_len = entry.argument(3)?.into();
    let slice_end = entry.append_op_result(arith::addi(slice_start, slice_len, location))?;

    let slice_lhs_bound = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Ule,
        slice_start,
        array_len,
        location,
    ))?;
    let slice_rhs_bound = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Ule,
        slice_end,
        array_len,
        location,
    ))?;
    let slice_bounds =
        entry.append_op_result(arith::andi(slice_lhs_bound, slice_rhs_bound, location))?;

    let valid_block = helper.append_block(Block::new(&[]));
    let error_block = helper.append_block(Block::new(&[]));
    entry.append_operation(cf::cond_br(
        context,
        slice_bounds,
        valid_block,
        error_block,
        &[],
        &[],
        location,
    ));

    {
        let (elem_ty, elem_layout) =
            registry.build_type_with_layout(context, helper, metadata, &info.ty)?;
        let elem_stride =
            valid_block.const_int(context, location, elem_layout.pad_to_align().size(), 64)?;

        let slice_size = valid_block.append_op_result(arith::extui(
            slice_len,
            IntegerType::new(context, 64).into(),
            location,
        ))?;
        let slice_size =
            valid_block.append_op_result(arith::muli(elem_stride, slice_size, location))?;
        let slice_size_with_offset = valid_block.append_op_result(arith::addi(
            slice_size,
            valid_block.const_int(context, location, calc_refcount_offset(elem_layout), 64)?,
            location,
        ))?;

        let array_ptr =
            valid_block.extract_value(context, location, entry.argument(1)?.into(), ptr_ty, 0)?;
        let null_ptr = valid_block.append_op_result(llvm::zero(ptr_ty, location))?;
        let is_null_source = valid_block.append_op_result(
            ods::llvm::icmp(
                context,
                IntegerType::new(context, 1).into(),
                array_ptr,
                null_ptr,
                IntegerAttribute::new(IntegerType::new(context, 64).into(), 0).into(),
                location,
            )
            .into(),
        )?;
        let slice_ptr = valid_block.append_op_result(scf::r#if(
            is_null_source,
            &[ptr_ty],
            {
                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                block.append_operation(scf::r#yield(&[null_ptr], location));
                region
            },
            {
                let region = Region::new();
                let block = region.append_block(Block::new(&[]));

                let slice_ptr = block.append_op_result(llvm::zero(ptr_ty, location))?;
                let slice_ptr = block.append_op_result(ReallocBindingsMeta::realloc(
                    context,
                    slice_ptr,
                    slice_size_with_offset,
                    location,
                )?)?;
                block.store(context, location, slice_ptr, k1)?;

                let slice_ptr = block.gep(
                    context,
                    location,
                    slice_ptr,
                    &[GepIndex::Const(calc_refcount_offset(elem_layout) as i32)],
                    IntegerType::new(context, 8).into(),
                )?;

                let is_shared = is_shared(context, &block, location, array_ptr, elem_layout)?;

                let offset =
                    block.append_op_result(arith::addi(array_start, slice_start, location))?;
                let offset = block.append_op_result(arith::extui(
                    offset,
                    IntegerType::new(context, 64).into(),
                    location,
                ))?;
                let offset = block.append_op_result(arith::muli(offset, elem_stride, location))?;

                let source_ptr = block.gep(
                    context,
                    location,
                    array_ptr,
                    &[GepIndex::Value(offset)],
                    IntegerType::new(context, 8).into(),
                )?;

                block.append_operation(scf::r#if(
                    is_shared,
                    &[],
                    {
                        let region = Region::new();
                        let block = region.append_block(Block::new(&[]));

                        match metadata.get::<DupOverridesMeta>() {
                            Some(dup_overrides_meta)
                                if dup_overrides_meta.is_overriden(&info.ty) =>
                            {
                                let k0 = block.const_int(context, location, 0, 64)?;
                                block.append_operation(scf::r#for(
                                    k0,
                                    slice_size,
                                    elem_stride,
                                    {
                                        let region = Region::new();
                                        let block = region.append_block(Block::new(&[(
                                            IntegerType::new(context, 64).into(),
                                            location,
                                        )]));

                                        let offset = block.argument(0)?.into();
                                        let source_ptr = block.gep(
                                            context,
                                            location,
                                            source_ptr,
                                            &[GepIndex::Value(offset)],
                                            IntegerType::new(context, 8).into(),
                                        )?;
                                        let target_ptr = block.gep(
                                            context,
                                            location,
                                            slice_ptr,
                                            &[GepIndex::Value(offset)],
                                            IntegerType::new(context, 8).into(),
                                        )?;

                                        let value =
                                            block.load(context, location, source_ptr, elem_ty)?;
                                        let values = dup_overrides_meta.invoke_override(
                                            context, &block, location, &info.ty, value,
                                        )?;
                                        block.store(context, location, source_ptr, values.0)?;
                                        block.store(context, location, target_ptr, values.1)?;

                                        block.append_operation(scf::r#yield(&[], location));
                                        region
                                    },
                                    location,
                                ));
                            }
                            _ => block.memcpy(context, location, source_ptr, slice_ptr, slice_size),
                        }

                        metadata
                            .get::<DropOverridesMeta>()
                            .to_native_assert_error("drop overrides meta should exist")?
                            .invoke_override(
                                context,
                                &block,
                                location,
                                &info.signature.param_signatures[1].ty,
                                entry.argument(1)?.into(),
                            )?;

                        block.append_operation(scf::r#yield(&[], location));
                        region
                    },
                    {
                        let region = Region::new();
                        let block = region.append_block(Block::new(&[]));

                        block.memcpy(context, location, source_ptr, slice_ptr, slice_size);

                        match metadata.get::<DropOverridesMeta>() {
                            Some(drop_overrides_meta)
                                if drop_overrides_meta.is_overriden(&info.ty) =>
                            {
                                let drop_loop = |o0, o1| {
                                    block.append_operation(scf::r#for(
                                        o0,
                                        o1,
                                        elem_stride,
                                        {
                                            let region = Region::new();
                                            let block = region.append_block(Block::new(&[(
                                                IntegerType::new(context, 64).into(),
                                                location,
                                            )]));

                                            let value_ptr = block.gep(
                                                context,
                                                location,
                                                array_ptr,
                                                &[GepIndex::Value(block.argument(0)?.into())],
                                                IntegerType::new(context, 8).into(),
                                            )?;
                                            let value = block
                                                .load(context, location, value_ptr, elem_ty)?;
                                            drop_overrides_meta.invoke_override(
                                                context, &block, location, &info.ty, value,
                                            )?;

                                            block.append_operation(scf::r#yield(&[], location));
                                            region
                                        },
                                        location,
                                    ));

                                    Result::Ok(())
                                };

                                let o0 = block.append_op_result(arith::extui(
                                    array_start,
                                    IntegerType::new(context, 64).into(),
                                    location,
                                ))?;
                                let o1 = block.append_op_result(arith::addi(
                                    array_start,
                                    slice_start,
                                    location,
                                ))?;
                                let o1 = block.append_op_result(arith::extui(
                                    o1,
                                    IntegerType::new(context, 64).into(),
                                    location,
                                ))?;
                                let o0 = block.append_op_result(arith::muli(
                                    o0,
                                    elem_stride,
                                    location,
                                ))?;
                                let o1 = block.append_op_result(arith::muli(
                                    o1,
                                    elem_stride,
                                    location,
                                ))?;
                                drop_loop(o0, o1)?;

                                let o0 = block
                                    .append_op_result(arith::addi(o1, slice_size, location))?;
                                let o1 = block.append_op_result(arith::extui(
                                    array_end,
                                    IntegerType::new(context, 64).into(),
                                    location,
                                ))?;
                                let o1 = block.append_op_result(arith::muli(
                                    o1,
                                    elem_stride,
                                    location,
                                ))?;
                                drop_loop(o0, o1)?;
                            }
                            _ => {}
                        }

                        let array_ptr = block.gep(
                            context,
                            location,
                            array_ptr,
                            &[GepIndex::Const(-(calc_refcount_offset(elem_layout) as i32))],
                            IntegerType::new(context, 8).into(),
                        )?;
                        block.append_operation(ReallocBindingsMeta::free(
                            context, array_ptr, location,
                        )?);

                        block.append_operation(scf::r#yield(&[], location));
                        region
                    },
                    location,
                ));

                block.append_operation(scf::r#yield(&[slice_ptr], location));
                region
            },
            location,
        ))?;

        let slice_value = valid_block.append_op_result(llvm::undef(self_ty, location))?;
        let slice_value = valid_block.insert_values(
            context,
            location,
            slice_value,
            &[slice_ptr, k0, slice_len, slice_len],
        )?;

        valid_block.append_operation(helper.br(0, &[range_check, slice_value], location));
    }

    {
        metadata
            .get::<DropOverridesMeta>()
            .ok_or(Error::MissingMetadata)?
            .invoke_override(
                context,
                error_block,
                location,
                &info.signature.param_signatures[1].ty,
                entry.argument(1)?.into(),
            )?;

        error_block.append_operation(helper.br(1, &[range_check], location));
    }

    Ok(())
}

/// Generate MLIR operations for the `array_len` libfunc.
pub fn build_len<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let len_ty = IntegerType::new(context, 32).into();

    let array_start =
        entry.extract_value(context, location, entry.argument(0)?.into(), len_ty, 1)?;
    let array_end = entry.extract_value(context, location, entry.argument(0)?.into(), len_ty, 2)?;

    let array_len = entry.append_op_result(arith::subi(array_end, array_start, location))?;

    metadata
        .get::<DropOverridesMeta>()
        .to_native_assert_error("drop overrides meta should exist")?
        .invoke_override(
            context,
            entry,
            location,
            &info.signature.param_signatures[0].ty,
            entry.argument(0)?.into(),
        )?;

    entry.append_operation(helper.br(0, &[array_len], location));
    Ok(())
}

fn calc_refcount_offset(layout: Layout) -> usize {
    get_integer_layout(32)
        .align_to(layout.align())
        .unwrap()
        .pad_to_align()
        .size()
}

fn is_shared<'ctx, 'this>(
    context: &'ctx Context,
    block: &'this Block<'ctx>,
    location: Location<'ctx>,
    array_ptr: Value<'ctx, 'this>,
    elem_layout: Layout,
) -> Result<Value<'ctx, 'this>> {
    let null_ptr =
        block.append_op_result(llvm::zero(llvm::r#type::pointer(context, 0), location))?;
    let ptr_is_null = block.append_op_result(
        ods::llvm::icmp(
            context,
            IntegerType::new(context, 1).into(),
            array_ptr,
            null_ptr,
            IntegerAttribute::new(IntegerType::new(context, 64).into(), 0).into(),
            location,
        )
        .into(),
    )?;

    let is_shared = block.append_op_result(scf::r#if(
        ptr_is_null,
        &[IntegerType::new(context, 1).into()],
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let k0 = block.const_int(context, location, 0, 1)?;

            block.append_operation(scf::r#yield(&[k0], location));
            region
        },
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let array_ptr = block.gep(
                context,
                location,
                array_ptr,
                &[GepIndex::Const(-(calc_refcount_offset(elem_layout) as i32))],
                IntegerType::new(context, 8).into(),
            )?;
            let ref_count = block.load(
                context,
                location,
                array_ptr,
                IntegerType::new(context, 32).into(),
            )?;

            #[cfg(debug_assertions)]
            {
                let k0 = block.const_int(context, location, 0, 32)?;
                let is_nonzero = block.append_op_result(arith::cmpi(
                    context,
                    CmpiPredicate::Ne,
                    ref_count,
                    k0,
                    location,
                ))?;

                block.append_operation(cf::assert(
                    context,
                    is_nonzero,
                    "ref_count must not be zero",
                    location,
                ));
            }

            let k1 = block.const_int(context, location, 1, 32)?;
            let is_shared = block.append_op_result(arith::cmpi(
                context,
                CmpiPredicate::Ne,
                ref_count,
                k1,
                location,
            ))?;

            block.append_operation(scf::r#yield(&[is_shared], location));
            region
        },
        location,
    ))?;

    Ok(is_shared)
}

#[cfg(test)]
mod test {
    use crate::{
        utils::{
            felt252_str,
            test::{jit_enum, jit_panic, jit_struct, load_cairo, run_program},
        },
        values::Value,
    };
    use pretty_assertions_sorted::assert_eq;
    use starknet_types_core::felt::Felt;

    #[test]
    fn run_roundtrip() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test(x: Array<u32>) -> Array<u32> {
                x
            }
        );
        let result = run_program(&program, "run_test", &[[1u32, 2u32].into()]).return_value;

        assert_eq!(result, Value::from([1u32, 2u32]));
    }

    #[test]
    fn run_append() {
        let program = load_cairo! {
            use array::ArrayTrait;

            fn run_test() -> Array<u32> {
                let mut numbers = ArrayTrait::new();
                numbers.append(4_u32);
                numbers
            }
        };
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(result, [4u32].into());
    }

    #[test]
    fn run_len() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> u32 {
                let mut numbers = ArrayTrait::new();
                numbers.append(4_u32);
                numbers.append(3_u32);
                numbers.append(2_u32);
                numbers.len()
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(result, 3u32.into());
    }

    #[test]
    fn run_get() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> (u32, u32, u32, u32) {
                let mut numbers = ArrayTrait::new();
                numbers.append(4_u32);
                numbers.append(3_u32);
                numbers.append(2_u32);
                numbers.append(1_u32);
                (
                    *numbers.at(0),
                    *numbers.at(1),
                    *numbers.at(2),
                    *numbers.at(3),
                )
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            jit_enum!(
                0,
                jit_struct!(jit_struct!(
                    4u32.into(),
                    3u32.into(),
                    2u32.into(),
                    1u32.into()
                ))
            )
        );
    }

    #[test]
    fn run_get_big() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> (u32, u32, u32, u32) {
                let mut numbers = ArrayTrait::new();
                numbers.append(4_u32);
                numbers.append(3_u32);
                numbers.append(2_u32);
                numbers.append(2_u32);
                numbers.append(2_u32);
                numbers.append(2_u32);
                numbers.append(2_u32);
                numbers.append(2_u32);
                numbers.append(2_u32);
                numbers.append(2_u32);
                numbers.append(2_u32);
                numbers.append(2_u32);
                numbers.append(2_u32);
                numbers.append(2_u32);
                numbers.append(2_u32);
                numbers.append(2_u32);
                numbers.append(17_u32);
                numbers.append(17_u32);
                numbers.append(18_u32);
                numbers.append(19_u32);
                numbers.append(20_u32);
                numbers.append(21_u32);
                numbers.append(22_u32);
                numbers.append(23_u32);
                (
                    *numbers.at(20),
                    *numbers.at(21),
                    *numbers.at(22),
                    *numbers.at(23),
                )
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            jit_enum!(
                0,
                jit_struct!(jit_struct!(
                    20u32.into(),
                    21u32.into(),
                    22u32.into(),
                    23u32.into()
                ))
            )
        );
    }

    #[test]
    fn run_pop_front() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> u32 {
                let mut numbers = ArrayTrait::new();
                numbers.append(4_u32);
                numbers.append(3_u32);
                let _ = numbers.pop_front();
                numbers.append(1_u32);
                *numbers.at(0)
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(result, jit_enum!(0, jit_struct!(3u32.into())));
    }

    #[test]
    fn run_pop_front_result() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> Option<u32> {
                let mut numbers = ArrayTrait::new();
                numbers.append(4_u32);
                numbers.append(3_u32);
                numbers.pop_front()
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(result, jit_enum!(0, 4u32.into()));

        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> Option<u32> {
                let mut numbers = ArrayTrait::new();
                numbers.pop_front()
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(result, jit_enum!(1, jit_struct!()));
    }

    #[test]
    fn run_pop_front_consume() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> u32 {
                let mut numbers = ArrayTrait::new();
                numbers.append(4_u32);
                numbers.append(3_u32);
                match numbers.pop_front_consume() {
                    Option::Some((_, x)) => x,
                    Option::None(()) => 0_u32,
                }
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(result, 4u32.into());
    }

    #[test]
    fn run_pop_back() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> (Option<@u32>, Option<@u32>, Option<@u32>, Option<@u32>) {
                let mut numbers = ArrayTrait::new();
                numbers.append(4_u32);
                numbers.append(3_u32);
                numbers.append(1_u32);
                let mut numbers = numbers.span();
                (
                    numbers.pop_back(),
                    numbers.pop_back(),
                    numbers.pop_back(),
                    numbers.pop_back(),
                )
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            jit_struct!(
                jit_enum!(0, 1u32.into()),
                jit_enum!(0, 3u32.into()),
                jit_enum!(0, 4u32.into()),
                jit_enum!(
                    1,
                    Value::Struct {
                        fields: Vec::new(),
                        debug_name: None,
                    }
                ),
            ),
        );
    }

    #[test]
    fn run_slice() {
        let program = load_cairo!(
            use array::Array;
            use array::ArrayTrait;
            use array::SpanTrait;
            use option::OptionTrait;
            use box::BoxTrait;

            fn run_test() -> u32 {
                let mut data: Array<u32> = ArrayTrait::new(); // Alloca (freed).
                data.append(1_u32);
                data.append(2_u32);
                data.append(3_u32);
                data.append(4_u32);
                let sp = data.span(); // Alloca (leaked).
                let slice = sp.slice(1, 2);
                data.append(5_u32);
                data.append(5_u32);
                data.append(5_u32);
                data.append(5_u32);
                data.append(5_u32); // Realloc (freed).
                data.append(5_u32);
                *slice.get(1).unwrap().unbox()
            }

        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(result, jit_enum!(0, jit_struct!(3u32.into())));
    }

    #[test]
    fn run_slice_fail() {
        let program = load_cairo!(
            use array::Array;
            use array::ArrayTrait;
            use array::SpanTrait;
            use option::OptionTrait;
            use box::BoxTrait;

            fn run_test() -> u32 {
                let mut data: Array<u32> = ArrayTrait::new();
                data.append(1_u32);
                data.append(2_u32);
                data.append(3_u32);
                data.append(4_u32);
                let sp = data.span();
                let slice = sp.slice(1, 4); // oob
                //data.append(5_u32);
                *slice.get(0).unwrap().unbox()
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            jit_panic!(felt252_str(
                "1637570914057682275393755530660268060279989363"
            ))
        );
    }

    #[test]
    fn run_slice_empty_array() {
        let program = load_cairo!(
            fn run_test() -> Span<felt252> {
                let x: Span<felt252> = array![].span();
                x.slice(0, 0)
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            Value::Enum {
                tag: 0,
                value: Box::new(Value::Struct {
                    fields: vec![Value::Struct {
                        fields: vec![Value::Array(vec![])],
                        debug_name: None,
                    }],
                    debug_name: None,
                }),
                debug_name: None
            },
        );
    }

    #[test]
    fn run_span_from_tuple() {
        let program = load_cairo!(
            mod felt252_span_from_tuple {
                pub extern fn span_from_tuple<T>(struct_like: Box<@T>) -> @Array<felt252> nopanic;
            }

            fn run_test() -> Array<felt252> {
                let span = felt252_span_from_tuple::span_from_tuple(BoxTrait::new(@(10, 20, 30)));
                span.clone()
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            jit_enum!(
                0,
                jit_struct!(Value::from([
                    Value::Felt252(Felt::from(10)),
                    Felt::from(20).into(),
                    Felt::from(30).into()
                ]))
            )
        );
    }

    #[test]
    fn run_span_from_multi_tuple() {
        let program = load_cairo!(
            mod tuple_span_from_tuple {
                pub extern fn span_from_tuple<T>(
                    struct_like: Box<@T>
                ) -> @Array<(felt252, felt252, felt252)> nopanic;
            }

            fn run_test() {
                let multi_tuple = ((10, 20, 30), (40, 50, 60), (70, 80, 90));
                let span = tuple_span_from_tuple::span_from_tuple(BoxTrait::new(@multi_tuple));
                assert!(*span[0] == (10, 20, 30));
                assert!(*span[1] == (40, 50, 60));
                assert!(*span[2] == (70, 80, 90));
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(result, jit_enum!(0, jit_struct!(jit_struct!())));
    }

    #[test]
    fn seq_append1() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> Array<u32> {
                let mut data = ArrayTrait::new();
                data.append(1);
                data
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            Value::from([1u32]),
        );
    }

    #[test]
    fn seq_append2() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> Array<u32> {
                let mut data = ArrayTrait::new();
                data.append(1);
                data.append(2);
                data
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            Value::from([1u32, 2u32]),
        );
    }

    #[test]
    fn seq_append2_popf1() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> Array<u32> {
                let mut data = ArrayTrait::new();
                data.append(1);
                data.append(2);
                let _ = data.pop_front();
                data
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            Value::from([2u32]),
        );
    }

    #[test]
    fn seq_append2_popb1() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> Span<u32> {
                let mut data = ArrayTrait::new();
                data.append(1);
                data.append(2);
                let mut data = data.span();
                let _ = data.pop_back();
                data
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            jit_struct!([1u32].into())
        );
    }

    #[test]
    fn seq_append1_popf1_append1() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> Array<u32> {
                let mut data = ArrayTrait::new();
                data.append(1);
                let _ = data.pop_front();
                data.append(2);
                data
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            Value::from([2u32]),
        );
    }

    #[test]
    fn seq_append1_first() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> u32 {
                let mut data = ArrayTrait::new();
                data.append(1);
                *data.at(0)
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            Value::Enum {
                tag: 0,
                value: Box::new(Value::Struct {
                    fields: vec![Value::from(1u32)],
                    debug_name: None,
                }),
                debug_name: None,
            },
        );
    }

    #[test]
    fn seq_append2_first() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> u32 {
                let mut data = ArrayTrait::new();
                data.append(1);
                data.append(2);
                *data.at(0)
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            Value::Enum {
                tag: 0,
                value: Box::new(Value::Struct {
                    fields: vec![Value::from(1u32)],
                    debug_name: None,
                }),
                debug_name: None,
            },
        );
    }

    #[test]
    fn seq_append2_popf1_first() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> u32 {
                let mut data = ArrayTrait::new();
                data.append(1);
                data.append(2);
                let _ = data.pop_front();
                *data.at(0)
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            Value::Enum {
                tag: 0,
                value: Box::new(Value::Struct {
                    fields: vec![Value::from(2u32)],
                    debug_name: None,
                }),
                debug_name: None,
            },
        );
    }

    #[test]
    fn seq_append2_popb1_last() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> u32 {
                let mut data = ArrayTrait::new();
                data.append(1);
                data.append(2);
                let mut data_span = data.span();
                let _ = data_span.pop_back();
                let last = data_span.len() - 1;
                *data_span.at(last)
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            Value::Enum {
                tag: 0,
                value: Box::new(Value::Struct {
                    fields: vec![Value::from(1u32)],
                    debug_name: None,
                }),
                debug_name: None,
            }
        );
    }

    #[test]
    fn seq_append1_popf1_append1_first() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> u32 {
                let mut data = ArrayTrait::new();
                data.append(1);
                let _ = data.pop_front();
                data.append(2);
                *data.at(0)
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            Value::Enum {
                tag: 0,
                value: Box::new(Value::Struct {
                    fields: vec![Value::from(2u32)],
                    debug_name: None,
                }),
                debug_name: None,
            },
        );
    }

    #[test]
    fn array_clone() {
        let program = load_cairo!(
            fn run_test() -> Array<u32> {
                let x = ArrayTrait::new();
                x.clone()
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            Value::Enum {
                tag: 0,
                value: Box::new(Value::Struct {
                    fields: vec![Value::Array(vec![])],
                    debug_name: None,
                }),
                debug_name: None,
            },
        );
    }

    #[test]
    fn array_pop_back_state() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> Span<u32> {
                let mut numbers = ArrayTrait::new();
                numbers.append(1_u32);
                numbers.append(2_u32);
                numbers.append(3_u32);
                let mut numbers = numbers.span();
                let _ = numbers.pop_back();
                numbers
            }
        );

        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(result, jit_struct!([1u32, 2u32].into()));
    }

    #[test]
    fn array_empty_span() {
        // Tests snapshot_take on a empty array.
        let program = load_cairo!(
            fn run_test() -> Span<u32> {
                let x = ArrayTrait::new();
                x.span()
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            jit_struct!(Value::Array(vec![])),
        );
    }

    #[test]
    fn array_span_modify_span() {
        // Tests pop_back on a span.
        let program = load_cairo!(
            use core::array::SpanTrait;
            fn pop_elem(mut self: Span<u64>) -> Option<@u64> {
                let x = self.pop_back();
                x
            }

            fn run_test() -> Option<@u64> {
                let mut data = array![2].span();
                let x = pop_elem(data);
                x
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            jit_enum!(0, 2u64.into()),
        );
    }

    #[test]
    fn array_span_check_array() {
        // Tests pop back on a span not modifying the original array.
        let program = load_cairo!(
            use core::array::SpanTrait;
            fn pop_elem(mut self: Span<u64>) -> Option<@u64> {
                let x = self.pop_back();
                x
            }

            fn run_test() -> Array<u64> {
                let mut data = array![1, 2];
                let _x = pop_elem(data.span());
                data
            }
        );

        assert_eq!(
            run_program(&program, "run_test", &[]).return_value,
            Value::Array(vec![1u64.into(), 2u64.into()]),
        );
    }

    #[test]
    fn tuple_from_span() {
        let program = load_cairo! {
            use core::array::{tuple_from_span, FixedSizedArrayInfoImpl};

            fn run_test(x: Array<felt252>) -> [felt252; 3] {
                (*tuple_from_span::<[felt252; 3], FixedSizedArrayInfoImpl<felt252, 3>>(@x).unwrap()).unbox()
            }
        };

        assert_eq!(
            run_program(
                &program,
                "run_test",
                &[Value::Array(vec![
                    Value::Felt252(1.into()),
                    Value::Felt252(2.into()),
                    Value::Felt252(3.into()),
                ])],
            )
            .return_value,
            Value::Enum {
                tag: 0,
                value: Box::new(Value::Struct {
                    fields: vec![Value::Struct {
                        fields: vec![
                            Value::Felt252(1.into()),
                            Value::Felt252(2.into()),
                            Value::Felt252(3.into()),
                        ],
                        debug_name: None
                    }],
                    debug_name: None
                }),
                debug_name: None
            }
        );
    }

    #[test]
    fn tuple_from_span_failed() {
        let program = load_cairo! {
            use core::array::{tuple_from_span, FixedSizedArrayInfoImpl};

            fn run_test(x: Array<felt252>) -> Option<@Box<[core::felt252; 3]>> {
                tuple_from_span::<[felt252; 3], FixedSizedArrayInfoImpl<felt252, 3>>(@x)
            }
        };

        assert_eq!(
            run_program(
                &program,
                "run_test",
                &[Value::Array(vec![
                    Value::Felt252(1.into()),
                    Value::Felt252(2.into()),
                ])],
            )
            .return_value,
            jit_enum!(1, jit_struct!())
        );
    }

    #[test]
    fn snapshot_multi_pop_front() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> (Span<felt252>, @Box<[felt252; 3]>) {
                let mut numbers = array![1, 2, 3, 4, 5, 6].span();
                let popped = numbers.multi_pop_front::<3>().unwrap();

                (numbers, popped)
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            // Panic result
            jit_enum!(
                0,
                jit_struct!(
                    // Tuple
                    jit_struct!(
                        // Span of original array
                        jit_struct!(Value::Array(vec![
                            Value::Felt252(4.into()),
                            Value::Felt252(5.into()),
                            Value::Felt252(6.into()),
                        ])),
                        // Box of fixed array
                        jit_struct!(
                            Value::Felt252(1.into()),
                            Value::Felt252(2.into()),
                            Value::Felt252(3.into())
                        ),
                    )
                )
            )
        );
    }

    #[test]
    fn snapshot_failed_multi_pop_front() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> Span<felt252> {
                let mut numbers = array![1, 2].span();

                // should fail (return none)
                assert!(numbers.multi_pop_front::<3>().is_none());

                numbers
            }
        );

        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            // Panic result
            jit_enum!(
                0,
                jit_struct!(
                    // Span of original array
                    jit_struct!(Value::Array(vec![
                        Value::Felt252(1.into()),
                        Value::Felt252(2.into()),
                    ]),)
                )
            )
        );
    }

    #[test]
    fn snapshot_multi_pop_back() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> (Span<felt252>, @Box<[felt252; 3]>) {
                let mut numbers = array![1, 2, 3, 4, 5, 6].span();
                let popped = numbers.multi_pop_back::<3>().unwrap();

                (numbers, popped)
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            // Panic result
            jit_enum!(
                0,
                jit_struct!(
                    // Tuple
                    jit_struct!(
                        // Span of original array
                        jit_struct!(Value::Array(vec![
                            Value::Felt252(1.into()),
                            Value::Felt252(2.into()),
                            Value::Felt252(3.into()),
                        ])),
                        // Box of fixed array
                        jit_struct!(
                            Value::Felt252(4.into()),
                            Value::Felt252(5.into()),
                            Value::Felt252(6.into())
                        ),
                    )
                )
            )
        );
    }

    #[test]
    fn snapshot_failed_multi_pop_back() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> Span<felt252> {
                let mut numbers = array![1, 2].span();

                // should fail (return none)
                assert!(numbers.multi_pop_back::<3>().is_none());

                numbers
            }
        );

        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            // Panic result
            jit_enum!(
                0,
                jit_struct!(
                    // Span of original array
                    jit_struct!(Value::Array(vec![
                        Value::Felt252(1.into()),
                        Value::Felt252(2.into()),
                    ]),)
                )
            )
        );
    }

    #[test]
    fn snapshot_multi_pop_back_front() {
        let program = load_cairo!(
            use array::ArrayTrait;

            fn run_test() -> (Span<felt252>, @Box<[felt252; 2]>, @Box<[felt252; 2]>) {
                let mut numbers = array![1, 2, 3, 4, 5, 6].span();
                let popped_front = numbers.multi_pop_front::<2>().unwrap();
                let popped_back = numbers.multi_pop_back::<2>().unwrap();

                (numbers, popped_front, popped_back)
            }
        );
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            // Panic result
            jit_enum!(
                0,
                jit_struct!(
                    // Tuple
                    jit_struct!(
                        // Span of original array
                        jit_struct!(Value::Array(vec![
                            Value::Felt252(3.into()),
                            Value::Felt252(4.into()),
                        ])),
                        // Box of fixed array
                        jit_struct!(Value::Felt252(1.into()), Value::Felt252(2.into()),),
                        // Box of fixed array
                        jit_struct!(Value::Felt252(5.into()), Value::Felt252(6.into())),
                    )
                )
            )
        );
    }

    /// Test to ensure that the returned element in `array_get` does NOT get dropped.
    #[test]
    fn array_get_avoid_dropping_element() {
        let program = load_cairo! {
            use core::{array::{array_append, array_at, array_new}, box::{into_box, unbox}};

            fn run_test() -> @Box<felt252> {
                let mut x: Array<Box<felt252>> = array_new();
                array_append(ref x, into_box(42));

                unbox(array_at(@x, 0))
            }
        };
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(result, jit_enum!(0, jit_struct!(Value::Felt252(42.into()))));
    }

    #[test]
    fn array_snapshot_pop_front_clone_offset() {
        let program = load_cairo! {
            fn run_test() -> Span<felt252> {
                let data = array![7, 3, 4, 193827];
                let mut data = data.span();

                assert(*data.pop_front().unwrap() == 7, 0);
                let data2 = data.clone();

                assert(*data.pop_front().unwrap() == 3, 1);

                drop(data2);
                data
            }
        };
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            jit_enum!(
                0,
                jit_struct!(jit_struct!(Value::Array(vec![
                    Value::Felt252(4.into()),
                    Value::Felt252(193827.into()),
                ])))
            ),
        );
    }

    #[test]
    fn array_snapshot_pop_back_clone_offset() {
        let program = load_cairo! {
            fn run_test() -> Span<felt252> {
                let data = array![7, 3, 4, 193827];
                let mut data = data.span();

                assert(*data.pop_front().unwrap() == 7, 0);
                let data2 = data.clone();

                assert(*data.pop_back().unwrap() == 193827, 1);

                drop(data2);
                data
            }
        };
        let result = run_program(&program, "run_test", &[]).return_value;

        assert_eq!(
            result,
            jit_enum!(
                0,
                jit_struct!(jit_struct!(Value::Array(vec![
                    Value::Felt252(3.into()),
                    Value::Felt252(4.into()),
                ])))
            ),
        );
    }
}
