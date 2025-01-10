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
        attribute::IntegerAttribute, r#type::IntegerType, Block, Location, Region, Value, ValueLike,
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
                .unwrap()
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

                    let data_ptr = if REVERSE {
                        array_ptr
                    } else {
                        let offset = block.append_op_result(arith::extui(
                            extract_len_value,
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
                        .unwrap()
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
                .unwrap()
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
                    .unwrap()
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
            .unwrap()
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
                            .unwrap()
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
        .unwrap()
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
            test::{jit_enum, jit_panic, jit_struct, run_sierra_program},
        },
        values::Value,
    };
    use cairo_lang_sierra::ProgramParser;
    use pretty_assertions_sorted::assert_eq;
    use starknet_types_core::felt::Felt;

    #[test]
    fn run_roundtrip() {
        // use array::ArrayTrait;
        // fn run_test(x: Array<u32>) -> Array<u32> {
        //     x
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
            type [1] = Array<[0]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [0] = u32 [storable: true, drop: true, dup: true, zero_sized: false];

            libfunc [0] = store_temp<[1]>;

            [0]([0]) -> ([0]); // 0
            return([0]); // 1

            [0]@0([0]: [1]) -> ([1]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let result = run_sierra_program(&program, &[[1u32, 2u32].into()]).return_value;

        assert_eq!(result, Value::from([1u32, 2u32]));
    }

    #[test]
    fn run_append() {
        // use array::ArrayTrait;
        // fn run_test() -> Array<u32> {
        //     let mut numbers = ArrayTrait::new();
        //     numbers.append(4_u32);
        //     numbers
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [1] = Array<[0]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [2] = Const<[0], 4> [storable: false, drop: false, dup: false, zero_sized: false];
                type [0] = u32 [storable: true, drop: true, dup: true, zero_sized: false];

                libfunc [1] = array_new<[0]>;
                libfunc [3] = const_as_immediate<[2]>;
                libfunc [4] = store_temp<[0]>;
                libfunc [0] = array_append<[0]>;
                libfunc [5] = store_temp<[1]>;

                [1]() -> ([0]); // 0
                [3]() -> ([1]); // 1
                [4]([1]) -> ([1]); // 2
                [0]([0], [1]) -> ([2]); // 3
                [5]([2]) -> ([2]); // 4
                return([2]); // 5

                [0]@0() -> ([1]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let result = run_sierra_program(&program, &[]).return_value;

        assert_eq!(result, [4u32].into());
    }

    #[test]
    fn run_len() {
        // use array::ArrayTrait;
        // fn run_test() -> u32 {
        //     let mut numbers = ArrayTrait::new();
        //     numbers.append(4_u32);
        //     numbers.append(3_u32);
        //     numbers.append(2_u32);
        //     numbers.len()
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [1] = Array<[0]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [2] = Snapshot<[1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [5] = Const<[0], 2> [storable: false, drop: false, dup: false, zero_sized: false];
                type [4] = Const<[0], 3> [storable: false, drop: false, dup: false, zero_sized: false];
                type [3] = Const<[0], 4> [storable: false, drop: false, dup: false, zero_sized: false];
                type [0] = u32 [storable: true, drop: true, dup: true, zero_sized: false];

                libfunc [2] = array_new<[0]>;
                libfunc [4] = const_as_immediate<[3]>;
                libfunc [9] = store_temp<[0]>;
                libfunc [1] = array_append<[0]>;
                libfunc [5] = const_as_immediate<[4]>;
                libfunc [6] = const_as_immediate<[5]>;
                libfunc [7] = snapshot_take<[1]>;
                libfunc [8] = drop<[1]>;
                libfunc [10] = store_temp<[2]>;
                libfunc [0] = array_len<[0]>;

                [2]() -> ([0]); // 0
                [4]() -> ([1]); // 1
                [9]([1]) -> ([1]); // 2
                [1]([0], [1]) -> ([2]); // 3
                [5]() -> ([3]); // 4
                [9]([3]) -> ([3]); // 5
                [1]([2], [3]) -> ([4]); // 6
                [6]() -> ([5]); // 7
                [9]([5]) -> ([5]); // 8
                [1]([4], [5]) -> ([6]); // 9
                [7]([6]) -> ([7], [8]); // 10
                [8]([7]) -> (); // 11
                [10]([8]) -> ([8]); // 12
                [0]([8]) -> ([9]); // 13
                [9]([9]) -> ([9]); // 14
                return([9]); // 15

                [0]@0() -> ([0]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let result = run_sierra_program(&program, &[]).return_value;

        assert_eq!(result, 3u32.into());
    }

    #[test]
    fn run_get() {
        // use array::ArrayTrait;
        // fn run_test() -> (u32, u32, u32, u32) {
        //     let mut numbers = ArrayTrait::new();
        //     numbers.append(4_u32);
        //     numbers.append(3_u32);
        //     numbers.append(2_u32);
        //     numbers.append(1_u32);
        //     (
        //         *numbers.at(0),
        //         *numbers.at(1),
        //         *numbers.at(2),
        //         *numbers.at(3),
        //     )
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [2] = Array<[1]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [7] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
                type [9] = Array<[8]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [10] = Struct<ut@Tuple, [7], [9]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [16] = Const<[8], 1637570914057682275393755530660268060279989363> [storable: false, drop: false, dup: false, zero_sized: false];
                type [8] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
                type [1] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
                type [5] = Struct<ut@Tuple, [1], [1], [1], [1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [6] = Struct<ut@Tuple, [5]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [11] = Enum<ut@core::panics::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32),)>, [6], [10]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
                type [4] = Box<[1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [3] = Snapshot<[2]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [15] = Const<[1], 1> [storable: false, drop: false, dup: false, zero_sized: false];
                type [14] = Const<[1], 2> [storable: false, drop: false, dup: false, zero_sized: false];
                type [13] = Const<[1], 3> [storable: false, drop: false, dup: false, zero_sized: false];
                type [12] = Const<[1], 4> [storable: false, drop: false, dup: false, zero_sized: false];

                libfunc [12] = array_new<[1]>;
                libfunc [14] = const_as_immediate<[12]>;
                libfunc [25] = store_temp<[1]>;
                libfunc [11] = array_append<[1]>;
                libfunc [15] = const_as_immediate<[13]>;
                libfunc [16] = const_as_immediate<[14]>;
                libfunc [17] = const_as_immediate<[15]>;
                libfunc [18] = snapshot_take<[2]>;
                libfunc [26] = store_temp<[3]>;
                libfunc [27] = store_temp<[2]>;
                libfunc [10] = array_snapshot_pop_front<[1]>;
                libfunc [19] = branch_align;
                libfunc [20] = drop<[3]>;
                libfunc [8] = unbox<[1]>;
                libfunc [9] = array_get<[1]>;
                libfunc [28] = store_temp<[4]>;
                libfunc [21] = drop<[2]>;
                libfunc [22] = rename<[1]>;
                libfunc [7] = struct_construct<[5]>;
                libfunc [6] = struct_construct<[6]>;
                libfunc [5] = enum_init<[11], 0>;
                libfunc [29] = store_temp<[0]>;
                libfunc [30] = store_temp<[11]>;
                libfunc [23] = drop<[1]>;
                libfunc [4] = array_new<[8]>;
                libfunc [24] = const_as_immediate<[16]>;
                libfunc [31] = store_temp<[8]>;
                libfunc [3] = array_append<[8]>;
                libfunc [2] = struct_construct<[7]>;
                libfunc [1] = struct_construct<[10]>;
                libfunc [0] = enum_init<[11], 1>;

                [12]() -> ([1]); // 0
                [14]() -> ([2]); // 1
                [25]([2]) -> ([2]); // 2
                [11]([1], [2]) -> ([3]); // 3
                [15]() -> ([4]); // 4
                [25]([4]) -> ([4]); // 5
                [11]([3], [4]) -> ([5]); // 6
                [16]() -> ([6]); // 7
                [25]([6]) -> ([6]); // 8
                [11]([5], [6]) -> ([7]); // 9
                [17]() -> ([8]); // 10
                [25]([8]) -> ([8]); // 11
                [11]([7], [8]) -> ([9]); // 12
                [18]([9]) -> ([10], [11]); // 13
                [26]([11]) -> ([11]); // 14
                [27]([10]) -> ([10]); // 15
                [10]([11]) { fallthrough([12], [13]) 96([14]) }; // 16
                [19]() -> (); // 17
                [20]([12]) -> (); // 18
                [8]([13]) -> ([15]); // 19
                [18]([10]) -> ([16], [17]); // 20
                [17]() -> ([18]); // 21
                [25]([18]) -> ([18]); // 22
                [25]([15]) -> ([15]); // 23
                [9]([0], [17], [18]) { fallthrough([19], [20]) 83([21]) }; // 24
                [19]() -> (); // 25
                [28]([20]) -> ([20]); // 26
                [8]([20]) -> ([22]); // 27
                [18]([16]) -> ([23], [24]); // 28
                [16]() -> ([25]); // 29
                [25]([25]) -> ([25]); // 30
                [25]([22]) -> ([22]); // 31
                [9]([19], [24], [25]) { fallthrough([26], [27]) 69([28]) }; // 32
                [19]() -> (); // 33
                [28]([27]) -> ([27]); // 34
                [8]([27]) -> ([29]); // 35
                [18]([23]) -> ([30], [31]); // 36
                [21]([30]) -> (); // 37
                [15]() -> ([32]); // 38
                [25]([32]) -> ([32]); // 39
                [25]([29]) -> ([29]); // 40
                [9]([26], [31], [32]) { fallthrough([33], [34]) 55([35]) }; // 41
                [19]() -> (); // 42
                [28]([34]) -> ([34]); // 43
                [8]([34]) -> ([36]); // 44
                [22]([15]) -> ([37]); // 45
                [22]([22]) -> ([38]); // 46
                [22]([29]) -> ([39]); // 47
                [22]([36]) -> ([40]); // 48
                [7]([37], [38], [39], [40]) -> ([41]); // 49
                [6]([41]) -> ([42]); // 50
                [5]([42]) -> ([43]); // 51
                [29]([33]) -> ([33]); // 52
                [30]([43]) -> ([43]); // 53
                return([33], [43]); // 54
                [19]() -> (); // 55
                [23]([29]) -> (); // 56
                [23]([22]) -> (); // 57
                [23]([15]) -> (); // 58
                [4]() -> ([44]); // 59
                [24]() -> ([45]); // 60
                [31]([45]) -> ([45]); // 61
                [3]([44], [45]) -> ([46]); // 62
                [2]() -> ([47]); // 63
                [1]([47], [46]) -> ([48]); // 64
                [0]([48]) -> ([49]); // 65
                [29]([35]) -> ([35]); // 66
                [30]([49]) -> ([49]); // 67
                return([35], [49]); // 68
                [19]() -> (); // 69
                [21]([23]) -> (); // 70
                [23]([22]) -> (); // 71
                [23]([15]) -> (); // 72
                [4]() -> ([50]); // 73
                [24]() -> ([51]); // 74
                [31]([51]) -> ([51]); // 75
                [3]([50], [51]) -> ([52]); // 76
                [2]() -> ([53]); // 77
                [1]([53], [52]) -> ([54]); // 78
                [0]([54]) -> ([55]); // 79
                [29]([28]) -> ([28]); // 80
                [30]([55]) -> ([55]); // 81
                return([28], [55]); // 82
                [19]() -> (); // 83
                [23]([15]) -> (); // 84
                [21]([16]) -> (); // 85
                [4]() -> ([56]); // 86
                [24]() -> ([57]); // 87
                [31]([57]) -> ([57]); // 88
                [3]([56], [57]) -> ([58]); // 89
                [2]() -> ([59]); // 90
                [1]([59], [58]) -> ([60]); // 91
                [0]([60]) -> ([61]); // 92
                [29]([21]) -> ([21]); // 93
                [30]([61]) -> ([61]); // 94
                return([21], [61]); // 95
                [19]() -> (); // 96
                [20]([14]) -> (); // 97
                [21]([10]) -> (); // 98
                [4]() -> ([62]); // 99
                [24]() -> ([63]); // 100
                [31]([63]) -> ([63]); // 101
                [3]([62], [63]) -> ([64]); // 102
                [2]() -> ([65]); // 103
                [1]([65], [64]) -> ([66]); // 104
                [0]([66]) -> ([67]); // 105
                [29]([0]) -> ([0]); // 106
                [30]([67]) -> ([67]); // 107
                return([0], [67]); // 108

                [0]@0([0]: [0]) -> ([0], [11]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let result = run_sierra_program(&program, &[]).return_value;

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
        // use array::ArrayTrait;
        // fn run_test() -> (u32, u32, u32, u32) {
        //     let mut numbers = ArrayTrait::new();
        //     numbers.append(4_u32);
        //     numbers.append(3_u32);
        //     numbers.append(2_u32);
        //     numbers.append(2_u32);
        //     numbers.append(2_u32);
        //     numbers.append(2_u32);
        //     numbers.append(2_u32);
        //     numbers.append(2_u32);
        //     numbers.append(2_u32);
        //     numbers.append(2_u32);
        //     numbers.append(2_u32);
        //     numbers.append(2_u32);
        //     numbers.append(2_u32);
        //     numbers.append(2_u32);
        //     numbers.append(2_u32);
        //     numbers.append(2_u32);
        //     numbers.append(17_u32);
        //     numbers.append(17_u32);
        //     numbers.append(18_u32);
        //     numbers.append(19_u32);
        //     numbers.append(20_u32);
        //     numbers.append(21_u32);
        //     numbers.append(22_u32);
        //     numbers.append(23_u32);
        //     (
        //         *numbers.at(20),
        //         *numbers.at(21),
        //         *numbers.at(22),
        //         *numbers.at(23),
        //     )
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [2] = Array<[1]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [7] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
                type [9] = Array<[8]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [10] = Struct<ut@Tuple, [7], [9]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [22] = Const<[8], 1637570914057682275393755530660268060279989363> [storable: false, drop: false, dup: false, zero_sized: false];
                type [8] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
                type [1] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
                type [5] = Struct<ut@Tuple, [1], [1], [1], [1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [6] = Struct<ut@Tuple, [5]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [11] = Enum<ut@core::panics::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32),)>, [6], [10]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [4] = Box<[1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
                type [3] = Snapshot<[2]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [21] = Const<[1], 23> [storable: false, drop: false, dup: false, zero_sized: false];
                type [20] = Const<[1], 22> [storable: false, drop: false, dup: false, zero_sized: false];
                type [19] = Const<[1], 21> [storable: false, drop: false, dup: false, zero_sized: false];
                type [18] = Const<[1], 20> [storable: false, drop: false, dup: false, zero_sized: false];
                type [17] = Const<[1], 19> [storable: false, drop: false, dup: false, zero_sized: false];
                type [16] = Const<[1], 18> [storable: false, drop: false, dup: false, zero_sized: false];
                type [15] = Const<[1], 17> [storable: false, drop: false, dup: false, zero_sized: false];
                type [14] = Const<[1], 2> [storable: false, drop: false, dup: false, zero_sized: false];
                type [13] = Const<[1], 3> [storable: false, drop: false, dup: false, zero_sized: false];
                type [12] = Const<[1], 4> [storable: false, drop: false, dup: false, zero_sized: false];

                libfunc [11] = array_new<[1]>;
                libfunc [13] = const_as_immediate<[12]>;
                libfunc [29] = store_temp<[1]>;
                libfunc [10] = array_append<[1]>;
                libfunc [14] = const_as_immediate<[13]>;
                libfunc [15] = const_as_immediate<[14]>;
                libfunc [16] = const_as_immediate<[15]>;
                libfunc [17] = const_as_immediate<[16]>;
                libfunc [18] = const_as_immediate<[17]>;
                libfunc [19] = const_as_immediate<[18]>;
                libfunc [20] = const_as_immediate<[19]>;
                libfunc [21] = const_as_immediate<[20]>;
                libfunc [22] = const_as_immediate<[21]>;
                libfunc [23] = snapshot_take<[2]>;
                libfunc [30] = store_temp<[3]>;
                libfunc [31] = store_temp<[2]>;
                libfunc [9] = array_get<[1]>;
                libfunc [24] = branch_align;
                libfunc [32] = store_temp<[4]>;
                libfunc [8] = unbox<[1]>;
                libfunc [25] = drop<[2]>;
                libfunc [26] = rename<[1]>;
                libfunc [7] = struct_construct<[5]>;
                libfunc [6] = struct_construct<[6]>;
                libfunc [5] = enum_init<[11], 0>;
                libfunc [33] = store_temp<[0]>;
                libfunc [34] = store_temp<[11]>;
                libfunc [27] = drop<[1]>;
                libfunc [4] = array_new<[8]>;
                libfunc [28] = const_as_immediate<[22]>;
                libfunc [35] = store_temp<[8]>;
                libfunc [3] = array_append<[8]>;
                libfunc [2] = struct_construct<[7]>;
                libfunc [1] = struct_construct<[10]>;
                libfunc [0] = enum_init<[11], 1>;

                [11]() -> ([1]); // 0
                [13]() -> ([2]); // 1
                [29]([2]) -> ([2]); // 2
                [10]([1], [2]) -> ([3]); // 3
                [14]() -> ([4]); // 4
                [29]([4]) -> ([4]); // 5
                [10]([3], [4]) -> ([5]); // 6
                [15]() -> ([6]); // 7
                [29]([6]) -> ([6]); // 8
                [10]([5], [6]) -> ([7]); // 9
                [15]() -> ([8]); // 10
                [29]([8]) -> ([8]); // 11
                [10]([7], [8]) -> ([9]); // 12
                [15]() -> ([10]); // 13
                [29]([10]) -> ([10]); // 14
                [10]([9], [10]) -> ([11]); // 15
                [15]() -> ([12]); // 16
                [29]([12]) -> ([12]); // 17
                [10]([11], [12]) -> ([13]); // 18
                [15]() -> ([14]); // 19
                [29]([14]) -> ([14]); // 20
                [10]([13], [14]) -> ([15]); // 21
                [15]() -> ([16]); // 22
                [29]([16]) -> ([16]); // 23
                [10]([15], [16]) -> ([17]); // 24
                [15]() -> ([18]); // 25
                [29]([18]) -> ([18]); // 26
                [10]([17], [18]) -> ([19]); // 27
                [15]() -> ([20]); // 28
                [29]([20]) -> ([20]); // 29
                [10]([19], [20]) -> ([21]); // 30
                [15]() -> ([22]); // 31
                [29]([22]) -> ([22]); // 32
                [10]([21], [22]) -> ([23]); // 33
                [15]() -> ([24]); // 34
                [29]([24]) -> ([24]); // 35
                [10]([23], [24]) -> ([25]); // 36
                [15]() -> ([26]); // 37
                [29]([26]) -> ([26]); // 38
                [10]([25], [26]) -> ([27]); // 39
                [15]() -> ([28]); // 40
                [29]([28]) -> ([28]); // 41
                [10]([27], [28]) -> ([29]); // 42
                [15]() -> ([30]); // 43
                [29]([30]) -> ([30]); // 44
                [10]([29], [30]) -> ([31]); // 45
                [15]() -> ([32]); // 46
                [29]([32]) -> ([32]); // 47
                [10]([31], [32]) -> ([33]); // 48
                [16]() -> ([34]); // 49
                [29]([34]) -> ([34]); // 50
                [10]([33], [34]) -> ([35]); // 51
                [16]() -> ([36]); // 52
                [29]([36]) -> ([36]); // 53
                [10]([35], [36]) -> ([37]); // 54
                [17]() -> ([38]); // 55
                [29]([38]) -> ([38]); // 56
                [10]([37], [38]) -> ([39]); // 57
                [18]() -> ([40]); // 58
                [29]([40]) -> ([40]); // 59
                [10]([39], [40]) -> ([41]); // 60
                [19]() -> ([42]); // 61
                [29]([42]) -> ([42]); // 62
                [10]([41], [42]) -> ([43]); // 63
                [20]() -> ([44]); // 64
                [29]([44]) -> ([44]); // 65
                [10]([43], [44]) -> ([45]); // 66
                [21]() -> ([46]); // 67
                [29]([46]) -> ([46]); // 68
                [10]([45], [46]) -> ([47]); // 69
                [22]() -> ([48]); // 70
                [29]([48]) -> ([48]); // 71
                [10]([47], [48]) -> ([49]); // 72
                [23]([49]) -> ([50], [51]); // 73
                [19]() -> ([52]); // 74
                [30]([51]) -> ([51]); // 75
                [29]([52]) -> ([52]); // 76
                [31]([50]) -> ([50]); // 77
                [9]([0], [51], [52]) { fallthrough([53], [54]) 158([55]) }; // 78
                [24]() -> (); // 79
                [32]([54]) -> ([54]); // 80
                [8]([54]) -> ([56]); // 81
                [23]([50]) -> ([57], [58]); // 82
                [20]() -> ([59]); // 83
                [29]([59]) -> ([59]); // 84
                [29]([56]) -> ([56]); // 85
                [9]([53], [58], [59]) { fallthrough([60], [61]) 145([62]) }; // 86
                [24]() -> (); // 87
                [32]([61]) -> ([61]); // 88
                [8]([61]) -> ([63]); // 89
                [23]([57]) -> ([64], [65]); // 90
                [21]() -> ([66]); // 91
                [29]([66]) -> ([66]); // 92
                [29]([63]) -> ([63]); // 93
                [9]([60], [65], [66]) { fallthrough([67], [68]) 131([69]) }; // 94
                [24]() -> (); // 95
                [32]([68]) -> ([68]); // 96
                [8]([68]) -> ([70]); // 97
                [23]([64]) -> ([71], [72]); // 98
                [25]([71]) -> (); // 99
                [22]() -> ([73]); // 100
                [29]([73]) -> ([73]); // 101
                [29]([70]) -> ([70]); // 102
                [9]([67], [72], [73]) { fallthrough([74], [75]) 117([76]) }; // 103
                [24]() -> (); // 104
                [32]([75]) -> ([75]); // 105
                [8]([75]) -> ([77]); // 106
                [26]([56]) -> ([78]); // 107
                [26]([63]) -> ([79]); // 108
                [26]([70]) -> ([80]); // 109
                [26]([77]) -> ([81]); // 110
                [7]([78], [79], [80], [81]) -> ([82]); // 111
                [6]([82]) -> ([83]); // 112
                [5]([83]) -> ([84]); // 113
                [33]([74]) -> ([74]); // 114
                [34]([84]) -> ([84]); // 115
                return([74], [84]); // 116
                [24]() -> (); // 117
                [27]([70]) -> (); // 118
                [27]([63]) -> (); // 119
                [27]([56]) -> (); // 120
                [4]() -> ([85]); // 121
                [28]() -> ([86]); // 122
                [35]([86]) -> ([86]); // 123
                [3]([85], [86]) -> ([87]); // 124
                [2]() -> ([88]); // 125
                [1]([88], [87]) -> ([89]); // 126
                [0]([89]) -> ([90]); // 127
                [33]([76]) -> ([76]); // 128
                [34]([90]) -> ([90]); // 129
                return([76], [90]); // 130
                [24]() -> (); // 131
                [25]([64]) -> (); // 132
                [27]([63]) -> (); // 133
                [27]([56]) -> (); // 134
                [4]() -> ([91]); // 135
                [28]() -> ([92]); // 136
                [35]([92]) -> ([92]); // 137
                [3]([91], [92]) -> ([93]); // 138
                [2]() -> ([94]); // 139
                [1]([94], [93]) -> ([95]); // 140
                [0]([95]) -> ([96]); // 141
                [33]([69]) -> ([69]); // 142
                [34]([96]) -> ([96]); // 143
                return([69], [96]); // 144
                [24]() -> (); // 145
                [27]([56]) -> (); // 146
                [25]([57]) -> (); // 147
                [4]() -> ([97]); // 148
                [28]() -> ([98]); // 149
                [35]([98]) -> ([98]); // 150
                [3]([97], [98]) -> ([99]); // 151
                [2]() -> ([100]); // 152
                [1]([100], [99]) -> ([101]); // 153
                [0]([101]) -> ([102]); // 154
                [33]([62]) -> ([62]); // 155
                [34]([102]) -> ([102]); // 156
                return([62], [102]); // 157
                [24]() -> (); // 158
                [25]([50]) -> (); // 159
                [4]() -> ([103]); // 160
                [28]() -> ([104]); // 161
                [35]([104]) -> ([104]); // 162
                [3]([103], [104]) -> ([105]); // 163
                [2]() -> ([106]); // 164
                [1]([106], [105]) -> ([107]); // 165
                [0]([107]) -> ([108]); // 166
                [33]([55]) -> ([55]); // 167
                [34]([108]) -> ([108]); // 168
                return([55], [108]); // 169

                [0]@0([0]: [0]) -> ([0], [11]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let result = run_sierra_program(&program, &[]).return_value;

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
        // use array::ArrayTrait;
        // fn run_test() -> u32 {
        //     let mut numbers = ArrayTrait::new();
        //     numbers.append(4_u32);
        //     numbers.append(3_u32);
        //     let _ = numbers.pop_front();
        //     numbers.append(1_u32);
        //     *numbers.at(0)
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [1] = Array<[0]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [5] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
                type [7] = Array<[6]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [8] = Struct<ut@Tuple, [5], [7]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [13] = Const<[6], 1637570914057682275393755530660268060279989363> [storable: false, drop: false, dup: false, zero_sized: false];
                type [6] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
                type [0] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
                type [4] = Struct<ut@Tuple, [0]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [9] = Enum<ut@core::panics::PanicResult::<(core::integer::u32,)>, [4], [8]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [3] = Snapshot<[1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [12] = Const<[0], 1> [storable: false, drop: false, dup: false, zero_sized: false];
                type [2] = Box<[0]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [11] = Const<[0], 3> [storable: false, drop: false, dup: false, zero_sized: false];
                type [10] = Const<[0], 4> [storable: false, drop: false, dup: false, zero_sized: false];

                libfunc [11] = array_new<[0]>;
                libfunc [13] = const_as_immediate<[10]>;
                libfunc [24] = store_temp<[0]>;
                libfunc [9] = array_append<[0]>;
                libfunc [14] = const_as_immediate<[11]>;
                libfunc [25] = store_temp<[1]>;
                libfunc [10] = array_pop_front<[0]>;
                libfunc [15] = branch_align;
                libfunc [7] = unbox<[0]>;
                libfunc [16] = drop<[0]>;
                libfunc [17] = jump;
                libfunc [18] = const_as_immediate<[12]>;
                libfunc [19] = snapshot_take<[1]>;
                libfunc [20] = drop<[1]>;
                libfunc [26] = store_temp<[3]>;
                libfunc [8] = array_snapshot_pop_front<[0]>;
                libfunc [21] = drop<[3]>;
                libfunc [22] = rename<[0]>;
                libfunc [6] = struct_construct<[4]>;
                libfunc [5] = enum_init<[9], 0>;
                libfunc [27] = store_temp<[9]>;
                libfunc [4] = array_new<[6]>;
                libfunc [23] = const_as_immediate<[13]>;
                libfunc [28] = store_temp<[6]>;
                libfunc [3] = array_append<[6]>;
                libfunc [2] = struct_construct<[5]>;
                libfunc [1] = struct_construct<[8]>;
                libfunc [0] = enum_init<[9], 1>;

                [11]() -> ([0]); // 0
                [13]() -> ([1]); // 1
                [24]([1]) -> ([1]); // 2
                [9]([0], [1]) -> ([2]); // 3
                [14]() -> ([3]); // 4
                [24]([3]) -> ([3]); // 5
                [9]([2], [3]) -> ([4]); // 6
                [25]([4]) -> ([4]); // 7
                [10]([4]) { fallthrough([5], [6]) 14([7]) }; // 8
                [15]() -> (); // 9
                [7]([6]) -> ([8]); // 10
                [16]([8]) -> (); // 11
                [25]([5]) -> ([9]); // 12
                [17]() { 16() }; // 13
                [15]() -> (); // 14
                [25]([7]) -> ([9]); // 15
                [18]() -> ([10]); // 16
                [24]([10]) -> ([10]); // 17
                [9]([9], [10]) -> ([11]); // 18
                [19]([11]) -> ([12], [13]); // 19
                [20]([12]) -> (); // 20
                [26]([13]) -> ([13]); // 21
                [8]([13]) { fallthrough([14], [15]) 31([16]) }; // 22
                [15]() -> (); // 23
                [21]([14]) -> (); // 24
                [7]([15]) -> ([17]); // 25
                [22]([17]) -> ([18]); // 26
                [6]([18]) -> ([19]); // 27
                [5]([19]) -> ([20]); // 28
                [27]([20]) -> ([20]); // 29
                return([20]); // 30
                [15]() -> (); // 31
                [21]([16]) -> (); // 32
                [4]() -> ([21]); // 33
                [23]() -> ([22]); // 34
                [28]([22]) -> ([22]); // 35
                [3]([21], [22]) -> ([23]); // 36
                [2]() -> ([24]); // 37
                [1]([24], [23]) -> ([25]); // 38
                [0]([25]) -> ([26]); // 39
                [27]([26]) -> ([26]); // 40
                return([26]); // 41

                [0]@0() -> ([9]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let result = run_sierra_program(&program, &[]).return_value;

        assert_eq!(result, jit_enum!(0, jit_struct!(3u32.into())));
    }

    #[test]
    fn run_pop_front_result() {
        // use array::ArrayTrait;
        // fn run_test() -> Option<u32> {
        //     let mut numbers = ArrayTrait::new();
        //     numbers.append(4_u32);
        //     numbers.append(3_u32);
        //     numbers.pop_front()
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [1] = Array<[0]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [3] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
                type [0] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
                type [4] = Enum<ut@core::option::Option::<core::integer::u32>, [0], [3]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [2] = Box<[0]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [6] = Const<[0], 3> [storable: false, drop: false, dup: false, zero_sized: false];
                type [5] = Const<[0], 4> [storable: false, drop: false, dup: false, zero_sized: false];

                libfunc [6] = array_new<[0]>;
                libfunc [8] = const_as_immediate<[5]>;
                libfunc [12] = store_temp<[0]>;
                libfunc [5] = array_append<[0]>;
                libfunc [9] = const_as_immediate<[6]>;
                libfunc [13] = store_temp<[1]>;
                libfunc [4] = array_pop_front<[0]>;
                libfunc [10] = branch_align;
                libfunc [11] = drop<[1]>;
                libfunc [3] = unbox<[0]>;
                libfunc [2] = enum_init<[4], 0>;
                libfunc [14] = store_temp<[4]>;
                libfunc [1] = struct_construct<[3]>;
                libfunc [0] = enum_init<[4], 1>;

                [6]() -> ([0]); // 0
                [8]() -> ([1]); // 1
                [12]([1]) -> ([1]); // 2
                [5]([0], [1]) -> ([2]); // 3
                [9]() -> ([3]); // 4
                [12]([3]) -> ([3]); // 5
                [5]([2], [3]) -> ([4]); // 6
                [13]([4]) -> ([4]); // 7
                [4]([4]) { fallthrough([5], [6]) 15([7]) }; // 8
                [10]() -> (); // 9
                [11]([5]) -> (); // 10
                [3]([6]) -> ([8]); // 11
                [2]([8]) -> ([9]); // 12
                [14]([9]) -> ([9]); // 13
                return([9]); // 14
                [10]() -> (); // 15
                [11]([7]) -> (); // 16
                [1]() -> ([10]); // 17
                [0]([10]) -> ([11]); // 18
                [14]([11]) -> ([11]); // 19
                return([11]); // 20

                [0]@0() -> ([4]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let result = run_sierra_program(&program, &[]).return_value;

        assert_eq!(result, jit_enum!(0, 4u32.into()));

        // use array::ArrayTrait;
        // fn run_test() -> Option<u32> {
        //     let mut numbers = ArrayTrait::new();
        //     numbers.pop_front()
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [1] = Array<[0]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [3] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
                type [0] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
                type [4] = Enum<ut@core::option::Option::<core::integer::u32>, [0], [3]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [2] = Box<[0]> [storable: true, drop: true, dup: true, zero_sized: false];

                libfunc [5] = array_new<[0]>;
                libfunc [4] = array_pop_front<[0]>;
                libfunc [7] = branch_align;
                libfunc [8] = drop<[1]>;
                libfunc [3] = unbox<[0]>;
                libfunc [2] = enum_init<[4], 0>;
                libfunc [9] = store_temp<[4]>;
                libfunc [1] = struct_construct<[3]>;
                libfunc [0] = enum_init<[4], 1>;

                [5]() -> ([0]); // 0
                [4]([0]) { fallthrough([1], [2]) 8([3]) }; // 1
                [7]() -> (); // 2
                [8]([1]) -> (); // 3
                [3]([2]) -> ([4]); // 4
                [2]([4]) -> ([5]); // 5
                [9]([5]) -> ([5]); // 6
                return([5]); // 7
                [7]() -> (); // 8
                [8]([3]) -> (); // 9
                [1]() -> ([6]); // 10
                [0]([6]) -> ([7]); // 11
                [9]([7]) -> ([7]); // 12
                return([7]); // 13

                [0]@0() -> ([4]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let result = run_sierra_program(&program, &[]).return_value;

        assert_eq!(result, jit_enum!(1, jit_struct!()));
    }

    #[test]
    fn run_pop_front_consume() {
        // use array::ArrayTrait;
        // fn run_test() -> u32 {
        //     let mut numbers = ArrayTrait::new();
        //     numbers.append(4_u32);
        //     numbers.append(3_u32);
        //     match numbers.pop_front_consume() {
        //         Option::Some((_, x)) => x,
        //         Option::None(()) => 0_u32,
        //     }
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [1] = Array<[0]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [5] = Const<[0], 0> [storable: false, drop: false, dup: false, zero_sized: false];
                type [2] = Box<[0]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [4] = Const<[0], 3> [storable: false, drop: false, dup: false, zero_sized: false];
                type [3] = Const<[0], 4> [storable: false, drop: false, dup: false, zero_sized: false];
                type [0] = u32 [storable: true, drop: true, dup: true, zero_sized: false];

                libfunc [3] = array_new<[0]>;
                libfunc [5] = const_as_immediate<[3]>;
                libfunc [10] = store_temp<[0]>;
                libfunc [2] = array_append<[0]>;
                libfunc [6] = const_as_immediate<[4]>;
                libfunc [11] = store_temp<[1]>;
                libfunc [1] = array_pop_front_consume<[0]>;
                libfunc [7] = branch_align;
                libfunc [8] = drop<[1]>;
                libfunc [0] = unbox<[0]>;
                libfunc [9] = const_as_immediate<[5]>;

                [3]() -> ([0]); // 0
                [5]() -> ([1]); // 1
                [10]([1]) -> ([1]); // 2
                [2]([0], [1]) -> ([2]); // 3
                [6]() -> ([3]); // 4
                [10]([3]) -> ([3]); // 5
                [2]([2], [3]) -> ([4]); // 6
                [11]([4]) -> ([4]); // 7
                [1]([4]) { fallthrough([5], [6]) 14() }; // 8
                [7]() -> (); // 9
                [8]([5]) -> (); // 10
                [0]([6]) -> ([7]); // 11
                [10]([7]) -> ([7]); // 12
                return([7]); // 13
                [7]() -> (); // 14
                [9]() -> ([8]); // 15
                [10]([8]) -> ([8]); // 16
                return([8]); // 17

                [0]@0() -> ([0]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let result = run_sierra_program(&program, &[]).return_value;

        assert_eq!(result, 4u32.into());
    }

    #[test]
    fn run_pop_back() {
        // use array::ArrayTrait;
        // fn run_test() -> (Option<@u32>, Option<@u32>, Option<@u32>, Option<@u32>) {
        //     let mut numbers = ArrayTrait::new();
        //     numbers.append(4_u32);
        //     numbers.append(3_u32);
        //     numbers.append(1_u32);
        //     let mut numbers = numbers.span();
        //     (
        //         numbers.pop_back(),
        //         numbers.pop_back(),
        //         numbers.pop_back(),
        //         numbers.pop_back(),
        //     )
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [1] = Array<[0]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [0] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
                type [4] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
                type [6] = Enum<ut@core::option::Option::<@core::integer::u32>, [0], [4]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [7] = Struct<ut@Tuple, [6], [6], [6], [6]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [3] = Box<[0]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [5] = Enum<ut@core::option::Option::<core::box::Box::<@core::integer::u32>>, [3], [4]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [2] = Snapshot<[1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [10] = Const<[0], 1> [storable: false, drop: false, dup: false, zero_sized: false];
                type [9] = Const<[0], 3> [storable: false, drop: false, dup: false, zero_sized: false];
                type [8] = Const<[0], 4> [storable: false, drop: false, dup: false, zero_sized: false];

                libfunc [10] = array_new<[0]>;
                libfunc [12] = const_as_immediate<[8]>;
                libfunc [21] = store_temp<[0]>;
                libfunc [9] = array_append<[0]>;
                libfunc [13] = const_as_immediate<[9]>;
                libfunc [14] = const_as_immediate<[10]>;
                libfunc [15] = snapshot_take<[1]>;
                libfunc [16] = drop<[1]>;
                libfunc [22] = store_temp<[2]>;
                libfunc [5] = array_snapshot_pop_back<[0]>;
                libfunc [17] = branch_align;
                libfunc [8] = enum_init<[5], 0>;
                libfunc [23] = store_temp<[5]>;
                libfunc [18] = jump;
                libfunc [2] = struct_construct<[4]>;
                libfunc [7] = enum_init<[5], 1>;
                libfunc [6] = enum_match<[5]>;
                libfunc [4] = unbox<[0]>;
                libfunc [3] = enum_init<[6], 0>;
                libfunc [24] = store_temp<[6]>;
                libfunc [19] = drop<[4]>;
                libfunc [1] = enum_init<[6], 1>;
                libfunc [20] = drop<[2]>;
                libfunc [25] = store_temp<[3]>;
                libfunc [0] = struct_construct<[7]>;
                libfunc [26] = store_temp<[7]>;

                [10]() -> ([0]); // 0
                [12]() -> ([1]); // 1
                [21]([1]) -> ([1]); // 2
                [9]([0], [1]) -> ([2]); // 3
                [13]() -> ([3]); // 4
                [21]([3]) -> ([3]); // 5
                [9]([2], [3]) -> ([4]); // 6
                [14]() -> ([5]); // 7
                [21]([5]) -> ([5]); // 8
                [9]([4], [5]) -> ([6]); // 9
                [15]([6]) -> ([7], [8]); // 10
                [16]([7]) -> (); // 11
                [22]([8]) -> ([8]); // 12
                [5]([8]) { fallthrough([9], [10]) 19([11]) }; // 13
                [17]() -> (); // 14
                [8]([10]) -> ([12]); // 15
                [22]([9]) -> ([13]); // 16
                [23]([12]) -> ([14]); // 17
                [18]() { 24() }; // 18
                [17]() -> (); // 19
                [2]() -> ([15]); // 20
                [7]([15]) -> ([16]); // 21
                [22]([11]) -> ([13]); // 22
                [23]([16]) -> ([14]); // 23
                [6]([14]) { fallthrough([17]) 30([18]) }; // 24
                [17]() -> (); // 25
                [4]([17]) -> ([19]); // 26
                [3]([19]) -> ([20]); // 27
                [24]([20]) -> ([21]); // 28
                [18]() { 35() }; // 29
                [17]() -> (); // 30
                [19]([18]) -> (); // 31
                [2]() -> ([22]); // 32
                [1]([22]) -> ([23]); // 33
                [24]([23]) -> ([21]); // 34
                [5]([13]) { fallthrough([24], [25]) 41([26]) }; // 35
                [17]() -> (); // 36
                [8]([25]) -> ([27]); // 37
                [22]([24]) -> ([28]); // 38
                [23]([27]) -> ([29]); // 39
                [18]() { 46() }; // 40
                [17]() -> (); // 41
                [2]() -> ([30]); // 42
                [7]([30]) -> ([31]); // 43
                [22]([26]) -> ([28]); // 44
                [23]([31]) -> ([29]); // 45
                [6]([29]) { fallthrough([32]) 52([33]) }; // 46
                [17]() -> (); // 47
                [4]([32]) -> ([34]); // 48
                [3]([34]) -> ([35]); // 49
                [24]([35]) -> ([36]); // 50
                [18]() { 57() }; // 51
                [17]() -> (); // 52
                [19]([33]) -> (); // 53
                [2]() -> ([37]); // 54
                [1]([37]) -> ([38]); // 55
                [24]([38]) -> ([36]); // 56
                [5]([28]) { fallthrough([39], [40]) 63([41]) }; // 57
                [17]() -> (); // 58
                [8]([40]) -> ([42]); // 59
                [22]([39]) -> ([43]); // 60
                [23]([42]) -> ([44]); // 61
                [18]() { 68() }; // 62
                [17]() -> (); // 63
                [2]() -> ([45]); // 64
                [7]([45]) -> ([46]); // 65
                [22]([41]) -> ([43]); // 66
                [23]([46]) -> ([44]); // 67
                [6]([44]) { fallthrough([47]) 74([48]) }; // 68
                [17]() -> (); // 69
                [4]([47]) -> ([49]); // 70
                [3]([49]) -> ([50]); // 71
                [24]([50]) -> ([51]); // 72
                [18]() { 79() }; // 73
                [17]() -> (); // 74
                [19]([48]) -> (); // 75
                [2]() -> ([52]); // 76
                [1]([52]) -> ([53]); // 77
                [24]([53]) -> ([51]); // 78
                [5]([43]) { fallthrough([54], [55]) 87([56]) }; // 79
                [17]() -> (); // 80
                [20]([54]) -> (); // 81
                [25]([55]) -> ([55]); // 82
                [4]([55]) -> ([57]); // 83
                [3]([57]) -> ([58]); // 84
                [24]([58]) -> ([59]); // 85
                [18]() { 92() }; // 86
                [17]() -> (); // 87
                [20]([56]) -> (); // 88
                [2]() -> ([60]); // 89
                [1]([60]) -> ([61]); // 90
                [24]([61]) -> ([59]); // 91
                [0]([21], [36], [51], [59]) -> ([62]); // 92
                [26]([62]) -> ([62]); // 93
                return([62]); // 94

                [0]@0() -> ([7]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let result = run_sierra_program(&program, &[]).return_value;

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
        // use array::Array;
        // use array::ArrayTrait;
        // use array::SpanTrait;
        // use option::OptionTrait;
        // use box::BoxTrait;
        // fn run_test() -> u32 {
        //     let mut data: Array<u32> = ArrayTrait::new(); // Alloca (freed).
        //     data.append(1_u32);
        //     data.append(2_u32);
        //     data.append(3_u32);
        //     data.append(4_u32);
        //     let sp = data.span(); // Alloca (leaked).
        //     let slice = sp.slice(1, 2);
        //     data.append(5_u32);
        //     data.append(5_u32);
        //     data.append(5_u32);
        //     data.append(5_u32);
        //     data.append(5_u32); // Realloc (freed).
        //     data.append(5_u32);
        //     *slice.get(1).unwrap().unbox()
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [2] = Array<[1]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [17] = Const<[7], 1637570914057682275393755530660268060279989363> [storable: false, drop: false, dup: false, zero_sized: false];
                type [6] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
                type [8] = Array<[7]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [9] = Struct<ut@Tuple, [6], [8]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [16] = Const<[7], 29721761890975875353235833581453094220424382983267374> [storable: false, drop: false, dup: false, zero_sized: false];
                type [7] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
                type [1] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
                type [5] = Struct<ut@Tuple, [1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [10] = Enum<ut@core::panics::PanicResult::<(core::integer::u32,)>, [5], [9]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [4] = Box<[1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [15] = Const<[1], 5> [storable: false, drop: false, dup: false, zero_sized: false];
                type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
                type [3] = Snapshot<[2]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [14] = Const<[1], 4> [storable: false, drop: false, dup: false, zero_sized: false];
                type [13] = Const<[1], 3> [storable: false, drop: false, dup: false, zero_sized: false];
                type [12] = Const<[1], 2> [storable: false, drop: false, dup: false, zero_sized: false];
                type [11] = Const<[1], 1> [storable: false, drop: false, dup: false, zero_sized: false];

                libfunc [11] = array_new<[1]>;
                libfunc [13] = const_as_immediate<[11]>;
                libfunc [24] = store_temp<[1]>;
                libfunc [9] = array_append<[1]>;
                libfunc [14] = const_as_immediate<[12]>;
                libfunc [15] = const_as_immediate<[13]>;
                libfunc [16] = const_as_immediate<[14]>;
                libfunc [17] = snapshot_take<[2]>;
                libfunc [25] = store_temp<[3]>;
                libfunc [26] = store_temp<[2]>;
                libfunc [10] = array_slice<[1]>;
                libfunc [18] = branch_align;
                libfunc [19] = const_as_immediate<[15]>;
                libfunc [20] = drop<[2]>;
                libfunc [8] = array_get<[1]>;
                libfunc [27] = store_temp<[4]>;
                libfunc [7] = unbox<[1]>;
                libfunc [21] = rename<[1]>;
                libfunc [6] = struct_construct<[5]>;
                libfunc [5] = enum_init<[10], 0>;
                libfunc [28] = store_temp<[0]>;
                libfunc [29] = store_temp<[10]>;
                libfunc [4] = array_new<[7]>;
                libfunc [22] = const_as_immediate<[16]>;
                libfunc [30] = store_temp<[7]>;
                libfunc [3] = array_append<[7]>;
                libfunc [2] = struct_construct<[6]>;
                libfunc [1] = struct_construct<[9]>;
                libfunc [0] = enum_init<[10], 1>;
                libfunc [23] = const_as_immediate<[17]>;

                [11]() -> ([1]); // 0
                [13]() -> ([2]); // 1
                [24]([2]) -> ([2]); // 2
                [9]([1], [2]) -> ([3]); // 3
                [14]() -> ([4]); // 4
                [24]([4]) -> ([4]); // 5
                [9]([3], [4]) -> ([5]); // 6
                [15]() -> ([6]); // 7
                [24]([6]) -> ([6]); // 8
                [9]([5], [6]) -> ([7]); // 9
                [16]() -> ([8]); // 10
                [24]([8]) -> ([8]); // 11
                [9]([7], [8]) -> ([9]); // 12
                [17]([9]) -> ([10], [11]); // 13
                [13]() -> ([12]); // 14
                [14]() -> ([13]); // 15
                [25]([11]) -> ([11]); // 16
                [24]([12]) -> ([12]); // 17
                [24]([13]) -> ([13]); // 18
                [26]([10]) -> ([10]); // 19
                [10]([0], [11], [12], [13]) { fallthrough([14], [15]) 65([16]) }; // 20
                [18]() -> (); // 21
                [19]() -> ([17]); // 22
                [24]([17]) -> ([17]); // 23
                [9]([10], [17]) -> ([18]); // 24
                [19]() -> ([19]); // 25
                [24]([19]) -> ([19]); // 26
                [9]([18], [19]) -> ([20]); // 27
                [19]() -> ([21]); // 28
                [24]([21]) -> ([21]); // 29
                [9]([20], [21]) -> ([22]); // 30
                [19]() -> ([23]); // 31
                [24]([23]) -> ([23]); // 32
                [9]([22], [23]) -> ([24]); // 33
                [19]() -> ([25]); // 34
                [24]([25]) -> ([25]); // 35
                [9]([24], [25]) -> ([26]); // 36
                [19]() -> ([27]); // 37
                [24]([27]) -> ([27]); // 38
                [9]([26], [27]) -> ([28]); // 39
                [20]([28]) -> (); // 40
                [13]() -> ([29]); // 41
                [25]([15]) -> ([15]); // 42
                [24]([29]) -> ([29]); // 43
                [8]([14], [15], [29]) { fallthrough([30], [31]) 54([32]) }; // 44
                [18]() -> (); // 45
                [27]([31]) -> ([31]); // 46
                [7]([31]) -> ([33]); // 47
                [21]([33]) -> ([34]); // 48
                [6]([34]) -> ([35]); // 49
                [5]([35]) -> ([36]); // 50
                [28]([30]) -> ([30]); // 51
                [29]([36]) -> ([36]); // 52
                return([30], [36]); // 53
                [18]() -> (); // 54
                [4]() -> ([37]); // 55
                [22]() -> ([38]); // 56
                [30]([38]) -> ([38]); // 57
                [3]([37], [38]) -> ([39]); // 58
                [2]() -> ([40]); // 59
                [1]([40], [39]) -> ([41]); // 60
                [0]([41]) -> ([42]); // 61
                [28]([32]) -> ([32]); // 62
                [29]([42]) -> ([42]); // 63
                return([32], [42]); // 64
                [18]() -> (); // 65
                [20]([10]) -> (); // 66
                [4]() -> ([43]); // 67
                [23]() -> ([44]); // 68
                [30]([44]) -> ([44]); // 69
                [3]([43], [44]) -> ([45]); // 70
                [2]() -> ([46]); // 71
                [1]([46], [45]) -> ([47]); // 72
                [0]([47]) -> ([48]); // 73
                [28]([16]) -> ([16]); // 74
                [29]([48]) -> ([48]); // 75
                return([16], [48]); // 76

                [0]@0([0]: [0]) -> ([0], [10]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let result = run_sierra_program(&program, &[]).return_value;

        assert_eq!(result, jit_enum!(0, jit_struct!(3u32.into())));
    }

    #[test]
    fn run_slice_fail() {
        // use array::Array;
        // use array::ArrayTrait;
        // use array::SpanTrait;
        // use option::OptionTrait;
        // use box::BoxTrait;
        // fn run_test() -> u32 {
        //     let mut data: Array<u32> = ArrayTrait::new();
        //     data.append(1_u32);
        //     data.append(2_u32);
        //     data.append(3_u32);
        //     data.append(4_u32);
        //     let sp = data.span();
        //     let slice = sp.slice(1, 4); // oob
        //     //data.append(5_u32);
        //     *slice.get(0).unwrap().unbox()
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [2] = Array<[1]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [16] = Const<[7], 1637570914057682275393755530660268060279989363> [storable: false, drop: false, dup: false, zero_sized: false];
                type [6] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
                type [8] = Array<[7]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [9] = Struct<ut@Tuple, [6], [8]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [15] = Const<[7], 29721761890975875353235833581453094220424382983267374> [storable: false, drop: false, dup: false, zero_sized: false];
                type [7] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
                type [1] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
                type [5] = Struct<ut@Tuple, [1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [10] = Enum<ut@core::panics::PanicResult::<(core::integer::u32,)>, [5], [9]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [4] = Box<[1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
                type [3] = Snapshot<[2]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [14] = Const<[1], 4> [storable: false, drop: false, dup: false, zero_sized: false];
                type [13] = Const<[1], 3> [storable: false, drop: false, dup: false, zero_sized: false];
                type [12] = Const<[1], 2> [storable: false, drop: false, dup: false, zero_sized: false];
                type [11] = Const<[1], 1> [storable: false, drop: false, dup: false, zero_sized: false];

                libfunc [11] = array_new<[1]>;
                libfunc [13] = const_as_immediate<[11]>;
                libfunc [24] = store_temp<[1]>;
                libfunc [10] = array_append<[1]>;
                libfunc [14] = const_as_immediate<[12]>;
                libfunc [15] = const_as_immediate<[13]>;
                libfunc [16] = const_as_immediate<[14]>;
                libfunc [17] = snapshot_take<[2]>;
                libfunc [18] = drop<[2]>;
                libfunc [25] = store_temp<[3]>;
                libfunc [9] = array_slice<[1]>;
                libfunc [19] = branch_align;
                libfunc [26] = store_temp<[0]>;
                libfunc [8] = array_snapshot_pop_front<[1]>;
                libfunc [20] = drop<[3]>;
                libfunc [7] = unbox<[1]>;
                libfunc [21] = rename<[1]>;
                libfunc [6] = struct_construct<[5]>;
                libfunc [5] = enum_init<[10], 0>;
                libfunc [27] = store_temp<[10]>;
                libfunc [4] = array_new<[7]>;
                libfunc [22] = const_as_immediate<[15]>;
                libfunc [28] = store_temp<[7]>;
                libfunc [3] = array_append<[7]>;
                libfunc [2] = struct_construct<[6]>;
                libfunc [1] = struct_construct<[9]>;
                libfunc [0] = enum_init<[10], 1>;
                libfunc [23] = const_as_immediate<[16]>;

                [11]() -> ([1]); // 0
                [13]() -> ([2]); // 1
                [24]([2]) -> ([2]); // 2
                [10]([1], [2]) -> ([3]); // 3
                [14]() -> ([4]); // 4
                [24]([4]) -> ([4]); // 5
                [10]([3], [4]) -> ([5]); // 6
                [15]() -> ([6]); // 7
                [24]([6]) -> ([6]); // 8
                [10]([5], [6]) -> ([7]); // 9
                [16]() -> ([8]); // 10
                [24]([8]) -> ([8]); // 11
                [10]([7], [8]) -> ([9]); // 12
                [17]([9]) -> ([10], [11]); // 13
                [18]([10]) -> (); // 14
                [13]() -> ([12]); // 15
                [16]() -> ([13]); // 16
                [25]([11]) -> ([11]); // 17
                [24]([12]) -> ([12]); // 18
                [24]([13]) -> ([13]); // 19
                [9]([0], [11], [12], [13]) { fallthrough([14], [15]) 46([16]) }; // 20
                [19]() -> (); // 21
                [25]([15]) -> ([15]); // 22
                [26]([14]) -> ([14]); // 23
                [8]([15]) { fallthrough([17], [18]) 34([19]) }; // 24
                [19]() -> (); // 25
                [20]([17]) -> (); // 26
                [7]([18]) -> ([20]); // 27
                [21]([20]) -> ([21]); // 28
                [6]([21]) -> ([22]); // 29
                [5]([22]) -> ([23]); // 30
                [26]([14]) -> ([14]); // 31
                [27]([23]) -> ([23]); // 32
                return([14], [23]); // 33
                [19]() -> (); // 34
                [20]([19]) -> (); // 35
                [4]() -> ([24]); // 36
                [22]() -> ([25]); // 37
                [28]([25]) -> ([25]); // 38
                [3]([24], [25]) -> ([26]); // 39
                [2]() -> ([27]); // 40
                [1]([27], [26]) -> ([28]); // 41
                [0]([28]) -> ([29]); // 42
                [26]([14]) -> ([14]); // 43
                [27]([29]) -> ([29]); // 44
                return([14], [29]); // 45
                [19]() -> (); // 46
                [4]() -> ([30]); // 47
                [23]() -> ([31]); // 48
                [28]([31]) -> ([31]); // 49
                [3]([30], [31]) -> ([32]); // 50
                [2]() -> ([33]); // 51
                [1]([33], [32]) -> ([34]); // 52
                [0]([34]) -> ([35]); // 53
                [26]([16]) -> ([16]); // 54
                [27]([35]) -> ([35]); // 55
                return([16], [35]); // 56

                [0]@0([0]: [0]) -> ([0], [10]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let result = run_sierra_program(&program, &[]).return_value;

        assert_eq!(
            result,
            jit_panic!(felt252_str(
                "1637570914057682275393755530660268060279989363"
            ))
        );
    }

    #[test]
    fn run_slice_empty_array() {
        // fn run_test() -> Span<felt252> {
        //     let x: Span<felt252> = array![].span();
        //     x.slice(0, 0)
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [2] = Array<[1]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [7] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
                type [8] = Struct<ut@Tuple, [7], [2]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [11] = Const<[1], 1637570914057682275393755530660268060279989363> [storable: false, drop: false, dup: false, zero_sized: false];
                type [3] = Snapshot<[2]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [5] = Struct<ut@core::array::Span::<core::felt252>, [3]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [6] = Struct<ut@Tuple, [5]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [9] = Enum<ut@core::panics::PanicResult::<(core::array::Span::<core::felt252>,)>, [6], [8]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
                type [10] = Const<[4], 0> [storable: false, drop: false, dup: false, zero_sized: false];
                type [4] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
                type [1] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];

                libfunc [4] = array_new<[1]>;
                libfunc [10] = snapshot_take<[2]>;
                libfunc [11] = drop<[2]>;
                libfunc [12] = const_as_immediate<[10]>;
                libfunc [15] = store_temp<[4]>;
                libfunc [8] = array_slice<[1]>;
                libfunc [13] = branch_align;
                libfunc [7] = struct_construct<[5]>;
                libfunc [6] = struct_construct<[6]>;
                libfunc [5] = enum_init<[9], 0>;
                libfunc [16] = store_temp<[0]>;
                libfunc [17] = store_temp<[9]>;
                libfunc [14] = const_as_immediate<[11]>;
                libfunc [18] = store_temp<[1]>;
                libfunc [3] = array_append<[1]>;
                libfunc [2] = struct_construct<[7]>;
                libfunc [1] = struct_construct<[8]>;
                libfunc [0] = enum_init<[9], 1>;

                [4]() -> ([1]); // 0
                [10]([1]) -> ([2], [3]); // 1
                [11]([2]) -> (); // 2
                [12]() -> ([4]); // 3
                [12]() -> ([5]); // 4
                [15]([4]) -> ([4]); // 5
                [15]([5]) -> ([5]); // 6
                [8]([0], [3], [4], [5]) { fallthrough([6], [7]) 15([8]) }; // 7
                [13]() -> (); // 8
                [7]([7]) -> ([9]); // 9
                [6]([9]) -> ([10]); // 10
                [5]([10]) -> ([11]); // 11
                [16]([6]) -> ([6]); // 12
                [17]([11]) -> ([11]); // 13
                return([6], [11]); // 14
                [13]() -> (); // 15
                [4]() -> ([12]); // 16
                [14]() -> ([13]); // 17
                [18]([13]) -> ([13]); // 18
                [3]([12], [13]) -> ([14]); // 19
                [2]() -> ([15]); // 20
                [1]([15], [14]) -> ([16]); // 21
                [0]([16]) -> ([17]); // 22
                [16]([8]) -> ([8]); // 23
                [17]([17]) -> ([17]); // 24
                return([8], [17]); // 25

                [0]@0([0]: [0]) -> ([0], [9]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let result = run_sierra_program(&program, &[]).return_value;

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
        // mod felt252_span_from_tuple {
        //     pub extern fn span_from_tuple<T>(struct_like: Box<@T>) -> @Array<felt252> nopanic;
        // }
        // fn run_test() -> Array<felt252> {
        //     let span = felt252_span_from_tuple::span_from_tuple(BoxTrait::new(@(10, 20, 30)));
        //     span.clone()
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [4] = Box<[3]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [10] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
                type [21] = Const<[2], 375233589013918064796019> [storable: false, drop: false, dup: false, zero_sized: false];
                type [19] = Box<[2]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [8] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
                type [20] = Enum<ut@core::option::Option::<core::box::Box::<@core::felt252>>, [19], [8]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [5] = Array<[2]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [13] = Struct<ut@Tuple, [5]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [11] = Struct<ut@Tuple, [10], [5]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [14] = Enum<ut@core::panics::PanicResult::<(core::array::Array::<core::felt252>,)>, [13], [11]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [6] = Snapshot<[5]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [7] = Struct<ut@core::array::Span::<core::felt252>, [6]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [9] = Struct<ut@Tuple, [7], [5], [8]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [12] = Enum<ut@core::panics::PanicResult::<(core::array::Span::<core::felt252>, core::array::Array::<core::felt252>, ())>, [9], [11]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [1] = GasBuiltin [storable: true, drop: false, dup: false, zero_sized: false];
                type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
                type [2] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
                type [3] = Struct<ut@Tuple, [2], [2], [2]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [18] = Const<[3], [15], [16], [17]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [17] = Const<[2], 30> [storable: false, drop: false, dup: false, zero_sized: false];
                type [16] = Const<[2], 20> [storable: false, drop: false, dup: false, zero_sized: false];
                type [15] = Const<[2], 10> [storable: false, drop: false, dup: false, zero_sized: false];

                libfunc [10] = disable_ap_tracking;
                libfunc [11] = const_as_box<[18], 0>;
                libfunc [9] = span_from_tuple<[3]>;
                libfunc [8] = array_new<[2]>;
                libfunc [7] = struct_construct<[7]>;
                libfunc [15] = store_temp<[0]>;
                libfunc [16] = store_temp<[1]>;
                libfunc [17] = store_temp<[7]>;
                libfunc [18] = store_temp<[5]>;
                libfunc [6] = function_call<user@[0]>;
                libfunc [5] = enum_match<[12]>;
                libfunc [12] = branch_align;
                libfunc [1] = redeposit_gas;
                libfunc [4] = struct_deconstruct<[9]>;
                libfunc [13] = drop<[7]>;
                libfunc [14] = drop<[8]>;
                libfunc [3] = struct_construct<[13]>;
                libfunc [2] = enum_init<[14], 0>;
                libfunc [19] = store_temp<[14]>;
                libfunc [0] = enum_init<[14], 1>;
                libfunc [33] = withdraw_gas;
                libfunc [32] = struct_deconstruct<[7]>;
                libfunc [34] = enable_ap_tracking;
                libfunc [31] = array_snapshot_pop_front<[2]>;
                libfunc [30] = enum_init<[20], 0>;
                libfunc [39] = store_temp<[6]>;
                libfunc [40] = store_temp<[20]>;
                libfunc [35] = jump;
                libfunc [26] = struct_construct<[8]>;
                libfunc [29] = enum_init<[20], 1>;
                libfunc [28] = enum_match<[20]>;
                libfunc [27] = unbox<[2]>;
                libfunc [36] = rename<[2]>;
                libfunc [41] = store_temp<[2]>;
                libfunc [23] = array_append<[2]>;
                libfunc [25] = struct_construct<[9]>;
                libfunc [24] = enum_init<[12], 0>;
                libfunc [42] = store_temp<[12]>;
                libfunc [37] = drop<[5]>;
                libfunc [38] = const_as_immediate<[21]>;
                libfunc [22] = struct_construct<[10]>;
                libfunc [21] = struct_construct<[11]>;
                libfunc [20] = enum_init<[12], 1>;

                [10]() -> (); // 0
                [11]() -> ([2]); // 1
                [9]([2]) -> ([3]); // 2
                [8]() -> ([4]); // 3
                [7]([3]) -> ([5]); // 4
                [15]([0]) -> ([0]); // 5
                [16]([1]) -> ([1]); // 6
                [17]([5]) -> ([5]); // 7
                [18]([4]) -> ([4]); // 8
                [6]([0], [1], [5], [4]) -> ([6], [7], [8]); // 9
                [5]([8]) { fallthrough([9]) 22([10]) }; // 10
                [12]() -> (); // 11
                [1]([7]) -> ([11]); // 12
                [4]([9]) -> ([12], [13], [14]); // 13
                [13]([12]) -> (); // 14
                [14]([14]) -> (); // 15
                [3]([13]) -> ([15]); // 16
                [2]([15]) -> ([16]); // 17
                [15]([6]) -> ([6]); // 18
                [16]([11]) -> ([11]); // 19
                [19]([16]) -> ([16]); // 20
                return([6], [11], [16]); // 21
                [12]() -> (); // 22
                [1]([7]) -> ([17]); // 23
                [0]([10]) -> ([18]); // 24
                [15]([6]) -> ([6]); // 25
                [16]([17]) -> ([17]); // 26
                [19]([18]) -> ([18]); // 27
                return([6], [17], [18]); // 28
                [10]() -> (); // 29
                [33]([0], [1]) { fallthrough([4], [5]) 78([6], [7]) }; // 30
                [12]() -> (); // 31
                [1]([5]) -> ([8]); // 32
                [32]([2]) -> ([9]); // 33
                [34]() -> (); // 34
                [15]([4]) -> ([4]); // 35
                [16]([8]) -> ([8]); // 36
                [31]([9]) { fallthrough([10], [11]) 45([12]) }; // 37
                [12]() -> (); // 38
                [1]([8]) -> ([13]); // 39
                [30]([11]) -> ([14]); // 40
                [16]([13]) -> ([15]); // 41
                [39]([10]) -> ([16]); // 42
                [40]([14]) -> ([17]); // 43
                [35]() { 52() }; // 44
                [12]() -> (); // 45
                [1]([8]) -> ([18]); // 46
                [26]() -> ([19]); // 47
                [29]([19]) -> ([20]); // 48
                [16]([18]) -> ([15]); // 49
                [39]([12]) -> ([16]); // 50
                [40]([20]) -> ([17]); // 51
                [7]([16]) -> ([21]); // 52
                [28]([17]) { fallthrough([22]) 67([23]) }; // 53
                [12]() -> (); // 54
                [10]() -> (); // 55
                [1]([15]) -> ([24]); // 56
                [27]([22]) -> ([25]); // 57
                [36]([25]) -> ([26]); // 58
                [41]([26]) -> ([26]); // 59
                [23]([3], [26]) -> ([27]); // 60
                [15]([4]) -> ([4]); // 61
                [16]([24]) -> ([24]); // 62
                [17]([21]) -> ([21]); // 63
                [18]([27]) -> ([27]); // 64
                [6]([4], [24], [21], [27]) -> ([28], [29], [30]); // 65
                return([28], [29], [30]); // 66
                [12]() -> (); // 67
                [10]() -> (); // 68
                [14]([23]) -> (); // 69
                [1]([15]) -> ([31]); // 70
                [26]() -> ([32]); // 71
                [25]([21], [3], [32]) -> ([33]); // 72
                [24]([33]) -> ([34]); // 73
                [15]([4]) -> ([4]); // 74
                [16]([31]) -> ([31]); // 75
                [42]([34]) -> ([34]); // 76
                return([4], [31], [34]); // 77
                [12]() -> (); // 78
                [13]([2]) -> (); // 79
                [37]([3]) -> (); // 80
                [1]([7]) -> ([35]); // 81
                [8]() -> ([36]); // 82
                [38]() -> ([37]); // 83
                [41]([37]) -> ([37]); // 84
                [23]([36], [37]) -> ([38]); // 85
                [22]() -> ([39]); // 86
                [21]([39], [38]) -> ([40]); // 87
                [20]([40]) -> ([41]); // 88
                [15]([6]) -> ([6]); // 89
                [16]([35]) -> ([35]); // 90
                [42]([41]) -> ([41]); // 91
                return([6], [35], [41]); // 92

                [1]@0([0]: [0], [1]: [1]) -> ([0], [1], [14]);
                [0]@29([0]: [0], [1]: [1], [2]: [7], [3]: [5]) -> ([0], [1], [12]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let result = run_sierra_program(&program, &[]).return_value;

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
        // mod tuple_span_from_tuple {
        //     pub extern fn span_from_tuple<T>(
        //         struct_like: Box<@T>
        //     ) -> @Array<(felt252, felt252, felt252)> nopanic;
        // }
        // fn run_test() {
        //     let multi_tuple = ((10, 20, 30), (40, 50, 60), (70, 80, 90));
        //     let span = tuple_span_from_tuple::span_from_tuple(BoxTrait::new(@multi_tuple));
        //     assert!(*span[0] == (10, 20, 30));
        //     assert!(*span[1] == (40, 50, 60));
        //     assert!(*span[2] == (70, 80, 90));
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [18] = Array<[17]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [2] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
                type [10] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
                type [19] = Struct<ut@core::byte_array::ByteArray, [18], [2], [10]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [89] = Uninitialized<[19]> [storable: false, drop: true, dup: false, zero_sized: false];
                type [66] = Const<[2], 573087285299505011920718992710461799> [storable: false, drop: false, dup: false, zero_sized: false];
                type [65] = Const<[29], [64]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [63] = Const<[29], [62]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [64] = Const<[28], 1329227995784915872903807060280344576> [storable: false, drop: false, dup: false, zero_sized: false];
                type [62] = Const<[28], 5192296858534827628530496329220096> [storable: false, drop: false, dup: false, zero_sized: false];
                type [61] = Const<[29], [60]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [59] = Const<[29], [58]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [60] = Const<[28], 20282409603651670423947251286016> [storable: false, drop: false, dup: false, zero_sized: false];
                type [58] = Const<[28], 79228162514264337593543950336> [storable: false, drop: false, dup: false, zero_sized: false];
                type [57] = Const<[29], [56]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [55] = Const<[29], [54]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [56] = Const<[28], 309485009821345068724781056> [storable: false, drop: false, dup: false, zero_sized: false];
                type [54] = Const<[28], 1208925819614629174706176> [storable: false, drop: false, dup: false, zero_sized: false];
                type [53] = Const<[29], [52]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [51] = Const<[29], [50]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [52] = Const<[28], 4722366482869645213696> [storable: false, drop: false, dup: false, zero_sized: false];
                type [50] = Const<[28], 18446744073709551616> [storable: false, drop: false, dup: false, zero_sized: false];
                type [49] = Const<[29], [48]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [47] = Const<[29], [46]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [48] = Const<[28], 72057594037927936> [storable: false, drop: false, dup: false, zero_sized: false];
                type [46] = Const<[28], 281474976710656> [storable: false, drop: false, dup: false, zero_sized: false];
                type [45] = Const<[29], [44]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [43] = Const<[29], [42]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [44] = Const<[28], 1099511627776> [storable: false, drop: false, dup: false, zero_sized: false];
                type [42] = Const<[28], 4294967296> [storable: false, drop: false, dup: false, zero_sized: false];
                type [41] = Const<[29], [40]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [39] = Const<[29], [38]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [40] = Const<[28], 16777216> [storable: false, drop: false, dup: false, zero_sized: false];
                type [38] = Const<[28], 65536> [storable: false, drop: false, dup: false, zero_sized: false];
                type [37] = Const<[29], [36]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [35] = Const<[29], [34]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [36] = Const<[28], 256> [storable: false, drop: false, dup: false, zero_sized: false];
                type [34] = Const<[28], 1> [storable: false, drop: false, dup: false, zero_sized: false];
                type [11] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
                type [33] = Enum<ut@index_enum_type<16>, [11], [11], [11], [11], [11], [11], [11], [11], [11], [11], [11], [11], [11], [11], [11], [11]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [32] = BoundedInt<0, 15> [storable: true, drop: true, dup: true, zero_sized: false];
                type [102] = Const<[2], 375233589013918064796019> [storable: false, drop: false, dup: false, zero_sized: false];
                type [101] = Box<[17]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [75] = Const<[2], 155785504323917466144735657540098748279> [storable: false, drop: false, dup: false, zero_sized: false];
                type [72] = Const<[2], 29721761890975875353235833581453094220424382983267374> [storable: false, drop: false, dup: false, zero_sized: false];
                type [71] = Const<[2], 155785504329508738615720351733824384887> [storable: false, drop: false, dup: false, zero_sized: false];
                type [70] = Const<[2], 340282366920938463463374607431768211456> [storable: false, drop: false, dup: false, zero_sized: false];
                type [28] = u128 [storable: true, drop: true, dup: true, zero_sized: false];
                type [29] = NonZero<[28]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [30] = Struct<ut@Tuple, [29]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [13] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
                type [14] = Array<[2]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [15] = Struct<ut@Tuple, [13], [14]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [31] = Enum<ut@core::panics::PanicResult::<(core::zeroable::NonZero::<core::integer::u128>,)>, [30], [15]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [69] = Const<[28], 0> [storable: false, drop: false, dup: false, zero_sized: false];
                type [68] = Const<[10], 16> [storable: false, drop: false, dup: false, zero_sized: false];
                type [27] = NonZero<[10]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [100] = Const<[2], 815193472734514792947287325237294> [storable: false, drop: false, dup: false, zero_sized: false];
                type [99] = Const<[2], 172180977190876322177717838039515195832848434314566438920480792630592879904> [storable: false, drop: false, dup: false, zero_sized: false];
                type [98] = Const<[2], 815431157222112926192301970645038> [storable: false, drop: false, dup: false, zero_sized: false];
                type [97] = Const<[2], 172180977190876322177717838039515195832848434314566438920480793730104507680> [storable: false, drop: false, dup: false, zero_sized: false];
                type [96] = Const<[2], 1637570914057682275393755530660268060279989363> [storable: false, drop: false, dup: false, zero_sized: false];
                type [25] = Struct<ut@Tuple, [14], [11]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [26] = Enum<ut@core::panics::PanicResult::<(core::array::Array::<core::felt252>, ())>, [25], [15]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [23] = Snapshot<[18]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [24] = Struct<ut@core::array::Span::<core::bytes_31::bytes31>, [23]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [22] = Snapshot<[19]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [95] = Const<[2], 1997209042069643135709344952807065910992472029923670688473712229447419591075> [storable: false, drop: false, dup: false, zero_sized: false];
                type [94] = Const<[10], 14> [storable: false, drop: false, dup: false, zero_sized: false];
                type [93] = Const<[2], 815668841709711059437316616052782> [storable: false, drop: false, dup: false, zero_sized: false];
                type [20] = Struct<ut@Tuple, [19], [11]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [21] = Enum<ut@core::panics::PanicResult::<(core::byte_array::ByteArray, ())>, [20], [15]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [67] = Const<[10], 31> [storable: false, drop: false, dup: false, zero_sized: false];
                type [92] = Const<[2], 172180977190876322177717838039515195832848434314566438920480794829616135456> [storable: false, drop: false, dup: false, zero_sized: false];
                type [74] = Const<[10], 0> [storable: false, drop: false, dup: false, zero_sized: false];
                type [73] = Const<[2], 0> [storable: false, drop: false, dup: false, zero_sized: false];
                type [17] = bytes31 [storable: true, drop: true, dup: true, zero_sized: false];
                type [12] = Struct<ut@Tuple, [11]> [storable: true, drop: true, dup: true, zero_sized: true];
                type [16] = Enum<ut@core::panics::PanicResult::<((),)>, [12], [15]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [86] = Const<[2], 90> [storable: false, drop: false, dup: false, zero_sized: false];
                type [85] = Const<[2], 80> [storable: false, drop: false, dup: false, zero_sized: false];
                type [84] = Const<[2], 70> [storable: false, drop: false, dup: false, zero_sized: false];
                type [91] = Const<[10], 2> [storable: false, drop: false, dup: false, zero_sized: false];
                type [82] = Const<[2], 60> [storable: false, drop: false, dup: false, zero_sized: false];
                type [81] = Const<[2], 50> [storable: false, drop: false, dup: false, zero_sized: false];
                type [80] = Const<[2], 40> [storable: false, drop: false, dup: false, zero_sized: false];
                type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
                type [90] = Const<[10], 1> [storable: false, drop: false, dup: false, zero_sized: false];
                type [9] = NonZero<[2]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [78] = Const<[2], 30> [storable: false, drop: false, dup: false, zero_sized: false];
                type [77] = Const<[2], 20> [storable: false, drop: false, dup: false, zero_sized: false];
                type [76] = Const<[2], 10> [storable: false, drop: false, dup: false, zero_sized: false];
                type [1] = GasBuiltin [storable: true, drop: false, dup: false, zero_sized: false];
                type [3] = Struct<ut@Tuple, [2], [2], [2]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [8] = Box<[3]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [4] = Struct<ut@Tuple, [3], [3], [3]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [6] = Array<[3]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [7] = Snapshot<[6]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [88] = Const<[4], [79], [83], [87]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [87] = Const<[3], [84], [85], [86]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [83] = Const<[3], [80], [81], [82]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [79] = Const<[3], [76], [77], [78]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [5] = Box<[4]> [storable: true, drop: true, dup: true, zero_sized: false];

                libfunc [104] = alloc_local<[19]>;
                libfunc [105] = finalize_locals;
                libfunc [106] = disable_ap_tracking;
                libfunc [107] = const_as_box<[88], 0>;
                libfunc [103] = span_from_tuple<[4]>;
                libfunc [144] = store_temp<[7]>;
                libfunc [108] = dup<[7]>;
                libfunc [102] = array_snapshot_pop_front<[3]>;
                libfunc [35] = branch_align;
                libfunc [109] = drop<[7]>;
                libfunc [5] = redeposit_gas;
                libfunc [100] = unbox<[3]>;
                libfunc [110] = rename<[3]>;
                libfunc [111] = snapshot_take<[3]>;
                libfunc [112] = drop<[3]>;
                libfunc [98] = struct_deconstruct<[3]>;
                libfunc [113] = const_as_immediate<[76]>;
                libfunc [114] = const_as_immediate<[77]>;
                libfunc [115] = const_as_immediate<[78]>;
                libfunc [99] = struct_construct<[3]>;
                libfunc [116] = rename<[2]>;
                libfunc [60] = store_temp<[2]>;
                libfunc [97] = felt252_sub;
                libfunc [145] = store_temp<[1]>;
                libfunc [96] = felt252_is_zero;
                libfunc [117] = drop<[89]>;
                libfunc [118] = const_as_immediate<[90]>;
                libfunc [86] = store_temp<[10]>;
                libfunc [101] = array_get<[3]>;
                libfunc [146] = store_temp<[8]>;
                libfunc [119] = const_as_immediate<[80]>;
                libfunc [120] = const_as_immediate<[81]>;
                libfunc [121] = const_as_immediate<[82]>;
                libfunc [122] = enable_ap_tracking;
                libfunc [57] = store_temp<[0]>;
                libfunc [123] = const_as_immediate<[91]>;
                libfunc [124] = const_as_immediate<[84]>;
                libfunc [125] = const_as_immediate<[85]>;
                libfunc [126] = const_as_immediate<[86]>;
                libfunc [20] = struct_construct<[11]>;
                libfunc [95] = struct_construct<[12]>;
                libfunc [94] = enum_init<[16], 0>;
                libfunc [147] = store_temp<[16]>;
                libfunc [127] = drop<[9]>;
                libfunc [39] = jump;
                libfunc [70] = drop<[2]>;
                libfunc [93] = array_new<[17]>;
                libfunc [82] = const_as_immediate<[73]>;
                libfunc [83] = const_as_immediate<[74]>;
                libfunc [128] = const_as_immediate<[92]>;
                libfunc [73] = const_as_immediate<[67]>;
                libfunc [21] = struct_construct<[19]>;
                libfunc [89] = store_temp<[19]>;
                libfunc [16] = function_call<user@[1]>;
                libfunc [15] = enum_match<[21]>;
                libfunc [14] = struct_deconstruct<[20]>;
                libfunc [37] = drop<[11]>;
                libfunc [129] = const_as_immediate<[93]>;
                libfunc [130] = const_as_immediate<[94]>;
                libfunc [4] = array_new<[2]>;
                libfunc [131] = const_as_immediate<[95]>;
                libfunc [3] = array_append<[2]>;
                libfunc [148] = store_local<[19]>;
                libfunc [132] = snapshot_take<[19]>;
                libfunc [133] = drop<[19]>;
                libfunc [134] = dup<[22]>;
                libfunc [8] = struct_snapshot_deconstruct<[19]>;
                libfunc [71] = drop<[10]>;
                libfunc [135] = dup<[23]>;
                libfunc [13] = array_len<[17]>;
                libfunc [7] = u32_to_felt252;
                libfunc [12] = struct_construct<[24]>;
                libfunc [149] = store_temp<[24]>;
                libfunc [88] = store_temp<[14]>;
                libfunc [11] = function_call<user@[0]>;
                libfunc [10] = enum_match<[26]>;
                libfunc [9] = struct_deconstruct<[25]>;
                libfunc [136] = drop<[23]>;
                libfunc [137] = rename<[10]>;
                libfunc [138] = drop<[22]>;
                libfunc [6] = struct_deconstruct<[15]>;
                libfunc [78] = drop<[13]>;
                libfunc [2] = struct_construct<[13]>;
                libfunc [1] = struct_construct<[15]>;
                libfunc [0] = enum_init<[16], 1>;
                libfunc [139] = const_as_immediate<[96]>;
                libfunc [140] = const_as_immediate<[97]>;
                libfunc [141] = const_as_immediate<[98]>;
                libfunc [142] = const_as_immediate<[99]>;
                libfunc [143] = const_as_immediate<[100]>;
                libfunc [69] = dup<[10]>;
                libfunc [62] = u32_is_zero;
                libfunc [19] = struct_construct<[20]>;
                libfunc [18] = enum_init<[21], 0>;
                libfunc [85] = store_temp<[21]>;
                libfunc [72] = drop<[27]>;
                libfunc [65] = struct_deconstruct<[19]>;
                libfunc [24] = u32_overflowing_add;
                libfunc [61] = u32_overflowing_sub;
                libfunc [68] = u32_eq;
                libfunc [74] = const_as_immediate<[68]>;
                libfunc [66] = u128s_from_felt252;
                libfunc [75] = const_as_immediate<[69]>;
                libfunc [87] = store_temp<[28]>;
                libfunc [29] = function_call<user@[2]>;
                libfunc [28] = enum_match<[31]>;
                libfunc [27] = struct_deconstruct<[30]>;
                libfunc [67] = u128_safe_divmod;
                libfunc [25] = u128_to_felt252;
                libfunc [76] = const_as_immediate<[70]>;
                libfunc [23] = felt252_mul;
                libfunc [22] = felt252_add;
                libfunc [26] = unwrap_non_zero<[28]>;
                libfunc [77] = drop<[18]>;
                libfunc [79] = const_as_immediate<[71]>;
                libfunc [64] = bytes31_try_from_felt252;
                libfunc [63] = array_append<[17]>;
                libfunc [80] = const_as_immediate<[72]>;
                libfunc [90] = rename<[0]>;
                libfunc [91] = rename<[14]>;
                libfunc [81] = drop<[28]>;
                libfunc [17] = enum_init<[21], 1>;
                libfunc [92] = rename<[19]>;
                libfunc [84] = const_as_immediate<[75]>;
                libfunc [157] = withdraw_gas;
                libfunc [156] = struct_deconstruct<[24]>;
                libfunc [155] = array_snapshot_pop_front<[17]>;
                libfunc [154] = unbox<[17]>;
                libfunc [158] = rename<[17]>;
                libfunc [153] = bytes31_to_felt252;
                libfunc [152] = struct_construct<[25]>;
                libfunc [151] = enum_init<[26], 0>;
                libfunc [162] = store_temp<[26]>;
                libfunc [159] = drop<[24]>;
                libfunc [160] = drop<[14]>;
                libfunc [161] = const_as_immediate<[102]>;
                libfunc [150] = enum_init<[26], 1>;
                libfunc [34] = downcast<[10], [32]>;
                libfunc [36] = enum_from_bounded_int<[33]>;
                libfunc [56] = store_temp<[33]>;
                libfunc [33] = enum_match<[33]>;
                libfunc [38] = const_as_immediate<[35]>;
                libfunc [58] = store_temp<[29]>;
                libfunc [40] = const_as_immediate<[37]>;
                libfunc [41] = const_as_immediate<[39]>;
                libfunc [42] = const_as_immediate<[41]>;
                libfunc [43] = const_as_immediate<[43]>;
                libfunc [44] = const_as_immediate<[45]>;
                libfunc [45] = const_as_immediate<[47]>;
                libfunc [46] = const_as_immediate<[49]>;
                libfunc [47] = const_as_immediate<[51]>;
                libfunc [48] = const_as_immediate<[53]>;
                libfunc [49] = const_as_immediate<[55]>;
                libfunc [50] = const_as_immediate<[57]>;
                libfunc [51] = const_as_immediate<[59]>;
                libfunc [52] = const_as_immediate<[61]>;
                libfunc [53] = const_as_immediate<[63]>;
                libfunc [54] = const_as_immediate<[65]>;
                libfunc [32] = struct_construct<[30]>;
                libfunc [31] = enum_init<[31], 0>;
                libfunc [59] = store_temp<[31]>;
                libfunc [55] = const_as_immediate<[66]>;
                libfunc [30] = enum_init<[31], 1>;

                [104]() -> ([3]); // 0
                [104]() -> ([5]); // 1
                [104]() -> ([7]); // 2
                [105]() -> (); // 3
                [106]() -> (); // 4
                [107]() -> ([8]); // 5
                [103]([8]) -> ([9]); // 6
                [144]([9]) -> ([9]); // 7
                [108]([9]) -> ([9], [10]); // 8
                [102]([10]) { fallthrough([11], [12]) 586([13]) }; // 9
                [35]() -> (); // 10
                [109]([11]) -> (); // 11
                [5]([1]) -> ([14]); // 12
                [100]([12]) -> ([15]); // 13
                [110]([15]) -> ([16]); // 14
                [111]([16]) -> ([17], [18]); // 15
                [112]([17]) -> (); // 16
                [98]([18]) -> ([19], [20], [21]); // 17
                [113]() -> ([22]); // 18
                [114]() -> ([23]); // 19
                [115]() -> ([24]); // 20
                [99]([22], [23], [24]) -> ([25]); // 21
                [111]([25]) -> ([26], [27]); // 22
                [112]([26]) -> (); // 23
                [98]([27]) -> ([28], [29], [30]); // 24
                [116]([19]) -> ([31]); // 25
                [116]([28]) -> ([32]); // 26
                [60]([31]) -> ([31]); // 27
                [60]([32]) -> ([32]); // 28
                [97]([31], [32]) -> ([33]); // 29
                [60]([33]) -> ([33]); // 30
                [145]([14]) -> ([14]); // 31
                [60]([20]) -> ([20]); // 32
                [60]([21]) -> ([21]); // 33
                [60]([29]) -> ([29]); // 34
                [60]([30]) -> ([30]); // 35
                [96]([33]) { fallthrough() 472([34]) }; // 36
                [35]() -> (); // 37
                [5]([14]) -> ([35]); // 38
                [116]([20]) -> ([36]); // 39
                [116]([29]) -> ([37]); // 40
                [97]([36], [37]) -> ([38]); // 41
                [60]([38]) -> ([38]); // 42
                [145]([35]) -> ([35]); // 43
                [96]([38]) { fallthrough() 462([39]) }; // 44
                [35]() -> (); // 45
                [5]([35]) -> ([40]); // 46
                [116]([21]) -> ([41]); // 47
                [116]([30]) -> ([42]); // 48
                [97]([41], [42]) -> ([43]); // 49
                [60]([43]) -> ([43]); // 50
                [145]([40]) -> ([40]); // 51
                [96]([43]) { fallthrough() 454([44]) }; // 52
                [35]() -> (); // 53
                [117]([3]) -> (); // 54
                [5]([40]) -> ([45]); // 55
                [118]() -> ([46]); // 56
                [108]([9]) -> ([9], [47]); // 57
                [86]([46]) -> ([46]); // 58
                [145]([45]) -> ([45]); // 59
                [101]([0], [47], [46]) { fallthrough([48], [49]) 438([50]) }; // 60
                [35]() -> (); // 61
                [5]([45]) -> ([51]); // 62
                [146]([49]) -> ([49]); // 63
                [100]([49]) -> ([52]); // 64
                [110]([52]) -> ([53]); // 65
                [111]([53]) -> ([54], [55]); // 66
                [112]([54]) -> (); // 67
                [98]([55]) -> ([56], [57], [58]); // 68
                [119]() -> ([59]); // 69
                [120]() -> ([60]); // 70
                [121]() -> ([61]); // 71
                [99]([59], [60], [61]) -> ([62]); // 72
                [111]([62]) -> ([63], [64]); // 73
                [112]([63]) -> (); // 74
                [98]([64]) -> ([65], [66], [67]); // 75
                [116]([56]) -> ([68]); // 76
                [116]([65]) -> ([69]); // 77
                [60]([68]) -> ([68]); // 78
                [60]([69]) -> ([69]); // 79
                [97]([68], [69]) -> ([70]); // 80
                [122]() -> (); // 81
                [60]([70]) -> ([70]); // 82
                [57]([48]) -> ([48]); // 83
                [145]([51]) -> ([51]); // 84
                [60]([57]) -> ([57]); // 85
                [60]([58]) -> ([58]); // 86
                [60]([66]) -> ([66]); // 87
                [60]([67]) -> ([67]); // 88
                [96]([70]) { fallthrough() 324([71]) }; // 89
                [35]() -> (); // 90
                [5]([51]) -> ([72]); // 91
                [116]([57]) -> ([73]); // 92
                [116]([66]) -> ([74]); // 93
                [97]([73], [74]) -> ([75]); // 94
                [60]([75]) -> ([75]); // 95
                [145]([72]) -> ([72]); // 96
                [96]([75]) { fallthrough() 315([76]) }; // 97
                [35]() -> (); // 98
                [5]([72]) -> ([77]); // 99
                [116]([58]) -> ([78]); // 100
                [116]([67]) -> ([79]); // 101
                [97]([78], [79]) -> ([80]); // 102
                [60]([80]) -> ([80]); // 103
                [145]([77]) -> ([77]); // 104
                [96]([80]) { fallthrough() 308([81]) }; // 105
                [35]() -> (); // 106
                [106]() -> (); // 107
                [117]([5]) -> (); // 108
                [5]([77]) -> ([82]); // 109
                [123]() -> ([83]); // 110
                [86]([83]) -> ([83]); // 111
                [145]([82]) -> ([82]); // 112
                [101]([48], [9], [83]) { fallthrough([84], [85]) 294([86]) }; // 113
                [35]() -> (); // 114
                [5]([82]) -> ([87]); // 115
                [146]([85]) -> ([85]); // 116
                [100]([85]) -> ([88]); // 117
                [110]([88]) -> ([89]); // 118
                [111]([89]) -> ([90], [91]); // 119
                [112]([90]) -> (); // 120
                [98]([91]) -> ([92], [93], [94]); // 121
                [124]() -> ([95]); // 122
                [125]() -> ([96]); // 123
                [126]() -> ([97]); // 124
                [99]([95], [96], [97]) -> ([98]); // 125
                [111]([98]) -> ([99], [100]); // 126
                [112]([99]) -> (); // 127
                [98]([100]) -> ([101], [102], [103]); // 128
                [116]([92]) -> ([104]); // 129
                [116]([101]) -> ([105]); // 130
                [60]([104]) -> ([104]); // 131
                [60]([105]) -> ([105]); // 132
                [97]([104], [105]) -> ([106]); // 133
                [122]() -> (); // 134
                [60]([106]) -> ([106]); // 135
                [57]([84]) -> ([84]); // 136
                [145]([87]) -> ([87]); // 137
                [60]([93]) -> ([93]); // 138
                [60]([94]) -> ([94]); // 139
                [60]([102]) -> ([102]); // 140
                [60]([103]) -> ([103]); // 141
                [96]([106]) { fallthrough() 182([107]) }; // 142
                [35]() -> (); // 143
                [5]([87]) -> ([108]); // 144
                [116]([93]) -> ([109]); // 145
                [116]([102]) -> ([110]); // 146
                [97]([109], [110]) -> ([111]); // 147
                [60]([111]) -> ([111]); // 148
                [145]([108]) -> ([108]); // 149
                [96]([111]) { fallthrough() 175([112]) }; // 150
                [35]() -> (); // 151
                [5]([108]) -> ([113]); // 152
                [116]([94]) -> ([114]); // 153
                [116]([103]) -> ([115]); // 154
                [97]([114], [115]) -> ([116]); // 155
                [60]([116]) -> ([116]); // 156
                [145]([113]) -> ([113]); // 157
                [96]([116]) { fallthrough() 170([117]) }; // 158
                [35]() -> (); // 159
                [106]() -> (); // 160
                [117]([7]) -> (); // 161
                [5]([113]) -> ([118]); // 162
                [20]() -> ([119]); // 163
                [95]([119]) -> ([120]); // 164
                [94]([120]) -> ([121]); // 165
                [57]([84]) -> ([84]); // 166
                [145]([118]) -> ([118]); // 167
                [147]([121]) -> ([121]); // 168
                return([84], [118], [121]); // 169
                [35]() -> (); // 170
                [127]([117]) -> (); // 171
                [5]([113]) -> ([122]); // 172
                [145]([122]) -> ([123]); // 173
                [39]() { 190() }; // 174
                [35]() -> (); // 175
                [127]([112]) -> (); // 176
                [70]([94]) -> (); // 177
                [70]([103]) -> (); // 178
                [5]([108]) -> ([124]); // 179
                [145]([124]) -> ([123]); // 180
                [39]() { 190() }; // 181
                [35]() -> (); // 182
                [127]([107]) -> (); // 183
                [70]([94]) -> (); // 184
                [70]([103]) -> (); // 185
                [70]([93]) -> (); // 186
                [70]([102]) -> (); // 187
                [5]([87]) -> ([125]); // 188
                [145]([125]) -> ([123]); // 189
                [106]() -> (); // 190
                [93]() -> ([126]); // 191
                [82]() -> ([127]); // 192
                [83]() -> ([128]); // 193
                [128]() -> ([129]); // 194
                [73]() -> ([130]); // 195
                [21]([126], [127], [128]) -> ([131]); // 196
                [57]([84]) -> ([84]); // 197
                [89]([131]) -> ([131]); // 198
                [60]([129]) -> ([129]); // 199
                [86]([130]) -> ([130]); // 200
                [16]([84], [131], [129], [130]) -> ([132], [133]); // 201
                [15]([133]) { fallthrough([134]) 286([135]) }; // 202
                [35]() -> (); // 203
                [5]([123]) -> ([136]); // 204
                [14]([134]) -> ([137], [138]); // 205
                [37]([138]) -> (); // 206
                [129]() -> ([139]); // 207
                [130]() -> ([140]); // 208
                [57]([132]) -> ([132]); // 209
                [89]([137]) -> ([137]); // 210
                [60]([139]) -> ([139]); // 211
                [86]([140]) -> ([140]); // 212
                [16]([132], [137], [139], [140]) -> ([141], [142]); // 213
                [145]([136]) -> ([136]); // 214
                [15]([142]) { fallthrough([143]) 278([144]) }; // 215
                [35]() -> (); // 216
                [5]([136]) -> ([145]); // 217
                [4]() -> ([146]); // 218
                [131]() -> ([147]); // 219
                [60]([147]) -> ([147]); // 220
                [3]([146], [147]) -> ([148]); // 221
                [14]([143]) -> ([6], [149]); // 222
                [37]([149]) -> (); // 223
                [148]([7], [6]) -> ([6]); // 224
                [132]([6]) -> ([150], [151]); // 225
                [133]([150]) -> (); // 226
                [134]([151]) -> ([151], [152]); // 227
                [8]([152]) -> ([153], [154], [155]); // 228
                [70]([154]) -> (); // 229
                [71]([155]) -> (); // 230
                [135]([153]) -> ([153], [156]); // 231
                [13]([156]) -> ([157]); // 232
                [7]([157]) -> ([158]); // 233
                [60]([158]) -> ([158]); // 234
                [3]([148], [158]) -> ([159]); // 235
                [12]([153]) -> ([160]); // 236
                [57]([141]) -> ([141]); // 237
                [145]([145]) -> ([145]); // 238
                [149]([160]) -> ([160]); // 239
                [88]([159]) -> ([159]); // 240
                [11]([141], [145], [160], [159]) -> ([161], [162], [163]); // 241
                [122]() -> (); // 242
                [10]([163]) { fallthrough([164]) 263([165]) }; // 243
                [35]() -> (); // 244
                [5]([162]) -> ([166]); // 245
                [9]([164]) -> ([167], [168]); // 246
                [37]([168]) -> (); // 247
                [134]([151]) -> ([151], [169]); // 248
                [8]([169]) -> ([170], [171], [172]); // 249
                [136]([170]) -> (); // 250
                [71]([172]) -> (); // 251
                [116]([171]) -> ([173]); // 252
                [3]([167], [173]) -> ([174]); // 253
                [8]([151]) -> ([175], [176], [177]); // 254
                [136]([175]) -> (); // 255
                [70]([176]) -> (); // 256
                [137]([177]) -> ([178]); // 257
                [7]([178]) -> ([179]); // 258
                [3]([174], [179]) -> ([180]); // 259
                [145]([166]) -> ([181]); // 260
                [88]([180]) -> ([182]); // 261
                [39]() { 270() }; // 262
                [35]() -> (); // 263
                [138]([151]) -> (); // 264
                [5]([162]) -> ([183]); // 265
                [6]([165]) -> ([184], [185]); // 266
                [78]([184]) -> (); // 267
                [145]([183]) -> ([181]); // 268
                [88]([185]) -> ([182]); // 269
                [106]() -> (); // 270
                [2]() -> ([186]); // 271
                [1]([186], [182]) -> ([187]); // 272
                [0]([187]) -> ([188]); // 273
                [57]([161]) -> ([161]); // 274
                [145]([181]) -> ([181]); // 275
                [147]([188]) -> ([188]); // 276
                return([161], [181], [188]); // 277
                [35]() -> (); // 278
                [117]([7]) -> (); // 279
                [5]([136]) -> ([189]); // 280
                [0]([144]) -> ([190]); // 281
                [57]([141]) -> ([141]); // 282
                [145]([189]) -> ([189]); // 283
                [147]([190]) -> ([190]); // 284
                return([141], [189], [190]); // 285
                [35]() -> (); // 286
                [117]([7]) -> (); // 287
                [5]([123]) -> ([191]); // 288
                [0]([135]) -> ([192]); // 289
                [57]([132]) -> ([132]); // 290
                [145]([191]) -> ([191]); // 291
                [147]([192]) -> ([192]); // 292
                return([132], [191], [192]); // 293
                [35]() -> (); // 294
                [117]([7]) -> (); // 295
                [5]([82]) -> ([193]); // 296
                [4]() -> ([194]); // 297
                [139]() -> ([195]); // 298
                [60]([195]) -> ([195]); // 299
                [3]([194], [195]) -> ([196]); // 300
                [2]() -> ([197]); // 301
                [1]([197], [196]) -> ([198]); // 302
                [0]([198]) -> ([199]); // 303
                [57]([86]) -> ([86]); // 304
                [145]([193]) -> ([193]); // 305
                [147]([199]) -> ([199]); // 306
                return([86], [193], [199]); // 307
                [35]() -> (); // 308
                [127]([81]) -> (); // 309
                [109]([9]) -> (); // 310
                [117]([7]) -> (); // 311
                [5]([77]) -> ([200]); // 312
                [145]([200]) -> ([201]); // 313
                [39]() { 334() }; // 314
                [35]() -> (); // 315
                [127]([76]) -> (); // 316
                [109]([9]) -> (); // 317
                [117]([7]) -> (); // 318
                [70]([58]) -> (); // 319
                [70]([67]) -> (); // 320
                [5]([72]) -> ([202]); // 321
                [145]([202]) -> ([201]); // 322
                [39]() { 334() }; // 323
                [35]() -> (); // 324
                [127]([71]) -> (); // 325
                [109]([9]) -> (); // 326
                [117]([7]) -> (); // 327
                [70]([58]) -> (); // 328
                [70]([67]) -> (); // 329
                [70]([57]) -> (); // 330
                [70]([66]) -> (); // 331
                [5]([51]) -> ([203]); // 332
                [145]([203]) -> ([201]); // 333
                [106]() -> (); // 334
                [93]() -> ([204]); // 335
                [82]() -> ([205]); // 336
                [83]() -> ([206]); // 337
                [140]() -> ([207]); // 338
                [73]() -> ([208]); // 339
                [21]([204], [205], [206]) -> ([209]); // 340
                [57]([48]) -> ([48]); // 341
                [89]([209]) -> ([209]); // 342
                [60]([207]) -> ([207]); // 343
                [86]([208]) -> ([208]); // 344
                [16]([48], [209], [207], [208]) -> ([210], [211]); // 345
                [15]([211]) { fallthrough([212]) 430([213]) }; // 346
                [35]() -> (); // 347
                [5]([201]) -> ([214]); // 348
                [14]([212]) -> ([215], [216]); // 349
                [37]([216]) -> (); // 350
                [141]() -> ([217]); // 351
                [130]() -> ([218]); // 352
                [57]([210]) -> ([210]); // 353
                [89]([215]) -> ([215]); // 354
                [60]([217]) -> ([217]); // 355
                [86]([218]) -> ([218]); // 356
                [16]([210], [215], [217], [218]) -> ([219], [220]); // 357
                [145]([214]) -> ([214]); // 358
                [15]([220]) { fallthrough([221]) 422([222]) }; // 359
                [35]() -> (); // 360
                [5]([214]) -> ([223]); // 361
                [4]() -> ([224]); // 362
                [131]() -> ([225]); // 363
                [60]([225]) -> ([225]); // 364
                [3]([224], [225]) -> ([226]); // 365
                [14]([221]) -> ([4], [227]); // 366
                [37]([227]) -> (); // 367
                [148]([5], [4]) -> ([4]); // 368
                [132]([4]) -> ([228], [229]); // 369
                [133]([228]) -> (); // 370
                [134]([229]) -> ([229], [230]); // 371
                [8]([230]) -> ([231], [232], [233]); // 372
                [70]([232]) -> (); // 373
                [71]([233]) -> (); // 374
                [135]([231]) -> ([231], [234]); // 375
                [13]([234]) -> ([235]); // 376
                [7]([235]) -> ([236]); // 377
                [60]([236]) -> ([236]); // 378
                [3]([226], [236]) -> ([237]); // 379
                [12]([231]) -> ([238]); // 380
                [57]([219]) -> ([219]); // 381
                [145]([223]) -> ([223]); // 382
                [149]([238]) -> ([238]); // 383
                [88]([237]) -> ([237]); // 384
                [11]([219], [223], [238], [237]) -> ([239], [240], [241]); // 385
                [122]() -> (); // 386
                [10]([241]) { fallthrough([242]) 407([243]) }; // 387
                [35]() -> (); // 388
                [5]([240]) -> ([244]); // 389
                [9]([242]) -> ([245], [246]); // 390
                [37]([246]) -> (); // 391
                [134]([229]) -> ([229], [247]); // 392
                [8]([247]) -> ([248], [249], [250]); // 393
                [136]([248]) -> (); // 394
                [71]([250]) -> (); // 395
                [116]([249]) -> ([251]); // 396
                [3]([245], [251]) -> ([252]); // 397
                [8]([229]) -> ([253], [254], [255]); // 398
                [136]([253]) -> (); // 399
                [70]([254]) -> (); // 400
                [137]([255]) -> ([256]); // 401
                [7]([256]) -> ([257]); // 402
                [3]([252], [257]) -> ([258]); // 403
                [145]([244]) -> ([259]); // 404
                [88]([258]) -> ([260]); // 405
                [39]() { 414() }; // 406
                [35]() -> (); // 407
                [138]([229]) -> (); // 408
                [5]([240]) -> ([261]); // 409
                [6]([243]) -> ([262], [263]); // 410
                [78]([262]) -> (); // 411
                [145]([261]) -> ([259]); // 412
                [88]([263]) -> ([260]); // 413
                [106]() -> (); // 414
                [2]() -> ([264]); // 415
                [1]([264], [260]) -> ([265]); // 416
                [0]([265]) -> ([266]); // 417
                [57]([239]) -> ([239]); // 418
                [145]([259]) -> ([259]); // 419
                [147]([266]) -> ([266]); // 420
                return([239], [259], [266]); // 421
                [35]() -> (); // 422
                [117]([5]) -> (); // 423
                [5]([214]) -> ([267]); // 424
                [0]([222]) -> ([268]); // 425
                [57]([219]) -> ([219]); // 426
                [145]([267]) -> ([267]); // 427
                [147]([268]) -> ([268]); // 428
                return([219], [267], [268]); // 429
                [35]() -> (); // 430
                [117]([5]) -> (); // 431
                [5]([201]) -> ([269]); // 432
                [0]([213]) -> ([270]); // 433
                [57]([210]) -> ([210]); // 434
                [145]([269]) -> ([269]); // 435
                [147]([270]) -> ([270]); // 436
                return([210], [269], [270]); // 437
                [35]() -> (); // 438
                [109]([9]) -> (); // 439
                [117]([7]) -> (); // 440
                [117]([5]) -> (); // 441
                [5]([45]) -> ([271]); // 442
                [4]() -> ([272]); // 443
                [139]() -> ([273]); // 444
                [60]([273]) -> ([273]); // 445
                [3]([272], [273]) -> ([274]); // 446
                [2]() -> ([275]); // 447
                [1]([275], [274]) -> ([276]); // 448
                [0]([276]) -> ([277]); // 449
                [57]([50]) -> ([50]); // 450
                [145]([271]) -> ([271]); // 451
                [147]([277]) -> ([277]); // 452
                return([50], [271], [277]); // 453
                [35]() -> (); // 454
                [127]([44]) -> (); // 455
                [109]([9]) -> (); // 456
                [117]([7]) -> (); // 457
                [117]([5]) -> (); // 458
                [5]([40]) -> ([278]); // 459
                [145]([278]) -> ([279]); // 460
                [39]() { 483() }; // 461
                [35]() -> (); // 462
                [127]([39]) -> (); // 463
                [109]([9]) -> (); // 464
                [117]([7]) -> (); // 465
                [117]([5]) -> (); // 466
                [70]([21]) -> (); // 467
                [70]([30]) -> (); // 468
                [5]([35]) -> ([280]); // 469
                [145]([280]) -> ([279]); // 470
                [39]() { 483() }; // 471
                [35]() -> (); // 472
                [127]([34]) -> (); // 473
                [109]([9]) -> (); // 474
                [117]([7]) -> (); // 475
                [117]([5]) -> (); // 476
                [70]([21]) -> (); // 477
                [70]([30]) -> (); // 478
                [70]([20]) -> (); // 479
                [70]([29]) -> (); // 480
                [5]([14]) -> ([281]); // 481
                [145]([281]) -> ([279]); // 482
                [93]() -> ([282]); // 483
                [82]() -> ([283]); // 484
                [83]() -> ([284]); // 485
                [142]() -> ([285]); // 486
                [73]() -> ([286]); // 487
                [21]([282], [283], [284]) -> ([287]); // 488
                [57]([0]) -> ([0]); // 489
                [89]([287]) -> ([287]); // 490
                [60]([285]) -> ([285]); // 491
                [86]([286]) -> ([286]); // 492
                [16]([0], [287], [285], [286]) -> ([288], [289]); // 493
                [15]([289]) { fallthrough([290]) 578([291]) }; // 494
                [35]() -> (); // 495
                [5]([279]) -> ([292]); // 496
                [14]([290]) -> ([293], [294]); // 497
                [37]([294]) -> (); // 498
                [143]() -> ([295]); // 499
                [130]() -> ([296]); // 500
                [57]([288]) -> ([288]); // 501
                [89]([293]) -> ([293]); // 502
                [60]([295]) -> ([295]); // 503
                [86]([296]) -> ([296]); // 504
                [16]([288], [293], [295], [296]) -> ([297], [298]); // 505
                [145]([292]) -> ([292]); // 506
                [15]([298]) { fallthrough([299]) 570([300]) }; // 507
                [35]() -> (); // 508
                [5]([292]) -> ([301]); // 509
                [4]() -> ([302]); // 510
                [131]() -> ([303]); // 511
                [60]([303]) -> ([303]); // 512
                [3]([302], [303]) -> ([304]); // 513
                [14]([299]) -> ([2], [305]); // 514
                [37]([305]) -> (); // 515
                [148]([3], [2]) -> ([2]); // 516
                [132]([2]) -> ([306], [307]); // 517
                [133]([306]) -> (); // 518
                [134]([307]) -> ([307], [308]); // 519
                [8]([308]) -> ([309], [310], [311]); // 520
                [70]([310]) -> (); // 521
                [71]([311]) -> (); // 522
                [135]([309]) -> ([309], [312]); // 523
                [13]([312]) -> ([313]); // 524
                [7]([313]) -> ([314]); // 525
                [60]([314]) -> ([314]); // 526
                [3]([304], [314]) -> ([315]); // 527
                [12]([309]) -> ([316]); // 528
                [57]([297]) -> ([297]); // 529
                [145]([301]) -> ([301]); // 530
                [149]([316]) -> ([316]); // 531
                [88]([315]) -> ([315]); // 532
                [11]([297], [301], [316], [315]) -> ([317], [318], [319]); // 533
                [122]() -> (); // 534
                [10]([319]) { fallthrough([320]) 555([321]) }; // 535
                [35]() -> (); // 536
                [5]([318]) -> ([322]); // 537
                [9]([320]) -> ([323], [324]); // 538
                [37]([324]) -> (); // 539
                [134]([307]) -> ([307], [325]); // 540
                [8]([325]) -> ([326], [327], [328]); // 541
                [136]([326]) -> (); // 542
                [71]([328]) -> (); // 543
                [116]([327]) -> ([329]); // 544
                [3]([323], [329]) -> ([330]); // 545
                [8]([307]) -> ([331], [332], [333]); // 546
                [136]([331]) -> (); // 547
                [70]([332]) -> (); // 548
                [137]([333]) -> ([334]); // 549
                [7]([334]) -> ([335]); // 550
                [3]([330], [335]) -> ([336]); // 551
                [145]([322]) -> ([337]); // 552
                [88]([336]) -> ([338]); // 553
                [39]() { 562() }; // 554
                [35]() -> (); // 555
                [138]([307]) -> (); // 556
                [5]([318]) -> ([339]); // 557
                [6]([321]) -> ([340], [341]); // 558
                [78]([340]) -> (); // 559
                [145]([339]) -> ([337]); // 560
                [88]([341]) -> ([338]); // 561
                [106]() -> (); // 562
                [2]() -> ([342]); // 563
                [1]([342], [338]) -> ([343]); // 564
                [0]([343]) -> ([344]); // 565
                [57]([317]) -> ([317]); // 566
                [145]([337]) -> ([337]); // 567
                [147]([344]) -> ([344]); // 568
                return([317], [337], [344]); // 569
                [35]() -> (); // 570
                [117]([3]) -> (); // 571
                [5]([292]) -> ([345]); // 572
                [0]([300]) -> ([346]); // 573
                [57]([297]) -> ([297]); // 574
                [145]([345]) -> ([345]); // 575
                [147]([346]) -> ([346]); // 576
                return([297], [345], [346]); // 577
                [35]() -> (); // 578
                [117]([3]) -> (); // 579
                [5]([279]) -> ([347]); // 580
                [0]([291]) -> ([348]); // 581
                [57]([288]) -> ([288]); // 582
                [145]([347]) -> ([347]); // 583
                [147]([348]) -> ([348]); // 584
                return([288], [347], [348]); // 585
                [35]() -> (); // 586
                [109]([13]) -> (); // 587
                [109]([9]) -> (); // 588
                [117]([7]) -> (); // 589
                [117]([5]) -> (); // 590
                [117]([3]) -> (); // 591
                [5]([1]) -> ([349]); // 592
                [4]() -> ([350]); // 593
                [139]() -> ([351]); // 594
                [60]([351]) -> ([351]); // 595
                [3]([350], [351]) -> ([352]); // 596
                [2]() -> ([353]); // 597
                [1]([353], [352]) -> ([354]); // 598
                [0]([354]) -> ([355]); // 599
                [57]([0]) -> ([0]); // 600
                [145]([349]) -> ([349]); // 601
                [147]([355]) -> ([355]); // 602
                return([0], [349], [355]); // 603
                [69]([3]) -> ([3], [4]); // 604
                [62]([4]) { fallthrough() 615([5]) }; // 605
                [35]() -> (); // 606
                [70]([2]) -> (); // 607
                [71]([3]) -> (); // 608
                [20]() -> ([6]); // 609
                [19]([1], [6]) -> ([7]); // 610
                [18]([7]) -> ([8]); // 611
                [57]([0]) -> ([0]); // 612
                [85]([8]) -> ([8]); // 613
                return([0], [8]); // 614
                [35]() -> (); // 615
                [72]([5]) -> (); // 616
                [65]([1]) -> ([9], [10], [11]); // 617
                [69]([11]) -> ([11], [12]); // 618
                [69]([3]) -> ([3], [13]); // 619
                [24]([0], [12], [13]) { fallthrough([14], [15]) 1429([16], [17]) }; // 620
                [35]() -> (); // 621
                [73]() -> ([18]); // 622
                [69]([15]) -> ([15], [19]); // 623
                [86]([18]) -> ([18]); // 624
                [61]([14], [19], [18]) { fallthrough([20], [21]) 1305([22], [23]) }; // 625
                [35]() -> (); // 626
                [71]([21]) -> (); // 627
                [73]() -> ([24]); // 628
                [69]([15]) -> ([15], [25]); // 629
                [57]([20]) -> ([20]); // 630
                [68]([25], [24]) { fallthrough() 1204() }; // 631
                [35]() -> (); // 632
                [71]([3]) -> (); // 633
                [73]() -> ([26]); // 634
                [86]([26]) -> ([26]); // 635
                [61]([20], [15], [26]) { fallthrough([27], [28]) 1188([29], [30]) }; // 636
                [35]() -> (); // 637
                [74]() -> ([31]); // 638
                [69]([28]) -> ([28], [32]); // 639
                [57]([27]) -> ([27]); // 640
                [68]([32], [31]) { fallthrough() 1042() }; // 641
                [35]() -> (); // 642
                [74]() -> ([33]); // 643
                [69]([28]) -> ([28], [34]); // 644
                [86]([33]) -> ([33]); // 645
                [61]([27], [34], [33]) { fallthrough([35], [36]) 832([37], [38]) }; // 646
                [35]() -> (); // 647
                [71]([36]) -> (); // 648
                [66]([35], [2]) { fallthrough([39], [40]) 656([41], [42], [43]) }; // 649
                [35]() -> (); // 650
                [75]() -> ([44]); // 651
                [57]([39]) -> ([45]); // 652
                [87]([40]) -> ([46]); // 653
                [87]([44]) -> ([47]); // 654
                [39]() { 660() }; // 655
                [35]() -> (); // 656
                [57]([41]) -> ([45]); // 657
                [87]([43]) -> ([46]); // 658
                [87]([42]) -> ([47]); // 659
                [74]() -> ([48]); // 660
                [69]([28]) -> ([28], [49]); // 661
                [86]([48]) -> ([48]); // 662
                [61]([45], [49], [48]) { fallthrough([50], [51]) 812([52], [53]) }; // 663
                [35]() -> (); // 664
                [57]([50]) -> ([50]); // 665
                [86]([51]) -> ([51]); // 666
                [29]([50], [51]) -> ([54], [55]); // 667
                [28]([55]) { fallthrough([56]) 800([57]) }; // 668
                [35]() -> (); // 669
                [27]([56]) -> ([58]); // 670
                [67]([54], [47], [58]) -> ([59], [60], [61]); // 671
                [25]([61]) -> ([62]); // 672
                [25]([46]) -> ([63]); // 673
                [25]([60]) -> ([64]); // 674
                [73]() -> ([65]); // 675
                [69]([11]) -> ([11], [66]); // 676
                [86]([65]) -> ([65]); // 677
                [61]([59], [65], [66]) { fallthrough([67], [68]) 784([69], [70]) }; // 678
                [35]() -> (); // 679
                [76]() -> ([71]); // 680
                [23]([62], [71]) -> ([72]); // 681
                [60]([72]) -> ([72]); // 682
                [22]([72], [63]) -> ([73]); // 683
                [74]() -> ([74]); // 684
                [69]([68]) -> ([68], [75]); // 685
                [86]([74]) -> ([74]); // 686
                [60]([73]) -> ([73]); // 687
                [61]([67], [75], [74]) { fallthrough([76], [77]) 735([78], [79]) }; // 688
                [35]() -> (); // 689
                [71]([77]) -> (); // 690
                [74]() -> ([80]); // 691
                [86]([80]) -> ([80]); // 692
                [61]([76], [68], [80]) { fallthrough([81], [82]) 720([83], [84]) }; // 693
                [35]() -> (); // 694
                [57]([81]) -> ([81]); // 695
                [86]([82]) -> ([82]); // 696
                [29]([81], [82]) -> ([85], [86]); // 697
                [28]([86]) { fallthrough([87]) 708([88]) }; // 698
                [35]() -> (); // 699
                [27]([87]) -> ([89]); // 700
                [26]([89]) -> ([90]); // 701
                [25]([90]) -> ([91]); // 702
                [76]() -> ([92]); // 703
                [23]([91], [92]) -> ([93]); // 704
                [57]([85]) -> ([94]); // 705
                [60]([93]) -> ([95]); // 706
                [39]() { 747() }; // 707
                [35]() -> (); // 708
                [77]([9]) -> (); // 709
                [71]([28]) -> (); // 710
                [71]([11]) -> (); // 711
                [70]([73]) -> (); // 712
                [70]([64]) -> (); // 713
                [70]([10]) -> (); // 714
                [6]([88]) -> ([96], [97]); // 715
                [78]([96]) -> (); // 716
                [57]([85]) -> ([98]); // 717
                [88]([97]) -> ([99]); // 718
                [39]() { 781() }; // 719
                [35]() -> (); // 720
                [71]([84]) -> (); // 721
                [77]([9]) -> (); // 722
                [71]([28]) -> (); // 723
                [71]([11]) -> (); // 724
                [70]([73]) -> (); // 725
                [70]([10]) -> (); // 726
                [70]([64]) -> (); // 727
                [4]() -> ([100]); // 728
                [79]() -> ([101]); // 729
                [60]([101]) -> ([101]); // 730
                [3]([100], [101]) -> ([102]); // 731
                [57]([83]) -> ([98]); // 732
                [88]([102]) -> ([99]); // 733
                [39]() { 781() }; // 734
                [35]() -> (); // 735
                [71]([79]) -> (); // 736
                [57]([78]) -> ([78]); // 737
                [86]([68]) -> ([68]); // 738
                [29]([78], [68]) -> ([103], [104]); // 739
                [28]([104]) { fallthrough([105]) 770([106]) }; // 740
                [35]() -> (); // 741
                [27]([105]) -> ([107]); // 742
                [26]([107]) -> ([108]); // 743
                [25]([108]) -> ([109]); // 744
                [57]([103]) -> ([94]); // 745
                [60]([109]) -> ([95]); // 746
                [23]([10], [95]) -> ([110]); // 747
                [60]([110]) -> ([110]); // 748
                [22]([64], [110]) -> ([111]); // 749
                [60]([111]) -> ([111]); // 750
                [64]([94], [111]) { fallthrough([112], [113]) 758([114]) }; // 751
                [35]() -> (); // 752
                [63]([9], [113]) -> ([115]); // 753
                [21]([115], [73], [11]) -> ([116]); // 754
                [57]([112]) -> ([117]); // 755
                [89]([116]) -> ([118]); // 756
                [39]() { 950() }; // 757
                [35]() -> (); // 758
                [77]([9]) -> (); // 759
                [71]([28]) -> (); // 760
                [71]([11]) -> (); // 761
                [70]([73]) -> (); // 762
                [4]() -> ([119]); // 763
                [80]() -> ([120]); // 764
                [60]([120]) -> ([120]); // 765
                [3]([119], [120]) -> ([121]); // 766
                [57]([114]) -> ([122]); // 767
                [88]([121]) -> ([123]); // 768
                [39]() { 826() }; // 769
                [35]() -> (); // 770
                [77]([9]) -> (); // 771
                [71]([28]) -> (); // 772
                [71]([11]) -> (); // 773
                [70]([73]) -> (); // 774
                [70]([64]) -> (); // 775
                [70]([10]) -> (); // 776
                [6]([106]) -> ([124], [125]); // 777
                [78]([124]) -> (); // 778
                [57]([103]) -> ([98]); // 779
                [88]([125]) -> ([99]); // 780
                [90]([98]) -> ([122]); // 781
                [91]([99]) -> ([123]); // 782
                [39]() { 826() }; // 783
                [35]() -> (); // 784
                [71]([70]) -> (); // 785
                [77]([9]) -> (); // 786
                [71]([28]) -> (); // 787
                [71]([11]) -> (); // 788
                [70]([62]) -> (); // 789
                [70]([10]) -> (); // 790
                [70]([64]) -> (); // 791
                [70]([63]) -> (); // 792
                [4]() -> ([126]); // 793
                [79]() -> ([127]); // 794
                [60]([127]) -> ([127]); // 795
                [3]([126], [127]) -> ([128]); // 796
                [57]([69]) -> ([122]); // 797
                [88]([128]) -> ([123]); // 798
                [39]() { 826() }; // 799
                [35]() -> (); // 800
                [77]([9]) -> (); // 801
                [71]([28]) -> (); // 802
                [71]([11]) -> (); // 803
                [81]([46]) -> (); // 804
                [70]([10]) -> (); // 805
                [81]([47]) -> (); // 806
                [6]([57]) -> ([129], [130]); // 807
                [78]([129]) -> (); // 808
                [57]([54]) -> ([122]); // 809
                [88]([130]) -> ([123]); // 810
                [39]() { 826() }; // 811
                [35]() -> (); // 812
                [71]([53]) -> (); // 813
                [77]([9]) -> (); // 814
                [71]([28]) -> (); // 815
                [71]([11]) -> (); // 816
                [81]([46]) -> (); // 817
                [70]([10]) -> (); // 818
                [81]([47]) -> (); // 819
                [4]() -> ([131]); // 820
                [79]() -> ([132]); // 821
                [60]([132]) -> ([132]); // 822
                [3]([131], [132]) -> ([133]); // 823
                [57]([52]) -> ([122]); // 824
                [88]([133]) -> ([123]); // 825
                [2]() -> ([134]); // 826
                [1]([134], [123]) -> ([135]); // 827
                [17]([135]) -> ([136]); // 828
                [57]([122]) -> ([122]); // 829
                [85]([136]) -> ([136]); // 830
                return([122], [136]); // 831
                [35]() -> (); // 832
                [71]([38]) -> (); // 833
                [66]([37], [2]) { fallthrough([137], [138]) 841([139], [140], [141]) }; // 834
                [35]() -> (); // 835
                [75]() -> ([142]); // 836
                [57]([137]) -> ([143]); // 837
                [87]([138]) -> ([144]); // 838
                [87]([142]) -> ([145]); // 839
                [39]() { 845() }; // 840
                [35]() -> (); // 841
                [57]([139]) -> ([143]); // 842
                [87]([141]) -> ([144]); // 843
                [87]([140]) -> ([145]); // 844
                [57]([143]) -> ([143]); // 845
                [69]([28]) -> ([28], [146]); // 846
                [86]([146]) -> ([146]); // 847
                [29]([143], [146]) -> ([147], [148]); // 848
                [28]([148]) { fallthrough([149]) 1025([150]) }; // 849
                [35]() -> (); // 850
                [27]([149]) -> ([151]); // 851
                [67]([147], [144], [151]) -> ([152], [153], [154]); // 852
                [25]([145]) -> ([155]); // 853
                [74]() -> ([156]); // 854
                [69]([28]) -> ([28], [157]); // 855
                [86]([156]) -> ([156]); // 856
                [61]([152], [156], [157]) { fallthrough([158], [159]) 1009([160], [161]) }; // 857
                [35]() -> (); // 858
                [57]([158]) -> ([158]); // 859
                [86]([159]) -> ([159]); // 860
                [29]([158], [159]) -> ([162], [163]); // 861
                [28]([163]) { fallthrough([164]) 996([165]) }; // 862
                [35]() -> (); // 863
                [27]([164]) -> ([166]); // 864
                [26]([166]) -> ([167]); // 865
                [25]([167]) -> ([168]); // 866
                [25]([153]) -> ([169]); // 867
                [25]([154]) -> ([170]); // 868
                [73]() -> ([171]); // 869
                [69]([11]) -> ([11], [172]); // 870
                [86]([171]) -> ([171]); // 871
                [61]([162], [171], [172]) { fallthrough([173], [174]) 979([175], [176]) }; // 872
                [35]() -> (); // 873
                [23]([155], [168]) -> ([177]); // 874
                [60]([177]) -> ([177]); // 875
                [22]([177], [169]) -> ([178]); // 876
                [74]() -> ([179]); // 877
                [69]([174]) -> ([174], [180]); // 878
                [86]([179]) -> ([179]); // 879
                [60]([178]) -> ([178]); // 880
                [61]([173], [180], [179]) { fallthrough([181], [182]) 928([183], [184]) }; // 881
                [35]() -> (); // 882
                [71]([182]) -> (); // 883
                [74]() -> ([185]); // 884
                [86]([185]) -> ([185]); // 885
                [61]([181], [174], [185]) { fallthrough([186], [187]) 913([188], [189]) }; // 886
                [35]() -> (); // 887
                [57]([186]) -> ([186]); // 888
                [86]([187]) -> ([187]); // 889
                [29]([186], [187]) -> ([190], [191]); // 890
                [28]([191]) { fallthrough([192]) 901([193]) }; // 891
                [35]() -> (); // 892
                [27]([192]) -> ([194]); // 893
                [26]([194]) -> ([195]); // 894
                [25]([195]) -> ([196]); // 895
                [76]() -> ([197]); // 896
                [23]([196], [197]) -> ([198]); // 897
                [57]([190]) -> ([199]); // 898
                [60]([198]) -> ([200]); // 899
                [39]() { 940() }; // 900
                [35]() -> (); // 901
                [77]([9]) -> (); // 902
                [71]([28]) -> (); // 903
                [71]([11]) -> (); // 904
                [70]([170]) -> (); // 905
                [70]([178]) -> (); // 906
                [70]([10]) -> (); // 907
                [6]([193]) -> ([201], [202]); // 908
                [78]([201]) -> (); // 909
                [57]([190]) -> ([203]); // 910
                [88]([202]) -> ([204]); // 911
                [39]() { 976() }; // 912
                [35]() -> (); // 913
                [71]([189]) -> (); // 914
                [77]([9]) -> (); // 915
                [71]([28]) -> (); // 916
                [71]([11]) -> (); // 917
                [70]([170]) -> (); // 918
                [70]([10]) -> (); // 919
                [70]([178]) -> (); // 920
                [4]() -> ([205]); // 921
                [79]() -> ([206]); // 922
                [60]([206]) -> ([206]); // 923
                [3]([205], [206]) -> ([207]); // 924
                [57]([188]) -> ([203]); // 925
                [88]([207]) -> ([204]); // 926
                [39]() { 976() }; // 927
                [35]() -> (); // 928
                [71]([184]) -> (); // 929
                [57]([183]) -> ([183]); // 930
                [86]([174]) -> ([174]); // 931
                [29]([183], [174]) -> ([208], [209]); // 932
                [28]([209]) { fallthrough([210]) 965([211]) }; // 933
                [35]() -> (); // 934
                [27]([210]) -> ([212]); // 935
                [26]([212]) -> ([213]); // 936
                [25]([213]) -> ([214]); // 937
                [57]([208]) -> ([199]); // 938
                [60]([214]) -> ([200]); // 939
                [23]([10], [200]) -> ([215]); // 940
                [60]([215]) -> ([215]); // 941
                [22]([178], [215]) -> ([216]); // 942
                [60]([216]) -> ([216]); // 943
                [64]([199], [216]) { fallthrough([217], [218]) 953([219]) }; // 944
                [35]() -> (); // 945
                [63]([9], [218]) -> ([220]); // 946
                [21]([220], [170], [11]) -> ([221]); // 947
                [57]([217]) -> ([117]); // 948
                [89]([221]) -> ([118]); // 949
                [90]([117]) -> ([222]); // 950
                [92]([118]) -> ([223]); // 951
                [39]() { 1133() }; // 952
                [35]() -> (); // 953
                [77]([9]) -> (); // 954
                [71]([28]) -> (); // 955
                [71]([11]) -> (); // 956
                [70]([170]) -> (); // 957
                [4]() -> ([224]); // 958
                [80]() -> ([225]); // 959
                [60]([225]) -> ([225]); // 960
                [3]([224], [225]) -> ([226]); // 961
                [57]([219]) -> ([227]); // 962
                [88]([226]) -> ([228]); // 963
                [39]() { 1036() }; // 964
                [35]() -> (); // 965
                [77]([9]) -> (); // 966
                [71]([28]) -> (); // 967
                [71]([11]) -> (); // 968
                [70]([170]) -> (); // 969
                [70]([178]) -> (); // 970
                [70]([10]) -> (); // 971
                [6]([211]) -> ([229], [230]); // 972
                [78]([229]) -> (); // 973
                [57]([208]) -> ([203]); // 974
                [88]([230]) -> ([204]); // 975
                [90]([203]) -> ([227]); // 976
                [91]([204]) -> ([228]); // 977
                [39]() { 1036() }; // 978
                [35]() -> (); // 979
                [71]([176]) -> (); // 980
                [77]([9]) -> (); // 981
                [71]([28]) -> (); // 982
                [71]([11]) -> (); // 983
                [70]([170]) -> (); // 984
                [70]([10]) -> (); // 985
                [70]([155]) -> (); // 986
                [70]([168]) -> (); // 987
                [70]([169]) -> (); // 988
                [4]() -> ([231]); // 989
                [79]() -> ([232]); // 990
                [60]([232]) -> ([232]); // 991
                [3]([231], [232]) -> ([233]); // 992
                [57]([175]) -> ([227]); // 993
                [88]([233]) -> ([228]); // 994
                [39]() { 1036() }; // 995
                [35]() -> (); // 996
                [77]([9]) -> (); // 997
                [71]([28]) -> (); // 998
                [71]([11]) -> (); // 999
                [70]([10]) -> (); // 1000
                [70]([155]) -> (); // 1001
                [81]([153]) -> (); // 1002
                [81]([154]) -> (); // 1003
                [6]([165]) -> ([234], [235]); // 1004
                [78]([234]) -> (); // 1005
                [57]([162]) -> ([227]); // 1006
                [88]([235]) -> ([228]); // 1007
                [39]() { 1036() }; // 1008
                [35]() -> (); // 1009
                [71]([161]) -> (); // 1010
                [77]([9]) -> (); // 1011
                [71]([28]) -> (); // 1012
                [71]([11]) -> (); // 1013
                [81]([154]) -> (); // 1014
                [70]([10]) -> (); // 1015
                [70]([155]) -> (); // 1016
                [81]([153]) -> (); // 1017
                [4]() -> ([236]); // 1018
                [79]() -> ([237]); // 1019
                [60]([237]) -> ([237]); // 1020
                [3]([236], [237]) -> ([238]); // 1021
                [57]([160]) -> ([227]); // 1022
                [88]([238]) -> ([228]); // 1023
                [39]() { 1036() }; // 1024
                [35]() -> (); // 1025
                [77]([9]) -> (); // 1026
                [71]([28]) -> (); // 1027
                [71]([11]) -> (); // 1028
                [81]([145]) -> (); // 1029
                [70]([10]) -> (); // 1030
                [81]([144]) -> (); // 1031
                [6]([150]) -> ([239], [240]); // 1032
                [78]([239]) -> (); // 1033
                [57]([147]) -> ([227]); // 1034
                [88]([240]) -> ([228]); // 1035
                [2]() -> ([241]); // 1036
                [1]([241], [228]) -> ([242]); // 1037
                [17]([242]) -> ([243]); // 1038
                [57]([227]) -> ([227]); // 1039
                [85]([243]) -> ([243]); // 1040
                return([227], [243]); // 1041
                [35]() -> (); // 1042
                [66]([27], [2]) { fallthrough([244], [245]) 1050([246], [247], [248]) }; // 1043
                [35]() -> (); // 1044
                [75]() -> ([249]); // 1045
                [57]([244]) -> ([250]); // 1046
                [87]([245]) -> ([251]); // 1047
                [87]([249]) -> ([252]); // 1048
                [39]() { 1054() }; // 1049
                [35]() -> (); // 1050
                [57]([246]) -> ([250]); // 1051
                [87]([248]) -> ([251]); // 1052
                [87]([247]) -> ([252]); // 1053
                [25]([252]) -> ([253]); // 1054
                [25]([251]) -> ([254]); // 1055
                [73]() -> ([255]); // 1056
                [69]([11]) -> ([11], [256]); // 1057
                [86]([255]) -> ([255]); // 1058
                [61]([250], [255], [256]) { fallthrough([257], [258]) 1168([259], [260]) }; // 1059
                [35]() -> (); // 1060
                [74]() -> ([261]); // 1061
                [69]([258]) -> ([258], [262]); // 1062
                [86]([261]) -> ([261]); // 1063
                [61]([257], [262], [261]) { fallthrough([263], [264]) 1111([265], [266]) }; // 1064
                [35]() -> (); // 1065
                [71]([264]) -> (); // 1066
                [74]() -> ([267]); // 1067
                [86]([267]) -> ([267]); // 1068
                [61]([263], [258], [267]) { fallthrough([268], [269]) 1096([270], [271]) }; // 1069
                [35]() -> (); // 1070
                [57]([268]) -> ([268]); // 1071
                [86]([269]) -> ([269]); // 1072
                [29]([268], [269]) -> ([272], [273]); // 1073
                [28]([273]) { fallthrough([274]) 1084([275]) }; // 1074
                [35]() -> (); // 1075
                [27]([274]) -> ([276]); // 1076
                [26]([276]) -> ([277]); // 1077
                [25]([277]) -> ([278]); // 1078
                [76]() -> ([279]); // 1079
                [23]([278], [279]) -> ([280]); // 1080
                [57]([272]) -> ([281]); // 1081
                [60]([280]) -> ([282]); // 1082
                [39]() { 1123() }; // 1083
                [35]() -> (); // 1084
                [77]([9]) -> (); // 1085
                [71]([28]) -> (); // 1086
                [71]([11]) -> (); // 1087
                [70]([254]) -> (); // 1088
                [70]([253]) -> (); // 1089
                [70]([10]) -> (); // 1090
                [6]([275]) -> ([283], [284]); // 1091
                [78]([283]) -> (); // 1092
                [57]([272]) -> ([285]); // 1093
                [88]([284]) -> ([286]); // 1094
                [39]() { 1165() }; // 1095
                [35]() -> (); // 1096
                [71]([271]) -> (); // 1097
                [77]([9]) -> (); // 1098
                [71]([28]) -> (); // 1099
                [71]([11]) -> (); // 1100
                [70]([254]) -> (); // 1101
                [70]([10]) -> (); // 1102
                [70]([253]) -> (); // 1103
                [4]() -> ([287]); // 1104
                [79]() -> ([288]); // 1105
                [60]([288]) -> ([288]); // 1106
                [3]([287], [288]) -> ([289]); // 1107
                [57]([270]) -> ([285]); // 1108
                [88]([289]) -> ([286]); // 1109
                [39]() { 1165() }; // 1110
                [35]() -> (); // 1111
                [71]([266]) -> (); // 1112
                [57]([265]) -> ([265]); // 1113
                [86]([258]) -> ([258]); // 1114
                [29]([265], [258]) -> ([290], [291]); // 1115
                [28]([291]) { fallthrough([292]) 1154([293]) }; // 1116
                [35]() -> (); // 1117
                [27]([292]) -> ([294]); // 1118
                [26]([294]) -> ([295]); // 1119
                [25]([295]) -> ([296]); // 1120
                [57]([290]) -> ([281]); // 1121
                [60]([296]) -> ([282]); // 1122
                [23]([10], [282]) -> ([297]); // 1123
                [60]([297]) -> ([297]); // 1124
                [22]([253], [297]) -> ([298]); // 1125
                [60]([298]) -> ([298]); // 1126
                [64]([281], [298]) { fallthrough([299], [300]) 1142([301]) }; // 1127
                [35]() -> (); // 1128
                [63]([9], [300]) -> ([302]); // 1129
                [21]([302], [254], [11]) -> ([303]); // 1130
                [57]([299]) -> ([222]); // 1131
                [89]([303]) -> ([223]); // 1132
                [65]([223]) -> ([304], [305], [306]); // 1133
                [71]([306]) -> (); // 1134
                [21]([304], [305], [28]) -> ([307]); // 1135
                [20]() -> ([308]); // 1136
                [19]([307], [308]) -> ([309]); // 1137
                [18]([309]) -> ([310]); // 1138
                [57]([222]) -> ([222]); // 1139
                [85]([310]) -> ([310]); // 1140
                return([222], [310]); // 1141
                [35]() -> (); // 1142
                [77]([9]) -> (); // 1143
                [71]([28]) -> (); // 1144
                [71]([11]) -> (); // 1145
                [70]([254]) -> (); // 1146
                [4]() -> ([311]); // 1147
                [80]() -> ([312]); // 1148
                [60]([312]) -> ([312]); // 1149
                [3]([311], [312]) -> ([313]); // 1150
                [57]([301]) -> ([314]); // 1151
                [88]([313]) -> ([315]); // 1152
                [39]() { 1182() }; // 1153
                [35]() -> (); // 1154
                [77]([9]) -> (); // 1155
                [71]([28]) -> (); // 1156
                [71]([11]) -> (); // 1157
                [70]([254]) -> (); // 1158
                [70]([253]) -> (); // 1159
                [70]([10]) -> (); // 1160
                [6]([293]) -> ([316], [317]); // 1161
                [78]([316]) -> (); // 1162
                [57]([290]) -> ([285]); // 1163
                [88]([317]) -> ([286]); // 1164
                [90]([285]) -> ([314]); // 1165
                [91]([286]) -> ([315]); // 1166
                [39]() { 1182() }; // 1167
                [35]() -> (); // 1168
                [71]([260]) -> (); // 1169
                [77]([9]) -> (); // 1170
                [71]([28]) -> (); // 1171
                [71]([11]) -> (); // 1172
                [70]([254]) -> (); // 1173
                [70]([10]) -> (); // 1174
                [70]([253]) -> (); // 1175
                [4]() -> ([318]); // 1176
                [79]() -> ([319]); // 1177
                [60]([319]) -> ([319]); // 1178
                [3]([318], [319]) -> ([320]); // 1179
                [57]([259]) -> ([314]); // 1180
                [88]([320]) -> ([315]); // 1181
                [2]() -> ([321]); // 1182
                [1]([321], [315]) -> ([322]); // 1183
                [17]([322]) -> ([323]); // 1184
                [57]([314]) -> ([314]); // 1185
                [85]([323]) -> ([323]); // 1186
                return([314], [323]); // 1187
                [35]() -> (); // 1188
                [71]([30]) -> (); // 1189
                [77]([9]) -> (); // 1190
                [70]([2]) -> (); // 1191
                [71]([11]) -> (); // 1192
                [70]([10]) -> (); // 1193
                [4]() -> ([324]); // 1194
                [79]() -> ([325]); // 1195
                [60]([325]) -> ([325]); // 1196
                [3]([324], [325]) -> ([326]); // 1197
                [2]() -> ([327]); // 1198
                [1]([327], [326]) -> ([328]); // 1199
                [17]([328]) -> ([329]); // 1200
                [57]([29]) -> ([29]); // 1201
                [85]([329]) -> ([329]); // 1202
                return([29], [329]); // 1203
                [35]() -> (); // 1204
                [71]([11]) -> (); // 1205
                [71]([15]) -> (); // 1206
                [74]() -> ([330]); // 1207
                [69]([3]) -> ([3], [331]); // 1208
                [86]([330]) -> ([330]); // 1209
                [61]([20], [331], [330]) { fallthrough([332], [333]) 1251([334], [335]) }; // 1210
                [35]() -> (); // 1211
                [71]([333]) -> (); // 1212
                [74]() -> ([336]); // 1213
                [86]([336]) -> ([336]); // 1214
                [61]([332], [3], [336]) { fallthrough([337], [338]) 1239([339], [340]) }; // 1215
                [35]() -> (); // 1216
                [57]([337]) -> ([337]); // 1217
                [86]([338]) -> ([338]); // 1218
                [29]([337], [338]) -> ([341], [342]); // 1219
                [28]([342]) { fallthrough([343]) 1230([344]) }; // 1220
                [35]() -> (); // 1221
                [27]([343]) -> ([345]); // 1222
                [26]([345]) -> ([346]); // 1223
                [25]([346]) -> ([347]); // 1224
                [76]() -> ([348]); // 1225
                [23]([347], [348]) -> ([349]); // 1226
                [57]([341]) -> ([350]); // 1227
                [60]([349]) -> ([351]); // 1228
                [39]() { 1263() }; // 1229
                [35]() -> (); // 1230
                [77]([9]) -> (); // 1231
                [70]([2]) -> (); // 1232
                [70]([10]) -> (); // 1233
                [6]([344]) -> ([352], [353]); // 1234
                [78]([352]) -> (); // 1235
                [57]([341]) -> ([354]); // 1236
                [88]([353]) -> ([355]); // 1237
                [39]() { 1299() }; // 1238
                [35]() -> (); // 1239
                [71]([340]) -> (); // 1240
                [77]([9]) -> (); // 1241
                [70]([10]) -> (); // 1242
                [70]([2]) -> (); // 1243
                [4]() -> ([356]); // 1244
                [79]() -> ([357]); // 1245
                [60]([357]) -> ([357]); // 1246
                [3]([356], [357]) -> ([358]); // 1247
                [57]([339]) -> ([354]); // 1248
                [88]([358]) -> ([355]); // 1249
                [39]() { 1299() }; // 1250
                [35]() -> (); // 1251
                [71]([335]) -> (); // 1252
                [57]([334]) -> ([334]); // 1253
                [86]([3]) -> ([3]); // 1254
                [29]([334], [3]) -> ([359], [360]); // 1255
                [28]([360]) { fallthrough([361]) 1291([362]) }; // 1256
                [35]() -> (); // 1257
                [27]([361]) -> ([363]); // 1258
                [26]([363]) -> ([364]); // 1259
                [25]([364]) -> ([365]); // 1260
                [57]([359]) -> ([350]); // 1261
                [60]([365]) -> ([351]); // 1262
                [23]([10], [351]) -> ([366]); // 1263
                [60]([366]) -> ([366]); // 1264
                [22]([2], [366]) -> ([367]); // 1265
                [60]([367]) -> ([367]); // 1266
                [64]([350], [367]) { fallthrough([368], [369]) 1279([370]) }; // 1267
                [35]() -> (); // 1268
                [63]([9], [369]) -> ([371]); // 1269
                [82]() -> ([372]); // 1270
                [83]() -> ([373]); // 1271
                [21]([371], [372], [373]) -> ([374]); // 1272
                [20]() -> ([375]); // 1273
                [19]([374], [375]) -> ([376]); // 1274
                [18]([376]) -> ([377]); // 1275
                [57]([368]) -> ([368]); // 1276
                [85]([377]) -> ([377]); // 1277
                return([368], [377]); // 1278
                [35]() -> (); // 1279
                [77]([9]) -> (); // 1280
                [4]() -> ([378]); // 1281
                [80]() -> ([379]); // 1282
                [60]([379]) -> ([379]); // 1283
                [3]([378], [379]) -> ([380]); // 1284
                [2]() -> ([381]); // 1285
                [1]([381], [380]) -> ([382]); // 1286
                [17]([382]) -> ([383]); // 1287
                [57]([370]) -> ([370]); // 1288
                [85]([383]) -> ([383]); // 1289
                return([370], [383]); // 1290
                [35]() -> (); // 1291
                [77]([9]) -> (); // 1292
                [70]([2]) -> (); // 1293
                [70]([10]) -> (); // 1294
                [6]([362]) -> ([384], [385]); // 1295
                [78]([384]) -> (); // 1296
                [57]([359]) -> ([354]); // 1297
                [88]([385]) -> ([355]); // 1298
                [2]() -> ([386]); // 1299
                [1]([386], [355]) -> ([387]); // 1300
                [17]([387]) -> ([388]); // 1301
                [57]([354]) -> ([354]); // 1302
                [85]([388]) -> ([388]); // 1303
                return([354], [388]); // 1304
                [35]() -> (); // 1305
                [71]([23]) -> (); // 1306
                [71]([15]) -> (); // 1307
                [69]([11]) -> ([11], [389]); // 1308
                [57]([22]) -> ([22]); // 1309
                [62]([389]) { fallthrough() 1321([390]) }; // 1310
                [35]() -> (); // 1311
                [70]([10]) -> (); // 1312
                [71]([11]) -> (); // 1313
                [21]([9], [2], [3]) -> ([391]); // 1314
                [20]() -> ([392]); // 1315
                [19]([391], [392]) -> ([393]); // 1316
                [18]([393]) -> ([394]); // 1317
                [57]([22]) -> ([22]); // 1318
                [85]([394]) -> ([394]); // 1319
                return([22], [394]); // 1320
                [35]() -> (); // 1321
                [72]([390]) -> (); // 1322
                [74]() -> ([395]); // 1323
                [69]([3]) -> ([3], [396]); // 1324
                [86]([395]) -> ([395]); // 1325
                [61]([22], [396], [395]) { fallthrough([397], [398]) 1372([399], [400]) }; // 1326
                [35]() -> (); // 1327
                [71]([398]) -> (); // 1328
                [74]() -> ([401]); // 1329
                [69]([3]) -> ([3], [402]); // 1330
                [86]([401]) -> ([401]); // 1331
                [61]([397], [402], [401]) { fallthrough([403], [404]) 1358([405], [406]) }; // 1332
                [35]() -> (); // 1333
                [57]([403]) -> ([403]); // 1334
                [86]([404]) -> ([404]); // 1335
                [29]([403], [404]) -> ([407], [408]); // 1336
                [28]([408]) { fallthrough([409]) 1347([410]) }; // 1337
                [35]() -> (); // 1338
                [27]([409]) -> ([411]); // 1339
                [26]([411]) -> ([412]); // 1340
                [25]([412]) -> ([413]); // 1341
                [76]() -> ([414]); // 1342
                [23]([413], [414]) -> ([415]); // 1343
                [57]([407]) -> ([416]); // 1344
                [60]([415]) -> ([417]); // 1345
                [39]() { 1385() }; // 1346
                [35]() -> (); // 1347
                [70]([10]) -> (); // 1348
                [77]([9]) -> (); // 1349
                [70]([2]) -> (); // 1350
                [71]([3]) -> (); // 1351
                [71]([11]) -> (); // 1352
                [6]([410]) -> ([418], [419]); // 1353
                [78]([418]) -> (); // 1354
                [57]([407]) -> ([420]); // 1355
                [88]([419]) -> ([421]); // 1356
                [39]() { 1423() }; // 1357
                [35]() -> (); // 1358
                [71]([406]) -> (); // 1359
                [70]([10]) -> (); // 1360
                [71]([11]) -> (); // 1361
                [77]([9]) -> (); // 1362
                [70]([2]) -> (); // 1363
                [71]([3]) -> (); // 1364
                [4]() -> ([422]); // 1365
                [79]() -> ([423]); // 1366
                [60]([423]) -> ([423]); // 1367
                [3]([422], [423]) -> ([424]); // 1368
                [57]([405]) -> ([420]); // 1369
                [88]([424]) -> ([421]); // 1370
                [39]() { 1423() }; // 1371
                [35]() -> (); // 1372
                [71]([400]) -> (); // 1373
                [57]([399]) -> ([399]); // 1374
                [69]([3]) -> ([3], [425]); // 1375
                [86]([425]) -> ([425]); // 1376
                [29]([399], [425]) -> ([426], [427]); // 1377
                [28]([427]) { fallthrough([428]) 1413([429]) }; // 1378
                [35]() -> (); // 1379
                [27]([428]) -> ([430]); // 1380
                [26]([430]) -> ([431]); // 1381
                [25]([431]) -> ([432]); // 1382
                [57]([426]) -> ([416]); // 1383
                [60]([432]) -> ([417]); // 1384
                [24]([416], [11], [3]) { fallthrough([433], [434]) 1397([435], [436]) }; // 1385
                [35]() -> (); // 1386
                [23]([10], [417]) -> ([437]); // 1387
                [60]([437]) -> ([437]); // 1388
                [22]([2], [437]) -> ([438]); // 1389
                [21]([9], [438], [434]) -> ([439]); // 1390
                [20]() -> ([440]); // 1391
                [19]([439], [440]) -> ([441]); // 1392
                [18]([441]) -> ([442]); // 1393
                [57]([433]) -> ([433]); // 1394
                [85]([442]) -> ([442]); // 1395
                return([433], [442]); // 1396
                [35]() -> (); // 1397
                [71]([436]) -> (); // 1398
                [70]([10]) -> (); // 1399
                [70]([417]) -> (); // 1400
                [77]([9]) -> (); // 1401
                [70]([2]) -> (); // 1402
                [4]() -> ([443]); // 1403
                [84]() -> ([444]); // 1404
                [60]([444]) -> ([444]); // 1405
                [3]([443], [444]) -> ([445]); // 1406
                [2]() -> ([446]); // 1407
                [1]([446], [445]) -> ([447]); // 1408
                [17]([447]) -> ([448]); // 1409
                [57]([435]) -> ([435]); // 1410
                [85]([448]) -> ([448]); // 1411
                return([435], [448]); // 1412
                [35]() -> (); // 1413
                [70]([10]) -> (); // 1414
                [77]([9]) -> (); // 1415
                [70]([2]) -> (); // 1416
                [71]([3]) -> (); // 1417
                [71]([11]) -> (); // 1418
                [6]([429]) -> ([449], [450]); // 1419
                [78]([449]) -> (); // 1420
                [57]([426]) -> ([420]); // 1421
                [88]([450]) -> ([421]); // 1422
                [2]() -> ([451]); // 1423
                [1]([451], [421]) -> ([452]); // 1424
                [17]([452]) -> ([453]); // 1425
                [57]([420]) -> ([420]); // 1426
                [85]([453]) -> ([453]); // 1427
                return([420], [453]); // 1428
                [35]() -> (); // 1429
                [71]([17]) -> (); // 1430
                [77]([9]) -> (); // 1431
                [70]([2]) -> (); // 1432
                [71]([11]) -> (); // 1433
                [70]([10]) -> (); // 1434
                [71]([3]) -> (); // 1435
                [4]() -> ([454]); // 1436
                [84]() -> ([455]); // 1437
                [60]([455]) -> ([455]); // 1438
                [3]([454], [455]) -> ([456]); // 1439
                [2]() -> ([457]); // 1440
                [1]([457], [456]) -> ([458]); // 1441
                [17]([458]) -> ([459]); // 1442
                [57]([16]) -> ([16]); // 1443
                [85]([459]) -> ([459]); // 1444
                return([16], [459]); // 1445
                [106]() -> (); // 1446
                [157]([0], [1]) { fallthrough([4], [5]) 1478([6], [7]) }; // 1447
                [35]() -> (); // 1448
                [5]([5]) -> ([8]); // 1449
                [156]([2]) -> ([9]); // 1450
                [57]([4]) -> ([4]); // 1451
                [145]([8]) -> ([8]); // 1452
                [155]([9]) { fallthrough([10], [11]) 1468([12]) }; // 1453
                [35]() -> (); // 1454
                [5]([8]) -> ([13]); // 1455
                [154]([11]) -> ([14]); // 1456
                [158]([14]) -> ([15]); // 1457
                [153]([15]) -> ([16]); // 1458
                [60]([16]) -> ([16]); // 1459
                [3]([3], [16]) -> ([17]); // 1460
                [12]([10]) -> ([18]); // 1461
                [57]([4]) -> ([4]); // 1462
                [145]([13]) -> ([13]); // 1463
                [149]([18]) -> ([18]); // 1464
                [88]([17]) -> ([17]); // 1465
                [11]([4], [13], [18], [17]) -> ([19], [20], [21]); // 1466
                return([19], [20], [21]); // 1467
                [35]() -> (); // 1468
                [136]([12]) -> (); // 1469
                [5]([8]) -> ([22]); // 1470
                [20]() -> ([23]); // 1471
                [152]([3], [23]) -> ([24]); // 1472
                [151]([24]) -> ([25]); // 1473
                [57]([4]) -> ([4]); // 1474
                [145]([22]) -> ([22]); // 1475
                [162]([25]) -> ([25]); // 1476
                return([4], [22], [25]); // 1477
                [35]() -> (); // 1478
                [159]([2]) -> (); // 1479
                [160]([3]) -> (); // 1480
                [5]([7]) -> ([26]); // 1481
                [4]() -> ([27]); // 1482
                [161]() -> ([28]); // 1483
                [60]([28]) -> ([28]); // 1484
                [3]([27], [28]) -> ([29]); // 1485
                [2]() -> ([30]); // 1486
                [1]([30], [29]) -> ([31]); // 1487
                [150]([31]) -> ([32]); // 1488
                [57]([6]) -> ([6]); // 1489
                [145]([26]) -> ([26]); // 1490
                [162]([32]) -> ([32]); // 1491
                return([6], [26], [32]); // 1492
                [34]([0], [1]) { fallthrough([2], [3]) 1583([4]) }; // 1493
                [35]() -> (); // 1494
                [36]([3]) -> ([5]); // 1495
                [56]([5]) -> ([5]); // 1496
                [57]([2]) -> ([2]); // 1497
                [33]([5]) { fallthrough([6]) 1504([7]) 1509([8]) 1514([9]) 1519([10]) 1524([11]) 1529([12]) 1534([13]) 1539([14]) 1544([15]) 1549([16]) 1554([17]) 1559([18]) 1564([19]) 1569([20]) 1574([21]) }; // 1498
                [35]() -> (); // 1499
                [37]([6]) -> (); // 1500
                [38]() -> ([22]); // 1501
                [58]([22]) -> ([23]); // 1502
                [39]() { 1578() }; // 1503
                [35]() -> (); // 1504
                [37]([7]) -> (); // 1505
                [40]() -> ([24]); // 1506
                [58]([24]) -> ([23]); // 1507
                [39]() { 1578() }; // 1508
                [35]() -> (); // 1509
                [37]([8]) -> (); // 1510
                [41]() -> ([25]); // 1511
                [58]([25]) -> ([23]); // 1512
                [39]() { 1578() }; // 1513
                [35]() -> (); // 1514
                [37]([9]) -> (); // 1515
                [42]() -> ([26]); // 1516
                [58]([26]) -> ([23]); // 1517
                [39]() { 1578() }; // 1518
                [35]() -> (); // 1519
                [37]([10]) -> (); // 1520
                [43]() -> ([27]); // 1521
                [58]([27]) -> ([23]); // 1522
                [39]() { 1578() }; // 1523
                [35]() -> (); // 1524
                [37]([11]) -> (); // 1525
                [44]() -> ([28]); // 1526
                [58]([28]) -> ([23]); // 1527
                [39]() { 1578() }; // 1528
                [35]() -> (); // 1529
                [37]([12]) -> (); // 1530
                [45]() -> ([29]); // 1531
                [58]([29]) -> ([23]); // 1532
                [39]() { 1578() }; // 1533
                [35]() -> (); // 1534
                [37]([13]) -> (); // 1535
                [46]() -> ([30]); // 1536
                [58]([30]) -> ([23]); // 1537
                [39]() { 1578() }; // 1538
                [35]() -> (); // 1539
                [37]([14]) -> (); // 1540
                [47]() -> ([31]); // 1541
                [58]([31]) -> ([23]); // 1542
                [39]() { 1578() }; // 1543
                [35]() -> (); // 1544
                [37]([15]) -> (); // 1545
                [48]() -> ([32]); // 1546
                [58]([32]) -> ([23]); // 1547
                [39]() { 1578() }; // 1548
                [35]() -> (); // 1549
                [37]([16]) -> (); // 1550
                [49]() -> ([33]); // 1551
                [58]([33]) -> ([23]); // 1552
                [39]() { 1578() }; // 1553
                [35]() -> (); // 1554
                [37]([17]) -> (); // 1555
                [50]() -> ([34]); // 1556
                [58]([34]) -> ([23]); // 1557
                [39]() { 1578() }; // 1558
                [35]() -> (); // 1559
                [37]([18]) -> (); // 1560
                [51]() -> ([35]); // 1561
                [58]([35]) -> ([23]); // 1562
                [39]() { 1578() }; // 1563
                [35]() -> (); // 1564
                [37]([19]) -> (); // 1565
                [52]() -> ([36]); // 1566
                [58]([36]) -> ([23]); // 1567
                [39]() { 1578() }; // 1568
                [35]() -> (); // 1569
                [37]([20]) -> (); // 1570
                [53]() -> ([37]); // 1571
                [58]([37]) -> ([23]); // 1572
                [39]() { 1578() }; // 1573
                [35]() -> (); // 1574
                [37]([21]) -> (); // 1575
                [54]() -> ([38]); // 1576
                [58]([38]) -> ([23]); // 1577
                [32]([23]) -> ([39]); // 1578
                [31]([39]) -> ([40]); // 1579
                [57]([2]) -> ([2]); // 1580
                [59]([40]) -> ([40]); // 1581
                return([2], [40]); // 1582
                [35]() -> (); // 1583
                [4]() -> ([41]); // 1584
                [55]() -> ([42]); // 1585
                [60]([42]) -> ([42]); // 1586
                [3]([41], [42]) -> ([43]); // 1587
                [2]() -> ([44]); // 1588
                [1]([44], [43]) -> ([45]); // 1589
                [30]([45]) -> ([46]); // 1590
                [57]([4]) -> ([4]); // 1591
                [59]([46]) -> ([46]); // 1592
                return([4], [46]); // 1593

                [3]@0([0]: [0], [1]: [1]) -> ([0], [1], [16]);
                [1]@604([0]: [0], [1]: [19], [2]: [2], [3]: [10]) -> ([0], [21]);
                [0]@1446([0]: [0], [1]: [1], [2]: [24], [3]: [14]) -> ([0], [1], [26]);
                [2]@1493([0]: [0], [1]: [10]) -> ([0], [31]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let result = run_sierra_program(&program, &[]).return_value;

        assert_eq!(result, jit_enum!(0, jit_struct!(jit_struct!())));
    }

    #[test]
    fn seq_append1() {
        // use array::ArrayTrait;
        // fn run_test() -> Array<u32> {
        //     let mut data = ArrayTrait::new();
        //     data.append(1);
        //     data
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [1] = Array<[0]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [2] = Const<[0], 1> [storable: false, drop: false, dup: false, zero_sized: false];
                type [0] = u32 [storable: true, drop: true, dup: true, zero_sized: false];

                libfunc [1] = array_new<[0]>;
                libfunc [3] = const_as_immediate<[2]>;
                libfunc [4] = store_temp<[0]>;
                libfunc [0] = array_append<[0]>;
                libfunc [5] = store_temp<[1]>;

                [1]() -> ([0]); // 0
                [3]() -> ([1]); // 1
                [4]([1]) -> ([1]); // 2
                [0]([0], [1]) -> ([2]); // 3
                [5]([2]) -> ([2]); // 4
                return([2]); // 5

                [0]@0() -> ([1]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        assert_eq!(
            run_sierra_program(&program, &[]).return_value,
            Value::from([1u32]),
        );
    }

    #[test]
    fn seq_append2() {
        // use array::ArrayTrait;
        // fn run_test() -> Array<u32> {
        //     let mut data = ArrayTrait::new();
        //     data.append(1);
        //     data.append(2);
        //     data
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [1] = Array<[0]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [3] = Const<[0], 2> [storable: false, drop: false, dup: false, zero_sized: false];
                type [2] = Const<[0], 1> [storable: false, drop: false, dup: false, zero_sized: false];
                type [0] = u32 [storable: true, drop: true, dup: true, zero_sized: false];

                libfunc [1] = array_new<[0]>;
                libfunc [3] = const_as_immediate<[2]>;
                libfunc [5] = store_temp<[0]>;
                libfunc [0] = array_append<[0]>;
                libfunc [4] = const_as_immediate<[3]>;
                libfunc [6] = store_temp<[1]>;

                [1]() -> ([0]); // 0
                [3]() -> ([1]); // 1
                [5]([1]) -> ([1]); // 2
                [0]([0], [1]) -> ([2]); // 3
                [4]() -> ([3]); // 4
                [5]([3]) -> ([3]); // 5
                [0]([2], [3]) -> ([4]); // 6
                [6]([4]) -> ([4]); // 7
                return([4]); // 8

                [0]@0() -> ([1]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        assert_eq!(
            run_sierra_program(&program, &[]).return_value,
            Value::from([1u32, 2u32]),
        );
    }

    #[test]
    fn seq_append2_popf1() {
        // use array::ArrayTrait;
        // fn run_test() -> Array<u32> {
        //     let mut data = ArrayTrait::new();
        //     data.append(1);
        //     data.append(2);
        //     let _ = data.pop_front();
        //     data
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [1] = Array<[0]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [2] = Box<[0]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [4] = Const<[0], 2> [storable: false, drop: false, dup: false, zero_sized: false];
                type [3] = Const<[0], 1> [storable: false, drop: false, dup: false, zero_sized: false];
                type [0] = u32 [storable: true, drop: true, dup: true, zero_sized: false];

                libfunc [3] = array_new<[0]>;
                libfunc [5] = const_as_immediate<[3]>;
                libfunc [9] = store_temp<[0]>;
                libfunc [2] = array_append<[0]>;
                libfunc [6] = const_as_immediate<[4]>;
                libfunc [10] = store_temp<[1]>;
                libfunc [1] = array_pop_front<[0]>;
                libfunc [7] = branch_align;
                libfunc [0] = unbox<[0]>;
                libfunc [8] = drop<[0]>;

                [3]() -> ([0]); // 0
                [5]() -> ([1]); // 1
                [9]([1]) -> ([1]); // 2
                [2]([0], [1]) -> ([2]); // 3
                [6]() -> ([3]); // 4
                [9]([3]) -> ([3]); // 5
                [2]([2], [3]) -> ([4]); // 6
                [10]([4]) -> ([4]); // 7
                [1]([4]) { fallthrough([5], [6]) 14([7]) }; // 8
                [7]() -> (); // 9
                [0]([6]) -> ([8]); // 10
                [8]([8]) -> (); // 11
                [10]([5]) -> ([5]); // 12
                return([5]); // 13
                [7]() -> (); // 14
                [10]([7]) -> ([7]); // 15
                return([7]); // 16

                [0]@0() -> ([1]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        assert_eq!(
            run_sierra_program(&program, &[]).return_value,
            Value::from([2u32]),
        );
    }

    #[test]
    fn seq_append2_popb1() {
        // use array::ArrayTrait;
        // fn run_test() -> Span<u32> {
        //     let mut data = ArrayTrait::new();
        //     data.append(1);
        //     data.append(2);
        //     let mut data = data.span();
        //     let _ = data.pop_back();
        //     data
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [1] = Array<[0]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [2] = Snapshot<[1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [6] = Struct<ut@core::array::Span::<core::integer::u32>, [2]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [4] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
                type [3] = Box<[0]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [5] = Enum<ut@core::option::Option::<core::box::Box::<@core::integer::u32>>, [3], [4]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [8] = Const<[0], 2> [storable: false, drop: false, dup: false, zero_sized: false];
                type [7] = Const<[0], 1> [storable: false, drop: false, dup: false, zero_sized: false];
                type [0] = u32 [storable: true, drop: true, dup: true, zero_sized: false];

                libfunc [8] = array_new<[0]>;
                libfunc [10] = const_as_immediate<[7]>;
                libfunc [18] = store_temp<[0]>;
                libfunc [7] = array_append<[0]>;
                libfunc [11] = const_as_immediate<[8]>;
                libfunc [12] = snapshot_take<[1]>;
                libfunc [13] = drop<[1]>;
                libfunc [19] = store_temp<[2]>;
                libfunc [6] = array_snapshot_pop_back<[0]>;
                libfunc [14] = branch_align;
                libfunc [5] = enum_init<[5], 0>;
                libfunc [20] = store_temp<[5]>;
                libfunc [15] = jump;
                libfunc [4] = struct_construct<[4]>;
                libfunc [3] = enum_init<[5], 1>;
                libfunc [2] = struct_construct<[6]>;
                libfunc [1] = enum_match<[5]>;
                libfunc [0] = unbox<[0]>;
                libfunc [16] = drop<[0]>;
                libfunc [21] = store_temp<[6]>;
                libfunc [17] = drop<[4]>;

                [8]() -> ([0]); // 0
                [10]() -> ([1]); // 1
                [18]([1]) -> ([1]); // 2
                [7]([0], [1]) -> ([2]); // 3
                [11]() -> ([3]); // 4
                [18]([3]) -> ([3]); // 5
                [7]([2], [3]) -> ([4]); // 6
                [12]([4]) -> ([5], [6]); // 7
                [13]([5]) -> (); // 8
                [19]([6]) -> ([6]); // 9
                [6]([6]) { fallthrough([7], [8]) 16([9]) }; // 10
                [14]() -> (); // 11
                [5]([8]) -> ([10]); // 12
                [19]([7]) -> ([11]); // 13
                [20]([10]) -> ([12]); // 14
                [15]() { 21() }; // 15
                [14]() -> (); // 16
                [4]() -> ([13]); // 17
                [3]([13]) -> ([14]); // 18
                [19]([9]) -> ([11]); // 19
                [20]([14]) -> ([12]); // 20
                [2]([11]) -> ([15]); // 21
                [1]([12]) { fallthrough([16]) 28([17]) }; // 22
                [14]() -> (); // 23
                [0]([16]) -> ([18]); // 24
                [16]([18]) -> (); // 25
                [21]([15]) -> ([15]); // 26
                return([15]); // 27
                [14]() -> (); // 28
                [17]([17]) -> (); // 29
                [21]([15]) -> ([15]); // 30
                return([15]); // 31

                [0]@0() -> ([6]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        assert_eq!(
            run_sierra_program(&program, &[]).return_value,
            jit_struct!([1u32].into())
        );
    }

    #[test]
    fn seq_append1_popf1_append1() {
        // use array::ArrayTrait;
        // fn run_test() -> Array<u32> {
        //     let mut data = ArrayTrait::new();
        //     data.append(1);
        //     let _ = data.pop_front();
        //     data.append(2);
        //     data
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [1] = Array<[0]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [4] = Const<[0], 2> [storable: false, drop: false, dup: false, zero_sized: false];
                type [2] = Box<[0]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [3] = Const<[0], 1> [storable: false, drop: false, dup: false, zero_sized: false];
                type [0] = u32 [storable: true, drop: true, dup: true, zero_sized: false];

                libfunc [3] = array_new<[0]>;
                libfunc [5] = const_as_immediate<[3]>;
                libfunc [10] = store_temp<[0]>;
                libfunc [0] = array_append<[0]>;
                libfunc [11] = store_temp<[1]>;
                libfunc [2] = array_pop_front<[0]>;
                libfunc [6] = branch_align;
                libfunc [1] = unbox<[0]>;
                libfunc [7] = drop<[0]>;
                libfunc [8] = jump;
                libfunc [9] = const_as_immediate<[4]>;

                [3]() -> ([0]); // 0
                [5]() -> ([1]); // 1
                [10]([1]) -> ([1]); // 2
                [0]([0], [1]) -> ([2]); // 3
                [11]([2]) -> ([2]); // 4
                [2]([2]) { fallthrough([3], [4]) 11([5]) }; // 5
                [6]() -> (); // 6
                [1]([4]) -> ([6]); // 7
                [7]([6]) -> (); // 8
                [11]([3]) -> ([7]); // 9
                [8]() { 13() }; // 10
                [6]() -> (); // 11
                [11]([5]) -> ([7]); // 12
                [9]() -> ([8]); // 13
                [10]([8]) -> ([8]); // 14
                [0]([7], [8]) -> ([9]); // 15
                [11]([9]) -> ([9]); // 16
                return([9]); // 17

                [0]@0() -> ([1]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        assert_eq!(
            run_sierra_program(&program, &[]).return_value,
            Value::from([2u32]),
        );
    }

    #[test]
    fn seq_append1_first() {
        // use array::ArrayTrait;
        // fn run_test() -> u32 {
        //     let mut data = ArrayTrait::new();
        //     data.append(1);
        //     *data.at(0)
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [1] = Array<[0]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [5] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
                type [7] = Array<[6]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [8] = Struct<ut@Tuple, [5], [7]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [11] = Const<[6], 1637570914057682275393755530660268060279989363> [storable: false, drop: false, dup: false, zero_sized: false];
                type [6] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
                type [0] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
                type [4] = Struct<ut@Tuple, [0]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [9] = Enum<ut@core::panics::PanicResult::<(core::integer::u32,)>, [4], [8]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [3] = Box<[0]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [2] = Snapshot<[1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [10] = Const<[0], 1> [storable: false, drop: false, dup: false, zero_sized: false];

                libfunc [10] = array_new<[0]>;
                libfunc [12] = const_as_immediate<[10]>;
                libfunc [19] = store_temp<[0]>;
                libfunc [9] = array_append<[0]>;
                libfunc [13] = snapshot_take<[1]>;
                libfunc [14] = drop<[1]>;
                libfunc [20] = store_temp<[2]>;
                libfunc [8] = array_snapshot_pop_front<[0]>;
                libfunc [15] = branch_align;
                libfunc [16] = drop<[2]>;
                libfunc [7] = unbox<[0]>;
                libfunc [17] = rename<[0]>;
                libfunc [6] = struct_construct<[4]>;
                libfunc [5] = enum_init<[9], 0>;
                libfunc [21] = store_temp<[9]>;
                libfunc [4] = array_new<[6]>;
                libfunc [18] = const_as_immediate<[11]>;
                libfunc [22] = store_temp<[6]>;
                libfunc [3] = array_append<[6]>;
                libfunc [2] = struct_construct<[5]>;
                libfunc [1] = struct_construct<[8]>;
                libfunc [0] = enum_init<[9], 1>;

                [10]() -> ([0]); // 0
                [12]() -> ([1]); // 1
                [19]([1]) -> ([1]); // 2
                [9]([0], [1]) -> ([2]); // 3
                [13]([2]) -> ([3], [4]); // 4
                [14]([3]) -> (); // 5
                [20]([4]) -> ([4]); // 6
                [8]([4]) { fallthrough([5], [6]) 16([7]) }; // 7
                [15]() -> (); // 8
                [16]([5]) -> (); // 9
                [7]([6]) -> ([8]); // 10
                [17]([8]) -> ([9]); // 11
                [6]([9]) -> ([10]); // 12
                [5]([10]) -> ([11]); // 13
                [21]([11]) -> ([11]); // 14
                return([11]); // 15
                [15]() -> (); // 16
                [16]([7]) -> (); // 17
                [4]() -> ([12]); // 18
                [18]() -> ([13]); // 19
                [22]([13]) -> ([13]); // 20
                [3]([12], [13]) -> ([14]); // 21
                [2]() -> ([15]); // 22
                [1]([15], [14]) -> ([16]); // 23
                [0]([16]) -> ([17]); // 24
                [21]([17]) -> ([17]); // 25
                return([17]); // 26

                [0]@0() -> ([9]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        assert_eq!(
            run_sierra_program(&program, &[]).return_value,
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
        // use array::ArrayTrait;
        // fn run_test() -> u32 {
        //     let mut data = ArrayTrait::new();
        //     data.append(1);
        //     data.append(2);
        //     *data.at(0)
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [1] = Array<[0]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [5] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
                type [7] = Array<[6]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [8] = Struct<ut@Tuple, [5], [7]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [12] = Const<[6], 1637570914057682275393755530660268060279989363> [storable: false, drop: false, dup: false, zero_sized: false];
                type [6] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
                type [0] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
                type [4] = Struct<ut@Tuple, [0]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [9] = Enum<ut@core::panics::PanicResult::<(core::integer::u32,)>, [4], [8]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [3] = Box<[0]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [2] = Snapshot<[1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [11] = Const<[0], 2> [storable: false, drop: false, dup: false, zero_sized: false];
                type [10] = Const<[0], 1> [storable: false, drop: false, dup: false, zero_sized: false];

                libfunc [10] = array_new<[0]>;
                libfunc [12] = const_as_immediate<[10]>;
                libfunc [20] = store_temp<[0]>;
                libfunc [9] = array_append<[0]>;
                libfunc [13] = const_as_immediate<[11]>;
                libfunc [14] = snapshot_take<[1]>;
                libfunc [15] = drop<[1]>;
                libfunc [21] = store_temp<[2]>;
                libfunc [8] = array_snapshot_pop_front<[0]>;
                libfunc [16] = branch_align;
                libfunc [17] = drop<[2]>;
                libfunc [7] = unbox<[0]>;
                libfunc [18] = rename<[0]>;
                libfunc [6] = struct_construct<[4]>;
                libfunc [5] = enum_init<[9], 0>;
                libfunc [22] = store_temp<[9]>;
                libfunc [4] = array_new<[6]>;
                libfunc [19] = const_as_immediate<[12]>;
                libfunc [23] = store_temp<[6]>;
                libfunc [3] = array_append<[6]>;
                libfunc [2] = struct_construct<[5]>;
                libfunc [1] = struct_construct<[8]>;
                libfunc [0] = enum_init<[9], 1>;

                [10]() -> ([0]); // 0
                [12]() -> ([1]); // 1
                [20]([1]) -> ([1]); // 2
                [9]([0], [1]) -> ([2]); // 3
                [13]() -> ([3]); // 4
                [20]([3]) -> ([3]); // 5
                [9]([2], [3]) -> ([4]); // 6
                [14]([4]) -> ([5], [6]); // 7
                [15]([5]) -> (); // 8
                [21]([6]) -> ([6]); // 9
                [8]([6]) { fallthrough([7], [8]) 19([9]) }; // 10
                [16]() -> (); // 11
                [17]([7]) -> (); // 12
                [7]([8]) -> ([10]); // 13
                [18]([10]) -> ([11]); // 14
                [6]([11]) -> ([12]); // 15
                [5]([12]) -> ([13]); // 16
                [22]([13]) -> ([13]); // 17
                return([13]); // 18
                [16]() -> (); // 19
                [17]([9]) -> (); // 20
                [4]() -> ([14]); // 21
                [19]() -> ([15]); // 22
                [23]([15]) -> ([15]); // 23
                [3]([14], [15]) -> ([16]); // 24
                [2]() -> ([17]); // 25
                [1]([17], [16]) -> ([18]); // 26
                [0]([18]) -> ([19]); // 27
                [22]([19]) -> ([19]); // 28
                return([19]); // 29

                [0]@0() -> ([9]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        assert_eq!(
            run_sierra_program(&program, &[]).return_value,
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
        // use array::ArrayTrait;
        // fn run_test() -> u32 {
        //     let mut data = ArrayTrait::new();
        //     data.append(1);
        //     data.append(2);
        //     let _ = data.pop_front();
        //     *data.at(0)
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [1] = Array<[0]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [5] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
                type [7] = Array<[6]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [8] = Struct<ut@Tuple, [5], [7]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [12] = Const<[6], 1637570914057682275393755530660268060279989363> [storable: false, drop: false, dup: false, zero_sized: false];
                type [6] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
                type [0] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
                type [4] = Struct<ut@Tuple, [0]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [9] = Enum<ut@core::panics::PanicResult::<(core::integer::u32,)>, [4], [8]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [3] = Snapshot<[1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [2] = Box<[0]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [11] = Const<[0], 2> [storable: false, drop: false, dup: false, zero_sized: false];
                type [10] = Const<[0], 1> [storable: false, drop: false, dup: false, zero_sized: false];

                libfunc [11] = array_new<[0]>;
                libfunc [13] = const_as_immediate<[10]>;
                libfunc [23] = store_temp<[0]>;
                libfunc [10] = array_append<[0]>;
                libfunc [14] = const_as_immediate<[11]>;
                libfunc [24] = store_temp<[1]>;
                libfunc [9] = array_pop_front<[0]>;
                libfunc [15] = branch_align;
                libfunc [7] = unbox<[0]>;
                libfunc [16] = drop<[0]>;
                libfunc [17] = jump;
                libfunc [18] = snapshot_take<[1]>;
                libfunc [19] = drop<[1]>;
                libfunc [8] = array_snapshot_pop_front<[0]>;
                libfunc [20] = drop<[3]>;
                libfunc [21] = rename<[0]>;
                libfunc [6] = struct_construct<[4]>;
                libfunc [5] = enum_init<[9], 0>;
                libfunc [25] = store_temp<[9]>;
                libfunc [4] = array_new<[6]>;
                libfunc [22] = const_as_immediate<[12]>;
                libfunc [26] = store_temp<[6]>;
                libfunc [3] = array_append<[6]>;
                libfunc [2] = struct_construct<[5]>;
                libfunc [1] = struct_construct<[8]>;
                libfunc [0] = enum_init<[9], 1>;

                [11]() -> ([0]); // 0
                [13]() -> ([1]); // 1
                [23]([1]) -> ([1]); // 2
                [10]([0], [1]) -> ([2]); // 3
                [14]() -> ([3]); // 4
                [23]([3]) -> ([3]); // 5
                [10]([2], [3]) -> ([4]); // 6
                [24]([4]) -> ([4]); // 7
                [9]([4]) { fallthrough([5], [6]) 14([7]) }; // 8
                [15]() -> (); // 9
                [7]([6]) -> ([8]); // 10
                [16]([8]) -> (); // 11
                [24]([5]) -> ([9]); // 12
                [17]() { 16() }; // 13
                [15]() -> (); // 14
                [24]([7]) -> ([9]); // 15
                [18]([9]) -> ([10], [11]); // 16
                [19]([10]) -> (); // 17
                [8]([11]) { fallthrough([12], [13]) 27([14]) }; // 18
                [15]() -> (); // 19
                [20]([12]) -> (); // 20
                [7]([13]) -> ([15]); // 21
                [21]([15]) -> ([16]); // 22
                [6]([16]) -> ([17]); // 23
                [5]([17]) -> ([18]); // 24
                [25]([18]) -> ([18]); // 25
                return([18]); // 26
                [15]() -> (); // 27
                [20]([14]) -> (); // 28
                [4]() -> ([19]); // 29
                [22]() -> ([20]); // 30
                [26]([20]) -> ([20]); // 31
                [3]([19], [20]) -> ([21]); // 32
                [2]() -> ([22]); // 33
                [1]([22], [21]) -> ([23]); // 34
                [0]([23]) -> ([24]); // 35
                [25]([24]) -> ([24]); // 36
                return([24]); // 37

                [0]@0() -> ([9]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        assert_eq!(
            run_sierra_program(&program, &[]).return_value,
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
        // use array::ArrayTrait;
        // fn run_test() -> u32 {
        //     let mut data = ArrayTrait::new();
        //     data.append(1);
        //     data.append(2);
        //     let mut data_span = data.span();
        //     let _ = data_span.pop_back();
        //     let last = data_span.len() - 1;
        //     *data_span.at(last)
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [2] = Array<[1]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [16] = Const<[9], 155785504329508738615720351733824384887> [storable: false, drop: false, dup: false, zero_sized: false];
                type [8] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
                type [10] = Array<[9]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [11] = Struct<ut@Tuple, [8], [10]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [15] = Const<[9], 1637570914057682275393755530660268060279989363> [storable: false, drop: false, dup: false, zero_sized: false];
                type [9] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
                type [1] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
                type [7] = Struct<ut@Tuple, [1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [12] = Enum<ut@core::panics::PanicResult::<(core::integer::u32,)>, [7], [11]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
                type [5] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
                type [4] = Box<[1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [6] = Enum<ut@core::option::Option::<core::box::Box::<@core::integer::u32>>, [4], [5]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [3] = Snapshot<[2]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [14] = Const<[1], 2> [storable: false, drop: false, dup: false, zero_sized: false];
                type [13] = Const<[1], 1> [storable: false, drop: false, dup: false, zero_sized: false];

                libfunc [17] = array_new<[1]>;
                libfunc [19] = const_as_immediate<[13]>;
                libfunc [32] = store_temp<[1]>;
                libfunc [16] = array_append<[1]>;
                libfunc [20] = const_as_immediate<[14]>;
                libfunc [21] = snapshot_take<[2]>;
                libfunc [22] = drop<[2]>;
                libfunc [33] = store_temp<[3]>;
                libfunc [15] = array_snapshot_pop_back<[1]>;
                libfunc [23] = branch_align;
                libfunc [14] = enum_init<[6], 0>;
                libfunc [34] = store_temp<[6]>;
                libfunc [24] = jump;
                libfunc [13] = struct_construct<[5]>;
                libfunc [12] = enum_init<[6], 1>;
                libfunc [11] = enum_match<[6]>;
                libfunc [7] = unbox<[1]>;
                libfunc [25] = drop<[1]>;
                libfunc [26] = drop<[5]>;
                libfunc [27] = dup<[3]>;
                libfunc [10] = array_len<[1]>;
                libfunc [9] = u32_overflowing_sub;
                libfunc [8] = array_get<[1]>;
                libfunc [35] = store_temp<[4]>;
                libfunc [28] = rename<[1]>;
                libfunc [6] = struct_construct<[7]>;
                libfunc [5] = enum_init<[12], 0>;
                libfunc [36] = store_temp<[0]>;
                libfunc [37] = store_temp<[12]>;
                libfunc [4] = array_new<[9]>;
                libfunc [29] = const_as_immediate<[15]>;
                libfunc [38] = store_temp<[9]>;
                libfunc [3] = array_append<[9]>;
                libfunc [2] = struct_construct<[8]>;
                libfunc [1] = struct_construct<[11]>;
                libfunc [0] = enum_init<[12], 1>;
                libfunc [30] = drop<[3]>;
                libfunc [31] = const_as_immediate<[16]>;

                [17]() -> ([1]); // 0
                [19]() -> ([2]); // 1
                [32]([2]) -> ([2]); // 2
                [16]([1], [2]) -> ([3]); // 3
                [20]() -> ([4]); // 4
                [32]([4]) -> ([4]); // 5
                [16]([3], [4]) -> ([5]); // 6
                [21]([5]) -> ([6], [7]); // 7
                [22]([6]) -> (); // 8
                [33]([7]) -> ([7]); // 9
                [15]([7]) { fallthrough([8], [9]) 16([10]) }; // 10
                [23]() -> (); // 11
                [14]([9]) -> ([11]); // 12
                [33]([8]) -> ([12]); // 13
                [34]([11]) -> ([13]); // 14
                [24]() { 21() }; // 15
                [23]() -> (); // 16
                [13]() -> ([14]); // 17
                [12]([14]) -> ([15]); // 18
                [33]([10]) -> ([12]); // 19
                [34]([15]) -> ([13]); // 20
                [11]([13]) { fallthrough([16]) 26([17]) }; // 21
                [23]() -> (); // 22
                [7]([16]) -> ([18]); // 23
                [25]([18]) -> (); // 24
                [24]() { 28() }; // 25
                [23]() -> (); // 26
                [26]([17]) -> (); // 27
                [27]([12]) -> ([12], [19]); // 28
                [10]([19]) -> ([20]); // 29
                [19]() -> ([21]); // 30
                [32]([20]) -> ([20]); // 31
                [32]([21]) -> ([21]); // 32
                [9]([0], [20], [21]) { fallthrough([22], [23]) 56([24], [25]) }; // 33
                [23]() -> (); // 34
                [8]([22], [12], [23]) { fallthrough([26], [27]) 45([28]) }; // 35
                [23]() -> (); // 36
                [35]([27]) -> ([27]); // 37
                [7]([27]) -> ([29]); // 38
                [28]([29]) -> ([30]); // 39
                [6]([30]) -> ([31]); // 40
                [5]([31]) -> ([32]); // 41
                [36]([26]) -> ([26]); // 42
                [37]([32]) -> ([32]); // 43
                return([26], [32]); // 44
                [23]() -> (); // 45
                [4]() -> ([33]); // 46
                [29]() -> ([34]); // 47
                [38]([34]) -> ([34]); // 48
                [3]([33], [34]) -> ([35]); // 49
                [2]() -> ([36]); // 50
                [1]([36], [35]) -> ([37]); // 51
                [0]([37]) -> ([38]); // 52
                [36]([28]) -> ([28]); // 53
                [37]([38]) -> ([38]); // 54
                return([28], [38]); // 55
                [23]() -> (); // 56
                [25]([25]) -> (); // 57
                [30]([12]) -> (); // 58
                [4]() -> ([39]); // 59
                [31]() -> ([40]); // 60
                [38]([40]) -> ([40]); // 61
                [3]([39], [40]) -> ([41]); // 62
                [2]() -> ([42]); // 63
                [1]([42], [41]) -> ([43]); // 64
                [0]([43]) -> ([44]); // 65
                [36]([24]) -> ([24]); // 66
                [37]([44]) -> ([44]); // 67
                return([24], [44]); // 68

                [0]@0([0]: [0]) -> ([0], [12]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        assert_eq!(
            run_sierra_program(&program, &[]).return_value,
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
        // use array::ArrayTrait;
        // fn run_test() -> u32 {
        //     let mut data = ArrayTrait::new();
        //     data.append(1);
        //     let _ = data.pop_front();
        //     data.append(2);
        //     *data.at(0)
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [1] = Array<[0]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [5] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
                type [7] = Array<[6]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [8] = Struct<ut@Tuple, [5], [7]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [12] = Const<[6], 1637570914057682275393755530660268060279989363> [storable: false, drop: false, dup: false, zero_sized: false];
                type [6] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
                type [0] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
                type [4] = Struct<ut@Tuple, [0]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [9] = Enum<ut@core::panics::PanicResult::<(core::integer::u32,)>, [4], [8]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [3] = Snapshot<[1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [11] = Const<[0], 2> [storable: false, drop: false, dup: false, zero_sized: false];
                type [2] = Box<[0]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [10] = Const<[0], 1> [storable: false, drop: false, dup: false, zero_sized: false];

                libfunc [11] = array_new<[0]>;
                libfunc [13] = const_as_immediate<[10]>;
                libfunc [23] = store_temp<[0]>;
                libfunc [9] = array_append<[0]>;
                libfunc [24] = store_temp<[1]>;
                libfunc [10] = array_pop_front<[0]>;
                libfunc [14] = branch_align;
                libfunc [7] = unbox<[0]>;
                libfunc [15] = drop<[0]>;
                libfunc [16] = jump;
                libfunc [17] = const_as_immediate<[11]>;
                libfunc [18] = snapshot_take<[1]>;
                libfunc [19] = drop<[1]>;
                libfunc [25] = store_temp<[3]>;
                libfunc [8] = array_snapshot_pop_front<[0]>;
                libfunc [20] = drop<[3]>;
                libfunc [21] = rename<[0]>;
                libfunc [6] = struct_construct<[4]>;
                libfunc [5] = enum_init<[9], 0>;
                libfunc [26] = store_temp<[9]>;
                libfunc [4] = array_new<[6]>;
                libfunc [22] = const_as_immediate<[12]>;
                libfunc [27] = store_temp<[6]>;
                libfunc [3] = array_append<[6]>;
                libfunc [2] = struct_construct<[5]>;
                libfunc [1] = struct_construct<[8]>;
                libfunc [0] = enum_init<[9], 1>;

                [11]() -> ([0]); // 0
                [13]() -> ([1]); // 1
                [23]([1]) -> ([1]); // 2
                [9]([0], [1]) -> ([2]); // 3
                [24]([2]) -> ([2]); // 4
                [10]([2]) { fallthrough([3], [4]) 11([5]) }; // 5
                [14]() -> (); // 6
                [7]([4]) -> ([6]); // 7
                [15]([6]) -> (); // 8
                [24]([3]) -> ([7]); // 9
                [16]() { 13() }; // 10
                [14]() -> (); // 11
                [24]([5]) -> ([7]); // 12
                [17]() -> ([8]); // 13
                [23]([8]) -> ([8]); // 14
                [9]([7], [8]) -> ([9]); // 15
                [18]([9]) -> ([10], [11]); // 16
                [19]([10]) -> (); // 17
                [25]([11]) -> ([11]); // 18
                [8]([11]) { fallthrough([12], [13]) 28([14]) }; // 19
                [14]() -> (); // 20
                [20]([12]) -> (); // 21
                [7]([13]) -> ([15]); // 22
                [21]([15]) -> ([16]); // 23
                [6]([16]) -> ([17]); // 24
                [5]([17]) -> ([18]); // 25
                [26]([18]) -> ([18]); // 26
                return([18]); // 27
                [14]() -> (); // 28
                [20]([14]) -> (); // 29
                [4]() -> ([19]); // 30
                [22]() -> ([20]); // 31
                [27]([20]) -> ([20]); // 32
                [3]([19], [20]) -> ([21]); // 33
                [2]() -> ([22]); // 34
                [1]([22], [21]) -> ([23]); // 35
                [0]([23]) -> ([24]); // 36
                [26]([24]) -> ([24]); // 37
                return([24]); // 38

                [0]@0() -> ([9]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        assert_eq!(
            run_sierra_program(&program, &[]).return_value,
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
        // fn run_test() -> Array<u32> {
        //     let x = ArrayTrait::new();
        //     x.clone()
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [3] = Array<[2]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [8] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
                type [17] = Const<[9], 375233589013918064796019> [storable: false, drop: false, dup: false, zero_sized: false];
                type [9] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
                type [10] = Array<[9]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [15] = Box<[2]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [6] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
                type [16] = Enum<ut@core::option::Option::<core::box::Box::<@core::integer::u32>>, [15], [6]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [13] = Struct<ut@Tuple, [3]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [11] = Struct<ut@Tuple, [8], [10]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [14] = Enum<ut@core::panics::PanicResult::<(core::array::Array::<core::integer::u32>,)>, [13], [11]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [4] = Snapshot<[3]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [5] = Struct<ut@core::array::Span::<core::integer::u32>, [4]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [7] = Struct<ut@Tuple, [5], [3], [6]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [12] = Enum<ut@core::panics::PanicResult::<(core::array::Span::<core::integer::u32>, core::array::Array::<core::integer::u32>, ())>, [7], [11]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [1] = GasBuiltin [storable: true, drop: false, dup: false, zero_sized: false];
                type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
                type [2] = u32 [storable: true, drop: true, dup: true, zero_sized: false];

                libfunc [10] = disable_ap_tracking;
                libfunc [8] = array_new<[2]>;
                libfunc [11] = snapshot_take<[3]>;
                libfunc [12] = drop<[3]>;
                libfunc [7] = struct_construct<[5]>;
                libfunc [16] = store_temp<[0]>;
                libfunc [17] = store_temp<[1]>;
                libfunc [18] = store_temp<[5]>;
                libfunc [19] = store_temp<[3]>;
                libfunc [6] = function_call<user@[0]>;
                libfunc [5] = enum_match<[12]>;
                libfunc [13] = branch_align;
                libfunc [1] = redeposit_gas;
                libfunc [4] = struct_deconstruct<[7]>;
                libfunc [14] = drop<[5]>;
                libfunc [15] = drop<[6]>;
                libfunc [3] = struct_construct<[13]>;
                libfunc [2] = enum_init<[14], 0>;
                libfunc [20] = store_temp<[14]>;
                libfunc [0] = enum_init<[14], 1>;
                libfunc [36] = withdraw_gas;
                libfunc [35] = struct_deconstruct<[5]>;
                libfunc [37] = enable_ap_tracking;
                libfunc [34] = array_snapshot_pop_front<[2]>;
                libfunc [33] = enum_init<[16], 0>;
                libfunc [41] = store_temp<[4]>;
                libfunc [42] = store_temp<[16]>;
                libfunc [38] = jump;
                libfunc [28] = struct_construct<[6]>;
                libfunc [32] = enum_init<[16], 1>;
                libfunc [31] = enum_match<[16]>;
                libfunc [30] = unbox<[2]>;
                libfunc [39] = rename<[2]>;
                libfunc [43] = store_temp<[2]>;
                libfunc [29] = array_append<[2]>;
                libfunc [27] = struct_construct<[7]>;
                libfunc [26] = enum_init<[12], 0>;
                libfunc [44] = store_temp<[12]>;
                libfunc [25] = array_new<[9]>;
                libfunc [40] = const_as_immediate<[17]>;
                libfunc [45] = store_temp<[9]>;
                libfunc [24] = array_append<[9]>;
                libfunc [23] = struct_construct<[8]>;
                libfunc [22] = struct_construct<[11]>;
                libfunc [21] = enum_init<[12], 1>;

                [10]() -> (); // 0
                [8]() -> ([2]); // 1
                [8]() -> ([3]); // 2
                [11]([2]) -> ([4], [5]); // 3
                [12]([4]) -> (); // 4
                [7]([5]) -> ([6]); // 5
                [16]([0]) -> ([0]); // 6
                [17]([1]) -> ([1]); // 7
                [18]([6]) -> ([6]); // 8
                [19]([3]) -> ([3]); // 9
                [6]([0], [1], [6], [3]) -> ([7], [8], [9]); // 10
                [5]([9]) { fallthrough([10]) 23([11]) }; // 11
                [13]() -> (); // 12
                [1]([8]) -> ([12]); // 13
                [4]([10]) -> ([13], [14], [15]); // 14
                [14]([13]) -> (); // 15
                [15]([15]) -> (); // 16
                [3]([14]) -> ([16]); // 17
                [2]([16]) -> ([17]); // 18
                [16]([7]) -> ([7]); // 19
                [17]([12]) -> ([12]); // 20
                [20]([17]) -> ([17]); // 21
                return([7], [12], [17]); // 22
                [13]() -> (); // 23
                [1]([8]) -> ([18]); // 24
                [0]([11]) -> ([19]); // 25
                [16]([7]) -> ([7]); // 26
                [17]([18]) -> ([18]); // 27
                [20]([19]) -> ([19]); // 28
                return([7], [18], [19]); // 29
                [10]() -> (); // 30
                [36]([0], [1]) { fallthrough([4], [5]) 79([6], [7]) }; // 31
                [13]() -> (); // 32
                [1]([5]) -> ([8]); // 33
                [35]([2]) -> ([9]); // 34
                [37]() -> (); // 35
                [16]([4]) -> ([4]); // 36
                [17]([8]) -> ([8]); // 37
                [34]([9]) { fallthrough([10], [11]) 46([12]) }; // 38
                [13]() -> (); // 39
                [1]([8]) -> ([13]); // 40
                [33]([11]) -> ([14]); // 41
                [17]([13]) -> ([15]); // 42
                [41]([10]) -> ([16]); // 43
                [42]([14]) -> ([17]); // 44
                [38]() { 53() }; // 45
                [13]() -> (); // 46
                [1]([8]) -> ([18]); // 47
                [28]() -> ([19]); // 48
                [32]([19]) -> ([20]); // 49
                [17]([18]) -> ([15]); // 50
                [41]([12]) -> ([16]); // 51
                [42]([20]) -> ([17]); // 52
                [7]([16]) -> ([21]); // 53
                [31]([17]) { fallthrough([22]) 68([23]) }; // 54
                [13]() -> (); // 55
                [10]() -> (); // 56
                [1]([15]) -> ([24]); // 57
                [30]([22]) -> ([25]); // 58
                [39]([25]) -> ([26]); // 59
                [43]([26]) -> ([26]); // 60
                [29]([3], [26]) -> ([27]); // 61
                [16]([4]) -> ([4]); // 62
                [17]([24]) -> ([24]); // 63
                [18]([21]) -> ([21]); // 64
                [19]([27]) -> ([27]); // 65
                [6]([4], [24], [21], [27]) -> ([28], [29], [30]); // 66
                return([28], [29], [30]); // 67
                [13]() -> (); // 68
                [10]() -> (); // 69
                [15]([23]) -> (); // 70
                [1]([15]) -> ([31]); // 71
                [28]() -> ([32]); // 72
                [27]([21], [3], [32]) -> ([33]); // 73
                [26]([33]) -> ([34]); // 74
                [16]([4]) -> ([4]); // 75
                [17]([31]) -> ([31]); // 76
                [44]([34]) -> ([34]); // 77
                return([4], [31], [34]); // 78
                [13]() -> (); // 79
                [14]([2]) -> (); // 80
                [12]([3]) -> (); // 81
                [1]([7]) -> ([35]); // 82
                [25]() -> ([36]); // 83
                [40]() -> ([37]); // 84
                [45]([37]) -> ([37]); // 85
                [24]([36], [37]) -> ([38]); // 86
                [23]() -> ([39]); // 87
                [22]([39], [38]) -> ([40]); // 88
                [21]([40]) -> ([41]); // 89
                [16]([6]) -> ([6]); // 90
                [17]([35]) -> ([35]); // 91
                [44]([41]) -> ([41]); // 92
                return([6], [35], [41]); // 93

                [1]@0([0]: [0], [1]: [1]) -> ([0], [1], [14]);
                [0]@30([0]: [0], [1]: [1], [2]: [5], [3]: [3]) -> ([0], [1], [12]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        assert_eq!(
            run_sierra_program(&program, &[]).return_value,
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
        // use array::ArrayTrait;
        // fn run_test() -> Span<u32> {
        //     let mut numbers = ArrayTrait::new();
        //     numbers.append(1_u32);
        //     numbers.append(2_u32);
        //     numbers.append(3_u32);
        //     let mut numbers = numbers.span();
        //     let _ = numbers.pop_back();
        //     numbers
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [1] = Array<[0]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [2] = Snapshot<[1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [6] = Struct<ut@core::array::Span::<core::integer::u32>, [2]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [4] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
                type [3] = Box<[0]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [5] = Enum<ut@core::option::Option::<core::box::Box::<@core::integer::u32>>, [3], [4]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [9] = Const<[0], 3> [storable: false, drop: false, dup: false, zero_sized: false];
                type [8] = Const<[0], 2> [storable: false, drop: false, dup: false, zero_sized: false];
                type [7] = Const<[0], 1> [storable: false, drop: false, dup: false, zero_sized: false];
                type [0] = u32 [storable: true, drop: true, dup: true, zero_sized: false];

                libfunc [8] = array_new<[0]>;
                libfunc [10] = const_as_immediate<[7]>;
                libfunc [19] = store_temp<[0]>;
                libfunc [7] = array_append<[0]>;
                libfunc [11] = const_as_immediate<[8]>;
                libfunc [12] = const_as_immediate<[9]>;
                libfunc [13] = snapshot_take<[1]>;
                libfunc [14] = drop<[1]>;
                libfunc [20] = store_temp<[2]>;
                libfunc [6] = array_snapshot_pop_back<[0]>;
                libfunc [15] = branch_align;
                libfunc [5] = enum_init<[5], 0>;
                libfunc [21] = store_temp<[5]>;
                libfunc [16] = jump;
                libfunc [4] = struct_construct<[4]>;
                libfunc [3] = enum_init<[5], 1>;
                libfunc [2] = struct_construct<[6]>;
                libfunc [1] = enum_match<[5]>;
                libfunc [0] = unbox<[0]>;
                libfunc [17] = drop<[0]>;
                libfunc [22] = store_temp<[6]>;
                libfunc [18] = drop<[4]>;

                [8]() -> ([0]); // 0
                [10]() -> ([1]); // 1
                [19]([1]) -> ([1]); // 2
                [7]([0], [1]) -> ([2]); // 3
                [11]() -> ([3]); // 4
                [19]([3]) -> ([3]); // 5
                [7]([2], [3]) -> ([4]); // 6
                [12]() -> ([5]); // 7
                [19]([5]) -> ([5]); // 8
                [7]([4], [5]) -> ([6]); // 9
                [13]([6]) -> ([7], [8]); // 10
                [14]([7]) -> (); // 11
                [20]([8]) -> ([8]); // 12
                [6]([8]) { fallthrough([9], [10]) 19([11]) }; // 13
                [15]() -> (); // 14
                [5]([10]) -> ([12]); // 15
                [20]([9]) -> ([13]); // 16
                [21]([12]) -> ([14]); // 17
                [16]() { 24() }; // 18
                [15]() -> (); // 19
                [4]() -> ([15]); // 20
                [3]([15]) -> ([16]); // 21
                [20]([11]) -> ([13]); // 22
                [21]([16]) -> ([14]); // 23
                [2]([13]) -> ([17]); // 24
                [1]([14]) { fallthrough([18]) 31([19]) }; // 25
                [15]() -> (); // 26
                [0]([18]) -> ([20]); // 27
                [17]([20]) -> (); // 28
                [22]([17]) -> ([17]); // 29
                return([17]); // 30
                [15]() -> (); // 31
                [18]([19]) -> (); // 32
                [22]([17]) -> ([17]); // 33
                return([17]); // 34

                [0]@0() -> ([6]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let result = run_sierra_program(&program, &[]).return_value;

        assert_eq!(result, jit_struct!([1u32, 2u32].into()));
    }

    #[test]
    fn array_empty_span() {
        // Tests snapshot_take on a empty array.
        // fn run_test() -> Span<u32> {
        //     let x = ArrayTrait::new();
        //     x.span()
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [1] = Array<[0]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [2] = Snapshot<[1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [3] = Struct<ut@core::array::Span::<core::integer::u32>, [2]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [0] = u32 [storable: true, drop: true, dup: true, zero_sized: false];

                libfunc [1] = array_new<[0]>;
                libfunc [3] = snapshot_take<[1]>;
                libfunc [4] = drop<[1]>;
                libfunc [0] = struct_construct<[3]>;
                libfunc [5] = store_temp<[3]>;

                [1]() -> ([0]); // 0
                [3]([0]) -> ([1], [2]); // 1
                [4]([1]) -> (); // 2
                [0]([2]) -> ([3]); // 3
                [5]([3]) -> ([3]); // 4
                return([3]); // 5

                [0]@0() -> ([3]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        assert_eq!(
            run_sierra_program(&program, &[]).return_value,
            jit_struct!(Value::Array(vec![])),
        );
    }

    #[test]
    fn array_span_modify_span() {
        // Tests pop_back on a span.
        // use core::array::SpanTrait;
        // fn pop_elem(mut self: Span<u64>) -> Option<@u64> {
        //     let x = self.pop_back();
        //     x
        // }
        // fn run_test() -> Option<@u64> {
        //     let mut data = array![2].span();
        //     let x = pop_elem(data);
        //     x
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [1] = Array<[0]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [2] = Snapshot<[1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [7] = Struct<ut@core::array::Span::<core::integer::u64>, [2]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [4] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
                type [0] = u64 [storable: true, drop: true, dup: true, zero_sized: false];
                type [5] = Enum<ut@core::option::Option::<@core::integer::u64>, [0], [4]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [3] = Box<[0]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [6] = Const<[0], 2> [storable: false, drop: false, dup: false, zero_sized: false];

                libfunc [6] = array_new<[0]>;
                libfunc [8] = const_as_immediate<[6]>;
                libfunc [13] = store_temp<[0]>;
                libfunc [5] = array_append<[0]>;
                libfunc [9] = snapshot_take<[1]>;
                libfunc [10] = drop<[1]>;
                libfunc [14] = store_temp<[2]>;
                libfunc [4] = array_snapshot_pop_back<[0]>;
                libfunc [11] = branch_align;
                libfunc [12] = drop<[2]>;
                libfunc [15] = store_temp<[3]>;
                libfunc [3] = unbox<[0]>;
                libfunc [2] = enum_init<[5], 0>;
                libfunc [16] = store_temp<[5]>;
                libfunc [1] = struct_construct<[4]>;
                libfunc [0] = enum_init<[5], 1>;
                libfunc [17] = struct_deconstruct<[7]>;

                [6]() -> ([0]); // 0
                [8]() -> ([1]); // 1
                [13]([1]) -> ([1]); // 2
                [5]([0], [1]) -> ([2]); // 3
                [9]([2]) -> ([3], [4]); // 4
                [10]([3]) -> (); // 5
                [14]([4]) -> ([4]); // 6
                [4]([4]) { fallthrough([5], [6]) 15([7]) }; // 7
                [11]() -> (); // 8
                [12]([5]) -> (); // 9
                [15]([6]) -> ([6]); // 10
                [3]([6]) -> ([8]); // 11
                [2]([8]) -> ([9]); // 12
                [16]([9]) -> ([9]); // 13
                return([9]); // 14
                [11]() -> (); // 15
                [12]([7]) -> (); // 16
                [1]() -> ([10]); // 17
                [0]([10]) -> ([11]); // 18
                [16]([11]) -> ([11]); // 19
                return([11]); // 20
                [17]([0]) -> ([1]); // 21
                [4]([1]) { fallthrough([2], [3]) 30([4]) }; // 22
                [11]() -> (); // 23
                [12]([2]) -> (); // 24
                [15]([3]) -> ([3]); // 25
                [3]([3]) -> ([5]); // 26
                [2]([5]) -> ([6]); // 27
                [16]([6]) -> ([6]); // 28
                return([6]); // 29
                [11]() -> (); // 30
                [12]([4]) -> (); // 31
                [1]() -> ([7]); // 32
                [0]([7]) -> ([8]); // 33
                [16]([8]) -> ([8]); // 34
                return([8]); // 35

                [0]@0() -> ([5]);
                [1]@21([0]: [7]) -> ([5]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        assert_eq!(
            run_sierra_program(&program, &[]).return_value,
            jit_enum!(0, 2u64.into()),
        );
    }

    #[test]
    fn array_span_check_array() {
        // Tests pop back on a span not modifying the original array.
        // use core::array::SpanTrait;
        // fn pop_elem(mut self: Span<u64>) -> Option<@u64> {
        //     let x = self.pop_back();
        //     x
        // }
        // fn run_test() -> Array<u64> {
        //     let mut data = array![1, 2];
        //     let _x = pop_elem(data.span());
        //     data
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [1] = Array<[0]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [7] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
                type [0] = u64 [storable: true, drop: true, dup: true, zero_sized: false];
                type [8] = Enum<ut@core::option::Option::<@core::integer::u64>, [0], [7]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [2] = Snapshot<[1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [6] = Struct<ut@core::array::Span::<core::integer::u64>, [2]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [3] = Box<[0]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [5] = Const<[0], 2> [storable: false, drop: false, dup: false, zero_sized: false];
                type [4] = Const<[0], 1> [storable: false, drop: false, dup: false, zero_sized: false];

                libfunc [3] = array_new<[0]>;
                libfunc [5] = const_as_immediate<[4]>;
                libfunc [11] = store_temp<[0]>;
                libfunc [2] = array_append<[0]>;
                libfunc [6] = const_as_immediate<[5]>;
                libfunc [7] = snapshot_take<[1]>;
                libfunc [12] = store_temp<[2]>;
                libfunc [13] = store_temp<[1]>;
                libfunc [1] = array_snapshot_pop_back<[0]>;
                libfunc [8] = branch_align;
                libfunc [9] = drop<[2]>;
                libfunc [14] = store_temp<[3]>;
                libfunc [0] = unbox<[0]>;
                libfunc [10] = drop<[0]>;
                libfunc [18] = struct_deconstruct<[6]>;
                libfunc [17] = enum_init<[8], 0>;
                libfunc [19] = store_temp<[8]>;
                libfunc [16] = struct_construct<[7]>;
                libfunc [15] = enum_init<[8], 1>;

                [3]() -> ([0]); // 0
                [5]() -> ([1]); // 1
                [11]([1]) -> ([1]); // 2
                [2]([0], [1]) -> ([2]); // 3
                [6]() -> ([3]); // 4
                [11]([3]) -> ([3]); // 5
                [2]([2], [3]) -> ([4]); // 6
                [7]([4]) -> ([5], [6]); // 7
                [12]([6]) -> ([6]); // 8
                [13]([5]) -> ([5]); // 9
                [1]([6]) { fallthrough([7], [8]) 18([9]) }; // 10
                [8]() -> (); // 11
                [9]([7]) -> (); // 12
                [14]([8]) -> ([8]); // 13
                [0]([8]) -> ([10]); // 14
                [10]([10]) -> (); // 15
                [13]([5]) -> ([5]); // 16
                return([5]); // 17
                [8]() -> (); // 18
                [9]([9]) -> (); // 19
                [13]([5]) -> ([5]); // 20
                return([5]); // 21
                [18]([0]) -> ([1]); // 22
                [1]([1]) { fallthrough([2], [3]) 31([4]) }; // 23
                [8]() -> (); // 24
                [9]([2]) -> (); // 25
                [14]([3]) -> ([3]); // 26
                [0]([3]) -> ([5]); // 27
                [17]([5]) -> ([6]); // 28
                [19]([6]) -> ([6]); // 29
                return([6]); // 30
                [8]() -> (); // 31
                [9]([4]) -> (); // 32
                [16]() -> ([7]); // 33
                [15]([7]) -> ([8]); // 34
                [19]([8]) -> ([8]); // 35
                return([8]); // 36

                [0]@0() -> ([1]);
                [1]@22([0]: [6]) -> ([8]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        assert_eq!(
            run_sierra_program(&program, &[]).return_value,
            Value::Array(vec![1u64.into(), 2u64.into()]),
        );
    }

    #[test]
    fn tuple_from_span() {
        // use core::array::{tuple_from_span, FixedSizedArrayInfoImpl};
        // fn run_test(x: Array<felt252>) -> [felt252; 3] {
        //     (*tuple_from_span::<[felt252; 3], FixedSizedArrayInfoImpl<felt252, 3>>(@x).unwrap()).unbox()
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [1] = Array<[0]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [6] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
                type [7] = Struct<ut@Tuple, [6], [1]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [9] = Const<[0], 29721761890975875353235833581453094220424382983267374> [storable: false, drop: false, dup: false, zero_sized: false];
                type [0] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
                type [3] = Struct<ut@Tuple, [0], [0], [0]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [5] = Struct<ut@Tuple, [3]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [8] = Enum<ut@core::panics::PanicResult::<([core::felt252; 3],)>, [5], [7]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [4] = Box<[3]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [2] = Snapshot<[1]> [storable: true, drop: true, dup: true, zero_sized: false];

                libfunc [9] = snapshot_take<[1]>;
                libfunc [10] = drop<[1]>;
                libfunc [8] = tuple_from_span<[3]>;
                libfunc [11] = branch_align;
                libfunc [12] = rename<[4]>;
                libfunc [7] = unbox<[3]>;
                libfunc [6] = struct_construct<[5]>;
                libfunc [5] = enum_init<[8], 0>;
                libfunc [14] = store_temp<[8]>;
                libfunc [4] = array_new<[0]>;
                libfunc [13] = const_as_immediate<[9]>;
                libfunc [15] = store_temp<[0]>;
                libfunc [3] = array_append<[0]>;
                libfunc [2] = struct_construct<[6]>;
                libfunc [1] = struct_construct<[7]>;
                libfunc [0] = enum_init<[8], 1>;

                [9]([0]) -> ([1], [2]); // 0
                [10]([1]) -> (); // 1
                [8]([2]) { fallthrough([3]) 10() }; // 2
                [11]() -> (); // 3
                [12]([3]) -> ([4]); // 4
                [7]([4]) -> ([5]); // 5
                [6]([5]) -> ([6]); // 6
                [5]([6]) -> ([7]); // 7
                [14]([7]) -> ([7]); // 8
                return([7]); // 9
                [11]() -> (); // 10
                [4]() -> ([8]); // 11
                [13]() -> ([9]); // 12
                [15]([9]) -> ([9]); // 13
                [3]([8], [9]) -> ([10]); // 14
                [2]() -> ([11]); // 15
                [1]([11], [10]) -> ([12]); // 16
                [0]([12]) -> ([13]); // 17
                [14]([13]) -> ([13]); // 18
                return([13]); // 19

                [0]@0([0]: [1]) -> ([8]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        assert_eq!(
            run_sierra_program(
                &program,
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
        // use core::array::{tuple_from_span, FixedSizedArrayInfoImpl};
        // fn run_test(x: Array<felt252>) -> Option<@Box<[core::felt252; 3]>> {
        //     tuple_from_span::<[felt252; 3], FixedSizedArrayInfoImpl<felt252, 3>>(@x)
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [1] = Array<[0]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [5] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
                type [0] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
                type [4] = Box<[3]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [6] = Enum<ut@core::option::Option::<@core::box::Box::<[core::felt252; 3]>>, [4], [5]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [3] = Struct<ut@Tuple, [0], [0], [0]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [2] = Snapshot<[1]> [storable: true, drop: true, dup: true, zero_sized: false];

                libfunc [4] = snapshot_take<[1]>;
                libfunc [5] = drop<[1]>;
                libfunc [3] = tuple_from_span<[3]>;
                libfunc [6] = branch_align;
                libfunc [2] = enum_init<[6], 0>;
                libfunc [7] = store_temp<[6]>;
                libfunc [1] = struct_construct<[5]>;
                libfunc [0] = enum_init<[6], 1>;

                [4]([0]) -> ([1], [2]); // 0
                [5]([1]) -> (); // 1
                [3]([2]) { fallthrough([3]) 7() }; // 2
                [6]() -> (); // 3
                [2]([3]) -> ([4]); // 4
                [7]([4]) -> ([4]); // 5
                return([4]); // 6
                [6]() -> (); // 7
                [1]() -> ([5]); // 8
                [0]([5]) -> ([6]); // 9
                [7]([6]) -> ([6]); // 10
                return([6]); // 11

                [0]@0([0]: [1]) -> ([6]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        assert_eq!(
            run_sierra_program(
                &program,
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
        // use array::ArrayTrait;
        // fn run_test() -> (Span<felt252>, @Box<[felt252; 3]>) {
        //     let mut numbers = array![1, 2, 3, 4, 5, 6].span();
        //     let popped = numbers.multi_pop_front::<3>().unwrap();
        //     (numbers, popped)
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [2] = Array<[1]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [9] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
                type [10] = Struct<ut@Tuple, [9], [2]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [18] = Const<[1], 29721761890975875353235833581453094220424382983267374> [storable: false, drop: false, dup: false, zero_sized: false];
                type [3] = Snapshot<[2]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [6] = Struct<ut@core::array::Span::<core::felt252>, [3]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [5] = Box<[4]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [7] = Struct<ut@Tuple, [6], [5]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [8] = Struct<ut@Tuple, [7]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [11] = Enum<ut@core::panics::PanicResult::<((core::array::Span::<core::felt252>, @core::box::Box::<[core::felt252; 3]>),)>, [8], [10]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [1] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
                type [4] = Struct<ut@Tuple, [1], [1], [1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
                type [17] = Const<[1], 6> [storable: false, drop: false, dup: false, zero_sized: false];
                type [16] = Const<[1], 5> [storable: false, drop: false, dup: false, zero_sized: false];
                type [15] = Const<[1], 4> [storable: false, drop: false, dup: false, zero_sized: false];
                type [14] = Const<[1], 3> [storable: false, drop: false, dup: false, zero_sized: false];
                type [13] = Const<[1], 2> [storable: false, drop: false, dup: false, zero_sized: false];
                type [12] = Const<[1], 1> [storable: false, drop: false, dup: false, zero_sized: false];

                libfunc [4] = array_new<[1]>;
                libfunc [11] = const_as_immediate<[12]>;
                libfunc [22] = store_temp<[1]>;
                libfunc [3] = array_append<[1]>;
                libfunc [12] = const_as_immediate<[13]>;
                libfunc [13] = const_as_immediate<[14]>;
                libfunc [14] = const_as_immediate<[15]>;
                libfunc [15] = const_as_immediate<[16]>;
                libfunc [16] = const_as_immediate<[17]>;
                libfunc [17] = snapshot_take<[2]>;
                libfunc [18] = drop<[2]>;
                libfunc [23] = store_temp<[3]>;
                libfunc [9] = array_snapshot_multi_pop_front<[4]>;
                libfunc [19] = branch_align;
                libfunc [8] = struct_construct<[6]>;
                libfunc [7] = struct_construct<[7]>;
                libfunc [6] = struct_construct<[8]>;
                libfunc [5] = enum_init<[11], 0>;
                libfunc [24] = store_temp<[0]>;
                libfunc [25] = store_temp<[11]>;
                libfunc [20] = drop<[3]>;
                libfunc [21] = const_as_immediate<[18]>;
                libfunc [2] = struct_construct<[9]>;
                libfunc [1] = struct_construct<[10]>;
                libfunc [0] = enum_init<[11], 1>;

                [4]() -> ([1]); // 0
                [11]() -> ([2]); // 1
                [22]([2]) -> ([2]); // 2
                [3]([1], [2]) -> ([3]); // 3
                [12]() -> ([4]); // 4
                [22]([4]) -> ([4]); // 5
                [3]([3], [4]) -> ([5]); // 6
                [13]() -> ([6]); // 7
                [22]([6]) -> ([6]); // 8
                [3]([5], [6]) -> ([7]); // 9
                [14]() -> ([8]); // 10
                [22]([8]) -> ([8]); // 11
                [3]([7], [8]) -> ([9]); // 12
                [15]() -> ([10]); // 13
                [22]([10]) -> ([10]); // 14
                [3]([9], [10]) -> ([11]); // 15
                [16]() -> ([12]); // 16
                [22]([12]) -> ([12]); // 17
                [3]([11], [12]) -> ([13]); // 18
                [17]([13]) -> ([14], [15]); // 19
                [18]([14]) -> (); // 20
                [23]([15]) -> ([15]); // 21
                [9]([0], [15]) { fallthrough([16], [17], [18]) 31([19], [20]) }; // 22
                [19]() -> (); // 23
                [8]([17]) -> ([21]); // 24
                [7]([21], [18]) -> ([22]); // 25
                [6]([22]) -> ([23]); // 26
                [5]([23]) -> ([24]); // 27
                [24]([16]) -> ([16]); // 28
                [25]([24]) -> ([24]); // 29
                return([16], [24]); // 30
                [19]() -> (); // 31
                [20]([20]) -> (); // 32
                [4]() -> ([25]); // 33
                [21]() -> ([26]); // 34
                [22]([26]) -> ([26]); // 35
                [3]([25], [26]) -> ([27]); // 36
                [2]() -> ([28]); // 37
                [1]([28], [27]) -> ([29]); // 38
                [0]([29]) -> ([30]); // 39
                [24]([19]) -> ([19]); // 40
                [25]([30]) -> ([30]); // 41
                return([19], [30]); // 42

                [0]@0([0]: [0]) -> ([0], [11]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let result = run_sierra_program(&program, &[]).return_value;

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
        // use array::ArrayTrait;
        // fn run_test() -> Span<felt252> {
        //     let mut numbers = array![1, 2].span();
        //     // should fail (return none)
        //     assert!(numbers.multi_pop_front::<3>().is_none());
        //     numbers
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [10] = Array<[9]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [2] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
                type [11] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
                type [12] = Struct<ut@core::byte_array::ByteArray, [10], [2], [11]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [80] = Uninitialized<[12]> [storable: false, drop: true, dup: false, zero_sized: false];
                type [64] = Const<[2], 573087285299505011920718992710461799> [storable: false, drop: false, dup: false, zero_sized: false];
                type [63] = Const<[27], [62]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [61] = Const<[27], [60]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [62] = Const<[26], 1329227995784915872903807060280344576> [storable: false, drop: false, dup: false, zero_sized: false];
                type [60] = Const<[26], 5192296858534827628530496329220096> [storable: false, drop: false, dup: false, zero_sized: false];
                type [59] = Const<[27], [58]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [57] = Const<[27], [56]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [58] = Const<[26], 20282409603651670423947251286016> [storable: false, drop: false, dup: false, zero_sized: false];
                type [56] = Const<[26], 79228162514264337593543950336> [storable: false, drop: false, dup: false, zero_sized: false];
                type [55] = Const<[27], [54]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [53] = Const<[27], [52]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [54] = Const<[26], 309485009821345068724781056> [storable: false, drop: false, dup: false, zero_sized: false];
                type [52] = Const<[26], 1208925819614629174706176> [storable: false, drop: false, dup: false, zero_sized: false];
                type [51] = Const<[27], [50]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [49] = Const<[27], [48]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [50] = Const<[26], 4722366482869645213696> [storable: false, drop: false, dup: false, zero_sized: false];
                type [48] = Const<[26], 18446744073709551616> [storable: false, drop: false, dup: false, zero_sized: false];
                type [47] = Const<[27], [46]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [45] = Const<[27], [44]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [46] = Const<[26], 72057594037927936> [storable: false, drop: false, dup: false, zero_sized: false];
                type [44] = Const<[26], 281474976710656> [storable: false, drop: false, dup: false, zero_sized: false];
                type [43] = Const<[27], [42]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [41] = Const<[27], [40]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [42] = Const<[26], 1099511627776> [storable: false, drop: false, dup: false, zero_sized: false];
                type [40] = Const<[26], 4294967296> [storable: false, drop: false, dup: false, zero_sized: false];
                type [39] = Const<[27], [38]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [37] = Const<[27], [36]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [38] = Const<[26], 16777216> [storable: false, drop: false, dup: false, zero_sized: false];
                type [36] = Const<[26], 65536> [storable: false, drop: false, dup: false, zero_sized: false];
                type [35] = Const<[27], [34]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [33] = Const<[27], [32]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [34] = Const<[26], 256> [storable: false, drop: false, dup: false, zero_sized: false];
                type [32] = Const<[26], 1> [storable: false, drop: false, dup: false, zero_sized: false];
                type [7] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
                type [31] = Enum<ut@index_enum_type<16>, [7], [7], [7], [7], [7], [7], [7], [7], [7], [7], [7], [7], [7], [7], [7], [7]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [30] = BoundedInt<0, 15> [storable: true, drop: true, dup: true, zero_sized: false];
                type [82] = Const<[2], 375233589013918064796019> [storable: false, drop: false, dup: false, zero_sized: false];
                type [81] = Box<[9]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [73] = Const<[2], 155785504323917466144735657540098748279> [storable: false, drop: false, dup: false, zero_sized: false];
                type [70] = Const<[2], 29721761890975875353235833581453094220424382983267374> [storable: false, drop: false, dup: false, zero_sized: false];
                type [69] = Const<[2], 155785504329508738615720351733824384887> [storable: false, drop: false, dup: false, zero_sized: false];
                type [68] = Const<[2], 340282366920938463463374607431768211456> [storable: false, drop: false, dup: false, zero_sized: false];
                type [26] = u128 [storable: true, drop: true, dup: true, zero_sized: false];
                type [27] = NonZero<[26]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [28] = Struct<ut@Tuple, [27]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [14] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
                type [3] = Array<[2]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [15] = Struct<ut@Tuple, [14], [3]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [29] = Enum<ut@core::panics::PanicResult::<(core::zeroable::NonZero::<core::integer::u128>,)>, [28], [15]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [67] = Const<[26], 0> [storable: false, drop: false, dup: false, zero_sized: false];
                type [66] = Const<[11], 16> [storable: false, drop: false, dup: false, zero_sized: false];
                type [25] = NonZero<[11]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [4] = Snapshot<[3]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [22] = Struct<ut@core::array::Span::<core::felt252>, [4]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [23] = Struct<ut@Tuple, [22]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [24] = Enum<ut@core::panics::PanicResult::<(core::array::Span::<core::felt252>,)>, [23], [15]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [20] = Struct<ut@Tuple, [3], [7]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [21] = Enum<ut@core::panics::PanicResult::<(core::array::Array::<core::felt252>, ())>, [20], [15]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [18] = Snapshot<[10]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [19] = Struct<ut@core::array::Span::<core::bytes_31::bytes31>, [18]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [17] = Snapshot<[12]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [79] = Const<[2], 1997209042069643135709344952807065910992472029923670688473712229447419591075> [storable: false, drop: false, dup: false, zero_sized: false];
                type [78] = Const<[11], 30> [storable: false, drop: false, dup: false, zero_sized: false];
                type [77] = Const<[2], 727256402166382750144834095173136006803505384629802774195555044099186734> [storable: false, drop: false, dup: false, zero_sized: false];
                type [13] = Struct<ut@Tuple, [12], [7]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [16] = Enum<ut@core::panics::PanicResult::<(core::byte_array::ByteArray, ())>, [13], [15]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [65] = Const<[11], 31> [storable: false, drop: false, dup: false, zero_sized: false];
                type [76] = Const<[2], 172180977190876322177717838039515195832848434335613823290676811071835434100> [storable: false, drop: false, dup: false, zero_sized: false];
                type [72] = Const<[11], 0> [storable: false, drop: false, dup: false, zero_sized: false];
                type [71] = Const<[2], 0> [storable: false, drop: false, dup: false, zero_sized: false];
                type [9] = bytes31 [storable: true, drop: true, dup: true, zero_sized: false];
                type [6] = Box<[5]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [8] = Enum<ut@core::option::Option::<@core::box::Box::<[core::felt252; 3]>>, [6], [7]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [1] = GasBuiltin [storable: true, drop: false, dup: false, zero_sized: false];
                type [5] = Struct<ut@Tuple, [2], [2], [2]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
                type [75] = Const<[2], 2> [storable: false, drop: false, dup: false, zero_sized: false];
                type [74] = Const<[2], 1> [storable: false, drop: false, dup: false, zero_sized: false];

                libfunc [102] = alloc_local<[12]>;
                libfunc [103] = finalize_locals;
                libfunc [104] = disable_ap_tracking;
                libfunc [17] = array_new<[2]>;
                libfunc [105] = const_as_immediate<[74]>;
                libfunc [63] = store_temp<[2]>;
                libfunc [8] = array_append<[2]>;
                libfunc [106] = const_as_immediate<[75]>;
                libfunc [107] = snapshot_take<[3]>;
                libfunc [108] = drop<[3]>;
                libfunc [127] = store_temp<[4]>;
                libfunc [100] = array_snapshot_multi_pop_front<[5]>;
                libfunc [38] = branch_align;
                libfunc [3] = redeposit_gas;
                libfunc [99] = enum_init<[8], 0>;
                libfunc [60] = store_temp<[0]>;
                libfunc [128] = store_temp<[1]>;
                libfunc [129] = store_temp<[8]>;
                libfunc [42] = jump;
                libfunc [23] = struct_construct<[7]>;
                libfunc [98] = enum_init<[8], 1>;
                libfunc [109] = snapshot_take<[8]>;
                libfunc [110] = drop<[8]>;
                libfunc [97] = enum_match<[8]>;
                libfunc [111] = drop<[6]>;
                libfunc [112] = drop<[4]>;
                libfunc [96] = array_new<[9]>;
                libfunc [85] = const_as_immediate<[71]>;
                libfunc [86] = const_as_immediate<[72]>;
                libfunc [113] = const_as_immediate<[76]>;
                libfunc [76] = const_as_immediate<[65]>;
                libfunc [24] = struct_construct<[12]>;
                libfunc [92] = store_temp<[12]>;
                libfunc [89] = store_temp<[11]>;
                libfunc [19] = function_call<user@[1]>;
                libfunc [18] = enum_match<[16]>;
                libfunc [16] = struct_deconstruct<[13]>;
                libfunc [40] = drop<[7]>;
                libfunc [114] = const_as_immediate<[77]>;
                libfunc [115] = const_as_immediate<[78]>;
                libfunc [116] = const_as_immediate<[79]>;
                libfunc [130] = store_local<[12]>;
                libfunc [117] = snapshot_take<[12]>;
                libfunc [118] = drop<[12]>;
                libfunc [119] = dup<[17]>;
                libfunc [10] = struct_snapshot_deconstruct<[12]>;
                libfunc [73] = drop<[2]>;
                libfunc [74] = drop<[11]>;
                libfunc [120] = dup<[18]>;
                libfunc [15] = array_len<[9]>;
                libfunc [9] = u32_to_felt252;
                libfunc [14] = struct_construct<[19]>;
                libfunc [131] = store_temp<[19]>;
                libfunc [91] = store_temp<[3]>;
                libfunc [13] = function_call<user@[0]>;
                libfunc [121] = enable_ap_tracking;
                libfunc [12] = enum_match<[21]>;
                libfunc [11] = struct_deconstruct<[20]>;
                libfunc [122] = drop<[18]>;
                libfunc [123] = rename<[2]>;
                libfunc [124] = rename<[11]>;
                libfunc [125] = drop<[17]>;
                libfunc [7] = struct_deconstruct<[15]>;
                libfunc [81] = drop<[14]>;
                libfunc [6] = struct_construct<[14]>;
                libfunc [5] = struct_construct<[15]>;
                libfunc [4] = enum_init<[24], 1>;
                libfunc [132] = store_temp<[24]>;
                libfunc [126] = drop<[80]>;
                libfunc [2] = struct_construct<[22]>;
                libfunc [1] = struct_construct<[23]>;
                libfunc [0] = enum_init<[24], 0>;
                libfunc [72] = dup<[11]>;
                libfunc [65] = u32_is_zero;
                libfunc [22] = struct_construct<[13]>;
                libfunc [21] = enum_init<[16], 0>;
                libfunc [88] = store_temp<[16]>;
                libfunc [75] = drop<[25]>;
                libfunc [68] = struct_deconstruct<[12]>;
                libfunc [27] = u32_overflowing_add;
                libfunc [64] = u32_overflowing_sub;
                libfunc [71] = u32_eq;
                libfunc [77] = const_as_immediate<[66]>;
                libfunc [69] = u128s_from_felt252;
                libfunc [78] = const_as_immediate<[67]>;
                libfunc [90] = store_temp<[26]>;
                libfunc [32] = function_call<user@[2]>;
                libfunc [31] = enum_match<[29]>;
                libfunc [30] = struct_deconstruct<[28]>;
                libfunc [70] = u128_safe_divmod;
                libfunc [28] = u128_to_felt252;
                libfunc [79] = const_as_immediate<[68]>;
                libfunc [26] = felt252_mul;
                libfunc [25] = felt252_add;
                libfunc [29] = unwrap_non_zero<[26]>;
                libfunc [80] = drop<[10]>;
                libfunc [82] = const_as_immediate<[69]>;
                libfunc [67] = bytes31_try_from_felt252;
                libfunc [66] = array_append<[9]>;
                libfunc [83] = const_as_immediate<[70]>;
                libfunc [93] = rename<[0]>;
                libfunc [94] = rename<[3]>;
                libfunc [84] = drop<[26]>;
                libfunc [20] = enum_init<[16], 1>;
                libfunc [95] = rename<[12]>;
                libfunc [87] = const_as_immediate<[73]>;
                libfunc [140] = withdraw_gas;
                libfunc [139] = struct_deconstruct<[19]>;
                libfunc [138] = array_snapshot_pop_front<[9]>;
                libfunc [137] = unbox<[9]>;
                libfunc [141] = rename<[9]>;
                libfunc [136] = bytes31_to_felt252;
                libfunc [135] = struct_construct<[20]>;
                libfunc [134] = enum_init<[21], 0>;
                libfunc [144] = store_temp<[21]>;
                libfunc [142] = drop<[19]>;
                libfunc [143] = const_as_immediate<[82]>;
                libfunc [133] = enum_init<[21], 1>;
                libfunc [37] = downcast<[11], [30]>;
                libfunc [39] = enum_from_bounded_int<[31]>;
                libfunc [59] = store_temp<[31]>;
                libfunc [36] = enum_match<[31]>;
                libfunc [41] = const_as_immediate<[33]>;
                libfunc [61] = store_temp<[27]>;
                libfunc [43] = const_as_immediate<[35]>;
                libfunc [44] = const_as_immediate<[37]>;
                libfunc [45] = const_as_immediate<[39]>;
                libfunc [46] = const_as_immediate<[41]>;
                libfunc [47] = const_as_immediate<[43]>;
                libfunc [48] = const_as_immediate<[45]>;
                libfunc [49] = const_as_immediate<[47]>;
                libfunc [50] = const_as_immediate<[49]>;
                libfunc [51] = const_as_immediate<[51]>;
                libfunc [52] = const_as_immediate<[53]>;
                libfunc [53] = const_as_immediate<[55]>;
                libfunc [54] = const_as_immediate<[57]>;
                libfunc [55] = const_as_immediate<[59]>;
                libfunc [56] = const_as_immediate<[61]>;
                libfunc [57] = const_as_immediate<[63]>;
                libfunc [35] = struct_construct<[28]>;
                libfunc [34] = enum_init<[29], 0>;
                libfunc [62] = store_temp<[29]>;
                libfunc [58] = const_as_immediate<[64]>;
                libfunc [33] = enum_init<[29], 1>;

                [102]() -> ([3]); // 0
                [103]() -> (); // 1
                [104]() -> (); // 2
                [17]() -> ([4]); // 3
                [105]() -> ([5]); // 4
                [63]([5]) -> ([5]); // 5
                [8]([4], [5]) -> ([6]); // 6
                [106]() -> ([7]); // 7
                [63]([7]) -> ([7]); // 8
                [8]([6], [7]) -> ([8]); // 9
                [107]([8]) -> ([9], [10]); // 10
                [108]([9]) -> (); // 11
                [127]([10]) -> ([10]); // 12
                [100]([0], [10]) { fallthrough([11], [12], [13]) 22([14], [15]) }; // 13
                [38]() -> (); // 14
                [3]([1]) -> ([16]); // 15
                [99]([13]) -> ([17]); // 16
                [60]([11]) -> ([18]); // 17
                [128]([16]) -> ([19]); // 18
                [127]([12]) -> ([20]); // 19
                [129]([17]) -> ([21]); // 20
                [42]() { 30() }; // 21
                [38]() -> (); // 22
                [3]([1]) -> ([22]); // 23
                [23]() -> ([23]); // 24
                [98]([23]) -> ([24]); // 25
                [60]([14]) -> ([18]); // 26
                [128]([22]) -> ([19]); // 27
                [127]([15]) -> ([20]); // 28
                [129]([24]) -> ([21]); // 29
                [109]([21]) -> ([25], [26]); // 30
                [110]([25]) -> (); // 31
                [97]([26]) { fallthrough([27]) 141([28]) }; // 32
                [38]() -> (); // 33
                [111]([27]) -> (); // 34
                [112]([20]) -> (); // 35
                [3]([19]) -> ([29]); // 36
                [96]() -> ([30]); // 37
                [85]() -> ([31]); // 38
                [86]() -> ([32]); // 39
                [113]() -> ([33]); // 40
                [76]() -> ([34]); // 41
                [24]([30], [31], [32]) -> ([35]); // 42
                [60]([18]) -> ([18]); // 43
                [92]([35]) -> ([35]); // 44
                [63]([33]) -> ([33]); // 45
                [89]([34]) -> ([34]); // 46
                [19]([18], [35], [33], [34]) -> ([36], [37]); // 47
                [128]([29]) -> ([29]); // 48
                [18]([37]) { fallthrough([38]) 133([39]) }; // 49
                [38]() -> (); // 50
                [3]([29]) -> ([40]); // 51
                [16]([38]) -> ([41], [42]); // 52
                [40]([42]) -> (); // 53
                [114]() -> ([43]); // 54
                [115]() -> ([44]); // 55
                [60]([36]) -> ([36]); // 56
                [92]([41]) -> ([41]); // 57
                [63]([43]) -> ([43]); // 58
                [89]([44]) -> ([44]); // 59
                [19]([36], [41], [43], [44]) -> ([45], [46]); // 60
                [128]([40]) -> ([40]); // 61
                [18]([46]) { fallthrough([47]) 125([48]) }; // 62
                [38]() -> (); // 63
                [3]([40]) -> ([49]); // 64
                [17]() -> ([50]); // 65
                [116]() -> ([51]); // 66
                [63]([51]) -> ([51]); // 67
                [8]([50], [51]) -> ([52]); // 68
                [16]([47]) -> ([2], [53]); // 69
                [40]([53]) -> (); // 70
                [130]([3], [2]) -> ([2]); // 71
                [117]([2]) -> ([54], [55]); // 72
                [118]([54]) -> (); // 73
                [119]([55]) -> ([55], [56]); // 74
                [10]([56]) -> ([57], [58], [59]); // 75
                [73]([58]) -> (); // 76
                [74]([59]) -> (); // 77
                [120]([57]) -> ([57], [60]); // 78
                [15]([60]) -> ([61]); // 79
                [9]([61]) -> ([62]); // 80
                [63]([62]) -> ([62]); // 81
                [8]([52], [62]) -> ([63]); // 82
                [14]([57]) -> ([64]); // 83
                [60]([45]) -> ([45]); // 84
                [128]([49]) -> ([49]); // 85
                [131]([64]) -> ([64]); // 86
                [91]([63]) -> ([63]); // 87
                [13]([45], [49], [64], [63]) -> ([65], [66], [67]); // 88
                [121]() -> (); // 89
                [12]([67]) { fallthrough([68]) 110([69]) }; // 90
                [38]() -> (); // 91
                [3]([66]) -> ([70]); // 92
                [11]([68]) -> ([71], [72]); // 93
                [40]([72]) -> (); // 94
                [119]([55]) -> ([55], [73]); // 95
                [10]([73]) -> ([74], [75], [76]); // 96
                [122]([74]) -> (); // 97
                [74]([76]) -> (); // 98
                [123]([75]) -> ([77]); // 99
                [8]([71], [77]) -> ([78]); // 100
                [10]([55]) -> ([79], [80], [81]); // 101
                [122]([79]) -> (); // 102
                [73]([80]) -> (); // 103
                [124]([81]) -> ([82]); // 104
                [9]([82]) -> ([83]); // 105
                [8]([78], [83]) -> ([84]); // 106
                [128]([70]) -> ([85]); // 107
                [91]([84]) -> ([86]); // 108
                [42]() { 117() }; // 109
                [38]() -> (); // 110
                [125]([55]) -> (); // 111
                [3]([66]) -> ([87]); // 112
                [7]([69]) -> ([88], [89]); // 113
                [81]([88]) -> (); // 114
                [128]([87]) -> ([85]); // 115
                [91]([89]) -> ([86]); // 116
                [104]() -> (); // 117
                [6]() -> ([90]); // 118
                [5]([90], [86]) -> ([91]); // 119
                [4]([91]) -> ([92]); // 120
                [60]([65]) -> ([65]); // 121
                [128]([85]) -> ([85]); // 122
                [132]([92]) -> ([92]); // 123
                return([65], [85], [92]); // 124
                [38]() -> (); // 125
                [126]([3]) -> (); // 126
                [3]([40]) -> ([93]); // 127
                [4]([48]) -> ([94]); // 128
                [60]([45]) -> ([45]); // 129
                [128]([93]) -> ([93]); // 130
                [132]([94]) -> ([94]); // 131
                return([45], [93], [94]); // 132
                [38]() -> (); // 133
                [126]([3]) -> (); // 134
                [3]([29]) -> ([95]); // 135
                [4]([39]) -> ([96]); // 136
                [60]([36]) -> ([36]); // 137
                [128]([95]) -> ([95]); // 138
                [132]([96]) -> ([96]); // 139
                return([36], [95], [96]); // 140
                [38]() -> (); // 141
                [40]([28]) -> (); // 142
                [126]([3]) -> (); // 143
                [3]([19]) -> ([97]); // 144
                [2]([20]) -> ([98]); // 145
                [1]([98]) -> ([99]); // 146
                [0]([99]) -> ([100]); // 147
                [60]([18]) -> ([18]); // 148
                [128]([97]) -> ([97]); // 149
                [132]([100]) -> ([100]); // 150
                return([18], [97], [100]); // 151
                [72]([3]) -> ([3], [4]); // 152
                [65]([4]) { fallthrough() 163([5]) }; // 153
                [38]() -> (); // 154
                [73]([2]) -> (); // 155
                [74]([3]) -> (); // 156
                [23]() -> ([6]); // 157
                [22]([1], [6]) -> ([7]); // 158
                [21]([7]) -> ([8]); // 159
                [60]([0]) -> ([0]); // 160
                [88]([8]) -> ([8]); // 161
                return([0], [8]); // 162
                [38]() -> (); // 163
                [75]([5]) -> (); // 164
                [68]([1]) -> ([9], [10], [11]); // 165
                [72]([11]) -> ([11], [12]); // 166
                [72]([3]) -> ([3], [13]); // 167
                [27]([0], [12], [13]) { fallthrough([14], [15]) 977([16], [17]) }; // 168
                [38]() -> (); // 169
                [76]() -> ([18]); // 170
                [72]([15]) -> ([15], [19]); // 171
                [89]([18]) -> ([18]); // 172
                [64]([14], [19], [18]) { fallthrough([20], [21]) 853([22], [23]) }; // 173
                [38]() -> (); // 174
                [74]([21]) -> (); // 175
                [76]() -> ([24]); // 176
                [72]([15]) -> ([15], [25]); // 177
                [60]([20]) -> ([20]); // 178
                [71]([25], [24]) { fallthrough() 752() }; // 179
                [38]() -> (); // 180
                [74]([3]) -> (); // 181
                [76]() -> ([26]); // 182
                [89]([26]) -> ([26]); // 183
                [64]([20], [15], [26]) { fallthrough([27], [28]) 736([29], [30]) }; // 184
                [38]() -> (); // 185
                [77]() -> ([31]); // 186
                [72]([28]) -> ([28], [32]); // 187
                [60]([27]) -> ([27]); // 188
                [71]([32], [31]) { fallthrough() 590() }; // 189
                [38]() -> (); // 190
                [77]() -> ([33]); // 191
                [72]([28]) -> ([28], [34]); // 192
                [89]([33]) -> ([33]); // 193
                [64]([27], [34], [33]) { fallthrough([35], [36]) 380([37], [38]) }; // 194
                [38]() -> (); // 195
                [74]([36]) -> (); // 196
                [69]([35], [2]) { fallthrough([39], [40]) 204([41], [42], [43]) }; // 197
                [38]() -> (); // 198
                [78]() -> ([44]); // 199
                [60]([39]) -> ([45]); // 200
                [90]([40]) -> ([46]); // 201
                [90]([44]) -> ([47]); // 202
                [42]() { 208() }; // 203
                [38]() -> (); // 204
                [60]([41]) -> ([45]); // 205
                [90]([43]) -> ([46]); // 206
                [90]([42]) -> ([47]); // 207
                [77]() -> ([48]); // 208
                [72]([28]) -> ([28], [49]); // 209
                [89]([48]) -> ([48]); // 210
                [64]([45], [49], [48]) { fallthrough([50], [51]) 360([52], [53]) }; // 211
                [38]() -> (); // 212
                [60]([50]) -> ([50]); // 213
                [89]([51]) -> ([51]); // 214
                [32]([50], [51]) -> ([54], [55]); // 215
                [31]([55]) { fallthrough([56]) 348([57]) }; // 216
                [38]() -> (); // 217
                [30]([56]) -> ([58]); // 218
                [70]([54], [47], [58]) -> ([59], [60], [61]); // 219
                [28]([61]) -> ([62]); // 220
                [28]([46]) -> ([63]); // 221
                [28]([60]) -> ([64]); // 222
                [76]() -> ([65]); // 223
                [72]([11]) -> ([11], [66]); // 224
                [89]([65]) -> ([65]); // 225
                [64]([59], [65], [66]) { fallthrough([67], [68]) 332([69], [70]) }; // 226
                [38]() -> (); // 227
                [79]() -> ([71]); // 228
                [26]([62], [71]) -> ([72]); // 229
                [63]([72]) -> ([72]); // 230
                [25]([72], [63]) -> ([73]); // 231
                [77]() -> ([74]); // 232
                [72]([68]) -> ([68], [75]); // 233
                [89]([74]) -> ([74]); // 234
                [63]([73]) -> ([73]); // 235
                [64]([67], [75], [74]) { fallthrough([76], [77]) 283([78], [79]) }; // 236
                [38]() -> (); // 237
                [74]([77]) -> (); // 238
                [77]() -> ([80]); // 239
                [89]([80]) -> ([80]); // 240
                [64]([76], [68], [80]) { fallthrough([81], [82]) 268([83], [84]) }; // 241
                [38]() -> (); // 242
                [60]([81]) -> ([81]); // 243
                [89]([82]) -> ([82]); // 244
                [32]([81], [82]) -> ([85], [86]); // 245
                [31]([86]) { fallthrough([87]) 256([88]) }; // 246
                [38]() -> (); // 247
                [30]([87]) -> ([89]); // 248
                [29]([89]) -> ([90]); // 249
                [28]([90]) -> ([91]); // 250
                [79]() -> ([92]); // 251
                [26]([91], [92]) -> ([93]); // 252
                [60]([85]) -> ([94]); // 253
                [63]([93]) -> ([95]); // 254
                [42]() { 295() }; // 255
                [38]() -> (); // 256
                [80]([9]) -> (); // 257
                [74]([28]) -> (); // 258
                [74]([11]) -> (); // 259
                [73]([73]) -> (); // 260
                [73]([64]) -> (); // 261
                [73]([10]) -> (); // 262
                [7]([88]) -> ([96], [97]); // 263
                [81]([96]) -> (); // 264
                [60]([85]) -> ([98]); // 265
                [91]([97]) -> ([99]); // 266
                [42]() { 329() }; // 267
                [38]() -> (); // 268
                [74]([84]) -> (); // 269
                [80]([9]) -> (); // 270
                [74]([28]) -> (); // 271
                [74]([11]) -> (); // 272
                [73]([73]) -> (); // 273
                [73]([10]) -> (); // 274
                [73]([64]) -> (); // 275
                [17]() -> ([100]); // 276
                [82]() -> ([101]); // 277
                [63]([101]) -> ([101]); // 278
                [8]([100], [101]) -> ([102]); // 279
                [60]([83]) -> ([98]); // 280
                [91]([102]) -> ([99]); // 281
                [42]() { 329() }; // 282
                [38]() -> (); // 283
                [74]([79]) -> (); // 284
                [60]([78]) -> ([78]); // 285
                [89]([68]) -> ([68]); // 286
                [32]([78], [68]) -> ([103], [104]); // 287
                [31]([104]) { fallthrough([105]) 318([106]) }; // 288
                [38]() -> (); // 289
                [30]([105]) -> ([107]); // 290
                [29]([107]) -> ([108]); // 291
                [28]([108]) -> ([109]); // 292
                [60]([103]) -> ([94]); // 293
                [63]([109]) -> ([95]); // 294
                [26]([10], [95]) -> ([110]); // 295
                [63]([110]) -> ([110]); // 296
                [25]([64], [110]) -> ([111]); // 297
                [63]([111]) -> ([111]); // 298
                [67]([94], [111]) { fallthrough([112], [113]) 306([114]) }; // 299
                [38]() -> (); // 300
                [66]([9], [113]) -> ([115]); // 301
                [24]([115], [73], [11]) -> ([116]); // 302
                [60]([112]) -> ([117]); // 303
                [92]([116]) -> ([118]); // 304
                [42]() { 498() }; // 305
                [38]() -> (); // 306
                [80]([9]) -> (); // 307
                [74]([28]) -> (); // 308
                [74]([11]) -> (); // 309
                [73]([73]) -> (); // 310
                [17]() -> ([119]); // 311
                [83]() -> ([120]); // 312
                [63]([120]) -> ([120]); // 313
                [8]([119], [120]) -> ([121]); // 314
                [60]([114]) -> ([122]); // 315
                [91]([121]) -> ([123]); // 316
                [42]() { 374() }; // 317
                [38]() -> (); // 318
                [80]([9]) -> (); // 319
                [74]([28]) -> (); // 320
                [74]([11]) -> (); // 321
                [73]([73]) -> (); // 322
                [73]([64]) -> (); // 323
                [73]([10]) -> (); // 324
                [7]([106]) -> ([124], [125]); // 325
                [81]([124]) -> (); // 326
                [60]([103]) -> ([98]); // 327
                [91]([125]) -> ([99]); // 328
                [93]([98]) -> ([122]); // 329
                [94]([99]) -> ([123]); // 330
                [42]() { 374() }; // 331
                [38]() -> (); // 332
                [74]([70]) -> (); // 333
                [80]([9]) -> (); // 334
                [74]([28]) -> (); // 335
                [74]([11]) -> (); // 336
                [73]([62]) -> (); // 337
                [73]([10]) -> (); // 338
                [73]([64]) -> (); // 339
                [73]([63]) -> (); // 340
                [17]() -> ([126]); // 341
                [82]() -> ([127]); // 342
                [63]([127]) -> ([127]); // 343
                [8]([126], [127]) -> ([128]); // 344
                [60]([69]) -> ([122]); // 345
                [91]([128]) -> ([123]); // 346
                [42]() { 374() }; // 347
                [38]() -> (); // 348
                [80]([9]) -> (); // 349
                [74]([28]) -> (); // 350
                [74]([11]) -> (); // 351
                [84]([46]) -> (); // 352
                [73]([10]) -> (); // 353
                [84]([47]) -> (); // 354
                [7]([57]) -> ([129], [130]); // 355
                [81]([129]) -> (); // 356
                [60]([54]) -> ([122]); // 357
                [91]([130]) -> ([123]); // 358
                [42]() { 374() }; // 359
                [38]() -> (); // 360
                [74]([53]) -> (); // 361
                [80]([9]) -> (); // 362
                [74]([28]) -> (); // 363
                [74]([11]) -> (); // 364
                [84]([46]) -> (); // 365
                [73]([10]) -> (); // 366
                [84]([47]) -> (); // 367
                [17]() -> ([131]); // 368
                [82]() -> ([132]); // 369
                [63]([132]) -> ([132]); // 370
                [8]([131], [132]) -> ([133]); // 371
                [60]([52]) -> ([122]); // 372
                [91]([133]) -> ([123]); // 373
                [6]() -> ([134]); // 374
                [5]([134], [123]) -> ([135]); // 375
                [20]([135]) -> ([136]); // 376
                [60]([122]) -> ([122]); // 377
                [88]([136]) -> ([136]); // 378
                return([122], [136]); // 379
                [38]() -> (); // 380
                [74]([38]) -> (); // 381
                [69]([37], [2]) { fallthrough([137], [138]) 389([139], [140], [141]) }; // 382
                [38]() -> (); // 383
                [78]() -> ([142]); // 384
                [60]([137]) -> ([143]); // 385
                [90]([138]) -> ([144]); // 386
                [90]([142]) -> ([145]); // 387
                [42]() { 393() }; // 388
                [38]() -> (); // 389
                [60]([139]) -> ([143]); // 390
                [90]([141]) -> ([144]); // 391
                [90]([140]) -> ([145]); // 392
                [60]([143]) -> ([143]); // 393
                [72]([28]) -> ([28], [146]); // 394
                [89]([146]) -> ([146]); // 395
                [32]([143], [146]) -> ([147], [148]); // 396
                [31]([148]) { fallthrough([149]) 573([150]) }; // 397
                [38]() -> (); // 398
                [30]([149]) -> ([151]); // 399
                [70]([147], [144], [151]) -> ([152], [153], [154]); // 400
                [28]([145]) -> ([155]); // 401
                [77]() -> ([156]); // 402
                [72]([28]) -> ([28], [157]); // 403
                [89]([156]) -> ([156]); // 404
                [64]([152], [156], [157]) { fallthrough([158], [159]) 557([160], [161]) }; // 405
                [38]() -> (); // 406
                [60]([158]) -> ([158]); // 407
                [89]([159]) -> ([159]); // 408
                [32]([158], [159]) -> ([162], [163]); // 409
                [31]([163]) { fallthrough([164]) 544([165]) }; // 410
                [38]() -> (); // 411
                [30]([164]) -> ([166]); // 412
                [29]([166]) -> ([167]); // 413
                [28]([167]) -> ([168]); // 414
                [28]([153]) -> ([169]); // 415
                [28]([154]) -> ([170]); // 416
                [76]() -> ([171]); // 417
                [72]([11]) -> ([11], [172]); // 418
                [89]([171]) -> ([171]); // 419
                [64]([162], [171], [172]) { fallthrough([173], [174]) 527([175], [176]) }; // 420
                [38]() -> (); // 421
                [26]([155], [168]) -> ([177]); // 422
                [63]([177]) -> ([177]); // 423
                [25]([177], [169]) -> ([178]); // 424
                [77]() -> ([179]); // 425
                [72]([174]) -> ([174], [180]); // 426
                [89]([179]) -> ([179]); // 427
                [63]([178]) -> ([178]); // 428
                [64]([173], [180], [179]) { fallthrough([181], [182]) 476([183], [184]) }; // 429
                [38]() -> (); // 430
                [74]([182]) -> (); // 431
                [77]() -> ([185]); // 432
                [89]([185]) -> ([185]); // 433
                [64]([181], [174], [185]) { fallthrough([186], [187]) 461([188], [189]) }; // 434
                [38]() -> (); // 435
                [60]([186]) -> ([186]); // 436
                [89]([187]) -> ([187]); // 437
                [32]([186], [187]) -> ([190], [191]); // 438
                [31]([191]) { fallthrough([192]) 449([193]) }; // 439
                [38]() -> (); // 440
                [30]([192]) -> ([194]); // 441
                [29]([194]) -> ([195]); // 442
                [28]([195]) -> ([196]); // 443
                [79]() -> ([197]); // 444
                [26]([196], [197]) -> ([198]); // 445
                [60]([190]) -> ([199]); // 446
                [63]([198]) -> ([200]); // 447
                [42]() { 488() }; // 448
                [38]() -> (); // 449
                [80]([9]) -> (); // 450
                [74]([28]) -> (); // 451
                [74]([11]) -> (); // 452
                [73]([170]) -> (); // 453
                [73]([178]) -> (); // 454
                [73]([10]) -> (); // 455
                [7]([193]) -> ([201], [202]); // 456
                [81]([201]) -> (); // 457
                [60]([190]) -> ([203]); // 458
                [91]([202]) -> ([204]); // 459
                [42]() { 524() }; // 460
                [38]() -> (); // 461
                [74]([189]) -> (); // 462
                [80]([9]) -> (); // 463
                [74]([28]) -> (); // 464
                [74]([11]) -> (); // 465
                [73]([170]) -> (); // 466
                [73]([10]) -> (); // 467
                [73]([178]) -> (); // 468
                [17]() -> ([205]); // 469
                [82]() -> ([206]); // 470
                [63]([206]) -> ([206]); // 471
                [8]([205], [206]) -> ([207]); // 472
                [60]([188]) -> ([203]); // 473
                [91]([207]) -> ([204]); // 474
                [42]() { 524() }; // 475
                [38]() -> (); // 476
                [74]([184]) -> (); // 477
                [60]([183]) -> ([183]); // 478
                [89]([174]) -> ([174]); // 479
                [32]([183], [174]) -> ([208], [209]); // 480
                [31]([209]) { fallthrough([210]) 513([211]) }; // 481
                [38]() -> (); // 482
                [30]([210]) -> ([212]); // 483
                [29]([212]) -> ([213]); // 484
                [28]([213]) -> ([214]); // 485
                [60]([208]) -> ([199]); // 486
                [63]([214]) -> ([200]); // 487
                [26]([10], [200]) -> ([215]); // 488
                [63]([215]) -> ([215]); // 489
                [25]([178], [215]) -> ([216]); // 490
                [63]([216]) -> ([216]); // 491
                [67]([199], [216]) { fallthrough([217], [218]) 501([219]) }; // 492
                [38]() -> (); // 493
                [66]([9], [218]) -> ([220]); // 494
                [24]([220], [170], [11]) -> ([221]); // 495
                [60]([217]) -> ([117]); // 496
                [92]([221]) -> ([118]); // 497
                [93]([117]) -> ([222]); // 498
                [95]([118]) -> ([223]); // 499
                [42]() { 681() }; // 500
                [38]() -> (); // 501
                [80]([9]) -> (); // 502
                [74]([28]) -> (); // 503
                [74]([11]) -> (); // 504
                [73]([170]) -> (); // 505
                [17]() -> ([224]); // 506
                [83]() -> ([225]); // 507
                [63]([225]) -> ([225]); // 508
                [8]([224], [225]) -> ([226]); // 509
                [60]([219]) -> ([227]); // 510
                [91]([226]) -> ([228]); // 511
                [42]() { 584() }; // 512
                [38]() -> (); // 513
                [80]([9]) -> (); // 514
                [74]([28]) -> (); // 515
                [74]([11]) -> (); // 516
                [73]([170]) -> (); // 517
                [73]([178]) -> (); // 518
                [73]([10]) -> (); // 519
                [7]([211]) -> ([229], [230]); // 520
                [81]([229]) -> (); // 521
                [60]([208]) -> ([203]); // 522
                [91]([230]) -> ([204]); // 523
                [93]([203]) -> ([227]); // 524
                [94]([204]) -> ([228]); // 525
                [42]() { 584() }; // 526
                [38]() -> (); // 527
                [74]([176]) -> (); // 528
                [80]([9]) -> (); // 529
                [74]([28]) -> (); // 530
                [74]([11]) -> (); // 531
                [73]([170]) -> (); // 532
                [73]([10]) -> (); // 533
                [73]([155]) -> (); // 534
                [73]([168]) -> (); // 535
                [73]([169]) -> (); // 536
                [17]() -> ([231]); // 537
                [82]() -> ([232]); // 538
                [63]([232]) -> ([232]); // 539
                [8]([231], [232]) -> ([233]); // 540
                [60]([175]) -> ([227]); // 541
                [91]([233]) -> ([228]); // 542
                [42]() { 584() }; // 543
                [38]() -> (); // 544
                [80]([9]) -> (); // 545
                [74]([28]) -> (); // 546
                [74]([11]) -> (); // 547
                [73]([10]) -> (); // 548
                [73]([155]) -> (); // 549
                [84]([153]) -> (); // 550
                [84]([154]) -> (); // 551
                [7]([165]) -> ([234], [235]); // 552
                [81]([234]) -> (); // 553
                [60]([162]) -> ([227]); // 554
                [91]([235]) -> ([228]); // 555
                [42]() { 584() }; // 556
                [38]() -> (); // 557
                [74]([161]) -> (); // 558
                [80]([9]) -> (); // 559
                [74]([28]) -> (); // 560
                [74]([11]) -> (); // 561
                [84]([154]) -> (); // 562
                [73]([10]) -> (); // 563
                [73]([155]) -> (); // 564
                [84]([153]) -> (); // 565
                [17]() -> ([236]); // 566
                [82]() -> ([237]); // 567
                [63]([237]) -> ([237]); // 568
                [8]([236], [237]) -> ([238]); // 569
                [60]([160]) -> ([227]); // 570
                [91]([238]) -> ([228]); // 571
                [42]() { 584() }; // 572
                [38]() -> (); // 573
                [80]([9]) -> (); // 574
                [74]([28]) -> (); // 575
                [74]([11]) -> (); // 576
                [84]([145]) -> (); // 577
                [73]([10]) -> (); // 578
                [84]([144]) -> (); // 579
                [7]([150]) -> ([239], [240]); // 580
                [81]([239]) -> (); // 581
                [60]([147]) -> ([227]); // 582
                [91]([240]) -> ([228]); // 583
                [6]() -> ([241]); // 584
                [5]([241], [228]) -> ([242]); // 585
                [20]([242]) -> ([243]); // 586
                [60]([227]) -> ([227]); // 587
                [88]([243]) -> ([243]); // 588
                return([227], [243]); // 589
                [38]() -> (); // 590
                [69]([27], [2]) { fallthrough([244], [245]) 598([246], [247], [248]) }; // 591
                [38]() -> (); // 592
                [78]() -> ([249]); // 593
                [60]([244]) -> ([250]); // 594
                [90]([245]) -> ([251]); // 595
                [90]([249]) -> ([252]); // 596
                [42]() { 602() }; // 597
                [38]() -> (); // 598
                [60]([246]) -> ([250]); // 599
                [90]([248]) -> ([251]); // 600
                [90]([247]) -> ([252]); // 601
                [28]([252]) -> ([253]); // 602
                [28]([251]) -> ([254]); // 603
                [76]() -> ([255]); // 604
                [72]([11]) -> ([11], [256]); // 605
                [89]([255]) -> ([255]); // 606
                [64]([250], [255], [256]) { fallthrough([257], [258]) 716([259], [260]) }; // 607
                [38]() -> (); // 608
                [77]() -> ([261]); // 609
                [72]([258]) -> ([258], [262]); // 610
                [89]([261]) -> ([261]); // 611
                [64]([257], [262], [261]) { fallthrough([263], [264]) 659([265], [266]) }; // 612
                [38]() -> (); // 613
                [74]([264]) -> (); // 614
                [77]() -> ([267]); // 615
                [89]([267]) -> ([267]); // 616
                [64]([263], [258], [267]) { fallthrough([268], [269]) 644([270], [271]) }; // 617
                [38]() -> (); // 618
                [60]([268]) -> ([268]); // 619
                [89]([269]) -> ([269]); // 620
                [32]([268], [269]) -> ([272], [273]); // 621
                [31]([273]) { fallthrough([274]) 632([275]) }; // 622
                [38]() -> (); // 623
                [30]([274]) -> ([276]); // 624
                [29]([276]) -> ([277]); // 625
                [28]([277]) -> ([278]); // 626
                [79]() -> ([279]); // 627
                [26]([278], [279]) -> ([280]); // 628
                [60]([272]) -> ([281]); // 629
                [63]([280]) -> ([282]); // 630
                [42]() { 671() }; // 631
                [38]() -> (); // 632
                [80]([9]) -> (); // 633
                [74]([28]) -> (); // 634
                [74]([11]) -> (); // 635
                [73]([254]) -> (); // 636
                [73]([253]) -> (); // 637
                [73]([10]) -> (); // 638
                [7]([275]) -> ([283], [284]); // 639
                [81]([283]) -> (); // 640
                [60]([272]) -> ([285]); // 641
                [91]([284]) -> ([286]); // 642
                [42]() { 713() }; // 643
                [38]() -> (); // 644
                [74]([271]) -> (); // 645
                [80]([9]) -> (); // 646
                [74]([28]) -> (); // 647
                [74]([11]) -> (); // 648
                [73]([254]) -> (); // 649
                [73]([10]) -> (); // 650
                [73]([253]) -> (); // 651
                [17]() -> ([287]); // 652
                [82]() -> ([288]); // 653
                [63]([288]) -> ([288]); // 654
                [8]([287], [288]) -> ([289]); // 655
                [60]([270]) -> ([285]); // 656
                [91]([289]) -> ([286]); // 657
                [42]() { 713() }; // 658
                [38]() -> (); // 659
                [74]([266]) -> (); // 660
                [60]([265]) -> ([265]); // 661
                [89]([258]) -> ([258]); // 662
                [32]([265], [258]) -> ([290], [291]); // 663
                [31]([291]) { fallthrough([292]) 702([293]) }; // 664
                [38]() -> (); // 665
                [30]([292]) -> ([294]); // 666
                [29]([294]) -> ([295]); // 667
                [28]([295]) -> ([296]); // 668
                [60]([290]) -> ([281]); // 669
                [63]([296]) -> ([282]); // 670
                [26]([10], [282]) -> ([297]); // 671
                [63]([297]) -> ([297]); // 672
                [25]([253], [297]) -> ([298]); // 673
                [63]([298]) -> ([298]); // 674
                [67]([281], [298]) { fallthrough([299], [300]) 690([301]) }; // 675
                [38]() -> (); // 676
                [66]([9], [300]) -> ([302]); // 677
                [24]([302], [254], [11]) -> ([303]); // 678
                [60]([299]) -> ([222]); // 679
                [92]([303]) -> ([223]); // 680
                [68]([223]) -> ([304], [305], [306]); // 681
                [74]([306]) -> (); // 682
                [24]([304], [305], [28]) -> ([307]); // 683
                [23]() -> ([308]); // 684
                [22]([307], [308]) -> ([309]); // 685
                [21]([309]) -> ([310]); // 686
                [60]([222]) -> ([222]); // 687
                [88]([310]) -> ([310]); // 688
                return([222], [310]); // 689
                [38]() -> (); // 690
                [80]([9]) -> (); // 691
                [74]([28]) -> (); // 692
                [74]([11]) -> (); // 693
                [73]([254]) -> (); // 694
                [17]() -> ([311]); // 695
                [83]() -> ([312]); // 696
                [63]([312]) -> ([312]); // 697
                [8]([311], [312]) -> ([313]); // 698
                [60]([301]) -> ([314]); // 699
                [91]([313]) -> ([315]); // 700
                [42]() { 730() }; // 701
                [38]() -> (); // 702
                [80]([9]) -> (); // 703
                [74]([28]) -> (); // 704
                [74]([11]) -> (); // 705
                [73]([254]) -> (); // 706
                [73]([253]) -> (); // 707
                [73]([10]) -> (); // 708
                [7]([293]) -> ([316], [317]); // 709
                [81]([316]) -> (); // 710
                [60]([290]) -> ([285]); // 711
                [91]([317]) -> ([286]); // 712
                [93]([285]) -> ([314]); // 713
                [94]([286]) -> ([315]); // 714
                [42]() { 730() }; // 715
                [38]() -> (); // 716
                [74]([260]) -> (); // 717
                [80]([9]) -> (); // 718
                [74]([28]) -> (); // 719
                [74]([11]) -> (); // 720
                [73]([254]) -> (); // 721
                [73]([10]) -> (); // 722
                [73]([253]) -> (); // 723
                [17]() -> ([318]); // 724
                [82]() -> ([319]); // 725
                [63]([319]) -> ([319]); // 726
                [8]([318], [319]) -> ([320]); // 727
                [60]([259]) -> ([314]); // 728
                [91]([320]) -> ([315]); // 729
                [6]() -> ([321]); // 730
                [5]([321], [315]) -> ([322]); // 731
                [20]([322]) -> ([323]); // 732
                [60]([314]) -> ([314]); // 733
                [88]([323]) -> ([323]); // 734
                return([314], [323]); // 735
                [38]() -> (); // 736
                [74]([30]) -> (); // 737
                [80]([9]) -> (); // 738
                [73]([2]) -> (); // 739
                [74]([11]) -> (); // 740
                [73]([10]) -> (); // 741
                [17]() -> ([324]); // 742
                [82]() -> ([325]); // 743
                [63]([325]) -> ([325]); // 744
                [8]([324], [325]) -> ([326]); // 745
                [6]() -> ([327]); // 746
                [5]([327], [326]) -> ([328]); // 747
                [20]([328]) -> ([329]); // 748
                [60]([29]) -> ([29]); // 749
                [88]([329]) -> ([329]); // 750
                return([29], [329]); // 751
                [38]() -> (); // 752
                [74]([11]) -> (); // 753
                [74]([15]) -> (); // 754
                [77]() -> ([330]); // 755
                [72]([3]) -> ([3], [331]); // 756
                [89]([330]) -> ([330]); // 757
                [64]([20], [331], [330]) { fallthrough([332], [333]) 799([334], [335]) }; // 758
                [38]() -> (); // 759
                [74]([333]) -> (); // 760
                [77]() -> ([336]); // 761
                [89]([336]) -> ([336]); // 762
                [64]([332], [3], [336]) { fallthrough([337], [338]) 787([339], [340]) }; // 763
                [38]() -> (); // 764
                [60]([337]) -> ([337]); // 765
                [89]([338]) -> ([338]); // 766
                [32]([337], [338]) -> ([341], [342]); // 767
                [31]([342]) { fallthrough([343]) 778([344]) }; // 768
                [38]() -> (); // 769
                [30]([343]) -> ([345]); // 770
                [29]([345]) -> ([346]); // 771
                [28]([346]) -> ([347]); // 772
                [79]() -> ([348]); // 773
                [26]([347], [348]) -> ([349]); // 774
                [60]([341]) -> ([350]); // 775
                [63]([349]) -> ([351]); // 776
                [42]() { 811() }; // 777
                [38]() -> (); // 778
                [80]([9]) -> (); // 779
                [73]([2]) -> (); // 780
                [73]([10]) -> (); // 781
                [7]([344]) -> ([352], [353]); // 782
                [81]([352]) -> (); // 783
                [60]([341]) -> ([354]); // 784
                [91]([353]) -> ([355]); // 785
                [42]() { 847() }; // 786
                [38]() -> (); // 787
                [74]([340]) -> (); // 788
                [80]([9]) -> (); // 789
                [73]([10]) -> (); // 790
                [73]([2]) -> (); // 791
                [17]() -> ([356]); // 792
                [82]() -> ([357]); // 793
                [63]([357]) -> ([357]); // 794
                [8]([356], [357]) -> ([358]); // 795
                [60]([339]) -> ([354]); // 796
                [91]([358]) -> ([355]); // 797
                [42]() { 847() }; // 798
                [38]() -> (); // 799
                [74]([335]) -> (); // 800
                [60]([334]) -> ([334]); // 801
                [89]([3]) -> ([3]); // 802
                [32]([334], [3]) -> ([359], [360]); // 803
                [31]([360]) { fallthrough([361]) 839([362]) }; // 804
                [38]() -> (); // 805
                [30]([361]) -> ([363]); // 806
                [29]([363]) -> ([364]); // 807
                [28]([364]) -> ([365]); // 808
                [60]([359]) -> ([350]); // 809
                [63]([365]) -> ([351]); // 810
                [26]([10], [351]) -> ([366]); // 811
                [63]([366]) -> ([366]); // 812
                [25]([2], [366]) -> ([367]); // 813
                [63]([367]) -> ([367]); // 814
                [67]([350], [367]) { fallthrough([368], [369]) 827([370]) }; // 815
                [38]() -> (); // 816
                [66]([9], [369]) -> ([371]); // 817
                [85]() -> ([372]); // 818
                [86]() -> ([373]); // 819
                [24]([371], [372], [373]) -> ([374]); // 820
                [23]() -> ([375]); // 821
                [22]([374], [375]) -> ([376]); // 822
                [21]([376]) -> ([377]); // 823
                [60]([368]) -> ([368]); // 824
                [88]([377]) -> ([377]); // 825
                return([368], [377]); // 826
                [38]() -> (); // 827
                [80]([9]) -> (); // 828
                [17]() -> ([378]); // 829
                [83]() -> ([379]); // 830
                [63]([379]) -> ([379]); // 831
                [8]([378], [379]) -> ([380]); // 832
                [6]() -> ([381]); // 833
                [5]([381], [380]) -> ([382]); // 834
                [20]([382]) -> ([383]); // 835
                [60]([370]) -> ([370]); // 836
                [88]([383]) -> ([383]); // 837
                return([370], [383]); // 838
                [38]() -> (); // 839
                [80]([9]) -> (); // 840
                [73]([2]) -> (); // 841
                [73]([10]) -> (); // 842
                [7]([362]) -> ([384], [385]); // 843
                [81]([384]) -> (); // 844
                [60]([359]) -> ([354]); // 845
                [91]([385]) -> ([355]); // 846
                [6]() -> ([386]); // 847
                [5]([386], [355]) -> ([387]); // 848
                [20]([387]) -> ([388]); // 849
                [60]([354]) -> ([354]); // 850
                [88]([388]) -> ([388]); // 851
                return([354], [388]); // 852
                [38]() -> (); // 853
                [74]([23]) -> (); // 854
                [74]([15]) -> (); // 855
                [72]([11]) -> ([11], [389]); // 856
                [60]([22]) -> ([22]); // 857
                [65]([389]) { fallthrough() 869([390]) }; // 858
                [38]() -> (); // 859
                [73]([10]) -> (); // 860
                [74]([11]) -> (); // 861
                [24]([9], [2], [3]) -> ([391]); // 862
                [23]() -> ([392]); // 863
                [22]([391], [392]) -> ([393]); // 864
                [21]([393]) -> ([394]); // 865
                [60]([22]) -> ([22]); // 866
                [88]([394]) -> ([394]); // 867
                return([22], [394]); // 868
                [38]() -> (); // 869
                [75]([390]) -> (); // 870
                [77]() -> ([395]); // 871
                [72]([3]) -> ([3], [396]); // 872
                [89]([395]) -> ([395]); // 873
                [64]([22], [396], [395]) { fallthrough([397], [398]) 920([399], [400]) }; // 874
                [38]() -> (); // 875
                [74]([398]) -> (); // 876
                [77]() -> ([401]); // 877
                [72]([3]) -> ([3], [402]); // 878
                [89]([401]) -> ([401]); // 879
                [64]([397], [402], [401]) { fallthrough([403], [404]) 906([405], [406]) }; // 880
                [38]() -> (); // 881
                [60]([403]) -> ([403]); // 882
                [89]([404]) -> ([404]); // 883
                [32]([403], [404]) -> ([407], [408]); // 884
                [31]([408]) { fallthrough([409]) 895([410]) }; // 885
                [38]() -> (); // 886
                [30]([409]) -> ([411]); // 887
                [29]([411]) -> ([412]); // 888
                [28]([412]) -> ([413]); // 889
                [79]() -> ([414]); // 890
                [26]([413], [414]) -> ([415]); // 891
                [60]([407]) -> ([416]); // 892
                [63]([415]) -> ([417]); // 893
                [42]() { 933() }; // 894
                [38]() -> (); // 895
                [73]([10]) -> (); // 896
                [80]([9]) -> (); // 897
                [73]([2]) -> (); // 898
                [74]([3]) -> (); // 899
                [74]([11]) -> (); // 900
                [7]([410]) -> ([418], [419]); // 901
                [81]([418]) -> (); // 902
                [60]([407]) -> ([420]); // 903
                [91]([419]) -> ([421]); // 904
                [42]() { 971() }; // 905
                [38]() -> (); // 906
                [74]([406]) -> (); // 907
                [73]([10]) -> (); // 908
                [74]([11]) -> (); // 909
                [80]([9]) -> (); // 910
                [73]([2]) -> (); // 911
                [74]([3]) -> (); // 912
                [17]() -> ([422]); // 913
                [82]() -> ([423]); // 914
                [63]([423]) -> ([423]); // 915
                [8]([422], [423]) -> ([424]); // 916
                [60]([405]) -> ([420]); // 917
                [91]([424]) -> ([421]); // 918
                [42]() { 971() }; // 919
                [38]() -> (); // 920
                [74]([400]) -> (); // 921
                [60]([399]) -> ([399]); // 922
                [72]([3]) -> ([3], [425]); // 923
                [89]([425]) -> ([425]); // 924
                [32]([399], [425]) -> ([426], [427]); // 925
                [31]([427]) { fallthrough([428]) 961([429]) }; // 926
                [38]() -> (); // 927
                [30]([428]) -> ([430]); // 928
                [29]([430]) -> ([431]); // 929
                [28]([431]) -> ([432]); // 930
                [60]([426]) -> ([416]); // 931
                [63]([432]) -> ([417]); // 932
                [27]([416], [11], [3]) { fallthrough([433], [434]) 945([435], [436]) }; // 933
                [38]() -> (); // 934
                [26]([10], [417]) -> ([437]); // 935
                [63]([437]) -> ([437]); // 936
                [25]([2], [437]) -> ([438]); // 937
                [24]([9], [438], [434]) -> ([439]); // 938
                [23]() -> ([440]); // 939
                [22]([439], [440]) -> ([441]); // 940
                [21]([441]) -> ([442]); // 941
                [60]([433]) -> ([433]); // 942
                [88]([442]) -> ([442]); // 943
                return([433], [442]); // 944
                [38]() -> (); // 945
                [74]([436]) -> (); // 946
                [73]([10]) -> (); // 947
                [73]([417]) -> (); // 948
                [80]([9]) -> (); // 949
                [73]([2]) -> (); // 950
                [17]() -> ([443]); // 951
                [87]() -> ([444]); // 952
                [63]([444]) -> ([444]); // 953
                [8]([443], [444]) -> ([445]); // 954
                [6]() -> ([446]); // 955
                [5]([446], [445]) -> ([447]); // 956
                [20]([447]) -> ([448]); // 957
                [60]([435]) -> ([435]); // 958
                [88]([448]) -> ([448]); // 959
                return([435], [448]); // 960
                [38]() -> (); // 961
                [73]([10]) -> (); // 962
                [80]([9]) -> (); // 963
                [73]([2]) -> (); // 964
                [74]([3]) -> (); // 965
                [74]([11]) -> (); // 966
                [7]([429]) -> ([449], [450]); // 967
                [81]([449]) -> (); // 968
                [60]([426]) -> ([420]); // 969
                [91]([450]) -> ([421]); // 970
                [6]() -> ([451]); // 971
                [5]([451], [421]) -> ([452]); // 972
                [20]([452]) -> ([453]); // 973
                [60]([420]) -> ([420]); // 974
                [88]([453]) -> ([453]); // 975
                return([420], [453]); // 976
                [38]() -> (); // 977
                [74]([17]) -> (); // 978
                [80]([9]) -> (); // 979
                [73]([2]) -> (); // 980
                [74]([11]) -> (); // 981
                [73]([10]) -> (); // 982
                [74]([3]) -> (); // 983
                [17]() -> ([454]); // 984
                [87]() -> ([455]); // 985
                [63]([455]) -> ([455]); // 986
                [8]([454], [455]) -> ([456]); // 987
                [6]() -> ([457]); // 988
                [5]([457], [456]) -> ([458]); // 989
                [20]([458]) -> ([459]); // 990
                [60]([16]) -> ([16]); // 991
                [88]([459]) -> ([459]); // 992
                return([16], [459]); // 993
                [104]() -> (); // 994
                [140]([0], [1]) { fallthrough([4], [5]) 1026([6], [7]) }; // 995
                [38]() -> (); // 996
                [3]([5]) -> ([8]); // 997
                [139]([2]) -> ([9]); // 998
                [60]([4]) -> ([4]); // 999
                [128]([8]) -> ([8]); // 1000
                [138]([9]) { fallthrough([10], [11]) 1016([12]) }; // 1001
                [38]() -> (); // 1002
                [3]([8]) -> ([13]); // 1003
                [137]([11]) -> ([14]); // 1004
                [141]([14]) -> ([15]); // 1005
                [136]([15]) -> ([16]); // 1006
                [63]([16]) -> ([16]); // 1007
                [8]([3], [16]) -> ([17]); // 1008
                [14]([10]) -> ([18]); // 1009
                [60]([4]) -> ([4]); // 1010
                [128]([13]) -> ([13]); // 1011
                [131]([18]) -> ([18]); // 1012
                [91]([17]) -> ([17]); // 1013
                [13]([4], [13], [18], [17]) -> ([19], [20], [21]); // 1014
                return([19], [20], [21]); // 1015
                [38]() -> (); // 1016
                [122]([12]) -> (); // 1017
                [3]([8]) -> ([22]); // 1018
                [23]() -> ([23]); // 1019
                [135]([3], [23]) -> ([24]); // 1020
                [134]([24]) -> ([25]); // 1021
                [60]([4]) -> ([4]); // 1022
                [128]([22]) -> ([22]); // 1023
                [144]([25]) -> ([25]); // 1024
                return([4], [22], [25]); // 1025
                [38]() -> (); // 1026
                [142]([2]) -> (); // 1027
                [108]([3]) -> (); // 1028
                [3]([7]) -> ([26]); // 1029
                [17]() -> ([27]); // 1030
                [143]() -> ([28]); // 1031
                [63]([28]) -> ([28]); // 1032
                [8]([27], [28]) -> ([29]); // 1033
                [6]() -> ([30]); // 1034
                [5]([30], [29]) -> ([31]); // 1035
                [133]([31]) -> ([32]); // 1036
                [60]([6]) -> ([6]); // 1037
                [128]([26]) -> ([26]); // 1038
                [144]([32]) -> ([32]); // 1039
                return([6], [26], [32]); // 1040
                [37]([0], [1]) { fallthrough([2], [3]) 1131([4]) }; // 1041
                [38]() -> (); // 1042
                [39]([3]) -> ([5]); // 1043
                [59]([5]) -> ([5]); // 1044
                [60]([2]) -> ([2]); // 1045
                [36]([5]) { fallthrough([6]) 1052([7]) 1057([8]) 1062([9]) 1067([10]) 1072([11]) 1077([12]) 1082([13]) 1087([14]) 1092([15]) 1097([16]) 1102([17]) 1107([18]) 1112([19]) 1117([20]) 1122([21]) }; // 1046
                [38]() -> (); // 1047
                [40]([6]) -> (); // 1048
                [41]() -> ([22]); // 1049
                [61]([22]) -> ([23]); // 1050
                [42]() { 1126() }; // 1051
                [38]() -> (); // 1052
                [40]([7]) -> (); // 1053
                [43]() -> ([24]); // 1054
                [61]([24]) -> ([23]); // 1055
                [42]() { 1126() }; // 1056
                [38]() -> (); // 1057
                [40]([8]) -> (); // 1058
                [44]() -> ([25]); // 1059
                [61]([25]) -> ([23]); // 1060
                [42]() { 1126() }; // 1061
                [38]() -> (); // 1062
                [40]([9]) -> (); // 1063
                [45]() -> ([26]); // 1064
                [61]([26]) -> ([23]); // 1065
                [42]() { 1126() }; // 1066
                [38]() -> (); // 1067
                [40]([10]) -> (); // 1068
                [46]() -> ([27]); // 1069
                [61]([27]) -> ([23]); // 1070
                [42]() { 1126() }; // 1071
                [38]() -> (); // 1072
                [40]([11]) -> (); // 1073
                [47]() -> ([28]); // 1074
                [61]([28]) -> ([23]); // 1075
                [42]() { 1126() }; // 1076
                [38]() -> (); // 1077
                [40]([12]) -> (); // 1078
                [48]() -> ([29]); // 1079
                [61]([29]) -> ([23]); // 1080
                [42]() { 1126() }; // 1081
                [38]() -> (); // 1082
                [40]([13]) -> (); // 1083
                [49]() -> ([30]); // 1084
                [61]([30]) -> ([23]); // 1085
                [42]() { 1126() }; // 1086
                [38]() -> (); // 1087
                [40]([14]) -> (); // 1088
                [50]() -> ([31]); // 1089
                [61]([31]) -> ([23]); // 1090
                [42]() { 1126() }; // 1091
                [38]() -> (); // 1092
                [40]([15]) -> (); // 1093
                [51]() -> ([32]); // 1094
                [61]([32]) -> ([23]); // 1095
                [42]() { 1126() }; // 1096
                [38]() -> (); // 1097
                [40]([16]) -> (); // 1098
                [52]() -> ([33]); // 1099
                [61]([33]) -> ([23]); // 1100
                [42]() { 1126() }; // 1101
                [38]() -> (); // 1102
                [40]([17]) -> (); // 1103
                [53]() -> ([34]); // 1104
                [61]([34]) -> ([23]); // 1105
                [42]() { 1126() }; // 1106
                [38]() -> (); // 1107
                [40]([18]) -> (); // 1108
                [54]() -> ([35]); // 1109
                [61]([35]) -> ([23]); // 1110
                [42]() { 1126() }; // 1111
                [38]() -> (); // 1112
                [40]([19]) -> (); // 1113
                [55]() -> ([36]); // 1114
                [61]([36]) -> ([23]); // 1115
                [42]() { 1126() }; // 1116
                [38]() -> (); // 1117
                [40]([20]) -> (); // 1118
                [56]() -> ([37]); // 1119
                [61]([37]) -> ([23]); // 1120
                [42]() { 1126() }; // 1121
                [38]() -> (); // 1122
                [40]([21]) -> (); // 1123
                [57]() -> ([38]); // 1124
                [61]([38]) -> ([23]); // 1125
                [35]([23]) -> ([39]); // 1126
                [34]([39]) -> ([40]); // 1127
                [60]([2]) -> ([2]); // 1128
                [62]([40]) -> ([40]); // 1129
                return([2], [40]); // 1130
                [38]() -> (); // 1131
                [17]() -> ([41]); // 1132
                [58]() -> ([42]); // 1133
                [63]([42]) -> ([42]); // 1134
                [8]([41], [42]) -> ([43]); // 1135
                [6]() -> ([44]); // 1136
                [5]([44], [43]) -> ([45]); // 1137
                [33]([45]) -> ([46]); // 1138
                [60]([4]) -> ([4]); // 1139
                [62]([46]) -> ([46]); // 1140
                return([4], [46]); // 1141

                [3]@0([0]: [0], [1]: [1]) -> ([0], [1], [24]);
                [1]@152([0]: [0], [1]: [12], [2]: [2], [3]: [11]) -> ([0], [16]);
                [0]@994([0]: [0], [1]: [1], [2]: [19], [3]: [3]) -> ([0], [1], [21]);
                [2]@1041([0]: [0], [1]: [11]) -> ([0], [29]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let result = run_sierra_program(&program, &[]).return_value;

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
        // use array::ArrayTrait;
        // fn run_test() -> (Span<felt252>, @Box<[felt252; 3]>) {
        //     let mut numbers = array![1, 2, 3, 4, 5, 6].span();
        //     let popped = numbers.multi_pop_back::<3>().unwrap();

        //     (numbers, popped)
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [2] = Array<[1]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [9] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
                type [10] = Struct<ut@Tuple, [9], [2]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [18] = Const<[1], 29721761890975875353235833581453094220424382983267374> [storable: false, drop: false, dup: false, zero_sized: false];
                type [3] = Snapshot<[2]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [6] = Struct<ut@core::array::Span::<core::felt252>, [3]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [5] = Box<[4]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [7] = Struct<ut@Tuple, [6], [5]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [8] = Struct<ut@Tuple, [7]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [11] = Enum<ut@core::panics::PanicResult::<((core::array::Span::<core::felt252>, @core::box::Box::<[core::felt252; 3]>),)>, [8], [10]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [1] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
                type [4] = Struct<ut@Tuple, [1], [1], [1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
                type [17] = Const<[1], 6> [storable: false, drop: false, dup: false, zero_sized: false];
                type [16] = Const<[1], 5> [storable: false, drop: false, dup: false, zero_sized: false];
                type [15] = Const<[1], 4> [storable: false, drop: false, dup: false, zero_sized: false];
                type [14] = Const<[1], 3> [storable: false, drop: false, dup: false, zero_sized: false];
                type [13] = Const<[1], 2> [storable: false, drop: false, dup: false, zero_sized: false];
                type [12] = Const<[1], 1> [storable: false, drop: false, dup: false, zero_sized: false];

                libfunc [4] = array_new<[1]>;
                libfunc [11] = const_as_immediate<[12]>;
                libfunc [22] = store_temp<[1]>;
                libfunc [3] = array_append<[1]>;
                libfunc [12] = const_as_immediate<[13]>;
                libfunc [13] = const_as_immediate<[14]>;
                libfunc [14] = const_as_immediate<[15]>;
                libfunc [15] = const_as_immediate<[16]>;
                libfunc [16] = const_as_immediate<[17]>;
                libfunc [17] = snapshot_take<[2]>;
                libfunc [18] = drop<[2]>;
                libfunc [23] = store_temp<[3]>;
                libfunc [9] = array_snapshot_multi_pop_back<[4]>;
                libfunc [19] = branch_align;
                libfunc [8] = struct_construct<[6]>;
                libfunc [7] = struct_construct<[7]>;
                libfunc [6] = struct_construct<[8]>;
                libfunc [5] = enum_init<[11], 0>;
                libfunc [24] = store_temp<[0]>;
                libfunc [25] = store_temp<[11]>;
                libfunc [20] = drop<[3]>;
                libfunc [21] = const_as_immediate<[18]>;
                libfunc [2] = struct_construct<[9]>;
                libfunc [1] = struct_construct<[10]>;
                libfunc [0] = enum_init<[11], 1>;

                [4]() -> ([1]); // 0
                [11]() -> ([2]); // 1
                [22]([2]) -> ([2]); // 2
                [3]([1], [2]) -> ([3]); // 3
                [12]() -> ([4]); // 4
                [22]([4]) -> ([4]); // 5
                [3]([3], [4]) -> ([5]); // 6
                [13]() -> ([6]); // 7
                [22]([6]) -> ([6]); // 8
                [3]([5], [6]) -> ([7]); // 9
                [14]() -> ([8]); // 10
                [22]([8]) -> ([8]); // 11
                [3]([7], [8]) -> ([9]); // 12
                [15]() -> ([10]); // 13
                [22]([10]) -> ([10]); // 14
                [3]([9], [10]) -> ([11]); // 15
                [16]() -> ([12]); // 16
                [22]([12]) -> ([12]); // 17
                [3]([11], [12]) -> ([13]); // 18
                [17]([13]) -> ([14], [15]); // 19
                [18]([14]) -> (); // 20
                [23]([15]) -> ([15]); // 21
                [9]([0], [15]) { fallthrough([16], [17], [18]) 31([19], [20]) }; // 22
                [19]() -> (); // 23
                [8]([17]) -> ([21]); // 24
                [7]([21], [18]) -> ([22]); // 25
                [6]([22]) -> ([23]); // 26
                [5]([23]) -> ([24]); // 27
                [24]([16]) -> ([16]); // 28
                [25]([24]) -> ([24]); // 29
                return([16], [24]); // 30
                [19]() -> (); // 31
                [20]([20]) -> (); // 32
                [4]() -> ([25]); // 33
                [21]() -> ([26]); // 34
                [22]([26]) -> ([26]); // 35
                [3]([25], [26]) -> ([27]); // 36
                [2]() -> ([28]); // 37
                [1]([28], [27]) -> ([29]); // 38
                [0]([29]) -> ([30]); // 39
                [24]([19]) -> ([19]); // 40
                [25]([30]) -> ([30]); // 41
                return([19], [30]); // 42

                [0]@0([0]: [0]) -> ([0], [11]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let result = run_sierra_program(&program, &[]).return_value;

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
        // use array::ArrayTrait;
        // fn run_test() -> Span<felt252> {
        //     let mut numbers = array![1, 2].span();
        //     // should fail (return none)
        //     assert!(numbers.multi_pop_back::<3>().is_none());
        //     numbers
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [10] = Array<[9]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [2] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
                type [11] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
                type [12] = Struct<ut@core::byte_array::ByteArray, [10], [2], [11]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [80] = Uninitialized<[12]> [storable: false, drop: true, dup: false, zero_sized: false];
                type [64] = Const<[2], 573087285299505011920718992710461799> [storable: false, drop: false, dup: false, zero_sized: false];
                type [63] = Const<[27], [62]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [61] = Const<[27], [60]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [62] = Const<[26], 1329227995784915872903807060280344576> [storable: false, drop: false, dup: false, zero_sized: false];
                type [60] = Const<[26], 5192296858534827628530496329220096> [storable: false, drop: false, dup: false, zero_sized: false];
                type [59] = Const<[27], [58]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [57] = Const<[27], [56]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [58] = Const<[26], 20282409603651670423947251286016> [storable: false, drop: false, dup: false, zero_sized: false];
                type [56] = Const<[26], 79228162514264337593543950336> [storable: false, drop: false, dup: false, zero_sized: false];
                type [55] = Const<[27], [54]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [53] = Const<[27], [52]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [54] = Const<[26], 309485009821345068724781056> [storable: false, drop: false, dup: false, zero_sized: false];
                type [52] = Const<[26], 1208925819614629174706176> [storable: false, drop: false, dup: false, zero_sized: false];
                type [51] = Const<[27], [50]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [49] = Const<[27], [48]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [50] = Const<[26], 4722366482869645213696> [storable: false, drop: false, dup: false, zero_sized: false];
                type [48] = Const<[26], 18446744073709551616> [storable: false, drop: false, dup: false, zero_sized: false];
                type [47] = Const<[27], [46]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [45] = Const<[27], [44]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [46] = Const<[26], 72057594037927936> [storable: false, drop: false, dup: false, zero_sized: false];
                type [44] = Const<[26], 281474976710656> [storable: false, drop: false, dup: false, zero_sized: false];
                type [43] = Const<[27], [42]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [41] = Const<[27], [40]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [42] = Const<[26], 1099511627776> [storable: false, drop: false, dup: false, zero_sized: false];
                type [40] = Const<[26], 4294967296> [storable: false, drop: false, dup: false, zero_sized: false];
                type [39] = Const<[27], [38]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [37] = Const<[27], [36]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [38] = Const<[26], 16777216> [storable: false, drop: false, dup: false, zero_sized: false];
                type [36] = Const<[26], 65536> [storable: false, drop: false, dup: false, zero_sized: false];
                type [35] = Const<[27], [34]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [33] = Const<[27], [32]> [storable: false, drop: false, dup: false, zero_sized: false];
                type [34] = Const<[26], 256> [storable: false, drop: false, dup: false, zero_sized: false];
                type [32] = Const<[26], 1> [storable: false, drop: false, dup: false, zero_sized: false];
                type [7] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
                type [31] = Enum<ut@index_enum_type<16>, [7], [7], [7], [7], [7], [7], [7], [7], [7], [7], [7], [7], [7], [7], [7], [7]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [30] = BoundedInt<0, 15> [storable: true, drop: true, dup: true, zero_sized: false];
                type [82] = Const<[2], 375233589013918064796019> [storable: false, drop: false, dup: false, zero_sized: false];
                type [81] = Box<[9]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [73] = Const<[2], 155785504323917466144735657540098748279> [storable: false, drop: false, dup: false, zero_sized: false];
                type [70] = Const<[2], 29721761890975875353235833581453094220424382983267374> [storable: false, drop: false, dup: false, zero_sized: false];
                type [69] = Const<[2], 155785504329508738615720351733824384887> [storable: false, drop: false, dup: false, zero_sized: false];
                type [68] = Const<[2], 340282366920938463463374607431768211456> [storable: false, drop: false, dup: false, zero_sized: false];
                type [26] = u128 [storable: true, drop: true, dup: true, zero_sized: false];
                type [27] = NonZero<[26]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [28] = Struct<ut@Tuple, [27]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [14] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
                type [3] = Array<[2]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [15] = Struct<ut@Tuple, [14], [3]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [29] = Enum<ut@core::panics::PanicResult::<(core::zeroable::NonZero::<core::integer::u128>,)>, [28], [15]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [67] = Const<[26], 0> [storable: false, drop: false, dup: false, zero_sized: false];
                type [66] = Const<[11], 16> [storable: false, drop: false, dup: false, zero_sized: false];
                type [25] = NonZero<[11]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [4] = Snapshot<[3]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [22] = Struct<ut@core::array::Span::<core::felt252>, [4]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [23] = Struct<ut@Tuple, [22]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [24] = Enum<ut@core::panics::PanicResult::<(core::array::Span::<core::felt252>,)>, [23], [15]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [20] = Struct<ut@Tuple, [3], [7]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [21] = Enum<ut@core::panics::PanicResult::<(core::array::Array::<core::felt252>, ())>, [20], [15]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [18] = Snapshot<[10]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [19] = Struct<ut@core::array::Span::<core::bytes_31::bytes31>, [18]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [17] = Snapshot<[12]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [79] = Const<[2], 1997209042069643135709344952807065910992472029923670688473712229447419591075> [storable: false, drop: false, dup: false, zero_sized: false];
                type [78] = Const<[11], 29> [storable: false, drop: false, dup: false, zero_sized: false];
                type [77] = Const<[2], 2840845320962432228251361402070922274004853694903599195568980986781742> [storable: false, drop: false, dup: false, zero_sized: false];
                type [13] = Struct<ut@Tuple, [12], [7]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [16] = Enum<ut@core::panics::PanicResult::<(core::byte_array::ByteArray, ())>, [13], [15]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [65] = Const<[11], 31> [storable: false, drop: false, dup: false, zero_sized: false];
                type [76] = Const<[2], 172180977190876322177717838039515195832848434335613823290676811071835434100> [storable: false, drop: false, dup: false, zero_sized: false];
                type [72] = Const<[11], 0> [storable: false, drop: false, dup: false, zero_sized: false];
                type [71] = Const<[2], 0> [storable: false, drop: false, dup: false, zero_sized: false];
                type [9] = bytes31 [storable: true, drop: true, dup: true, zero_sized: false];
                type [6] = Box<[5]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [8] = Enum<ut@core::option::Option::<@core::box::Box::<[core::felt252; 3]>>, [6], [7]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [1] = GasBuiltin [storable: true, drop: false, dup: false, zero_sized: false];
                type [5] = Struct<ut@Tuple, [2], [2], [2]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
                type [75] = Const<[2], 2> [storable: false, drop: false, dup: false, zero_sized: false];
                type [74] = Const<[2], 1> [storable: false, drop: false, dup: false, zero_sized: false];

                libfunc [102] = alloc_local<[12]>;
                libfunc [103] = finalize_locals;
                libfunc [104] = disable_ap_tracking;
                libfunc [17] = array_new<[2]>;
                libfunc [105] = const_as_immediate<[74]>;
                libfunc [63] = store_temp<[2]>;
                libfunc [8] = array_append<[2]>;
                libfunc [106] = const_as_immediate<[75]>;
                libfunc [107] = snapshot_take<[3]>;
                libfunc [108] = drop<[3]>;
                libfunc [127] = store_temp<[4]>;
                libfunc [100] = array_snapshot_multi_pop_back<[5]>;
                libfunc [38] = branch_align;
                libfunc [3] = redeposit_gas;
                libfunc [99] = enum_init<[8], 0>;
                libfunc [60] = store_temp<[0]>;
                libfunc [128] = store_temp<[1]>;
                libfunc [129] = store_temp<[8]>;
                libfunc [42] = jump;
                libfunc [23] = struct_construct<[7]>;
                libfunc [98] = enum_init<[8], 1>;
                libfunc [109] = snapshot_take<[8]>;
                libfunc [110] = drop<[8]>;
                libfunc [97] = enum_match<[8]>;
                libfunc [111] = drop<[6]>;
                libfunc [112] = drop<[4]>;
                libfunc [96] = array_new<[9]>;
                libfunc [85] = const_as_immediate<[71]>;
                libfunc [86] = const_as_immediate<[72]>;
                libfunc [113] = const_as_immediate<[76]>;
                libfunc [76] = const_as_immediate<[65]>;
                libfunc [24] = struct_construct<[12]>;
                libfunc [92] = store_temp<[12]>;
                libfunc [89] = store_temp<[11]>;
                libfunc [19] = function_call<user@[1]>;
                libfunc [18] = enum_match<[16]>;
                libfunc [16] = struct_deconstruct<[13]>;
                libfunc [40] = drop<[7]>;
                libfunc [114] = const_as_immediate<[77]>;
                libfunc [115] = const_as_immediate<[78]>;
                libfunc [116] = const_as_immediate<[79]>;
                libfunc [130] = store_local<[12]>;
                libfunc [117] = snapshot_take<[12]>;
                libfunc [118] = drop<[12]>;
                libfunc [119] = dup<[17]>;
                libfunc [10] = struct_snapshot_deconstruct<[12]>;
                libfunc [73] = drop<[2]>;
                libfunc [74] = drop<[11]>;
                libfunc [120] = dup<[18]>;
                libfunc [15] = array_len<[9]>;
                libfunc [9] = u32_to_felt252;
                libfunc [14] = struct_construct<[19]>;
                libfunc [131] = store_temp<[19]>;
                libfunc [91] = store_temp<[3]>;
                libfunc [13] = function_call<user@[0]>;
                libfunc [121] = enable_ap_tracking;
                libfunc [12] = enum_match<[21]>;
                libfunc [11] = struct_deconstruct<[20]>;
                libfunc [122] = drop<[18]>;
                libfunc [123] = rename<[2]>;
                libfunc [124] = rename<[11]>;
                libfunc [125] = drop<[17]>;
                libfunc [7] = struct_deconstruct<[15]>;
                libfunc [81] = drop<[14]>;
                libfunc [6] = struct_construct<[14]>;
                libfunc [5] = struct_construct<[15]>;
                libfunc [4] = enum_init<[24], 1>;
                libfunc [132] = store_temp<[24]>;
                libfunc [126] = drop<[80]>;
                libfunc [2] = struct_construct<[22]>;
                libfunc [1] = struct_construct<[23]>;
                libfunc [0] = enum_init<[24], 0>;
                libfunc [72] = dup<[11]>;
                libfunc [65] = u32_is_zero;
                libfunc [22] = struct_construct<[13]>;
                libfunc [21] = enum_init<[16], 0>;
                libfunc [88] = store_temp<[16]>;
                libfunc [75] = drop<[25]>;
                libfunc [68] = struct_deconstruct<[12]>;
                libfunc [27] = u32_overflowing_add;
                libfunc [64] = u32_overflowing_sub;
                libfunc [71] = u32_eq;
                libfunc [77] = const_as_immediate<[66]>;
                libfunc [69] = u128s_from_felt252;
                libfunc [78] = const_as_immediate<[67]>;
                libfunc [90] = store_temp<[26]>;
                libfunc [32] = function_call<user@[2]>;
                libfunc [31] = enum_match<[29]>;
                libfunc [30] = struct_deconstruct<[28]>;
                libfunc [70] = u128_safe_divmod;
                libfunc [28] = u128_to_felt252;
                libfunc [79] = const_as_immediate<[68]>;
                libfunc [26] = felt252_mul;
                libfunc [25] = felt252_add;
                libfunc [29] = unwrap_non_zero<[26]>;
                libfunc [80] = drop<[10]>;
                libfunc [82] = const_as_immediate<[69]>;
                libfunc [67] = bytes31_try_from_felt252;
                libfunc [66] = array_append<[9]>;
                libfunc [83] = const_as_immediate<[70]>;
                libfunc [93] = rename<[0]>;
                libfunc [94] = rename<[3]>;
                libfunc [84] = drop<[26]>;
                libfunc [20] = enum_init<[16], 1>;
                libfunc [95] = rename<[12]>;
                libfunc [87] = const_as_immediate<[73]>;
                libfunc [140] = withdraw_gas;
                libfunc [139] = struct_deconstruct<[19]>;
                libfunc [138] = array_snapshot_pop_front<[9]>;
                libfunc [137] = unbox<[9]>;
                libfunc [141] = rename<[9]>;
                libfunc [136] = bytes31_to_felt252;
                libfunc [135] = struct_construct<[20]>;
                libfunc [134] = enum_init<[21], 0>;
                libfunc [144] = store_temp<[21]>;
                libfunc [142] = drop<[19]>;
                libfunc [143] = const_as_immediate<[82]>;
                libfunc [133] = enum_init<[21], 1>;
                libfunc [37] = downcast<[11], [30]>;
                libfunc [39] = enum_from_bounded_int<[31]>;
                libfunc [59] = store_temp<[31]>;
                libfunc [36] = enum_match<[31]>;
                libfunc [41] = const_as_immediate<[33]>;
                libfunc [61] = store_temp<[27]>;
                libfunc [43] = const_as_immediate<[35]>;
                libfunc [44] = const_as_immediate<[37]>;
                libfunc [45] = const_as_immediate<[39]>;
                libfunc [46] = const_as_immediate<[41]>;
                libfunc [47] = const_as_immediate<[43]>;
                libfunc [48] = const_as_immediate<[45]>;
                libfunc [49] = const_as_immediate<[47]>;
                libfunc [50] = const_as_immediate<[49]>;
                libfunc [51] = const_as_immediate<[51]>;
                libfunc [52] = const_as_immediate<[53]>;
                libfunc [53] = const_as_immediate<[55]>;
                libfunc [54] = const_as_immediate<[57]>;
                libfunc [55] = const_as_immediate<[59]>;
                libfunc [56] = const_as_immediate<[61]>;
                libfunc [57] = const_as_immediate<[63]>;
                libfunc [35] = struct_construct<[28]>;
                libfunc [34] = enum_init<[29], 0>;
                libfunc [62] = store_temp<[29]>;
                libfunc [58] = const_as_immediate<[64]>;
                libfunc [33] = enum_init<[29], 1>;

                [102]() -> ([3]); // 0
                [103]() -> (); // 1
                [104]() -> (); // 2
                [17]() -> ([4]); // 3
                [105]() -> ([5]); // 4
                [63]([5]) -> ([5]); // 5
                [8]([4], [5]) -> ([6]); // 6
                [106]() -> ([7]); // 7
                [63]([7]) -> ([7]); // 8
                [8]([6], [7]) -> ([8]); // 9
                [107]([8]) -> ([9], [10]); // 10
                [108]([9]) -> (); // 11
                [127]([10]) -> ([10]); // 12
                [100]([0], [10]) { fallthrough([11], [12], [13]) 22([14], [15]) }; // 13
                [38]() -> (); // 14
                [3]([1]) -> ([16]); // 15
                [99]([13]) -> ([17]); // 16
                [60]([11]) -> ([18]); // 17
                [128]([16]) -> ([19]); // 18
                [127]([12]) -> ([20]); // 19
                [129]([17]) -> ([21]); // 20
                [42]() { 30() }; // 21
                [38]() -> (); // 22
                [3]([1]) -> ([22]); // 23
                [23]() -> ([23]); // 24
                [98]([23]) -> ([24]); // 25
                [60]([14]) -> ([18]); // 26
                [128]([22]) -> ([19]); // 27
                [127]([15]) -> ([20]); // 28
                [129]([24]) -> ([21]); // 29
                [109]([21]) -> ([25], [26]); // 30
                [110]([25]) -> (); // 31
                [97]([26]) { fallthrough([27]) 141([28]) }; // 32
                [38]() -> (); // 33
                [111]([27]) -> (); // 34
                [112]([20]) -> (); // 35
                [3]([19]) -> ([29]); // 36
                [96]() -> ([30]); // 37
                [85]() -> ([31]); // 38
                [86]() -> ([32]); // 39
                [113]() -> ([33]); // 40
                [76]() -> ([34]); // 41
                [24]([30], [31], [32]) -> ([35]); // 42
                [60]([18]) -> ([18]); // 43
                [92]([35]) -> ([35]); // 44
                [63]([33]) -> ([33]); // 45
                [89]([34]) -> ([34]); // 46
                [19]([18], [35], [33], [34]) -> ([36], [37]); // 47
                [128]([29]) -> ([29]); // 48
                [18]([37]) { fallthrough([38]) 133([39]) }; // 49
                [38]() -> (); // 50
                [3]([29]) -> ([40]); // 51
                [16]([38]) -> ([41], [42]); // 52
                [40]([42]) -> (); // 53
                [114]() -> ([43]); // 54
                [115]() -> ([44]); // 55
                [60]([36]) -> ([36]); // 56
                [92]([41]) -> ([41]); // 57
                [63]([43]) -> ([43]); // 58
                [89]([44]) -> ([44]); // 59
                [19]([36], [41], [43], [44]) -> ([45], [46]); // 60
                [128]([40]) -> ([40]); // 61
                [18]([46]) { fallthrough([47]) 125([48]) }; // 62
                [38]() -> (); // 63
                [3]([40]) -> ([49]); // 64
                [17]() -> ([50]); // 65
                [116]() -> ([51]); // 66
                [63]([51]) -> ([51]); // 67
                [8]([50], [51]) -> ([52]); // 68
                [16]([47]) -> ([2], [53]); // 69
                [40]([53]) -> (); // 70
                [130]([3], [2]) -> ([2]); // 71
                [117]([2]) -> ([54], [55]); // 72
                [118]([54]) -> (); // 73
                [119]([55]) -> ([55], [56]); // 74
                [10]([56]) -> ([57], [58], [59]); // 75
                [73]([58]) -> (); // 76
                [74]([59]) -> (); // 77
                [120]([57]) -> ([57], [60]); // 78
                [15]([60]) -> ([61]); // 79
                [9]([61]) -> ([62]); // 80
                [63]([62]) -> ([62]); // 81
                [8]([52], [62]) -> ([63]); // 82
                [14]([57]) -> ([64]); // 83
                [60]([45]) -> ([45]); // 84
                [128]([49]) -> ([49]); // 85
                [131]([64]) -> ([64]); // 86
                [91]([63]) -> ([63]); // 87
                [13]([45], [49], [64], [63]) -> ([65], [66], [67]); // 88
                [121]() -> (); // 89
                [12]([67]) { fallthrough([68]) 110([69]) }; // 90
                [38]() -> (); // 91
                [3]([66]) -> ([70]); // 92
                [11]([68]) -> ([71], [72]); // 93
                [40]([72]) -> (); // 94
                [119]([55]) -> ([55], [73]); // 95
                [10]([73]) -> ([74], [75], [76]); // 96
                [122]([74]) -> (); // 97
                [74]([76]) -> (); // 98
                [123]([75]) -> ([77]); // 99
                [8]([71], [77]) -> ([78]); // 100
                [10]([55]) -> ([79], [80], [81]); // 101
                [122]([79]) -> (); // 102
                [73]([80]) -> (); // 103
                [124]([81]) -> ([82]); // 104
                [9]([82]) -> ([83]); // 105
                [8]([78], [83]) -> ([84]); // 106
                [128]([70]) -> ([85]); // 107
                [91]([84]) -> ([86]); // 108
                [42]() { 117() }; // 109
                [38]() -> (); // 110
                [125]([55]) -> (); // 111
                [3]([66]) -> ([87]); // 112
                [7]([69]) -> ([88], [89]); // 113
                [81]([88]) -> (); // 114
                [128]([87]) -> ([85]); // 115
                [91]([89]) -> ([86]); // 116
                [104]() -> (); // 117
                [6]() -> ([90]); // 118
                [5]([90], [86]) -> ([91]); // 119
                [4]([91]) -> ([92]); // 120
                [60]([65]) -> ([65]); // 121
                [128]([85]) -> ([85]); // 122
                [132]([92]) -> ([92]); // 123
                return([65], [85], [92]); // 124
                [38]() -> (); // 125
                [126]([3]) -> (); // 126
                [3]([40]) -> ([93]); // 127
                [4]([48]) -> ([94]); // 128
                [60]([45]) -> ([45]); // 129
                [128]([93]) -> ([93]); // 130
                [132]([94]) -> ([94]); // 131
                return([45], [93], [94]); // 132
                [38]() -> (); // 133
                [126]([3]) -> (); // 134
                [3]([29]) -> ([95]); // 135
                [4]([39]) -> ([96]); // 136
                [60]([36]) -> ([36]); // 137
                [128]([95]) -> ([95]); // 138
                [132]([96]) -> ([96]); // 139
                return([36], [95], [96]); // 140
                [38]() -> (); // 141
                [40]([28]) -> (); // 142
                [126]([3]) -> (); // 143
                [3]([19]) -> ([97]); // 144
                [2]([20]) -> ([98]); // 145
                [1]([98]) -> ([99]); // 146
                [0]([99]) -> ([100]); // 147
                [60]([18]) -> ([18]); // 148
                [128]([97]) -> ([97]); // 149
                [132]([100]) -> ([100]); // 150
                return([18], [97], [100]); // 151
                [72]([3]) -> ([3], [4]); // 152
                [65]([4]) { fallthrough() 163([5]) }; // 153
                [38]() -> (); // 154
                [73]([2]) -> (); // 155
                [74]([3]) -> (); // 156
                [23]() -> ([6]); // 157
                [22]([1], [6]) -> ([7]); // 158
                [21]([7]) -> ([8]); // 159
                [60]([0]) -> ([0]); // 160
                [88]([8]) -> ([8]); // 161
                return([0], [8]); // 162
                [38]() -> (); // 163
                [75]([5]) -> (); // 164
                [68]([1]) -> ([9], [10], [11]); // 165
                [72]([11]) -> ([11], [12]); // 166
                [72]([3]) -> ([3], [13]); // 167
                [27]([0], [12], [13]) { fallthrough([14], [15]) 977([16], [17]) }; // 168
                [38]() -> (); // 169
                [76]() -> ([18]); // 170
                [72]([15]) -> ([15], [19]); // 171
                [89]([18]) -> ([18]); // 172
                [64]([14], [19], [18]) { fallthrough([20], [21]) 853([22], [23]) }; // 173
                [38]() -> (); // 174
                [74]([21]) -> (); // 175
                [76]() -> ([24]); // 176
                [72]([15]) -> ([15], [25]); // 177
                [60]([20]) -> ([20]); // 178
                [71]([25], [24]) { fallthrough() 752() }; // 179
                [38]() -> (); // 180
                [74]([3]) -> (); // 181
                [76]() -> ([26]); // 182
                [89]([26]) -> ([26]); // 183
                [64]([20], [15], [26]) { fallthrough([27], [28]) 736([29], [30]) }; // 184
                [38]() -> (); // 185
                [77]() -> ([31]); // 186
                [72]([28]) -> ([28], [32]); // 187
                [60]([27]) -> ([27]); // 188
                [71]([32], [31]) { fallthrough() 590() }; // 189
                [38]() -> (); // 190
                [77]() -> ([33]); // 191
                [72]([28]) -> ([28], [34]); // 192
                [89]([33]) -> ([33]); // 193
                [64]([27], [34], [33]) { fallthrough([35], [36]) 380([37], [38]) }; // 194
                [38]() -> (); // 195
                [74]([36]) -> (); // 196
                [69]([35], [2]) { fallthrough([39], [40]) 204([41], [42], [43]) }; // 197
                [38]() -> (); // 198
                [78]() -> ([44]); // 199
                [60]([39]) -> ([45]); // 200
                [90]([40]) -> ([46]); // 201
                [90]([44]) -> ([47]); // 202
                [42]() { 208() }; // 203
                [38]() -> (); // 204
                [60]([41]) -> ([45]); // 205
                [90]([43]) -> ([46]); // 206
                [90]([42]) -> ([47]); // 207
                [77]() -> ([48]); // 208
                [72]([28]) -> ([28], [49]); // 209
                [89]([48]) -> ([48]); // 210
                [64]([45], [49], [48]) { fallthrough([50], [51]) 360([52], [53]) }; // 211
                [38]() -> (); // 212
                [60]([50]) -> ([50]); // 213
                [89]([51]) -> ([51]); // 214
                [32]([50], [51]) -> ([54], [55]); // 215
                [31]([55]) { fallthrough([56]) 348([57]) }; // 216
                [38]() -> (); // 217
                [30]([56]) -> ([58]); // 218
                [70]([54], [47], [58]) -> ([59], [60], [61]); // 219
                [28]([61]) -> ([62]); // 220
                [28]([46]) -> ([63]); // 221
                [28]([60]) -> ([64]); // 222
                [76]() -> ([65]); // 223
                [72]([11]) -> ([11], [66]); // 224
                [89]([65]) -> ([65]); // 225
                [64]([59], [65], [66]) { fallthrough([67], [68]) 332([69], [70]) }; // 226
                [38]() -> (); // 227
                [79]() -> ([71]); // 228
                [26]([62], [71]) -> ([72]); // 229
                [63]([72]) -> ([72]); // 230
                [25]([72], [63]) -> ([73]); // 231
                [77]() -> ([74]); // 232
                [72]([68]) -> ([68], [75]); // 233
                [89]([74]) -> ([74]); // 234
                [63]([73]) -> ([73]); // 235
                [64]([67], [75], [74]) { fallthrough([76], [77]) 283([78], [79]) }; // 236
                [38]() -> (); // 237
                [74]([77]) -> (); // 238
                [77]() -> ([80]); // 239
                [89]([80]) -> ([80]); // 240
                [64]([76], [68], [80]) { fallthrough([81], [82]) 268([83], [84]) }; // 241
                [38]() -> (); // 242
                [60]([81]) -> ([81]); // 243
                [89]([82]) -> ([82]); // 244
                [32]([81], [82]) -> ([85], [86]); // 245
                [31]([86]) { fallthrough([87]) 256([88]) }; // 246
                [38]() -> (); // 247
                [30]([87]) -> ([89]); // 248
                [29]([89]) -> ([90]); // 249
                [28]([90]) -> ([91]); // 250
                [79]() -> ([92]); // 251
                [26]([91], [92]) -> ([93]); // 252
                [60]([85]) -> ([94]); // 253
                [63]([93]) -> ([95]); // 254
                [42]() { 295() }; // 255
                [38]() -> (); // 256
                [80]([9]) -> (); // 257
                [74]([28]) -> (); // 258
                [74]([11]) -> (); // 259
                [73]([73]) -> (); // 260
                [73]([64]) -> (); // 261
                [73]([10]) -> (); // 262
                [7]([88]) -> ([96], [97]); // 263
                [81]([96]) -> (); // 264
                [60]([85]) -> ([98]); // 265
                [91]([97]) -> ([99]); // 266
                [42]() { 329() }; // 267
                [38]() -> (); // 268
                [74]([84]) -> (); // 269
                [80]([9]) -> (); // 270
                [74]([28]) -> (); // 271
                [74]([11]) -> (); // 272
                [73]([73]) -> (); // 273
                [73]([10]) -> (); // 274
                [73]([64]) -> (); // 275
                [17]() -> ([100]); // 276
                [82]() -> ([101]); // 277
                [63]([101]) -> ([101]); // 278
                [8]([100], [101]) -> ([102]); // 279
                [60]([83]) -> ([98]); // 280
                [91]([102]) -> ([99]); // 281
                [42]() { 329() }; // 282
                [38]() -> (); // 283
                [74]([79]) -> (); // 284
                [60]([78]) -> ([78]); // 285
                [89]([68]) -> ([68]); // 286
                [32]([78], [68]) -> ([103], [104]); // 287
                [31]([104]) { fallthrough([105]) 318([106]) }; // 288
                [38]() -> (); // 289
                [30]([105]) -> ([107]); // 290
                [29]([107]) -> ([108]); // 291
                [28]([108]) -> ([109]); // 292
                [60]([103]) -> ([94]); // 293
                [63]([109]) -> ([95]); // 294
                [26]([10], [95]) -> ([110]); // 295
                [63]([110]) -> ([110]); // 296
                [25]([64], [110]) -> ([111]); // 297
                [63]([111]) -> ([111]); // 298
                [67]([94], [111]) { fallthrough([112], [113]) 306([114]) }; // 299
                [38]() -> (); // 300
                [66]([9], [113]) -> ([115]); // 301
                [24]([115], [73], [11]) -> ([116]); // 302
                [60]([112]) -> ([117]); // 303
                [92]([116]) -> ([118]); // 304
                [42]() { 498() }; // 305
                [38]() -> (); // 306
                [80]([9]) -> (); // 307
                [74]([28]) -> (); // 308
                [74]([11]) -> (); // 309
                [73]([73]) -> (); // 310
                [17]() -> ([119]); // 311
                [83]() -> ([120]); // 312
                [63]([120]) -> ([120]); // 313
                [8]([119], [120]) -> ([121]); // 314
                [60]([114]) -> ([122]); // 315
                [91]([121]) -> ([123]); // 316
                [42]() { 374() }; // 317
                [38]() -> (); // 318
                [80]([9]) -> (); // 319
                [74]([28]) -> (); // 320
                [74]([11]) -> (); // 321
                [73]([73]) -> (); // 322
                [73]([64]) -> (); // 323
                [73]([10]) -> (); // 324
                [7]([106]) -> ([124], [125]); // 325
                [81]([124]) -> (); // 326
                [60]([103]) -> ([98]); // 327
                [91]([125]) -> ([99]); // 328
                [93]([98]) -> ([122]); // 329
                [94]([99]) -> ([123]); // 330
                [42]() { 374() }; // 331
                [38]() -> (); // 332
                [74]([70]) -> (); // 333
                [80]([9]) -> (); // 334
                [74]([28]) -> (); // 335
                [74]([11]) -> (); // 336
                [73]([62]) -> (); // 337
                [73]([10]) -> (); // 338
                [73]([64]) -> (); // 339
                [73]([63]) -> (); // 340
                [17]() -> ([126]); // 341
                [82]() -> ([127]); // 342
                [63]([127]) -> ([127]); // 343
                [8]([126], [127]) -> ([128]); // 344
                [60]([69]) -> ([122]); // 345
                [91]([128]) -> ([123]); // 346
                [42]() { 374() }; // 347
                [38]() -> (); // 348
                [80]([9]) -> (); // 349
                [74]([28]) -> (); // 350
                [74]([11]) -> (); // 351
                [84]([46]) -> (); // 352
                [73]([10]) -> (); // 353
                [84]([47]) -> (); // 354
                [7]([57]) -> ([129], [130]); // 355
                [81]([129]) -> (); // 356
                [60]([54]) -> ([122]); // 357
                [91]([130]) -> ([123]); // 358
                [42]() { 374() }; // 359
                [38]() -> (); // 360
                [74]([53]) -> (); // 361
                [80]([9]) -> (); // 362
                [74]([28]) -> (); // 363
                [74]([11]) -> (); // 364
                [84]([46]) -> (); // 365
                [73]([10]) -> (); // 366
                [84]([47]) -> (); // 367
                [17]() -> ([131]); // 368
                [82]() -> ([132]); // 369
                [63]([132]) -> ([132]); // 370
                [8]([131], [132]) -> ([133]); // 371
                [60]([52]) -> ([122]); // 372
                [91]([133]) -> ([123]); // 373
                [6]() -> ([134]); // 374
                [5]([134], [123]) -> ([135]); // 375
                [20]([135]) -> ([136]); // 376
                [60]([122]) -> ([122]); // 377
                [88]([136]) -> ([136]); // 378
                return([122], [136]); // 379
                [38]() -> (); // 380
                [74]([38]) -> (); // 381
                [69]([37], [2]) { fallthrough([137], [138]) 389([139], [140], [141]) }; // 382
                [38]() -> (); // 383
                [78]() -> ([142]); // 384
                [60]([137]) -> ([143]); // 385
                [90]([138]) -> ([144]); // 386
                [90]([142]) -> ([145]); // 387
                [42]() { 393() }; // 388
                [38]() -> (); // 389
                [60]([139]) -> ([143]); // 390
                [90]([141]) -> ([144]); // 391
                [90]([140]) -> ([145]); // 392
                [60]([143]) -> ([143]); // 393
                [72]([28]) -> ([28], [146]); // 394
                [89]([146]) -> ([146]); // 395
                [32]([143], [146]) -> ([147], [148]); // 396
                [31]([148]) { fallthrough([149]) 573([150]) }; // 397
                [38]() -> (); // 398
                [30]([149]) -> ([151]); // 399
                [70]([147], [144], [151]) -> ([152], [153], [154]); // 400
                [28]([145]) -> ([155]); // 401
                [77]() -> ([156]); // 402
                [72]([28]) -> ([28], [157]); // 403
                [89]([156]) -> ([156]); // 404
                [64]([152], [156], [157]) { fallthrough([158], [159]) 557([160], [161]) }; // 405
                [38]() -> (); // 406
                [60]([158]) -> ([158]); // 407
                [89]([159]) -> ([159]); // 408
                [32]([158], [159]) -> ([162], [163]); // 409
                [31]([163]) { fallthrough([164]) 544([165]) }; // 410
                [38]() -> (); // 411
                [30]([164]) -> ([166]); // 412
                [29]([166]) -> ([167]); // 413
                [28]([167]) -> ([168]); // 414
                [28]([153]) -> ([169]); // 415
                [28]([154]) -> ([170]); // 416
                [76]() -> ([171]); // 417
                [72]([11]) -> ([11], [172]); // 418
                [89]([171]) -> ([171]); // 419
                [64]([162], [171], [172]) { fallthrough([173], [174]) 527([175], [176]) }; // 420
                [38]() -> (); // 421
                [26]([155], [168]) -> ([177]); // 422
                [63]([177]) -> ([177]); // 423
                [25]([177], [169]) -> ([178]); // 424
                [77]() -> ([179]); // 425
                [72]([174]) -> ([174], [180]); // 426
                [89]([179]) -> ([179]); // 427
                [63]([178]) -> ([178]); // 428
                [64]([173], [180], [179]) { fallthrough([181], [182]) 476([183], [184]) }; // 429
                [38]() -> (); // 430
                [74]([182]) -> (); // 431
                [77]() -> ([185]); // 432
                [89]([185]) -> ([185]); // 433
                [64]([181], [174], [185]) { fallthrough([186], [187]) 461([188], [189]) }; // 434
                [38]() -> (); // 435
                [60]([186]) -> ([186]); // 436
                [89]([187]) -> ([187]); // 437
                [32]([186], [187]) -> ([190], [191]); // 438
                [31]([191]) { fallthrough([192]) 449([193]) }; // 439
                [38]() -> (); // 440
                [30]([192]) -> ([194]); // 441
                [29]([194]) -> ([195]); // 442
                [28]([195]) -> ([196]); // 443
                [79]() -> ([197]); // 444
                [26]([196], [197]) -> ([198]); // 445
                [60]([190]) -> ([199]); // 446
                [63]([198]) -> ([200]); // 447
                [42]() { 488() }; // 448
                [38]() -> (); // 449
                [80]([9]) -> (); // 450
                [74]([28]) -> (); // 451
                [74]([11]) -> (); // 452
                [73]([170]) -> (); // 453
                [73]([178]) -> (); // 454
                [73]([10]) -> (); // 455
                [7]([193]) -> ([201], [202]); // 456
                [81]([201]) -> (); // 457
                [60]([190]) -> ([203]); // 458
                [91]([202]) -> ([204]); // 459
                [42]() { 524() }; // 460
                [38]() -> (); // 461
                [74]([189]) -> (); // 462
                [80]([9]) -> (); // 463
                [74]([28]) -> (); // 464
                [74]([11]) -> (); // 465
                [73]([170]) -> (); // 466
                [73]([10]) -> (); // 467
                [73]([178]) -> (); // 468
                [17]() -> ([205]); // 469
                [82]() -> ([206]); // 470
                [63]([206]) -> ([206]); // 471
                [8]([205], [206]) -> ([207]); // 472
                [60]([188]) -> ([203]); // 473
                [91]([207]) -> ([204]); // 474
                [42]() { 524() }; // 475
                [38]() -> (); // 476
                [74]([184]) -> (); // 477
                [60]([183]) -> ([183]); // 478
                [89]([174]) -> ([174]); // 479
                [32]([183], [174]) -> ([208], [209]); // 480
                [31]([209]) { fallthrough([210]) 513([211]) }; // 481
                [38]() -> (); // 482
                [30]([210]) -> ([212]); // 483
                [29]([212]) -> ([213]); // 484
                [28]([213]) -> ([214]); // 485
                [60]([208]) -> ([199]); // 486
                [63]([214]) -> ([200]); // 487
                [26]([10], [200]) -> ([215]); // 488
                [63]([215]) -> ([215]); // 489
                [25]([178], [215]) -> ([216]); // 490
                [63]([216]) -> ([216]); // 491
                [67]([199], [216]) { fallthrough([217], [218]) 501([219]) }; // 492
                [38]() -> (); // 493
                [66]([9], [218]) -> ([220]); // 494
                [24]([220], [170], [11]) -> ([221]); // 495
                [60]([217]) -> ([117]); // 496
                [92]([221]) -> ([118]); // 497
                [93]([117]) -> ([222]); // 498
                [95]([118]) -> ([223]); // 499
                [42]() { 681() }; // 500
                [38]() -> (); // 501
                [80]([9]) -> (); // 502
                [74]([28]) -> (); // 503
                [74]([11]) -> (); // 504
                [73]([170]) -> (); // 505
                [17]() -> ([224]); // 506
                [83]() -> ([225]); // 507
                [63]([225]) -> ([225]); // 508
                [8]([224], [225]) -> ([226]); // 509
                [60]([219]) -> ([227]); // 510
                [91]([226]) -> ([228]); // 511
                [42]() { 584() }; // 512
                [38]() -> (); // 513
                [80]([9]) -> (); // 514
                [74]([28]) -> (); // 515
                [74]([11]) -> (); // 516
                [73]([170]) -> (); // 517
                [73]([178]) -> (); // 518
                [73]([10]) -> (); // 519
                [7]([211]) -> ([229], [230]); // 520
                [81]([229]) -> (); // 521
                [60]([208]) -> ([203]); // 522
                [91]([230]) -> ([204]); // 523
                [93]([203]) -> ([227]); // 524
                [94]([204]) -> ([228]); // 525
                [42]() { 584() }; // 526
                [38]() -> (); // 527
                [74]([176]) -> (); // 528
                [80]([9]) -> (); // 529
                [74]([28]) -> (); // 530
                [74]([11]) -> (); // 531
                [73]([170]) -> (); // 532
                [73]([10]) -> (); // 533
                [73]([155]) -> (); // 534
                [73]([168]) -> (); // 535
                [73]([169]) -> (); // 536
                [17]() -> ([231]); // 537
                [82]() -> ([232]); // 538
                [63]([232]) -> ([232]); // 539
                [8]([231], [232]) -> ([233]); // 540
                [60]([175]) -> ([227]); // 541
                [91]([233]) -> ([228]); // 542
                [42]() { 584() }; // 543
                [38]() -> (); // 544
                [80]([9]) -> (); // 545
                [74]([28]) -> (); // 546
                [74]([11]) -> (); // 547
                [73]([10]) -> (); // 548
                [73]([155]) -> (); // 549
                [84]([153]) -> (); // 550
                [84]([154]) -> (); // 551
                [7]([165]) -> ([234], [235]); // 552
                [81]([234]) -> (); // 553
                [60]([162]) -> ([227]); // 554
                [91]([235]) -> ([228]); // 555
                [42]() { 584() }; // 556
                [38]() -> (); // 557
                [74]([161]) -> (); // 558
                [80]([9]) -> (); // 559
                [74]([28]) -> (); // 560
                [74]([11]) -> (); // 561
                [84]([154]) -> (); // 562
                [73]([10]) -> (); // 563
                [73]([155]) -> (); // 564
                [84]([153]) -> (); // 565
                [17]() -> ([236]); // 566
                [82]() -> ([237]); // 567
                [63]([237]) -> ([237]); // 568
                [8]([236], [237]) -> ([238]); // 569
                [60]([160]) -> ([227]); // 570
                [91]([238]) -> ([228]); // 571
                [42]() { 584() }; // 572
                [38]() -> (); // 573
                [80]([9]) -> (); // 574
                [74]([28]) -> (); // 575
                [74]([11]) -> (); // 576
                [84]([145]) -> (); // 577
                [73]([10]) -> (); // 578
                [84]([144]) -> (); // 579
                [7]([150]) -> ([239], [240]); // 580
                [81]([239]) -> (); // 581
                [60]([147]) -> ([227]); // 582
                [91]([240]) -> ([228]); // 583
                [6]() -> ([241]); // 584
                [5]([241], [228]) -> ([242]); // 585
                [20]([242]) -> ([243]); // 586
                [60]([227]) -> ([227]); // 587
                [88]([243]) -> ([243]); // 588
                return([227], [243]); // 589
                [38]() -> (); // 590
                [69]([27], [2]) { fallthrough([244], [245]) 598([246], [247], [248]) }; // 591
                [38]() -> (); // 592
                [78]() -> ([249]); // 593
                [60]([244]) -> ([250]); // 594
                [90]([245]) -> ([251]); // 595
                [90]([249]) -> ([252]); // 596
                [42]() { 602() }; // 597
                [38]() -> (); // 598
                [60]([246]) -> ([250]); // 599
                [90]([248]) -> ([251]); // 600
                [90]([247]) -> ([252]); // 601
                [28]([252]) -> ([253]); // 602
                [28]([251]) -> ([254]); // 603
                [76]() -> ([255]); // 604
                [72]([11]) -> ([11], [256]); // 605
                [89]([255]) -> ([255]); // 606
                [64]([250], [255], [256]) { fallthrough([257], [258]) 716([259], [260]) }; // 607
                [38]() -> (); // 608
                [77]() -> ([261]); // 609
                [72]([258]) -> ([258], [262]); // 610
                [89]([261]) -> ([261]); // 611
                [64]([257], [262], [261]) { fallthrough([263], [264]) 659([265], [266]) }; // 612
                [38]() -> (); // 613
                [74]([264]) -> (); // 614
                [77]() -> ([267]); // 615
                [89]([267]) -> ([267]); // 616
                [64]([263], [258], [267]) { fallthrough([268], [269]) 644([270], [271]) }; // 617
                [38]() -> (); // 618
                [60]([268]) -> ([268]); // 619
                [89]([269]) -> ([269]); // 620
                [32]([268], [269]) -> ([272], [273]); // 621
                [31]([273]) { fallthrough([274]) 632([275]) }; // 622
                [38]() -> (); // 623
                [30]([274]) -> ([276]); // 624
                [29]([276]) -> ([277]); // 625
                [28]([277]) -> ([278]); // 626
                [79]() -> ([279]); // 627
                [26]([278], [279]) -> ([280]); // 628
                [60]([272]) -> ([281]); // 629
                [63]([280]) -> ([282]); // 630
                [42]() { 671() }; // 631
                [38]() -> (); // 632
                [80]([9]) -> (); // 633
                [74]([28]) -> (); // 634
                [74]([11]) -> (); // 635
                [73]([254]) -> (); // 636
                [73]([253]) -> (); // 637
                [73]([10]) -> (); // 638
                [7]([275]) -> ([283], [284]); // 639
                [81]([283]) -> (); // 640
                [60]([272]) -> ([285]); // 641
                [91]([284]) -> ([286]); // 642
                [42]() { 713() }; // 643
                [38]() -> (); // 644
                [74]([271]) -> (); // 645
                [80]([9]) -> (); // 646
                [74]([28]) -> (); // 647
                [74]([11]) -> (); // 648
                [73]([254]) -> (); // 649
                [73]([10]) -> (); // 650
                [73]([253]) -> (); // 651
                [17]() -> ([287]); // 652
                [82]() -> ([288]); // 653
                [63]([288]) -> ([288]); // 654
                [8]([287], [288]) -> ([289]); // 655
                [60]([270]) -> ([285]); // 656
                [91]([289]) -> ([286]); // 657
                [42]() { 713() }; // 658
                [38]() -> (); // 659
                [74]([266]) -> (); // 660
                [60]([265]) -> ([265]); // 661
                [89]([258]) -> ([258]); // 662
                [32]([265], [258]) -> ([290], [291]); // 663
                [31]([291]) { fallthrough([292]) 702([293]) }; // 664
                [38]() -> (); // 665
                [30]([292]) -> ([294]); // 666
                [29]([294]) -> ([295]); // 667
                [28]([295]) -> ([296]); // 668
                [60]([290]) -> ([281]); // 669
                [63]([296]) -> ([282]); // 670
                [26]([10], [282]) -> ([297]); // 671
                [63]([297]) -> ([297]); // 672
                [25]([253], [297]) -> ([298]); // 673
                [63]([298]) -> ([298]); // 674
                [67]([281], [298]) { fallthrough([299], [300]) 690([301]) }; // 675
                [38]() -> (); // 676
                [66]([9], [300]) -> ([302]); // 677
                [24]([302], [254], [11]) -> ([303]); // 678
                [60]([299]) -> ([222]); // 679
                [92]([303]) -> ([223]); // 680
                [68]([223]) -> ([304], [305], [306]); // 681
                [74]([306]) -> (); // 682
                [24]([304], [305], [28]) -> ([307]); // 683
                [23]() -> ([308]); // 684
                [22]([307], [308]) -> ([309]); // 685
                [21]([309]) -> ([310]); // 686
                [60]([222]) -> ([222]); // 687
                [88]([310]) -> ([310]); // 688
                return([222], [310]); // 689
                [38]() -> (); // 690
                [80]([9]) -> (); // 691
                [74]([28]) -> (); // 692
                [74]([11]) -> (); // 693
                [73]([254]) -> (); // 694
                [17]() -> ([311]); // 695
                [83]() -> ([312]); // 696
                [63]([312]) -> ([312]); // 697
                [8]([311], [312]) -> ([313]); // 698
                [60]([301]) -> ([314]); // 699
                [91]([313]) -> ([315]); // 700
                [42]() { 730() }; // 701
                [38]() -> (); // 702
                [80]([9]) -> (); // 703
                [74]([28]) -> (); // 704
                [74]([11]) -> (); // 705
                [73]([254]) -> (); // 706
                [73]([253]) -> (); // 707
                [73]([10]) -> (); // 708
                [7]([293]) -> ([316], [317]); // 709
                [81]([316]) -> (); // 710
                [60]([290]) -> ([285]); // 711
                [91]([317]) -> ([286]); // 712
                [93]([285]) -> ([314]); // 713
                [94]([286]) -> ([315]); // 714
                [42]() { 730() }; // 715
                [38]() -> (); // 716
                [74]([260]) -> (); // 717
                [80]([9]) -> (); // 718
                [74]([28]) -> (); // 719
                [74]([11]) -> (); // 720
                [73]([254]) -> (); // 721
                [73]([10]) -> (); // 722
                [73]([253]) -> (); // 723
                [17]() -> ([318]); // 724
                [82]() -> ([319]); // 725
                [63]([319]) -> ([319]); // 726
                [8]([318], [319]) -> ([320]); // 727
                [60]([259]) -> ([314]); // 728
                [91]([320]) -> ([315]); // 729
                [6]() -> ([321]); // 730
                [5]([321], [315]) -> ([322]); // 731
                [20]([322]) -> ([323]); // 732
                [60]([314]) -> ([314]); // 733
                [88]([323]) -> ([323]); // 734
                return([314], [323]); // 735
                [38]() -> (); // 736
                [74]([30]) -> (); // 737
                [80]([9]) -> (); // 738
                [73]([2]) -> (); // 739
                [74]([11]) -> (); // 740
                [73]([10]) -> (); // 741
                [17]() -> ([324]); // 742
                [82]() -> ([325]); // 743
                [63]([325]) -> ([325]); // 744
                [8]([324], [325]) -> ([326]); // 745
                [6]() -> ([327]); // 746
                [5]([327], [326]) -> ([328]); // 747
                [20]([328]) -> ([329]); // 748
                [60]([29]) -> ([29]); // 749
                [88]([329]) -> ([329]); // 750
                return([29], [329]); // 751
                [38]() -> (); // 752
                [74]([11]) -> (); // 753
                [74]([15]) -> (); // 754
                [77]() -> ([330]); // 755
                [72]([3]) -> ([3], [331]); // 756
                [89]([330]) -> ([330]); // 757
                [64]([20], [331], [330]) { fallthrough([332], [333]) 799([334], [335]) }; // 758
                [38]() -> (); // 759
                [74]([333]) -> (); // 760
                [77]() -> ([336]); // 761
                [89]([336]) -> ([336]); // 762
                [64]([332], [3], [336]) { fallthrough([337], [338]) 787([339], [340]) }; // 763
                [38]() -> (); // 764
                [60]([337]) -> ([337]); // 765
                [89]([338]) -> ([338]); // 766
                [32]([337], [338]) -> ([341], [342]); // 767
                [31]([342]) { fallthrough([343]) 778([344]) }; // 768
                [38]() -> (); // 769
                [30]([343]) -> ([345]); // 770
                [29]([345]) -> ([346]); // 771
                [28]([346]) -> ([347]); // 772
                [79]() -> ([348]); // 773
                [26]([347], [348]) -> ([349]); // 774
                [60]([341]) -> ([350]); // 775
                [63]([349]) -> ([351]); // 776
                [42]() { 811() }; // 777
                [38]() -> (); // 778
                [80]([9]) -> (); // 779
                [73]([2]) -> (); // 780
                [73]([10]) -> (); // 781
                [7]([344]) -> ([352], [353]); // 782
                [81]([352]) -> (); // 783
                [60]([341]) -> ([354]); // 784
                [91]([353]) -> ([355]); // 785
                [42]() { 847() }; // 786
                [38]() -> (); // 787
                [74]([340]) -> (); // 788
                [80]([9]) -> (); // 789
                [73]([10]) -> (); // 790
                [73]([2]) -> (); // 791
                [17]() -> ([356]); // 792
                [82]() -> ([357]); // 793
                [63]([357]) -> ([357]); // 794
                [8]([356], [357]) -> ([358]); // 795
                [60]([339]) -> ([354]); // 796
                [91]([358]) -> ([355]); // 797
                [42]() { 847() }; // 798
                [38]() -> (); // 799
                [74]([335]) -> (); // 800
                [60]([334]) -> ([334]); // 801
                [89]([3]) -> ([3]); // 802
                [32]([334], [3]) -> ([359], [360]); // 803
                [31]([360]) { fallthrough([361]) 839([362]) }; // 804
                [38]() -> (); // 805
                [30]([361]) -> ([363]); // 806
                [29]([363]) -> ([364]); // 807
                [28]([364]) -> ([365]); // 808
                [60]([359]) -> ([350]); // 809
                [63]([365]) -> ([351]); // 810
                [26]([10], [351]) -> ([366]); // 811
                [63]([366]) -> ([366]); // 812
                [25]([2], [366]) -> ([367]); // 813
                [63]([367]) -> ([367]); // 814
                [67]([350], [367]) { fallthrough([368], [369]) 827([370]) }; // 815
                [38]() -> (); // 816
                [66]([9], [369]) -> ([371]); // 817
                [85]() -> ([372]); // 818
                [86]() -> ([373]); // 819
                [24]([371], [372], [373]) -> ([374]); // 820
                [23]() -> ([375]); // 821
                [22]([374], [375]) -> ([376]); // 822
                [21]([376]) -> ([377]); // 823
                [60]([368]) -> ([368]); // 824
                [88]([377]) -> ([377]); // 825
                return([368], [377]); // 826
                [38]() -> (); // 827
                [80]([9]) -> (); // 828
                [17]() -> ([378]); // 829
                [83]() -> ([379]); // 830
                [63]([379]) -> ([379]); // 831
                [8]([378], [379]) -> ([380]); // 832
                [6]() -> ([381]); // 833
                [5]([381], [380]) -> ([382]); // 834
                [20]([382]) -> ([383]); // 835
                [60]([370]) -> ([370]); // 836
                [88]([383]) -> ([383]); // 837
                return([370], [383]); // 838
                [38]() -> (); // 839
                [80]([9]) -> (); // 840
                [73]([2]) -> (); // 841
                [73]([10]) -> (); // 842
                [7]([362]) -> ([384], [385]); // 843
                [81]([384]) -> (); // 844
                [60]([359]) -> ([354]); // 845
                [91]([385]) -> ([355]); // 846
                [6]() -> ([386]); // 847
                [5]([386], [355]) -> ([387]); // 848
                [20]([387]) -> ([388]); // 849
                [60]([354]) -> ([354]); // 850
                [88]([388]) -> ([388]); // 851
                return([354], [388]); // 852
                [38]() -> (); // 853
                [74]([23]) -> (); // 854
                [74]([15]) -> (); // 855
                [72]([11]) -> ([11], [389]); // 856
                [60]([22]) -> ([22]); // 857
                [65]([389]) { fallthrough() 869([390]) }; // 858
                [38]() -> (); // 859
                [73]([10]) -> (); // 860
                [74]([11]) -> (); // 861
                [24]([9], [2], [3]) -> ([391]); // 862
                [23]() -> ([392]); // 863
                [22]([391], [392]) -> ([393]); // 864
                [21]([393]) -> ([394]); // 865
                [60]([22]) -> ([22]); // 866
                [88]([394]) -> ([394]); // 867
                return([22], [394]); // 868
                [38]() -> (); // 869
                [75]([390]) -> (); // 870
                [77]() -> ([395]); // 871
                [72]([3]) -> ([3], [396]); // 872
                [89]([395]) -> ([395]); // 873
                [64]([22], [396], [395]) { fallthrough([397], [398]) 920([399], [400]) }; // 874
                [38]() -> (); // 875
                [74]([398]) -> (); // 876
                [77]() -> ([401]); // 877
                [72]([3]) -> ([3], [402]); // 878
                [89]([401]) -> ([401]); // 879
                [64]([397], [402], [401]) { fallthrough([403], [404]) 906([405], [406]) }; // 880
                [38]() -> (); // 881
                [60]([403]) -> ([403]); // 882
                [89]([404]) -> ([404]); // 883
                [32]([403], [404]) -> ([407], [408]); // 884
                [31]([408]) { fallthrough([409]) 895([410]) }; // 885
                [38]() -> (); // 886
                [30]([409]) -> ([411]); // 887
                [29]([411]) -> ([412]); // 888
                [28]([412]) -> ([413]); // 889
                [79]() -> ([414]); // 890
                [26]([413], [414]) -> ([415]); // 891
                [60]([407]) -> ([416]); // 892
                [63]([415]) -> ([417]); // 893
                [42]() { 933() }; // 894
                [38]() -> (); // 895
                [73]([10]) -> (); // 896
                [80]([9]) -> (); // 897
                [73]([2]) -> (); // 898
                [74]([3]) -> (); // 899
                [74]([11]) -> (); // 900
                [7]([410]) -> ([418], [419]); // 901
                [81]([418]) -> (); // 902
                [60]([407]) -> ([420]); // 903
                [91]([419]) -> ([421]); // 904
                [42]() { 971() }; // 905
                [38]() -> (); // 906
                [74]([406]) -> (); // 907
                [73]([10]) -> (); // 908
                [74]([11]) -> (); // 909
                [80]([9]) -> (); // 910
                [73]([2]) -> (); // 911
                [74]([3]) -> (); // 912
                [17]() -> ([422]); // 913
                [82]() -> ([423]); // 914
                [63]([423]) -> ([423]); // 915
                [8]([422], [423]) -> ([424]); // 916
                [60]([405]) -> ([420]); // 917
                [91]([424]) -> ([421]); // 918
                [42]() { 971() }; // 919
                [38]() -> (); // 920
                [74]([400]) -> (); // 921
                [60]([399]) -> ([399]); // 922
                [72]([3]) -> ([3], [425]); // 923
                [89]([425]) -> ([425]); // 924
                [32]([399], [425]) -> ([426], [427]); // 925
                [31]([427]) { fallthrough([428]) 961([429]) }; // 926
                [38]() -> (); // 927
                [30]([428]) -> ([430]); // 928
                [29]([430]) -> ([431]); // 929
                [28]([431]) -> ([432]); // 930
                [60]([426]) -> ([416]); // 931
                [63]([432]) -> ([417]); // 932
                [27]([416], [11], [3]) { fallthrough([433], [434]) 945([435], [436]) }; // 933
                [38]() -> (); // 934
                [26]([10], [417]) -> ([437]); // 935
                [63]([437]) -> ([437]); // 936
                [25]([2], [437]) -> ([438]); // 937
                [24]([9], [438], [434]) -> ([439]); // 938
                [23]() -> ([440]); // 939
                [22]([439], [440]) -> ([441]); // 940
                [21]([441]) -> ([442]); // 941
                [60]([433]) -> ([433]); // 942
                [88]([442]) -> ([442]); // 943
                return([433], [442]); // 944
                [38]() -> (); // 945
                [74]([436]) -> (); // 946
                [73]([10]) -> (); // 947
                [73]([417]) -> (); // 948
                [80]([9]) -> (); // 949
                [73]([2]) -> (); // 950
                [17]() -> ([443]); // 951
                [87]() -> ([444]); // 952
                [63]([444]) -> ([444]); // 953
                [8]([443], [444]) -> ([445]); // 954
                [6]() -> ([446]); // 955
                [5]([446], [445]) -> ([447]); // 956
                [20]([447]) -> ([448]); // 957
                [60]([435]) -> ([435]); // 958
                [88]([448]) -> ([448]); // 959
                return([435], [448]); // 960
                [38]() -> (); // 961
                [73]([10]) -> (); // 962
                [80]([9]) -> (); // 963
                [73]([2]) -> (); // 964
                [74]([3]) -> (); // 965
                [74]([11]) -> (); // 966
                [7]([429]) -> ([449], [450]); // 967
                [81]([449]) -> (); // 968
                [60]([426]) -> ([420]); // 969
                [91]([450]) -> ([421]); // 970
                [6]() -> ([451]); // 971
                [5]([451], [421]) -> ([452]); // 972
                [20]([452]) -> ([453]); // 973
                [60]([420]) -> ([420]); // 974
                [88]([453]) -> ([453]); // 975
                return([420], [453]); // 976
                [38]() -> (); // 977
                [74]([17]) -> (); // 978
                [80]([9]) -> (); // 979
                [73]([2]) -> (); // 980
                [74]([11]) -> (); // 981
                [73]([10]) -> (); // 982
                [74]([3]) -> (); // 983
                [17]() -> ([454]); // 984
                [87]() -> ([455]); // 985
                [63]([455]) -> ([455]); // 986
                [8]([454], [455]) -> ([456]); // 987
                [6]() -> ([457]); // 988
                [5]([457], [456]) -> ([458]); // 989
                [20]([458]) -> ([459]); // 990
                [60]([16]) -> ([16]); // 991
                [88]([459]) -> ([459]); // 992
                return([16], [459]); // 993
                [104]() -> (); // 994
                [140]([0], [1]) { fallthrough([4], [5]) 1026([6], [7]) }; // 995
                [38]() -> (); // 996
                [3]([5]) -> ([8]); // 997
                [139]([2]) -> ([9]); // 998
                [60]([4]) -> ([4]); // 999
                [128]([8]) -> ([8]); // 1000
                [138]([9]) { fallthrough([10], [11]) 1016([12]) }; // 1001
                [38]() -> (); // 1002
                [3]([8]) -> ([13]); // 1003
                [137]([11]) -> ([14]); // 1004
                [141]([14]) -> ([15]); // 1005
                [136]([15]) -> ([16]); // 1006
                [63]([16]) -> ([16]); // 1007
                [8]([3], [16]) -> ([17]); // 1008
                [14]([10]) -> ([18]); // 1009
                [60]([4]) -> ([4]); // 1010
                [128]([13]) -> ([13]); // 1011
                [131]([18]) -> ([18]); // 1012
                [91]([17]) -> ([17]); // 1013
                [13]([4], [13], [18], [17]) -> ([19], [20], [21]); // 1014
                return([19], [20], [21]); // 1015
                [38]() -> (); // 1016
                [122]([12]) -> (); // 1017
                [3]([8]) -> ([22]); // 1018
                [23]() -> ([23]); // 1019
                [135]([3], [23]) -> ([24]); // 1020
                [134]([24]) -> ([25]); // 1021
                [60]([4]) -> ([4]); // 1022
                [128]([22]) -> ([22]); // 1023
                [144]([25]) -> ([25]); // 1024
                return([4], [22], [25]); // 1025
                [38]() -> (); // 1026
                [142]([2]) -> (); // 1027
                [108]([3]) -> (); // 1028
                [3]([7]) -> ([26]); // 1029
                [17]() -> ([27]); // 1030
                [143]() -> ([28]); // 1031
                [63]([28]) -> ([28]); // 1032
                [8]([27], [28]) -> ([29]); // 1033
                [6]() -> ([30]); // 1034
                [5]([30], [29]) -> ([31]); // 1035
                [133]([31]) -> ([32]); // 1036
                [60]([6]) -> ([6]); // 1037
                [128]([26]) -> ([26]); // 1038
                [144]([32]) -> ([32]); // 1039
                return([6], [26], [32]); // 1040
                [37]([0], [1]) { fallthrough([2], [3]) 1131([4]) }; // 1041
                [38]() -> (); // 1042
                [39]([3]) -> ([5]); // 1043
                [59]([5]) -> ([5]); // 1044
                [60]([2]) -> ([2]); // 1045
                [36]([5]) { fallthrough([6]) 1052([7]) 1057([8]) 1062([9]) 1067([10]) 1072([11]) 1077([12]) 1082([13]) 1087([14]) 1092([15]) 1097([16]) 1102([17]) 1107([18]) 1112([19]) 1117([20]) 1122([21]) }; // 1046
                [38]() -> (); // 1047
                [40]([6]) -> (); // 1048
                [41]() -> ([22]); // 1049
                [61]([22]) -> ([23]); // 1050
                [42]() { 1126() }; // 1051
                [38]() -> (); // 1052
                [40]([7]) -> (); // 1053
                [43]() -> ([24]); // 1054
                [61]([24]) -> ([23]); // 1055
                [42]() { 1126() }; // 1056
                [38]() -> (); // 1057
                [40]([8]) -> (); // 1058
                [44]() -> ([25]); // 1059
                [61]([25]) -> ([23]); // 1060
                [42]() { 1126() }; // 1061
                [38]() -> (); // 1062
                [40]([9]) -> (); // 1063
                [45]() -> ([26]); // 1064
                [61]([26]) -> ([23]); // 1065
                [42]() { 1126() }; // 1066
                [38]() -> (); // 1067
                [40]([10]) -> (); // 1068
                [46]() -> ([27]); // 1069
                [61]([27]) -> ([23]); // 1070
                [42]() { 1126() }; // 1071
                [38]() -> (); // 1072
                [40]([11]) -> (); // 1073
                [47]() -> ([28]); // 1074
                [61]([28]) -> ([23]); // 1075
                [42]() { 1126() }; // 1076
                [38]() -> (); // 1077
                [40]([12]) -> (); // 1078
                [48]() -> ([29]); // 1079
                [61]([29]) -> ([23]); // 1080
                [42]() { 1126() }; // 1081
                [38]() -> (); // 1082
                [40]([13]) -> (); // 1083
                [49]() -> ([30]); // 1084
                [61]([30]) -> ([23]); // 1085
                [42]() { 1126() }; // 1086
                [38]() -> (); // 1087
                [40]([14]) -> (); // 1088
                [50]() -> ([31]); // 1089
                [61]([31]) -> ([23]); // 1090
                [42]() { 1126() }; // 1091
                [38]() -> (); // 1092
                [40]([15]) -> (); // 1093
                [51]() -> ([32]); // 1094
                [61]([32]) -> ([23]); // 1095
                [42]() { 1126() }; // 1096
                [38]() -> (); // 1097
                [40]([16]) -> (); // 1098
                [52]() -> ([33]); // 1099
                [61]([33]) -> ([23]); // 1100
                [42]() { 1126() }; // 1101
                [38]() -> (); // 1102
                [40]([17]) -> (); // 1103
                [53]() -> ([34]); // 1104
                [61]([34]) -> ([23]); // 1105
                [42]() { 1126() }; // 1106
                [38]() -> (); // 1107
                [40]([18]) -> (); // 1108
                [54]() -> ([35]); // 1109
                [61]([35]) -> ([23]); // 1110
                [42]() { 1126() }; // 1111
                [38]() -> (); // 1112
                [40]([19]) -> (); // 1113
                [55]() -> ([36]); // 1114
                [61]([36]) -> ([23]); // 1115
                [42]() { 1126() }; // 1116
                [38]() -> (); // 1117
                [40]([20]) -> (); // 1118
                [56]() -> ([37]); // 1119
                [61]([37]) -> ([23]); // 1120
                [42]() { 1126() }; // 1121
                [38]() -> (); // 1122
                [40]([21]) -> (); // 1123
                [57]() -> ([38]); // 1124
                [61]([38]) -> ([23]); // 1125
                [35]([23]) -> ([39]); // 1126
                [34]([39]) -> ([40]); // 1127
                [60]([2]) -> ([2]); // 1128
                [62]([40]) -> ([40]); // 1129
                return([2], [40]); // 1130
                [38]() -> (); // 1131
                [17]() -> ([41]); // 1132
                [58]() -> ([42]); // 1133
                [63]([42]) -> ([42]); // 1134
                [8]([41], [42]) -> ([43]); // 1135
                [6]() -> ([44]); // 1136
                [5]([44], [43]) -> ([45]); // 1137
                [33]([45]) -> ([46]); // 1138
                [60]([4]) -> ([4]); // 1139
                [62]([46]) -> ([46]); // 1140
                return([4], [46]); // 1141

                [3]@0([0]: [0], [1]: [1]) -> ([0], [1], [24]);
                [1]@152([0]: [0], [1]: [12], [2]: [2], [3]: [11]) -> ([0], [16]);
                [0]@994([0]: [0], [1]: [1], [2]: [19], [3]: [3]) -> ([0], [1], [21]);
                [2]@1041([0]: [0], [1]: [11]) -> ([0], [29]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let result = run_sierra_program(&program, &[]).return_value;

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
        // use array::ArrayTrait;
        // fn run_test() -> (Span<felt252>, @Box<[felt252; 2]>, @Box<[felt252; 2]>) {
        //     let mut numbers = array![1, 2, 3, 4, 5, 6].span();
        //     let popped_front = numbers.multi_pop_front::<2>().unwrap();
        //     let popped_back = numbers.multi_pop_back::<2>().unwrap();
        //     (numbers, popped_front, popped_back)
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [2] = Array<[1]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [9] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
                type [10] = Struct<ut@Tuple, [9], [2]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [18] = Const<[1], 29721761890975875353235833581453094220424382983267374> [storable: false, drop: false, dup: false, zero_sized: false];
                type [3] = Snapshot<[2]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [6] = Struct<ut@core::array::Span::<core::felt252>, [3]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [5] = Box<[4]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [7] = Struct<ut@Tuple, [6], [5], [5]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [8] = Struct<ut@Tuple, [7]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [11] = Enum<ut@core::panics::PanicResult::<((core::array::Span::<core::felt252>, @core::box::Box::<[core::felt252; 2]>, @core::box::Box::<[core::felt252; 2]>),)>, [8], [10]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [1] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
                type [4] = Struct<ut@Tuple, [1], [1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
                type [17] = Const<[1], 6> [storable: false, drop: false, dup: false, zero_sized: false];
                type [16] = Const<[1], 5> [storable: false, drop: false, dup: false, zero_sized: false];
                type [15] = Const<[1], 4> [storable: false, drop: false, dup: false, zero_sized: false];
                type [14] = Const<[1], 3> [storable: false, drop: false, dup: false, zero_sized: false];
                type [13] = Const<[1], 2> [storable: false, drop: false, dup: false, zero_sized: false];
                type [12] = Const<[1], 1> [storable: false, drop: false, dup: false, zero_sized: false];

                libfunc [4] = array_new<[1]>;
                libfunc [12] = const_as_immediate<[12]>;
                libfunc [24] = store_temp<[1]>;
                libfunc [3] = array_append<[1]>;
                libfunc [13] = const_as_immediate<[13]>;
                libfunc [14] = const_as_immediate<[14]>;
                libfunc [15] = const_as_immediate<[15]>;
                libfunc [16] = const_as_immediate<[16]>;
                libfunc [17] = const_as_immediate<[17]>;
                libfunc [18] = snapshot_take<[2]>;
                libfunc [19] = drop<[2]>;
                libfunc [25] = store_temp<[3]>;
                libfunc [10] = array_snapshot_multi_pop_front<[4]>;
                libfunc [20] = branch_align;
                libfunc [9] = array_snapshot_multi_pop_back<[4]>;
                libfunc [8] = struct_construct<[6]>;
                libfunc [7] = struct_construct<[7]>;
                libfunc [6] = struct_construct<[8]>;
                libfunc [5] = enum_init<[11], 0>;
                libfunc [26] = store_temp<[0]>;
                libfunc [27] = store_temp<[11]>;
                libfunc [21] = drop<[3]>;
                libfunc [22] = drop<[5]>;
                libfunc [23] = const_as_immediate<[18]>;
                libfunc [2] = struct_construct<[9]>;
                libfunc [1] = struct_construct<[10]>;
                libfunc [0] = enum_init<[11], 1>;

                [4]() -> ([1]); // 0
                [12]() -> ([2]); // 1
                [24]([2]) -> ([2]); // 2
                [3]([1], [2]) -> ([3]); // 3
                [13]() -> ([4]); // 4
                [24]([4]) -> ([4]); // 5
                [3]([3], [4]) -> ([5]); // 6
                [14]() -> ([6]); // 7
                [24]([6]) -> ([6]); // 8
                [3]([5], [6]) -> ([7]); // 9
                [15]() -> ([8]); // 10
                [24]([8]) -> ([8]); // 11
                [3]([7], [8]) -> ([9]); // 12
                [16]() -> ([10]); // 13
                [24]([10]) -> ([10]); // 14
                [3]([9], [10]) -> ([11]); // 15
                [17]() -> ([12]); // 16
                [24]([12]) -> ([12]); // 17
                [3]([11], [12]) -> ([13]); // 18
                [18]([13]) -> ([14], [15]); // 19
                [19]([14]) -> (); // 20
                [25]([15]) -> ([15]); // 21
                [10]([0], [15]) { fallthrough([16], [17], [18]) 46([19], [20]) }; // 22
                [20]() -> (); // 23
                [9]([16], [17]) { fallthrough([21], [22], [23]) 33([24], [25]) }; // 24
                [20]() -> (); // 25
                [8]([22]) -> ([26]); // 26
                [7]([26], [18], [23]) -> ([27]); // 27
                [6]([27]) -> ([28]); // 28
                [5]([28]) -> ([29]); // 29
                [26]([21]) -> ([21]); // 30
                [27]([29]) -> ([29]); // 31
                return([21], [29]); // 32
                [20]() -> (); // 33
                [21]([25]) -> (); // 34
                [22]([18]) -> (); // 35
                [4]() -> ([30]); // 36
                [23]() -> ([31]); // 37
                [24]([31]) -> ([31]); // 38
                [3]([30], [31]) -> ([32]); // 39
                [2]() -> ([33]); // 40
                [1]([33], [32]) -> ([34]); // 41
                [0]([34]) -> ([35]); // 42
                [26]([24]) -> ([24]); // 43
                [27]([35]) -> ([35]); // 44
                return([24], [35]); // 45
                [20]() -> (); // 46
                [21]([20]) -> (); // 47
                [4]() -> ([36]); // 48
                [23]() -> ([37]); // 49
                [24]([37]) -> ([37]); // 50
                [3]([36], [37]) -> ([38]); // 51
                [2]() -> ([39]); // 52
                [1]([39], [38]) -> ([40]); // 53
                [0]([40]) -> ([41]); // 54
                [26]([19]) -> ([19]); // 55
                [27]([41]) -> ([41]); // 56
                return([19], [41]); // 57

                [0]@0([0]: [0]) -> ([0], [11]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let result = run_sierra_program(&program, &[]).return_value;

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
        // use core::{array::{array_append, array_at, array_new}, box::{into_box, unbox}};
        // fn run_test() -> @Box<felt252> {
        //     let mut x: Array<Box<felt252>> = array_new();
        //     array_append(ref x, into_box(42));
        //     unbox(array_at(@x, 0))
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [2] = Array<[1]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [6] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
                type [7] = Array<[0]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [8] = Struct<ut@Tuple, [6], [7]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [11] = Const<[0], 1637570914057682275393755530660268060279989363> [storable: false, drop: false, dup: false, zero_sized: false];
                type [0] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
                type [1] = Box<[0]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [5] = Struct<ut@Tuple, [1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [9] = Enum<ut@core::panics::PanicResult::<(@core::box::Box::<core::felt252>,)>, [5], [8]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [4] = Box<[1]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [3] = Snapshot<[2]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [10] = Const<[0], 42> [storable: false, drop: false, dup: false, zero_sized: false];

                libfunc [10] = array_new<[1]>;
                libfunc [12] = const_as_box<[10], 0>;
                libfunc [9] = array_append<[1]>;
                libfunc [13] = snapshot_take<[2]>;
                libfunc [14] = drop<[2]>;
                libfunc [18] = store_temp<[3]>;
                libfunc [8] = array_snapshot_pop_front<[1]>;
                libfunc [15] = branch_align;
                libfunc [16] = drop<[3]>;
                libfunc [7] = unbox<[1]>;
                libfunc [6] = struct_construct<[5]>;
                libfunc [5] = enum_init<[9], 0>;
                libfunc [19] = store_temp<[9]>;
                libfunc [4] = array_new<[0]>;
                libfunc [17] = const_as_immediate<[11]>;
                libfunc [20] = store_temp<[0]>;
                libfunc [3] = array_append<[0]>;
                libfunc [2] = struct_construct<[6]>;
                libfunc [1] = struct_construct<[8]>;
                libfunc [0] = enum_init<[9], 1>;

                [10]() -> ([0]); // 0
                [12]() -> ([1]); // 1
                [9]([0], [1]) -> ([2]); // 2
                [13]([2]) -> ([3], [4]); // 3
                [14]([3]) -> (); // 4
                [18]([4]) -> ([4]); // 5
                [8]([4]) { fallthrough([5], [6]) 14([7]) }; // 6
                [15]() -> (); // 7
                [16]([5]) -> (); // 8
                [7]([6]) -> ([8]); // 9
                [6]([8]) -> ([9]); // 10
                [5]([9]) -> ([10]); // 11
                [19]([10]) -> ([10]); // 12
                return([10]); // 13
                [15]() -> (); // 14
                [16]([7]) -> (); // 15
                [4]() -> ([11]); // 16
                [17]() -> ([12]); // 17
                [20]([12]) -> ([12]); // 18
                [3]([11], [12]) -> ([13]); // 19
                [2]() -> ([14]); // 20
                [1]([14], [13]) -> ([15]); // 21
                [0]([15]) -> ([16]); // 22
                [19]([16]) -> ([16]); // 23
                return([16]); // 24

                [0]@0() -> ([9]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let result = run_sierra_program(&program, &[]).return_value;

        assert_eq!(result, jit_enum!(0, jit_struct!(Value::Felt252(42.into()))));
    }
}
