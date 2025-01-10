//! # Array type
//!
//! An array type is a dynamically allocated list of items.
//!
//! ## Layout
//!
//! Being dynamically allocated, we just need to keep the pointer to the data, its length and
//! its capacity:
//!
//! | Index | Type           | Description              |
//! | ----- | -------------- | ------------------------ |
//! |   0   | `!llvm.ptr<T>` | Pointer to the data[^1]. |
//! |   1   | `i32`          | Array start offset[^2].  |
//! |   1   | `i32`          | Array end offset[^2].    |
//! |   2   | `i32`          | Allocated capacity[^2].  |
//!
//! The pointer to the allocation (which is **not the data**) contains:
//!   1. Reference counter.
//!   2. Padding.
//!   3. Array data. Its address is the pointer to the data stored in the type.
//!
//! [^1]: When capacity is zero, this field is not guaranteed to be valid.
//! [^2]: Those numbers are number of items, **not bytes**.

use super::{TypeBuilder, WithSelf};
use crate::{
    error::Result,
    metadata::{
        drop_overrides::DropOverridesMeta, dup_overrides::DupOverridesMeta,
        realloc_bindings::ReallocBindingsMeta, MetadataStorage,
    },
    utils::{get_integer_layout, BlockExt, GepIndex, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        types::InfoAndTypeConcreteType,
    },
    program_registry::ProgramRegistry,
};
use melior::dialect::{arith::CmpiPredicate, func, scf};
use melior::ir::Region;
use melior::{
    dialect::{arith, llvm},
    ir::{r#type::IntegerType, Block, Location, Module, Type},
    Context,
};

/// Build the MLIR type.
///
/// Check out [the module](self) for more info.
pub fn build<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: WithSelf<InfoAndTypeConcreteType>,
) -> Result<Type<'ctx>> {
    DupOverridesMeta::register_with(
        context,
        module,
        registry,
        metadata,
        info.self_ty(),
        |metadata| {
            // There's no need to build the type here because it'll always be built within
            // `build_dup`.

            Ok(Some(build_dup(context, module, registry, metadata, &info)?))
        },
    )?;
    DropOverridesMeta::register_with(
        context,
        module,
        registry,
        metadata,
        info.self_ty(),
        |metadata| {
            // There's no need to build the type here because it'll always be built within
            // `build_drop`.

            Ok(Some(build_drop(
                context, module, registry, metadata, &info,
            )?))
        },
    )?;

    let ptr_ty = llvm::r#type::pointer(context, 0);
    let len_ty = IntegerType::new(context, 32).into();

    Ok(llvm::r#type::r#struct(
        context,
        &[ptr_ty, len_ty, len_ty, len_ty],
        false,
    ))
}

/// This function clones the array shallowly. That is, it'll increment the reference counter but not
/// actually clone anything. The deep clone implementation is provided in `src/libfuncs/array.rs` as
/// part of some libfuncs's implementations.
fn build_dup<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: &WithSelf<InfoAndTypeConcreteType>,
) -> Result<Region<'ctx>> {
    let location = Location::unknown(context);
    let value_ty = registry.build_type(context, module, metadata, info.self_ty())?;

    let elem_layout = registry.get_type(&info.ty)?.layout(registry)?;
    let refcount_offset = get_integer_layout(32)
        .align_to(elem_layout.align())
        .unwrap()
        .pad_to_align()
        .size();

    let region = Region::new();
    let entry = region.append_block(Block::new(&[(value_ty, location)]));

    let array_cap = entry.extract_value(
        context,
        location,
        entry.argument(0)?.into(),
        IntegerType::new(context, 32).into(),
        3,
    )?;
    let k0 = entry.const_int(context, location, 0, 32)?;
    let is_empty = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Eq,
        array_cap,
        k0,
        location,
    ))?;

    entry.append_operation(scf::r#if(
        is_empty,
        &[],
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            block.append_operation(scf::r#yield(&[], location));
            region
        },
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let array_ptr = block.extract_value(
                context,
                location,
                entry.argument(0)?.into(),
                llvm::r#type::pointer(context, 0),
                0,
            )?;

            let refcount_ptr = block.gep(
                context,
                location,
                array_ptr,
                &[GepIndex::Const(-(refcount_offset as i32))],
                IntegerType::new(context, 8).into(),
            )?;
            let ref_count = block.load(
                context,
                location,
                refcount_ptr,
                IntegerType::new(context, 32).into(),
            )?;

            let k1 = block.const_int(context, location, 1, 32)?;
            let ref_count = block.append_op_result(arith::addi(ref_count, k1, location))?;
            block.store(context, location, refcount_ptr, ref_count)?;

            block.append_operation(scf::r#yield(&[], location));
            region
        },
        location,
    ));

    entry.append_operation(func::r#return(
        &[entry.argument(0)?.into(), entry.argument(0)?.into()],
        location,
    ));
    Ok(region)
}

/// This function decreases the reference counter of the array by one.
/// If the reference counter reaches zero, then all the resources are freed.
fn build_drop<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    metadata: &mut MetadataStorage,
    info: &WithSelf<InfoAndTypeConcreteType>,
) -> Result<Region<'ctx>> {
    let location = Location::unknown(context);
    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, module));
    }

    let value_ty = registry.build_type(context, module, metadata, info.self_ty())?;

    let elem_ty = registry.get_type(&info.ty)?;
    let elem_stride = elem_ty.layout(registry)?.pad_to_align().size();
    let elem_ty = elem_ty.build(context, module, registry, metadata, &info.ty)?;
    let elem_layout = registry.get_type(&info.ty)?.layout(registry)?;
    let refcount_offset = get_integer_layout(32)
        .align_to(elem_layout.align())
        .unwrap()
        .pad_to_align()
        .size();

    let region = Region::new();
    let entry = region.append_block(Block::new(&[(value_ty, location)]));

    let array_ptr = entry.extract_value(
        context,
        location,
        entry.argument(0)?.into(),
        llvm::r#type::pointer(context, 0),
        0,
    )?;

    let array_cap = entry.extract_value(
        context,
        location,
        entry.argument(0)?.into(),
        IntegerType::new(context, 32).into(),
        3,
    )?;
    let k0 = entry.const_int(context, location, 0, 32)?;
    let zero_capacity = entry.append_op_result(arith::cmpi(
        context,
        CmpiPredicate::Eq,
        array_cap,
        k0,
        location,
    ))?;

    entry.append_operation(scf::r#if(
        zero_capacity,
        &[],
        {
            // if the array has no capacity, do nothing, as there is no allocation

            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            block.append_operation(scf::r#yield(&[], location));
            region
        },
        {
            // if the array has capacity, decrease the reference counter
            // and, in case it reaches zero, free all the resources.

            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            // obtain the reference counter
            let refcount_ptr = block.gep(
                context,
                location,
                array_ptr,
                &[GepIndex::Const(-(refcount_offset as i32))],
                IntegerType::new(context, 8).into(),
            )?;
            let ref_count = block.load(
                context,
                location,
                refcount_ptr,
                IntegerType::new(context, 32).into(),
            )?;

            // if the reference counter is greater than 1, then it's shared
            let k1 = block.const_int(context, location, 1, 32)?;
            let is_shared = block.append_op_result(arith::cmpi(
                context,
                CmpiPredicate::Ne,
                ref_count,
                k1,
                location,
            ))?;

            block.append_operation(scf::r#if(
                is_shared,
                &[],
                {
                    // if the array is shared, decrease the reference counter by one
                    let region = Region::new();
                    let block = region.append_block(Block::new(&[]));

                    let ref_count = block.append_op_result(arith::subi(ref_count, k1, location))?;
                    block.store(context, location, refcount_ptr, ref_count)?;

                    block.append_operation(scf::r#yield(&[], location));
                    region
                },
                {
                    // if the array is not shared, drop all elements and free the memory
                    let region = Region::new();
                    let block = region.append_block(Block::new(&[]));

                    match metadata.get::<DropOverridesMeta>() {
                        Some(drop_overrides_meta) if drop_overrides_meta.is_overriden(&info.ty) => {
                            let value_start = block.extract_value(
                                context,
                                location,
                                entry.argument(0)?.into(),
                                IntegerType::new(context, 32).into(),
                                1,
                            )?;
                            let value_end = block.extract_value(
                                context,
                                location,
                                entry.argument(0)?.into(),
                                IntegerType::new(context, 32).into(),
                                2,
                            )?;

                            let value_start = block.append_op_result(arith::extui(
                                value_start,
                                IntegerType::new(context, 64).into(),
                                location,
                            ))?;
                            let value_end = block.append_op_result(arith::extui(
                                value_end,
                                IntegerType::new(context, 64).into(),
                                location,
                            ))?;

                            let elem_stride =
                                block.const_int(context, location, elem_stride, 64)?;
                            let offset_start = block.append_op_result(arith::muli(
                                value_start,
                                elem_stride,
                                location,
                            ))?;
                            let offset_end = block.append_op_result(arith::muli(
                                value_end,
                                elem_stride,
                                location,
                            ))?;

                            // for each element in the aray, invoke its drop implementation
                            block.append_operation(scf::r#for(
                                offset_start,
                                offset_end,
                                elem_stride,
                                {
                                    let region = Region::new();
                                    let block = region.append_block(Block::new(&[(
                                        IntegerType::new(context, 64).into(),
                                        location,
                                    )]));

                                    let elem_offset = block.argument(0)?.into();
                                    let elem_ptr = block.gep(
                                        context,
                                        location,
                                        array_ptr,
                                        &[GepIndex::Value(elem_offset)],
                                        IntegerType::new(context, 8).into(),
                                    )?;
                                    let elem_val =
                                        block.load(context, location, elem_ptr, elem_ty)?;

                                    drop_overrides_meta.invoke_override(
                                        context, &block, location, &info.ty, elem_val,
                                    )?;

                                    block.append_operation(scf::r#yield(&[], location));
                                    region
                                },
                                location,
                            ));
                        }
                        _ => {}
                    }

                    // finally, free the array allocation
                    block.append_operation(ReallocBindingsMeta::free(
                        context,
                        refcount_ptr,
                        location,
                    )?);
                    block.append_operation(scf::r#yield(&[], location));
                    region
                },
                location,
            ));

            block.append_operation(scf::r#yield(&[], location));
            region
        },
        location,
    ));

    entry.append_operation(func::r#return(&[], location));
    Ok(region)
}

#[cfg(test)]
mod test {
    use crate::{utils::test::run_sierra_program, values::Value};
    use cairo_lang_sierra::ProgramParser;
    use pretty_assertions_sorted::assert_eq;

    #[test]
    fn test_array_snapshot_deep_clone() {
        // fn run_test() -> @Array<Array<felt252>> {
        //     let mut inputs: Array<Array<felt252>> = ArrayTrait::new();
        //     inputs.append(array![1, 2, 3]);
        //     inputs.append(array![4, 5, 6]);

        //     @inputs
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
            type [2] = Array<[1]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [3] = Snapshot<[2]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [9] = Const<[0], 6> [storable: false, drop: false, dup: false, zero_sized: false];
            type [8] = Const<[0], 5> [storable: false, drop: false, dup: false, zero_sized: false];
            type [7] = Const<[0], 4> [storable: false, drop: false, dup: false, zero_sized: false];
            type [6] = Const<[0], 3> [storable: false, drop: false, dup: false, zero_sized: false];
            type [5] = Const<[0], 2> [storable: false, drop: false, dup: false, zero_sized: false];
            type [4] = Const<[0], 1> [storable: false, drop: false, dup: false, zero_sized: false];
            type [0] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = Array<[0]> [storable: true, drop: true, dup: false, zero_sized: false];

            libfunc [3] = array_new<[1]>;
            libfunc [2] = array_new<[0]>;
            libfunc [5] = const_as_immediate<[4]>;
            libfunc [13] = store_temp<[0]>;
            libfunc [1] = array_append<[0]>;
            libfunc [6] = const_as_immediate<[5]>;
            libfunc [7] = const_as_immediate<[6]>;
            libfunc [14] = store_temp<[1]>;
            libfunc [0] = array_append<[1]>;
            libfunc [8] = const_as_immediate<[7]>;
            libfunc [9] = const_as_immediate<[8]>;
            libfunc [10] = const_as_immediate<[9]>;
            libfunc [11] = snapshot_take<[2]>;
            libfunc [12] = drop<[2]>;
            libfunc [15] = store_temp<[3]>;

            [3]() -> ([0]); // 0
            [2]() -> ([1]); // 1
            [5]() -> ([2]); // 2
            [13]([2]) -> ([2]); // 3
            [1]([1], [2]) -> ([3]); // 4
            [6]() -> ([4]); // 5
            [13]([4]) -> ([4]); // 6
            [1]([3], [4]) -> ([5]); // 7
            [7]() -> ([6]); // 8
            [13]([6]) -> ([6]); // 9
            [1]([5], [6]) -> ([7]); // 10
            [14]([7]) -> ([7]); // 11
            [0]([0], [7]) -> ([8]); // 12
            [2]() -> ([9]); // 13
            [8]() -> ([10]); // 14
            [13]([10]) -> ([10]); // 15
            [1]([9], [10]) -> ([11]); // 16
            [9]() -> ([12]); // 17
            [13]([12]) -> ([12]); // 18
            [1]([11], [12]) -> ([13]); // 19
            [10]() -> ([14]); // 20
            [13]([14]) -> ([14]); // 21
            [1]([13], [14]) -> ([15]); // 22
            [14]([15]) -> ([15]); // 23
            [0]([8], [15]) -> ([16]); // 24
            [11]([16]) -> ([17], [18]); // 25
            [12]([17]) -> (); // 26
            [15]([18]) -> ([18]); // 27
            return([18]); // 28

            [0]@0() -> ([3]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let result = run_sierra_program(program, &[]).return_value;

        assert_eq!(
            result,
            Value::Array(vec![
                Value::Array(vec![
                    Value::Felt252(1.into()),
                    Value::Felt252(2.into()),
                    Value::Felt252(3.into()),
                ]),
                Value::Array(vec![
                    Value::Felt252(4.into()),
                    Value::Felt252(5.into()),
                    Value::Felt252(6.into()),
                ]),
            ]),
        );
    }
}
