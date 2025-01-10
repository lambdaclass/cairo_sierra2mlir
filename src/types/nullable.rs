//! # Nullable type
//!
//! Nullable is represented as a pointer, usually the null value will point to a alloca in the stack.
//!
//! A nullable is functionally equivalent to Rust's `Option<Box<T>>`. Since it's always paired with
//! `Box<T>` we can reuse its pointer, just leaving it null when there's no value.

use super::{TypeBuilder, WithSelf};
use crate::{
    error::Result,
    metadata::{
        drop_overrides::DropOverridesMeta, dup_overrides::DupOverridesMeta,
        realloc_bindings::ReallocBindingsMeta, MetadataStorage,
    },
    utils::{BlockExt, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        types::InfoAndTypeConcreteType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{cf, func},
    ir::Region,
};
use melior::{
    dialect::{llvm, ods},
    ir::{attribute::IntegerAttribute, r#type::IntegerType, Block, Location, Module, Type},
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

    // A nullable is represented by a pointer (equivalent to a box). A null value means no value.
    Ok(llvm::r#type::pointer(context, 0))
}

fn build_dup<'ctx>(
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

    let inner_ty = registry.get_type(&info.ty)?;
    let inner_len = inner_ty.layout(registry)?.pad_to_align().size();
    let inner_ty = inner_ty.build(context, module, registry, metadata, &info.ty)?;

    let region = Region::new();
    let entry = region.append_block(Block::new(&[(llvm::r#type::pointer(context, 0), location)]));

    let null_ptr =
        entry.append_op_result(llvm::zero(llvm::r#type::pointer(context, 0), location))?;
    let inner_len_val = entry.const_int(context, location, inner_len, 64)?;

    let src_value = entry.arg(0)?;
    let src_is_null = entry.append_op_result(
        ods::llvm::icmp(
            context,
            IntegerType::new(context, 1).into(),
            src_value,
            null_ptr,
            IntegerAttribute::new(IntegerType::new(context, 64).into(), 0).into(),
            location,
        )
        .into(),
    )?;

    let block_realloc = region.append_block(Block::new(&[]));
    let block_finish =
        region.append_block(Block::new(&[(llvm::r#type::pointer(context, 0), location)]));
    entry.append_operation(cf::cond_br(
        context,
        src_is_null,
        &block_finish,
        &block_realloc,
        &[null_ptr],
        &[],
        location,
    ));

    {
        let dst_value = block_realloc.append_op_result(ReallocBindingsMeta::realloc(
            context,
            null_ptr,
            inner_len_val,
            location,
        )?)?;

        match metadata.get::<DupOverridesMeta>() {
            Some(dup_override_meta) if dup_override_meta.is_overriden(&info.ty) => {
                let value = block_realloc.load(context, location, src_value, inner_ty)?;
                let values = dup_override_meta.invoke_override(
                    context,
                    &block_realloc,
                    location,
                    &info.ty,
                    value,
                )?;
                block_realloc.store(context, location, src_value, values.0)?;
                block_realloc.store(context, location, dst_value, values.1)?;
            }
            _ => {
                block_realloc.append_operation(
                    ods::llvm::intr_memcpy_inline(
                        context,
                        dst_value,
                        src_value,
                        IntegerAttribute::new(
                            IntegerType::new(context, 64).into(),
                            inner_len as i64,
                        ),
                        IntegerAttribute::new(IntegerType::new(context, 1).into(), 0),
                        location,
                    )
                    .into(),
                );
            }
        }

        block_realloc.append_operation(cf::br(&block_finish, &[dst_value], location));
    }

    block_finish.append_operation(func::r#return(&[src_value, block_finish.arg(0)?], location));
    Ok(region)
}

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

    let inner_ty = registry.build_type(context, module, metadata, &info.ty)?;

    let region = Region::new();
    let entry = region.append_block(Block::new(&[(llvm::r#type::pointer(context, 0), location)]));

    let null_ptr =
        entry.append_op_result(llvm::zero(llvm::r#type::pointer(context, 0), location))?;

    let value = entry.arg(0)?;
    let is_null = entry.append_op_result(
        ods::llvm::icmp(
            context,
            IntegerType::new(context, 1).into(),
            value,
            null_ptr,
            IntegerAttribute::new(IntegerType::new(context, 64).into(), 0).into(),
            location,
        )
        .into(),
    )?;

    let block_free = region.append_block(Block::new(&[]));
    let block_finish =
        region.append_block(Block::new(&[(llvm::r#type::pointer(context, 0), location)]));
    entry.append_operation(cf::cond_br(
        context,
        is_null,
        &block_finish,
        &block_free,
        &[null_ptr],
        &[],
        location,
    ));

    {
        match metadata.get::<DropOverridesMeta>() {
            Some(drop_override_meta) if drop_override_meta.is_overriden(&info.ty) => {
                let value = block_free.load(context, location, value, inner_ty)?;
                drop_override_meta.invoke_override(
                    context,
                    &block_free,
                    location,
                    &info.ty,
                    value,
                )?;
            }
            _ => {}
        }

        block_free.append_operation(ReallocBindingsMeta::free(context, value, location)?);
        block_free.append_operation(func::r#return(&[], location));
    }

    block_finish.append_operation(func::r#return(&[], location));
    Ok(region)
}

#[cfg(test)]
mod test {
    use crate::{
        utils::test::{jit_enum, jit_struct, run_sierra_program},
        values::Value,
    };
    use cairo_lang_sierra::ProgramParser;
    use pretty_assertions_sorted::assert_eq;

    #[test]
    fn test_nullable_deep_clone() {
        // use core::array::ArrayTrait;
        // use core::NullableTrait;
        // fn run_test() -> @Nullable<Array<felt252>> {
        //     let mut x = NullableTrait::new(array![1, 2, 3]);
        //     let x_s = @x;
        //     let mut y = NullableTrait::deref(x);
        //     y.append(4);
        //     x_s
        // }
        let program = ProgramParser::new().parse(r#"
            type [1] = Array<[0]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [3] = Nullable<[1]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [4] = Snapshot<[3]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [7] = Struct<ut@Tuple, [4]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [13] = Const<[0], 4> [storable: false, drop: false, dup: false, zero_sized: false];
            type [5] = Struct<ut@core::panics::Panic> [storable: true, drop: true, dup: true, zero_sized: true];
            type [6] = Struct<ut@Tuple, [5], [1]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [8] = Enum<ut@core::panics::PanicResult::<(@core::nullable::Nullable::<core::array::Array::<core::felt252>>,)>, [7], [6]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [12] = Const<[0], 1764660641818210475137527732331061317596259760618687855268902447379813> [storable: false, drop: false, dup: false, zero_sized: false];
            type [2] = Box<[1]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [11] = Const<[0], 3> [storable: false, drop: false, dup: false, zero_sized: false];
            type [10] = Const<[0], 2> [storable: false, drop: false, dup: false, zero_sized: false];
            type [9] = Const<[0], 1> [storable: false, drop: false, dup: false, zero_sized: false];
            type [0] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];

            libfunc [7] = array_new<[0]>;
            libfunc [12] = const_as_immediate<[9]>;
            libfunc [21] = store_temp<[0]>;
            libfunc [2] = array_append<[0]>;
            libfunc [13] = const_as_immediate<[10]>;
            libfunc [14] = const_as_immediate<[11]>;
            libfunc [22] = store_temp<[1]>;
            libfunc [10] = into_box<[1]>;
            libfunc [9] = nullable_from_box<[1]>;
            libfunc [15] = snapshot_take<[3]>;
            libfunc [8] = match_nullable<[1]>;
            libfunc [16] = branch_align;
            libfunc [17] = drop<[4]>;
            libfunc [18] = const_as_immediate<[12]>;
            libfunc [6] = struct_construct<[5]>;
            libfunc [5] = struct_construct<[6]>;
            libfunc [4] = enum_init<[8], 1>;
            libfunc [23] = store_temp<[8]>;
            libfunc [3] = unbox<[1]>;
            libfunc [19] = const_as_immediate<[13]>;
            libfunc [20] = drop<[1]>;
            libfunc [1] = struct_construct<[7]>;
            libfunc [0] = enum_init<[8], 0>;

            [7]() -> ([0]); // 0
            [12]() -> ([1]); // 1
            [21]([1]) -> ([1]); // 2
            [2]([0], [1]) -> ([2]); // 3
            [13]() -> ([3]); // 4
            [21]([3]) -> ([3]); // 5
            [2]([2], [3]) -> ([4]); // 6
            [14]() -> ([5]); // 7
            [21]([5]) -> ([5]); // 8
            [2]([4], [5]) -> ([6]); // 9
            [22]([6]) -> ([6]); // 10
            [10]([6]) -> ([7]); // 11
            [9]([7]) -> ([8]); // 12
            [15]([8]) -> ([9], [10]); // 13
            [8]([9]) { fallthrough() 26([11]) }; // 14
            [16]() -> (); // 15
            [17]([10]) -> (); // 16
            [7]() -> ([12]); // 17
            [18]() -> ([13]); // 18
            [21]([13]) -> ([13]); // 19
            [2]([12], [13]) -> ([14]); // 20
            [6]() -> ([15]); // 21
            [5]([15], [14]) -> ([16]); // 22
            [4]([16]) -> ([17]); // 23
            [23]([17]) -> ([17]); // 24
            return([17]); // 25
            [16]() -> (); // 26
            [3]([11]) -> ([18]); // 27
            [19]() -> ([19]); // 28
            [22]([18]) -> ([18]); // 29
            [21]([19]) -> ([19]); // 30
            [2]([18], [19]) -> ([20]); // 31
            [20]([20]) -> (); // 32
            [1]([10]) -> ([21]); // 33
            [0]([21]) -> ([22]); // 34
            [23]([22]) -> ([22]); // 35
            return([22]); // 36

            [0]@0() -> ([8]);
        "#).map_err(|e| e.to_string()).unwrap();

        let result = run_sierra_program(program, &[]).return_value;

        assert_eq!(
            result,
            jit_enum!(
                0,
                jit_struct!(Value::Array(vec![
                    Value::Felt252(1.into()),
                    Value::Felt252(2.into()),
                    Value::Felt252(3.into()),
                ]))
            ),
        );
    }
}
