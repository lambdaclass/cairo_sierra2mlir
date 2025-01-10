//! # `Felt` dictionary type
//!
//! A key value storage for values whose type implement Copy. The key is always a felt.
//!
//! This type is represented as a pointer to a tuple of a heap allocated Rust hashmap along with a u64
//! used to count accesses to the dictionary. The type is interacted through the runtime functions to
//! insert, get elements and increment the access counter.

use super::WithSelf;
use crate::{
    error::{Error, Result},
    metadata::{
        drop_overrides::DropOverridesMeta, dup_overrides::DupOverridesMeta,
        felt252_dict::Felt252DictOverrides, realloc_bindings::ReallocBindingsMeta,
        runtime_bindings::RuntimeBindingsMeta, MetadataStorage,
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
    dialect::{func, llvm},
    ir::{Block, Location, Module, Region, Type},
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

    let value_ty = registry.build_type(context, module, metadata, info.self_ty())?;

    {
        let mut dict_overrides = metadata
            .remove::<Felt252DictOverrides>()
            .unwrap_or_default();
        dict_overrides.build_dup_fn(context, module, registry, metadata, &info.ty)?;
        metadata.insert(dict_overrides);
    }

    let region = Region::new();
    let entry = region.append_block(Block::new(&[(value_ty, location)]));

    // The following unwrap is unreachable because the registration logic will always insert it.
    let value0 = entry.arg(0)?;
    let value1 = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .ok_or(Error::MissingMetadata)?
        .dict_dup(context, module, &entry, value0, location)?;

    entry.append_operation(func::r#return(&[value0, value1], location));
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

    let value_ty = registry.build_type(context, module, metadata, info.self_ty())?;

    {
        let mut dict_overrides = metadata
            .remove::<Felt252DictOverrides>()
            .unwrap_or_default();
        dict_overrides.build_drop_fn(context, module, registry, metadata, &info.ty)?;
        metadata.insert(dict_overrides);
    }

    let region = Region::new();
    let entry = region.append_block(Block::new(&[(value_ty, location)]));

    // The following unwrap is unreachable because the registration logic will always insert it.
    let runtime_bindings_meta = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .ok_or(Error::MissingMetadata)?;
    runtime_bindings_meta.dict_drop(context, module, &entry, entry.arg(0)?, location)?;

    entry.append_operation(func::r#return(&[], location));
    Ok(region)
}

#[cfg(test)]
mod test {
    use crate::{
        utils::test::{jit_dict, run_sierra_program},
        values::Value,
    };
    use cairo_lang_sierra::ProgramParser;
    use pretty_assertions_sorted::assert_eq;
    use starknet_types_core::felt::Felt;
    use std::collections::HashMap;

    #[test]
    fn dict_snapshot_take() {
        // fn run_test() -> @Felt252Dict<u32> {
        //     let mut dict: Felt252Dict<u32> = Default::default();
        //     dict.insert(2, 1_u32);
        //     @dict
        // }
        let program = ProgramParser::new().parse(r#"
            type [6] = Felt252DictEntry<[3]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [11] = Uninitialized<[6]> [storable: false, drop: true, dup: false, zero_sized: false];
            type [8] = SquashedFelt252Dict<[3]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [2] = GasBuiltin [storable: true, drop: false, dup: false, zero_sized: false];
            type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
            type [4] = Felt252Dict<[3]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [7] = Snapshot<[4]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [10] = Const<[3], 1> [storable: false, drop: false, dup: false, zero_sized: false];
            type [9] = Const<[5], 2> [storable: false, drop: false, dup: false, zero_sized: false];
            type [5] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [3] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = SegmentArena [storable: true, drop: false, dup: false, zero_sized: false];

            libfunc [11] = alloc_local<[6]>;
            libfunc [12] = finalize_locals;
            libfunc [3] = disable_ap_tracking;
            libfunc [10] = felt252_dict_new<[3]>;
            libfunc [13] = const_as_immediate<[9]>;
            libfunc [18] = store_temp<[4]>;
            libfunc [19] = store_temp<[5]>;
            libfunc [9] = felt252_dict_entry_get<[3]>;
            libfunc [14] = drop<[3]>;
            libfunc [15] = const_as_immediate<[10]>;
            libfunc [20] = store_local<[6]>;
            libfunc [21] = store_temp<[3]>;
            libfunc [8] = felt252_dict_entry_finalize<[3]>;
            libfunc [16] = snapshot_take<[4]>;
            libfunc [4] = store_temp<[0]>;
            libfunc [5] = store_temp<[1]>;
            libfunc [6] = store_temp<[2]>;
            libfunc [0] = function_call<user@[0]>;
            libfunc [17] = drop<[8]>;
            libfunc [22] = store_temp<[7]>;
            libfunc [1] = felt252_dict_squash<[3]>;
            libfunc [7] = store_temp<[8]>;

            [11]() -> ([4]); // 0
            [12]() -> (); // 1
            [3]() -> (); // 2
            [10]([1]) -> ([5], [6]); // 3
            [13]() -> ([7]); // 4
            [18]([6]) -> ([6]); // 5
            [19]([7]) -> ([7]); // 6
            [9]([6], [7]) -> ([3], [8]); // 7
            [14]([8]) -> (); // 8
            [15]() -> ([9]); // 9
            [20]([4], [3]) -> ([3]); // 10
            [21]([9]) -> ([9]); // 11
            [8]([3], [9]) -> ([10]); // 12
            [16]([10]) -> ([11], [12]); // 13
            [4]([0]) -> ([0]); // 14
            [5]([5]) -> ([5]); // 15
            [6]([2]) -> ([2]); // 16
            [18]([11]) -> ([11]); // 17
            [0]([0], [5], [2], [11]) -> ([13], [14], [15], [16]); // 18
            [17]([16]) -> (); // 19
            [4]([13]) -> ([13]); // 20
            [5]([14]) -> ([14]); // 21
            [6]([15]) -> ([15]); // 22
            [22]([12]) -> ([12]); // 23
            return([13], [14], [15], [12]); // 24
            [3]() -> (); // 25
            [1]([0], [2], [1], [3]) -> ([4], [5], [6], [7]); // 26
            [4]([4]) -> ([4]); // 27
            [5]([6]) -> ([6]); // 28
            [6]([5]) -> ([5]); // 29
            [7]([7]) -> ([7]); // 30
            return([4], [6], [5], [7]); // 31

            [1]@0([0]: [0], [1]: [1], [2]: [2]) -> ([0], [1], [2], [7]);
            [0]@25([0]: [0], [1]: [1], [2]: [2], [3]: [4]) -> ([0], [1], [2], [8]);
        "#).map_err(|e| e.to_string()).unwrap();

        let result = run_sierra_program(program, &[]).return_value;

        assert_eq!(
            result,
            jit_dict!(
                2 => 1u32
            ),
        );
    }

    #[test]
    fn dict_snapshot_take_complex() {
        // fn run_test() -> @Felt252Dict<Nullable<Array<u32>>> {
        //     let mut dict: Felt252Dict<Nullable<Array<u32>>> = Default::default();
        //     dict.insert(2, NullableTrait::new(array![3, 4]));
        //     @dict
        // }
        let program = ProgramParser::new().parse(r#"
            type [9] = Felt252DictEntry<[5]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [15] = Uninitialized<[9]> [storable: false, drop: true, dup: false, zero_sized: false];
            type [11] = SquashedFelt252Dict<[5]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [2] = GasBuiltin [storable: true, drop: false, dup: false, zero_sized: false];
            type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
            type [6] = Felt252Dict<[5]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [10] = Snapshot<[6]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [14] = Const<[8], 2> [storable: false, drop: false, dup: false, zero_sized: false];
            type [8] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [7] = Box<[4]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [13] = Const<[3], 4> [storable: false, drop: false, dup: false, zero_sized: false];
            type [12] = Const<[3], 3> [storable: false, drop: false, dup: false, zero_sized: false];
            type [3] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
            type [4] = Array<[3]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [5] = Nullable<[4]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [1] = SegmentArena [storable: true, drop: false, dup: false, zero_sized: false];

            libfunc [15] = alloc_local<[9]>;
            libfunc [16] = finalize_locals;
            libfunc [3] = disable_ap_tracking;
            libfunc [14] = felt252_dict_new<[5]>;
            libfunc [13] = array_new<[3]>;
            libfunc [17] = const_as_immediate<[12]>;
            libfunc [23] = store_temp<[3]>;
            libfunc [12] = array_append<[3]>;
            libfunc [18] = const_as_immediate<[13]>;
            libfunc [24] = store_temp<[4]>;
            libfunc [11] = into_box<[4]>;
            libfunc [10] = nullable_from_box<[4]>;
            libfunc [19] = const_as_immediate<[14]>;
            libfunc [25] = store_temp<[6]>;
            libfunc [26] = store_temp<[8]>;
            libfunc [9] = felt252_dict_entry_get<[5]>;
            libfunc [20] = drop<[5]>;
            libfunc [27] = store_local<[9]>;
            libfunc [8] = felt252_dict_entry_finalize<[5]>;
            libfunc [21] = snapshot_take<[6]>;
            libfunc [4] = store_temp<[0]>;
            libfunc [5] = store_temp<[1]>;
            libfunc [6] = store_temp<[2]>;
            libfunc [0] = function_call<user@[0]>;
            libfunc [22] = drop<[11]>;
            libfunc [28] = store_temp<[10]>;
            libfunc [1] = felt252_dict_squash<[5]>;
            libfunc [7] = store_temp<[11]>;

            [15]() -> ([4]); // 0
            [16]() -> (); // 1
            [3]() -> (); // 2
            [14]([1]) -> ([5], [6]); // 3
            [13]() -> ([7]); // 4
            [17]() -> ([8]); // 5
            [23]([8]) -> ([8]); // 6
            [12]([7], [8]) -> ([9]); // 7
            [18]() -> ([10]); // 8
            [23]([10]) -> ([10]); // 9
            [12]([9], [10]) -> ([11]); // 10
            [24]([11]) -> ([11]); // 11
            [11]([11]) -> ([12]); // 12
            [10]([12]) -> ([13]); // 13
            [19]() -> ([14]); // 14
            [25]([6]) -> ([6]); // 15
            [26]([14]) -> ([14]); // 16
            [9]([6], [14]) -> ([3], [15]); // 17
            [20]([15]) -> (); // 18
            [27]([4], [3]) -> ([3]); // 19
            [8]([3], [13]) -> ([16]); // 20
            [21]([16]) -> ([17], [18]); // 21
            [4]([0]) -> ([0]); // 22
            [5]([5]) -> ([5]); // 23
            [6]([2]) -> ([2]); // 24
            [25]([17]) -> ([17]); // 25
            [0]([0], [5], [2], [17]) -> ([19], [20], [21], [22]); // 26
            [22]([22]) -> (); // 27
            [4]([19]) -> ([19]); // 28
            [5]([20]) -> ([20]); // 29
            [6]([21]) -> ([21]); // 30
            [28]([18]) -> ([18]); // 31
            return([19], [20], [21], [18]); // 32
            [3]() -> (); // 33
            [1]([0], [2], [1], [3]) -> ([4], [5], [6], [7]); // 34
            [4]([4]) -> ([4]); // 35
            [5]([6]) -> ([6]); // 36
            [6]([5]) -> ([5]); // 37
            [7]([7]) -> ([7]); // 38
            return([4], [6], [5], [7]); // 39

            [1]@0([0]: [0], [1]: [1], [2]: [2]) -> ([0], [1], [2], [10]);
            [0]@33([0]: [0], [1]: [1], [2]: [2], [3]: [6]) -> ([0], [1], [2], [11]);
        "#).map_err(|e| e.to_string()).unwrap();

        let result = run_sierra_program(program, &[]).return_value;

        assert_eq!(
            result,
            jit_dict!(
                2 => Value::Array(vec![3u32.into(), 4u32.into()])
            ),
        );
    }

    #[test]
    fn dict_snapshot_take_compare() {
        // fn run_test() -> @Felt252Dict<Nullable<Array<u32>>> {
        //     let mut dict: Felt252Dict<Nullable<Array<u32>>> = Default::default();
        //     dict.insert(2, NullableTrait::new(array![3, 4]));
        //     @dict
        // }
        let program = ProgramParser::new()
            .parse(r#"
                type [9] = Felt252DictEntry<[5]> [storable: true, drop: false, dup: false, zero_sized: false];
                type [15] = Uninitialized<[9]> [storable: false, drop: true, dup: false, zero_sized: false];
                type [11] = SquashedFelt252Dict<[5]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [2] = GasBuiltin [storable: true, drop: false, dup: false, zero_sized: false];
                type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
                type [6] = Felt252Dict<[5]> [storable: true, drop: false, dup: false, zero_sized: false];
                type [10] = Snapshot<[6]> [storable: true, drop: true, dup: true, zero_sized: false];
                type [14] = Const<[8], 2> [storable: false, drop: false, dup: false, zero_sized: false];
                type [8] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
                type [7] = Box<[4]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [13] = Const<[3], 4> [storable: false, drop: false, dup: false, zero_sized: false];
                type [12] = Const<[3], 3> [storable: false, drop: false, dup: false, zero_sized: false];
                type [3] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
                type [4] = Array<[3]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [5] = Nullable<[4]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [1] = SegmentArena [storable: true, drop: false, dup: false, zero_sized: false];

                libfunc [15] = alloc_local<[9]>;
                libfunc [16] = finalize_locals;
                libfunc [3] = disable_ap_tracking;
                libfunc [14] = felt252_dict_new<[5]>;
                libfunc [13] = array_new<[3]>;
                libfunc [17] = const_as_immediate<[12]>;
                libfunc [23] = store_temp<[3]>;
                libfunc [12] = array_append<[3]>;
                libfunc [18] = const_as_immediate<[13]>;
                libfunc [24] = store_temp<[4]>;
                libfunc [11] = into_box<[4]>;
                libfunc [10] = nullable_from_box<[4]>;
                libfunc [19] = const_as_immediate<[14]>;
                libfunc [25] = store_temp<[6]>;
                libfunc [26] = store_temp<[8]>;
                libfunc [9] = felt252_dict_entry_get<[5]>;
                libfunc [20] = drop<[5]>;
                libfunc [27] = store_local<[9]>;
                libfunc [8] = felt252_dict_entry_finalize<[5]>;
                libfunc [21] = snapshot_take<[6]>;
                libfunc [4] = store_temp<[0]>;
                libfunc [5] = store_temp<[1]>;
                libfunc [6] = store_temp<[2]>;
                libfunc [0] = function_call<user@[0]>;
                libfunc [22] = drop<[11]>;
                libfunc [28] = store_temp<[10]>;
                libfunc [1] = felt252_dict_squash<[5]>;
                libfunc [7] = store_temp<[11]>;

                [15]() -> ([4]); // 0
                [16]() -> (); // 1
                [3]() -> (); // 2
                [14]([1]) -> ([5], [6]); // 3
                [13]() -> ([7]); // 4
                [17]() -> ([8]); // 5
                [23]([8]) -> ([8]); // 6
                [12]([7], [8]) -> ([9]); // 7
                [18]() -> ([10]); // 8
                [23]([10]) -> ([10]); // 9
                [12]([9], [10]) -> ([11]); // 10
                [24]([11]) -> ([11]); // 11
                [11]([11]) -> ([12]); // 12
                [10]([12]) -> ([13]); // 13
                [19]() -> ([14]); // 14
                [25]([6]) -> ([6]); // 15
                [26]([14]) -> ([14]); // 16
                [9]([6], [14]) -> ([3], [15]); // 17
                [20]([15]) -> (); // 18
                [27]([4], [3]) -> ([3]); // 19
                [8]([3], [13]) -> ([16]); // 20
                [21]([16]) -> ([17], [18]); // 21
                [4]([0]) -> ([0]); // 22
                [5]([5]) -> ([5]); // 23
                [6]([2]) -> ([2]); // 24
                [25]([17]) -> ([17]); // 25
                [0]([0], [5], [2], [17]) -> ([19], [20], [21], [22]); // 26
                [22]([22]) -> (); // 27
                [4]([19]) -> ([19]); // 28
                [5]([20]) -> ([20]); // 29
                [6]([21]) -> ([21]); // 30
                [28]([18]) -> ([18]); // 31
                return([19], [20], [21], [18]); // 32
                [3]() -> (); // 33
                [1]([0], [2], [1], [3]) -> ([4], [5], [6], [7]); // 34
                [4]([4]) -> ([4]); // 35
                [5]([6]) -> ([6]); // 36
                [6]([5]) -> ([5]); // 37
                [7]([7]) -> ([7]); // 38
                return([4], [6], [5], [7]); // 39

                [1]@0([0]: [0], [1]: [1], [2]: [2]) -> ([0], [1], [2], [10]);
                [0]@33([0]: [0], [1]: [1], [2]: [2], [3]: [6]) -> ([0], [1], [2], [11]);
            "#)
            .map_err(|e| e.to_string())
            .unwrap();

        // fn run_test() -> Felt252Dict<Nullable<Array<u32>>> {
        //     let mut dict: Felt252Dict<Nullable<Array<u32>>> = Default::default();
        //     dict.insert(2, NullableTrait::new(array![3, 4]));
        //     dict
        // }
        let program2 = ProgramParser::new()
            .parse(r#"
                type [0] = SegmentArena [storable: true, drop: false, dup: false, zero_sized: false];
                type [7] = Felt252DictEntry<[3]> [storable: true, drop: false, dup: false, zero_sized: false];
                type [10] = Const<[6], 2> [storable: false, drop: false, dup: false, zero_sized: false];
                type [6] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
                type [5] = Box<[2]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [9] = Const<[1], 4> [storable: false, drop: false, dup: false, zero_sized: false];
                type [8] = Const<[1], 3> [storable: false, drop: false, dup: false, zero_sized: false];
                type [1] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
                type [2] = Array<[1]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [3] = Nullable<[2]> [storable: true, drop: true, dup: false, zero_sized: false];
                type [4] = Felt252Dict<[3]> [storable: true, drop: false, dup: false, zero_sized: false];

                libfunc [6] = felt252_dict_new<[3]>;
                libfunc [5] = array_new<[1]>;
                libfunc [8] = const_as_immediate<[8]>;
                libfunc [12] = store_temp<[1]>;
                libfunc [4] = array_append<[1]>;
                libfunc [9] = const_as_immediate<[9]>;
                libfunc [13] = store_temp<[2]>;
                libfunc [3] = into_box<[2]>;
                libfunc [2] = nullable_from_box<[2]>;
                libfunc [10] = const_as_immediate<[10]>;
                libfunc [14] = store_temp<[4]>;
                libfunc [15] = store_temp<[6]>;
                libfunc [1] = felt252_dict_entry_get<[3]>;
                libfunc [11] = drop<[3]>;
                libfunc [0] = felt252_dict_entry_finalize<[3]>;
                libfunc [16] = store_temp<[0]>;

                [6]([0]) -> ([1], [2]); // 0
                [5]() -> ([3]); // 1
                [8]() -> ([4]); // 2
                [12]([4]) -> ([4]); // 3
                [4]([3], [4]) -> ([5]); // 4
                [9]() -> ([6]); // 5
                [12]([6]) -> ([6]); // 6
                [4]([5], [6]) -> ([7]); // 7
                [13]([7]) -> ([7]); // 8
                [3]([7]) -> ([8]); // 9
                [2]([8]) -> ([9]); // 10
                [10]() -> ([10]); // 11
                [14]([2]) -> ([2]); // 12
                [15]([10]) -> ([10]); // 13
                [1]([2], [10]) -> ([11], [12]); // 14
                [11]([12]) -> (); // 15
                [0]([11], [9]) -> ([13]); // 16
                [16]([1]) -> ([1]); // 17
                [14]([13]) -> ([13]); // 18
                return([1], [13]); // 19

                [0]@0([0]: [0]) -> ([0], [4]);
            "#)
            .map_err(|e| e.to_string())
            .unwrap();

        let result1 = run_sierra_program(program, &[]).return_value;
        let result2 = run_sierra_program(program2, &[]).return_value;

        assert_eq!(result1, result2);
    }

    /// Ensure that a dictionary of booleans compiles.
    #[test]
    fn dict_type_bool() {
        // fn run_program() -> Felt252Dict<bool> {
        //     let mut x: Felt252Dict<bool> = Default::default();
        //     x.insert(0, false);
        //     x.insert(1, true);
        //     x
        // }
        let program = ProgramParser::new().parse(r#"
            type [0] = SegmentArena [storable: true, drop: false, dup: false, zero_sized: false];
            type [7] = Const<[4], 1> [storable: false, drop: false, dup: false, zero_sized: false];
            type [1] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
            type [5] = Felt252DictEntry<[2]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [6] = Const<[4], 0> [storable: false, drop: false, dup: false, zero_sized: false];
            type [4] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [2] = Enum<ut@core::bool, [1], [1]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [3] = Felt252Dict<[2]> [storable: true, drop: false, dup: false, zero_sized: false];

            libfunc [5] = felt252_dict_new<[2]>;
            libfunc [7] = const_as_immediate<[6]>;
            libfunc [10] = store_temp<[3]>;
            libfunc [11] = store_temp<[4]>;
            libfunc [3] = felt252_dict_entry_get<[2]>;
            libfunc [8] = drop<[2]>;
            libfunc [2] = struct_construct<[1]>;
            libfunc [4] = enum_init<[2], 0>;
            libfunc [12] = store_temp<[2]>;
            libfunc [0] = felt252_dict_entry_finalize<[2]>;
            libfunc [9] = const_as_immediate<[7]>;
            libfunc [1] = enum_init<[2], 1>;
            libfunc [13] = store_temp<[0]>;

            [5]([0]) -> ([1], [2]); // 0
            [7]() -> ([3]); // 1
            [10]([2]) -> ([2]); // 2
            [11]([3]) -> ([3]); // 3
            [3]([2], [3]) -> ([4], [5]); // 4
            [8]([5]) -> (); // 5
            [2]() -> ([6]); // 6
            [4]([6]) -> ([7]); // 7
            [12]([7]) -> ([7]); // 8
            [0]([4], [7]) -> ([8]); // 9
            [9]() -> ([9]); // 10
            [11]([9]) -> ([9]); // 11
            [3]([8], [9]) -> ([10], [11]); // 12
            [8]([11]) -> (); // 13
            [2]() -> ([12]); // 14
            [1]([12]) -> ([13]); // 15
            [12]([13]) -> ([13]); // 16
            [0]([10], [13]) -> ([14]); // 17
            [13]([1]) -> ([1]); // 18
            [10]([14]) -> ([14]); // 19
            return([1], [14]); // 20

            [0]@0([0]: [0]) -> ([0], [3]);
        "#).map_err(|e| e.to_string()).unwrap();

        let result = run_sierra_program(program, &[]);
        assert_eq!(
            result.return_value,
            Value::Felt252Dict {
                value: HashMap::from([
                    (
                        Felt::ZERO,
                        Value::Enum {
                            tag: 0,
                            value: Box::new(Value::Struct {
                                fields: Vec::new(),
                                debug_name: None
                            }),
                            debug_name: None,
                        },
                    ),
                    (
                        Felt::ONE,
                        Value::Enum {
                            tag: 1,
                            value: Box::new(Value::Struct {
                                fields: Vec::new(),
                                debug_name: None
                            }),
                            debug_name: None,
                        },
                    ),
                ]),
                debug_name: None,
            },
        );
    }

    /// Ensure that a dictionary of felts compiles.
    #[test]
    fn dict_type_felt252() {
        // fn run_program() -> Felt252Dict<felt252> {
        //     let mut x: Felt252Dict<felt252> = Default::default();
        //     x.insert(0, 0);
        //     x.insert(1, 1);
        //     x.insert(2, 2);
        //     x.insert(3, 3);
        //     x
        // }
        let program = ProgramParser::new().parse(r#"
            type [0] = SegmentArena [storable: true, drop: false, dup: false, zero_sized: false];
            type [7] = Const<[1], 3> [storable: false, drop: false, dup: false, zero_sized: false];
            type [6] = Const<[1], 2> [storable: false, drop: false, dup: false, zero_sized: false];
            type [5] = Const<[1], 1> [storable: false, drop: false, dup: false, zero_sized: false];
            type [3] = Felt252DictEntry<[1]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [4] = Const<[1], 0> [storable: false, drop: false, dup: false, zero_sized: false];
            type [1] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [2] = Felt252Dict<[1]> [storable: true, drop: false, dup: false, zero_sized: false];

            libfunc [2] = felt252_dict_new<[1]>;
            libfunc [4] = const_as_immediate<[4]>;
            libfunc [9] = store_temp<[2]>;
            libfunc [10] = store_temp<[1]>;
            libfunc [1] = felt252_dict_entry_get<[1]>;
            libfunc [5] = drop<[1]>;
            libfunc [0] = felt252_dict_entry_finalize<[1]>;
            libfunc [6] = const_as_immediate<[5]>;
            libfunc [7] = const_as_immediate<[6]>;
            libfunc [8] = const_as_immediate<[7]>;
            libfunc [11] = store_temp<[0]>;

            [2]([0]) -> ([1], [2]); // 0
            [4]() -> ([3]); // 1
            [9]([2]) -> ([2]); // 2
            [10]([3]) -> ([3]); // 3
            [1]([2], [3]) -> ([4], [5]); // 4
            [5]([5]) -> (); // 5
            [4]() -> ([6]); // 6
            [10]([6]) -> ([6]); // 7
            [0]([4], [6]) -> ([7]); // 8
            [6]() -> ([8]); // 9
            [10]([8]) -> ([8]); // 10
            [1]([7], [8]) -> ([9], [10]); // 11
            [5]([10]) -> (); // 12
            [6]() -> ([11]); // 13
            [10]([11]) -> ([11]); // 14
            [0]([9], [11]) -> ([12]); // 15
            [7]() -> ([13]); // 16
            [10]([13]) -> ([13]); // 17
            [1]([12], [13]) -> ([14], [15]); // 18
            [5]([15]) -> (); // 19
            [7]() -> ([16]); // 20
            [10]([16]) -> ([16]); // 21
            [0]([14], [16]) -> ([17]); // 22
            [8]() -> ([18]); // 23
            [10]([18]) -> ([18]); // 24
            [1]([17], [18]) -> ([19], [20]); // 25
            [5]([20]) -> (); // 26
            [8]() -> ([21]); // 27
            [10]([21]) -> ([21]); // 28
            [0]([19], [21]) -> ([22]); // 29
            [11]([1]) -> ([1]); // 30
            [9]([22]) -> ([22]); // 31
            return([1], [22]); // 32

            [0]@0([0]: [0]) -> ([0], [2]);
        "#).map_err(|e| e.to_string()).unwrap();

        let result = run_sierra_program(program, &[]);
        assert_eq!(
            result.return_value,
            Value::Felt252Dict {
                value: HashMap::from([
                    (Felt::ZERO, Value::Felt252(Felt::ZERO)),
                    (Felt::ONE, Value::Felt252(Felt::ONE)),
                    (Felt::TWO, Value::Felt252(Felt::TWO)),
                    (Felt::THREE, Value::Felt252(Felt::THREE)),
                ]),
                debug_name: None,
            },
        );
    }

    /// Ensure that a dictionary of nullables compiles.
    #[test]
    fn dict_type_nullable() {
        // #[derive(Drop)]
        // struct MyStruct {
        //     a: u8,
        //     b: i16,
        //     c: felt252,
        // }
        // fn run_program() -> Felt252Dict<Nullable<MyStruct>> {
        //     let mut x: Felt252Dict<Nullable<MyStruct>> = Default::default();
        //     x.insert(0, Default::default());
        //     x.insert(1, NullableTrait::new(MyStruct { a: 0, b: 1, c: 2 }));
        //     x.insert(2, NullableTrait::new(MyStruct { a: 1, b: -2, c: 3 }));
        //     x.insert(3, NullableTrait::new(MyStruct { a: 2, b: 3, c: 4 }));
        //     x
        // }
        let program = ProgramParser::new().parse(r#"
            type [0] = SegmentArena [storable: true, drop: false, dup: false, zero_sized: false];
            type [17] = Const<[3], 3> [storable: false, drop: false, dup: false, zero_sized: false];
            type [22] = Const<[4], [19], [20], [21]> [storable: false, drop: false, dup: false, zero_sized: false];
            type [12] = Const<[3], 2> [storable: false, drop: false, dup: false, zero_sized: false];
            type [21] = Const<[3], 4> [storable: false, drop: false, dup: false, zero_sized: false];
            type [20] = Const<[2], 3> [storable: false, drop: false, dup: false, zero_sized: false];
            type [19] = Const<[1], 2> [storable: false, drop: false, dup: false, zero_sized: false];
            type [2] = i16 [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = u8 [storable: true, drop: true, dup: true, zero_sized: false];
            type [18] = Const<[4], [15], [16], [17]> [storable: false, drop: false, dup: false, zero_sized: false];
            type [14] = Const<[3], 1> [storable: false, drop: false, dup: false, zero_sized: false];
            type [16] = Const<[2], -2> [storable: false, drop: false, dup: false, zero_sized: false];
            type [15] = Const<[1], 1> [storable: false, drop: false, dup: false, zero_sized: false];
            type [13] = Const<[4], [10], [11], [12]> [storable: false, drop: false, dup: false, zero_sized: false];
            type [8] = Box<[4]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [11] = Const<[2], 1> [storable: false, drop: false, dup: false, zero_sized: false];
            type [10] = Const<[1], 0> [storable: false, drop: false, dup: false, zero_sized: false];
            type [7] = Felt252DictEntry<[5]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [9] = Const<[3], 0> [storable: false, drop: false, dup: false, zero_sized: false];
            type [3] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [4] = Struct<ut@program::program::MyStruct, [1], [2], [3]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [5] = Nullable<[4]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [6] = Felt252Dict<[5]> [storable: true, drop: false, dup: false, zero_sized: false];

            libfunc [4] = felt252_dict_new<[5]>;
            libfunc [3] = null<[4]>;
            libfunc [6] = const_as_immediate<[9]>;
            libfunc [14] = store_temp<[6]>;
            libfunc [15] = store_temp<[3]>;
            libfunc [1] = felt252_dict_entry_get<[5]>;
            libfunc [7] = drop<[5]>;
            libfunc [16] = store_temp<[5]>;
            libfunc [0] = felt252_dict_entry_finalize<[5]>;
            libfunc [8] = const_as_box<[13], 0>;
            libfunc [2] = nullable_from_box<[4]>;
            libfunc [9] = const_as_immediate<[14]>;
            libfunc [10] = const_as_box<[18], 0>;
            libfunc [11] = const_as_immediate<[12]>;
            libfunc [12] = const_as_box<[22], 0>;
            libfunc [13] = const_as_immediate<[17]>;
            libfunc [17] = store_temp<[0]>;

            [4]([0]) -> ([1], [2]); // 0
            [3]() -> ([3]); // 1
            [6]() -> ([4]); // 2
            [14]([2]) -> ([2]); // 3
            [15]([4]) -> ([4]); // 4
            [1]([2], [4]) -> ([5], [6]); // 5
            [7]([6]) -> (); // 6
            [16]([3]) -> ([3]); // 7
            [0]([5], [3]) -> ([7]); // 8
            [8]() -> ([8]); // 9
            [2]([8]) -> ([9]); // 10
            [9]() -> ([10]); // 11
            [15]([10]) -> ([10]); // 12
            [1]([7], [10]) -> ([11], [12]); // 13
            [7]([12]) -> (); // 14
            [0]([11], [9]) -> ([13]); // 15
            [10]() -> ([14]); // 16
            [2]([14]) -> ([15]); // 17
            [11]() -> ([16]); // 18
            [15]([16]) -> ([16]); // 19
            [1]([13], [16]) -> ([17], [18]); // 20
            [7]([18]) -> (); // 21
            [0]([17], [15]) -> ([19]); // 22
            [12]() -> ([20]); // 23
            [2]([20]) -> ([21]); // 24
            [13]() -> ([22]); // 25
            [15]([22]) -> ([22]); // 26
            [1]([19], [22]) -> ([23], [24]); // 27
            [7]([24]) -> (); // 28
            [0]([23], [21]) -> ([25]); // 29
            [17]([1]) -> ([1]); // 30
            [14]([25]) -> ([25]); // 31
            return([1], [25]); // 32

            [0]@0([0]: [0]) -> ([0], [6]);
        "#).map_err(|e| e.to_string()).unwrap();

        let result = run_sierra_program(program, &[]);
        pretty_assertions_sorted::assert_eq_sorted!(
            result.return_value,
            Value::Felt252Dict {
                value: HashMap::from([
                    (Felt::ZERO, Value::Null),
                    (
                        Felt::ONE,
                        Value::Struct {
                            fields: Vec::from([
                                Value::Uint8(0),
                                Value::Sint16(1),
                                Value::Felt252(2.into()),
                            ]),
                            debug_name: None,
                        },
                    ),
                    (
                        Felt::TWO,
                        Value::Struct {
                            fields: Vec::from([
                                Value::Uint8(1),
                                Value::Sint16(-2),
                                Value::Felt252(3.into()),
                            ]),
                            debug_name: None,
                        },
                    ),
                    (
                        Felt::THREE,
                        Value::Struct {
                            fields: Vec::from([
                                Value::Uint8(2),
                                Value::Sint16(3),
                                Value::Felt252(4.into()),
                            ]),
                            debug_name: None,
                        },
                    ),
                ]),
                debug_name: None,
            },
        );
    }

    /// Ensure that a dictionary of unsigned integers compiles.
    #[test]
    fn dict_type_unsigned() {
        // fn run_program() -> Felt252Dict<u128> {
        //     let mut x: Felt252Dict<u128> = Default::default();
        //     x.insert(0, 0_u128);
        //     x.insert(1, 1_u128);
        //     x.insert(2, 2_u128);
        //     x.insert(3, 3_u128);
        //     x
        // }
        let program = ProgramParser::new().parse(r#"
            type [0] = SegmentArena [storable: true, drop: false, dup: false, zero_sized: false];
            type [12] = Const<[1], 3> [storable: false, drop: false, dup: false, zero_sized: false];
            type [11] = Const<[3], 3> [storable: false, drop: false, dup: false, zero_sized: false];
            type [10] = Const<[1], 2> [storable: false, drop: false, dup: false, zero_sized: false];
            type [9] = Const<[3], 2> [storable: false, drop: false, dup: false, zero_sized: false];
            type [8] = Const<[1], 1> [storable: false, drop: false, dup: false, zero_sized: false];
            type [7] = Const<[3], 1> [storable: false, drop: false, dup: false, zero_sized: false];
            type [6] = Const<[1], 0> [storable: false, drop: false, dup: false, zero_sized: false];
            type [4] = Felt252DictEntry<[1]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [5] = Const<[3], 0> [storable: false, drop: false, dup: false, zero_sized: false];
            type [3] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = u128 [storable: true, drop: true, dup: true, zero_sized: false];
            type [2] = Felt252Dict<[1]> [storable: true, drop: false, dup: false, zero_sized: false];

            libfunc [2] = felt252_dict_new<[1]>;
            libfunc [4] = const_as_immediate<[5]>;
            libfunc [13] = store_temp<[2]>;
            libfunc [14] = store_temp<[3]>;
            libfunc [1] = felt252_dict_entry_get<[1]>;
            libfunc [5] = drop<[1]>;
            libfunc [6] = const_as_immediate<[6]>;
            libfunc [15] = store_temp<[1]>;
            libfunc [0] = felt252_dict_entry_finalize<[1]>;
            libfunc [7] = const_as_immediate<[7]>;
            libfunc [8] = const_as_immediate<[8]>;
            libfunc [9] = const_as_immediate<[9]>;
            libfunc [10] = const_as_immediate<[10]>;
            libfunc [11] = const_as_immediate<[11]>;
            libfunc [12] = const_as_immediate<[12]>;
            libfunc [16] = store_temp<[0]>;

            [2]([0]) -> ([1], [2]); // 0
            [4]() -> ([3]); // 1
            [13]([2]) -> ([2]); // 2
            [14]([3]) -> ([3]); // 3
            [1]([2], [3]) -> ([4], [5]); // 4
            [5]([5]) -> (); // 5
            [6]() -> ([6]); // 6
            [15]([6]) -> ([6]); // 7
            [0]([4], [6]) -> ([7]); // 8
            [7]() -> ([8]); // 9
            [14]([8]) -> ([8]); // 10
            [1]([7], [8]) -> ([9], [10]); // 11
            [5]([10]) -> (); // 12
            [8]() -> ([11]); // 13
            [15]([11]) -> ([11]); // 14
            [0]([9], [11]) -> ([12]); // 15
            [9]() -> ([13]); // 16
            [14]([13]) -> ([13]); // 17
            [1]([12], [13]) -> ([14], [15]); // 18
            [5]([15]) -> (); // 19
            [10]() -> ([16]); // 20
            [15]([16]) -> ([16]); // 21
            [0]([14], [16]) -> ([17]); // 22
            [11]() -> ([18]); // 23
            [14]([18]) -> ([18]); // 24
            [1]([17], [18]) -> ([19], [20]); // 25
            [5]([20]) -> (); // 26
            [12]() -> ([21]); // 27
            [15]([21]) -> ([21]); // 28
            [0]([19], [21]) -> ([22]); // 29
            [16]([1]) -> ([1]); // 30
            [13]([22]) -> ([22]); // 31
            return([1], [22]); // 32

            [0]@0([0]: [0]) -> ([0], [2]);
        "#).map_err(|e| e.to_string()).unwrap();

        let result = run_sierra_program(program, &[]);
        assert_eq!(
            result.return_value,
            Value::Felt252Dict {
                value: HashMap::from([
                    (Felt::ZERO, Value::Uint128(0)),
                    (Felt::ONE, Value::Uint128(1)),
                    (Felt::TWO, Value::Uint128(2)),
                    (Felt::THREE, Value::Uint128(3)),
                ]),
                debug_name: None,
            },
        );
    }
}
