//! # `Felt` dictionary entry libfuncs

use super::LibfuncHelper;
use crate::{
    error::{Error, Result},
    metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
    types::TypeBuilder,
    utils::{BlockExt, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        felt252_dict::Felt252DictEntryConcreteLibfunc,
        lib_func::SignatureAndTypeConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{llvm, scf},
    ir::{r#type::IntegerType, Block, Location, Region},
    Context,
};
use std::cell::Cell;

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &Felt252DictEntryConcreteLibfunc,
) -> Result<()> {
    match selector {
        Felt252DictEntryConcreteLibfunc::Get(info) => {
            build_get(context, registry, entry, location, helper, metadata, info)
        }
        Felt252DictEntryConcreteLibfunc::Finalize(info) => {
            build_finalize(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// The felt252_dict_entry_get libfunc receives the dictionary and the key and
/// returns the associated dict entry, along with it's value.
///
/// The dict entry also contains a pointer to the dictionary.
///
/// If the key doesn't yet exist, it is created and the type's default value is returned.
///
/// # Cairo Signature
///
/// ```cairo
/// fn felt252_dict_entry_get<T>(dict: Felt252Dict<T>, key: felt252) -> (Felt252DictEntry<T>, T) nopanic;
/// ```
pub fn build_get<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let (key_ty, key_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.param_signatures()[1].ty,
    )?;
    let entry_ty = registry.build_type(
        context,
        helper,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;
    let concrete_value_type = registry.get_type(&info.ty)?;
    let value_ty = concrete_value_type.build(context, helper, registry, metadata, &info.ty)?;

    let dict_ptr = entry.arg(0)?;
    let entry_key = entry.arg(1)?;

    let entry_key_ptr =
        helper
            .init_block()
            .alloca1(context, location, key_ty, key_layout.align())?;
    entry.store(context, location, entry_key_ptr, entry_key)?;

    let (is_present, value_ptr) = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .ok_or(Error::MissingMetadata)?
        .dict_get(context, helper, entry, dict_ptr, entry_key_ptr, location)?;
    let is_present = entry.trunci(is_present, IntegerType::new(context, 1).into(), location)?;

    let value = entry.append_op_result(scf::r#if(
        is_present,
        &[value_ty],
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            // If the entry is present we can load the current value.
            let value = block.load(context, location, value_ptr, value_ty)?;

            block.append_operation(scf::r#yield(&[value], location));
            region
        },
        {
            let region = Region::new();
            let block = region.append_block(Block::new(&[]));

            let helper = LibfuncHelper {
                module: helper.module,
                init_block: helper.init_block,
                region: &region,
                blocks_arena: helper.blocks_arena,
                last_block: Cell::new(&block),
                branches: Vec::new(),
                results: Vec::new(),
            };

            // When the entry is vacant we need to create the default value.
            let value = concrete_value_type.build_default(
                context, registry, &block, location, &helper, metadata, &info.ty,
            )?;

            block.append_operation(scf::r#yield(&[value], location));
            region
        },
        location,
    ))?;

    let dict_entry = entry.append_op_result(llvm::undef(entry_ty, location))?;
    let dict_entry = entry.insert_values(context, location, dict_entry, &[dict_ptr, value_ptr])?;

    // The `Felt252DictEntry<T>` holds both the `Felt252Dict<T>` and the pointer to the space where
    // the new value will be written when the entry is finalized. If the entry were to be dropped
    // (without being consumed by the finalizer), which shouldn't be possible under normal
    // conditions, and the type `T` requires a custom drop implementation (ex. arrays, dicts...),
    // it'll cause undefined behavior because when the value is moved out of the dictionary (on
    // `get`), the memory it occupied is not modified because we're expecting it to be overwritten
    // by the finalizer (in other words, the extracted element will be dropped twice).

    entry.append_operation(helper.br(0, &[dict_entry, value], location));
    Ok(())
}

/// The felt252_dict_entry_finalize libfunc receives the dict entry and a new value,
/// inserts the new value in the entry, and returns the full dictionary.
///
/// # Cairo Signature
///
/// ```cairo
/// fn felt252_dict_entry_finalize<T>(dict_entry: Felt252DictEntry<T>, new_value: T) -> Felt252Dict<T> nopanic;
/// ```
pub fn build_finalize<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    // Get the dict entry struct: `crate::types::felt252_dict_entry`.
    let dict_entry = entry.arg(0)?;
    let new_value = entry.arg(1)?;

    let dict_ptr = entry.extract_value(
        context,
        location,
        dict_entry,
        llvm::r#type::pointer(context, 0),
        0,
    )?;
    let value_ptr = entry.extract_value(
        context,
        location,
        dict_entry,
        llvm::r#type::pointer(context, 0),
        1,
    )?;

    entry.store(context, location, value_ptr, new_value)?;

    entry.append_operation(helper.br(0, &[dict_ptr], location));
    Ok(())
}

#[cfg(test)]
mod test {
    use cairo_lang_sierra::ProgramParser;

    use crate::{
        utils::test::{jit_dict, run_sierra_program},
        Value,
    };

    #[test]
    fn run_dict_insert() {
        // use traits::Default;
        // use dict::Felt252DictTrait;
        // fn run_test() -> u32 {
        //     let mut dict: Felt252Dict<u32> = Default::default();
        //     dict.insert(2, 1_u32);
        //     dict.get(2)
        // }
        let program = ProgramParser::new().parse(r#"
            type [3] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
            type [10] = Uninitialized<[3]> [storable: false, drop: true, dup: false, zero_sized: false];
            type [7] = SquashedFelt252Dict<[3]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [2] = GasBuiltin [storable: true, drop: false, dup: false, zero_sized: false];
            type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
            type [9] = Const<[3], 1> [storable: false, drop: false, dup: false, zero_sized: false];
            type [6] = Felt252DictEntry<[3]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [8] = Const<[5], 2> [storable: false, drop: false, dup: false, zero_sized: false];
            type [5] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [4] = Felt252Dict<[3]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [1] = SegmentArena [storable: true, drop: false, dup: false, zero_sized: false];

            libfunc [11] = alloc_local<[3]>;
            libfunc [12] = finalize_locals;
            libfunc [3] = disable_ap_tracking;
            libfunc [10] = felt252_dict_new<[3]>;
            libfunc [13] = const_as_immediate<[8]>;
            libfunc [18] = store_temp<[4]>;
            libfunc [19] = store_temp<[5]>;
            libfunc [9] = felt252_dict_entry_get<[3]>;
            libfunc [14] = drop<[3]>;
            libfunc [15] = const_as_immediate<[9]>;
            libfunc [20] = store_temp<[3]>;
            libfunc [8] = felt252_dict_entry_finalize<[3]>;
            libfunc [21] = store_local<[3]>;
            libfunc [16] = dup<[3]>;
            libfunc [4] = store_temp<[0]>;
            libfunc [5] = store_temp<[1]>;
            libfunc [6] = store_temp<[2]>;
            libfunc [0] = function_call<user@[0]>;
            libfunc [17] = drop<[7]>;
            libfunc [1] = felt252_dict_squash<[3]>;
            libfunc [7] = store_temp<[7]>;

            [11]() -> ([4]); // 0
            [12]() -> (); // 1
            [3]() -> (); // 2
            [10]([1]) -> ([5], [6]); // 3
            [13]() -> ([7]); // 4
            [18]([6]) -> ([6]); // 5
            [19]([7]) -> ([7]); // 6
            [9]([6], [7]) -> ([8], [9]); // 7
            [14]([9]) -> (); // 8
            [15]() -> ([10]); // 9
            [20]([10]) -> ([10]); // 10
            [8]([8], [10]) -> ([11]); // 11
            [13]() -> ([12]); // 12
            [19]([12]) -> ([12]); // 13
            [9]([11], [12]) -> ([13], [3]); // 14
            [21]([4], [3]) -> ([3]); // 15
            [16]([3]) -> ([3], [14]); // 16
            [8]([13], [14]) -> ([15]); // 17
            [4]([0]) -> ([0]); // 18
            [5]([5]) -> ([5]); // 19
            [6]([2]) -> ([2]); // 20
            [18]([15]) -> ([15]); // 21
            [0]([0], [5], [2], [15]) -> ([16], [17], [18], [19]); // 22
            [17]([19]) -> (); // 23
            [4]([16]) -> ([16]); // 24
            [5]([17]) -> ([17]); // 25
            [6]([18]) -> ([18]); // 26
            [20]([3]) -> ([3]); // 27
            return([16], [17], [18], [3]); // 28
            [3]() -> (); // 29
            [1]([0], [2], [1], [3]) -> ([4], [5], [6], [7]); // 30
            [4]([4]) -> ([4]); // 31
            [5]([6]) -> ([6]); // 32
            [6]([5]) -> ([5]); // 33
            [7]([7]) -> ([7]); // 34
            return([4], [6], [5], [7]); // 35

            [1]@0([0]: [0], [1]: [1], [2]: [2]) -> ([0], [1], [2], [3]);
            [0]@29([0]: [0], [1]: [1], [2]: [2], [3]: [4]) -> ([0], [1], [2], [7]);
        "#).map_err(|e| e.to_string()).unwrap();

        let return_value = run_sierra_program(&program, &[]).return_value;

        assert_eq!(Value::from(1u32), return_value);
    }

    #[test]
    fn run_dict_insert_big() {
        // use traits::Default;
        // use dict::Felt252DictTrait;
        // fn run_test() -> u64 {
        //     let mut dict: Felt252Dict<u64> = Default::default();
        //     dict.insert(200000000, 4_u64);
        //     dict.get(200000000)
        // }
        let program = ProgramParser::new().parse(r#"
            type [3] = u64 [storable: true, drop: true, dup: true, zero_sized: false];
            type [10] = Uninitialized<[3]> [storable: false, drop: true, dup: false, zero_sized: false];
            type [7] = SquashedFelt252Dict<[3]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [2] = GasBuiltin [storable: true, drop: false, dup: false, zero_sized: false];
            type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
            type [9] = Const<[3], 4> [storable: false, drop: false, dup: false, zero_sized: false];
            type [6] = Felt252DictEntry<[3]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [8] = Const<[5], 200000000> [storable: false, drop: false, dup: false, zero_sized: false];
            type [5] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [4] = Felt252Dict<[3]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [1] = SegmentArena [storable: true, drop: false, dup: false, zero_sized: false];

            libfunc [11] = alloc_local<[3]>;
            libfunc [12] = finalize_locals;
            libfunc [3] = disable_ap_tracking;
            libfunc [10] = felt252_dict_new<[3]>;
            libfunc [13] = const_as_immediate<[8]>;
            libfunc [18] = store_temp<[4]>;
            libfunc [19] = store_temp<[5]>;
            libfunc [9] = felt252_dict_entry_get<[3]>;
            libfunc [14] = drop<[3]>;
            libfunc [15] = const_as_immediate<[9]>;
            libfunc [20] = store_temp<[3]>;
            libfunc [8] = felt252_dict_entry_finalize<[3]>;
            libfunc [21] = store_local<[3]>;
            libfunc [16] = dup<[3]>;
            libfunc [4] = store_temp<[0]>;
            libfunc [5] = store_temp<[1]>;
            libfunc [6] = store_temp<[2]>;
            libfunc [0] = function_call<user@[0]>;
            libfunc [17] = drop<[7]>;
            libfunc [1] = felt252_dict_squash<[3]>;
            libfunc [7] = store_temp<[7]>;

            [11]() -> ([4]); // 0
            [12]() -> (); // 1
            [3]() -> (); // 2
            [10]([1]) -> ([5], [6]); // 3
            [13]() -> ([7]); // 4
            [18]([6]) -> ([6]); // 5
            [19]([7]) -> ([7]); // 6
            [9]([6], [7]) -> ([8], [9]); // 7
            [14]([9]) -> (); // 8
            [15]() -> ([10]); // 9
            [20]([10]) -> ([10]); // 10
            [8]([8], [10]) -> ([11]); // 11
            [13]() -> ([12]); // 12
            [19]([12]) -> ([12]); // 13
            [9]([11], [12]) -> ([13], [3]); // 14
            [21]([4], [3]) -> ([3]); // 15
            [16]([3]) -> ([3], [14]); // 16
            [8]([13], [14]) -> ([15]); // 17
            [4]([0]) -> ([0]); // 18
            [5]([5]) -> ([5]); // 19
            [6]([2]) -> ([2]); // 20
            [18]([15]) -> ([15]); // 21
            [0]([0], [5], [2], [15]) -> ([16], [17], [18], [19]); // 22
            [17]([19]) -> (); // 23
            [4]([16]) -> ([16]); // 24
            [5]([17]) -> ([17]); // 25
            [6]([18]) -> ([18]); // 26
            [20]([3]) -> ([3]); // 27
            return([16], [17], [18], [3]); // 28
            [3]() -> (); // 29
            [1]([0], [2], [1], [3]) -> ([4], [5], [6], [7]); // 30
            [4]([4]) -> ([4]); // 31
            [5]([6]) -> ([6]); // 32
            [6]([5]) -> ([5]); // 33
            [7]([7]) -> ([7]); // 34
            return([4], [6], [5], [7]); // 35

            [1]@0([0]: [0], [1]: [1], [2]: [2]) -> ([0], [1], [2], [3]);
            [0]@29([0]: [0], [1]: [1], [2]: [2], [3]: [4]) -> ([0], [1], [2], [7]);
        "#).map_err(|e| e.to_string()).unwrap();

        let return_value = run_sierra_program(&program, &[]).return_value;

        assert_eq!(Value::from(4u64), return_value);
    }

    #[test]
    fn run_dict_insert_ret_dict() {
        // use traits::Default;
        // use dict::Felt252DictTrait;
        // fn run_test() -> Felt252Dict<u32> {
        //     let mut dict: Felt252Dict<u32> = Default::default();
        //     dict.insert(2, 1_u32);
        //     dict
        // }
        let program = ProgramParser::new().parse(r#"
            type [0] = SegmentArena [storable: true, drop: false, dup: false, zero_sized: false];
            type [6] = Const<[1], 1> [storable: false, drop: false, dup: false, zero_sized: false];
            type [4] = Felt252DictEntry<[1]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [5] = Const<[3], 2> [storable: false, drop: false, dup: false, zero_sized: false];
            type [3] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
            type [2] = Felt252Dict<[1]> [storable: true, drop: false, dup: false, zero_sized: false];

            libfunc [2] = felt252_dict_new<[1]>;
            libfunc [4] = const_as_immediate<[5]>;
            libfunc [7] = store_temp<[2]>;
            libfunc [8] = store_temp<[3]>;
            libfunc [1] = felt252_dict_entry_get<[1]>;
            libfunc [5] = drop<[1]>;
            libfunc [6] = const_as_immediate<[6]>;
            libfunc [9] = store_temp<[1]>;
            libfunc [0] = felt252_dict_entry_finalize<[1]>;
            libfunc [10] = store_temp<[0]>;

            [2]([0]) -> ([1], [2]); // 0
            [4]() -> ([3]); // 1
            [7]([2]) -> ([2]); // 2
            [8]([3]) -> ([3]); // 3
            [1]([2], [3]) -> ([4], [5]); // 4
            [5]([5]) -> (); // 5
            [6]() -> ([6]); // 6
            [9]([6]) -> ([6]); // 7
            [0]([4], [6]) -> ([7]); // 8
            [10]([1]) -> ([1]); // 9
            [7]([7]) -> ([7]); // 10
            return([1], [7]); // 11

            [0]@0([0]: [0]) -> ([0], [2]);
        "#).map_err(|e| e.to_string()).unwrap();

        let return_value = run_sierra_program(&program, &[]).return_value;

        assert_eq!(
            jit_dict!(
                2 => 1u32
            ),
            return_value
        );
    }

    #[test]
    fn run_dict_insert_multiple() {
        // use traits::Default;
        // use dict::Felt252DictTrait;
        // fn run_test() -> u32 {
        //     let mut dict: Felt252Dict<u32> = Default::default();
        //     dict.insert(2, 1_u32);
        //     dict.insert(3, 1_u32);
        //     dict.insert(4, 1_u32);
        //     dict.insert(5, 1_u32);
        //     dict.insert(6, 1_u32);
        //     dict.insert(7, 1_u32);
        //     dict.insert(8, 1_u32);
        //     dict.insert(9, 1_u32);
        //     dict.insert(10, 1_u32);
        //     dict.insert(11, 1_u32);
        //     dict.insert(12, 1_u32);
        //     dict.insert(13, 1_u32);
        //     dict.insert(14, 1_u32);
        //     dict.insert(15, 1_u32);
        //     dict.insert(16, 1_u32);
        //     dict.insert(17, 1_u32);
        //     dict.insert(18, 1345432_u32);
        //     dict.get(18)
        // }
        let program = ProgramParser::new().parse(r#"
            type [3] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
            type [27] = Uninitialized<[3]> [storable: false, drop: true, dup: false, zero_sized: false];
            type [7] = SquashedFelt252Dict<[3]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [2] = GasBuiltin [storable: true, drop: false, dup: false, zero_sized: false];
            type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
            type [26] = Const<[3], 1345432> [storable: false, drop: false, dup: false, zero_sized: false];
            type [25] = Const<[5], 18> [storable: false, drop: false, dup: false, zero_sized: false];
            type [24] = Const<[5], 17> [storable: false, drop: false, dup: false, zero_sized: false];
            type [23] = Const<[5], 16> [storable: false, drop: false, dup: false, zero_sized: false];
            type [22] = Const<[5], 15> [storable: false, drop: false, dup: false, zero_sized: false];
            type [21] = Const<[5], 14> [storable: false, drop: false, dup: false, zero_sized: false];
            type [20] = Const<[5], 13> [storable: false, drop: false, dup: false, zero_sized: false];
            type [19] = Const<[5], 12> [storable: false, drop: false, dup: false, zero_sized: false];
            type [18] = Const<[5], 11> [storable: false, drop: false, dup: false, zero_sized: false];
            type [17] = Const<[5], 10> [storable: false, drop: false, dup: false, zero_sized: false];
            type [16] = Const<[5], 9> [storable: false, drop: false, dup: false, zero_sized: false];
            type [15] = Const<[5], 8> [storable: false, drop: false, dup: false, zero_sized: false];
            type [14] = Const<[5], 7> [storable: false, drop: false, dup: false, zero_sized: false];
            type [13] = Const<[5], 6> [storable: false, drop: false, dup: false, zero_sized: false];
            type [12] = Const<[5], 5> [storable: false, drop: false, dup: false, zero_sized: false];
            type [11] = Const<[5], 4> [storable: false, drop: false, dup: false, zero_sized: false];
            type [10] = Const<[5], 3> [storable: false, drop: false, dup: false, zero_sized: false];
            type [9] = Const<[3], 1> [storable: false, drop: false, dup: false, zero_sized: false];
            type [6] = Felt252DictEntry<[3]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [8] = Const<[5], 2> [storable: false, drop: false, dup: false, zero_sized: false];
            type [5] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [4] = Felt252Dict<[3]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [1] = SegmentArena [storable: true, drop: false, dup: false, zero_sized: false];

            libfunc [11] = alloc_local<[3]>;
            libfunc [12] = finalize_locals;
            libfunc [3] = disable_ap_tracking;
            libfunc [10] = felt252_dict_new<[3]>;
            libfunc [13] = const_as_immediate<[8]>;
            libfunc [35] = store_temp<[4]>;
            libfunc [36] = store_temp<[5]>;
            libfunc [9] = felt252_dict_entry_get<[3]>;
            libfunc [14] = drop<[3]>;
            libfunc [15] = const_as_immediate<[9]>;
            libfunc [37] = store_temp<[3]>;
            libfunc [8] = felt252_dict_entry_finalize<[3]>;
            libfunc [16] = const_as_immediate<[10]>;
            libfunc [17] = const_as_immediate<[11]>;
            libfunc [18] = const_as_immediate<[12]>;
            libfunc [19] = const_as_immediate<[13]>;
            libfunc [20] = const_as_immediate<[14]>;
            libfunc [21] = const_as_immediate<[15]>;
            libfunc [22] = const_as_immediate<[16]>;
            libfunc [23] = const_as_immediate<[17]>;
            libfunc [24] = const_as_immediate<[18]>;
            libfunc [25] = const_as_immediate<[19]>;
            libfunc [26] = const_as_immediate<[20]>;
            libfunc [27] = const_as_immediate<[21]>;
            libfunc [28] = const_as_immediate<[22]>;
            libfunc [29] = const_as_immediate<[23]>;
            libfunc [30] = const_as_immediate<[24]>;
            libfunc [31] = const_as_immediate<[25]>;
            libfunc [32] = const_as_immediate<[26]>;
            libfunc [38] = store_local<[3]>;
            libfunc [33] = dup<[3]>;
            libfunc [4] = store_temp<[0]>;
            libfunc [5] = store_temp<[1]>;
            libfunc [6] = store_temp<[2]>;
            libfunc [0] = function_call<user@[0]>;
            libfunc [34] = drop<[7]>;
            libfunc [1] = felt252_dict_squash<[3]>;
            libfunc [7] = store_temp<[7]>;

            [11]() -> ([4]); // 0
            [12]() -> (); // 1
            [3]() -> (); // 2
            [10]([1]) -> ([5], [6]); // 3
            [13]() -> ([7]); // 4
            [35]([6]) -> ([6]); // 5
            [36]([7]) -> ([7]); // 6
            [9]([6], [7]) -> ([8], [9]); // 7
            [14]([9]) -> (); // 8
            [15]() -> ([10]); // 9
            [37]([10]) -> ([10]); // 10
            [8]([8], [10]) -> ([11]); // 11
            [16]() -> ([12]); // 12
            [36]([12]) -> ([12]); // 13
            [9]([11], [12]) -> ([13], [14]); // 14
            [14]([14]) -> (); // 15
            [15]() -> ([15]); // 16
            [37]([15]) -> ([15]); // 17
            [8]([13], [15]) -> ([16]); // 18
            [17]() -> ([17]); // 19
            [36]([17]) -> ([17]); // 20
            [9]([16], [17]) -> ([18], [19]); // 21
            [14]([19]) -> (); // 22
            [15]() -> ([20]); // 23
            [37]([20]) -> ([20]); // 24
            [8]([18], [20]) -> ([21]); // 25
            [18]() -> ([22]); // 26
            [36]([22]) -> ([22]); // 27
            [9]([21], [22]) -> ([23], [24]); // 28
            [14]([24]) -> (); // 29
            [15]() -> ([25]); // 30
            [37]([25]) -> ([25]); // 31
            [8]([23], [25]) -> ([26]); // 32
            [19]() -> ([27]); // 33
            [36]([27]) -> ([27]); // 34
            [9]([26], [27]) -> ([28], [29]); // 35
            [14]([29]) -> (); // 36
            [15]() -> ([30]); // 37
            [37]([30]) -> ([30]); // 38
            [8]([28], [30]) -> ([31]); // 39
            [20]() -> ([32]); // 40
            [36]([32]) -> ([32]); // 41
            [9]([31], [32]) -> ([33], [34]); // 42
            [14]([34]) -> (); // 43
            [15]() -> ([35]); // 44
            [37]([35]) -> ([35]); // 45
            [8]([33], [35]) -> ([36]); // 46
            [21]() -> ([37]); // 47
            [36]([37]) -> ([37]); // 48
            [9]([36], [37]) -> ([38], [39]); // 49
            [14]([39]) -> (); // 50
            [15]() -> ([40]); // 51
            [37]([40]) -> ([40]); // 52
            [8]([38], [40]) -> ([41]); // 53
            [22]() -> ([42]); // 54
            [36]([42]) -> ([42]); // 55
            [9]([41], [42]) -> ([43], [44]); // 56
            [14]([44]) -> (); // 57
            [15]() -> ([45]); // 58
            [37]([45]) -> ([45]); // 59
            [8]([43], [45]) -> ([46]); // 60
            [23]() -> ([47]); // 61
            [36]([47]) -> ([47]); // 62
            [9]([46], [47]) -> ([48], [49]); // 63
            [14]([49]) -> (); // 64
            [15]() -> ([50]); // 65
            [37]([50]) -> ([50]); // 66
            [8]([48], [50]) -> ([51]); // 67
            [24]() -> ([52]); // 68
            [36]([52]) -> ([52]); // 69
            [9]([51], [52]) -> ([53], [54]); // 70
            [14]([54]) -> (); // 71
            [15]() -> ([55]); // 72
            [37]([55]) -> ([55]); // 73
            [8]([53], [55]) -> ([56]); // 74
            [25]() -> ([57]); // 75
            [36]([57]) -> ([57]); // 76
            [9]([56], [57]) -> ([58], [59]); // 77
            [14]([59]) -> (); // 78
            [15]() -> ([60]); // 79
            [37]([60]) -> ([60]); // 80
            [8]([58], [60]) -> ([61]); // 81
            [26]() -> ([62]); // 82
            [36]([62]) -> ([62]); // 83
            [9]([61], [62]) -> ([63], [64]); // 84
            [14]([64]) -> (); // 85
            [15]() -> ([65]); // 86
            [37]([65]) -> ([65]); // 87
            [8]([63], [65]) -> ([66]); // 88
            [27]() -> ([67]); // 89
            [36]([67]) -> ([67]); // 90
            [9]([66], [67]) -> ([68], [69]); // 91
            [14]([69]) -> (); // 92
            [15]() -> ([70]); // 93
            [37]([70]) -> ([70]); // 94
            [8]([68], [70]) -> ([71]); // 95
            [28]() -> ([72]); // 96
            [36]([72]) -> ([72]); // 97
            [9]([71], [72]) -> ([73], [74]); // 98
            [14]([74]) -> (); // 99
            [15]() -> ([75]); // 100
            [37]([75]) -> ([75]); // 101
            [8]([73], [75]) -> ([76]); // 102
            [29]() -> ([77]); // 103
            [36]([77]) -> ([77]); // 104
            [9]([76], [77]) -> ([78], [79]); // 105
            [14]([79]) -> (); // 106
            [15]() -> ([80]); // 107
            [37]([80]) -> ([80]); // 108
            [8]([78], [80]) -> ([81]); // 109
            [30]() -> ([82]); // 110
            [36]([82]) -> ([82]); // 111
            [9]([81], [82]) -> ([83], [84]); // 112
            [14]([84]) -> (); // 113
            [15]() -> ([85]); // 114
            [37]([85]) -> ([85]); // 115
            [8]([83], [85]) -> ([86]); // 116
            [31]() -> ([87]); // 117
            [36]([87]) -> ([87]); // 118
            [9]([86], [87]) -> ([88], [89]); // 119
            [14]([89]) -> (); // 120
            [32]() -> ([90]); // 121
            [37]([90]) -> ([90]); // 122
            [8]([88], [90]) -> ([91]); // 123
            [31]() -> ([92]); // 124
            [36]([92]) -> ([92]); // 125
            [9]([91], [92]) -> ([93], [3]); // 126
            [38]([4], [3]) -> ([3]); // 127
            [33]([3]) -> ([3], [94]); // 128
            [8]([93], [94]) -> ([95]); // 129
            [4]([0]) -> ([0]); // 130
            [5]([5]) -> ([5]); // 131
            [6]([2]) -> ([2]); // 132
            [35]([95]) -> ([95]); // 133
            [0]([0], [5], [2], [95]) -> ([96], [97], [98], [99]); // 134
            [34]([99]) -> (); // 135
            [4]([96]) -> ([96]); // 136
            [5]([97]) -> ([97]); // 137
            [6]([98]) -> ([98]); // 138
            [37]([3]) -> ([3]); // 139
            return([96], [97], [98], [3]); // 140
            [3]() -> (); // 141
            [1]([0], [2], [1], [3]) -> ([4], [5], [6], [7]); // 142
            [4]([4]) -> ([4]); // 143
            [5]([6]) -> ([6]); // 144
            [6]([5]) -> ([5]); // 145
            [7]([7]) -> ([7]); // 146
            return([4], [6], [5], [7]); // 147

            [1]@0([0]: [0], [1]: [1], [2]: [2]) -> ([0], [1], [2], [3]);
            [0]@141([0]: [0], [1]: [1], [2]: [2], [3]: [4]) -> ([0], [1], [2], [7]);
        "#).map_err(|e| e.to_string()).unwrap();

        let return_value = run_sierra_program(&program, &[]).return_value;

        assert_eq!(Value::from(1345432_u32), return_value);
    }
}
