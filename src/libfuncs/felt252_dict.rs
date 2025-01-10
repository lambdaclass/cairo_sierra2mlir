//! # `Felt` dictionary libfuncs

use super::LibfuncHelper;
use crate::{
    error::Result,
    metadata::{
        felt252_dict::Felt252DictOverrides, runtime_bindings::RuntimeBindingsMeta, MetadataStorage,
    },
    native_panic,
    types::TypeBuilder,
    utils::BlockExt,
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        felt252_dict::Felt252DictConcreteLibfunc,
        lib_func::SignatureOnlyConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{llvm, ods},
    ir::{Block, Location},
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
    selector: &Felt252DictConcreteLibfunc,
) -> Result<()> {
    match selector {
        Felt252DictConcreteLibfunc::New(info) => {
            build_new(context, registry, entry, location, helper, metadata, info)
        }
        Felt252DictConcreteLibfunc::Squash(info) => {
            build_squash(context, registry, entry, location, helper, metadata, info)
        }
    }
}

pub fn build_new<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let segment_arena = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let value_type_id = match registry.get_type(&info.signature.branch_signatures[0].vars[1].ty)? {
        CoreTypeConcrete::Felt252Dict(info) => &info.ty,
        _ => native_panic!("entered unreachable code"),
    };

    let (dup_fn, drop_fn) = {
        let mut dict_overrides = metadata
            .remove::<Felt252DictOverrides>()
            .unwrap_or_default();

        let dup_fn = match dict_overrides.build_dup_fn(
            context,
            helper,
            registry,
            metadata,
            value_type_id,
        )? {
            Some(dup_fn) => Some(
                entry.append_op_result(
                    ods::llvm::mlir_addressof(
                        context,
                        llvm::r#type::pointer(context, 0),
                        dup_fn,
                        location,
                    )
                    .into(),
                )?,
            ),
            None => None,
        };
        let drop_fn = match dict_overrides.build_drop_fn(
            context,
            helper,
            registry,
            metadata,
            value_type_id,
        )? {
            Some(drop_fn_symbol) => Some(
                entry.append_op_result(
                    ods::llvm::mlir_addressof(
                        context,
                        llvm::r#type::pointer(context, 0),
                        drop_fn_symbol,
                        location,
                    )
                    .into(),
                )?,
            ),
            None => None,
        };

        metadata.insert(dict_overrides);
        (dup_fn, drop_fn)
    };

    let runtime_bindings = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.");
    let dict_ptr = runtime_bindings.dict_new(
        context,
        helper,
        entry,
        location,
        dup_fn,
        drop_fn,
        registry.get_type(value_type_id)?.layout(registry)?,
    )?;

    entry.append_operation(helper.br(0, &[segment_arena, dict_ptr], location));
    Ok(())
}

pub fn build_squash<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;
    let gas_builtin = entry.arg(1)?;
    let segment_arena = super::increment_builtin_counter(context, entry, location, entry.arg(2)?)?;
    let dict_ptr = entry.arg(3)?;

    let runtime_bindings = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.");

    let gas_refund = runtime_bindings
        .dict_gas_refund(context, helper, entry, dict_ptr, location)?
        .result(0)?
        .into();

    let new_gas_builtin = entry.addi(gas_builtin, gas_refund, location)?;

    entry.append_operation(helper.br(
        0,
        &[range_check, new_gas_builtin, segment_arena, entry.arg(3)?],
        location,
    ));

    Ok(())
}

#[cfg(test)]
mod test {
    use cairo_lang_sierra::ProgramParser;

    use crate::{
        utils::test::{jit_dict, jit_enum, jit_struct, run_sierra_program},
        values::Value,
    };

    #[test]
    fn run_dict_new() {
        // use traits::Default;
        // use dict::Felt252DictTrait;
        // fn run_test() {
        //     let mut _dict: Felt252Dict<u32> = Default::default();
        // }
        let program = ProgramParser::new().parse(r#"
            type [1] = SegmentArena [storable: true, drop: false, dup: false, zero_sized: false];
            type [6] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
            type [5] = SquashedFelt252Dict<[3]> [storable: true, drop: true, dup: false, zero_sized: false];
            type [2] = GasBuiltin [storable: true, drop: false, dup: false, zero_sized: false];
            type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
            type [3] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
            type [4] = Felt252Dict<[3]> [storable: true, drop: false, dup: false, zero_sized: false];

            libfunc [3] = disable_ap_tracking;
            libfunc [8] = felt252_dict_new<[3]>;
            libfunc [4] = store_temp<[0]>;
            libfunc [5] = store_temp<[1]>;
            libfunc [6] = store_temp<[2]>;
            libfunc [10] = store_temp<[4]>;
            libfunc [0] = function_call<user@[0]>;
            libfunc [9] = drop<[5]>;
            libfunc [1] = felt252_dict_squash<[3]>;
            libfunc [7] = store_temp<[5]>;

            [3]() -> (); // 0
            [8]([1]) -> ([3], [4]); // 1
            [4]([0]) -> ([0]); // 2
            [5]([3]) -> ([3]); // 3
            [6]([2]) -> ([2]); // 4
            [10]([4]) -> ([4]); // 5
            [0]([0], [3], [2], [4]) -> ([5], [6], [7], [8]); // 6
            [9]([8]) -> (); // 7
            [4]([5]) -> ([5]); // 8
            [5]([6]) -> ([6]); // 9
            [6]([7]) -> ([7]); // 10
            return([5], [6], [7]); // 11
            [3]() -> (); // 12
            [1]([0], [2], [1], [3]) -> ([4], [5], [6], [7]); // 13
            [4]([4]) -> ([4]); // 14
            [5]([6]) -> ([6]); // 15
            [6]([5]) -> ([5]); // 16
            [7]([7]) -> ([7]); // 17
            return([4], [6], [5], [7]); // 18

            [1]@0([0]: [0], [1]: [1], [2]: [2]) -> ([0], [1], [2]);
            [0]@12([0]: [0], [1]: [1], [2]: [2], [3]: [4]) -> ([0], [1], [2], [5]);
        "#).map_err(|e| e.to_string()).unwrap();

        let return_value = run_sierra_program(&program, &[]).return_value;

        assert_eq!(jit_struct!(), return_value)
    }

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
    fn run_dict_insert_ret_dict() {
        // use traits::Default;
        // use dict::Felt252DictTrait;
        // fn run_test() -> Felt252Dict<u32> {
        //     let mut dict: Felt252Dict<u32> = Default::default();
        //     dict.insert(1, 2_u32);
        //     dict.insert(2, 3_u32);
        //     dict.insert(3, 4_u32);
        //     dict.insert(4, 5_u32);
        //     dict.insert(5, 6_u32);
        //     dict
        // }
        let program = ProgramParser::new().parse(r#"
            type [0] = SegmentArena [storable: true, drop: false, dup: false, zero_sized: false];
            type [14] = Const<[1], 6> [storable: false, drop: false, dup: false, zero_sized: false];
            type [13] = Const<[3], 5> [storable: false, drop: false, dup: false, zero_sized: false];
            type [12] = Const<[1], 5> [storable: false, drop: false, dup: false, zero_sized: false];
            type [11] = Const<[3], 4> [storable: false, drop: false, dup: false, zero_sized: false];
            type [10] = Const<[1], 4> [storable: false, drop: false, dup: false, zero_sized: false];
            type [9] = Const<[3], 3> [storable: false, drop: false, dup: false, zero_sized: false];
            type [8] = Const<[1], 3> [storable: false, drop: false, dup: false, zero_sized: false];
            type [7] = Const<[3], 2> [storable: false, drop: false, dup: false, zero_sized: false];
            type [6] = Const<[1], 2> [storable: false, drop: false, dup: false, zero_sized: false];
            type [4] = Felt252DictEntry<[1]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [5] = Const<[3], 1> [storable: false, drop: false, dup: false, zero_sized: false];
            type [3] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
            type [2] = Felt252Dict<[1]> [storable: true, drop: false, dup: false, zero_sized: false];

            libfunc [2] = felt252_dict_new<[1]>;
            libfunc [4] = const_as_immediate<[5]>;
            libfunc [15] = store_temp<[2]>;
            libfunc [16] = store_temp<[3]>;
            libfunc [1] = felt252_dict_entry_get<[1]>;
            libfunc [5] = drop<[1]>;
            libfunc [6] = const_as_immediate<[6]>;
            libfunc [17] = store_temp<[1]>;
            libfunc [0] = felt252_dict_entry_finalize<[1]>;
            libfunc [7] = const_as_immediate<[7]>;
            libfunc [8] = const_as_immediate<[8]>;
            libfunc [9] = const_as_immediate<[9]>;
            libfunc [10] = const_as_immediate<[10]>;
            libfunc [11] = const_as_immediate<[11]>;
            libfunc [12] = const_as_immediate<[12]>;
            libfunc [13] = const_as_immediate<[13]>;
            libfunc [14] = const_as_immediate<[14]>;
            libfunc [18] = store_temp<[0]>;

            [2]([0]) -> ([1], [2]); // 0
            [4]() -> ([3]); // 1
            [15]([2]) -> ([2]); // 2
            [16]([3]) -> ([3]); // 3
            [1]([2], [3]) -> ([4], [5]); // 4
            [5]([5]) -> (); // 5
            [6]() -> ([6]); // 6
            [17]([6]) -> ([6]); // 7
            [0]([4], [6]) -> ([7]); // 8
            [7]() -> ([8]); // 9
            [16]([8]) -> ([8]); // 10
            [1]([7], [8]) -> ([9], [10]); // 11
            [5]([10]) -> (); // 12
            [8]() -> ([11]); // 13
            [17]([11]) -> ([11]); // 14
            [0]([9], [11]) -> ([12]); // 15
            [9]() -> ([13]); // 16
            [16]([13]) -> ([13]); // 17
            [1]([12], [13]) -> ([14], [15]); // 18
            [5]([15]) -> (); // 19
            [10]() -> ([16]); // 20
            [17]([16]) -> ([16]); // 21
            [0]([14], [16]) -> ([17]); // 22
            [11]() -> ([18]); // 23
            [16]([18]) -> ([18]); // 24
            [1]([17], [18]) -> ([19], [20]); // 25
            [5]([20]) -> (); // 26
            [12]() -> ([21]); // 27
            [17]([21]) -> ([21]); // 28
            [0]([19], [21]) -> ([22]); // 29
            [13]() -> ([23]); // 30
            [16]([23]) -> ([23]); // 31
            [1]([22], [23]) -> ([24], [25]); // 32
            [5]([25]) -> (); // 33
            [14]() -> ([26]); // 34
            [17]([26]) -> ([26]); // 35
            [0]([24], [26]) -> ([27]); // 36
            [18]([1]) -> ([1]); // 37
            [15]([27]) -> ([27]); // 38
            return([1], [27]); // 39

            [0]@0([0]: [0]) -> ([0], [2]);
        "#).map_err(|e| e.to_string()).unwrap();

        let return_value = run_sierra_program(&program, &[]).return_value;

        assert_eq!(
            jit_dict!(
                1 => 2u32,
                2 => 3u32,
                3 => 4u32,
                4 => 5u32,
                5 => 6u32,
            ),
            return_value
        );
    }

    #[test]
    fn run_dict_deserialize() {
        // use traits::Default;
        // use dict::Felt252DictTrait;
        // fn run_test(mut dict: Felt252Dict<u32>) -> Felt252Dict<u32> {
        //     dict
        // }
        let program = ProgramParser::new().parse(r#"
            type [1] = Felt252Dict<[0]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [0] = u32 [storable: true, drop: true, dup: true, zero_sized: false];

            libfunc [0] = store_temp<[1]>;

            [0]([0]) -> ([0]); // 0
            return([0]); // 1

            [0]@0([0]: [1]) -> ([1]);
        "#).map_err(|e| e.to_string()).unwrap();

        let return_value = run_sierra_program(
            &program,
            &[jit_dict!(
                1 => 2u32,
                2 => 3u32,
                3 => 4u32,
                4 => 5u32,
                5 => 6u32,
            )],
        )
        .return_value;

        assert_eq!(
            jit_dict!(
                1 => 2u32,
                2 => 3u32,
                3 => 4u32,
                4 => 5u32,
                5 => 6u32,
            ),
            return_value
        );
    }

    #[test]
    fn run_dict_deserialize2() {
        // use traits::Default;
        // use dict::Felt252DictTrait;
        // fn run_test(mut dict: Felt252Dict<u32>) -> (felt252, Felt252Dict<u32>) {
        //     (0, dict)
        // }
        let program = ProgramParser::new().parse(r#"
            type [2] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = Felt252Dict<[0]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [3] = Struct<ut@Tuple, [2], [1]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [0] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
            type [4] = Const<[2], 0> [storable: false, drop: false, dup: false, zero_sized: false];

            libfunc [1] = const_as_immediate<[4]>;
            libfunc [0] = struct_construct<[3]>;
            libfunc [2] = store_temp<[3]>;

            [1]() -> ([1]); // 0
            [0]([1], [0]) -> ([2]); // 1
            [2]([2]) -> ([2]); // 2
            return([2]); // 3

            [0]@0([0]: [1]) -> ([3]);
        "#).map_err(|e| e.to_string()).unwrap();

        let return_value = run_sierra_program(
            &program,
            &[jit_dict!(
                1 => 2u32,
                2 => 3u32,
                3 => 4u32,
                4 => 5u32,
                5 => 6u32,
            )],
        )
        .return_value;

        assert_eq!(
            jit_struct!(
                Value::Felt252(0.into()),
                jit_dict!(
                    1 => 2u32,
                    2 => 3u32,
                    3 => 4u32,
                    4 => 5u32,
                    5 => 6u32,
                )
            ),
            return_value
        );
    }

    #[test]
    fn run_dict_deserialize_struct() {
        // use core::{dict::Felt252DictTrait, nullable::Nullable};
        // fn run_test() -> Felt252Dict<Nullable<(u32, u64, u128)>> {
        //     let mut x: Felt252Dict<Nullable<(u32, u64, u128)>> = Default::default();
        //     x.insert(0, NullableTrait::new((1_u32, 2_u64, 3_u128)));
        //     x.insert(1, NullableTrait::new((2_u32, 3_u64, 4_u128)));
        //     x.insert(2, NullableTrait::new((3_u32, 4_u64, 5_u128)));
        //     x
        // }
        let program = ProgramParser::new().parse(r#"
            type [0] = SegmentArena [storable: true, drop: false, dup: false, zero_sized: false];
            type [24] = Const<[8], 2> [storable: false, drop: false, dup: false, zero_sized: false];
            type [23] = Const<[4], [20], [21], [22]> [storable: false, drop: false, dup: false, zero_sized: false];
            type [19] = Const<[8], 1> [storable: false, drop: false, dup: false, zero_sized: false];
            type [22] = Const<[3], 5> [storable: false, drop: false, dup: false, zero_sized: false];
            type [21] = Const<[2], 4> [storable: false, drop: false, dup: false, zero_sized: false];
            type [3] = u128 [storable: true, drop: true, dup: true, zero_sized: false];
            type [2] = u64 [storable: true, drop: true, dup: true, zero_sized: false];
            type [20] = Const<[1], 3> [storable: false, drop: false, dup: false, zero_sized: false];
            type [18] = Const<[4], [15], [16], [17]> [storable: false, drop: false, dup: false, zero_sized: false];
            type [1] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
            type [17] = Const<[3], 4> [storable: false, drop: false, dup: false, zero_sized: false];
            type [16] = Const<[2], 3> [storable: false, drop: false, dup: false, zero_sized: false];
            type [15] = Const<[1], 2> [storable: false, drop: false, dup: false, zero_sized: false];
            type [9] = Felt252DictEntry<[5]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [14] = Const<[8], 0> [storable: false, drop: false, dup: false, zero_sized: false];
            type [8] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [4] = Struct<ut@Tuple, [1], [2], [3]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [13] = Const<[4], [10], [11], [12]> [storable: false, drop: false, dup: false, zero_sized: false];
            type [7] = Box<[4]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [12] = Const<[3], 3> [storable: false, drop: false, dup: false, zero_sized: false];
            type [11] = Const<[2], 2> [storable: false, drop: false, dup: false, zero_sized: false];
            type [10] = Const<[1], 1> [storable: false, drop: false, dup: false, zero_sized: false];
            type [5] = Nullable<[4]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [6] = Felt252Dict<[5]> [storable: true, drop: false, dup: false, zero_sized: false];

            libfunc [3] = felt252_dict_new<[5]>;
            libfunc [5] = const_as_box<[13], 0>;
            libfunc [2] = nullable_from_box<[4]>;
            libfunc [6] = const_as_immediate<[14]>;
            libfunc [12] = store_temp<[6]>;
            libfunc [13] = store_temp<[8]>;
            libfunc [1] = felt252_dict_entry_get<[5]>;
            libfunc [7] = drop<[5]>;
            libfunc [0] = felt252_dict_entry_finalize<[5]>;
            libfunc [8] = const_as_box<[18], 0>;
            libfunc [9] = const_as_immediate<[19]>;
            libfunc [10] = const_as_box<[23], 0>;
            libfunc [11] = const_as_immediate<[24]>;
            libfunc [14] = store_temp<[0]>;

            [3]([0]) -> ([1], [2]); // 0
            [5]() -> ([3]); // 1
            [2]([3]) -> ([4]); // 2
            [6]() -> ([5]); // 3
            [12]([2]) -> ([2]); // 4
            [13]([5]) -> ([5]); // 5
            [1]([2], [5]) -> ([6], [7]); // 6
            [7]([7]) -> (); // 7
            [0]([6], [4]) -> ([8]); // 8
            [8]() -> ([9]); // 9
            [2]([9]) -> ([10]); // 10
            [9]() -> ([11]); // 11
            [13]([11]) -> ([11]); // 12
            [1]([8], [11]) -> ([12], [13]); // 13
            [7]([13]) -> (); // 14
            [0]([12], [10]) -> ([14]); // 15
            [10]() -> ([15]); // 16
            [2]([15]) -> ([16]); // 17
            [11]() -> ([17]); // 18
            [13]([17]) -> ([17]); // 19
            [1]([14], [17]) -> ([18], [19]); // 20
            [7]([19]) -> (); // 21
            [0]([18], [16]) -> ([20]); // 22
            [14]([1]) -> ([1]); // 23
            [12]([20]) -> ([20]); // 24
            return([1], [20]); // 25

            [0]@0([0]: [0]) -> ([0], [6]);
        "#).map_err(|e| e.to_string()).unwrap();

        let return_value = run_sierra_program(&program, &[]).return_value;

        assert_eq!(
            jit_dict!(
                0 => jit_struct!(1u32.into(), 2u64.into(), 3u128.into()),
                1 => jit_struct!(2u32.into(), 3u64.into(), 4u128.into()),
                2 => jit_struct!(3u32.into(), 4u64.into(), 5u128.into()),
            ),
            return_value
        );
    }

    #[test]
    fn run_dict_deserialize_enum() {
        // use core::{dict::Felt252DictTrait, nullable::Nullable};
        // #[derive(Drop)]
        // enum MyEnum {
        //     A: u32,
        //     B: u64,
        //     C: u128,
        // }
        // fn run_test() -> Felt252Dict<Nullable<MyEnum>> {
        //     let mut x: Felt252Dict<Nullable<MyEnum>> = Default::default();
        //     x.insert(0, NullableTrait::new(MyEnum::A(1)));
        //     x.insert(1, NullableTrait::new(MyEnum::B(2)));
        //     x.insert(2, NullableTrait::new(MyEnum::C(3)));
        //     x
        // }
        let program = ProgramParser::new().parse(r#"
            type [0] = SegmentArena [storable: true, drop: false, dup: false, zero_sized: false];
            type [18] = Const<[8], 2> [storable: false, drop: false, dup: false, zero_sized: false];
            type [17] = Const<[4], 2, [16]> [storable: false, drop: false, dup: false, zero_sized: false];
            type [15] = Const<[8], 1> [storable: false, drop: false, dup: false, zero_sized: false];
            type [16] = Const<[3], 3> [storable: false, drop: false, dup: false, zero_sized: false];
            type [14] = Const<[4], 1, [13]> [storable: false, drop: false, dup: false, zero_sized: false];
            type [3] = u128 [storable: true, drop: true, dup: true, zero_sized: false];
            type [13] = Const<[2], 2> [storable: false, drop: false, dup: false, zero_sized: false];
            type [9] = Felt252DictEntry<[5]> [storable: true, drop: false, dup: false, zero_sized: false];
            type [2] = u64 [storable: true, drop: true, dup: true, zero_sized: false];
            type [12] = Const<[8], 0> [storable: false, drop: false, dup: false, zero_sized: false];
            type [8] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
            type [4] = Enum<ut@program::program::MyEnum, [1], [2], [3]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [11] = Const<[4], 0, [10]> [storable: false, drop: false, dup: false, zero_sized: false];
            type [7] = Box<[4]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [10] = Const<[1], 1> [storable: false, drop: false, dup: false, zero_sized: false];
            type [5] = Nullable<[4]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [6] = Felt252Dict<[5]> [storable: true, drop: false, dup: false, zero_sized: false];

            libfunc [3] = felt252_dict_new<[5]>;
            libfunc [5] = const_as_box<[11], 0>;
            libfunc [2] = nullable_from_box<[4]>;
            libfunc [6] = const_as_immediate<[12]>;
            libfunc [12] = store_temp<[6]>;
            libfunc [13] = store_temp<[8]>;
            libfunc [1] = felt252_dict_entry_get<[5]>;
            libfunc [7] = drop<[5]>;
            libfunc [0] = felt252_dict_entry_finalize<[5]>;
            libfunc [8] = const_as_box<[14], 0>;
            libfunc [9] = const_as_immediate<[15]>;
            libfunc [10] = const_as_box<[17], 0>;
            libfunc [11] = const_as_immediate<[18]>;
            libfunc [14] = store_temp<[0]>;

            [3]([0]) -> ([1], [2]); // 0
            [5]() -> ([3]); // 1
            [2]([3]) -> ([4]); // 2
            [6]() -> ([5]); // 3
            [12]([2]) -> ([2]); // 4
            [13]([5]) -> ([5]); // 5
            [1]([2], [5]) -> ([6], [7]); // 6
            [7]([7]) -> (); // 7
            [0]([6], [4]) -> ([8]); // 8
            [8]() -> ([9]); // 9
            [2]([9]) -> ([10]); // 10
            [9]() -> ([11]); // 11
            [13]([11]) -> ([11]); // 12
            [1]([8], [11]) -> ([12], [13]); // 13
            [7]([13]) -> (); // 14
            [0]([12], [10]) -> ([14]); // 15
            [10]() -> ([15]); // 16
            [2]([15]) -> ([16]); // 17
            [11]() -> ([17]); // 18
            [13]([17]) -> ([17]); // 19
            [1]([14], [17]) -> ([18], [19]); // 20
            [7]([19]) -> (); // 21
            [0]([18], [16]) -> ([20]); // 22
            [14]([1]) -> ([1]); // 23
            [12]([20]) -> ([20]); // 24
            return([1], [20]); // 25

            [0]@0([0]: [0]) -> ([0], [6]);
        "#).map_err(|e| e.to_string()).unwrap();

        let return_value = run_sierra_program(&program, &[]).return_value;

        assert_eq!(
            jit_dict!(
                0 => jit_enum!(0, 1u32.into()),
                1 => jit_enum!(1, 2u64.into()),
                2 => jit_enum!(2, 3u128.into()),
            ),
            return_value
        );
    }
}
