//! # Casting libfuncs

use super::LibfuncHelper;
use crate::{
    error::Result,
    metadata::MetadataStorage,
    native_panic,
    types::TypeBuilder,
    utils::{BlockExt, RangeExt, HALF_PRIME, PRIME},
};
use cairo_lang_sierra::{
    extensions::{
        casts::{CastConcreteLibfunc, DowncastConcreteLibfunc},
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
        utils::Range,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::arith::{self, CmpiPredicate},
    ir::{r#type::IntegerType, Block, Location, Value, ValueLike},
    Context,
};
use num_bigint::{BigInt, Sign};
use num_traits::One;

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &CastConcreteLibfunc,
) -> Result<()> {
    match selector {
        CastConcreteLibfunc::Downcast(info) => {
            build_downcast(context, registry, entry, location, helper, metadata, info)
        }
        CastConcreteLibfunc::Upcast(info) => {
            build_upcast(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `downcast` libfunc.
pub fn build_downcast<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &DowncastConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;
    let src_value: Value = entry.arg(1)?;

    if info.signature.param_signatures[1].ty == info.signature.branch_signatures[0].vars[1].ty {
        let k0 = entry.const_int(context, location, 0, 1)?;
        entry.append_operation(helper.cond_br(
            context,
            k0,
            [0, 1],
            [&[range_check, src_value], &[range_check]],
            location,
        ));
        return Ok(());
    }

    let src_ty = registry.get_type(&info.signature.param_signatures[1].ty)?;
    let dst_ty = registry.get_type(&info.signature.branch_signatures[0].vars[1].ty)?;

    let dst_range = dst_ty.integer_range(registry)?;
    let src_range = if src_ty.is_felt252(registry)? && dst_range.lower.sign() == Sign::Minus {
        if dst_range.upper.sign() != Sign::Plus {
            Range {
                lower: BigInt::from_biguint(Sign::Minus, PRIME.clone()) + 1,
                upper: BigInt::one(),
            }
        } else {
            Range {
                lower: BigInt::from_biguint(Sign::Minus, HALF_PRIME.clone()),
                upper: BigInt::from_biguint(Sign::Plus, HALF_PRIME.clone()) + BigInt::one(),
            }
        }
    } else {
        src_ty.integer_range(registry)?
    };

    let src_width = if src_ty.is_bounded_int(registry)? {
        src_range.offset_bit_width()
    } else {
        src_ty.integer_range(registry)?.zero_based_bit_width()
    };
    let dst_width = if dst_ty.is_bounded_int(registry)? {
        dst_range.offset_bit_width()
    } else {
        dst_range.zero_based_bit_width()
    };

    let compute_width = src_range
        .zero_based_bit_width()
        .max(dst_range.zero_based_bit_width());

    let is_signed = src_range.lower.sign() == Sign::Minus;

    let src_value = if compute_width > src_width {
        if is_signed && !src_ty.is_bounded_int(registry)? && !src_ty.is_felt252(registry)? {
            entry.extsi(
                src_value,
                IntegerType::new(context, compute_width).into(),
                location,
            )?
        } else {
            entry.extui(
                src_value,
                IntegerType::new(context, compute_width).into(),
                location,
            )?
        }
    } else {
        src_value
    };

    let src_value = if is_signed && src_ty.is_felt252(registry)? {
        if src_range.upper.is_one() {
            let adj_offset =
                entry.const_int_from_type(context, location, PRIME.clone(), src_value.r#type())?;
            entry.append_op_result(arith::subi(src_value, adj_offset, location))?
        } else {
            let adj_offset = entry.const_int_from_type(
                context,
                location,
                HALF_PRIME.clone(),
                src_value.r#type(),
            )?;
            let is_negative =
                entry.cmpi(context, CmpiPredicate::Ugt, src_value, adj_offset, location)?;

            let k_prime =
                entry.const_int_from_type(context, location, PRIME.clone(), src_value.r#type())?;
            let adj_value = entry.append_op_result(arith::subi(src_value, k_prime, location))?;

            entry.append_op_result(arith::select(is_negative, adj_value, src_value, location))?
        }
    } else if src_ty.is_bounded_int(registry)? && src_range.lower != BigInt::ZERO {
        let dst_offset = entry.const_int_from_type(
            context,
            location,
            src_range.lower.clone(),
            src_value.r#type(),
        )?;
        entry.addi(src_value, dst_offset, location)?
    } else {
        src_value
    };

    if !(dst_range.lower > src_range.lower || dst_range.upper < src_range.upper) {
        let dst_value = if dst_ty.is_bounded_int(registry)? && dst_range.lower != BigInt::ZERO {
            let dst_offset = entry.const_int_from_type(
                context,
                location,
                dst_range.lower,
                src_value.r#type(),
            )?;
            entry.append_op_result(arith::subi(src_value, dst_offset, location))?
        } else {
            src_value
        };

        let dst_value = if dst_width < compute_width {
            entry.trunci(
                dst_value,
                IntegerType::new(context, dst_width).into(),
                location,
            )?
        } else {
            dst_value
        };

        let is_in_bounds = entry.const_int(context, location, 1, 1)?;

        entry.append_operation(helper.cond_br(
            context,
            is_in_bounds,
            [0, 1],
            [&[range_check, dst_value], &[range_check]],
            location,
        ));
    } else {
        let lower_check = if dst_range.lower > src_range.lower {
            let dst_lower = entry.const_int_from_type(
                context,
                location,
                dst_range.lower.clone(),
                src_value.r#type(),
            )?;
            Some(entry.cmpi(
                context,
                if !is_signed {
                    CmpiPredicate::Uge
                } else {
                    CmpiPredicate::Sge
                },
                src_value,
                dst_lower,
                location,
            )?)
        } else {
            None
        };
        let upper_check = if dst_range.upper < src_range.upper {
            let dst_upper = entry.const_int_from_type(
                context,
                location,
                dst_range.upper.clone(),
                src_value.r#type(),
            )?;
            Some(entry.cmpi(
                context,
                if !is_signed {
                    CmpiPredicate::Ult
                } else {
                    CmpiPredicate::Slt
                },
                src_value,
                dst_upper,
                location,
            )?)
        } else {
            None
        };

        let is_in_bounds = match (lower_check, upper_check) {
            (Some(lower_check), Some(upper_check)) => {
                entry.append_op_result(arith::andi(lower_check, upper_check, location))?
            }
            (Some(lower_check), None) => lower_check,
            (None, Some(upper_check)) => upper_check,
            // its always in bounds since dst is larger than src (i.e no bounds checks needed)
            (None, None) => {
                native_panic!("matched an unreachable: no bounds checks are being performed")
            }
        };

        let dst_value = if dst_ty.is_bounded_int(registry)? && dst_range.lower != BigInt::ZERO {
            let dst_offset = entry.const_int_from_type(
                context,
                location,
                dst_range.lower,
                src_value.r#type(),
            )?;
            entry.append_op_result(arith::subi(src_value, dst_offset, location))?
        } else {
            src_value
        };

        let dst_value = if dst_width < compute_width {
            entry.trunci(
                dst_value,
                IntegerType::new(context, dst_width).into(),
                location,
            )?
        } else {
            dst_value
        };
        entry.append_operation(helper.cond_br(
            context,
            is_in_bounds,
            [0, 1],
            [&[range_check, dst_value], &[range_check]],
            location,
        ));
    }

    Ok(())
}

/// Generate MLIR operations for the `upcast` libfunc.
pub fn build_upcast<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let src_value = entry.arg(0)?;

    if info.signature.param_signatures[0].ty == info.signature.branch_signatures[0].vars[0].ty {
        entry.append_operation(helper.br(0, &[src_value], location));
        return Ok(());
    }

    let src_ty = registry.get_type(&info.signature.param_signatures[0].ty)?;
    let dst_ty = registry.get_type(&info.signature.branch_signatures[0].vars[0].ty)?;

    let src_range = src_ty.integer_range(registry)?;
    let dst_range = dst_ty.integer_range(registry)?;
    assert!(
        if dst_ty.is_felt252(registry)? {
            let alt_range = Range {
                lower: BigInt::from_biguint(Sign::Minus, HALF_PRIME.clone()),
                upper: BigInt::from_biguint(Sign::Plus, HALF_PRIME.clone()) + BigInt::one(),
            };

            (dst_range.lower <= src_range.lower && dst_range.upper >= src_range.upper)
                || (alt_range.lower <= src_range.lower && alt_range.upper >= src_range.upper)
        } else {
            dst_range.lower <= src_range.lower && dst_range.upper >= src_range.upper
        },
        "invalid upcast `{:?}` into `{:?}`: target range doesn't contain the source range",
        info.signature.param_signatures[0].ty,
        info.signature.branch_signatures[0].vars[0].ty
    );

    let src_width = if src_ty.is_bounded_int(registry)? {
        src_range.offset_bit_width()
    } else {
        src_range.zero_based_bit_width()
    };
    let dst_width = if dst_ty.is_bounded_int(registry)? {
        dst_range.offset_bit_width()
    } else {
        dst_range.zero_based_bit_width()
    };

    // If the source can be negative, the target type must also contain negatives when upcasting.
    assert!(
        src_range.lower.sign() != Sign::Minus
            || dst_ty.is_felt252(registry)?
            || dst_range.lower.sign() == Sign::Minus
    );
    let is_signed = src_range.lower.sign() == Sign::Minus;

    let dst_value = if dst_width > src_width {
        if is_signed && !src_ty.is_bounded_int(registry)? {
            entry.extsi(
                src_value,
                IntegerType::new(context, dst_width).into(),
                location,
            )?
        } else {
            entry.extui(
                src_value,
                IntegerType::new(context, dst_width).into(),
                location,
            )?
        }
    } else {
        src_value
    };

    let dst_value = if src_ty.is_bounded_int(registry)? && src_range.lower != BigInt::ZERO {
        let dst_offset = entry.const_int_from_type(
            context,
            location,
            if dst_ty.is_bounded_int(registry)? {
                &src_range.lower - &dst_range.lower
            } else {
                src_range.lower.clone()
            },
            dst_value.r#type(),
        )?;
        entry.addi(dst_value, dst_offset, location)?
    } else {
        dst_value
    };

    let dst_value = if dst_ty.is_felt252(registry)? && src_range.lower.sign() == Sign::Minus {
        let k0 = entry.const_int(context, location, 0, 252)?;
        let is_negative = entry.cmpi(context, CmpiPredicate::Slt, dst_value, k0, location)?;

        let k_prime = entry.const_int(context, location, PRIME.clone(), 252)?;
        let adj_value = entry.addi(dst_value, k_prime, location)?;

        entry.append_op_result(arith::select(is_negative, adj_value, dst_value, location))?
    } else {
        dst_value
    };

    entry.append_operation(helper.br(0, &[dst_value], location));
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{
        utils::test::{jit_enum, jit_struct, run_sierra_program},
        values::Value,
    };
    use cairo_lang_sierra::{program::Program, ProgramParser};
    use lazy_static::lazy_static;

    lazy_static! {
        // use core::integer::downcast;
        // fn run_test(
        //     v8: u8, v16: u16, v32: u32, v64: u64, v128: u128
        // ) -> (
        //     (Option<u8>, Option<u8>, Option<u8>, Option<u8>, Option<u8>),
        //     (Option<u16>, Option<u16>, Option<u16>, Option<u16>),
        //     (Option<u32>, Option<u32>, Option<u32>),
        //     (Option<u64>, Option<u64>),
        //     (Option<u128>,),
        // ) {
        //     (
        //         (downcast(v128), downcast(v64), downcast(v32), downcast(v16), downcast(v8)),
        //         (downcast(v128), downcast(v64), downcast(v32), downcast(v16)),
        //         (downcast(v128), downcast(v64), downcast(v32)),
        //         (downcast(v128), downcast(v64)),
        //         (downcast(v128),),
        //     )
        // }
        static ref DOWNCAST: Program = ProgramParser::new().parse(r#"
            type [5] = u128 [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = u8 [storable: true, drop: true, dup: true, zero_sized: false];
            type [6] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
            type [7] = Enum<ut@core::option::Option::<core::integer::u8>, [1], [6]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [8] = Struct<ut@Tuple, [7], [7], [7], [7], [7]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [2] = u16 [storable: true, drop: true, dup: true, zero_sized: false];
            type [9] = Enum<ut@core::option::Option::<core::integer::u16>, [2], [6]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [10] = Struct<ut@Tuple, [9], [9], [9], [9]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [3] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
            type [11] = Enum<ut@core::option::Option::<core::integer::u32>, [3], [6]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [12] = Struct<ut@Tuple, [11], [11], [11]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [4] = u64 [storable: true, drop: true, dup: true, zero_sized: false];
            type [13] = Enum<ut@core::option::Option::<core::integer::u64>, [4], [6]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [14] = Struct<ut@Tuple, [13], [13]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [15] = Enum<ut@core::option::Option::<core::integer::u128>, [5], [6]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [16] = Struct<ut@Tuple, [15]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [17] = Struct<ut@Tuple, [8], [10], [12], [14], [16]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];

            libfunc [32] = dup<[5]>;
            libfunc [31] = downcast<[5], [1]>;
            libfunc [33] = branch_align;
            libfunc [26] = enum_init<[7], 0>;
            libfunc [38] = store_temp<[0]>;
            libfunc [39] = store_temp<[7]>;
            libfunc [34] = jump;
            libfunc [3] = struct_construct<[6]>;
            libfunc [25] = enum_init<[7], 1>;
            libfunc [35] = dup<[4]>;
            libfunc [30] = downcast<[4], [1]>;
            libfunc [36] = dup<[3]>;
            libfunc [29] = downcast<[3], [1]>;
            libfunc [37] = dup<[2]>;
            libfunc [28] = downcast<[2], [1]>;
            libfunc [27] = downcast<[1], [1]>;
            libfunc [24] = struct_construct<[8]>;
            libfunc [40] = store_temp<[8]>;
            libfunc [23] = downcast<[5], [2]>;
            libfunc [19] = enum_init<[9], 0>;
            libfunc [41] = store_temp<[9]>;
            libfunc [18] = enum_init<[9], 1>;
            libfunc [22] = downcast<[4], [2]>;
            libfunc [21] = downcast<[3], [2]>;
            libfunc [20] = downcast<[2], [2]>;
            libfunc [17] = struct_construct<[10]>;
            libfunc [42] = store_temp<[10]>;
            libfunc [16] = downcast<[5], [3]>;
            libfunc [13] = enum_init<[11], 0>;
            libfunc [43] = store_temp<[11]>;
            libfunc [12] = enum_init<[11], 1>;
            libfunc [15] = downcast<[4], [3]>;
            libfunc [14] = downcast<[3], [3]>;
            libfunc [11] = struct_construct<[12]>;
            libfunc [44] = store_temp<[12]>;
            libfunc [10] = downcast<[5], [4]>;
            libfunc [8] = enum_init<[13], 0>;
            libfunc [45] = store_temp<[13]>;
            libfunc [7] = enum_init<[13], 1>;
            libfunc [9] = downcast<[4], [4]>;
            libfunc [6] = struct_construct<[14]>;
            libfunc [46] = store_temp<[14]>;
            libfunc [5] = downcast<[5], [5]>;
            libfunc [4] = enum_init<[15], 0>;
            libfunc [47] = store_temp<[15]>;
            libfunc [2] = enum_init<[15], 1>;
            libfunc [1] = struct_construct<[16]>;
            libfunc [0] = struct_construct<[17]>;
            libfunc [48] = store_temp<[17]>;

            [32]([5]) -> ([5], [6]); // 0
            [31]([0], [6]) { fallthrough([7], [8]) 7([9]) }; // 1
            [33]() -> (); // 2
            [26]([8]) -> ([10]); // 3
            [38]([7]) -> ([11]); // 4
            [39]([10]) -> ([12]); // 5
            [34]() { 12() }; // 6
            [33]() -> (); // 7
            [3]() -> ([13]); // 8
            [25]([13]) -> ([14]); // 9
            [38]([9]) -> ([11]); // 10
            [39]([14]) -> ([12]); // 11
            [35]([4]) -> ([4], [15]); // 12
            [30]([11], [15]) { fallthrough([16], [17]) 19([18]) }; // 13
            [33]() -> (); // 14
            [26]([17]) -> ([19]); // 15
            [38]([16]) -> ([20]); // 16
            [39]([19]) -> ([21]); // 17
            [34]() { 24() }; // 18
            [33]() -> (); // 19
            [3]() -> ([22]); // 20
            [25]([22]) -> ([23]); // 21
            [38]([18]) -> ([20]); // 22
            [39]([23]) -> ([21]); // 23
            [36]([3]) -> ([3], [24]); // 24
            [29]([20], [24]) { fallthrough([25], [26]) 31([27]) }; // 25
            [33]() -> (); // 26
            [26]([26]) -> ([28]); // 27
            [38]([25]) -> ([29]); // 28
            [39]([28]) -> ([30]); // 29
            [34]() { 36() }; // 30
            [33]() -> (); // 31
            [3]() -> ([31]); // 32
            [25]([31]) -> ([32]); // 33
            [38]([27]) -> ([29]); // 34
            [39]([32]) -> ([30]); // 35
            [37]([2]) -> ([2], [33]); // 36
            [28]([29], [33]) { fallthrough([34], [35]) 43([36]) }; // 37
            [33]() -> (); // 38
            [26]([35]) -> ([37]); // 39
            [38]([34]) -> ([38]); // 40
            [39]([37]) -> ([39]); // 41
            [34]() { 48() }; // 42
            [33]() -> (); // 43
            [3]() -> ([40]); // 44
            [25]([40]) -> ([41]); // 45
            [38]([36]) -> ([38]); // 46
            [39]([41]) -> ([39]); // 47
            [27]([38], [1]) { fallthrough([42], [43]) 54([44]) }; // 48
            [33]() -> (); // 49
            [26]([43]) -> ([45]); // 50
            [38]([42]) -> ([46]); // 51
            [39]([45]) -> ([47]); // 52
            [34]() { 59() }; // 53
            [33]() -> (); // 54
            [3]() -> ([48]); // 55
            [25]([48]) -> ([49]); // 56
            [38]([44]) -> ([46]); // 57
            [39]([49]) -> ([47]); // 58
            [24]([12], [21], [30], [39], [47]) -> ([50]); // 59
            [32]([5]) -> ([5], [51]); // 60
            [40]([50]) -> ([50]); // 61
            [23]([46], [51]) { fallthrough([52], [53]) 68([54]) }; // 62
            [33]() -> (); // 63
            [19]([53]) -> ([55]); // 64
            [38]([52]) -> ([56]); // 65
            [41]([55]) -> ([57]); // 66
            [34]() { 73() }; // 67
            [33]() -> (); // 68
            [3]() -> ([58]); // 69
            [18]([58]) -> ([59]); // 70
            [38]([54]) -> ([56]); // 71
            [41]([59]) -> ([57]); // 72
            [35]([4]) -> ([4], [60]); // 73
            [22]([56], [60]) { fallthrough([61], [62]) 80([63]) }; // 74
            [33]() -> (); // 75
            [19]([62]) -> ([64]); // 76
            [38]([61]) -> ([65]); // 77
            [41]([64]) -> ([66]); // 78
            [34]() { 85() }; // 79
            [33]() -> (); // 80
            [3]() -> ([67]); // 81
            [18]([67]) -> ([68]); // 82
            [38]([63]) -> ([65]); // 83
            [41]([68]) -> ([66]); // 84
            [36]([3]) -> ([3], [69]); // 85
            [21]([65], [69]) { fallthrough([70], [71]) 92([72]) }; // 86
            [33]() -> (); // 87
            [19]([71]) -> ([73]); // 88
            [38]([70]) -> ([74]); // 89
            [41]([73]) -> ([75]); // 90
            [34]() { 97() }; // 91
            [33]() -> (); // 92
            [3]() -> ([76]); // 93
            [18]([76]) -> ([77]); // 94
            [38]([72]) -> ([74]); // 95
            [41]([77]) -> ([75]); // 96
            [20]([74], [2]) { fallthrough([78], [79]) 103([80]) }; // 97
            [33]() -> (); // 98
            [19]([79]) -> ([81]); // 99
            [38]([78]) -> ([82]); // 100
            [41]([81]) -> ([83]); // 101
            [34]() { 108() }; // 102
            [33]() -> (); // 103
            [3]() -> ([84]); // 104
            [18]([84]) -> ([85]); // 105
            [38]([80]) -> ([82]); // 106
            [41]([85]) -> ([83]); // 107
            [17]([57], [66], [75], [83]) -> ([86]); // 108
            [32]([5]) -> ([5], [87]); // 109
            [42]([86]) -> ([86]); // 110
            [16]([82], [87]) { fallthrough([88], [89]) 117([90]) }; // 111
            [33]() -> (); // 112
            [13]([89]) -> ([91]); // 113
            [38]([88]) -> ([92]); // 114
            [43]([91]) -> ([93]); // 115
            [34]() { 122() }; // 116
            [33]() -> (); // 117
            [3]() -> ([94]); // 118
            [12]([94]) -> ([95]); // 119
            [38]([90]) -> ([92]); // 120
            [43]([95]) -> ([93]); // 121
            [35]([4]) -> ([4], [96]); // 122
            [15]([92], [96]) { fallthrough([97], [98]) 129([99]) }; // 123
            [33]() -> (); // 124
            [13]([98]) -> ([100]); // 125
            [38]([97]) -> ([101]); // 126
            [43]([100]) -> ([102]); // 127
            [34]() { 134() }; // 128
            [33]() -> (); // 129
            [3]() -> ([103]); // 130
            [12]([103]) -> ([104]); // 131
            [38]([99]) -> ([101]); // 132
            [43]([104]) -> ([102]); // 133
            [14]([101], [3]) { fallthrough([105], [106]) 140([107]) }; // 134
            [33]() -> (); // 135
            [13]([106]) -> ([108]); // 136
            [38]([105]) -> ([109]); // 137
            [43]([108]) -> ([110]); // 138
            [34]() { 145() }; // 139
            [33]() -> (); // 140
            [3]() -> ([111]); // 141
            [12]([111]) -> ([112]); // 142
            [38]([107]) -> ([109]); // 143
            [43]([112]) -> ([110]); // 144
            [11]([93], [102], [110]) -> ([113]); // 145
            [32]([5]) -> ([5], [114]); // 146
            [44]([113]) -> ([113]); // 147
            [10]([109], [114]) { fallthrough([115], [116]) 154([117]) }; // 148
            [33]() -> (); // 149
            [8]([116]) -> ([118]); // 150
            [38]([115]) -> ([119]); // 151
            [45]([118]) -> ([120]); // 152
            [34]() { 159() }; // 153
            [33]() -> (); // 154
            [3]() -> ([121]); // 155
            [7]([121]) -> ([122]); // 156
            [38]([117]) -> ([119]); // 157
            [45]([122]) -> ([120]); // 158
            [9]([119], [4]) { fallthrough([123], [124]) 165([125]) }; // 159
            [33]() -> (); // 160
            [8]([124]) -> ([126]); // 161
            [38]([123]) -> ([127]); // 162
            [45]([126]) -> ([128]); // 163
            [34]() { 170() }; // 164
            [33]() -> (); // 165
            [3]() -> ([129]); // 166
            [7]([129]) -> ([130]); // 167
            [38]([125]) -> ([127]); // 168
            [45]([130]) -> ([128]); // 169
            [6]([120], [128]) -> ([131]); // 170
            [46]([131]) -> ([131]); // 171
            [5]([127], [5]) { fallthrough([132], [133]) 178([134]) }; // 172
            [33]() -> (); // 173
            [4]([133]) -> ([135]); // 174
            [38]([132]) -> ([136]); // 175
            [47]([135]) -> ([137]); // 176
            [34]() { 183() }; // 177
            [33]() -> (); // 178
            [3]() -> ([138]); // 179
            [2]([138]) -> ([139]); // 180
            [38]([134]) -> ([136]); // 181
            [47]([139]) -> ([137]); // 182
            [1]([137]) -> ([140]); // 183
            [0]([50], [86], [113], [131], [140]) -> ([141]); // 184
            [38]([136]) -> ([136]); // 185
            [48]([141]) -> ([141]); // 186
            return([136], [141]); // 187

            [0]@0([0]: [0], [1]: [1], [2]: [2], [3]: [3], [4]: [4], [5]: [5]) -> ([0], [17]);
        "#).map_err(|e| e.to_string()).unwrap();

        // use core::integer::upcast;
        // fn run_test(
        //     v8: u8, v16: u16, v32: u32, v64: u64, v128: u128, v248: bytes31
        // ) -> (
        //     (u8,),
        //     (u16, u16),
        //     (u32, u32, u32),
        //     (u64, u64, u64, u64),
        //     (u128, u128, u128, u128, u128),
        //     (bytes31, bytes31, bytes31, bytes31, bytes31, bytes31)
        // ) {
        //     (
        //         (upcast(v8),),
        //         (upcast(v8), upcast(v16)),
        //         (upcast(v8), upcast(v16), upcast(v32)),
        //         (upcast(v8), upcast(v16), upcast(v32), upcast(v64)),
        //         (upcast(v8), upcast(v16), upcast(v32), upcast(v64), upcast(v128)),
        //         (upcast(v8), upcast(v16), upcast(v32), upcast(v64), upcast(v128), upcast(v248)),
        //     )
        // }
        static ref UPCAST: Program = ProgramParser::new().parse(r#"
            type [0] = u8 [storable: true, drop: true, dup: true, zero_sized: false];
            type [6] = Struct<ut@Tuple, [0]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = u16 [storable: true, drop: true, dup: true, zero_sized: false];
            type [7] = Struct<ut@Tuple, [1], [1]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [2] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
            type [8] = Struct<ut@Tuple, [2], [2], [2]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [3] = u64 [storable: true, drop: true, dup: true, zero_sized: false];
            type [9] = Struct<ut@Tuple, [3], [3], [3], [3]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [4] = u128 [storable: true, drop: true, dup: true, zero_sized: false];
            type [10] = Struct<ut@Tuple, [4], [4], [4], [4], [4]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [5] = bytes31 [storable: true, drop: true, dup: true, zero_sized: false];
            type [11] = Struct<ut@Tuple, [5], [5], [5], [5], [5], [5]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [12] = Struct<ut@Tuple, [6], [7], [8], [9], [10], [11]> [storable: true, drop: true, dup: true, zero_sized: false];

            libfunc [29] = dup<[0]>;
            libfunc [27] = upcast<[0], [0]>;
            libfunc [26] = upcast<[0], [1]>;
            libfunc [30] = dup<[1]>;
            libfunc [25] = upcast<[1], [1]>;
            libfunc [24] = upcast<[0], [2]>;
            libfunc [23] = upcast<[1], [2]>;
            libfunc [31] = dup<[2]>;
            libfunc [22] = upcast<[2], [2]>;
            libfunc [21] = upcast<[0], [3]>;
            libfunc [20] = upcast<[1], [3]>;
            libfunc [19] = upcast<[2], [3]>;
            libfunc [32] = dup<[3]>;
            libfunc [18] = upcast<[3], [3]>;
            libfunc [17] = upcast<[0], [4]>;
            libfunc [16] = upcast<[1], [4]>;
            libfunc [15] = upcast<[2], [4]>;
            libfunc [14] = upcast<[3], [4]>;
            libfunc [33] = dup<[4]>;
            libfunc [13] = upcast<[4], [4]>;
            libfunc [12] = upcast<[0], [5]>;
            libfunc [11] = upcast<[1], [5]>;
            libfunc [10] = upcast<[2], [5]>;
            libfunc [9] = upcast<[3], [5]>;
            libfunc [8] = upcast<[4], [5]>;
            libfunc [7] = upcast<[5], [5]>;
            libfunc [6] = struct_construct<[6]>;
            libfunc [5] = struct_construct<[7]>;
            libfunc [4] = struct_construct<[8]>;
            libfunc [3] = struct_construct<[9]>;
            libfunc [2] = struct_construct<[10]>;
            libfunc [1] = struct_construct<[11]>;
            libfunc [0] = struct_construct<[12]>;
            libfunc [34] = store_temp<[12]>;

            [29]([0]) -> ([0], [6]); // 0
            [27]([6]) -> ([7]); // 1
            [29]([0]) -> ([0], [8]); // 2
            [26]([8]) -> ([9]); // 3
            [30]([1]) -> ([1], [10]); // 4
            [25]([10]) -> ([11]); // 5
            [29]([0]) -> ([0], [12]); // 6
            [24]([12]) -> ([13]); // 7
            [30]([1]) -> ([1], [14]); // 8
            [23]([14]) -> ([15]); // 9
            [31]([2]) -> ([2], [16]); // 10
            [22]([16]) -> ([17]); // 11
            [29]([0]) -> ([0], [18]); // 12
            [21]([18]) -> ([19]); // 13
            [30]([1]) -> ([1], [20]); // 14
            [20]([20]) -> ([21]); // 15
            [31]([2]) -> ([2], [22]); // 16
            [19]([22]) -> ([23]); // 17
            [32]([3]) -> ([3], [24]); // 18
            [18]([24]) -> ([25]); // 19
            [29]([0]) -> ([0], [26]); // 20
            [17]([26]) -> ([27]); // 21
            [30]([1]) -> ([1], [28]); // 22
            [16]([28]) -> ([29]); // 23
            [31]([2]) -> ([2], [30]); // 24
            [15]([30]) -> ([31]); // 25
            [32]([3]) -> ([3], [32]); // 26
            [14]([32]) -> ([33]); // 27
            [33]([4]) -> ([4], [34]); // 28
            [13]([34]) -> ([35]); // 29
            [12]([0]) -> ([36]); // 30
            [11]([1]) -> ([37]); // 31
            [10]([2]) -> ([38]); // 32
            [9]([3]) -> ([39]); // 33
            [8]([4]) -> ([40]); // 34
            [7]([5]) -> ([41]); // 35
            [6]([7]) -> ([42]); // 36
            [5]([9], [11]) -> ([43]); // 37
            [4]([13], [15], [17]) -> ([44]); // 38
            [3]([19], [21], [23], [25]) -> ([45]); // 39
            [2]([27], [29], [31], [33], [35]) -> ([46]); // 40
            [1]([36], [37], [38], [39], [40], [41]) -> ([47]); // 41
            [0]([42], [43], [44], [45], [46], [47]) -> ([48]); // 42
            [34]([48]) -> ([48]); // 43
            return([48]); // 44

            [0]@0([0]: [0], [1]: [1], [2]: [2], [3]: [3], [4]: [4], [5]: [5]) -> ([12]);
        "#).map_err(|e| e.to_string()).unwrap();
    }

    #[test]
    fn downcast() {
        let return_value = run_sierra_program(
            &DOWNCAST,
            &[
                u8::MAX.into(),
                u16::MAX.into(),
                u32::MAX.into(),
                u64::MAX.into(),
                u128::MAX.into(),
            ],
        )
        .return_value;

        assert_eq!(
            jit_struct!(
                jit_struct!(
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                ),
                jit_struct!(
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                ),
                jit_struct!(
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                    jit_enum!(1, jit_struct!()),
                ),
                jit_struct!(jit_enum!(1, jit_struct!()), jit_enum!(1, jit_struct!())),
                jit_struct!(jit_enum!(1, jit_struct!())),
            ),
            return_value
        );
    }

    #[test]
    fn upcast() {
        let return_value = run_sierra_program(
            &UPCAST,
            &[
                u8::MAX.into(),
                u16::MAX.into(),
                u32::MAX.into(),
                u64::MAX.into(),
                u128::MAX.into(),
                Value::Bytes31([0xFF; 31]),
            ],
        )
        .return_value;

        assert_eq!(
            jit_struct!(
                jit_struct!(u8::MAX.into()),
                jit_struct!((u8::MAX as u16).into(), u16::MAX.into()),
                jit_struct!(
                    (u8::MAX as u32).into(),
                    (u16::MAX as u32).into(),
                    u32::MAX.into()
                ),
                jit_struct!(
                    (u8::MAX as u64).into(),
                    (u16::MAX as u64).into(),
                    (u32::MAX as u64).into(),
                    u64::MAX.into()
                ),
                jit_struct!(
                    (u8::MAX as u128).into(),
                    (u16::MAX as u128).into(),
                    (u32::MAX as u128).into(),
                    (u64::MAX as u128).into(),
                    u128::MAX.into()
                ),
                jit_struct!(
                    Value::Bytes31([
                        u8::MAX,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]),
                    Value::Bytes31([
                        u8::MAX,
                        u8::MAX,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]),
                    Value::Bytes31([
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]),
                    Value::Bytes31([
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]),
                    Value::Bytes31([
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        u8::MAX,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]),
                    Value::Bytes31([u8::MAX; 31]),
                ),
            ),
            return_value
        );
    }
}
