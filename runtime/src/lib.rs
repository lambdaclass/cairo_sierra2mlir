#![allow(non_snake_case)]

use cairo_felt::Felt252;
use cairo_lang_runner::short_string::as_cairo_short_string;
use starknet_crypto::FieldElement;
use std::{fs::File, io::Write, os::fd::FromRawFd, slice};

/// Based on `cairo-lang-runner`'s implementation.
///
/// Source: https://github.com/starkware-libs/cairo/blob/main/crates/cairo-lang-runner/src/casm_run/mod.rs#L1789-L1800
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__libfunc__debug__print(
    target_fd: i32,
    data: *const [u8; 32],
    len: usize,
) -> i32 {
    let mut target = File::from_raw_fd(target_fd);

    for i in 0..len {
        let mut data = *data.add(i);
        data.reverse();

        let value = Felt252::from_bytes_be(&data);
        if let Some(shortstring) = as_cairo_short_string(&value) {
            if writeln!(
                target,
                "[DEBUG]\t{shortstring: <31}\t(raw: {})",
                value.to_bigint()
            )
            .is_err()
            {
                return 1;
            };
        } else if writeln!(target, "[DEBUG]\t{:<31}\t(raw: {})", ' ', value.to_bigint()).is_err() {
            return 1;
        }
    }
    if writeln!(target).is_err() {
        return 1;
    };

    // Avoid closing `stdout`.
    std::mem::forget(target);

    0
}

/// Compute `pedersen(lhs, rhs)` and store it into `dst`.
///
/// All its operands need the values in big endian.
///
/// # Panics
///
/// This function will panic if either operand is out of range for a felt.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
#[no_mangle]
pub unsafe extern "C" fn cairo_native__libfunc_pedersen(
    dst: *mut u8,
    lhs: *const u8,
    rhs: *const u8,
) {
    // Extract arrays from the pointers.
    let dst = slice::from_raw_parts_mut(dst, 32);
    let lhs = slice::from_raw_parts(lhs, 32);
    let rhs = slice::from_raw_parts(rhs, 32);

    // Convert to FieldElement.
    let lhs = FieldElement::from_byte_slice_be(lhs).unwrap();
    let rhs = FieldElement::from_byte_slice_be(rhs).unwrap();

    // Compute pedersen hash and copy the result into `dst`.
    let res = starknet_crypto::pedersen_hash(&lhs, &rhs);
    dst.copy_from_slice(&res.to_bytes_be());
}
