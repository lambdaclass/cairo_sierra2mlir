#![allow(non_snake_case)]

use crate::utils::BuiltinCosts;
use cairo_lang_sierra_gas::core_libfunc_cost::{
    DICT_SQUASH_REPEATED_ACCESS_COST, DICT_SQUASH_UNIQUE_KEY_COST,
};
use itertools::Itertools;
use lazy_static::lazy_static;
use num_traits::{ToPrimitive, Zero};
use rand::Rng;
use starknet_curve::curve_params::BETA;
use starknet_types_core::{
    curve::{AffinePoint, ProjectivePoint},
    felt::Felt,
    hash::StarkHash,
};
use std::{
    alloc::{alloc, dealloc, realloc, Layout},
    cell::Cell,
    collections::{hash_map::Entry, HashMap},
    ffi::{c_int, c_void},
    fs::File,
    io::Write,
    mem::{forget, ManuallyDrop},
    os::fd::FromRawFd,
    ptr,
    rc::Rc,
};
use std::{ops::Mul, vec::IntoIter};

lazy_static! {
    pub static ref HALF_PRIME: Felt = Felt::from_dec_str(
        "1809251394333065606848661391547535052811553607665798349986546028067936010240"
    )
    .unwrap();
    pub static ref DICT_GAS_REFUND_PER_ACCESS: u64 =
        (DICT_SQUASH_UNIQUE_KEY_COST.cost() - DICT_SQUASH_REPEATED_ACCESS_COST.cost()) as u64;
}

/// Based on `cairo-lang-runner`'s implementation.
///
/// Source: <https://github.com/starkware-libs/cairo/blob/main/crates/cairo-lang-runner/src/casm_run/mod.rs#L1946-L1948>
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
pub unsafe extern "C" fn cairo_native__libfunc__debug__print(
    target_fd: i32,
    data: *const [u8; 32],
    len: u32,
) -> i32 {
    // Avoid closing `stdout` on all branches.
    let mut target = ManuallyDrop::new(File::from_raw_fd(target_fd));

    let mut items = Vec::with_capacity(len as usize);

    for i in 0..len as usize {
        let mut data = *data.add(i);
        data[31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).

        let value = Felt::from_bytes_le(&data);
        items.push(value);
    }

    let value = format_for_debug(items.into_iter());

    if write!(target, "{}", value).is_err() {
        return 1;
    };

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
pub unsafe extern "C" fn cairo_native__libfunc__pedersen(
    dst: &mut [u8; 32],
    lhs: &[u8; 32],
    rhs: &[u8; 32],
) {
    // Extract arrays from the pointers.
    let mut lhs = *lhs;
    let mut rhs = *rhs;

    lhs[31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
    rhs[31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).

    // Convert to FieldElement.
    let lhs = Felt::from_bytes_le(&lhs);
    let rhs = Felt::from_bytes_le(&rhs);

    // Compute pedersen hash and copy the result into `dst`.
    let res = starknet_types_core::hash::Pedersen::hash(&lhs, &rhs);
    *dst = res.to_bytes_le();
}

/// Compute `hades_permutation(op0, op1, op2)` and replace the operands with the results.
///
/// All operands need the values in big endian.
///
/// # Panics
///
/// This function will panic if either operand is out of range for a felt.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
pub unsafe extern "C" fn cairo_native__libfunc__hades_permutation(
    op0: &mut [u8; 32],
    op1: &mut [u8; 32],
    op2: &mut [u8; 32],
) {
    op0[31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
    op1[31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
    op2[31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).

    // Convert to FieldElement.
    let mut state = [
        Felt::from_bytes_le(op0),
        Felt::from_bytes_le(op1),
        Felt::from_bytes_le(op2),
    ];

    // Compute Poseidon permutation.
    starknet_types_core::hash::Poseidon::hades_permutation(&mut state);

    // Write back the results.
    *op0 = state[0].to_bytes_le();
    *op1 = state[1].to_bytes_le();
    *op2 = state[2].to_bytes_le();
}

/// Felt252 type used in cairo native runtime
#[derive(Debug)]
pub struct FeltDict {
    pub mappings: HashMap<[u8; 32], usize>,

    pub layout: Layout,
    pub elements: *mut (),

    pub dup_fn: Option<extern "C" fn(*mut c_void, *mut c_void)>,
    pub drop_fn: Option<extern "C" fn(*mut c_void)>,

    pub count: u64,
}

impl Clone for FeltDict {
    fn clone(&self) -> Self {
        let mut new_dict = FeltDict {
            mappings: HashMap::with_capacity(self.mappings.len()),

            layout: self.layout,
            elements: if self.mappings.is_empty() {
                ptr::null_mut()
            } else {
                unsafe {
                    alloc(Layout::from_size_align_unchecked(
                        self.layout.pad_to_align().size() * self.mappings.len(),
                        self.layout.align(),
                    ))
                    .cast()
                }
            },

            dup_fn: self.dup_fn,
            drop_fn: self.drop_fn,

            // TODO: Check if `0` is fine or otherwise we should copy the value from `old_dict` too.
            count: 0,
        };

        for (&key, &old_index) in self.mappings.iter() {
            let old_value_ptr = unsafe {
                self.elements
                    .byte_add(self.layout.pad_to_align().size() * old_index)
            };

            let new_index = new_dict.mappings.len();
            let new_value_ptr = unsafe {
                new_dict
                    .elements
                    .byte_add(new_dict.layout.pad_to_align().size() * new_index)
            };

            new_dict.mappings.insert(key, new_index);
            match self.dup_fn {
                Some(dup_fn) => dup_fn(old_value_ptr.cast(), new_value_ptr.cast()),
                None => unsafe {
                    ptr::copy_nonoverlapping::<u8>(
                        old_value_ptr.cast(),
                        new_value_ptr.cast(),
                        self.layout.size(),
                    )
                },
            }
        }

        new_dict
    }
}

impl Drop for FeltDict {
    fn drop(&mut self) {
        // Free the entries manually.
        if let Some(drop_fn) = self.drop_fn {
            for (_, &index) in self.mappings.iter() {
                let value_ptr = unsafe {
                    self.elements
                        .byte_add(self.layout.pad_to_align().size() * index)
                };

                drop_fn(value_ptr.cast());
            }
        }

        // Free the value data.
        if !self.elements.is_null() {
            unsafe {
                dealloc(
                    self.elements.cast(),
                    Layout::from_size_align_unchecked(
                        self.layout.pad_to_align().size() * self.mappings.capacity(),
                        self.layout.align(),
                    ),
                )
            };
        }
    }
}

/// Allocate a new dictionary.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
pub unsafe extern "C" fn cairo_native__dict_new(
    size: u64,
    align: u64,
    dup_fn: Option<extern "C" fn(*mut c_void, *mut c_void)>,
    drop_fn: Option<extern "C" fn(*mut c_void)>,
) -> *const FeltDict {
    Rc::into_raw(Rc::new(FeltDict {
        mappings: HashMap::default(),

        layout: Layout::from_size_align_unchecked(size as usize, align as usize),
        elements: ptr::null_mut(),

        dup_fn,
        drop_fn,

        count: 0,
    }))
}

/// Free a dictionary using an optional callback to drop each element.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
// Note: Using `Option<extern "C" fn(*mut c_void)>` is ffi-safe thanks to Option's null
//   pointer optimization. Check out
//   https://doc.rust-lang.org/nomicon/ffi.html#the-nullable-pointer-optimization for more info.
pub unsafe extern "C" fn cairo_native__dict_drop(ptr: *const FeltDict) {
    drop(Rc::from_raw(ptr));
}

/// Duplicate a dictionary using a provided callback to clone each element.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
pub unsafe extern "C" fn cairo_native__dict_dup(dict_ptr: *const FeltDict) -> *const FeltDict {
    let old_dict = Rc::from_raw(dict_ptr);
    let new_dict = Rc::clone(&old_dict);

    forget(old_dict);
    Rc::into_raw(new_dict)
}

/// Return a pointer to the entry's value pointer for a given key, inserting a null pointer if not
/// present. Increment the access count.
///
/// The null pointer will be either updated by `felt252_dict_entry_finalize` or removed (along with
/// everything else in the dict) by the entry's drop implementation.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
pub unsafe extern "C" fn cairo_native__dict_get(
    dict: *const FeltDict,
    key: &[u8; 32],
    value_ptr: *mut *mut c_void,
) -> c_int {
    let mut dict_rc = Rc::from_raw(dict);
    let dict = Rc::make_mut(&mut dict_rc);

    let num_mappings = dict.mappings.len();
    let has_capacity = num_mappings != dict.mappings.capacity();

    let (is_present, index) = match dict.mappings.entry(*key) {
        Entry::Occupied(entry) => (true, *entry.get()),
        Entry::Vacant(entry) => {
            entry.insert(num_mappings);
            (false, num_mappings)
        }
    };

    // Maybe realloc (conditions: !has_capacity && !is_present).
    if !has_capacity && !is_present {
        dict.elements = realloc(
            dict.elements.cast(),
            Layout::from_size_align_unchecked(
                dict.layout.pad_to_align().size() * dict.mappings.len(),
                dict.layout.align(),
            ),
            dict.layout.pad_to_align().size() * dict.mappings.capacity(),
        )
        .cast();
    }

    *value_ptr = dict
        .elements
        .byte_add(dict.layout.pad_to_align().size() * index)
        .cast();

    dict.count += 1;
    forget(dict_rc);

    is_present as c_int
}

/// Compute the total gas refund for the dictionary at squash time.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
pub unsafe extern "C" fn cairo_native__dict_gas_refund(ptr: *const FeltDict) -> u64 {
    let dict = Rc::from_raw(ptr);
    let amount =
        (dict.count.saturating_sub(dict.mappings.len() as u64)) * *DICT_GAS_REFUND_PER_ACCESS;

    forget(dict);
    amount
}

/// Compute `ec_point_from_x_nz(x)` and store it.
///
/// # Panics
///
/// This function will panic if either operand is out of range for a felt.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_point_from_x_nz(
    point_ptr: &mut [[u8; 32]; 2],
) -> bool {
    point_ptr[0][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
    let x = Felt::from_bytes_le(&point_ptr[0]);

    // https://github.com/starkware-libs/cairo/blob/aaad921bba52e729dc24ece07fab2edf09ccfa15/crates/cairo-lang-sierra-to-casm/src/invocations/ec.rs#L63

    let x2 = x * x;
    let x3 = x2 * x;
    let alpha_x_plus_beta = x + BETA;
    let rhs = x3 + alpha_x_plus_beta;
    // https://github.com/starkware-libs/cairo/blob/9b603b88c2e5a98eec1bb8f323260b7765e94911/crates/cairo-lang-runner/src/casm_run/mod.rs#L1825
    let y = rhs
        .sqrt()
        .unwrap_or_else(|| (Felt::THREE * rhs).sqrt().unwrap());
    let y = y.min(-y);

    match AffinePoint::new(x, y) {
        Ok(point) => {
            point_ptr[1] = point.y().to_bytes_le();
            true
        }
        Err(_) => false,
    }
}

/// Compute `ec_point_try_new_nz(x)`.
///
/// # Panics
///
/// This function will panic if either operand is out of range for a felt.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_point_try_new_nz(
    point_ptr: &mut [[u8; 32]; 2],
) -> bool {
    point_ptr[0][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
    point_ptr[1][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).

    let x = Felt::from_bytes_le(&point_ptr[0]);
    let y = Felt::from_bytes_le(&point_ptr[1]);

    match AffinePoint::new(x, y) {
        Ok(point) => {
            point_ptr[0] = point.x().to_bytes_le();
            point_ptr[1] = point.y().to_bytes_le();
            true
        }
        Err(_) => false,
    }
}

/// Compute `ec_state_init()` and store the state back.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_state_init(state_ptr: &mut [[u8; 32]; 4]) {
    // https://github.com/starkware-libs/cairo/blob/aaad921bba52e729dc24ece07fab2edf09ccfa15/crates/cairo-lang-runner/src/casm_run/mod.rs#L1802
    let mut rng = rand::rng();
    let (random_x, random_y) = loop {
        // Randominzing 31 bytes to make sure is in range.
        let x_bytes: [u8; 31] = rng.random();
        let random_x = Felt::from_bytes_be_slice(&x_bytes);
        let random_y_squared = random_x * random_x * random_x + random_x + BETA;
        if let Some(random_y) = random_y_squared.sqrt() {
            break (random_x, random_y);
        }
    };

    // We already made sure its a valid point.
    let state = AffinePoint::new_unchecked(random_x, random_y);

    state_ptr[0] = state.x().to_bytes_le();
    state_ptr[1] = state.y().to_bytes_le();
    state_ptr[2] = state_ptr[0];
    state_ptr[3] = state_ptr[1];
}

/// Compute `ec_state_add(state, point)` and store the state back.
///
/// # Panics
///
/// This function will panic if either operand is out of range for a felt.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_state_add(
    state_ptr: &mut [[u8; 32]; 4],
    point_ptr: &[[u8; 32]; 2],
) {
    state_ptr[0][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
    state_ptr[1][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).

    let mut point_ptr = *point_ptr;
    point_ptr[0][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
    point_ptr[1][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).

    // We use unchecked methods because the inputs must already be valid points.
    let mut state = ProjectivePoint::from_affine_unchecked(
        Felt::from_bytes_le(&state_ptr[0]),
        Felt::from_bytes_le(&state_ptr[1]),
    );
    let point = AffinePoint::new_unchecked(
        Felt::from_bytes_le(&point_ptr[0]),
        Felt::from_bytes_le(&point_ptr[1]),
    );

    state += &point;
    let state = state.to_affine().unwrap();

    state_ptr[0] = state.x().to_bytes_le();
    state_ptr[1] = state.y().to_bytes_le();
}

/// Compute `ec_state_add_mul(state, scalar, point)` and store the state back.
///
/// # Panics
///
/// This function will panic if either operand is out of range for a felt.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_state_add_mul(
    state_ptr: &mut [[u8; 32]; 4],
    scalar_ptr: &[u8; 32],
    point_ptr: &[[u8; 32]; 2],
) {
    state_ptr[0][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
    state_ptr[1][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).

    let mut point_ptr = *point_ptr;
    point_ptr[0][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
    point_ptr[1][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).

    let mut scalar_ptr = *scalar_ptr;
    scalar_ptr[31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).

    // Here the points should already be checked as valid, so we can use unchecked.
    let mut state = ProjectivePoint::from_affine_unchecked(
        Felt::from_bytes_le(&state_ptr[0]),
        Felt::from_bytes_le(&state_ptr[1]),
    );
    let point = ProjectivePoint::from_affine_unchecked(
        Felt::from_bytes_le(&point_ptr[0]),
        Felt::from_bytes_le(&point_ptr[1]),
    );
    let scalar = Felt::from_bytes_le(&scalar_ptr);

    state += &point.mul(scalar);
    let state = state.to_affine().unwrap();

    state_ptr[0] = state.x().to_bytes_le();
    state_ptr[1] = state.y().to_bytes_le();
}

/// Compute `ec_state_try_finalize_nz(state)` and store the result.
///
/// # Panics
///
/// This function will panic if either operand is out of range for a felt.
///
/// # Safety
///
/// This function is intended to be called from MLIR, deals with pointers, and is therefore
/// definitely unsafe to use manually.
pub unsafe extern "C" fn cairo_native__libfunc__ec__ec_state_try_finalize_nz(
    point_ptr: &mut [[u8; 32]; 2],
    state_ptr: &[[u8; 32]; 4],
) -> bool {
    let mut state_ptr = *state_ptr;
    state_ptr[0][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
    state_ptr[1][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
    state_ptr[2][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
    state_ptr[3][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).

    // We use unchecked methods because the inputs must already be valid points.
    let state = ProjectivePoint::from_affine_unchecked(
        Felt::from_bytes_le(&state_ptr[0]),
        Felt::from_bytes_le(&state_ptr[1]),
    );
    let random = ProjectivePoint::from_affine_unchecked(
        Felt::from_bytes_le(&state_ptr[2]),
        Felt::from_bytes_le(&state_ptr[3]),
    );

    if state.x() == random.x() && state.y() == random.y() {
        false
    } else {
        let point = &state - &random;
        let point = point.to_affine().unwrap();

        point_ptr[0] = point.x().to_bytes_le();
        point_ptr[1] = point.y().to_bytes_le();

        true
    }
}

thread_local! {
    pub(crate) static BUILTIN_COSTS: Cell<BuiltinCosts> = const {
        // These default values shouldn't be accessible, they will be overriden before entering
        // compiled code.
        Cell::new(BuiltinCosts {
            r#const: 0,
            pedersen: 0,
            bitwise: 0,
            ecop: 0,
            poseidon: 0,
            add_mod: 0,
            mul_mod: 0,
        })
    };
}

/// Get the gas builtin from the internal thread local.
pub extern "C" fn cairo_native__get_costs_builtin() -> *const [u64; 7] {
    BUILTIN_COSTS.with(|x| x.as_ptr()) as *const [u64; 7]
}

// Utility methods for the print runtime function

/// Formats the given felts as a debug string.
fn format_for_debug(mut felts: IntoIter<Felt>) -> String {
    let mut items = Vec::new();
    while let Some(item) = format_next_item(&mut felts) {
        items.push(item);
    }
    if let [item] = &items[..] {
        if item.is_string {
            return item.item.clone();
        }
    }
    items
        .into_iter()
        .map(|item| {
            if item.is_string {
                format!("{}\n", item.item)
            } else {
                format!("[DEBUG]\t{}\n", item.item)
            }
        })
        .join("")
}

/// A formatted string representation of anything formattable (e.g. ByteArray, felt, short-string).
pub struct FormattedItem {
    /// The formatted string representing the item.
    item: String,
    /// Whether the item is a string.
    is_string: bool,
}
impl FormattedItem {
    /// Returns the formatted item as is.
    #[must_use]
    pub fn get(self) -> String {
        self.item
    }
    /// Wraps the formatted item with quote, if it's a string. Otherwise returns it as is.
    #[must_use]
    pub fn quote_if_string(self) -> String {
        if self.is_string {
            format!("\"{}\"", self.item)
        } else {
            self.item
        }
    }
}

pub const BYTE_ARRAY_MAGIC: &str =
    "46a6158a16a947e5916b2a2ca68501a45e93d7110e81aa2d6438b1c57c879a3";
pub const BYTES_IN_WORD: usize = 31;

/// Formats a string or a short string / `felt252`. Returns the formatted string and a boolean
/// indicating whether it's a string. If can't format the item, returns None.
pub fn format_next_item<T>(values: &mut T) -> Option<FormattedItem>
where
    T: Iterator<Item = Felt> + Clone,
{
    let first_felt = values.next()?;

    if first_felt == Felt::from_hex(BYTE_ARRAY_MAGIC).unwrap() {
        if let Some(string) = try_format_string(values) {
            return Some(FormattedItem {
                item: string,
                is_string: true,
            });
        }
    }
    Some(FormattedItem {
        item: format_short_string(&first_felt),
        is_string: false,
    })
}

/// Formats a `Felt252`, as a short string if possible.
fn format_short_string(value: &Felt) -> String {
    let hex_value = value.to_biguint();
    match as_cairo_short_string(value) {
        Some(as_string) => format!("{hex_value:#x} ('{as_string}')"),
        None => format!("{hex_value:#x}"),
    }
}

/// Tries to format a string, represented as a sequence of `Felt252`s.
/// If the sequence is not a valid serialization of a `ByteArray`, returns None and doesn't change the
/// given iterator (`values`).
fn try_format_string<T>(values: &mut T) -> Option<String>
where
    T: Iterator<Item = Felt> + Clone,
{
    // Clone the iterator and work with the clone. If the extraction of the string is successful,
    // change the original iterator to the one we worked with. If not, continue with the
    // original iterator at the original point.
    let mut cloned_values_iter = values.clone();

    let num_full_words = cloned_values_iter.next()?.to_usize()?;
    let full_words = cloned_values_iter
        .by_ref()
        .take(num_full_words)
        .collect_vec();
    let pending_word = cloned_values_iter.next()?;
    let pending_word_len = cloned_values_iter.next()?.to_usize()?;

    let full_words_string = full_words
        .into_iter()
        .map(|word| as_cairo_short_string_ex(&word, BYTES_IN_WORD))
        .collect::<Option<Vec<String>>>()?
        .join("");
    let pending_word_string = as_cairo_short_string_ex(&pending_word, pending_word_len)?;

    // Extraction was successful, change the original iterator to the one we worked with.
    *values = cloned_values_iter;

    Some(format!("{full_words_string}{pending_word_string}"))
}

/// Converts a bigint representing a felt252 to a Cairo short-string.
#[must_use]
pub fn as_cairo_short_string(value: &Felt) -> Option<String> {
    let mut as_string = String::default();
    let mut is_end = false;
    for byte in value.to_biguint().to_bytes_be() {
        if byte == 0 {
            is_end = true;
        } else if is_end {
            return None;
        } else if byte.is_ascii_graphic() || byte.is_ascii_whitespace() {
            as_string.push(byte as char);
        } else {
            return None;
        }
    }
    Some(as_string)
}

/// Converts a bigint representing a felt252 to a Cairo short-string of the given length.
/// Nulls are allowed and length must be <= 31.
#[must_use]
pub fn as_cairo_short_string_ex(value: &Felt, length: usize) -> Option<String> {
    if length == 0 {
        return if value.is_zero() {
            Some(String::new())
        } else {
            None
        };
    }
    if length > 31 {
        // A short string can't be longer than 31 bytes.
        return None;
    }

    // We pass through biguint as felt252.to_bytes_be() does not trim leading zeros.
    let bytes = value.to_biguint().to_bytes_be();
    let bytes_len = bytes.len();
    if bytes_len > length {
        // `value` has more bytes than expected.
        return None;
    }

    let mut as_string = String::new();
    for byte in bytes {
        if byte == 0 {
            as_string.push_str(r"\0");
        } else if byte.is_ascii_graphic() || byte.is_ascii_whitespace() {
            as_string.push(byte as char);
        } else {
            as_string.push_str(format!(r"\x{:02x}", byte).as_str());
        }
    }

    // `to_bytes_be` misses starting nulls. Prepend them as needed.
    let missing_nulls = length - bytes_len;
    as_string.insert_str(0, &r"\0".repeat(missing_nulls));

    Some(as_string)
}

#[cfg(feature = "with-trace-dump")]
pub mod trace_dump {
    use cairo_lang_sierra::{
        extensions::{
            bounded_int::BoundedIntConcreteType,
            circuit::CircuitTypeConcrete,
            core::{CoreLibfunc, CoreType, CoreTypeConcrete},
            starknet::{secp256::Secp256PointTypeConcrete, StarkNetTypeConcrete},
        },
        ids::{ConcreteTypeId, VarId},
        program::{GenericArg, StatementIdx},
        program_registry::ProgramRegistry,
    };
    use cairo_lang_utils::ordered_hash_map::OrderedHashMap;
    use itertools::Itertools;
    use num_bigint::{BigInt, BigUint};
    use num_traits::One;
    use sierra_emu::{
        starknet::{
            Secp256k1Point as EmuSecp256k1Point, Secp256r1Point as EmuSecp256r1Point,
            U256 as EmuU256,
        },
        ProgramTrace, StateDump, Value,
    };
    use starknet_types_core::felt::Felt;
    use std::{
        alloc::Layout,
        collections::HashMap,
        mem::swap,
        ops::Range,
        ptr::NonNull,
        sync::{LazyLock, Mutex},
    };

    use super::FeltDict;

    pub static TRACE_DUMP: LazyLock<Mutex<HashMap<u64, TraceDump>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

    pub struct TraceDump {
        pub trace: ProgramTrace,
        state: OrderedHashMap<VarId, Value>,
        registry: ProgramRegistry<CoreType, CoreLibfunc>,

        get_layout: fn(&CoreTypeConcrete, &ProgramRegistry<CoreType, CoreLibfunc>) -> Layout,
    }

    impl TraceDump {
        pub fn new(
            registry: ProgramRegistry<CoreType, CoreLibfunc>,
            get_layout: fn(&CoreTypeConcrete, &ProgramRegistry<CoreType, CoreLibfunc>) -> Layout,
        ) -> Self {
            Self {
                trace: ProgramTrace::default(),
                state: OrderedHashMap::default(),
                registry,

                get_layout,
            }
        }
    }

    pub extern "C" fn get_trace_dump_ptr() -> *const Mutex<HashMap<u64, TraceDump>> {
        &*TRACE_DUMP as *const _
    }

    pub unsafe extern "C" fn cairo_native__trace_dump__state(
        trace_id: u64,
        var_id: u64,
        type_id: u64,
        value_ptr: NonNull<()>,
    ) {
        let mut trace_dump = TRACE_DUMP.lock().unwrap();
        let trace_dump = trace_dump.get_mut(&trace_id).unwrap();

        let type_id = ConcreteTypeId::new(type_id);
        let value = read_value_ptr(
            &trace_dump.registry,
            &type_id,
            value_ptr,
            trace_dump.get_layout,
        );

        trace_dump.state.insert(VarId::new(var_id), value);
    }

    pub unsafe extern "C" fn cairo_native__trace_dump__push(trace_id: u64, statement_idx: u64) {
        let mut trace_dump = TRACE_DUMP.lock().unwrap();
        let trace_dump = trace_dump.get_mut(&trace_id).unwrap();

        let mut items = OrderedHashMap::default();
        swap(&mut items, &mut trace_dump.state);

        trace_dump
            .trace
            .push(StateDump::new(StatementIdx(statement_idx as usize), items));
    }

    unsafe fn read_value_ptr(
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        type_id: &ConcreteTypeId,
        value_ptr: NonNull<()>,
        get_layout: fn(&CoreTypeConcrete, &ProgramRegistry<CoreType, CoreLibfunc>) -> Layout,
    ) -> Value {
        let type_info = registry.get_type(type_id).unwrap();
        match type_info {
            CoreTypeConcrete::Felt252(_)
            | CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::ContractAddress(_))
            | CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::ClassHash(_))
            | CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::StorageAddress(_))
            | CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::StorageBaseAddress(_)) => {
                Value::Felt(Felt::from_bytes_le(value_ptr.cast().as_ref()))
            }
            CoreTypeConcrete::Uint8(_) => Value::U8(value_ptr.cast().read()),
            CoreTypeConcrete::Uint16(_) => Value::U16(value_ptr.cast().read()),
            CoreTypeConcrete::Uint32(_) => Value::U32(value_ptr.cast().read()),
            CoreTypeConcrete::Uint64(_) | CoreTypeConcrete::GasBuiltin(_) => {
                Value::U64(value_ptr.cast().read())
            }
            CoreTypeConcrete::Uint128(_) => Value::U128(value_ptr.cast().read()),

            CoreTypeConcrete::BoundedInt(BoundedIntConcreteType { range, .. }) => {
                let n_bits = ((range.size() - BigInt::one()).bits() as u32).max(1);
                let n_bytes = n_bits.next_multiple_of(8) >> 3;

                let data = NonNull::slice_from_raw_parts(value_ptr.cast::<u8>(), n_bytes as usize);

                let value = BigInt::from_bytes_le(num_bigint::Sign::Plus, data.as_ref());

                Value::BoundedInt {
                    range: Range {
                        start: range.lower.clone(),
                        end: range.upper.clone(),
                    },
                    value: value + &range.lower,
                }
            }

            CoreTypeConcrete::EcPoint(_) => {
                let layout = Layout::new::<()>();
                let (x, layout) = {
                    let (layout, offset) = layout.extend(Layout::new::<[u128; 2]>()).unwrap();
                    (
                        Felt::from_bytes_le(value_ptr.byte_add(offset).cast().as_ref()),
                        layout,
                    )
                };
                let (y, _) = {
                    let (layout, offset) = layout.extend(Layout::new::<[u128; 2]>()).unwrap();
                    (
                        Felt::from_bytes_le(value_ptr.byte_add(offset).cast().as_ref()),
                        layout,
                    )
                };

                Value::EcPoint { x, y }
            }
            CoreTypeConcrete::EcState(_) => {
                let layout = Layout::new::<()>();
                let (x0, layout) = {
                    let (layout, offset) = layout.extend(Layout::new::<[u128; 2]>()).unwrap();
                    (
                        Felt::from_bytes_le(value_ptr.byte_add(offset).cast().as_ref()),
                        layout,
                    )
                };
                let (y0, layout) = {
                    let (layout, offset) = layout.extend(Layout::new::<[u128; 2]>()).unwrap();
                    (
                        Felt::from_bytes_le(value_ptr.byte_add(offset).cast().as_ref()),
                        layout,
                    )
                };
                let (x1, layout) = {
                    let (layout, offset) = layout.extend(Layout::new::<[u128; 2]>()).unwrap();
                    (
                        Felt::from_bytes_le(value_ptr.byte_add(offset).cast().as_ref()),
                        layout,
                    )
                };
                let (y1, _) = {
                    let (layout, offset) = layout.extend(Layout::new::<[u128; 2]>()).unwrap();
                    (
                        Felt::from_bytes_le(value_ptr.byte_add(offset).cast().as_ref()),
                        layout,
                    )
                };

                Value::EcState { x0, y0, x1, y1 }
            }

            CoreTypeConcrete::Uninitialized(info) => Value::Uninitialized {
                ty: info.ty.clone(),
            },
            CoreTypeConcrete::Box(info) => read_value_ptr(
                registry,
                &info.ty,
                value_ptr.cast::<NonNull<()>>().read(),
                get_layout,
            ),
            CoreTypeConcrete::Array(info) => {
                let layout = Layout::new::<()>();
                let (array_ptr, layout) = {
                    let (layout, offset) = layout.extend(Layout::new::<*mut ()>()).unwrap();
                    (value_ptr.byte_add(offset).cast::<*mut ()>().read(), layout)
                };
                let (array_begin, layout) = {
                    let (layout, offset) = layout.extend(Layout::new::<u32>()).unwrap();
                    (value_ptr.byte_add(offset).cast::<u32>().read(), layout)
                };
                let (array_end, _) = {
                    let (layout, offset) = layout.extend(Layout::new::<u32>()).unwrap();
                    (value_ptr.byte_add(offset).cast::<u32>().read(), layout)
                };

                let layout =
                    get_layout(registry.get_type(&info.ty).unwrap(), registry).pad_to_align();

                let mut data = Vec::with_capacity((array_end - array_begin) as usize);
                for index in array_begin..array_end {
                    let index = index as usize;

                    data.push(read_value_ptr(
                        registry,
                        &info.ty,
                        NonNull::new(array_ptr.byte_add(layout.size() * index)).unwrap(),
                        get_layout,
                    ));
                }

                Value::Array {
                    ty: info.ty.clone(),
                    data,
                }
            }

            CoreTypeConcrete::Struct(info) => {
                let mut layout = Layout::new::<()>();
                let mut members = Vec::with_capacity(info.members.len());
                for member_ty in &info.members {
                    let type_info = registry.get_type(member_ty).unwrap();
                    let member_layout = get_layout(type_info, registry);

                    let offset;
                    (layout, offset) = layout.extend(member_layout).unwrap();

                    let current_ptr = value_ptr.byte_add(offset);
                    members.push(read_value_ptr(registry, member_ty, current_ptr, get_layout));
                }

                Value::Struct(members)
            }
            CoreTypeConcrete::Enum(info) => {
                let tag_bits = info.variants.len().next_power_of_two().trailing_zeros();
                let (tag_value, layout) = match tag_bits {
                    0 => (0, Layout::new::<()>()),
                    width if width <= 8 => {
                        (value_ptr.cast::<u8>().read() as usize, Layout::new::<u8>())
                    }
                    width if width <= 16 => (
                        value_ptr.cast::<u16>().read() as usize,
                        Layout::new::<u16>(),
                    ),
                    width if width <= 32 => (
                        value_ptr.cast::<u32>().read() as usize,
                        Layout::new::<u32>(),
                    ),
                    width if width <= 64 => (
                        value_ptr.cast::<u64>().read() as usize,
                        Layout::new::<u64>(),
                    ),
                    width if width <= 128 => (
                        value_ptr.cast::<u128>().read() as usize,
                        Layout::new::<u128>(),
                    ),
                    _ => todo!(),
                };

                let payload = {
                    let (_, offset) = layout
                        .extend(get_layout(
                            registry.get_type(&info.variants[tag_value]).unwrap(),
                            registry,
                        ))
                        .unwrap();

                    read_value_ptr(
                        registry,
                        &info.variants[tag_value],
                        value_ptr.byte_add(offset),
                        get_layout,
                    )
                };

                Value::Enum {
                    self_ty: type_id.clone(),
                    index: tag_value,
                    payload: Box::new(payload),
                }
            }

            CoreTypeConcrete::NonZero(info) | CoreTypeConcrete::Snapshot(info) => {
                read_value_ptr(registry, &info.ty, value_ptr, get_layout)
            }

            // Builtins and other unit types:
            CoreTypeConcrete::Bitwise(_)
            | CoreTypeConcrete::EcOp(_)
            | CoreTypeConcrete::Pedersen(_)
            | CoreTypeConcrete::Poseidon(_)
            | CoreTypeConcrete::RangeCheck96(_)
            | CoreTypeConcrete::RangeCheck(_)
            | CoreTypeConcrete::SegmentArena(_)
            | CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::System(_))
            | CoreTypeConcrete::Uint128MulGuarantee(_) => Value::Unit,

            CoreTypeConcrete::BuiltinCosts(_) => {
                let builtin_costs = value_ptr.cast::<&[u64; 7]>().read();
                Value::BuiltinCosts(sierra_emu::BuiltinCosts {
                    r#const: builtin_costs[0],
                    pedersen: builtin_costs[1],
                    bitwise: builtin_costs[2],
                    ecop: builtin_costs[3],
                    poseidon: builtin_costs[4],
                    add_mod: builtin_costs[5],
                    mul_mod: builtin_costs[6],
                })
            }

            // TODO:
            CoreTypeConcrete::Coupon(_) => todo!("CoreTypeConcrete::Coupon"),
            CoreTypeConcrete::Circuit(circuit) => match circuit {
                CircuitTypeConcrete::AddMod(_) => Value::Unit,
                CircuitTypeConcrete::MulMod(_) => Value::Unit,
                CircuitTypeConcrete::AddModGate(_) => Value::Unit,
                CircuitTypeConcrete::Circuit(_) => Value::Unit,
                CircuitTypeConcrete::CircuitData(info) => {
                    let Some(GenericArg::Type(circuit_type_id)) =
                        info.info.long_id.generic_args.first()
                    else {
                        panic!("generic arg should be a type");
                    };
                    let CoreTypeConcrete::Circuit(CircuitTypeConcrete::Circuit(circuit)) =
                        registry.get_type(circuit_type_id).unwrap()
                    else {
                        panic!("generic arg should be a Circuit");
                    };

                    let u384_layout = Layout::from_size_align(48, 16).unwrap();

                    let n_inputs = circuit.circuit_info.n_inputs;
                    let mut values = Vec::with_capacity(n_inputs);

                    let value_ptr = value_ptr.cast::<[u8; 48]>();

                    for i in 0..n_inputs {
                        let size = u384_layout.pad_to_align().size();
                        let current_ptr = value_ptr.byte_add(size * i);
                        let current_value = current_ptr.as_ref();
                        values.push(BigUint::from_bytes_le(current_value));
                    }

                    Value::Circuit(values)
                }
                CircuitTypeConcrete::CircuitOutputs(info) => {
                    let Some(GenericArg::Type(circuit_type_id)) =
                        info.info.long_id.generic_args.first()
                    else {
                        panic!("generic arg should be a type");
                    };
                    let CoreTypeConcrete::Circuit(CircuitTypeConcrete::Circuit(circuit)) =
                        registry.get_type(circuit_type_id).unwrap()
                    else {
                        panic!("generic arg should be a Circuit");
                    };

                    let u384_layout = Layout::from_size_align(48, 16).unwrap();

                    let n_outputs = circuit.circuit_info.values.len();
                    let mut values = Vec::with_capacity(n_outputs);

                    let value_ptr = value_ptr.cast::<[u8; 48]>();

                    for i in 0..n_outputs {
                        let size = u384_layout.pad_to_align().size();
                        let current_ptr = value_ptr.byte_add(size * i);
                        let current_value = current_ptr.as_ref();
                        values.push(BigUint::from_bytes_le(current_value));
                    }

                    Value::CircuitOutputs(values)
                }
                CircuitTypeConcrete::CircuitPartialOutputs(_) => {
                    todo!("CircuitTypeConcrete::CircuitPartialOutputs")
                }
                CircuitTypeConcrete::CircuitDescriptor(_) => Value::Unit,
                CircuitTypeConcrete::CircuitFailureGuarantee(_) => {
                    todo!("CircuitTypeConcrete::CircuitFailureGuarantee")
                }
                CircuitTypeConcrete::CircuitInput(_) => {
                    todo!("CircuitTypeConcrete::CircuitInput")
                }
                CircuitTypeConcrete::CircuitInputAccumulator(info) => {
                    let Some(GenericArg::Type(circuit_type_id)) =
                        info.info.long_id.generic_args.first()
                    else {
                        panic!("generic arg should be a type");
                    };
                    let CoreTypeConcrete::Circuit(CircuitTypeConcrete::Circuit(_)) =
                        registry.get_type(circuit_type_id).unwrap()
                    else {
                        panic!("generic arg should be a Circuit");
                    };

                    let u64_layout = Layout::new::<u64>();
                    let u384_layout = Layout::from_size_align(48, 16).unwrap();

                    let length = unsafe { *value_ptr.cast::<u64>().as_ptr() };

                    let (_, input_start_offset) = u64_layout.extend(u384_layout).unwrap();
                    let start_ptr = value_ptr.byte_add(input_start_offset).cast::<[u8; 48]>();

                    let mut values = Vec::with_capacity(length as usize);

                    for i in 0..length {
                        let size = u384_layout.pad_to_align().size();
                        let current_ptr = start_ptr.byte_add(size * i as usize);
                        let current_value = current_ptr.as_ref();
                        values.push(BigUint::from_bytes_le(current_value));
                    }

                    Value::Circuit(values)
                }
                CircuitTypeConcrete::CircuitModulus(_) => {
                    let value_ptr = value_ptr.cast::<[u8; 48]>();
                    let value = unsafe { value_ptr.as_ref() };
                    Value::CircuitModulus(BigUint::from_bytes_le(value))
                }

                CircuitTypeConcrete::InverseGate(_) => Value::Unit,
                CircuitTypeConcrete::MulModGate(_) => Value::Unit,
                CircuitTypeConcrete::SubModGate(_) => Value::Unit,
                CircuitTypeConcrete::U96Guarantee(_) => {
                    let value_ptr = value_ptr.cast::<[u8; 12]>();
                    let value = unsafe { value_ptr.as_ref() };

                    let mut array_value = [0u8; 16];
                    array_value[..12].clone_from_slice(value);

                    Value::U128(u128::from_le_bytes(array_value))
                }
                CircuitTypeConcrete::U96LimbsLessThanGuarantee(_) => Value::Unit,
            },
            CoreTypeConcrete::Const(_) => todo!("CoreTypeConcrete::Const"),
            CoreTypeConcrete::Sint8(_) => Value::I8(value_ptr.cast().read()),
            CoreTypeConcrete::Sint16(_) => todo!("CoreTypeConcrete::Sint16"),
            CoreTypeConcrete::Sint32(_) => Value::I32(value_ptr.cast().read()),
            CoreTypeConcrete::Sint64(_) => todo!("CoreTypeConcrete::Sint64"),
            CoreTypeConcrete::Sint128(_) => Value::I128(value_ptr.cast().read()),
            CoreTypeConcrete::Nullable(info) => {
                let inner_ptr = value_ptr.cast::<*mut ()>().read();
                match NonNull::new(inner_ptr) {
                    Some(inner_ptr) => read_value_ptr(registry, &info.ty, inner_ptr, get_layout),
                    None => Value::Uninitialized {
                        ty: info.ty.clone(),
                    },
                }
            }

            CoreTypeConcrete::SquashedFelt252Dict(info) | CoreTypeConcrete::Felt252Dict(info) => {
                let value = value_ptr.cast::<&FeltDict>().read();

                let data = value
                    .mappings
                    .iter()
                    .map(|(k, &i)| {
                        let p = value
                            .elements
                            .byte_offset((value.layout.size() * i) as isize);
                        let v = match NonNull::new(p) {
                            Some(value_ptr) => {
                                read_value_ptr(registry, &info.ty, value_ptr.cast(), get_layout)
                            }
                            None => Value::Uninitialized {
                                ty: info.ty.clone(),
                            },
                        };
                        let k = Felt::from_bytes_le(k);
                        (k, v)
                    })
                    .collect::<HashMap<Felt, Value>>();

                Value::FeltDict {
                    ty: info.ty.clone(),
                    count: value.count,
                    data,
                }
            }
            CoreTypeConcrete::Felt252DictEntry(info) => {
                let value = value_ptr.cast::<FeltDictEntry>().read();

                let data = value
                    .dict
                    .mappings
                    .iter()
                    .map(|(k, &i)| {
                        let p = value
                            .dict
                            .elements
                            .byte_offset((value.dict.layout.size() * i) as isize);
                        let v = match NonNull::new(p) {
                            Some(value_ptr) => {
                                read_value_ptr(registry, &info.ty, value_ptr.cast(), get_layout)
                            }
                            None => Value::Uninitialized {
                                ty: info.ty.clone(),
                            },
                        };
                        let k = Felt::from_bytes_le(k);
                        (k, v)
                    })
                    .collect::<HashMap<Felt, Value>>();
                let key = Felt::from_bytes_le(value.key);

                Value::FeltDictEntry {
                    ty: info.ty.clone(),
                    data,
                    count: value.dict.count,
                    key,
                }
            }
            CoreTypeConcrete::Span(_) => todo!("CoreTypeConcrete::Span"),
            CoreTypeConcrete::StarkNet(selector) => match selector {
                StarkNetTypeConcrete::Secp256Point(selector) => match selector {
                    Secp256PointTypeConcrete::K1(_) => {
                        let point: Secp256Point = value_ptr.cast().read();
                        let emu_point = EmuSecp256k1Point {
                            x: EmuU256 {
                                lo: point.x.lo,
                                hi: point.x.hi,
                            },
                            y: EmuU256 {
                                lo: point.y.lo,
                                hi: point.y.hi,
                            },
                        };
                        emu_point.into_value()
                    }
                    Secp256PointTypeConcrete::R1(_) => {
                        let point: Secp256Point = value_ptr.cast().read();
                        let emu_point = EmuSecp256r1Point {
                            x: EmuU256 {
                                lo: point.x.lo,
                                hi: point.x.hi,
                            },
                            y: EmuU256 {
                                lo: point.y.lo,
                                hi: point.y.hi,
                            },
                        };
                        emu_point.into_value()
                    }
                },
                StarkNetTypeConcrete::Sha256StateHandle(_) => {
                    let raw_data = value_ptr.cast::<NonNull<[u32; 8]>>().read().read();
                    let data = raw_data.into_iter().map(Value::U32).collect_vec();
                    Value::Struct(data)
                }
                _ => unreachable!(),
            },
            CoreTypeConcrete::Bytes31(_) => {
                let original_data: [u8; 31] = value_ptr.cast().read();
                let mut data = [0u8; 32];
                for (i, v) in original_data.into_iter().enumerate() {
                    data[i] = v
                }

                Value::Bytes31(Felt::from_bytes_le(&data))
            }
            CoreTypeConcrete::IntRange(_) => todo!(),
        }
    }

    #[derive(Debug)]
    struct FeltDictEntry<'a> {
        dict: &'a FeltDict,
        key: &'a [u8; 32],
    }

    #[repr(C, align(16))]
    pub struct Secp256Point {
        pub x: U256,
        pub y: U256,
        pub is_infinity: bool,
    }

    #[repr(C, align(16))]
    pub struct U256 {
        pub lo: u128,
        pub hi: u128,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        env, fs,
        io::{Read, Seek},
        os::fd::AsRawFd,
    };

    pub fn felt252_short_str(value: &str) -> Felt {
        let values: Vec<_> = value
            .chars()
            .filter_map(|c| c.is_ascii().then_some(c as u8))
            .collect();

        assert!(values.len() < 32);
        Felt::from_bytes_be_slice(&values)
    }

    #[test]
    fn test_debug_print() {
        let dir = env::temp_dir();
        fs::remove_file(dir.join("print.txt")).ok();
        let mut file = File::create_new(dir.join("print.txt")).unwrap();
        {
            let fd = file.as_raw_fd();
            let data = felt252_short_str("hello world");
            let data = data.to_bytes_le();
            unsafe { cairo_native__libfunc__debug__print(fd, &data, 1) };
        }
        file.seek(std::io::SeekFrom::Start(0)).unwrap();

        let mut result = String::new();
        file.read_to_string(&mut result).unwrap();

        assert_eq!(
            result,
            "[DEBUG]\t0x68656c6c6f20776f726c64 ('hello world')\n"
        );
    }

    #[test]
    fn test_pederesen() {
        let mut dst = [0; 32];
        let lhs = Felt::from(1).to_bytes_le();
        let rhs = Felt::from(3).to_bytes_le();

        unsafe {
            cairo_native__libfunc__pedersen(&mut dst, &lhs, &rhs);
        }

        assert_eq!(
            dst,
            [
                84, 98, 174, 134, 3, 124, 237, 179, 166, 110, 159, 98, 170, 35, 83, 237, 130, 154,
                236, 0, 205, 134, 200, 185, 39, 92, 0, 228, 132, 217, 130, 5
            ]
        )
    }

    #[test]
    fn test_hades_permutation() {
        let mut op0 = Felt::from(1).to_bytes_le();
        let mut op1 = Felt::from(1).to_bytes_le();
        let mut op2 = Felt::from(1).to_bytes_le();

        unsafe {
            cairo_native__libfunc__hades_permutation(&mut op0, &mut op1, &mut op2);
        }

        assert_eq!(
            Felt::from_bytes_le(&op0),
            Felt::from_hex("0x4ebdde1149fcacbb41e4fc342432a48c97994fd045f432ad234ae9279269779")
                .unwrap()
        );
        assert_eq!(
            Felt::from_bytes_le(&op1),
            Felt::from_hex("0x7f4cec57dd08b69414f7de7dffa230fc90fa3993673c422408af05831e0cc98")
                .unwrap()
        );
        assert_eq!(
            Felt::from_bytes_le(&op2),
            Felt::from_hex("0x5b5d00fd09caade43caffe70527fa84d5d9cd51e22c2ce115693ecbb5854d6a")
                .unwrap()
        );
    }

    #[test]
    fn test_dict() {
        let dict = unsafe {
            cairo_native__dict_new(
                size_of::<u64>() as u64,
                align_of::<u64>() as u64,
                None,
                None,
            )
        };

        let key = Felt::ONE.to_bytes_le();
        let mut ptr = ptr::null_mut::<u64>();

        assert_eq!(
            unsafe { cairo_native__dict_get(dict, &key, (&raw mut ptr).cast()) },
            0,
        );
        assert!(!ptr.is_null());
        unsafe { *ptr = 24 };

        assert_eq!(
            unsafe { cairo_native__dict_get(dict, &key, (&raw mut ptr).cast()) },
            1,
        );
        assert!(!ptr.is_null());
        assert_eq!(unsafe { *ptr }, 24);
        unsafe { *ptr = 42 };

        let refund = unsafe { cairo_native__dict_gas_refund(dict) };
        assert_eq!(refund, 4050);

        let cloned_dict = unsafe { cairo_native__dict_dup(&*dict) };
        unsafe { cairo_native__dict_drop(dict) };

        assert_eq!(
            unsafe { cairo_native__dict_get(cloned_dict, &key, (&raw mut ptr).cast()) },
            1,
        );
        assert!(!ptr.is_null());
        assert_eq!(unsafe { *ptr }, 42);

        unsafe { cairo_native__dict_drop(cloned_dict) };
    }

    #[test]
    fn test_ec__ec_point() {
        let mut state = [
            Felt::ZERO.to_bytes_le(),
            Felt::ZERO.to_bytes_le(),
            Felt::ZERO.to_bytes_le(),
            Felt::ZERO.to_bytes_le(),
        ];

        unsafe { cairo_native__libfunc__ec__ec_state_init(&mut state) };

        let points: &mut [[u8; 32]; 2] = (&mut state[..2]).try_into().unwrap();

        let result = unsafe { cairo_native__libfunc__ec__ec_point_try_new_nz(points) };

        // point should be valid since it was made with state init
        assert!(result);
    }

    #[test]
    fn test_ec__ec_point_add() {
        // Test values taken from starknet-rs
        let mut state = [
            Felt::from_dec_str(
                "874739451078007766457464989774322083649278607533249481151382481072868806602",
            )
            .unwrap()
            .to_bytes_le(),
            Felt::from_dec_str(
                "152666792071518830868575557812948353041420400780739481342941381225525861407",
            )
            .unwrap()
            .to_bytes_le(),
            Felt::from_dec_str(
                "874739451078007766457464989774322083649278607533249481151382481072868806602",
            )
            .unwrap()
            .to_bytes_le(),
            Felt::from_dec_str(
                "152666792071518830868575557812948353041420400780739481342941381225525861407",
            )
            .unwrap()
            .to_bytes_le(),
        ];

        let point = [
            Felt::from_dec_str(
                "874739451078007766457464989774322083649278607533249481151382481072868806602",
            )
            .unwrap()
            .to_bytes_le(),
            Felt::from_dec_str(
                "152666792071518830868575557812948353041420400780739481342941381225525861407",
            )
            .unwrap()
            .to_bytes_le(),
        ];

        unsafe {
            cairo_native__libfunc__ec__ec_state_add(&mut state, &point);
        };

        assert_eq!(
            state[0],
            Felt::from_dec_str(
                "3324833730090626974525872402899302150520188025637965566623476530814354734325",
            )
            .unwrap()
            .to_bytes_le()
        );
        assert_eq!(
            state[1],
            Felt::from_dec_str(
                "3147007486456030910661996439995670279305852583596209647900952752170983517249",
            )
            .unwrap()
            .to_bytes_le()
        );
    }
}
