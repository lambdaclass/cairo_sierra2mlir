//! # Params and return values de/serialization
//!
//! A Rusty interface to provide parameters to cairo-native entry point calls.

use crate::{
    error::{panic::ToNativeAssertError, CompilerError, Error},
    native_assert, native_panic,
    runtime::FeltDict,
    starknet::{Secp256k1Point, Secp256r1Point},
    types::TypeBuilder,
    utils::{
        felt252_bigint, get_integer_layout, layout_repeat, libc_free, libc_malloc, RangeExt, PRIME,
    },
};
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType, CoreTypeConcrete},
        starknet::{secp256::Secp256PointTypeConcrete, StarkNetTypeConcrete},
        utils::Range,
    },
    ids::ConcreteTypeId,
    program_registry::ProgramRegistry,
};
use educe::Educe;
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::{Euclid, One};
use starknet_types_core::felt::Felt;
use std::{
    alloc::{alloc, Layout},
    collections::HashMap,
    ffi::c_void,
    mem::forget,
    ptr::{null_mut, NonNull},
    rc::Rc,
    slice,
};

/// A Value is a value that can be passed to either the JIT engine or a compiled program as an argument or received as a result.
///
/// They map to the cairo/sierra types.
///
/// The debug_name field on some variants is `Some` when receiving a [`Value`] as a result.
///
/// A Boxed value or a non-null Nullable value is returned with it's inner value.
#[derive(Clone, Educe, serde::Serialize, serde::Deserialize)]
#[educe(Debug, Eq, PartialEq)]
pub enum Value {
    Felt252(#[educe(Debug(method(std::fmt::Display::fmt)))] Felt),
    Bytes31([u8; 31]),
    /// all elements need to be same type
    Array(Vec<Self>),
    Struct {
        fields: Vec<Self>,
        #[educe(PartialEq(ignore))]
        debug_name: Option<String>,
    }, // element types can differ
    Enum {
        tag: usize,
        value: Box<Self>,
        #[educe(PartialEq(ignore))]
        debug_name: Option<String>,
    },
    Felt252Dict {
        value: HashMap<Felt, Self>,
        #[educe(PartialEq(ignore))]
        debug_name: Option<String>,
    },
    Uint8(u8),
    Uint16(u16),
    Uint32(u32),
    Uint64(u64),
    Uint128(u128),
    Sint8(i8),
    Sint16(i16),
    Sint32(i32),
    Sint64(i64),
    Sint128(i128),
    EcPoint(Felt, Felt),
    EcState(Felt, Felt, Felt, Felt),
    Secp256K1Point(Secp256k1Point),
    Secp256R1Point(Secp256r1Point),
    BoundedInt {
        value: Felt,
        #[serde(with = "range_serde")]
        range: Range,
    },
    IntRange {
        x: Box<Value>,
        y: Box<Value>,
    },
    /// Used as return value for Nullables that are null.
    Null,
}

// Conversions

macro_rules! impl_conversions {
    ( $( $t:ty as $i:ident ; )+ ) => { $(
        impl From<$t> for Value {
            fn from(value: $t) -> Self {
                Self::$i(value)
            }
        }

        impl TryFrom<Value> for $t {
            type Error = Value;

            fn try_from(value: Value) -> Result<Self, Self::Error> {
                match value {
                    Value::$i(value) => Ok(value),
                    _ => Err(value),
                }
            }
        }
    )+ };
}

impl_conversions! {
    Felt as Felt252;
    u8   as Uint8;
    u16  as Uint16;
    u32  as Uint32;
    u64  as Uint64;
    u128 as Uint128;
    i8   as Sint8;
    i16  as Sint16;
    i32  as Sint32;
    i64  as Sint64;
    i128 as Sint128;
}

impl<T: Into<Value> + Clone> From<&[T]> for Value {
    fn from(value: &[T]) -> Self {
        Self::Array(value.iter().map(|x| x.clone().into()).collect())
    }
}

impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(value: Vec<T>) -> Self {
        Self::Array(value.into_iter().map(Into::into).collect())
    }
}

impl<T: Into<Value>, const N: usize> From<[T; N]> for Value {
    fn from(value: [T; N]) -> Self {
        Self::Array(value.into_iter().map(Into::into).collect())
    }
}

impl Value {
    pub(crate) fn resolve_type<'a>(
        ty: &'a CoreTypeConcrete,
        registry: &'a ProgramRegistry<CoreType, CoreLibfunc>,
    ) -> Result<&'a CoreTypeConcrete, Error> {
        Ok(match ty {
            CoreTypeConcrete::Snapshot(info) => registry.get_type(&info.ty)?,
            x => x,
        })
    }

    /// Allocates the value in the given arena so it can be passed to the JIT engine or a compiled program.
    pub(crate) fn to_ptr(
        &self,
        arena: &Bump,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        type_id: &ConcreteTypeId,
        find_dict_overrides: impl Copy
            + Fn(
                &ConcreteTypeId,
            ) -> (
                Option<extern "C" fn(*mut c_void, *mut c_void)>,
                Option<extern "C" fn(*mut c_void)>,
            ),
    ) -> Result<NonNull<()>, Error> {
        let ty = registry.get_type(type_id)?;

        match ty {
            CoreTypeConcrete::Box(info)
            | CoreTypeConcrete::NonZero(info)
            | CoreTypeConcrete::Nullable(info)
            | CoreTypeConcrete::Snapshot(info) => {
                return self.to_ptr(arena, registry, &info.ty, find_dict_overrides);
            }
            _ => {}
        }

        Ok(unsafe {
            match self {
                Self::Felt252(value) => {
                    let ptr = arena.alloc_layout(get_integer_layout(252)).cast();

                    let data = felt252_bigint(value.to_bigint()).to_bytes_le();
                    ptr.cast::<[u8; 32]>().as_mut().copy_from_slice(&data);
                    ptr
                }
                Self::BoundedInt {
                    value,
                    range: Range { lower, upper },
                } => {
                    let value = value.to_bigint();

                    if lower >= upper {
                        // If lower bound is greater than or equal to upper bound
                        // Should not happen with correct range definition
                        return Err(CompilerError::BoundedIntOutOfRange {
                            value: Box::new(value),
                            range: Box::new((lower.clone(), upper.clone())),
                        }
                        .into());
                    }

                    let prime = BigInt::from_biguint(Sign::Plus, PRIME.clone());
                    let lower = lower.rem_euclid(&prime);
                    let upper = upper.rem_euclid(&prime);

                    // Check if value is within the valid range
                    if !(lower <= value && value < upper) {
                        return Err(CompilerError::BoundedIntOutOfRange {
                            value: Box::new(value),
                            range: Box::new((lower, upper)),
                        }
                        .into());
                    }

                    let ptr = arena.alloc_layout(get_integer_layout(252)).cast();
                    let data = felt252_bigint(value).to_bytes_le();
                    ptr.cast::<[u8; 32]>().as_mut().copy_from_slice(&data);
                    ptr
                }

                Self::Bytes31(_) => native_panic!("todo: allocate type Bytes31"),
                Self::Array(data) => {
                    if let CoreTypeConcrete::Array(info) = Self::resolve_type(ty, registry)? {
                        let elem_ty = registry.get_type(&info.ty)?;
                        let elem_layout = elem_ty.layout(registry)?.pad_to_align();

                        let refcount_offset =
                            crate::types::array::calc_data_prefix_offset(elem_layout);
                        let len: u32 = data
                            .len()
                            .try_into()
                            .map_err(|_| Error::IntegerConversion)?;
                        let ptr: *mut () = match len {
                            0 => std::ptr::null_mut(),
                            _ => {
                                let ptr: *mut () =
                                    libc_malloc(elem_layout.size() * data.len() + refcount_offset)
                                        .cast();

                                // Write reference count.
                                ptr.cast::<(u32, u32)>().write((1, len));
                                ptr.byte_add(refcount_offset).cast()
                            }
                        };

                        // Write the data.
                        for (idx, elem) in data.iter().enumerate() {
                            let elem =
                                elem.to_ptr(arena, registry, &info.ty, find_dict_overrides)?;

                            std::ptr::copy_nonoverlapping(
                                elem.cast::<u8>().as_ptr(),
                                ptr.byte_add(idx * elem_layout.size()).cast::<u8>(),
                                elem_layout.size(),
                            );
                        }

                        // Make double pointer.
                        let ptr_ptr = if ptr.is_null() {
                            null_mut()
                        } else {
                            let ptr_ptr: *mut *mut () = libc_malloc(8).cast();
                            ptr_ptr.write(ptr);
                            ptr_ptr
                        };

                        let target = arena
                            .alloc_layout(
                                Layout::new::<*mut ()>() // ptr
                                    .extend(Layout::new::<u32>())? // start
                                    .0
                                    .extend(Layout::new::<u32>())? // end
                                    .0
                                    .extend(Layout::new::<u32>())? // capacity
                                    .0
                                    .pad_to_align(),
                            )
                            .as_ptr();

                        *target.cast::<*mut ()>() = ptr_ptr.cast();

                        let (layout, offset) =
                            Layout::new::<*mut NonNull<()>>().extend(Layout::new::<u32>())?;
                        *target.byte_add(offset).cast::<u32>() = 0;

                        let (layout, offset) = layout.extend(Layout::new::<u32>())?;
                        *target.byte_add(offset).cast::<u32>() = len;

                        let (_, offset) = layout.extend(Layout::new::<u32>())?;
                        *target.byte_add(offset).cast::<u32>() = len;
                        NonNull::new_unchecked(target).cast()
                    } else {
                        Err(Error::UnexpectedValue(format!(
                            "expected value of type {:?} but got an array",
                            type_id.debug_name
                        )))?
                    }
                }
                Self::Struct {
                    fields: members, ..
                } => {
                    if let CoreTypeConcrete::Struct(info) = Self::resolve_type(ty, registry)? {
                        let mut layout: Option<Layout> = None;
                        let mut data = Vec::with_capacity(info.members.len());

                        let mut is_memory_allocated = false;
                        for (member_type_id, member) in info.members.iter().zip(members) {
                            let member_ty = registry.get_type(member_type_id)?;
                            let member_layout = member_ty.layout(registry)?;

                            let (new_layout, offset) = match layout {
                                Some(layout) => layout.extend(member_layout)?,
                                None => (member_layout, 0),
                            };
                            layout = Some(new_layout);

                            let member_ptr = member.to_ptr(
                                arena,
                                registry,
                                member_type_id,
                                find_dict_overrides,
                            )?;
                            data.push((
                                member_layout,
                                offset,
                                if member_ty.is_memory_allocated(registry)? {
                                    is_memory_allocated = true;

                                    // Undo the wrapper pointer added because the member's memory
                                    // allocated flag.
                                    *member_ptr.cast::<NonNull<()>>().as_ref()
                                } else {
                                    member_ptr
                                },
                            ));
                        }

                        let ptr = arena
                            .alloc_layout(layout.unwrap_or(Layout::new::<()>()).pad_to_align())
                            .as_ptr();

                        for (layout, offset, member_ptr) in data {
                            std::ptr::copy_nonoverlapping(
                                member_ptr.cast::<u8>().as_ptr(),
                                ptr.byte_add(offset),
                                layout.size(),
                            );
                        }

                        if is_memory_allocated {
                            // alloc returns a ref, so its never null
                            NonNull::new_unchecked(arena.alloc(ptr) as *mut _).cast()
                        } else {
                            NonNull::new_unchecked(ptr).cast()
                        }
                    } else {
                        Err(Error::UnexpectedValue(format!(
                            "expected value of type {:?} but got a struct",
                            type_id.debug_name
                        )))?
                    }
                }
                Self::Enum { tag, value, .. } => {
                    if let CoreTypeConcrete::Enum(info) = Self::resolve_type(ty, registry)? {
                        native_assert!(*tag < info.variants.len(), "Variant index out of range.");

                        let payload_type_id = &info.variants[*tag];
                        let payload =
                            value.to_ptr(arena, registry, payload_type_id, find_dict_overrides)?;

                        let (layout, tag_layout, variant_layouts) =
                            crate::types::r#enum::get_layout_for_variants(
                                registry,
                                &info.variants,
                            )?;
                        let ptr = arena.alloc_layout(layout).cast::<()>().as_ptr();

                        match tag_layout.size() {
                            0 => native_panic!("An enum without variants cannot be instantiated."),
                            1 => *ptr.cast::<u8>() = *tag as u8,
                            2 => *ptr.cast::<u16>() = *tag as u16,
                            4 => *ptr.cast::<u32>() = *tag as u32,
                            8 => *ptr.cast::<u64>() = *tag as u64,
                            _ => native_panic!("reached the maximum size for an enum"),
                        }

                        std::ptr::copy_nonoverlapping(
                            payload.cast::<u8>().as_ptr(),
                            ptr.byte_add(tag_layout.extend(variant_layouts[*tag])?.1)
                                .cast(),
                            variant_layouts[*tag].size(),
                        );

                        // alloc returns a reference so its never null
                        NonNull::new_unchecked(arena.alloc(ptr) as *mut _).cast()
                    } else {
                        Err(Error::UnexpectedValue(format!(
                            "expected value of type {:?} but got an enum value",
                            type_id.debug_name
                        )))?
                    }
                }
                Self::Felt252Dict { value: map, .. } => {
                    if let CoreTypeConcrete::Felt252Dict(info) = Self::resolve_type(ty, registry)? {
                        let elem_ty = registry.get_type(&info.ty)?;
                        let elem_layout = elem_ty.layout(registry)?.pad_to_align();

                        // We need `find_dict_overrides` to obtain the function pointers of the dup and drop
                        // implementations (if any) for the value type. This is required to be able to clone and drop
                        // the dictionary automatically when their reference count drops to zero.
                        let (dup_fn, drop_fn) = find_dict_overrides(&info.ty);
                        let mut value_map = FeltDict {
                            mappings: HashMap::with_capacity(map.len()),

                            layout: elem_layout,
                            elements: if map.is_empty() {
                                null_mut()
                            } else {
                                alloc(Layout::from_size_align_unchecked(
                                    elem_layout.pad_to_align().size() * map.len(),
                                    elem_layout.align(),
                                ))
                                .cast()
                            },

                            dup_fn,
                            drop_fn,

                            count: 0,
                        };

                        // next key must be called before next_value

                        for (key, value) in map.iter() {
                            let key = key.to_bytes_le();
                            let value =
                                value.to_ptr(arena, registry, &info.ty, find_dict_overrides)?;

                            let index = value_map.mappings.len();
                            value_map.mappings.insert(key, index);

                            std::ptr::copy_nonoverlapping(
                                value.cast::<u8>().as_ptr(),
                                value_map
                                    .elements
                                    .byte_add(elem_layout.pad_to_align().size() * index)
                                    .cast(),
                                elem_layout.size(),
                            );
                        }

                        NonNull::new_unchecked(Rc::into_raw(Rc::new(value_map)) as *mut ()).cast()
                    } else {
                        Err(Error::UnexpectedValue(format!(
                            "expected value of type {:?} but got a felt dict",
                            type_id.debug_name
                        )))?
                    }
                }
                Self::Uint8(value) => {
                    let ptr = arena.alloc_layout(Layout::new::<u8>()).cast();
                    *ptr.cast::<u8>().as_mut() = *value;

                    ptr
                }
                Self::Uint16(value) => {
                    let ptr = arena.alloc_layout(Layout::new::<u16>()).cast();
                    *ptr.cast::<u16>().as_mut() = *value;

                    ptr
                }
                Self::Uint32(value) => {
                    let ptr = arena.alloc_layout(Layout::new::<u32>()).cast();
                    *ptr.cast::<u32>().as_mut() = *value;

                    ptr
                }
                Self::Uint64(value) => {
                    let ptr = arena.alloc_layout(Layout::new::<u64>()).cast();
                    *ptr.cast::<u64>().as_mut() = *value;

                    ptr
                }
                Self::Uint128(value) => {
                    let ptr = arena.alloc_layout(Layout::new::<u128>()).cast();
                    *ptr.cast::<u128>().as_mut() = *value;

                    ptr
                }
                Self::Sint8(value) => {
                    let ptr = arena.alloc_layout(Layout::new::<i8>()).cast();
                    *ptr.cast::<i8>().as_mut() = *value;

                    ptr
                }
                Self::Sint16(value) => {
                    let ptr = arena.alloc_layout(Layout::new::<i16>()).cast();
                    *ptr.cast::<i16>().as_mut() = *value;

                    ptr
                }
                Self::Sint32(value) => {
                    let ptr = arena.alloc_layout(Layout::new::<i32>()).cast();
                    *ptr.cast::<i32>().as_mut() = *value;

                    ptr
                }
                Self::Sint64(value) => {
                    let ptr = arena.alloc_layout(Layout::new::<i64>()).cast();
                    *ptr.cast::<i64>().as_mut() = *value;

                    ptr
                }
                Self::Sint128(value) => {
                    let ptr = arena.alloc_layout(Layout::new::<i128>()).cast();
                    *ptr.cast::<i128>().as_mut() = *value;

                    ptr
                }
                Self::EcPoint(a, b) => {
                    let ptr = arena
                        .alloc_layout(layout_repeat(&get_integer_layout(252), 2)?.0.pad_to_align())
                        .cast();

                    let a = felt252_bigint(a.to_bigint()).to_bytes_le();
                    let b = felt252_bigint(b.to_bigint()).to_bytes_le();
                    let data = [a, b];

                    ptr.cast::<[[u8; 32]; 2]>().as_mut().copy_from_slice(&data);

                    ptr
                }
                Self::EcState(a, b, c, d) => {
                    let ptr = arena
                        .alloc_layout(layout_repeat(&get_integer_layout(252), 4)?.0.pad_to_align())
                        .cast();

                    let a = felt252_bigint(a.to_bigint()).to_bytes_le();
                    let b = felt252_bigint(b.to_bigint()).to_bytes_le();
                    let c = felt252_bigint(c.to_bigint()).to_bytes_le();
                    let d = felt252_bigint(d.to_bigint()).to_bytes_le();
                    let data = [a, b, c, d];

                    ptr.cast::<[[u8; 32]; 4]>().as_mut().copy_from_slice(&data);

                    ptr
                }
                Self::Secp256K1Point { .. } => native_panic!("todo: allocate type Secp256K1Point"),
                Self::Secp256R1Point { .. } => native_panic!("todo: allocate type Secp256R1Point"),
                Self::Null => {
                    native_panic!(
                        "unimplemented: null is meant as return value for nullable for now"
                    )
                }
                Self::IntRange { x, y } => {
                    if let CoreTypeConcrete::IntRange(info) = Self::resolve_type(ty, registry)? {
                        let inner = registry.get_type(&info.ty)?;
                        let inner_layout = inner.layout(registry)?;

                        let x_ptr = x.to_ptr(arena, registry, &info.ty, find_dict_overrides)?;

                        let (struct_layout, y_offset) = inner_layout.extend(inner_layout)?;

                        let y_ptr = y.to_ptr(arena, registry, &info.ty, find_dict_overrides)?;

                        let ptr = arena.alloc_layout(struct_layout.pad_to_align()).as_ptr();

                        std::ptr::copy_nonoverlapping(
                            x_ptr.cast::<u8>().as_ptr(),
                            ptr,
                            inner_layout.size(),
                        );

                        std::ptr::copy_nonoverlapping(
                            y_ptr.cast::<u8>().as_ptr(),
                            ptr.byte_add(y_offset),
                            inner_layout.size(),
                        );

                        NonNull::new_unchecked(ptr).cast()
                    } else {
                        native_panic!(
                            "an IntRange value should always have an IntRange CoreTypeConcrete"
                        )
                    }
                }
            }
        })
    }

    /// From the given pointer acquired from the either the JIT / compiled program outputs, convert it to a [`Self`]
    pub(crate) fn from_ptr(
        ptr: NonNull<()>,
        type_id: &ConcreteTypeId,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        should_drop: bool,
    ) -> Result<Self, Error> {
        let ty = registry.get_type(type_id)?;

        Ok(unsafe {
            match ty {
                CoreTypeConcrete::Array(info) => {
                    let elem_ty = registry.get_type(&info.ty)?;

                    let elem_layout = elem_ty.layout(registry)?;
                    let elem_stride = elem_layout.pad_to_align().size();

                    let ptr_layout = Layout::new::<*mut ()>();
                    let len_layout = crate::utils::get_integer_layout(32);

                    let (ptr_layout, offset) = ptr_layout.extend(len_layout)?;
                    let start_offset_value = *NonNull::new(ptr.as_ptr().byte_add(offset))
                        .to_native_assert_error("tried to make a non-null ptr out of a null one")?
                        .cast::<u32>()
                        .as_ref();
                    let (_, offset) = ptr_layout.extend(len_layout)?;
                    let end_offset_value = *NonNull::new(ptr.as_ptr().byte_add(offset))
                        .to_native_assert_error("tried to make a non-null ptr out of a null one")?
                        .cast::<u32>()
                        .as_ref();

                    // This pointer can be null if the array is empty.
                    let array_ptr_ptr = *ptr.cast::<*mut *mut ()>().as_ref();

                    let refcount_offset = crate::types::array::calc_data_prefix_offset(elem_layout);
                    let array_value = if array_ptr_ptr.is_null() {
                        Vec::new()
                    } else {
                        let array_ptr = array_ptr_ptr.read();
                        let ref_count = array_ptr
                            .byte_sub(refcount_offset)
                            .cast::<u32>()
                            .as_mut()
                            .unwrap();
                        if should_drop {
                            *ref_count -= 1;
                        }

                        native_assert!(
                            end_offset_value >= start_offset_value,
                            "can't have an array with negative length"
                        );
                        let num_elems = (end_offset_value - start_offset_value) as usize;

                        if *ref_count == 0 {
                            // Drop prefix elements.
                            for i in 0..start_offset_value {
                                let cur_elem_ptr =
                                    NonNull::new(array_ptr.byte_add(elem_stride * i as usize))
                                        .to_native_assert_error(
                                            "tried to make a non-null ptr out of a null one",
                                        )?;
                                drop(Self::from_ptr(
                                    cur_elem_ptr,
                                    &info.ty,
                                    registry,
                                    should_drop,
                                )?);
                            }
                        }

                        let mut array_value = Vec::with_capacity(num_elems);
                        for i in start_offset_value..end_offset_value {
                            let cur_elem_ptr =
                                NonNull::new(array_ptr.byte_add(elem_stride * i as usize))
                                    .to_native_assert_error(
                                        "tried to make a non-null ptr out of a null one",
                                    )?;
                            array_value.push(Self::from_ptr(
                                cur_elem_ptr,
                                &info.ty,
                                registry,
                                *ref_count == 0,
                            )?);
                        }

                        if *ref_count == 0 {
                            // Drop suffix elements.
                            let array_max_len = array_ptr
                                .byte_sub(refcount_offset - size_of::<u32>())
                                .cast::<u32>()
                                .read();
                            for i in end_offset_value..array_max_len {
                                let cur_elem_ptr =
                                    NonNull::new(array_ptr.byte_add(elem_stride * i as usize))
                                        .to_native_assert_error(
                                            "tried to make a non-null ptr out of a null one",
                                        )?;
                                drop(Self::from_ptr(
                                    cur_elem_ptr,
                                    &info.ty,
                                    registry,
                                    should_drop,
                                )?);
                            }

                            // Free array storage.
                            libc_free(array_ptr.byte_sub(refcount_offset).cast());
                            libc_free(array_ptr_ptr.cast());
                        }

                        array_value
                    };

                    Self::Array(array_value)
                }
                CoreTypeConcrete::Box(info) => {
                    let inner = *ptr.cast::<NonNull<()>>().as_ptr();
                    let value = Self::from_ptr(inner, &info.ty, registry, should_drop)?;

                    if should_drop {
                        libc_free(inner.as_ptr().cast());
                    }

                    value
                }
                CoreTypeConcrete::EcPoint(_) => {
                    let data = ptr.cast::<[[u8; 32]; 2]>().as_mut();

                    data[0][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
                    data[1][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).

                    Self::EcPoint(Felt::from_bytes_le(&data[0]), Felt::from_bytes_le(&data[1]))
                }
                CoreTypeConcrete::EcState(_) => {
                    let data = ptr.cast::<[[u8; 32]; 4]>().as_mut();

                    data[0][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
                    data[1][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
                    data[2][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
                    data[3][31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).

                    Self::EcState(
                        Felt::from_bytes_le(&data[0]),
                        Felt::from_bytes_le(&data[1]),
                        Felt::from_bytes_le(&data[2]),
                        Felt::from_bytes_le(&data[3]),
                    )
                }
                CoreTypeConcrete::Felt252(_) => {
                    let data = ptr.cast::<[u8; 32]>().as_mut();
                    data[31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
                    let data = Felt::from_bytes_le_slice(data);
                    Self::Felt252(data)
                }
                CoreTypeConcrete::Uint8(_) => Self::Uint8(*ptr.cast::<u8>().as_ref()),
                CoreTypeConcrete::Uint16(_) => Self::Uint16(*ptr.cast::<u16>().as_ref()),
                CoreTypeConcrete::Uint32(_) => Self::Uint32(*ptr.cast::<u32>().as_ref()),
                CoreTypeConcrete::Uint64(_) => Self::Uint64(*ptr.cast::<u64>().as_ref()),
                CoreTypeConcrete::Uint128(_) => Self::Uint128(*ptr.cast::<u128>().as_ref()),
                CoreTypeConcrete::Uint128MulGuarantee(_) => Self::Null,
                CoreTypeConcrete::Sint8(_) => Self::Sint8(*ptr.cast::<i8>().as_ref()),
                CoreTypeConcrete::Sint16(_) => Self::Sint16(*ptr.cast::<i16>().as_ref()),
                CoreTypeConcrete::Sint32(_) => Self::Sint32(*ptr.cast::<i32>().as_ref()),
                CoreTypeConcrete::Sint64(_) => Self::Sint64(*ptr.cast::<i64>().as_ref()),
                CoreTypeConcrete::Sint128(_) => Self::Sint128(*ptr.cast::<i128>().as_ref()),
                CoreTypeConcrete::NonZero(info) => {
                    Self::from_ptr(ptr, &info.ty, registry, should_drop)?
                }
                CoreTypeConcrete::Nullable(info) => {
                    let inner_ptr = *ptr.cast::<*mut ()>().as_ptr();
                    if inner_ptr.is_null() {
                        Self::Null
                    } else {
                        let value = Self::from_ptr(
                            NonNull::new_unchecked(inner_ptr).cast(),
                            &info.ty,
                            registry,
                            should_drop,
                        )?;

                        if should_drop {
                            libc_free(inner_ptr.cast());
                        }

                        value
                    }
                }
                CoreTypeConcrete::Uninitialized(_) => {
                    native_panic!("todo: implement uninit from_ptr or ignore the return value")
                }
                CoreTypeConcrete::Enum(info) => {
                    let tag_layout = crate::utils::get_integer_layout(match info.variants.len() {
                        0 | 1 => 0,
                        num_variants => (num_variants.next_power_of_two().next_multiple_of(8) >> 3)
                            .try_into()
                            .map_err(|_| Error::IntegerConversion)?,
                    });
                    let tag_value = match info.variants.len() {
                        0 => {
                            // An enum without variants is basically the `!` (never) type in Rust.
                            native_panic!("An enum without variants is not a valid type.")
                        }
                        1 => 0,
                        _ => match tag_layout.size() {
                            1 => *ptr.cast::<u8>().as_ref() as usize,
                            2 => *ptr.cast::<u16>().as_ref() as usize,
                            4 => *ptr.cast::<u32>().as_ref() as usize,
                            8 => *ptr.cast::<u64>().as_ref() as usize,
                            _ => native_panic!("reached the maximum size for an enum"),
                        },
                    };

                    // Filter out bits that are not part of the enum's tag.
                    let tag_value = tag_value
                        & 1usize
                            .wrapping_shl(info.variants.len().next_power_of_two().trailing_zeros())
                            .wrapping_sub(1);

                    let payload_ty = registry.get_type(&info.variants[tag_value])?;
                    let payload_layout = payload_ty.layout(registry)?;

                    let payload_ptr = NonNull::new(
                        ptr.as_ptr().byte_add(tag_layout.extend(payload_layout)?.1),
                    )
                    .to_native_assert_error("tried to make a non-null ptr out of a null one")?;
                    let payload = Self::from_ptr(
                        payload_ptr,
                        &info.variants[tag_value],
                        registry,
                        should_drop,
                    )?;

                    Self::Enum {
                        tag: tag_value,
                        value: Box::new(payload),
                        debug_name: type_id.debug_name.as_ref().map(|x| x.to_string()),
                    }
                }
                CoreTypeConcrete::Struct(info) => {
                    let mut layout: Option<Layout> = None;
                    let mut members = Vec::with_capacity(info.members.len());

                    for member_ty in &info.members {
                        let member = registry.get_type(member_ty)?;
                        let member_layout = member.layout(registry)?;

                        let (new_layout, offset) = match layout {
                            Some(layout) => layout.extend(member_layout)?,
                            None => (member_layout, 0),
                        };
                        layout = Some(new_layout);

                        members.push(Self::from_ptr(
                            NonNull::new(ptr.as_ptr().byte_add(offset)).to_native_assert_error(
                                "tried to make a non-null ptr out of a null one",
                            )?,
                            member_ty,
                            registry,
                            should_drop,
                        )?);
                    }

                    Self::Struct {
                        fields: members,
                        debug_name: type_id.debug_name.as_ref().map(|x| x.to_string()),
                    }
                }
                CoreTypeConcrete::Felt252Dict(info)
                | CoreTypeConcrete::SquashedFelt252Dict(info) => {
                    let dict = Rc::from_raw(ptr.cast::<*const FeltDict>().read());

                    let mut output_map = HashMap::with_capacity(dict.mappings.len());
                    for (&key, &index) in dict.mappings.iter() {
                        let mut key = key;
                        key[31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).

                        let key = Felt::from_bytes_le(&key);
                        // The dictionary items are not being dropped here. They'll be dropped along
                        // with the dictionary (if requested using `should_drop`).
                        output_map.insert(
                            key,
                            Self::from_ptr(
                                NonNull::new(
                                    dict.elements
                                        .byte_add(dict.layout.pad_to_align().size() * index),
                                )
                                .to_native_assert_error(
                                    "tried to make a non-null ptr out of a null one",
                                )?
                                .cast(),
                                &info.ty,
                                registry,
                                false,
                            )?,
                        );
                    }

                    if should_drop {
                        drop(dict);
                    } else {
                        forget(dict);
                    }

                    Self::Felt252Dict {
                        value: output_map,
                        debug_name: type_id.debug_name.as_ref().map(|x| x.to_string()),
                    }
                }
                CoreTypeConcrete::Felt252DictEntry(_) => {
                    native_panic!("unimplemented: should be impossible to return")
                }
                CoreTypeConcrete::Pedersen(_)
                | CoreTypeConcrete::Poseidon(_)
                | CoreTypeConcrete::Bitwise(_)
                | CoreTypeConcrete::BuiltinCosts(_)
                | CoreTypeConcrete::RangeCheck(_)
                | CoreTypeConcrete::EcOp(_)
                | CoreTypeConcrete::GasBuiltin(_)
                | CoreTypeConcrete::SegmentArena(_) => {
                    native_panic!("handled before: {:?}", type_id)
                }
                // Does it make sense for programs to return this? Should it be implemented
                CoreTypeConcrete::StarkNet(selector) => match selector {
                    StarkNetTypeConcrete::ClassHash(_)
                    | StarkNetTypeConcrete::ContractAddress(_)
                    | StarkNetTypeConcrete::StorageBaseAddress(_)
                    | StarkNetTypeConcrete::StorageAddress(_) => {
                        // felt values
                        let data = ptr.cast::<[u8; 32]>().as_mut();
                        data[31] &= 0x0F; // Filter out first 4 bits (they're outside an i252).
                        let data = Felt::from_bytes_le(data);
                        Self::Felt252(data)
                    }
                    StarkNetTypeConcrete::System(_) => {
                        native_panic!("should be handled before")
                    }
                    StarkNetTypeConcrete::Secp256Point(info) => match info {
                        Secp256PointTypeConcrete::K1(_) => {
                            let data = ptr.cast::<Secp256k1Point>().as_ref();
                            Self::Secp256K1Point(*data)
                        }
                        Secp256PointTypeConcrete::R1(_) => {
                            let data = ptr.cast::<Secp256r1Point>().as_ref();
                            Self::Secp256R1Point(*data)
                        }
                    },
                    StarkNetTypeConcrete::Sha256StateHandle(_) => {
                        native_panic!("todo: implement Sha256StateHandle from_ptr")
                    }
                },
                CoreTypeConcrete::Span(_) => native_panic!("implement span from_ptr"),
                CoreTypeConcrete::Snapshot(info) => {
                    Self::from_ptr(ptr, &info.ty, registry, should_drop)?
                }
                CoreTypeConcrete::Bytes31(_) => {
                    let data = *ptr.cast::<[u8; 31]>().as_ref();
                    Self::Bytes31(data)
                }

                CoreTypeConcrete::Const(_) => native_panic!("implement const from_ptr"),
                CoreTypeConcrete::BoundedInt(info) => {
                    let mut data = BigInt::from_biguint(
                        Sign::Plus,
                        BigUint::from_bytes_le(slice::from_raw_parts(
                            ptr.cast::<u8>().as_ptr(),
                            (info.range.offset_bit_width().next_multiple_of(8) >> 3) as usize,
                        )),
                    );

                    data &= (BigInt::one() << info.range.offset_bit_width()) - BigInt::one();
                    data += &info.range.lower;

                    Self::BoundedInt {
                        value: data.into(),
                        range: info.range.clone(),
                    }
                }
                CoreTypeConcrete::Coupon(_)
                | CoreTypeConcrete::Circuit(_)
                | CoreTypeConcrete::RangeCheck96(_) => native_panic!("implement from_ptr"),
                CoreTypeConcrete::IntRange(info) => {
                    let member = registry.get_type(&info.ty)?;
                    let member_layout = member.layout(registry)?;

                    let x = Self::from_ptr(
                        NonNull::new(ptr.as_ptr()).to_native_assert_error(
                            "tried to make a non-null ptr out of a null one",
                        )?,
                        &info.ty,
                        registry,
                        should_drop,
                    )?;

                    let y = Self::from_ptr(
                        NonNull::new(
                            ptr.as_ptr()
                                .byte_add(member_layout.extend(member_layout)?.1),
                        )
                        .to_native_assert_error("tried to make a non-null ptr out of a null one")?,
                        &info.ty,
                        registry,
                        should_drop,
                    )?;

                    Self::IntRange {
                        x: x.into(),
                        y: y.into(),
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use bumpalo::Bump;
    use cairo_lang_sierra::extensions::types::{InfoAndTypeConcreteType, TypeInfo};
    use cairo_lang_sierra::program::ConcreteTypeLongId;
    use cairo_lang_sierra::program::Program;
    use cairo_lang_sierra::program::TypeDeclaration;
    use cairo_lang_sierra::ProgramParser;

    #[test]
    fn test_jit_value_conversion_felt() {
        let felt_value: Felt = 42.into();
        let jit_value: Value = felt_value.into();
        assert_eq!(jit_value, Value::Felt252(Felt::from(42)));
    }

    #[test]
    fn test_jit_value_conversion_u8() {
        let u8_value: u8 = 10;
        let jit_value: Value = u8_value.into();
        assert_eq!(jit_value, Value::Uint8(10));
    }

    #[test]
    fn test_jit_value_conversion_u16() {
        let u8_value: u16 = 10;
        let jit_value: Value = u8_value.into();
        assert_eq!(jit_value, Value::Uint16(10));
    }

    #[test]
    fn test_jit_value_conversion_u32() {
        let u32_value: u32 = 10;
        let jit_value: Value = u32_value.into();
        assert_eq!(jit_value, Value::Uint32(10));
    }

    #[test]
    fn test_jit_value_conversion_u64() {
        let u64_value: u64 = 10;
        let jit_value: Value = u64_value.into();
        assert_eq!(jit_value, Value::Uint64(10));
    }

    #[test]
    fn test_jit_value_conversion_u128() {
        let u128_value: u128 = 10;
        let jit_value: Value = u128_value.into();
        assert_eq!(jit_value, Value::Uint128(10));
    }

    #[test]
    fn test_jit_value_conversion_i8() {
        let i8_value: i8 = -10;
        let jit_value: Value = i8_value.into();
        assert_eq!(jit_value, Value::Sint8(-10));
    }

    #[test]
    fn test_jit_value_conversion_i16() {
        let i16_value: i16 = -10;
        let jit_value: Value = i16_value.into();
        assert_eq!(jit_value, Value::Sint16(-10));
    }

    #[test]
    fn test_jit_value_conversion_i32() {
        let i32_value: i32 = -10;
        let jit_value: Value = i32_value.into();
        assert_eq!(jit_value, Value::Sint32(-10));
    }

    #[test]
    fn test_jit_value_conversion_i64() {
        let i64_value: i64 = -10;
        let jit_value: Value = i64_value.into();
        assert_eq!(jit_value, Value::Sint64(-10));
    }

    #[test]
    fn test_jit_value_conversion_i128() {
        let i128_value: i128 = -10;
        let jit_value: Value = i128_value.into();
        assert_eq!(jit_value, Value::Sint128(-10));
    }

    #[test]
    fn test_jit_value_conversion_array_from_slice() {
        let array_slice: &[u8] = &[1, 2, 3];
        let jit_value: Value = array_slice.into();
        assert_eq!(
            jit_value,
            Value::Array(vec![Value::Uint8(1), Value::Uint8(2), Value::Uint8(3)])
        );
    }

    #[test]
    fn test_jit_value_conversion_array_from_vec() {
        let array_vec: Vec<u8> = vec![1, 2, 3];
        let jit_value: Value = array_vec.into();
        assert_eq!(
            jit_value,
            Value::Array(vec![Value::Uint8(1), Value::Uint8(2), Value::Uint8(3)])
        );
    }

    #[test]
    fn test_jit_value_conversion_array_from_fixed_size_array() {
        let array_fixed: [u8; 3] = [1, 2, 3];
        let jit_value: Value = array_fixed.into();
        assert_eq!(
            jit_value,
            Value::Array(vec![Value::Uint8(1), Value::Uint8(2), Value::Uint8(3)])
        );
    }

    #[test]
    fn test_resolve_type_snapshot() {
        let ty = CoreTypeConcrete::Snapshot(InfoAndTypeConcreteType {
            info: TypeInfo {
                long_id: ConcreteTypeLongId {
                    generic_id: "generic_type_id".into(),
                    generic_args: vec![],
                },
                storable: false,
                droppable: false,
                duplicatable: false,
                zero_sized: false,
            },
            ty: "test_id".into(),
        });

        let program = Program {
            type_declarations: vec![TypeDeclaration {
                id: "test_id".into(),
                long_id: ConcreteTypeLongId {
                    generic_id: "u128".into(),
                    generic_args: vec![],
                },
                declared_type_info: None,
            }],
            libfunc_declarations: vec![],
            statements: vec![],
            funcs: vec![],
        };

        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        assert_eq!(
            Value::resolve_type(&ty, &registry)
                .unwrap()
                .integer_range(&registry)
                .unwrap(),
            Range {
                lower: BigInt::from(u128::MIN),
                upper: BigInt::from(u128::MAX) + BigInt::one(),
            }
        );
    }

    #[test]
    fn test_to_jit_felt252() {
        let program = ProgramParser::new()
            .parse("type felt252 = felt252;")
            .unwrap();

        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        assert_eq!(
            unsafe {
                *Value::Felt252(Felt::from(42))
                    .to_ptr(
                        &Bump::new(),
                        &registry,
                        &program.type_declarations[0].id,
                        |_| todo!(),
                    )
                    .unwrap()
                    .cast::<[u32; 8]>()
                    .as_ptr()
            },
            [42, 0, 0, 0, 0, 0, 0, 0]
        );

        assert_eq!(
            unsafe {
                *Value::Felt252(Felt::MAX)
                    .to_ptr(
                        &Bump::new(),
                        &registry,
                        &program.type_declarations[0].id,
                        |_| todo!(),
                    )
                    .unwrap()
                    .cast::<[u32; 8]>()
                    .as_ptr()
            },
            // 0x800000000000011000000000000000000000000000000000000000000000001 - 1
            [0, 0, 0, 0, 0, 0, 17, 134217728]
        );

        assert_eq!(
            unsafe {
                *Value::Felt252(Felt::MAX + Felt::ONE)
                    .to_ptr(
                        &Bump::new(),
                        &registry,
                        &program.type_declarations[0].id,
                        |_| todo!(),
                    )
                    .unwrap()
                    .cast::<[u32; 8]>()
                    .as_ptr()
            },
            [0, 0, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn test_to_jit_uint8() {
        let program = ProgramParser::new().parse("type u8 = u8;").unwrap();

        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        assert_eq!(
            unsafe {
                *Value::Uint8(9)
                    .to_ptr(
                        &Bump::new(),
                        &registry,
                        &program.type_declarations[0].id,
                        |_| todo!(),
                    )
                    .unwrap()
                    .cast::<u8>()
                    .as_ptr()
            },
            9
        );
    }

    #[test]
    fn test_to_jit_uint16() {
        let program = ProgramParser::new().parse("type u16 = u16;").unwrap();

        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        assert_eq!(
            unsafe {
                *Value::Uint16(17)
                    .to_ptr(
                        &Bump::new(),
                        &registry,
                        &program.type_declarations[0].id,
                        |_| todo!(),
                    )
                    .unwrap()
                    .cast::<u16>()
                    .as_ptr()
            },
            17
        );
    }

    #[test]
    fn test_to_jit_uint32() {
        let program = ProgramParser::new().parse("type u32 = u32;").unwrap();

        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        assert_eq!(
            unsafe {
                *Value::Uint32(33)
                    .to_ptr(
                        &Bump::new(),
                        &registry,
                        &program.type_declarations[0].id,
                        |_| todo!(),
                    )
                    .unwrap()
                    .cast::<u32>()
                    .as_ptr()
            },
            33
        );
    }

    #[test]
    fn test_to_jit_uint64() {
        let program = ProgramParser::new().parse("type u64 = u64;").unwrap();

        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        assert_eq!(
            unsafe {
                *Value::Uint64(65)
                    .to_ptr(
                        &Bump::new(),
                        &registry,
                        &program.type_declarations[0].id,
                        |_| todo!(),
                    )
                    .unwrap()
                    .cast::<u64>()
                    .as_ptr()
            },
            65
        );
    }

    #[test]
    fn test_to_jit_uint128() {
        let program = ProgramParser::new().parse("type u128 = u128;").unwrap();

        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        assert_eq!(
            unsafe {
                *Value::Uint128(129)
                    .to_ptr(
                        &Bump::new(),
                        &registry,
                        &program.type_declarations[0].id,
                        |_| todo!(),
                    )
                    .unwrap()
                    .cast::<u128>()
                    .as_ptr()
            },
            129
        );
    }

    #[test]
    fn test_to_jit_sint8() {
        let program = ProgramParser::new().parse("type i8 = i8;").unwrap();

        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        assert_eq!(
            unsafe {
                *Value::Sint8(-9)
                    .to_ptr(
                        &Bump::new(),
                        &registry,
                        &program.type_declarations[0].id,
                        |_| todo!(),
                    )
                    .unwrap()
                    .cast::<i8>()
                    .as_ptr()
            },
            -9
        );
    }

    #[test]
    fn test_to_jit_sint16() {
        let program = ProgramParser::new().parse("type i16 = i16;").unwrap();

        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        assert_eq!(
            unsafe {
                *Value::Sint16(-17)
                    .to_ptr(
                        &Bump::new(),
                        &registry,
                        &program.type_declarations[0].id,
                        |_| todo!(),
                    )
                    .unwrap()
                    .cast::<i16>()
                    .as_ptr()
            },
            -17
        );
    }

    #[test]
    fn test_to_jit_sint32() {
        let program = ProgramParser::new().parse("type i32 = i32;").unwrap();

        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        assert_eq!(
            unsafe {
                *Value::Sint32(-33)
                    .to_ptr(
                        &Bump::new(),
                        &registry,
                        &program.type_declarations[0].id,
                        |_| todo!(),
                    )
                    .unwrap()
                    .cast::<i32>()
                    .as_ptr()
            },
            -33
        );
    }

    #[test]
    fn test_to_jit_sint64() {
        let program = ProgramParser::new().parse("type i64 = i64;").unwrap();

        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        assert_eq!(
            unsafe {
                *Value::Sint64(-65)
                    .to_ptr(
                        &Bump::new(),
                        &registry,
                        &program.type_declarations[0].id,
                        |_| todo!(),
                    )
                    .unwrap()
                    .cast::<i64>()
                    .as_ptr()
            },
            -65
        );
    }

    #[test]
    fn test_to_jit_sint128() {
        let program = ProgramParser::new().parse("type i128 = i128;").unwrap();

        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        assert_eq!(
            unsafe {
                *Value::Sint128(-129)
                    .to_ptr(
                        &Bump::new(),
                        &registry,
                        &program.type_declarations[0].id,
                        |_| todo!(),
                    )
                    .unwrap()
                    .cast::<i128>()
                    .as_ptr()
            },
            -129
        );
    }

    #[test]
    fn test_to_jit_ec_point() {
        let program = ProgramParser::new()
            .parse("type EcPoint = EcPoint;")
            .unwrap();

        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        assert_eq!(
            unsafe {
                *Value::EcPoint(Felt::from(1234), Felt::from(4321))
                    .to_ptr(
                        &Bump::new(),
                        &registry,
                        &program.type_declarations[0].id,
                        |_| todo!(),
                    )
                    .unwrap()
                    .cast::<[[u32; 8]; 2]>()
                    .as_ptr()
            },
            [[1234, 0, 0, 0, 0, 0, 0, 0], [4321, 0, 0, 0, 0, 0, 0, 0]]
        );
    }

    #[test]
    fn test_to_jit_ec_state() {
        let program = ProgramParser::new()
            .parse("type EcState = EcState;")
            .unwrap();

        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        assert_eq!(
            unsafe {
                *Value::EcState(
                    Felt::from(1234),
                    Felt::from(4321),
                    Felt::from(3333),
                    Felt::from(4444),
                )
                .to_ptr(
                    &Bump::new(),
                    &registry,
                    &program.type_declarations[0].id,
                    |_| todo!(),
                )
                .unwrap()
                .cast::<[[u32; 8]; 4]>()
                .as_ptr()
            },
            [
                [1234, 0, 0, 0, 0, 0, 0, 0],
                [4321, 0, 0, 0, 0, 0, 0, 0],
                [3333, 0, 0, 0, 0, 0, 0, 0],
                [4444, 0, 0, 0, 0, 0, 0, 0]
            ]
        );
    }

    #[test]
    fn test_to_jit_enum() {
        // Parse the program
        let program = ProgramParser::new()
            .parse(
                "type u8 = u8;
                type MyEnum = Enum<ut@MyEnum, u8, u8>;",
            )
            .unwrap();

        // Create the registry for the program
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        // Call to_jit to get the value of the enum
        let result = Value::Enum {
            tag: 0,
            value: Box::new(Value::Uint8(10)),
            debug_name: None,
        }
        .to_ptr(
            &Bump::new(),
            &registry,
            &program.type_declarations[1].id,
            |_| todo!(),
        );

        // Assertion to verify that the value returned by to_jit is not NULL
        assert!(result.is_ok());
    }

    #[test]
    fn test_to_jit_bounded_int_valid() {
        // Parse the program
        let program = ProgramParser::new()
            .parse(
                "type felt252 = felt252;
            type BoundedInt = BoundedInt<10, 510>;",
            )
            .unwrap();

        // Create the registry for the program
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        // Valid case
        assert_eq!(
            unsafe {
                *Value::BoundedInt {
                    value: Felt::from(16),
                    range: Range {
                        lower: BigInt::from(10),
                        upper: BigInt::from(510),
                    },
                }
                .to_ptr(
                    &Bump::new(),
                    &registry,
                    &program.type_declarations[1].id,
                    |_| todo!(),
                )
                .unwrap()
                .cast::<[u32; 8]>()
                .as_ptr()
            },
            [16, 0, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn test_to_jit_bounded_int_lower_bound_greater_than_upper() {
        // Parse the program
        let program = ProgramParser::new()
            .parse(
                "type felt252 = felt252;
            type BoundedInt = BoundedInt<10, 510>;", // Note: lower > upper
            )
            .unwrap();

        // Create the registry for the program
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        // Error case: lower bound greater than upper bound
        let result = Value::BoundedInt {
            value: Felt::from(16),
            range: Range {
                lower: BigInt::from(510),
                upper: BigInt::from(10),
            },
        }
        .to_ptr(
            &Bump::new(),
            &registry,
            &program.type_declarations[1].id,
            |_| todo!(),
        );

        assert!(matches!(
            result,
            Err(Error::Compiler(CompilerError::BoundedIntOutOfRange { .. }))
        ));
    }

    #[test]
    fn test_to_jit_bounded_int_value_less_than_lower_bound() {
        // Parse the program
        let program = ProgramParser::new()
            .parse(
                "type felt252 = felt252;
            type BoundedInt = BoundedInt<10, 510>;",
            )
            .unwrap();

        // Create the registry for the program
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        // Error case: value less than lower bound
        let result = Value::BoundedInt {
            value: Felt::from(9),
            range: Range {
                lower: BigInt::from(10),
                upper: BigInt::from(510),
            },
        }
        .to_ptr(
            &Bump::new(),
            &registry,
            &program.type_declarations[1].id,
            |_| todo!(),
        );

        assert!(matches!(
            result,
            Err(Error::Compiler(CompilerError::BoundedIntOutOfRange { .. }))
        ));
    }

    #[test]
    fn test_to_jit_bounded_int_value_greater_than_or_equal_to_upper_bound() {
        // Parse the program
        let program = ProgramParser::new()
            .parse(
                "type felt252 = felt252;
            type BoundedInt = BoundedInt<10, 510>;",
            )
            .unwrap();

        // Create the registry for the program
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        // Error case: value greater than or equal to upper bound
        let result = Value::BoundedInt {
            value: Felt::from(512),
            range: Range {
                lower: BigInt::from(10),
                upper: BigInt::from(510),
            },
        }
        .to_ptr(
            &Bump::new(),
            &registry,
            &program.type_declarations[1].id,
            |_| todo!(),
        );

        assert!(matches!(
            result,
            Err(Error::Compiler(CompilerError::BoundedIntOutOfRange { .. }))
        ));
    }

    #[test]
    fn test_to_jit_bounded_int_equal_bounds_and_value() {
        // Parse the program
        let program = ProgramParser::new()
            .parse(
                "type felt252 = felt252;
            type BoundedInt = BoundedInt<10, 10>;", // Note: lower = upper
            )
            .unwrap();

        // Create the registry for the program
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        // Error case: value equals lower and upper bound (upper bound is exclusive)
        let result = Value::BoundedInt {
            value: Felt::from(10),
            range: Range {
                lower: BigInt::from(10),
                upper: BigInt::from(10),
            },
        }
        .to_ptr(
            &Bump::new(),
            &registry,
            &program.type_declarations[1].id,
            |_| todo!(),
        );

        assert!(matches!(
            result,
            Err(Error::Compiler(CompilerError::BoundedIntOutOfRange { .. }))
        ));
    }

    #[test]
    fn test_to_jit_enum_variant_out_of_range() {
        // Parse the program
        let program = ProgramParser::new()
            .parse(
                "type u8 = u8;
            type MyEnum = Enum<ut@MyEnum, u8, u8>;",
            )
            .unwrap();

        // Create the registry for the program
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        // Call to_jit to get the value of the enum with tag value out of range
        let result = Value::Enum {
            tag: 2,
            value: Box::new(Value::Uint8(10)),
            debug_name: None,
        }
        .to_ptr(
            &Bump::new(),
            &registry,
            &program.type_declarations[1].id,
            |_| todo!(),
        )
        .unwrap_err();

        let error = result.to_string().clone();
        let error_msg = error.split("\n").collect::<Vec<&str>>()[0];

        assert_eq!(error_msg, "Variant index out of range.");
    }

    #[test]
    fn test_to_jit_enum_no_variant() {
        let program = ProgramParser::new()
            .parse(
                "type u8 = u8;
                type MyEnum = Enum<ut@MyEnum, u8>;",
            )
            .unwrap();

        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        let result = Value::Enum {
            tag: 0,
            value: Box::new(Value::Uint8(10)),
            debug_name: None,
        }
        .to_ptr(
            &Bump::new(),
            &registry,
            &program.type_declarations[1].id,
            |_| todo!(),
        )
        .unwrap_err();

        let error = result.to_string().clone();
        let error_msg = error.split("\n").collect::<Vec<&str>>()[0];

        assert_eq!(
            error_msg,
            "An enum without variants cannot be instantiated."
        );
    }

    #[test]
    fn test_to_jit_enum_type_error() {
        // Parse the program
        let program = ProgramParser::new()
            .parse(
                "type felt252 = felt252;
                type MyEnum = Enum<ut@MyEnum, felt252, felt252>;",
            )
            .unwrap();

        // Creating a registry for the program.
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        // Invoking to_jit method on a JitValue::Enum to convert it to a JIT representation.
        // Generating an error by providing an enum value instead of the expected type.
        let result = Value::Enum {
            tag: 0,
            value: Box::new(Value::Struct {
                fields: vec![Value::from(2u32)],
                debug_name: None,
            }),
            debug_name: None,
        }
        .to_ptr(
            &Bump::new(),
            &registry,
            &program.type_declarations[0].id,
            |_| todo!(),
        )
        .unwrap_err(); // Unwrapping the error

        // Matching the error result to verify the error type and message.
        match result {
            Error::UnexpectedValue(expected_msg) => {
                // Asserting that the error message matches the expected message.
                assert_eq!(
                    expected_msg,
                    format!(
                        "expected value of type {:?} but got an enum value",
                        program.type_declarations[0].id.debug_name
                    )
                );
            }
            _ => panic!("Unexpected error type: {:?}", result),
        }
    }

    #[test]
    fn test_to_jit_struct_type_error() {
        // Parse the program
        let program = ProgramParser::new()
            .parse(
                "type felt252 = felt252;
                type MyEnum = Enum<ut@MyEnum, felt252, felt252>;",
            )
            .unwrap();

        // Creating a registry for the program.
        let registry = ProgramRegistry::<CoreType, CoreLibfunc>::new(&program).unwrap();

        // Invoking to_jit method on a JitValue::Struct to convert it to a JIT representation.
        // Generating an error by providing a struct value instead of the expected type.
        let result = Value::Struct {
            fields: vec![Value::from(2u32)],
            debug_name: None,
        }
        .to_ptr(
            &Bump::new(),
            &registry,
            &program.type_declarations[0].id,
            |_| todo!(),
        )
        .unwrap_err(); // Unwrapping the error

        // Matching the error result to verify the error type and message.
        match result {
            Error::UnexpectedValue(expected_msg) => {
                // Asserting that the error message matches the expected message.
                assert_eq!(
                    expected_msg,
                    format!(
                        "expected value of type {:?} but got a struct",
                        program.type_declarations[0].id.debug_name
                    )
                );
            }
            _ => panic!("Unexpected error type: {:?}", result),
        }
    }
}

mod range_serde {
    use std::fmt;

    use cairo_lang_sierra::extensions::utils::Range;
    use serde::{
        de::{self, Visitor},
        ser::SerializeStruct,
        Deserializer, Serializer,
    };

    pub fn serialize<S>(range: &Range, ser: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = ser.serialize_struct("Range", 2)?;

        state.serialize_field("lower", &range.lower)?;
        state.serialize_field("upper", &range.upper)?;

        state.end()
    }

    pub fn deserialize<'de, D>(de: D) -> Result<Range, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct RangeVisitor;

        impl<'de> Visitor<'de> for RangeVisitor {
            type Value = Range;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("an integer between -2^31 and 2^31")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                let lower = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let upper = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;

                Ok(Range { lower, upper })
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: de::MapAccess<'de>,
            {
                let mut lower = None;
                let mut upper = None;

                while let Some((field, value)) = map.next_entry()? {
                    match field {
                        "lower" => {
                            lower = Some(value);
                        }
                        "upper" => {
                            upper = Some(value);
                        }
                        _ => return Err(de::Error::unknown_field(field, &["lower", "upper"])),
                    }
                }

                Ok(Range {
                    lower: lower.ok_or_else(|| de::Error::missing_field("lower"))?,
                    upper: upper.ok_or_else(|| de::Error::missing_field("upper"))?,
                })
            }
        }

        de.deserialize_struct("Range", &["lower", "upper"], RangeVisitor)
    }
}
