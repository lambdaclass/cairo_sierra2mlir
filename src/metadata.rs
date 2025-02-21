//! # Code generation metadata
//!
//! The metadata is used for various stuff that would be otherwise difficult or plain impossible
//! with the current design, such as:
//!   - Pass compile-time constants that would otherwise have to be hardcoded in various places (ex.
//!     [PrimeModuloMeta](self::prime_modulo)).
//!   - Declare FFI bindings to external libraries (ex.
//!     [ReallocBindingsMeta](self::realloc_bindings)).
//!   - Pass extra compilation info to the libfunc generators (ex.
//!     [TailRecursionMeta](self::tail_recursion)).

use std::{
    any::{Any, TypeId},
    collections::{hash_map::Entry, HashMap},
};

pub mod auto_breakpoint;
pub mod debug_utils;
pub mod drop_overrides;
pub mod dup_overrides;
pub mod enum_snapshot_variants;
pub mod felt252_dict;
pub mod gas;
pub mod realloc_bindings;
pub mod runtime_bindings;
pub mod tail_recursion;
pub mod trace_dump;

/// Metadata container.
#[cfg_attr(not(feature = "with-debug-utils"), derive(Default))]
#[derive(Debug)]
pub struct MetadataStorage {
    entries: HashMap<TypeId, Box<dyn Any>>,
}

impl MetadataStorage {
    /// Create an empty metadata container.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert some metadata and return a mutable reference.
    ///
    /// The insertion will fail if there is already some metadata with the same type, in which case
    /// it'll return `None`.
    pub fn insert<T>(&mut self, meta: T) -> Option<&mut T>
    where
        T: Any,
    {
        if let Entry::Vacant(e) = self.entries.entry(TypeId::of::<T>()) {
            e.insert(Box::new(meta));
            self.get_mut::<T>()
        } else {
            None
        }
    }

    /// Remove some metadata and return its last value.
    ///
    /// The removal will fail if there is no metadata with the requested type, in which case it'll
    /// return `None`.
    pub fn remove<T>(&mut self) -> Option<T>
    where
        T: Any,
    {
        self.entries.remove(&TypeId::of::<T>()).map(|meta| {
            *(Box::<(dyn Any + 'static)>::downcast::<T>(meta).expect(
                "attempt to downcast a boxed value to a type which does not match the actual",
            ))
        })
    }

    /// Retrieve a reference to some metadata.
    ///
    /// The retrieval will fail if there is no metadata with the requested type, in which case it'll
    /// return `None`.
    pub fn get<T>(&self) -> Option<&T>
    where
        T: Any,
    {
        self.entries.get(&TypeId::of::<T>()).map(|meta| {
            meta.downcast_ref::<T>().expect(
                "attempt to downcast a boxed value to a type which does not match the actual",
            )
        })
    }

    /// Retrieve a mutable reference to some metadata.
    ///
    /// The retrieval will fail if there is no metadata with the requested type, in which case it'll
    /// return `None`.
    pub fn get_mut<T>(&mut self) -> Option<&mut T>
    where
        T: Any,
    {
        self.entries.get_mut(&TypeId::of::<T>()).map(|meta| {
            meta.downcast_mut::<T>()
                .expect("the given type does not match the actual")
        })
    }

    pub fn get_or_insert_with<T>(&mut self, meta_gen: impl FnOnce() -> T) -> &mut T
    where
        T: Any,
    {
        self.entries
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(meta_gen()))
            .downcast_mut::<T>()
            .expect("the given type does not match the actual")
    }
}

#[cfg(feature = "with-debug-utils")]
impl Default for MetadataStorage {
    fn default() -> Self {
        let mut metadata = Self {
            entries: Default::default(),
        };

        metadata.insert(debug_utils::DebugUtils::default());

        metadata
    }
}

#[cfg(test)]
mod test {
    use super::{runtime_bindings::RuntimeBindingsMeta, *};

    #[test]
    fn runtime_library_insert_works() {
        // Create a new instance of MetadataStorage.
        let mut metadata = MetadataStorage::new();

        // Insert a new instance of RuntimeBindingsMeta.
        let ret = metadata.insert(RuntimeBindingsMeta::default());

        // Check that the insertion was successful by asserting that `ret` is not `None`.
        assert!(ret.is_some());
    }

    #[test]
    fn runtime_library_insert_non_vacant() {
        // Create a new instance of MetadataStorage.
        let mut metadata = MetadataStorage::new();

        // Insert a new instance of RuntimeBindingsMeta.
        let ret_vacant = metadata.insert(RuntimeBindingsMeta::default());

        // Check that the first insertion was successful.
        assert!(ret_vacant.is_some());

        // Attempt to insert another instance of RuntimeBindingsMeta.
        let ret_non_vacant = metadata.insert(RuntimeBindingsMeta::default());

        // Check that the second insertion failed because an instance of the same type already exists.
        assert!(ret_non_vacant.is_none());
    }
}
