use std::{
    alloc::Layout,
    cell::UnsafeCell,
    collections::{hash_map::Entry, HashMap},
    ptr,
};

thread_local! {
    static ALLOCATOR: UnsafeCell<ManagedAllocator> = UnsafeCell::new(ManagedAllocator::default());
}

// TODO: Replace `crate::utils::libc_free`, `crate::utils::libc_malloc`,
//   `crate::utils::libc_realloc` with our implementation.
// TODO: Merge runtime crate into library (after #1051).
// TODO: Register runtime symbols (after #1051).

pub fn register_runtime_symbols(find_symbol: impl Fn(&str) -> Option<*mut ()>) {
    if let Some(symbol) = find_symbol("cairo_native__alloc") {
        unsafe {
            *symbol.cast::<*const ()>() =
                impl_alloc as *const extern "C" fn(u64, u64) -> *mut () as *const ()
        }
    }

    if let Some(symbol) = find_symbol("cairo_native__realloc") {
        unsafe {
            *symbol.cast::<*const ()>() =
                impl_realloc as *const extern "C" fn(*mut (), u64) -> *mut () as *const ()
        }
    }

    if let Some(symbol) = find_symbol("cairo_native__free") {
        unsafe {
            *symbol.cast::<*const ()>() = impl_free as *const extern "C" fn(*mut ()) as *const ()
        }
    }
}

pub fn run_with_allocator<T>(f: impl FnOnce() -> T) -> T {
    let prev_allocator =
        ALLOCATOR.with(|x| unsafe { ptr::replace(x.get(), ManagedAllocator::default()) });

    let result = f();

    ALLOCATOR.with(|x| unsafe { ptr::write(x.get(), prev_allocator) });
    result
}

#[derive(Debug, Default)]
struct ManagedAllocator {
    allocs: HashMap<*mut u8, Layout>,
}

impl ManagedAllocator {
    pub fn alloc(&mut self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { std::alloc::alloc(layout) };
        self.allocs.insert(ptr, layout);

        ptr
    }

    pub fn realloc(&mut self, ptr: *mut u8, new_size: usize) -> *mut u8 {
        assert!(!ptr.is_null());
        match self.allocs.entry(ptr) {
            Entry::Occupied(mut entry) => {
                let new_ptr = unsafe { std::alloc::realloc(ptr, *entry.get(), new_size) };
                let new_layout = {
                    let layout = *entry.get();
                    Layout::from_size_align(layout.size(), layout.align()).unwrap()
                };

                if ptr == new_ptr {
                    entry.insert(new_layout);
                } else {
                    entry.remove();
                    self.allocs.insert(new_ptr, new_layout);
                }

                new_ptr
            }
            Entry::Vacant(_) => panic!(),
        }
    }

    pub fn dealloc(&mut self, ptr: *mut u8) {
        let layout = self.allocs.remove(&ptr).unwrap();
        unsafe { std::alloc::dealloc(ptr, layout) }
    }
}

impl Drop for ManagedAllocator {
    fn drop(&mut self) {
        for (ptr, layout) in self.allocs.drain() {
            unsafe { std::alloc::dealloc(ptr, layout) }
        }
    }
}

extern "C" fn impl_alloc(size: u64, align: u64) -> *mut () {
    // let layout = Layout::from_size_align(size, align).unwrap();

    todo!()
}

extern "C" fn impl_realloc(ptr: *mut (), new_size: u64) -> *mut () {
    todo!()
}

extern "C" fn impl_free(ptr: *mut ()) {}
