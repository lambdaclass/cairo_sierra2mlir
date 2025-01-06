use libc::{
    c_int, sigaction, siginfo_t, sigset_t, stack_t, ucontext_t, MAP_ANONYMOUS, MAP_FAILED,
    MAP_FIXED, MAP_FIXED_NOREPLACE, MAP_NORESERVE, MAP_PRIVATE, MAP_SHARED, MAP_STACK, PROT_NONE,
    PROT_READ, PROT_WRITE, REG_RSP, SA_ONSTACK, SA_SIGINFO, SIGSEGV, SIGSTKSZ, _SC_PAGESIZE,
};
use std::{
    backtrace::Backtrace,
    cell::{Cell, UnsafeCell},
    mem::MaybeUninit,
    ptr::null_mut,
};

extern "C" {
    fn setjmp(env: *mut ()) -> c_int;
    fn longjmp(env: *mut (), val: c_int);
}

thread_local! {
    static JMP_BUF: UnsafeCell<MaybeUninit::<[u8; 1024]>> = const { UnsafeCell::new(MaybeUninit::uninit()) };
    static CURRENT_RUNNER: Cell<*mut SafeRunner> = const { Cell::new(null_mut()) };
}

#[derive(Debug)]
pub struct SafeRunner {
    stack_fd: c_int,

    page_size: usize,
    stack_maps: Vec<(*mut (), usize)>,

    signal_stack: Box<MaybeUninit<[u8; SIGSTKSZ]>>,
    error_result: Option<SafeRunnerError>,
}

impl SafeRunner {
    pub fn new(num_pages: usize) -> Self {
        let page_size = unsafe { libc::sysconf(_SC_PAGESIZE) } as usize;

        let stack_fd = unsafe { libc::memfd_create(b"stack_fd\0".as_ptr() as _, 0) };

        let mut self_value = Self {
            stack_fd,

            page_size,
            stack_maps: vec![],

            signal_stack: Box::new_uninit(),
            error_result: None,
        };
        self_value.grow_stack(num_pages);

        self_value
    }

    pub fn run_without_stack_swap<F>(&mut self, closure: F) -> Result<(), SafeRunnerError>
    where
        F: FnOnce(*mut ()),
    {
        let mut prev_signal_stack = MaybeUninit::<stack_t>::uninit();
        let mut prev_signal_handler = MaybeUninit::<sigaction>::uninit();

        // Configure early return (crash handler).
        let jmp_buf = JMP_BUF.with(|x| unsafe { (*x.get()).as_mut_ptr().cast() });
        if unsafe { setjmp(jmp_buf) } == 0 {
            // Configure signal stack.
            unsafe {
                libc::sigaltstack(
                    &{
                        stack_t {
                            ss_sp: self.signal_stack.as_mut_ptr().cast(),
                            ss_flags: 0,
                            ss_size: SIGSTKSZ,
                        }
                    },
                    prev_signal_stack.as_mut_ptr(),
                );
            }

            // Configure signal handler.
            unsafe {
                libc::sigaction(
                    SIGSEGV,
                    &sigaction {
                        sa_sigaction: segfault_handler
                            as *const extern "C" fn(c_int, &siginfo_t, &mut ucontext_t)
                            as usize,
                        sa_mask: MaybeUninit::<sigset_t>::zeroed().assume_init(),
                        sa_flags: SA_ONSTACK | SA_SIGINFO,
                        sa_restorer: None,
                    },
                    prev_signal_handler.as_mut_ptr(),
                );
            }

            let (stack_ptr, num_pages) = self.stack_maps.last().copied().unwrap();

            let prev_runner = CURRENT_RUNNER.replace(self);
            closure(unsafe { stack_ptr.byte_add(num_pages * self.page_size + self.page_size) });
            CURRENT_RUNNER.set(prev_runner);
        }

        // Restore signal handler.
        unsafe {
            libc::sigaction(SIGSEGV, prev_signal_handler.as_ptr(), null_mut());
        }

        // Restore signal stack.
        unsafe {
            libc::sigaltstack(prev_signal_stack.as_mut_ptr(), null_mut());
        }

        match self.error_result.take() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn grow_stack(&mut self, extra_pages: usize) -> isize {
        unsafe fn inner(
            runner: &SafeRunner,
            stack_ptr: *mut (),
            num_pages: usize,
            page_offset: usize,
        ) -> Result<*mut (), ()> {
            // Attempt to mmap the new region.
            let new_stack_ptr = libc::mmap(
                stack_ptr.cast(),
                num_pages * runner.page_size
                    + stack_ptr
                        .is_null()
                        .then_some(runner.page_size)
                        .unwrap_or_default(),
                PROT_NONE,
                MAP_ANONYMOUS
                    | MAP_NORESERVE
                    | MAP_PRIVATE
                    | MAP_STACK
                    | (!stack_ptr.is_null())
                        .then_some(MAP_FIXED_NOREPLACE)
                        .unwrap_or_default(),
                -1,
                0,
            );
            if new_stack_ptr == MAP_FAILED {
                return Err(());
            }

            // Ensure the backing file has enough space.
            assert_eq!(
                libc::fallocate(
                    runner.stack_fd,
                    0,
                    (page_offset * runner.page_size) as i64,
                    (num_pages * runner.page_size) as i64,
                ),
                0
            );

            // Map individual pages (in reverse order, the stack grows that way).
            for i in 1..=num_pages {
                let stack_ptr = new_stack_ptr.byte_add(i * runner.page_size);
                let fd_offset = (page_offset + num_pages - i) * runner.page_size;

                let result_ptr = libc::mmap(
                    stack_ptr,
                    runner.page_size,
                    PROT_READ | PROT_WRITE,
                    MAP_FIXED | MAP_NORESERVE | MAP_SHARED | MAP_STACK,
                    runner.stack_fd,
                    fd_offset as i64,
                );
                assert_ne!(
                    result_ptr,
                    MAP_FAILED,
                    "{}",
                    std::ffi::CStr::from_ptr(libc::strerror(*libc::__errno_location()))
                        .to_string_lossy()
                );
            }

            Ok(new_stack_ptr.cast())
        }

        match self.stack_maps.pop() {
            Some((stack_ptr, num_pages)) => unsafe {
                match inner(
                    self,
                    stack_ptr.byte_sub(extra_pages * self.page_size),
                    extra_pages,
                    num_pages,
                ) {
                    Ok(new_stack_ptr) => {
                        self.stack_maps
                            .push((new_stack_ptr, num_pages + extra_pages));

                        0
                    }
                    Err(_) => {
                        self.stack_maps.push((stack_ptr, num_pages));

                        println!("REMAPPING");
                        let total_pages = num_pages + extra_pages;
                        let new_stack_ptr = inner(self, null_mut(), total_pages, 0).unwrap();
                        self.stack_maps.push((new_stack_ptr, total_pages));

                        new_stack_ptr
                            .byte_add(num_pages * self.page_size)
                            .byte_offset_from(stack_ptr)
                    }
                }
            },
            None => unsafe {
                let new_stack_ptr = inner(self, null_mut(), extra_pages, 0).unwrap();
                self.stack_maps.push((new_stack_ptr, extra_pages));

                0
            },
        }
    }
}

impl Drop for SafeRunner {
    fn drop(&mut self) {
        for (stack_ptr, num_pages) in self.stack_maps.drain(..) {
            let stack_size = num_pages * self.page_size + self.page_size;
            assert_eq!(unsafe { libc::munmap(stack_ptr.cast(), stack_size) }, 0);
        }

        assert_eq!(unsafe { libc::close(self.stack_fd) }, 0);
    }
}

#[derive(Debug)]
pub enum SafeRunnerError {
    SegmentationFault,
}

unsafe extern "C" fn segfault_handler(sig: c_int, info: &siginfo_t, context: &mut ucontext_t) {
    let fault_addr = info.si_addr();
    println!("SEGFAULT at addr {:?}", fault_addr);

    let runner = unsafe { &mut *CURRENT_RUNNER.get() };

    // If any segment's guard (except the last segment's) contains the fault address, just replace
    // the stack pointer with the last allocation.
    let (&(stack_ptr, _), segments) = runner.stack_maps.split_last().unwrap();

    if let Some(stack_ptr) = segments.iter().copied().find_map(|(stack_ptr, _)| {
        let stack_guard = stack_ptr..stack_ptr.byte_add(runner.page_size);
        stack_guard
            .contains(&fault_addr.cast())
            .then_some(stack_ptr)
    }) {
        todo!("asdf")
    }

    let stack_guard = stack_ptr..stack_ptr.byte_add(runner.page_size);
    if stack_guard.contains(&fault_addr.cast()) {
        let rsp_offset = runner.grow_stack(4);
        println!(
            "Stack size: {} bytes",
            runner.stack_maps.last().copied().unwrap().1 * runner.page_size
        );

        let rsp = context.uc_mcontext.gregs[REG_RSP as usize] as *mut ();
        context.uc_mcontext.gregs[REG_RSP as usize] = rsp.byte_offset(rsp_offset) as i64;
    } else {
        runner.error_result = Some(SafeRunnerError::SegmentationFault);

        let jmp_buf = JMP_BUF.with(|x| unsafe { (*x.get()).as_mut_ptr().cast() });
        longjmp(jmp_buf, 0);
    }
}

#[cfg(test)]
mod test {
    use super::SafeRunner;
    use std::{arch::asm, hint::black_box, mem::forget, ptr::null_mut};

    fn wrap_stack<F>(stack_ptr: *mut (), mut f: F)
    where
        F: FnOnce(),
    {
        unsafe extern "C" fn trampoline<F>(f: *mut F)
        where
            F: FnOnce(),
        {
            (f.read())();
        }

        unsafe {
            // Using `r12` as it won't be modified by the function call. Any other preserved
            // register should work too.
            asm!(
                "xchg rsp, r12",
                "call {f}",
                "mov rsp, r12",
                f = in(reg) trampoline::<F>,
                in("rdi") &raw mut f,
                in("r12") stack_ptr,
            );
        }

        forget(f);
    }

    #[test]
    fn safe_runner_ok() {
        let mut runner = SafeRunner::new(4);
        runner
            .run_without_stack_swap(|stack_ptr| {
                wrap_stack(stack_ptr, || {
                    println!("Hello, world!");
                })
            })
            .unwrap();
    }

    #[test]
    #[should_panic]
    fn safe_runner_segfault() {
        let mut runner = SafeRunner::new(4);
        runner
            .run_without_stack_swap(|stack_ptr| {
                wrap_stack(stack_ptr, || unsafe {
                    *null_mut::<u32>().byte_add(4) = 0;
                })
            })
            .unwrap();
    }

    #[test]
    fn safe_runner_stack_overflow() {
        let mut runner = SafeRunner::new(4);
        runner
            .run_without_stack_swap(|stack_ptr| {
                wrap_stack(stack_ptr, || {
                    fn f(n: usize) -> usize {
                        match n {
                            0 | 1 => 1,
                            _ => black_box(n.wrapping_mul(f(n - 1))),
                        }
                    }

                    println!("{}", f(2047));
                })
            })
            .unwrap();
    }
}
