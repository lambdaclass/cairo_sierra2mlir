# Execution Walkthrough

Let's walk through the execution of the following Cairo program:

```rust,ignore
// This is the cairo program. It just adds two numbers together and returns the
// result in an enum whose variant is selected using the result's parity.
enum Parity<T> {
    Even: T,
    Odd: T,
}

/// Add `lhs` and `rhs` together and return the result in `Parity::Even` if it's
/// even or `Parity::Odd` otherwise.
fn run(lhs: u128, rhs: u128) -> Parity<u128> {
    let res = lhs + rhs;
    if (res & 1) == 0 {
        Parity::Even(res)
    } else {
        Parity::Odd(res)
    }
}
```

First, we need to compile the program to Sierra and then MLIR:

```rust,ignore
// Compile the Cairo to Sierra (using the Cairo compiler).
let program = get_sierra_program();

// Compile the Sierra to MLIR (using Cairo native, not covered here).
let module = get_native_module(&program);
```

## Execution engine preparation

Once we have the lowered MLIR module (using only the LLVM dialect) we can
instantiate an execution engine.

There's two kind of execution engines:

- The just-in-time (JIT) engine: Generates machine code on the fly. Can be
  optimized further taking into account hot paths and other metrics.
- The ahead-of-time (AOT) engine: Uses pre-generated machine code. Has lower
  overhead because the machine code is fixed and already compiled, but cannot be
  optimized further.

### Using the JIT executor

Using the JIT executor is the easiest option, since we just need to create it
and we're done:

```rust,ignore
let program = get_sierra_program();
let module = get_native_module(&program);

// The JIT engine accepts an optimization level. The available optimization
// levels are:
//   - `OptLevel::None`: Applies no optimization (other than what's already been
//     optimized by earlier passes).
//   - `OptLevel::Less`: Uses only a reduced set of optimizations.
//   - `OptLevel::Default`: The default.
//   - `OptLevel::Aggressive`: Tries to apply all the (safe) optimizations.
// They're equivalent to using `-O0`, `-O1`, `-O2` and `-O3` when compiling
// C/C++ respectively.
let engine = JitNativeExecutor::from_native_module(module, OptLevel::Default);
```

### Using the AOT executor

Using the AOT executor is a bit more complicated because we need to compile it
into a shared library on disk, but all that complexity has been hidden within
the `AotNativeExecutor::from_native_module` method:

```rust,ignore
let program = get_sierra_program();
let module = get_native_module(&program);

// Check out the previous section for information about `OptLevel`.
let engine = AotNativeExecutor::from_native_module(module, OptLevel::Default);
```

### Caching the compiled programs

Some use cases may benefit from storing the final (machine code) programs. Both
the JIT and AOT programs can be cached within the same process using the
`JitProgramCache` or `AotProgramCache` respectively, or just `ProgramCache` for
a cache that supports both. However, only the AOT supports persisting programs
between runs. They are stored using a different API from the `AotProgramCache`.

```rust,ignore
// An `Option<...>` is returned, indicating whether the program was present or
// not.
let executor = program_cache.get(key).unwrap();
```

## Invoking the program

Invoking the program involves the same steps for both AOT and JIT executors.
There are various methods that may help with invoking both normal programs and
Starknet contracts:

- `invoke_dynamic`: Call into a normal program that doesn't require a syscall
  handler.
- `invoke_dynamic_with_syscall_handler`: Same as before, but providing a syscall
  handler in case the program needs it.
- `invoke_contract_dynamic`: Call a contract's entry point. It accepts the entry
  point's ABI (a span of felts) instead of `Value`s and requires a syscall
  handler.

There's an extra, more performant way to invoke programs and contracts when we
know the exact signature of the function: we should obtain the function pointer,
cast it into an `extern "C" fn(...) -> ...` and invoke it directly from Rust. It
requires the user to convert the inputs and outputs into/from the expected
internal representation, and to manage the builtins manually. Because of that,
it has not been covered here.

All those methods for invoking the program need to know which entrypoint we're
trying to call. We can use the Sierra's function id directly.

Then we'll need the arguments. Since they can have any supported type in any
order we need to wrap them all in `Value`s and send those to the invoke method.
Builtins are automatically added by the invoke method and should be skipped.

```rust,ignore
let engine = get_execution_engine(); // This creates the execution engine (covered before).

let args = [
  Value::Uint128(1234),
  Value::Uint128(4321),
];
```

Finally we can invoke the program like this:

```rust,ignore
let engine = get_execution_engine();

let function_id = find_function_id(&program, "program::program::main(f0)");
let args = [
  Value::Uint128(1234),
  Value::Uint128(4321),
];

let execution_result = engine.invoke_dynamic(
  function_id, // The entry point function id.
  args,        // The slice of `JitValue`s.
  None,        // The available gas (if any).
)?;

// The return value has some useful information about the execution, like:
//   - The remaining gas, if any was supplied.
//   - The program's return value.
//   - The builtin usage statistics. These contain the number of times each builtin has been used.
println!("Remaining gas: {:?}",  execution_result.remaining_gas);
println!("Return value:  {:#?}", execution_result.return_value);
println!("Builtin stats: {:?}",  execution_result.builtin_stats);
```

Running the code above should print the following:

```rust,ignore
Remaining gas: None
Return value:  Enum {
  tag: 0,
  value: Struct {
    fields: [
      Enum {
        tag: 1,
        value: Uint128(
          5555,
        ),
        debug_name: Some(
          "sample::sample::Parity::<core::integer::u128>",
        ),
      },
    ],
    debug_name: Some(
      "Tuple<sample::sample::Parity::<core::integer::u128>>",
    ),
  },
  debug_name: Some(
    "core::panics::PanicResult::<(sample::sample::Parity::<core::integer::u128>,)>",
  ),
}
Builtin stats: BuiltinStats { bitwise: 1, ec_op: 0, range_check: 1, pedersen: 0, poseidon: 0, segment_arena: 0 }
```

### Contracts

Contracts always have the same interface, therefore they have an alternative to
`invoke_dynamic` called `invoke_contract_dynamic`.

```rust,ignore
fn(Span<felt252>) -> PanicResult<Span<felt252>>;
```

This wrapper will attempt to deserialize the real contract arguments from the
span of felts, invoke the contracts, and finally serialize and return the
result. When this deserialization fails, the contract will panic with the
mythical `Failed to deserialize param #N` error.

If the example program had the same interface as a contract (a span of felts)
then it'd be invoked like this:

```rust,ignore
let engine = get_execution_engine();

let function_id = find_function_id(&program, "program::program::main(f0)");
let args = [
  Felt::from(1234),
  Felt::from(4321),
];

let execution_result = engine.invoke_dynamic(
  function_id, // The entry point function id.
  args,        // The slice of `JitValue`s.
  None,        // The available gas (if any).
)?;

// The return value has some useful information about the execution, like:
//   - The remaining gas, if any was supplied.
//   - Whether the contract execution panicked.
//   - The contract's return values.
//   - The builtin usage statistics. These contain the number of times each builtin has been used.
println!("Remaining gas: {:?}", execution_result.remaining_gas);
println!("Failure flag:  {:?}", execution_result.failure_flag);
println!("Return value:  {:?}", execution_result.return_value);
println!("Builtin stats: {:?}", execution_result.builtin_stats);
```

Running the code above should print the following:

```text
Remaining gas: None
Failure flag:  false
Return value:  [
  JitValue::Felt252(1),
  JitValue::Felt252(5555),
]
Builtin stats: BuiltinStats { bitwise: 1, ec_op: 0, range_check: 1, pedersen: 0, poseidon: 0, segment_arena: 0 }
```

## The Cairo Native runtime

Sometimes we need to use stuff that would be too complicated or error-prone to
implement in MLIR, but that we have readily available from Rust. That's when we
use the runtime library.

When using the JIT it'll be automatically linked (if compiled with support for
it, which is enabled by default). If using the AOT, the
`CAIRO_NATIVE_RUNTIME_LIBRARY` environment variable will have to be modified to
point to the `libcairo_native_runtime.a` file, which is built and placed in said
folder by `make build`.

Although it's implemented in Rust, its functions use the C ABI and have Rust's
name mangling disabled. This means that to the extern observer it's technically
indistinguishible from a library written in C. By doing this we're making the
functions callable from MLIR.

### Syscall handlers

The syscall handler is similar to the runtime in the sense that we have
C-compatible functions called from MLIR, but it's different in that they're
built into Cairo Native itself rather than an external library, and that their
implementation is user-dependent.

To allow for user-provided syscall handler implementations we pass a pointer to
a vtable every time we detect a `System` builtin. We need a vtable and cannot
use function names because the methods themselves are generic over the syscall
handler implementation.

> Note: The `System` is used only for syscalls; every syscall has it, therefore
> it's a perfect candidate for this use.

Those wrappers then receive a mutable reference to the syscall handler
implementation. They are responsible of converting the MLIR-compatible inputs to
the Rust representations, calling the implementation, and then converting the
results back into MLIR-compatible formats.

This means that as far as the user is concerned, writing a syscall handler is
equivalent to implementing the trait `StarknetSyscallHandler` for a custom type.

## Appendix: The C ABI and the trampoline

Normally, calling FFI functions in Rust is as easy as defining an extern
function using C-compatible types. We can't do this here because we don't know
the function's signature.

It all boils down to the
[SystemV ABI](https://refspecs.linuxbase.org/elf/x86_64-abi-0.99.pdf) in
`x86_64` or its equivalent for ARM. Both of them are really similar:

- The stack must be aligned to 16 bytes before calling.
- Function arguments are spread between some registers and the stack.
- Return values use either a few registers or require a pointer.

There's a few other quirks, like which registers are caller vs callee-saved, but they're not that relevant in this case.

### Arguments

Argument location in `x86_64`:
| # | Reg. | Description |
|----|-------|------------------------|
| 1 | rdi | A single 64-bit value. |
| 2 | rsi | A single 64-bit value. |
| 3 | rdx | A single 64-bit value. |
| 4 | rcx | A single 64-bit value. |
| 5 | r8 | A single 64-bit value. |
| 6 | r9 | A single 64-bit value. |
| 7+ | Stack | Everything else. |

Argument location in `aarch64`:

| #   | Reg.  | Description            |
| --- | ----- | ---------------------- |
| 1   | x0    | A single 64-bit value. |
| 2   | x1    | A single 64-bit value. |
| 3   | x2    | A single 64-bit value. |
| 4   | x3    | A single 64-bit value. |
| 5   | x4    | A single 64-bit value. |
| 6   | x5    | A single 64-bit value. |
| 7   | x6    | A single 64-bit value. |
| 8   | x7    | A single 64-bit value. |
| 9+  | Stack | Everything else.       |

Usually function calls have arguments of types other than just 64-bit integers.
In those cases, for values smaller than 64 bits the smaller register variants
are written. For values larger than 64 bits the value is split into multiple
registers, but there's a catch: if when splitting the value only one value would
remain in registers then that register is padded and the entire value goes into
the stack. For example, an `u128` that would be split between registers and the
stack is always padded and written entirely in the stack.

For complex values like structs, the types are flattened into a list of values
when written into registers, or just written into the stack the same way they
would be written into memory (aka. with the correct alignment, etc).

### Return values

As mentioned before, return values may be either returned in registers or memory
(most likely the stack, but not necessarily).

Argument location in `x86_64`:

| #   | Reg | Description                 |
| --- | --- | --------------------------- |
| 1   | rax | A single 64-bit value.      |
| 2   | rdx | The "continuation" of `rax` |

Argument location in `aarch64`:

| #   | Reg | Description                |
| --- | --- | -------------------------- |
| 1   | x0  | A single 64-bit value      |
| 2   | x1  | The "continuation" of `x0` |
| 3   | x2  | The "continuation" of `x1` |
| 4   | x3  | The "continuation" of `x2` |

Values are different that arguments in that only a single value is returned. If
more than a single value needs to be returned then it'll use a pointer.

When a pointer is involved we need to pass it as the first argument. This means
that every actual argument has to be shifted down one slot, pushing more stuff
into the stack in the process.

### The trampoline

We cannot really influence what values are in the register or the stack from
Rust, therefore we need something written in assembler to put everything into
place and invoke the function pointer.

This is where the trampoline comes in. It's a simple assembler function that
does three things:

1. Fill in the 6 or 8 argument registers with the first values in the data
   pointer and copy the rest into the stack as-is (no stack alignment or anything,
   we guarantee from the Rust side that the stack will end up properly aligned).
2. Invoke the function pointer.
3. Write the return values (in registers only) into the return pointer.

This function always has the same signature, which is C-compatible, and
therefore can be used with Rust's FFI facilities without problems.

#### AOT calling convention:

##### Arguments

- Written on registers, then the stack.
- Structs' fields are treated as individual arguments (flattened).
- Enums are structs internally, therefore they are also flattened (including the padding).
  - The default payload works as expected since it has the correct signature.
  - All other payloads require breaking it down into bytes and scattering it
    through the padding and default payload's space.

##### Return values

- Indivisible values that do not fit within a single register (ex. felt252) use multiple registers (x0-x3 for felt252).
- Struct arguments, etc... use the stack.

In other words, complex values require a return pointer while simple values do
not but may still use multiple registers if they don't fit within one.
