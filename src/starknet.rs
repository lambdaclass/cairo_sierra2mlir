#![allow(clippy::type_complexity)]
#![allow(dead_code)]

use cairo_felt::Felt252;

type SyscallResult<T> = std::result::Result<T, Vec<Felt252>>;

/// Binary representation of a `felt252` (in MLIR).
#[derive(Debug, Clone)]
#[repr(C, align(8))]
struct Felt252Abi(pub [u8; 32]);
/// Binary representation of a `u256` (in MLIR).
// TODO: This shouldn't need to be public.
#[derive(Debug, Clone)]
#[repr(C, align(8))]
pub struct U256(pub [u8; 32]);

pub struct ExecutionInfo {
    // TODO: Add fields.
}
pub struct Secp256k1Point {
    // TODO: Add fields.
}
pub struct Secp256r1Point {
    // TODO: Add fields.
}

pub trait StarkNetSyscallHandler {
    fn get_block_hash(&self, block_number: u64) -> SyscallResult<Felt252>;
    fn get_execution_info(&self) -> SyscallResult<ExecutionInfo>;

    fn deploy(
        &self,
        class_hash: Felt252,
        contract_address_salt: Felt252,
        calldata: &[Felt252],
        deploy_from_zero: bool,
    ) -> SyscallResult<(Felt252, Vec<Felt252>)>;
    fn replace_class(&self, class_hash: Felt252) -> SyscallResult<()>;

    fn library_call(
        &self,
        class_hash: Felt252,
        function_selector: Felt252,
        calldata: &[Felt252],
    ) -> SyscallResult<Vec<Felt252>>;
    fn call_contract(
        &self,
        address: Felt252,
        entry_point_selector: Felt252,
        calldata: &[Felt252],
    ) -> SyscallResult<Vec<Felt252>>;

    fn storage_read(&self, address_domain: u32, address: Felt252) -> SyscallResult<Felt252>;
    fn storage_write(
        &self,
        address_domain: u32,
        address: Felt252,
        value: Felt252,
    ) -> SyscallResult<()>;

    fn emit_event(&self, keys: &[Felt252], data: &[Felt252]) -> SyscallResult<()>;
    fn send_message_to_l1(&self, to_address: Felt252, payload: &[Felt252]) -> SyscallResult<()>;

    fn keccak(&self, input: &[u64]) -> SyscallResult<U256>;

    // TODO: secp256k1 syscalls
    fn secp256k1_add(
        &self,
        p0: Secp256k1Point,
        p1: Secp256k1Point,
    ) -> SyscallResult<Option<Secp256k1Point>>;
    fn secp256k1_get_point_from_x(
        &self,
        x: U256,
        y_parity: bool,
    ) -> SyscallResult<Option<Secp256k1Point>>;
    fn secp256k1_get_xy(&self, p: Secp256k1Point) -> SyscallResult<(U256, U256)>;
    fn secp256k1_mul(&self, p: Secp256k1Point, m: U256) -> SyscallResult<Option<Secp256k1Point>>;
    fn secp256k1_new(&self, x: U256, y: U256) -> SyscallResult<Option<Secp256k1Point>>;

    // TODO: secp256r1 syscalls
    fn secp256r1_add(
        &self,
        p0: Secp256k1Point,
        p1: Secp256k1Point,
    ) -> SyscallResult<Option<Secp256k1Point>>;
    fn secp256r1_get_point_from_x(
        &self,
        x: U256,
        y_parity: bool,
    ) -> SyscallResult<Option<Secp256k1Point>>;
    fn secp256r1_get_xy(&self, p: Secp256k1Point) -> SyscallResult<(U256, U256)>;
    fn secp256r1_mul(&self, p: Secp256k1Point, m: U256) -> SyscallResult<Option<Secp256k1Point>>;
    fn secp256r1_new(&self, x: U256, y: U256) -> SyscallResult<Option<Secp256k1Point>>;

    // Testing syscalls.
    // TODO: Make them optional. Crash if called but not implemented.
    fn pop_log(&self);
    fn set_account_contract_address(&self, contract_address: Felt252);
    fn set_block_number(&self, block_number: u64);
    fn set_block_timestamp(&self, block_timestamp: u64);
    fn set_caller_address(&self, address: Felt252);
    fn set_chain_id(&self, chain_id: Felt252);
    fn set_contract_address(&self, address: Felt252);
    fn set_max_fee(&self, max_fee: u128);
    fn set_nonce(&self, nonce: Felt252);
    fn set_sequencer_address(&self, address: Felt252);
    fn set_signature(&self, signature: &[Felt252]);
    fn set_transaction_hash(&self, transaction_hash: Felt252);
    fn set_version(&self, version: Felt252);
}

// TODO: Move to the correct place or remove if unused.
pub(crate) mod handler {
    use super::*;
    use std::{alloc::Layout, mem::ManuallyDrop, ptr::NonNull};

    #[repr(C)]
    struct SyscallResultAbi<T> {
        tag: u8,
        payload: SyscallResultPayloadAbi<T>,
    }

    #[repr(C)]
    union SyscallResultPayloadAbi<T> {
        ok: ManuallyDrop<T>,
        err: (NonNull<Felt252Abi>, u32, u32),
    }

    #[repr(C)]
    pub struct StarkNetSyscallHandlerCallbacks<'a, T>
    where
        T: StarkNetSyscallHandler,
    {
        self_ptr: &'a T,

        get_block_hash: extern "C" fn(
            result_ptr: &mut SyscallResultAbi<Felt252Abi>,
            ptr: &mut T,
            gas: &mut u64,
            block_number: u64,
        ),
        get_execution_info: extern "C" fn(
            result_ptr: &mut SyscallResultAbi<ExecutionInfo>,
            ptr: &mut T,
            gas: &mut u64,
        ),
        deploy: extern "C" fn(
            result_ptr: &mut SyscallResultAbi<(Felt252Abi, (NonNull<Felt252Abi>, u32, u32))>,
            ptr: &mut T,
            gas: &mut u64,
            class_hash: &Felt252Abi,
            contract_address_salt: &Felt252Abi,
            calldata: *const (*const Felt252Abi, u32, u32),
            deploy_from_zero: bool,
        ),
        replace_class: extern "C" fn(
            result_ptr: &mut SyscallResultAbi<()>,
            ptr: &mut T,
            _gas: &mut u64,
            class_hash: &Felt252Abi,
        ),
        library_call: extern "C" fn(
            result_ptr: &mut SyscallResultAbi<(NonNull<Felt252Abi>, u32, u32)>,
            ptr: &mut T,
            gas: &mut u64,
            class_hash: &Felt252Abi,
            function_selector: &Felt252Abi,
            calldata: *const (*const Felt252Abi, u32, u32),
        ),
        call_contract: extern "C" fn(
            result_ptr: &mut SyscallResultAbi<(NonNull<Felt252Abi>, u32, u32)>,
            ptr: &mut T,
            gas: &mut u64,
            address: &Felt252Abi,
            entry_point_selector: &Felt252Abi,
            calldata: *const (*const Felt252Abi, u32, u32),
        ),

        storage_read: extern "C" fn(
            result_ptr: &mut SyscallResultAbi<Felt252Abi>,
            ptr: &mut T,
            gas: &mut u64,
            address_domain: u32,
            address: &Felt252Abi,
        ),
        storage_write: extern "C" fn(
            result_ptr: &mut SyscallResultAbi<()>,
            ptr: &mut T,
            gas: &mut u64,
            address_domain: u32,
            address: &Felt252Abi,
            value: &Felt252Abi,
        ),
    }

    impl<'a, T> StarkNetSyscallHandlerCallbacks<'a, T>
    where
        T: StarkNetSyscallHandler + 'a,
    {
        pub fn new(handler: &'a T) -> Self {
            Self {
                self_ptr: handler,
                get_block_hash: Self::wrap_get_block_hash,
                get_execution_info: Self::wrap_get_execution_info,
                deploy: Self::wrap_deploy,
                replace_class: Self::wrap_replace_class,
                library_call: Self::wrap_library_call,
                call_contract: Self::wrap_call_contract,
                storage_read: Self::wrap_storage_read,
                storage_write: Self::wrap_storage_write,
            }
        }

        unsafe fn alloc_mlir_array<E: Clone>(data: &[E]) -> (NonNull<E>, u32, u32) {
            let ptr = libc::malloc(Layout::array::<E>(data.len()).unwrap().size()) as *mut E;

            let len: u32 = data.len().try_into().unwrap();
            for (i, val) in data.iter().enumerate() {
                ptr.add(i).write(val.clone());
            }

            (NonNull::new(ptr).unwrap(), len, len)
        }

        fn wrap_error<E>(e: &[Felt252]) -> SyscallResultAbi<E> {
            SyscallResultAbi {
                tag: 1u8,
                payload: unsafe {
                    let data: Vec<_> = e.iter().map(|x| Felt252Abi(x.to_le_bytes())).collect();

                    SyscallResultPayloadAbi {
                        err: Self::alloc_mlir_array(&data),
                    }
                },
            }
        }

        extern "C" fn wrap_get_block_hash(
            result_ptr: &mut SyscallResultAbi<Felt252Abi>,
            ptr: &mut T,
            _gas: &mut u64,
            block_number: u64,
        ) {
            // TODO: Handle gas.
            let result = ptr.get_block_hash(block_number);

            *result_ptr = match result {
                Ok(x) => SyscallResultAbi {
                    tag: 0u8,
                    payload: SyscallResultPayloadAbi {
                        ok: ManuallyDrop::new(Felt252Abi(x.to_le_bytes())),
                    },
                },
                Err(e) => Self::wrap_error(&e),
            };
        }

        extern "C" fn wrap_get_execution_info(
            result_ptr: &mut SyscallResultAbi<ExecutionInfo>,
            ptr: &mut T,
            _gas: &mut u64,
        ) {
            // TODO: handle gas
            let result = ptr.get_execution_info();

            *result_ptr = match result {
                Ok(x) => SyscallResultAbi {
                    tag: 0u8,
                    payload: SyscallResultPayloadAbi {
                        ok: ManuallyDrop::new(x),
                    },
                },
                Err(e) => Self::wrap_error(&e),
            };
        }

        // TODO: change all from_bytes_be to from_bytes_ne when added.

        extern "C" fn wrap_deploy(
            result_ptr: &mut SyscallResultAbi<(Felt252Abi, (NonNull<Felt252Abi>, u32, u32))>,
            ptr: &mut T,
            _gas: &mut u64,
            class_hash: &Felt252Abi,
            contract_address_salt: &Felt252Abi,
            calldata: *const (*const Felt252Abi, u32, u32),
            deploy_from_zero: bool,
        ) {
            // TODO: handle gas
            let class_hash = Felt252::from_bytes_be(&class_hash.0);
            let contract_address_salt = Felt252::from_bytes_be(&contract_address_salt.0);

            let calldata: Vec<_> = unsafe {
                let len = (*calldata).1 as usize;

                std::slice::from_raw_parts((*calldata).0, len)
            }
            .iter()
            .map(|x| Felt252::from_bytes_be(&x.0))
            .collect();

            let result = ptr.deploy(
                class_hash,
                contract_address_salt,
                &calldata,
                deploy_from_zero,
            );

            *result_ptr = match result {
                Ok(x) => {
                    let felts: Vec<_> = x.1.iter().map(|x| Felt252Abi(x.to_le_bytes())).collect();
                    let felts_ptr = unsafe { Self::alloc_mlir_array(&felts) };
                    SyscallResultAbi {
                        tag: 0u8,
                        payload: SyscallResultPayloadAbi {
                            ok: ManuallyDrop::new((Felt252Abi(x.0.to_le_bytes()), felts_ptr)),
                        },
                    }
                }
                Err(e) => Self::wrap_error(&e),
            };
        }

        extern "C" fn wrap_replace_class(
            result_ptr: &mut SyscallResultAbi<()>,
            ptr: &mut T,
            _gas: &mut u64,
            class_hash: &Felt252Abi,
        ) {
            // TODO: Handle gas.
            let class_hash = Felt252::from_bytes_be(&class_hash.0);
            let result = ptr.replace_class(class_hash);

            *result_ptr = match result {
                Ok(_) => SyscallResultAbi {
                    tag: 0u8,
                    payload: SyscallResultPayloadAbi {
                        ok: ManuallyDrop::new(()),
                    },
                },
                Err(e) => Self::wrap_error(&e),
            };
        }

        extern "C" fn wrap_library_call(
            result_ptr: &mut SyscallResultAbi<(NonNull<Felt252Abi>, u32, u32)>,
            ptr: &mut T,
            _gas: &mut u64,
            class_hash: &Felt252Abi,
            function_selector: &Felt252Abi,
            calldata: *const (*const Felt252Abi, u32, u32),
        ) {
            // TODO: handle gas
            let class_hash = Felt252::from_bytes_be(&class_hash.0);
            let function_selector = Felt252::from_bytes_be(&function_selector.0);

            let calldata: Vec<_> = unsafe {
                let len = (*calldata).1 as usize;

                std::slice::from_raw_parts((*calldata).0, len)
            }
            .iter()
            .map(|x| Felt252::from_bytes_be(&x.0))
            .collect();

            let result = ptr.library_call(class_hash, function_selector, &calldata);

            *result_ptr = match result {
                Ok(x) => {
                    let felts: Vec<_> = x.iter().map(|x| Felt252Abi(x.to_le_bytes())).collect();
                    let felts_ptr = unsafe { Self::alloc_mlir_array(&felts) };
                    SyscallResultAbi {
                        tag: 0u8,
                        payload: SyscallResultPayloadAbi {
                            ok: ManuallyDrop::new(felts_ptr),
                        },
                    }
                }
                Err(e) => Self::wrap_error(&e),
            };
        }

        extern "C" fn wrap_call_contract(
            result_ptr: &mut SyscallResultAbi<(NonNull<Felt252Abi>, u32, u32)>,
            ptr: &mut T,
            _gas: &mut u64,
            address: &Felt252Abi,
            entry_point_selector: &Felt252Abi,
            calldata: *const (*const Felt252Abi, u32, u32),
        ) {
            // TODO: handle gas
            let address = Felt252::from_bytes_be(&address.0);
            let entry_point_selector = Felt252::from_bytes_be(&entry_point_selector.0);

            let calldata: Vec<_> = unsafe {
                let len = (*calldata).1 as usize;

                std::slice::from_raw_parts((*calldata).0, len)
            }
            .iter()
            .map(|x| Felt252::from_bytes_be(&x.0))
            .collect();

            let result = ptr.call_contract(address, entry_point_selector, &calldata);

            *result_ptr = match result {
                Ok(x) => {
                    let felts: Vec<_> = x.iter().map(|x| Felt252Abi(x.to_le_bytes())).collect();
                    let felts_ptr = unsafe { Self::alloc_mlir_array(&felts) };
                    SyscallResultAbi {
                        tag: 0u8,
                        payload: SyscallResultPayloadAbi {
                            ok: ManuallyDrop::new(felts_ptr),
                        },
                    }
                }
                Err(e) => Self::wrap_error(&e),
            };
        }

        extern "C" fn wrap_storage_read(
            result_ptr: &mut SyscallResultAbi<Felt252Abi>,
            ptr: &mut T,
            _gas: &mut u64,
            address_domain: u32,
            address: &Felt252Abi,
        ) {
            // TODO: Handle gas.
            let address = Felt252::from_bytes_be(&address.0);
            let result = ptr.storage_read(address_domain, address);

            *result_ptr = match result {
                Ok(res) => SyscallResultAbi {
                    tag: 0u8,
                    payload: SyscallResultPayloadAbi {
                        ok: ManuallyDrop::new(Felt252Abi(res.to_le_bytes())),
                    },
                },
                Err(e) => Self::wrap_error(&e),
            };
        }

        extern "C" fn wrap_storage_write(
            result_ptr: &mut SyscallResultAbi<()>,
            ptr: &mut T,
            _gas: &mut u64,
            address_domain: u32,
            address: &Felt252Abi,
            value: &Felt252Abi,
        ) {
            // TODO: Handle gas.
            let address = Felt252::from_bytes_be(&address.0);
            let value = Felt252::from_bytes_be(&value.0);
            let result = ptr.storage_write(address_domain, address, value);

            *result_ptr = match result {
                Ok(res) => SyscallResultAbi {
                    tag: 0u8,
                    payload: SyscallResultPayloadAbi {
                        ok: ManuallyDrop::new(()),
                    },
                },
                Err(e) => Self::wrap_error(&e),
            };
        }
    }
}
