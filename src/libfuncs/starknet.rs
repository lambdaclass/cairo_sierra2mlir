//! # Starknet libfuncs

use super::LibfuncHelper;
use crate::{
    error::{Error, Result},
    ffi::get_struct_field_type_at,
    metadata::{drop_overrides::DropOverridesMeta, MetadataStorage},
    starknet::handler::StarknetSyscallHandlerCallbacks,
    utils::{get_integer_layout, BlockExt, GepIndex, ProgramRegistryExt, PRIME},
};
use cairo_lang_sierra::{
    extensions::{
        consts::SignatureAndConstConcreteLibfunc,
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
        starknet::{testing::TestingConcreteLibfunc, StarkNetConcreteLibfunc},
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        llvm::{self, r#type::pointer, LoadStoreOptions},
    },
    ir::{
        attribute::DenseI64ArrayAttribute, operation::OperationBuilder, r#type::IntegerType,
        Attribute, Block, Location, Type, ValueLike,
    },
    Context,
};
use num_bigint::Sign;
use std::alloc::Layout;

mod secp256;
mod testing;

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &StarkNetConcreteLibfunc,
) -> Result<()> {
    match selector {
        StarkNetConcreteLibfunc::ClassHashToFelt252(info)
        | StarkNetConcreteLibfunc::ContractAddressToFelt252(info)
        | StarkNetConcreteLibfunc::StorageAddressFromBase(info)
        | StarkNetConcreteLibfunc::StorageAddressToFelt252(info)
        | StarkNetConcreteLibfunc::Sha256StateHandleInit(info)
        | StarkNetConcreteLibfunc::Sha256StateHandleDigest(info) => super::build_noop::<1, true>(
            context,
            registry,
            entry,
            location,
            helper,
            metadata,
            &info.signature.param_signatures,
        ),
        StarkNetConcreteLibfunc::CallContract(info) => {
            build_call_contract(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::ClassHashConst(info) => {
            build_class_hash_const(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::ClassHashTryFromFelt252(info) => {
            build_class_hash_try_from_felt252(
                context, registry, entry, location, helper, metadata, info,
            )
        }
        StarkNetConcreteLibfunc::ContractAddressConst(info) => {
            build_contract_address_const(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::ContractAddressTryFromFelt252(info) => {
            build_contract_address_try_from_felt252(
                context, registry, entry, location, helper, metadata, info,
            )
        }
        StarkNetConcreteLibfunc::StorageRead(info) => {
            build_storage_read(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::StorageWrite(info) => {
            build_storage_write(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::StorageBaseAddressConst(info) => build_storage_base_address_const(
            context, registry, entry, location, helper, metadata, info,
        ),
        StarkNetConcreteLibfunc::StorageBaseAddressFromFelt252(info) => {
            build_storage_base_address_from_felt252(
                context, registry, entry, location, helper, metadata, info,
            )
        }
        StarkNetConcreteLibfunc::StorageAddressFromBaseAndOffset(info) => {
            build_storage_address_from_base_and_offset(
                context, registry, entry, location, helper, metadata, info,
            )
        }
        StarkNetConcreteLibfunc::StorageAddressTryFromFelt252(info) => {
            build_storage_address_try_from_felt252(
                context, registry, entry, location, helper, metadata, info,
            )
        }
        StarkNetConcreteLibfunc::EmitEvent(info) => {
            build_emit_event(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::GetBlockHash(info) => {
            build_get_block_hash(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::GetClassHashAt(info) => {
            build_get_class_hash_at(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::GetExecutionInfo(info) => {
            build_get_execution_info(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::GetExecutionInfoV2(info) => {
            build_get_execution_info_v2(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::Deploy(info) => {
            build_deploy(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::Keccak(info) => {
            build_keccak(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::LibraryCall(info) => {
            build_library_call(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::ReplaceClass(info) => {
            build_replace_class(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::SendMessageToL1(info) => {
            build_send_message_to_l1(context, registry, entry, location, helper, metadata, info)
        }
        StarkNetConcreteLibfunc::Secp256(selector) => self::secp256::build(
            context, registry, entry, location, helper, metadata, selector,
        ),
        StarkNetConcreteLibfunc::Sha256ProcessBlock(info) => build_sha256_process_block_syscall(
            context, registry, entry, location, helper, metadata, info,
        ),
        #[cfg(feature = "with-cheatcode")]
        StarkNetConcreteLibfunc::Testing(TestingConcreteLibfunc::Cheatcode(info)) => {
            self::testing::build(context, registry, entry, location, helper, metadata, info)
        }
        #[cfg(not(feature = "with-cheatcode"))]
        StarkNetConcreteLibfunc::Testing(TestingConcreteLibfunc::Cheatcode(_)) => {
            unimplemented!("feature 'with-cheatcode' is required to compile with cheatcode syscall")
        }
    }
}

pub fn build_call_contract<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry.load(
        context,
        location,
        entry.arg(1)?,
        llvm::r#type::pointer(context, 0),
    )?;

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let result_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
        result_layout.align(),
    )?;

    // Allocate space and write the current gas.
    let (gas_ty, gas_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.param_signatures()[0].ty,
    )?;
    let gas_builtin_ptr =
        helper
            .init_block()
            .alloca1(context, location, gas_ty, gas_layout.align())?;
    entry.append_operation(llvm::store(
        context,
        entry.arg(0)?,
        gas_builtin_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `address` argument and write the value.
    let address_arg_ptr = helper.init_block().alloca_int(context, location, 252)?;
    entry.append_operation(llvm::store(
        context,
        entry.arg(2)?,
        address_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `entry_point_selector` argument and write the value.
    let entry_point_selector_arg_ptr = helper.init_block().alloca_int(context, location, 252)?;
    entry.append_operation(llvm::store(
        context,
        entry.arg(3)?,
        entry_point_selector_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `calldata` argument and write the value.
    let calldata_arg_ty = llvm::r#type::r#struct(
        context,
        &[llvm::r#type::r#struct(
            context,
            &[
                llvm::r#type::pointer(context, 0), // ptr to felt
                IntegerType::new(context, 32).into(),
                IntegerType::new(context, 32).into(),
                IntegerType::new(context, 32).into(),
            ],
            false,
        )],
        false,
    );
    let calldata_arg_ptr = helper.init_block().alloca1(
        context,
        location,
        calldata_arg_ty,
        get_integer_layout(64).align(),
    )?;
    entry.store(context, location, calldata_arg_ptr, entry.arg(4)?)?;

    // Extract function pointer.
    let fn_ptr = entry.gep(
        context,
        location,
        entry.arg(1)?,
        &[GepIndex::Const(
            StarknetSyscallHandlerCallbacks::<()>::CALL_CONTRACT.try_into()?,
        )],
        pointer(context, 0),
    )?;
    let fn_ptr = entry.load(context, location, fn_ptr, llvm::r#type::pointer(context, 0))?;

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[
                fn_ptr,
                result_ptr,
                ptr,
                gas_builtin_ptr,
                address_arg_ptr,
                entry_point_selector_arg_ptr,
                calldata_arg_ptr,
            ])
            .build()?,
    );

    let result = entry.load(
        context,
        location,
        result_ptr,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
    )?;
    let result_tag = entry.extract_value(
        context,
        location,
        result,
        IntegerType::new(context, 1).into(),
        0,
    )?;

    let payload_ok = {
        let ptr = entry.gep(
            context,
            location,
            result_ptr,
            &[GepIndex::Const(
                result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?,
            )],
            IntegerType::new(context, 8).into(),
        )?;
        entry.load(context, location, ptr, variant_tys[0].0)?
    };
    let payload_err = {
        let ptr = entry.gep(
            context,
            location,
            result_ptr,
            &[GepIndex::Const(
                result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?,
            )],
            IntegerType::new(context, 8).into(),
        )?;
        entry.load(context, location, ptr, variant_tys[1].0)?
    };

    let remaining_gas = entry.load(
        context,
        location,
        gas_builtin_ptr,
        IntegerType::new(context, 64).into(),
    )?;

    entry.append_operation(helper.cond_br(
        context,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.arg(1)?, payload_err],
            &[remaining_gas, entry.arg(1)?, payload_ok],
        ],
        location,
    ));
    Ok(())
}

pub fn build_class_hash_const<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureAndConstConcreteLibfunc,
) -> Result<()> {
    let value = entry.const_int(
        context,
        location,
        match info.c.sign() {
            Sign::Minus => &*PRIME - info.c.magnitude(),
            _ => info.c.magnitude().clone(),
        },
        252,
    )?;

    entry.append_operation(helper.br(0, &[value], location));
    Ok(())
}

pub fn build_class_hash_try_from_felt252<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let value = entry.arg(1)?;

    let limit = entry.append_op_result(arith::constant(
        context,
        Attribute::parse(
            context,
            "3618502788666131106986593281521497120414687020801267626233049500247285301248 : i252",
        )
        .ok_or(Error::ParseAttributeError)?,
        location,
    ))?;
    let is_in_range = entry.cmpi(context, CmpiPredicate::Ult, value, limit, location)?;

    entry.append_operation(helper.cond_br(
        context,
        is_in_range,
        [0, 1],
        [&[range_check, value], &[range_check]],
        location,
    ));
    Ok(())
}

pub fn build_contract_address_const<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureAndConstConcreteLibfunc,
) -> Result<()> {
    let value = entry.const_int(
        context,
        location,
        match info.c.sign() {
            Sign::Minus => &*PRIME - info.c.magnitude(),
            _ => info.c.magnitude().clone(),
        },
        252,
    )?;

    entry.append_operation(helper.br(0, &[value], location));
    Ok(())
}

pub fn build_contract_address_try_from_felt252<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let value = entry.arg(1)?;

    let limit = entry.append_op_result(arith::constant(
        context,
        Attribute::parse(
            context,
            "3618502788666131106986593281521497120414687020801267626233049500247285301248 : i252",
        )
        .ok_or(Error::ParseAttributeError)?,
        location,
    ))?;
    let is_in_range = entry.cmpi(context, CmpiPredicate::Ult, value, limit, location)?;

    entry.append_operation(helper.cond_br(
        context,
        is_in_range,
        [0, 1],
        [&[range_check, value], &[range_check]],
        location,
    ));
    Ok(())
}

pub fn build_storage_read<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry.load(
        context,
        location,
        entry.arg(1)?,
        llvm::r#type::pointer(context, 0),
    )?;

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let result_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
        result_layout.align(),
    )?;

    // Allocate space and write the current gas.
    let (gas_ty, gas_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.param_signatures()[0].ty,
    )?;
    let gas_builtin_ptr =
        helper
            .init_block()
            .alloca1(context, location, gas_ty, gas_layout.align())?;
    entry.store(context, location, gas_builtin_ptr, entry.arg(0)?)?;

    // Allocate `address` argument and write the value.
    let address_arg_ptr = helper.init_block().alloca_int(context, location, 252)?;
    entry.store(context, location, address_arg_ptr, entry.arg(3)?)?;

    // Extract function pointer.
    let fn_ptr = entry.gep(
        context,
        location,
        entry.arg(1)?,
        &[GepIndex::Const(
            StarknetSyscallHandlerCallbacks::<()>::STORAGE_READ.try_into()?,
        )],
        pointer(context, 0),
    )?;
    let fn_ptr = entry.load(context, location, fn_ptr, llvm::r#type::pointer(context, 0))?;

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[
                fn_ptr,
                result_ptr,
                ptr,
                gas_builtin_ptr,
                entry.arg(2)?,
                address_arg_ptr,
            ])
            .build()?,
    );

    let result = entry.load(
        context,
        location,
        result_ptr,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
    )?;
    let result_tag = entry.extract_value(
        context,
        location,
        result,
        IntegerType::new(context, 1).into(),
        0,
    )?;

    let payload_ok = {
        let ptr = entry.gep(
            context,
            location,
            result_ptr,
            &[GepIndex::Const(
                result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?,
            )],
            IntegerType::new(context, 8).into(),
        )?;
        entry.load(context, location, ptr, variant_tys[0].0)?
    };
    let payload_err = {
        let ptr = entry.gep(
            context,
            location,
            result_ptr,
            &[GepIndex::Const(
                result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?,
            )],
            IntegerType::new(context, 8).into(),
        )?;
        entry.load(context, location, ptr, variant_tys[1].0)?
    };

    let remaining_gas = entry.load(context, location, gas_builtin_ptr, gas_ty)?;

    entry.append_operation(helper.cond_br(
        context,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.arg(1)?, payload_err],
            &[remaining_gas, entry.arg(1)?, payload_ok],
        ],
        location,
    ));
    Ok(())
}

pub fn build_storage_write<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry.load(
        context,
        location,
        entry.arg(1)?,
        llvm::r#type::pointer(context, 0),
    )?;

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                // The branch is deliberately duplicated because:
                //   - There is no `[0].vars[2]` (it returns `()`).
                //   - We need a variant to make the length be 2.
                //   - It requires a `ConcreteTypeId`, we can't pass an MLIR type.
                info.branch_signatures()[1].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let result_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
        result_layout.align(),
    )?;

    // Allocate space and write the current gas.
    let (gas_ty, gas_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.param_signatures()[0].ty,
    )?;
    let gas_builtin_ptr =
        helper
            .init_block()
            .alloca1(context, location, gas_ty, gas_layout.align())?;
    entry.store(context, location, gas_builtin_ptr, entry.arg(0)?)?;

    // Allocate `address` argument and write the value.
    let address_arg_ptr = helper.init_block().alloca_int(context, location, 252)?;
    entry.store(context, location, address_arg_ptr, entry.arg(3)?)?;

    // Allocate `value` argument and write the value.
    let value_arg_ptr = helper.init_block().alloca_int(context, location, 252)?;
    entry.store(context, location, value_arg_ptr, entry.arg(4)?)?;

    let fn_ptr = entry.gep(
        context,
        location,
        entry.arg(1)?,
        &[GepIndex::Const(
            StarknetSyscallHandlerCallbacks::<()>::STORAGE_WRITE.try_into()?,
        )],
        pointer(context, 0),
    )?;
    let fn_ptr = entry.load(context, location, fn_ptr, llvm::r#type::pointer(context, 0))?;

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[
                fn_ptr,
                result_ptr,
                ptr,
                gas_builtin_ptr,
                entry.arg(2)?,
                address_arg_ptr,
                value_arg_ptr,
            ])
            .build()?,
    );

    let result = entry.load(
        context,
        location,
        result_ptr,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
    )?;
    let result_tag = entry.extract_value(
        context,
        location,
        result,
        IntegerType::new(context, 1).into(),
        0,
    )?;

    let payload_ok = {
        let ptr = entry.gep(
            context,
            location,
            result_ptr,
            &[GepIndex::Const(
                result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?,
            )],
            IntegerType::new(context, 8).into(),
        )?;
        entry.load(context, location, ptr, variant_tys[0].0)?
    };
    let payload_err = {
        let ptr = entry.gep(
            context,
            location,
            result_ptr,
            &[GepIndex::Const(
                result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?,
            )],
            IntegerType::new(context, 8).into(),
        )?;
        entry.load(context, location, ptr, variant_tys[1].0)?
    };

    let remaining_gas = entry.load(context, location, gas_builtin_ptr, gas_ty)?;

    entry.append_operation(helper.cond_br(
        context,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.arg(1)?, payload_err],
            &[remaining_gas, entry.arg(1)?, payload_ok],
        ],
        location,
    ));
    Ok(())
}

pub fn build_storage_base_address_const<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    info: &SignatureAndConstConcreteLibfunc,
) -> Result<()> {
    let value = entry.const_int(
        context,
        location,
        match info.c.sign() {
            Sign::Minus => &*PRIME - info.c.magnitude(),
            _ => info.c.magnitude().clone(),
        },
        252,
    )?;

    entry.append_operation(helper.br(0, &[value], location));
    Ok(())
}

pub fn build_storage_base_address_from_felt252<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let k_limit = entry.append_op_result(arith::constant(
        context,
        Attribute::parse(
            context,
            "3618502788666131106986593281521497120414687020801267626233049500247285300992 : i252",
        )
        .ok_or(Error::ParseAttributeError)?,
        location,
    ))?;

    let limited_value = entry.append_op_result(arith::subi(entry.arg(1)?, k_limit, location))?;

    let is_within_limit = entry.cmpi(
        context,
        CmpiPredicate::Ult,
        entry.arg(1)?,
        k_limit,
        location,
    )?;
    let value = entry.append_op_result(arith::select(
        is_within_limit,
        entry.arg(1)?,
        limited_value,
        location,
    ))?;

    entry.append_operation(helper.br(0, &[range_check, value], location));
    Ok(())
}

pub fn build_storage_address_from_base_and_offset<'ctx, 'this>(
    _context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let offset = entry.extui(entry.arg(1)?, entry.argument(0)?.r#type(), location)?;
    let addr = entry.addi(entry.arg(0)?, offset, location)?;

    entry.append_operation(helper.br(0, &[addr], location));
    Ok(())
}

pub fn build_storage_address_try_from_felt252<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let value = entry.arg(1)?;

    let limit = entry.append_op_result(arith::constant(
        context,
        Attribute::parse(
            context,
            "3618502788666131106986593281521497120414687020801267626233049500247285301248 : i252",
        )
        .ok_or(Error::ParseAttributeError)?,
        location,
    ))?;
    let is_in_range = entry.cmpi(context, CmpiPredicate::Ult, value, limit, location)?;

    entry.append_operation(helper.cond_br(
        context,
        is_in_range,
        [0, 1],
        [&[range_check, value], &[range_check]],
        location,
    ));
    Ok(())
}

pub fn build_emit_event<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry.load(
        context,
        location,
        entry.arg(1)?,
        llvm::r#type::pointer(context, 0),
    )?;

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                // The branch is deliberately duplicated because:
                //   - There is no `[0].vars[2]` (it returns `()`).
                //   - We need a variant to make the length be 2.
                //   - It requires a `ConcreteTypeId`, we can't pass an MLIR type.
                info.branch_signatures()[1].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let result_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
        result_layout.align(),
    )?;

    // Allocate space and write the current gas.
    let (gas_ty, gas_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.param_signatures()[0].ty,
    )?;
    let gas_builtin_ptr =
        helper
            .init_block()
            .alloca1(context, location, gas_ty, gas_layout.align())?;
    entry.append_operation(llvm::store(
        context,
        entry.arg(0)?,
        gas_builtin_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `keys` argument and write the value.
    let keys_arg_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[llvm::r#type::r#struct(
                context,
                &[
                    llvm::r#type::pointer(context, 0), // ptr to felt
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                ],
                false,
            )],
            false,
        ),
        get_integer_layout(64).align(),
    )?;
    entry.store(context, location, keys_arg_ptr, entry.arg(2)?)?;

    // Allocate `data` argument and write the value.
    let data_arg_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[llvm::r#type::r#struct(
                context,
                &[
                    llvm::r#type::pointer(context, 0), // ptr to felt
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                ],
                false,
            )],
            false,
        ),
        get_integer_layout(64).align(),
    )?;
    entry.store(context, location, data_arg_ptr, entry.arg(3)?)?;

    let fn_ptr = entry.gep(
        context,
        location,
        entry.arg(1)?,
        &[GepIndex::Const(
            StarknetSyscallHandlerCallbacks::<()>::EMIT_EVENT.try_into()?,
        )],
        pointer(context, 0),
    )?;
    let fn_ptr = entry.load(context, location, fn_ptr, llvm::r#type::pointer(context, 0))?;

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[
                fn_ptr,
                result_ptr,
                ptr,
                gas_builtin_ptr,
                keys_arg_ptr,
                data_arg_ptr,
            ])
            .build()?,
    );

    let result = entry.load(
        context,
        location,
        result_ptr,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
    )?;
    let result_tag = entry.extract_value(
        context,
        location,
        result,
        IntegerType::new(context, 1).into(),
        0,
    )?;

    let payload_ok = {
        let ptr = entry.gep(
            context,
            location,
            result_ptr,
            &[GepIndex::Const(
                result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?,
            )],
            IntegerType::new(context, 8).into(),
        )?;
        entry.load(context, location, ptr, variant_tys[0].0)?
    };
    let payload_err = {
        let ptr = entry.gep(
            context,
            location,
            result_ptr,
            &[GepIndex::Const(
                result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?,
            )],
            IntegerType::new(context, 8).into(),
        )?;
        entry.load(context, location, ptr, variant_tys[1].0)?
    };

    let remaining_gas = entry.load(context, location, gas_builtin_ptr, gas_ty)?;

    entry.append_operation(helper.cond_br(
        context,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.arg(1)?, payload_err],
            &[remaining_gas, entry.arg(1)?, payload_ok],
        ],
        location,
    ));
    Ok(())
}

pub fn build_get_block_hash<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry.load(
        context,
        location,
        entry.arg(1)?,
        llvm::r#type::pointer(context, 0),
    )?;

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let result_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
        result_layout.align(),
    )?;

    // Allocate space and write the current gas.
    let (gas_ty, gas_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.param_signatures()[0].ty,
    )?;
    let gas_builtin_ptr =
        helper
            .init_block()
            .alloca1(context, location, gas_ty, gas_layout.align())?;
    entry.append_operation(llvm::store(
        context,
        entry.arg(0)?,
        gas_builtin_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Extract function pointer.
    let fn_ptr = entry.gep(
        context,
        location,
        entry.arg(1)?,
        &[GepIndex::Const(
            StarknetSyscallHandlerCallbacks::<()>::GET_BLOCK_HASH.try_into()?,
        )],
        pointer(context, 0),
    )?;
    let fn_ptr = entry.load(context, location, fn_ptr, llvm::r#type::pointer(context, 0))?;

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[fn_ptr, result_ptr, ptr, gas_builtin_ptr, entry.arg(2)?])
            .build()?,
    );

    let result = entry.load(
        context,
        location,
        result_ptr,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
    )?;
    let result_tag = entry.extract_value(
        context,
        location,
        result,
        IntegerType::new(context, 1).into(),
        0,
    )?;

    let payload_ok = {
        let ptr = entry.gep(
            context,
            location,
            result_ptr,
            &[GepIndex::Const(
                result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?,
            )],
            IntegerType::new(context, 8).into(),
        )?;
        entry.load(context, location, ptr, variant_tys[0].0)?
    };
    let payload_err = {
        let ptr = entry.gep(
            context,
            location,
            result_ptr,
            &[GepIndex::Const(
                result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?,
            )],
            IntegerType::new(context, 8).into(),
        )?;
        entry.load(context, location, ptr, variant_tys[1].0)?
    };

    let remaining_gas = entry.load(context, location, gas_builtin_ptr, gas_ty)?;

    entry.append_operation(helper.cond_br(
        context,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.arg(1)?, payload_err],
            &[remaining_gas, entry.arg(1)?, payload_ok],
        ],
        location,
    ));
    Ok(())
}

pub fn build_get_execution_info<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry.load(
        context,
        location,
        entry.arg(1)?,
        llvm::r#type::pointer(context, 0),
    )?;

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let result_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
        result_layout.align(),
    )?;

    // Allocate space and write the current gas.
    let (gas_ty, gas_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.param_signatures()[0].ty,
    )?;
    let gas_builtin_ptr =
        helper
            .init_block()
            .alloca1(context, location, gas_ty, gas_layout.align())?;
    entry.store(context, location, gas_builtin_ptr, entry.arg(0)?)?;

    // Extract function pointer.
    let fn_ptr = entry.gep(
        context,
        location,
        entry.arg(1)?,
        &[GepIndex::Const(
            StarknetSyscallHandlerCallbacks::<()>::GET_EXECUTION_INFO.try_into()?,
        )],
        pointer(context, 0),
    )?;
    let fn_ptr = entry.load(context, location, fn_ptr, llvm::r#type::pointer(context, 0))?;

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[fn_ptr, result_ptr, ptr, gas_builtin_ptr])
            .build()?,
    );

    let result = entry.load(
        context,
        location,
        result_ptr,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
    )?;
    let result_tag = entry.extract_value(
        context,
        location,
        result,
        IntegerType::new(context, 1).into(),
        0,
    )?;

    let payload_ok = {
        let ptr = entry.gep(
            context,
            location,
            result_ptr,
            &[GepIndex::Const(
                result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?,
            )],
            IntegerType::new(context, 8).into(),
        )?;
        entry.load(context, location, ptr, variant_tys[0].0)?
    };
    let payload_err = {
        let ptr = entry.gep(
            context,
            location,
            result_ptr,
            &[GepIndex::Const(
                result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?,
            )],
            IntegerType::new(context, 8).into(),
        )?;
        entry.load(context, location, ptr, variant_tys[1].0)?
    };

    let remaining_gas = entry.load(context, location, gas_builtin_ptr, gas_ty)?;

    entry.append_operation(helper.cond_br(
        context,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.arg(1)?, payload_err],
            &[remaining_gas, entry.arg(1)?, payload_ok],
        ],
        location,
    ));
    Ok(())
}

pub fn build_get_execution_info_v2<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry.load(
        context,
        location,
        entry.arg(1)?,
        llvm::r#type::pointer(context, 0),
    )?;

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let result_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
        result_layout.align(),
    )?;

    // Allocate space and write the current gas.
    let (gas_ty, gas_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.param_signatures()[0].ty,
    )?;
    let gas_builtin_ptr =
        helper
            .init_block()
            .alloca1(context, location, gas_ty, gas_layout.align())?;
    entry.store(context, location, gas_builtin_ptr, entry.arg(0)?)?;

    // Extract function pointer.
    let fn_ptr = entry.gep(
        context,
        location,
        entry.arg(1)?,
        &[GepIndex::Const(
            StarknetSyscallHandlerCallbacks::<()>::GET_EXECUTION_INFOV2.try_into()?,
        )],
        pointer(context, 0),
    )?;
    let fn_ptr = entry.load(context, location, fn_ptr, llvm::r#type::pointer(context, 0))?;

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[fn_ptr, result_ptr, ptr, gas_builtin_ptr])
            .build()?,
    );

    let result = entry.load(
        context,
        location,
        result_ptr,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
    )?;
    let result_tag = entry.extract_value(
        context,
        location,
        result,
        IntegerType::new(context, 1).into(),
        0,
    )?;

    let payload_ok = {
        let ptr = entry.gep(
            context,
            location,
            result_ptr,
            &[GepIndex::Const(
                result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?,
            )],
            IntegerType::new(context, 8).into(),
        )?;
        entry.load(context, location, ptr, variant_tys[0].0)?
    };
    let payload_err = {
        let ptr = entry.gep(
            context,
            location,
            result_ptr,
            &[GepIndex::Const(
                result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?,
            )],
            IntegerType::new(context, 8).into(),
        )?;
        entry.load(context, location, ptr, variant_tys[1].0)?
    };

    let remaining_gas = entry.load(context, location, gas_builtin_ptr, gas_ty)?;

    entry.append_operation(helper.cond_br(
        context,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.arg(1)?, payload_err],
            &[remaining_gas, entry.arg(1)?, payload_ok],
        ],
        location,
    ));
    Ok(())
}

pub fn build_deploy<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry.load(
        context,
        location,
        entry.arg(1)?,
        llvm::r#type::pointer(context, 0),
    )?;

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) = {
        let tag_layout = get_integer_layout(1);
        let tag_ty: Type = IntegerType::new(context, 1).into();

        let mut layout = tag_layout;
        let output = [
            {
                let (p0_ty, p0_layout) = registry.build_type_with_layout(
                    context,
                    helper,
                    metadata,
                    &info.branch_signatures()[0].vars[2].ty,
                )?;
                let (p1_ty, p1_layout) = registry.build_type_with_layout(
                    context,
                    helper,
                    metadata,
                    &info.branch_signatures()[0].vars[3].ty,
                )?;

                let payload_ty = llvm::r#type::r#struct(context, &[p0_ty, p1_ty], false);
                let payload_layout = p0_layout.extend(p1_layout)?.0;

                let full_layout = tag_layout.extend(payload_layout)?.0;
                layout = Layout::from_size_align(
                    layout.size().max(full_layout.size()),
                    layout.align().max(full_layout.align()),
                )?;

                (payload_ty, payload_layout)
            },
            {
                let (payload_ty, payload_layout) = registry.build_type_with_layout(
                    context,
                    helper,
                    metadata,
                    &info.branch_signatures()[1].vars[2].ty,
                )?;

                let full_layout = tag_layout.extend(payload_layout)?.0;
                layout = Layout::from_size_align(
                    layout.size().max(full_layout.size()),
                    layout.align().max(full_layout.align()),
                )?;

                (payload_ty, payload_layout)
            },
        ];

        (layout, (tag_ty, tag_layout), output)
    };

    let result_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
        result_layout.align(),
    )?;

    // Allocate space and write the current gas.
    let (gas_ty, gas_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.param_signatures()[0].ty,
    )?;
    let gas_builtin_ptr =
        helper
            .init_block()
            .alloca1(context, location, gas_ty, gas_layout.align())?;
    entry.store(context, location, gas_builtin_ptr, entry.arg(0)?)?;

    // Allocate `class_hash` argument and write the value.
    let class_hash_arg_ptr = helper.init_block().alloca_int(context, location, 252)?;
    entry.store(context, location, class_hash_arg_ptr, entry.arg(2)?)?;

    // Allocate `entry_point_selector` argument and write the value.
    let contract_address_salt_arg_ptr = helper.init_block().alloca_int(context, location, 252)?;
    entry.store(
        context,
        location,
        contract_address_salt_arg_ptr,
        entry.arg(3)?,
    )?;

    // Allocate `calldata` argument and write the value.
    let calldata_arg_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[llvm::r#type::r#struct(
                context,
                &[
                    llvm::r#type::pointer(context, 0), // ptr to felt
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                ],
                false,
            )],
            false,
        ),
        get_integer_layout(64).align(),
    )?;
    entry.store(context, location, calldata_arg_ptr, entry.arg(4)?)?;

    let fn_ptr = entry.gep(
        context,
        location,
        entry.arg(1)?,
        &[GepIndex::Const(
            StarknetSyscallHandlerCallbacks::<()>::DEPLOY.try_into()?,
        )],
        pointer(context, 0),
    )?;
    let fn_ptr = entry.load(context, location, fn_ptr, llvm::r#type::pointer(context, 0))?;

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[
                fn_ptr,
                result_ptr,
                ptr,
                gas_builtin_ptr,
                class_hash_arg_ptr,
                contract_address_salt_arg_ptr,
                calldata_arg_ptr,
                entry
                    .append_operation(llvm::extract_value(
                        context,
                        entry.arg(5)?,
                        DenseI64ArrayAttribute::new(context, &[0]),
                        IntegerType::new(context, 1).into(),
                        location,
                    ))
                    .result(0)?
                    .into(),
            ])
            .build()?,
    );

    let result = entry.load(
        context,
        location,
        result_ptr,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
    )?;
    let result_tag = entry.extract_value(
        context,
        location,
        result,
        IntegerType::new(context, 1).into(),
        0,
    )?;

    let payload_ok = {
        let ptr = entry.gep(
            context,
            location,
            result_ptr,
            &[GepIndex::Const(
                result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?,
            )],
            IntegerType::new(context, 8).into(),
        )?;
        entry.load(context, location, ptr, variant_tys[0].0)?
    };
    let payload_err = {
        let ptr = entry.gep(
            context,
            location,
            result_ptr,
            &[GepIndex::Const(
                result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?,
            )],
            IntegerType::new(context, 8).into(),
        )?;
        entry.load(context, location, ptr, variant_tys[1].0)?
    };

    let remaining_gas = entry.load(context, location, gas_builtin_ptr, gas_ty)?;

    entry.append_operation(helper.cond_br(
        context,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.arg(1)?, payload_err],
            &[
                remaining_gas,
                entry.arg(1)?,
                entry.extract_value(
                    context,
                    location,
                    payload_ok,
                    get_struct_field_type_at(&variant_tys[0].0, 0),
                    0,
                )?,
                entry.extract_value(
                    context,
                    location,
                    payload_ok,
                    get_struct_field_type_at(&variant_tys[0].0, 1),
                    1,
                )?,
            ],
        ],
        location,
    ));
    Ok(())
}

pub fn build_keccak<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry.load(
        context,
        location,
        entry.arg(1)?,
        llvm::r#type::pointer(context, 0),
    )?;

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let result_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
        result_layout.align(),
    )?;

    // Allocate space and write the current gas.
    let (gas_ty, gas_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.param_signatures()[0].ty,
    )?;
    let gas_builtin_ptr =
        helper
            .init_block()
            .alloca1(context, location, gas_ty, gas_layout.align())?;
    entry.store(context, location, gas_builtin_ptr, entry.arg(0)?)?;

    // Allocate `input` argument and write the value.
    let input_arg_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[
                llvm::r#type::pointer(context, 0), // ptr to u64
                IntegerType::new(context, 32).into(),
                IntegerType::new(context, 32).into(),
                IntegerType::new(context, 32).into(),
            ],
            false,
        ),
        get_integer_layout(64).align(),
    )?;
    entry.store(context, location, input_arg_ptr, entry.arg(2)?)?;

    let fn_ptr = entry.gep(
        context,
        location,
        entry.arg(1)?,
        &[GepIndex::Const(
            StarknetSyscallHandlerCallbacks::<()>::KECCAK.try_into()?,
        )],
        pointer(context, 0),
    )?;
    let fn_ptr = entry.load(context, location, fn_ptr, llvm::r#type::pointer(context, 0))?;

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[fn_ptr, result_ptr, ptr, gas_builtin_ptr, input_arg_ptr])
            .build()?,
    );

    let result = entry.load(
        context,
        location,
        result_ptr,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
    )?;
    let result_tag = entry.extract_value(
        context,
        location,
        result,
        IntegerType::new(context, 1).into(),
        0,
    )?;

    let payload_ok = {
        let ptr = entry.gep(
            context,
            location,
            result_ptr,
            &[GepIndex::Const(
                result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?,
            )],
            IntegerType::new(context, 8).into(),
        )?;
        entry.load(context, location, ptr, variant_tys[0].0)?
    };
    let payload_err = {
        let ptr = entry.gep(
            context,
            location,
            result_ptr,
            &[GepIndex::Const(
                result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?,
            )],
            IntegerType::new(context, 8).into(),
        )?;
        entry.load(context, location, ptr, variant_tys[1].0)?
    };
    let remaining_gas = entry.load(context, location, gas_builtin_ptr, gas_ty)?;

    entry.append_operation(helper.cond_br(
        context,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.arg(1)?, payload_err],
            &[remaining_gas, entry.arg(1)?, payload_ok],
        ],
        location,
    ));
    Ok(())
}

pub fn build_library_call<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry.load(
        context,
        location,
        entry.arg(1)?,
        llvm::r#type::pointer(context, 0),
    )?;

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let result_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
        result_layout.align(),
    )?;

    // Allocate space and write the current gas.
    let (gas_ty, gas_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.param_signatures()[0].ty,
    )?;
    let gas_builtin_ptr =
        helper
            .init_block()
            .alloca1(context, location, gas_ty, gas_layout.align())?;
    entry.store(context, location, gas_builtin_ptr, entry.arg(0)?)?;

    // Allocate `class_hash` argument and write the value.
    let class_hash_arg_ptr = helper.init_block().alloca_int(context, location, 252)?;
    entry.store(context, location, class_hash_arg_ptr, entry.arg(2)?)?;

    // Allocate `entry_point_selector` argument and write the value.
    let function_selector_arg_ptr = helper.init_block().alloca_int(context, location, 252)?;
    entry.store(context, location, function_selector_arg_ptr, entry.arg(3)?)?;

    // Allocate `calldata` argument and write the value.
    let calldata_arg_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[llvm::r#type::r#struct(
                context,
                &[
                    llvm::r#type::pointer(context, 0), // ptr to felt
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                    IntegerType::new(context, 32).into(),
                ],
                false,
            )],
            false,
        ),
        get_integer_layout(64).align(),
    )?;
    entry.store(context, location, calldata_arg_ptr, entry.arg(4)?)?;

    let fn_ptr = entry.gep(
        context,
        location,
        entry.arg(1)?,
        &[GepIndex::Const(
            StarknetSyscallHandlerCallbacks::<()>::LIBRARY_CALL.try_into()?,
        )],
        pointer(context, 0),
    )?;
    let fn_ptr = entry.load(context, location, fn_ptr, llvm::r#type::pointer(context, 0))?;

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[
                fn_ptr,
                result_ptr,
                ptr,
                gas_builtin_ptr,
                class_hash_arg_ptr,
                function_selector_arg_ptr,
                calldata_arg_ptr,
            ])
            .build()?,
    );

    let result = entry.load(
        context,
        location,
        result_ptr,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
    )?;
    let result_tag = entry.extract_value(
        context,
        location,
        result,
        IntegerType::new(context, 1).into(),
        0,
    )?;

    let payload_ok = {
        let ptr = entry.gep(
            context,
            location,
            result_ptr,
            &[GepIndex::Const(
                result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?,
            )],
            IntegerType::new(context, 8).into(),
        )?;
        entry.load(context, location, ptr, variant_tys[0].0)?
    };
    let payload_err = {
        let ptr = entry.gep(
            context,
            location,
            result_ptr,
            &[GepIndex::Const(
                result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?,
            )],
            IntegerType::new(context, 8).into(),
        )?;
        entry.load(context, location, ptr, variant_tys[1].0)?
    };

    let remaining_gas = entry.load(context, location, gas_builtin_ptr, gas_ty)?;

    entry.append_operation(helper.cond_br(
        context,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.arg(1)?, payload_err],
            &[remaining_gas, entry.arg(1)?, payload_ok],
        ],
        location,
    ));
    Ok(())
}

pub fn build_replace_class<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry.load(
        context,
        location,
        entry.arg(1)?,
        llvm::r#type::pointer(context, 0),
    )?;

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                // The branch is deliberately duplicated because:
                //   - There is no `[0].vars[2]` (it returns `()`).
                //   - We need a variant to make the length be 2.
                //   - It requires a `ConcreteTypeId`, we can't pass an MLIR type.
                info.branch_signatures()[1].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let result_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
        result_layout.align(),
    )?;

    // Allocate space and write the current gas.
    let (gas_ty, gas_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.param_signatures()[0].ty,
    )?;
    let gas_builtin_ptr =
        helper
            .init_block()
            .alloca1(context, location, gas_ty, gas_layout.align())?;
    entry.store(context, location, gas_builtin_ptr, entry.arg(0)?)?;

    // Allocate `class_hash` argument and write the value.
    let class_hash_arg_ptr = helper.init_block().alloca_int(context, location, 252)?;
    entry.store(context, location, class_hash_arg_ptr, entry.arg(2)?)?;

    let fn_ptr = entry.gep(
        context,
        location,
        entry.arg(1)?,
        &[GepIndex::Const(
            StarknetSyscallHandlerCallbacks::<()>::REPLACE_CLASS.try_into()?,
        )],
        pointer(context, 0),
    )?;
    let fn_ptr = entry.load(context, location, fn_ptr, llvm::r#type::pointer(context, 0))?;

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[fn_ptr, result_ptr, ptr, gas_builtin_ptr, class_hash_arg_ptr])
            .build()?,
    );

    let result = entry.load(
        context,
        location,
        result_ptr,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
    )?;
    let result_tag = entry.extract_value(
        context,
        location,
        result,
        IntegerType::new(context, 1).into(),
        0,
    )?;

    let payload_ok = {
        let ptr = entry.gep(
            context,
            location,
            result_ptr,
            &[GepIndex::Const(
                result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?,
            )],
            IntegerType::new(context, 8).into(),
        )?;
        entry.load(context, location, ptr, variant_tys[0].0)?
    };
    let payload_err = {
        let ptr = entry.gep(
            context,
            location,
            result_ptr,
            &[GepIndex::Const(
                result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?,
            )],
            IntegerType::new(context, 8).into(),
        )?;
        entry.load(context, location, ptr, variant_tys[1].0)?
    };

    let remaining_gas = entry.load(context, location, gas_builtin_ptr, gas_ty)?;

    entry.append_operation(helper.cond_br(
        context,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.arg(1)?, payload_err],
            &[remaining_gas, entry.arg(1)?, payload_ok],
        ],
        location,
    ));
    Ok(())
}

pub fn build_send_message_to_l1<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry.load(
        context,
        location,
        entry.arg(1)?,
        llvm::r#type::pointer(context, 0),
    )?;

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                // The branch is deliberately duplicated because:
                //   - There is no `[0].vars[2]` (it returns `()`).
                //   - We need a variant to make the length be 2.
                //   - It requires a `ConcreteTypeId`, we can't pass an MLIR type.
                info.branch_signatures()[1].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let result_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
        result_layout.align(),
    )?;

    // Allocate space and write the current gas.
    let (gas_ty, gas_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.param_signatures()[0].ty,
    )?;
    let gas_builtin_ptr =
        helper
            .init_block()
            .alloca1(context, location, gas_ty, gas_layout.align())?;
    entry.store(context, location, gas_builtin_ptr, entry.arg(0)?)?;

    // Allocate `to_address` argument and write the value.
    let to_address_arg_ptr = helper.init_block().alloca_int(context, location, 252)?;
    entry.store(context, location, to_address_arg_ptr, entry.arg(2)?)?;

    // Allocate `payload` argument and write the value.
    let payload_arg_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[
                llvm::r#type::pointer(context, 0), // ptr to felt
                IntegerType::new(context, 32).into(),
                IntegerType::new(context, 32).into(),
                IntegerType::new(context, 32).into(),
            ],
            false,
        ),
        get_integer_layout(64).align(),
    )?;
    entry.store(context, location, payload_arg_ptr, entry.arg(3)?)?;

    let fn_ptr = entry.gep(
        context,
        location,
        entry.arg(1)?,
        &[GepIndex::Const(
            StarknetSyscallHandlerCallbacks::<()>::SEND_MESSAGE_TO_L1.try_into()?,
        )],
        pointer(context, 0),
    )?;
    let fn_ptr = entry.load(context, location, fn_ptr, llvm::r#type::pointer(context, 0))?;

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[
                fn_ptr,
                result_ptr,
                ptr,
                gas_builtin_ptr,
                to_address_arg_ptr,
                payload_arg_ptr,
            ])
            .build()?,
    );

    let result = entry.load(
        context,
        location,
        result_ptr,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
    )?;
    let result_tag = entry.extract_value(
        context,
        location,
        result,
        IntegerType::new(context, 1).into(),
        0,
    )?;

    let payload_ok = {
        let ptr = entry.gep(
            context,
            location,
            result_ptr,
            &[GepIndex::Const(
                result_tag_layout.extend(variant_tys[0].1)?.1.try_into()?,
            )],
            IntegerType::new(context, 8).into(),
        )?;
        entry.load(context, location, ptr, variant_tys[0].0)?
    };
    let payload_err = {
        let ptr = entry.gep(
            context,
            location,
            result_ptr,
            &[GepIndex::Const(
                result_tag_layout.extend(variant_tys[1].1)?.1.try_into()?,
            )],
            IntegerType::new(context, 8).into(),
        )?;
        entry.load(context, location, ptr, variant_tys[1].0)?
    };

    let remaining_gas = entry.load(context, location, gas_builtin_ptr, gas_ty)?;

    entry.append_operation(helper.cond_br(
        context,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.arg(1)?, payload_err],
            &[remaining_gas, entry.arg(1)?, payload_ok],
        ],
        location,
    ));
    Ok(())
}

pub fn build_sha256_process_block_syscall<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry
        .append_operation(llvm::load(
            context,
            entry.arg(1)?,
            llvm::r#type::pointer(context, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, _), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let result_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
        result_layout.align(),
    )?;

    // Allocate space and write the current gas.
    let (gas_ty, gas_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.param_signatures()[0].ty,
    )?;
    let gas_builtin_ptr =
        helper
            .init_block()
            .alloca1(context, location, gas_ty, gas_layout.align())?;
    entry.append_operation(llvm::store(
        context,
        entry.arg(0)?,
        gas_builtin_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    let sha256_prev_state_ptr = entry.arg(2)?;
    let sha256_current_block_ptr = entry.arg(3)?;

    let fn_ptr = entry.gep(
        context,
        location,
        entry.arg(1)?,
        &[GepIndex::Const(
            StarknetSyscallHandlerCallbacks::<()>::SHA256_PROCESS_BLOCK.try_into()?,
        )],
        pointer(context, 0),
    )?;
    let fn_ptr = entry.load(context, location, fn_ptr, llvm::r#type::pointer(context, 0))?;

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[
                fn_ptr,
                result_ptr,
                ptr,
                gas_builtin_ptr,
                sha256_prev_state_ptr,
                sha256_current_block_ptr,
            ])
            .build()?,
    );

    registry.build_type(
        context,
        helper,
        metadata,
        &info.signature.param_signatures[3].ty,
    )?;
    metadata
        .get::<DropOverridesMeta>()
        .ok_or(Error::MissingMetadata)?
        .invoke_override(
            context,
            entry,
            location,
            &info.signature.param_signatures[3].ty,
            sha256_current_block_ptr,
        )?;

    let result_tag = entry.load(context, location, result_ptr, result_tag_ty)?;

    let payload_ok = {
        let value = entry.load(
            context,
            location,
            result_ptr,
            llvm::r#type::r#struct(context, &[result_tag_ty, variant_tys[0].0], false),
        )?;
        entry.extract_value(context, location, value, variant_tys[0].0, 1)?
    };
    let payload_err = {
        let value = entry.load(
            context,
            location,
            result_ptr,
            llvm::r#type::r#struct(context, &[result_tag_ty, variant_tys[1].0], false),
        )?;
        entry.extract_value(context, location, value, variant_tys[1].0, 1)?
    };

    let remaining_gas = entry.load(context, location, gas_builtin_ptr, gas_ty)?;

    entry.append_operation(helper.cond_br(
        context,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.arg(1)?, payload_err],
            &[remaining_gas, entry.arg(1)?, payload_ok],
        ],
        location,
    ));
    Ok(())
}

pub fn build_get_class_hash_at<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // Extract self pointer.
    let ptr = entry.load(
        context,
        location,
        entry.arg(1)?,
        llvm::r#type::pointer(context, 0),
    )?;

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, _), variant_tys) =
        crate::types::r#enum::get_type_for_variants(
            context,
            helper,
            registry,
            metadata,
            &[
                info.branch_signatures()[0].vars[2].ty.clone(),
                info.branch_signatures()[1].vars[2].ty.clone(),
            ],
        )?;

    let result_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
        result_layout.align(),
    )?;

    // Allocate space and write the current gas.
    let (gas_ty, gas_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.param_signatures()[0].ty,
    )?;
    let gas_builtin_ptr =
        helper
            .init_block()
            .alloca1(context, location, gas_ty, gas_layout.align())?;
    entry.append_operation(llvm::store(
        context,
        entry.arg(0)?,
        gas_builtin_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `contract_address` argument and write the value.
    let contract_address_ptr = helper.init_block().alloca_int(context, location, 252)?;
    entry.store(context, location, contract_address_ptr, entry.arg(2)?)?;

    // Extract function pointer.
    let fn_ptr = entry.gep(
        context,
        location,
        entry.arg(1)?,
        &[GepIndex::Const(
            StarknetSyscallHandlerCallbacks::<()>::GET_CLASS_HASH_AT.try_into()?,
        )],
        pointer(context, 0),
    )?;
    let fn_ptr = entry.load(context, location, fn_ptr, llvm::r#type::pointer(context, 0))?;

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[
                fn_ptr,
                result_ptr,
                ptr,
                gas_builtin_ptr,
                contract_address_ptr,
            ])
            .build()?,
    );

    let result = entry.load(
        context,
        location,
        result_ptr,
        llvm::r#type::r#struct(
            context,
            &[
                result_tag_ty,
                llvm::r#type::array(
                    IntegerType::new(context, 8).into(),
                    (result_layout.size() - 1).try_into()?,
                ),
            ],
            false,
        ),
    )?;
    let result_tag = entry.extract_value(
        context,
        location,
        result,
        IntegerType::new(context, 1).into(),
        0,
    )?;

    let payload_ok = {
        let value = entry.load(
            context,
            location,
            result_ptr,
            llvm::r#type::r#struct(context, &[result_tag_ty, variant_tys[0].0], false),
        )?;
        entry.extract_value(context, location, value, variant_tys[0].0, 1)?
    };
    let payload_err = {
        let value = entry.load(
            context,
            location,
            result_ptr,
            llvm::r#type::r#struct(context, &[result_tag_ty, variant_tys[1].0], false),
        )?;
        entry.extract_value(context, location, value, variant_tys[1].0, 1)?
    };

    let remaining_gas = entry.load(context, location, gas_builtin_ptr, gas_ty)?;

    entry.append_operation(helper.cond_br(
        context,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.arg(1)?, payload_err],
            &[remaining_gas, entry.arg(1)?, payload_ok],
        ],
        location,
    ));
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{
        utils::test::{jit_enum, jit_struct, run_sierra_program},
        Value,
    };
    use cairo_lang_sierra::{program::Program, ProgramParser};
    use lazy_static::lazy_static;
    use starknet_types_core::felt::Felt;

    lazy_static! {
        // use starknet::storage_access::{StorageBaseAddress, storage_base_address_from_felt252};
        // fn run_program(value: felt252) -> StorageBaseAddress {
        //     storage_base_address_from_felt252(value)
        // }
        static ref STORAGE_BASE_ADDRESS_FROM_FELT252: Program = ProgramParser::new().parse(r#"
            type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
            type [2] = StorageBaseAddress [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];

            libfunc [0] = storage_base_address_from_felt252;
            libfunc [2] = store_temp<[0]>;
            libfunc [3] = store_temp<[2]>;

            [0]([0], [1]) -> ([2], [3]); // 0
            [2]([2]) -> ([2]); // 1
            [3]([3]) -> ([3]); // 2
            return([2], [3]); // 3

            [0]@0([0]: [0], [1]: [1]) -> ([0], [2]);
        "#).map_err(|e| e.to_string()).unwrap();
        // use starknet::storage_access::{StorageAddress, StorageBaseAddress, storage_address_from_base};
        // fn run_program(value: StorageBaseAddress) -> StorageAddress {
        //     storage_address_from_base(value)
        // }
        static ref STORAGE_ADDRESS_FROM_BASE: Program = ProgramParser::new().parse(r#"
            type [0] = StorageBaseAddress [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = StorageAddress [storable: true, drop: true, dup: true, zero_sized: false];

            libfunc [0] = storage_address_from_base;
            libfunc [2] = store_temp<[1]>;

            [0]([0]) -> ([1]); // 0
            [2]([1]) -> ([1]); // 1
            return([1]); // 2

            [0]@0([0]: [0]) -> ([1]);
        "#).map_err(|e| e.to_string()).unwrap();
        // use starknet::storage_access::{StorageAddress, StorageBaseAddress, storage_address_from_base_and_offset};
        // fn run_program(addr: StorageBaseAddress, offset: u8) -> StorageAddress {
        //     storage_address_from_base_and_offset(addr, offset)
        // }
        static ref STORAGE_ADDRESS_FROM_BASE_AND_OFFSET: Program = ProgramParser::new().parse(r#"
            type [0] = StorageBaseAddress [storable: true, drop: true, dup: true, zero_sized: false];
            type [2] = StorageAddress [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = u8 [storable: true, drop: true, dup: true, zero_sized: false];

            libfunc [0] = storage_address_from_base_and_offset;
            libfunc [2] = store_temp<[2]>;

            [0]([0], [1]) -> ([2]); // 0
            [2]([2]) -> ([2]); // 1
            return([2]); // 2

            [0]@0([0]: [0], [1]: [1]) -> ([2]);
        "#).map_err(|e| e.to_string()).unwrap();
        // use starknet::storage_access::{StorageAddress, storage_address_to_felt252};
        // fn run_program(value: StorageAddress) -> felt252 {
        //     storage_address_to_felt252(value)
        // }
        static ref STORAGE_ADDRESS_TO_FELT252: Program = ProgramParser::new().parse(r#"
            type [0] = StorageAddress [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];

            libfunc [0] = storage_address_to_felt252;
            libfunc [2] = store_temp<[1]>;

            [0]([0]) -> ([1]); // 0
            [2]([1]) -> ([1]); // 1
            return([1]); // 2

            [0]@0([0]: [0]) -> ([1]);
        "#).map_err(|e| e.to_string()).unwrap();
        // use starknet::storage_access::{StorageAddress, storage_address_try_from_felt252};
        // fn run_program(value: felt252) -> Option<StorageAddress> {
        //     storage_address_try_from_felt252(value)
        // }
        static ref STORAGE_ADDRESS_TRY_FROM_FELT252: Program = ProgramParser::new().parse(r#"
            type [0] = RangeCheck [storable: true, drop: false, dup: false, zero_sized: false];
            type [3] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
            type [2] = StorageAddress [storable: true, drop: true, dup: true, zero_sized: false];
            type [4] = Enum<ut@core::option::Option::<core::starknet::storage_access::StorageAddress>, [2], [3]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];

            libfunc [3] = storage_address_try_from_felt252;
            libfunc [4] = branch_align;
            libfunc [2] = enum_init<[4], 0>;
            libfunc [5] = store_temp<[0]>;
            libfunc [6] = store_temp<[4]>;
            libfunc [1] = struct_construct<[3]>;
            libfunc [0] = enum_init<[4], 1>;

            [3]([0], [1]) { fallthrough([2], [3]) 6([4]) }; // 0
            [4]() -> (); // 1
            [2]([3]) -> ([5]); // 2
            [5]([2]) -> ([2]); // 3
            [6]([5]) -> ([5]); // 4
            return([2], [5]); // 5
            [4]() -> (); // 6
            [1]() -> ([6]); // 7
            [0]([6]) -> ([7]); // 8
            [5]([4]) -> ([4]); // 9
            [6]([7]) -> ([7]); // 10
            return([4], [7]); // 11

            [0]@0([0]: [0], [1]: [1]) -> ([0], [4]);
        "#).map_err(|e| e.to_string()).unwrap();
        // use starknet::class_hash::{class_hash_const, ClassHash};
        // fn run_program() -> ClassHash {
        //     class_hash_const::<0>()
        // }
        static ref CLASS_HASH_CONST: Program = ProgramParser::new().parse(r#"
            type [0] = ClassHash [storable: true, drop: true, dup: true, zero_sized: false];

            libfunc [0] = class_hash_const<0>;
            libfunc [2] = store_temp<[0]>;

            [0]() -> ([0]); // 0
            [2]([0]) -> ([0]); // 1
            return([0]); // 2

            [0]@0() -> ([0]);
        "#).map_err(|e| e.to_string()).unwrap();
    }

    #[test]
    fn class_hash_const() {
        let return_value = run_sierra_program(&CLASS_HASH_CONST, &[]).return_value;
        assert_eq!(Value::from(Felt::ZERO), return_value);
    }

    #[test]
    fn storage_base_address_from_felt252() {
        let return_value =
            run_sierra_program(&STORAGE_BASE_ADDRESS_FROM_FELT252, &[Felt::ZERO.into()])
                .return_value;

        assert_eq!(Value::from(Felt::ZERO), return_value);

        let return_value =
            run_sierra_program(&STORAGE_BASE_ADDRESS_FROM_FELT252, &[Felt::ONE.into()])
                .return_value;

        assert_eq!(Value::from(Felt::ONE), return_value);

        let return_value =
            run_sierra_program(&STORAGE_BASE_ADDRESS_FROM_FELT252, &[Felt::from(-1).into()])
                .return_value;

        assert_eq!(
            Value::from(
                Felt::from_dec_str("106710729501573572985208420194530329073740042555888586719488")
                    .unwrap()
            ),
            return_value
        );

        let return_value = run_sierra_program(
            &STORAGE_BASE_ADDRESS_FROM_FELT252,
            &[Felt::from_dec_str(
                "3618502788666131106986593281521497120414687020801267626233049500247285300992",
            )
            .unwrap()
            .into()],
        )
        .return_value;

        assert_eq!(Value::from(Felt::ZERO), return_value);
    }

    #[test]
    fn storage_address_from_base() {
        let return_value =
            run_sierra_program(&STORAGE_ADDRESS_FROM_BASE, &[Felt::ZERO.into()]).return_value;

        assert_eq!(Value::from(Felt::ZERO), return_value);

        let return_value =
            run_sierra_program(&STORAGE_ADDRESS_FROM_BASE, &[Felt::ONE.into()]).return_value;

        assert_eq!(Value::from(Felt::ONE), return_value);

        let return_value = run_sierra_program(
            &STORAGE_ADDRESS_FROM_BASE,
            &[
                Felt::from_dec_str("106710729501573572985208420194530329073740042555888586719488")
                    .unwrap()
                    .into(),
            ],
        )
        .return_value;

        assert_eq!(
            Value::from(
                Felt::from_dec_str("106710729501573572985208420194530329073740042555888586719488")
                    .unwrap()
            ),
            return_value
        );
    }

    #[test]
    fn storage_address_from_base_and_offset() {
        let return_value = run_sierra_program(
            &STORAGE_ADDRESS_FROM_BASE_AND_OFFSET,
            &[Felt::ZERO.into(), 0u8.into()],
        )
        .return_value;

        assert_eq!(Value::from(Felt::ZERO), return_value);

        let return_value = run_sierra_program(
            &STORAGE_ADDRESS_FROM_BASE_AND_OFFSET,
            &[Felt::ONE.into(), 0u8.into()],
        )
        .return_value;

        assert_eq!(Value::from(Felt::ONE), return_value);

        let return_value = run_sierra_program(
            &STORAGE_ADDRESS_FROM_BASE_AND_OFFSET,
            &[
                Felt::from_dec_str("106710729501573572985208420194530329073740042555888586719488")
                    .unwrap()
                    .into(),
                0u8.into(),
            ],
        )
        .return_value;

        assert_eq!(
            Value::from(
                Felt::from_dec_str("106710729501573572985208420194530329073740042555888586719488")
                    .unwrap()
            ),
            return_value
        );

        let return_value = run_sierra_program(
            &STORAGE_ADDRESS_FROM_BASE_AND_OFFSET,
            &[Felt::ZERO.into(), 1u8.into()],
        )
        .return_value;

        assert_eq!(Value::from(Felt::ONE), return_value);

        let return_value = run_sierra_program(
            &STORAGE_ADDRESS_FROM_BASE_AND_OFFSET,
            &[Felt::ONE.into(), 1u8.into()],
        )
        .return_value;

        assert_eq!(Value::from(Felt::from(2)), return_value);

        let return_value = run_sierra_program(
            &STORAGE_ADDRESS_FROM_BASE_AND_OFFSET,
            &[
                Felt::from_dec_str("106710729501573572985208420194530329073740042555888586719488")
                    .unwrap()
                    .into(),
                1u8.into(),
            ],
        )
        .return_value;

        assert_eq!(
            Value::from(
                Felt::from_dec_str("106710729501573572985208420194530329073740042555888586719489")
                    .unwrap()
            ),
            return_value
        );

        let return_value = run_sierra_program(
            &STORAGE_ADDRESS_FROM_BASE_AND_OFFSET,
            &[Felt::ZERO.into(), 255u8.into()],
        )
        .return_value;

        assert_eq!(Value::from(Felt::from(255)), return_value);

        let return_value = run_sierra_program(
            &STORAGE_ADDRESS_FROM_BASE_AND_OFFSET,
            &[Felt::ONE.into(), 255u8.into()],
        )
        .return_value;

        assert_eq!(Value::from(Felt::from(256)), return_value);

        let return_value = run_sierra_program(
            &STORAGE_ADDRESS_FROM_BASE_AND_OFFSET,
            &[
                Felt::from_dec_str("106710729501573572985208420194530329073740042555888586719488")
                    .unwrap()
                    .into(),
                255u8.into(),
            ],
        )
        .return_value;

        assert_eq!(
            Value::from(
                Felt::from_dec_str("106710729501573572985208420194530329073740042555888586719743")
                    .unwrap()
            ),
            return_value
        )
    }

    #[test]
    fn storage_address_to_felt252() {
        let return_value =
            run_sierra_program(&STORAGE_ADDRESS_TO_FELT252, &[Felt::ZERO.into()]).return_value;

        assert_eq!(Value::from(Felt::ZERO), return_value);

        let return_value =
            run_sierra_program(&STORAGE_ADDRESS_TO_FELT252, &[Felt::ONE.into()]).return_value;

        assert_eq!(Value::from(Felt::ONE), return_value);

        let return_value = run_sierra_program(
            &STORAGE_ADDRESS_TO_FELT252,
            &[
                Felt::from_dec_str("106710729501573572985208420194530329073740042555888586719488")
                    .unwrap()
                    .into(),
            ],
        )
        .return_value;

        assert_eq!(
            Value::from(
                Felt::from_dec_str("106710729501573572985208420194530329073740042555888586719488")
                    .unwrap()
            ),
            return_value
        );
    }

    #[test]
    fn storage_address_try_from_felt252() {
        let return_value =
            run_sierra_program(&STORAGE_ADDRESS_TRY_FROM_FELT252, &[Felt::ZERO.into()])
                .return_value;

        assert_eq!(jit_enum!(0, Felt::ZERO.into()), return_value);

        let return_value =
            run_sierra_program(&STORAGE_ADDRESS_TRY_FROM_FELT252, &[Felt::ONE.into()]).return_value;

        assert_eq!(jit_enum!(0, Felt::ONE.into()), return_value);

        let return_value = run_sierra_program(
            &STORAGE_ADDRESS_TRY_FROM_FELT252,
            &[
                Felt::from_dec_str("106710729501573572985208420194530329073740042555888586719488")
                    .unwrap()
                    .into(),
            ],
        )
        .return_value;

        assert_eq!(
            jit_enum!(
                0,
                Felt::from_dec_str("106710729501573572985208420194530329073740042555888586719488")
                    .unwrap()
                    .into()
            ),
            return_value
        );

        let return_value =
            run_sierra_program(&STORAGE_ADDRESS_TRY_FROM_FELT252, &[Felt::from(-1).into()])
                .return_value;

        assert_eq!(jit_enum!(1, jit_struct!()), return_value);
    }
}
