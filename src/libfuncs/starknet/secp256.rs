use crate::{
    error::Result,
    libfuncs::LibfuncHelper,
    metadata::MetadataStorage,
    starknet::handler::StarknetSyscallHandlerCallbacks,
    utils::{get_integer_layout, BlockExt, GepIndex, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
        starknet::secp256::{Secp256ConcreteLibfunc, Secp256OpConcreteLibfunc},
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::llvm::{self, LoadStoreOptions},
    ir::{
        attribute::DenseI32ArrayAttribute, operation::OperationBuilder, r#type::IntegerType, Block,
        BlockLike, Location,
    },
    Context,
};
use std::alloc::Layout;

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &Secp256ConcreteLibfunc,
) -> Result<()> {
    match selector {
        Secp256ConcreteLibfunc::K1(selector) => match selector {
            Secp256OpConcreteLibfunc::New(info) => {
                build_k1_new(context, registry, entry, location, helper, metadata, info)
            }
            Secp256OpConcreteLibfunc::Add(info) => {
                build_k1_add(context, registry, entry, location, helper, metadata, info)
            }
            Secp256OpConcreteLibfunc::Mul(info) => {
                build_k1_mul(context, registry, entry, location, helper, metadata, info)
            }
            Secp256OpConcreteLibfunc::GetPointFromX(info) => build_k1_get_point_from_x(
                context, registry, entry, location, helper, metadata, info,
            ),
            Secp256OpConcreteLibfunc::GetXy(info) => {
                build_k1_get_xy(context, registry, entry, location, helper, metadata, info)
            }
        },
        Secp256ConcreteLibfunc::R1(selector) => match selector {
            Secp256OpConcreteLibfunc::New(info) => {
                build_r1_new(context, registry, entry, location, helper, metadata, info)
            }
            Secp256OpConcreteLibfunc::Add(info) => {
                build_r1_add(context, registry, entry, location, helper, metadata, info)
            }
            Secp256OpConcreteLibfunc::Mul(info) => {
                build_r1_mul(context, registry, entry, location, helper, metadata, info)
            }
            Secp256OpConcreteLibfunc::GetPointFromX(info) => build_r1_get_point_from_x(
                context, registry, entry, location, helper, metadata, info,
            ),
            Secp256OpConcreteLibfunc::GetXy(info) => {
                build_r1_get_xy(context, registry, entry, location, helper, metadata, info)
            }
        },
    }
}

pub fn build_k1_new<'ctx, 'this>(
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
            entry.argument(1)?.into(),
            llvm::r#type::pointer(context, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

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
        entry.argument(0)?.into(),
        gas_builtin_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    let (x_ty, x_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.signature.param_signatures[2].ty,
    )?;
    let (y_ty, y_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.signature.param_signatures[3].ty,
    )?;

    // Allocate `x` argument and write the value.
    let x_arg_ptr = helper
        .init_block()
        .alloca1(context, location, x_ty, x_layout.align())?;
    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        x_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `y` argument and write the value.
    let y_arg_ptr = helper
        .init_block()
        .alloca1(context, location, y_ty, y_layout.align())?;
    entry.append_operation(llvm::store(
        context,
        entry.argument(3)?.into(),
        y_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarknetSyscallHandlerCallbacks::<()>::SECP256K1_NEW.try_into()?],
            ),
            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
            location,
        ))
        .result(0)?
        .into();
    let fn_ptr = entry
        .append_operation(llvm::load(
            context,
            fn_ptr,
            llvm::r#type::pointer(context, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[
                fn_ptr,
                result_ptr,
                ptr,
                gas_builtin_ptr,
                x_arg_ptr,
                y_arg_ptr,
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

    helper.cond_br(
        context,
        entry,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.argument(1)?.into(), payload_err],
            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
        ],
        location,
    )
}

pub fn build_k1_add<'ctx, 'this>(
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
            entry.argument(1)?.into(),
            llvm::r#type::pointer(context, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

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
        entry.argument(0)?.into(),
        gas_builtin_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    let (p0_ty, p0_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.signature.param_signatures[2].ty,
    )?;
    let (p1_ty, p1_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.signature.param_signatures[3].ty,
    )?;

    // Allocate `p0` argument and write the value.
    let p0_arg_ptr = helper
        .init_block()
        .alloca1(context, location, p0_ty, p0_layout.align())?;
    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        p0_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `p1` argument and write the value.
    let p1_arg_ptr = helper
        .init_block()
        .alloca1(context, location, p1_ty, p1_layout.align())?;

    entry.append_operation(llvm::store(
        context,
        entry.argument(3)?.into(),
        p1_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarknetSyscallHandlerCallbacks::<()>::SECP256K1_ADD.try_into()?],
            ),
            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
            location,
        ))
        .result(0)?
        .into();
    let fn_ptr = entry
        .append_operation(llvm::load(
            context,
            fn_ptr,
            llvm::r#type::pointer(context, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[
                fn_ptr,
                result_ptr,
                ptr,
                gas_builtin_ptr,
                p0_arg_ptr,
                p1_arg_ptr,
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

    helper.cond_br(
        context,
        entry,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.argument(1)?.into(), payload_err],
            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
        ],
        location,
    )
}

pub fn build_k1_mul<'ctx, 'this>(
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
            entry.argument(1)?.into(),
            llvm::r#type::pointer(context, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

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
        entry.argument(0)?.into(),
        gas_builtin_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    let (p_ty, p_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.signature.param_signatures[2].ty,
    )?;
    let (scalar_ty, scalar_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.signature.param_signatures[3].ty,
    )?;

    // Allocate `p` argument and write the value.
    let p_arg_ptr = helper
        .init_block()
        .alloca1(context, location, p_ty, p_layout.align())?;
    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        p_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `scalar` argument and write the value.
    let scalar_arg_ptr =
        helper
            .init_block()
            .alloca1(context, location, scalar_ty, scalar_layout.align())?;
    entry.append_operation(llvm::store(
        context,
        entry.argument(3)?.into(),
        scalar_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarknetSyscallHandlerCallbacks::<()>::SECP256K1_MUL.try_into()?],
            ),
            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
            location,
        ))
        .result(0)?
        .into();
    let fn_ptr = entry
        .append_operation(llvm::load(
            context,
            fn_ptr,
            llvm::r#type::pointer(context, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[
                fn_ptr,
                result_ptr,
                ptr,
                gas_builtin_ptr,
                p_arg_ptr,
                scalar_arg_ptr,
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

    helper.cond_br(
        context,
        entry,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.argument(1)?.into(), payload_err],
            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
        ],
        location,
    )
}

pub fn build_k1_get_point_from_x<'ctx, 'this>(
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
            entry.argument(1)?.into(),
            llvm::r#type::pointer(context, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

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
        entry.argument(0)?.into(),
        gas_builtin_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    let (x_ty, x_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.signature.param_signatures[2].ty,
    )?;

    // Allocate `x` argument and write the value.
    let x_arg_ptr = helper
        .init_block()
        .alloca1(context, location, x_ty, x_layout.align())?;

    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        x_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `y_parity` argument and write the value.
    let y_parity_arg_ptr = helper.init_block().alloca_int(context, location, 1)?;
    entry.append_operation(llvm::store(
        context,
        entry.argument(3)?.into(),
        y_parity_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarknetSyscallHandlerCallbacks::<()>::SECP256K1_GET_POINT_FROM_X.try_into()?],
            ),
            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
            location,
        ))
        .result(0)?
        .into();
    let fn_ptr = entry
        .append_operation(llvm::load(
            context,
            fn_ptr,
            llvm::r#type::pointer(context, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[
                fn_ptr,
                result_ptr,
                ptr,
                gas_builtin_ptr,
                x_arg_ptr,
                y_parity_arg_ptr,
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

    // Load the two variants of the result returned by the syscall handler.
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

    helper.cond_br(
        context,
        entry,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.argument(1)?.into(), payload_err],
            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
        ],
        location,
    )
}

pub fn build_k1_get_xy<'ctx, 'this>(
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
            entry.argument(1)?.into(),
            llvm::r#type::pointer(context, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) = {
        // Note: This libfunc has multiple return values when successful, therefore the method used
        //   for the other libfuncs cannot be reused here.

        let u128_layout = get_integer_layout(128);
        let u256_layout = u128_layout.extend(u128_layout)?.0;
        let u256_ty = llvm::r#type::r#struct(
            context,
            &[
                IntegerType::new(context, 128).into(),
                IntegerType::new(context, 128).into(),
            ],
            false,
        );

        let (ok_ty, ok_layout) = (
            llvm::r#type::r#struct(context, &[u256_ty, u256_ty], false),
            u256_layout.extend(u256_layout)?.0,
        );
        let (err_ty, err_layout) = registry.build_type_with_layout(
            context,
            helper,
            metadata,
            &info.branch_signatures()[1].vars[2].ty,
        )?;

        let (tag_ty, tag_layout) = (IntegerType::new(context, 1).into(), get_integer_layout(1));

        (
            tag_layout
                .extend(Layout::from_size_align(
                    ok_layout.size().max(err_layout.size()),
                    ok_layout.align().max(err_layout.align()),
                )?)?
                .0,
            (tag_ty, tag_layout),
            [(ok_ty, ok_layout), (err_ty, err_layout)],
        )
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
    entry.append_operation(llvm::store(
        context,
        entry.argument(0)?.into(),
        gas_builtin_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    let (p_ty, p_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.signature.param_signatures[2].ty,
    )?;

    // Allocate `p` argument and write the value.
    let p_arg_ptr = helper
        .init_block()
        .alloca1(context, location, p_ty, p_layout.align())?;
    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        p_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarknetSyscallHandlerCallbacks::<()>::SECP256K1_GET_XY.try_into()?],
            ),
            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
            location,
        ))
        .result(0)?
        .into();
    let fn_ptr = entry
        .append_operation(llvm::load(
            context,
            fn_ptr,
            llvm::r#type::pointer(context, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[fn_ptr, result_ptr, ptr, gas_builtin_ptr, p_arg_ptr])
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

        let value = entry.load(context, location, ptr, variant_tys[0].0)?;

        let x_value = entry.extract_value(
            context,
            location,
            value,
            llvm::r#type::r#struct(
                context,
                &[
                    IntegerType::new(context, 128).into(),
                    IntegerType::new(context, 128).into(),
                ],
                false,
            ),
            0,
        )?;
        let y_value = entry.extract_value(
            context,
            location,
            value,
            llvm::r#type::r#struct(
                context,
                &[
                    IntegerType::new(context, 128).into(),
                    IntegerType::new(context, 128).into(),
                ],
                false,
            ),
            1,
        )?;

        (x_value, y_value)
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

    helper.cond_br(
        context,
        entry,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.argument(1)?.into(), payload_err],
            &[
                remaining_gas,
                entry.argument(1)?.into(),
                payload_ok.0,
                payload_ok.1,
            ],
        ],
        location,
    )
}

pub fn build_r1_new<'ctx, 'this>(
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
            entry.argument(1)?.into(),
            llvm::r#type::pointer(context, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

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
        entry.argument(0)?.into(),
        gas_builtin_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    let (x_ty, x_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.signature.param_signatures[2].ty,
    )?;
    let (y_ty, y_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.signature.param_signatures[3].ty,
    )?;

    // Allocate `x` argument and write the value.
    let x_arg_ptr = helper
        .init_block()
        .alloca1(context, location, x_ty, x_layout.align())?;
    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        x_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `y` argument and write the value.
    let y_arg_ptr = helper
        .init_block()
        .alloca1(context, location, y_ty, y_layout.align())?;
    entry.append_operation(llvm::store(
        context,
        entry.argument(3)?.into(),
        y_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarknetSyscallHandlerCallbacks::<()>::SECP256R1_NEW.try_into()?],
            ),
            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
            location,
        ))
        .result(0)?
        .into();
    let fn_ptr = entry
        .append_operation(llvm::load(
            context,
            fn_ptr,
            llvm::r#type::pointer(context, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[
                fn_ptr,
                result_ptr,
                ptr,
                gas_builtin_ptr,
                x_arg_ptr,
                y_arg_ptr,
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

    // Load the two variants of the result returned by the syscall handler.
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

    helper.cond_br(
        context,
        entry,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.argument(1)?.into(), payload_err],
            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
        ],
        location,
    )
}

pub fn build_r1_add<'ctx, 'this>(
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
            entry.argument(1)?.into(),
            llvm::r#type::pointer(context, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

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
        entry.argument(0)?.into(),
        gas_builtin_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    let (p0_ty, p0_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.signature.param_signatures[2].ty,
    )?;
    let (p1_ty, p1_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.signature.param_signatures[3].ty,
    )?;

    // Allocate `p0` argument and write the value.
    let p0_arg_ptr = helper
        .init_block()
        .alloca1(context, location, p0_ty, p0_layout.align())?;
    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        p0_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `p1` argument and write the value.
    let p1_arg_ptr = helper
        .init_block()
        .alloca1(context, location, p1_ty, p1_layout.align())?;
    entry.append_operation(llvm::store(
        context,
        entry.argument(3)?.into(),
        p1_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Extract function pointer.
    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarknetSyscallHandlerCallbacks::<()>::SECP256R1_ADD.try_into()?],
            ),
            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
            location,
        ))
        .result(0)?
        .into();
    let fn_ptr = entry
        .append_operation(llvm::load(
            context,
            fn_ptr,
            llvm::r#type::pointer(context, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[
                fn_ptr,
                result_ptr,
                ptr,
                gas_builtin_ptr,
                p0_arg_ptr,
                p1_arg_ptr,
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

    helper.cond_br(
        context,
        entry,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.argument(1)?.into(), payload_err],
            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
        ],
        location,
    )
}

pub fn build_r1_mul<'ctx, 'this>(
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
            entry.argument(1)?.into(),
            llvm::r#type::pointer(context, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

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
        entry.argument(0)?.into(),
        gas_builtin_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    let (p_ty, p_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.signature.param_signatures[2].ty,
    )?;

    let (scalar_ty, scalar_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.signature.param_signatures[3].ty,
    )?;

    // Allocate `p` argument and write the value.
    let p_arg_ptr = helper
        .init_block()
        .alloca1(context, location, p_ty, p_layout.align())?;

    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        p_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `scalar` argument and write the value.
    let scalar_arg_ptr =
        helper
            .init_block()
            .alloca1(context, location, scalar_ty, scalar_layout.align())?;
    entry.append_operation(llvm::store(
        context,
        entry.argument(3)?.into(),
        scalar_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Extract function pointer.
    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarknetSyscallHandlerCallbacks::<()>::SECP256R1_MUL.try_into()?],
            ),
            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
            location,
        ))
        .result(0)?
        .into();
    let fn_ptr = entry
        .append_operation(llvm::load(
            context,
            fn_ptr,
            llvm::r#type::pointer(context, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[
                fn_ptr,
                result_ptr,
                ptr,
                gas_builtin_ptr,
                p_arg_ptr,
                scalar_arg_ptr,
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

    helper.cond_br(
        context,
        entry,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.argument(1)?.into(), payload_err],
            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
        ],
        location,
    )
}

pub fn build_r1_get_point_from_x<'ctx, 'this>(
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
            entry.argument(1)?.into(),
            llvm::r#type::pointer(context, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

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
        entry.argument(0)?.into(),
        gas_builtin_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `x` argument and write the value.
    let x_arg_ptr = helper.init_block().alloca1(
        context,
        location,
        llvm::r#type::r#struct(
            context,
            &[
                IntegerType::new(context, 128).into(),
                IntegerType::new(context, 128).into(),
            ],
            false,
        ),
        get_integer_layout(128).align(),
    )?;
    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        x_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Allocate `y_parity` argument and write the value.
    let y_parity_arg_ptr = helper.init_block().alloca_int(context, location, 1)?;
    entry.append_operation(llvm::store(
        context,
        entry.argument(3)?.into(),
        y_parity_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Extract function pointer.
    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarknetSyscallHandlerCallbacks::<()>::SECP256R1_GET_POINT_FROM_X.try_into()?],
            ),
            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
            location,
        ))
        .result(0)?
        .into();
    let fn_ptr = entry
        .append_operation(llvm::load(
            context,
            fn_ptr,
            llvm::r#type::pointer(context, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[
                fn_ptr,
                result_ptr,
                ptr,
                gas_builtin_ptr,
                x_arg_ptr,
                y_parity_arg_ptr,
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

    helper.cond_br(
        context,
        entry,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.argument(1)?.into(), payload_err],
            &[remaining_gas, entry.argument(1)?.into(), payload_ok],
        ],
        location,
    )
}

pub fn build_r1_get_xy<'ctx, 'this>(
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
            entry.argument(1)?.into(),
            llvm::r#type::pointer(context, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    // Allocate space for the return value.
    let (result_layout, (result_tag_ty, result_tag_layout), variant_tys) = {
        // Note: This libfunc has multiple return values when successful, therefore the method used
        //   for the other libfuncs cannot be reused here.

        let u128_layout = get_integer_layout(128);
        let u256_layout = u128_layout.extend(u128_layout)?.0;
        let u256_ty = llvm::r#type::r#struct(
            context,
            &[
                IntegerType::new(context, 128).into(),
                IntegerType::new(context, 128).into(),
            ],
            false,
        );

        let (ok_ty, ok_layout) = (
            llvm::r#type::r#struct(context, &[u256_ty, u256_ty], false),
            u256_layout.extend(u256_layout)?.0,
        );
        let (err_ty, err_layout) = registry.build_type_with_layout(
            context,
            helper,
            metadata,
            &info.branch_signatures()[1].vars[2].ty,
        )?;

        let (tag_ty, tag_layout) = (IntegerType::new(context, 1).into(), get_integer_layout(1));

        (
            tag_layout
                .extend(Layout::from_size_align(
                    ok_layout.size().max(err_layout.size()),
                    ok_layout.align().max(err_layout.align()),
                )?)?
                .0,
            (tag_ty, tag_layout),
            [(ok_ty, ok_layout), (err_ty, err_layout)],
        )
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
    entry.append_operation(llvm::store(
        context,
        entry.argument(0)?.into(),
        gas_builtin_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    let (p_ty, p_layout) = registry.build_type_with_layout(
        context,
        helper,
        metadata,
        &info.signature.param_signatures[2].ty,
    )?;

    // Allocate `p` argument and write the value.
    let p_arg_ptr = helper
        .init_block()
        .alloca1(context, location, p_ty, p_layout.align())?;

    entry.append_operation(llvm::store(
        context,
        entry.argument(2)?.into(),
        p_arg_ptr,
        location,
        LoadStoreOptions::default(),
    ));

    // Extract function pointer.
    let fn_ptr = entry
        .append_operation(llvm::get_element_ptr(
            context,
            entry.argument(1)?.into(),
            DenseI32ArrayAttribute::new(
                context,
                &[StarknetSyscallHandlerCallbacks::<()>::SECP256R1_GET_XY.try_into()?],
            ),
            llvm::r#type::pointer(context, 0),
            llvm::r#type::pointer(context, 0),
            location,
        ))
        .result(0)?
        .into();
    let fn_ptr = entry
        .append_operation(llvm::load(
            context,
            fn_ptr,
            llvm::r#type::pointer(context, 0),
            location,
            LoadStoreOptions::default(),
        ))
        .result(0)?
        .into();

    entry.append_operation(
        OperationBuilder::new("llvm.call", location)
            .add_operands(&[fn_ptr, result_ptr, ptr, gas_builtin_ptr, p_arg_ptr])
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

        let value = entry.load(context, location, ptr, variant_tys[0].0)?;

        let x_value = entry.extract_value(
            context,
            location,
            value,
            llvm::r#type::r#struct(
                context,
                &[
                    IntegerType::new(context, 128).into(),
                    IntegerType::new(context, 128).into(),
                ],
                false,
            ),
            0,
        )?;
        let y_value = entry.extract_value(
            context,
            location,
            value,
            llvm::r#type::r#struct(
                context,
                &[
                    IntegerType::new(context, 128).into(),
                    IntegerType::new(context, 128).into(),
                ],
                false,
            ),
            1,
        )?;

        (x_value, y_value)
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

    helper.cond_br(
        context,
        entry,
        result_tag,
        [1, 0],
        [
            &[remaining_gas, entry.argument(1)?.into(), payload_err],
            &[
                remaining_gas,
                entry.argument(1)?.into(),
                payload_ok.0,
                payload_ok.1,
            ],
        ],
        location,
    )
}
