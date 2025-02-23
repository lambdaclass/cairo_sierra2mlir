//! # Elliptic curve libfuncs

use super::LibfuncHelper;
use crate::{
    error::{Error, Result},
    metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
    utils::{get_integer_layout, BlockExt, ProgramRegistryExt, PRIME},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        ec::EcConcreteLibfunc,
        lib_func::SignatureOnlyConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        llvm,
    },
    ir::{operation::OperationBuilder, r#type::IntegerType, Block, BlockLike, Location},
    Context,
};

/// Select and call the correct libfunc builder function from the selector.
pub fn build<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &EcConcreteLibfunc,
) -> Result<()> {
    match selector {
        EcConcreteLibfunc::IsZero(info) => {
            build_is_zero(context, registry, entry, location, helper, metadata, info)
        }
        EcConcreteLibfunc::Neg(info) => {
            build_neg(context, registry, entry, location, helper, metadata, info)
        }
        EcConcreteLibfunc::PointFromX(info) => {
            build_point_from_x(context, registry, entry, location, helper, metadata, info)
        }
        EcConcreteLibfunc::StateAdd(info) => {
            build_state_add(context, registry, entry, location, helper, metadata, info)
        }
        EcConcreteLibfunc::StateAddMul(info) => {
            build_state_add_mul(context, registry, entry, location, helper, metadata, info)
        }
        EcConcreteLibfunc::StateFinalize(info) => {
            build_state_finalize(context, registry, entry, location, helper, metadata, info)
        }
        EcConcreteLibfunc::StateInit(info) => {
            build_state_init(context, registry, entry, location, helper, metadata, info)
        }
        EcConcreteLibfunc::TryNew(info) => {
            build_try_new(context, registry, entry, location, helper, metadata, info)
        }
        EcConcreteLibfunc::UnwrapPoint(info) => {
            build_unwrap_point(context, registry, entry, location, helper, metadata, info)
        }
        EcConcreteLibfunc::Zero(info) => {
            build_zero(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `ec_point_is_zero` libfunc.
pub fn build_is_zero<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    // To check whether `(x, y) = (0, 0)` (the zero point), it is enough to check
    // whether `y = 0`, since there is no point on the curve with y = 0.
    let y = entry.extract_value(
        context,
        location,
        entry.arg(0)?,
        IntegerType::new(context, 252).into(),
        1,
    )?;

    let k0 = entry.const_int(context, location, 0, 252)?;
    let y_is_zero = entry.cmpi(context, CmpiPredicate::Eq, y, k0, location)?;

    entry.append_operation(helper.cond_br(
        context,
        y_is_zero,
        [0, 1],
        [&[], &[entry.arg(0)?]],
        location,
    ));
    Ok(())
}

/// Generate MLIR operations for the `ec_neg` libfunc.
pub fn build_neg<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let y = entry.extract_value(
        context,
        location,
        entry.arg(0)?,
        IntegerType::new(context, 252).into(),
        1,
    )?;

    let k_prime = entry.const_int(context, location, PRIME.clone(), 252)?;
    let k0 = entry.const_int(context, location, 0, 252)?;

    let y_is_zero = entry.cmpi(context, CmpiPredicate::Eq, y, k0, location)?;

    let y_neg = entry.append_op_result(arith::subi(k_prime, y, location))?;
    let y_neg = entry.append_op_result(
        OperationBuilder::new("arith.select", location)
            .add_operands(&[y_is_zero, k0, y_neg])
            .add_results(&[IntegerType::new(context, 252).into()])
            .build()?,
    )?;

    let result = entry.insert_value(context, location, entry.arg(0)?, y_neg, 1)?;

    entry.append_operation(helper.br(0, &[result], location));
    Ok(())
}

/// Generate MLIR operations for the `ec_point_from_x_nz` libfunc.
pub fn build_point_from_x<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let range_check = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let ec_point_ty = llvm::r#type::r#struct(
        context,
        &[
            IntegerType::new(context, 252).into(),
            IntegerType::new(context, 252).into(),
        ],
        false,
    );

    let point_ptr = helper.init_block().alloca1(
        context,
        location,
        ec_point_ty,
        get_integer_layout(252).align(),
    )?;

    let point = entry.append_op_result(llvm::undef(ec_point_ty, location))?;
    let point = entry.insert_value(context, location, point, entry.arg(1)?, 0)?;

    entry.store(context, location, point_ptr, point)?;
    let result = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .ok_or(Error::MissingMetadata)?
        .libfunc_ec_point_from_x_nz(context, helper, entry, point_ptr, location)?
        .result(0)?
        .into();

    let point = entry.load(context, location, point_ptr, ec_point_ty)?;

    entry.append_operation(helper.cond_br(
        context,
        result,
        [0, 1],
        [&[range_check, point], &[range_check]],
        location,
    ));
    Ok(())
}

/// Generate MLIR operations for the `ec_state_add` libfunc.
pub fn build_state_add<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let ec_state_ty = llvm::r#type::r#struct(
        context,
        &[
            IntegerType::new(context, 252).into(),
            IntegerType::new(context, 252).into(),
            IntegerType::new(context, 252).into(),
            IntegerType::new(context, 252).into(),
        ],
        false,
    );

    let state_ptr = helper.init_block().alloca1(
        context,
        location,
        ec_state_ty,
        get_integer_layout(252).align(),
    )?;
    let point_ptr = helper.init_block().alloca1(
        context,
        location,
        ec_state_ty,
        get_integer_layout(252).align(),
    )?;

    entry.store(context, location, state_ptr, entry.arg(0)?)?;
    entry.store(context, location, point_ptr, entry.arg(1)?)?;

    metadata
        .get_mut::<RuntimeBindingsMeta>()
        .ok_or(Error::MissingMetadata)?
        .libfunc_ec_state_add(context, helper, entry, state_ptr, point_ptr, location)?;

    let state = entry.load(context, location, state_ptr, ec_state_ty)?;

    entry.append_operation(helper.br(0, &[state], location));
    Ok(())
}

/// Generate MLIR operations for the `ec_state_add_mul` libfunc.
pub fn build_state_add_mul<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let ec_op = super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let felt252_ty = IntegerType::new(context, 252).into();
    let ec_state_ty = llvm::r#type::r#struct(
        context,
        &[felt252_ty, felt252_ty, felt252_ty, felt252_ty],
        false,
    );
    let ec_point_ty = llvm::r#type::r#struct(context, &[felt252_ty, felt252_ty], false);

    let state_ptr = helper.init_block().alloca1(
        context,
        location,
        ec_state_ty,
        get_integer_layout(252).align(),
    )?;
    let scalar_ptr = helper.init_block().alloca1(
        context,
        location,
        felt252_ty,
        get_integer_layout(252).align(),
    )?;
    let point_ptr = helper.init_block().alloca1(
        context,
        location,
        ec_point_ty,
        get_integer_layout(252).align(),
    )?;

    entry.store(context, location, state_ptr, entry.arg(1)?)?;
    entry.store(context, location, scalar_ptr, entry.arg(2)?)?;
    entry.store(context, location, point_ptr, entry.arg(3)?)?;

    metadata
        .get_mut::<RuntimeBindingsMeta>()
        .ok_or(Error::MissingMetadata)?
        .libfunc_ec_state_add_mul(
            context, helper, entry, state_ptr, scalar_ptr, point_ptr, location,
        )?;

    let state = entry.load(context, location, state_ptr, ec_state_ty)?;

    entry.append_operation(helper.br(0, &[ec_op, state], location));
    Ok(())
}

/// Generate MLIR operations for the `ec_state_try_finalize_nz` libfunc.
pub fn build_state_finalize<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let felt252_ty = IntegerType::new(context, 252).into();
    let ec_state_ty = llvm::r#type::r#struct(
        context,
        &[felt252_ty, felt252_ty, felt252_ty, felt252_ty],
        false,
    );
    let ec_point_ty = llvm::r#type::r#struct(context, &[felt252_ty, felt252_ty], false);

    let point_ptr = helper.init_block().alloca1(
        context,
        location,
        ec_point_ty,
        get_integer_layout(252).align(),
    )?;
    let state_ptr = helper.init_block().alloca1(
        context,
        location,
        ec_state_ty,
        get_integer_layout(252).align(),
    )?;

    entry.store(context, location, state_ptr, entry.arg(0)?)?;

    let is_zero = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .ok_or(Error::MissingMetadata)?
        .libfunc_ec_state_try_finalize_nz(context, helper, entry, point_ptr, state_ptr, location)?
        .result(0)?
        .into();

    let point = entry.load(context, location, point_ptr, ec_point_ty)?;

    entry.append_operation(helper.cond_br(context, is_zero, [0, 1], [&[point], &[]], location));
    Ok(())
}

/// Generate MLIR operations for the `ec_state_init` libfunc.
pub fn build_state_init<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let ec_state_ty = llvm::r#type::r#struct(
        context,
        &[
            IntegerType::new(context, 252).into(),
            IntegerType::new(context, 252).into(),
            IntegerType::new(context, 252).into(),
            IntegerType::new(context, 252).into(),
        ],
        false,
    );

    let state_ptr = helper.init_block().alloca1(
        context,
        location,
        ec_state_ty,
        get_integer_layout(252).align(),
    )?;

    metadata
        .get_mut::<RuntimeBindingsMeta>()
        .ok_or(Error::MissingMetadata)?
        .libfunc_ec_state_init(context, helper, entry, state_ptr, location)?;

    let state = entry.load(context, location, state_ptr, ec_state_ty)?;

    entry.append_operation(helper.br(0, &[state], location));
    Ok(())
}

/// Generate MLIR operations for the `ec_point_try_new_nz` libfunc.
pub fn build_try_new<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let ec_point_ty = llvm::r#type::r#struct(
        context,
        &[
            IntegerType::new(context, 252).into(),
            IntegerType::new(context, 252).into(),
        ],
        false,
    );

    let point_ptr = helper.init_block().alloca1(
        context,
        location,
        ec_point_ty,
        get_integer_layout(252).align(),
    )?;

    let point = entry.append_op_result(llvm::undef(ec_point_ty, location))?;
    let point = entry.insert_value(context, location, point, entry.arg(0)?, 0)?;
    let point = entry.insert_value(context, location, point, entry.arg(1)?, 1)?;

    entry.store(context, location, point_ptr, point)?;

    let result = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .ok_or(Error::MissingMetadata)?
        .libfunc_ec_point_try_new_nz(context, helper, entry, point_ptr, location)?
        .result(0)?
        .into();

    entry.append_operation(helper.cond_br(context, result, [0, 1], [&[point], &[]], location));
    Ok(())
}

/// Generate MLIR operations for the `ec_point_unwrap` libfunc.
pub fn build_unwrap_point<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let x = entry.extract_value(
        context,
        location,
        entry.arg(0)?,
        registry.build_type(
            context,
            helper,
            metadata,
            &info.branch_signatures()[0].vars[0].ty,
        )?,
        0,
    )?;

    let y = entry.extract_value(
        context,
        location,
        entry.arg(0)?,
        registry.build_type(
            context,
            helper,
            metadata,
            &info.branch_signatures()[0].vars[1].ty,
        )?,
        1,
    )?;

    entry.append_operation(helper.br(0, &[x, y], location));
    Ok(())
}

/// Generate MLIR operations for the `ec_point_zero` libfunc.
pub fn build_zero<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let ec_point_ty = registry.build_type(
        context,
        helper,
        metadata,
        &info.branch_signatures()[0].vars[0].ty,
    )?;

    let point = entry.append_op_result(llvm::undef(ec_point_ty, location))?;

    let k0 = entry.const_int(context, location, 0, 252)?;

    let point = entry.insert_value(context, location, point, k0, 0)?;

    let point = entry.insert_value(context, location, point, k0, 1)?;

    entry.append_operation(helper.br(0, &[point], location));
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{
        utils::{
            sierra_gen::SierraGenerator,
            test::{jit_enum, jit_struct, run_sierra_program},
        },
        values::Value,
    };
    use cairo_lang_sierra::{
        extensions::ec::{
            EcCreatePointLibfunc, EcIsZeroLibfunc, EcNegLibfunc, EcPointFromXLibfunc,
            EcStateAddLibfunc, EcStateAddMulLibfunc, EcStateFinalizeLibfunc, EcStateInitLibfunc,
            EcUnwrapPointLibfunc, EcZeroLibfunc,
        },
        program::Program,
    };
    use lazy_static::lazy_static;
    use starknet_types_core::felt::Felt;
    use std::ops::Neg;

    lazy_static! {
        static ref EC_POINT_IS_ZERO: Program = {
            let generator = SierraGenerator::<EcIsZeroLibfunc>::default();

            generator.build(&[])
        };
        static ref EC_NEG: Program = {
            let generator = SierraGenerator::<EcNegLibfunc>::default();

            generator.build(&[])
        };
        static ref EC_POINT_FROM_X_NZ: Program = {
            let generator = SierraGenerator::<EcPointFromXLibfunc>::default();

            generator.build(&[])
        };
        static ref EC_STATE_ADD: Program = {
            let generator = SierraGenerator::<EcStateAddLibfunc>::default();

            generator.build(&[])
        };
        static ref EC_STATE_ADD_MUL: Program = {
            let generator = SierraGenerator::<EcStateAddMulLibfunc>::default();

            generator.build(&[])
        };
        static ref EC_STATE_FINALIZE: Program = {
            let generator = SierraGenerator::<EcStateFinalizeLibfunc>::default();

            generator.build(&[])
        };
        static ref EC_STATE_INIT: Program = {
            let generator = SierraGenerator::<EcStateInitLibfunc>::default();

            generator.build(&[])
        };
        static ref EC_POINT_TRY_NEW_NZ: Program = {
            let generator = SierraGenerator::<EcCreatePointLibfunc>::default();

            generator.build(&[])
        };
        static ref EC_POINT_UNWRAP: Program = {
            let generator = SierraGenerator::<EcUnwrapPointLibfunc>::default();

            generator.build(&[])
        };
        static ref EC_POINT_ZERO: Program = {
            let generator = SierraGenerator::<EcZeroLibfunc>::default();

            generator.build(&[])
        };
    }

    #[test]
    fn ec_point_is_zero() {
        println!("{}", EC_POINT_IS_ZERO.clone());
        let r = |x, y| run_sierra_program(&EC_POINT_IS_ZERO, &[Value::EcPoint(x, y)]).return_value;

        assert_eq!(r(0.into(), 0.into()), jit_enum!(0, jit_struct!()));
        assert_eq!(
            r(0.into(), 1.into()),
            jit_enum!(1, Value::EcPoint(0.into(), 1.into()))
        );
        assert_eq!(r(1.into(), 0.into()), jit_enum!(0, jit_struct!()));
        assert_eq!(
            r(1.into(), 1.into()),
            jit_enum!(1, Value::EcPoint(1.into(), 1.into()))
        );
    }

    #[test]
    fn ec_neg() {
        let r = |x, y| run_sierra_program(&EC_NEG, &[Value::EcPoint(x, y)]).return_value;

        assert_eq!(r(0.into(), 0.into()), Value::EcPoint(0.into(), 0.into()));
        assert_eq!(
            r(0.into(), 1.into()),
            Value::EcPoint(0.into(), Felt::from(-1))
        );
        assert_eq!(r(1.into(), 0.into()), Value::EcPoint(1.into(), 0.into()));
        assert_eq!(
            r(1.into(), 1.into()),
            Value::EcPoint(1.into(), Felt::from(-1))
        );
    }

    #[test]
    fn ec_point_from_x() {
        let r = |x| run_sierra_program(&EC_POINT_FROM_X_NZ, &[Value::Felt252(x)]).return_value;

        assert_eq!(r(0.into()), jit_enum!(1, jit_struct!()));
        assert_eq!(r(1234.into()), jit_enum!(0, Value::EcPoint(
            Felt::from(1234),
            Felt::from_dec_str("1301976514684871091717790968549291947487646995000837413367950573852273027507").unwrap()
        )));
    }

    #[test]
    fn ec_state_add() {
        let result = run_sierra_program(&EC_STATE_ADD, &[
            Value::EcState(
                Felt::from_dec_str("3151312365169595090315724863753927489909436624354740709748557281394568342450").unwrap(),
                Felt::from_dec_str("2835232394579952276045648147338966184268723952674536708929458753792035266179").unwrap(),
                Felt::from_dec_str("3151312365169595090315724863753927489909436624354740709748557281394568342450").unwrap(),
                Felt::from_dec_str("2835232394579952276045648147338966184268723952674536708929458753792035266179").unwrap()
            ),
            Value::EcPoint(
                Felt::from_dec_str("1234").unwrap(),
                Felt::from_dec_str("1301976514684871091717790968549291947487646995000837413367950573852273027507").unwrap()
            )
        ]).return_value;

        assert_eq!(
            Value::EcState(
                Felt::from_dec_str(
                    "763975897824944497806946001227010133599886598340174017198031710397718335159"
                )
                .unwrap(),
                Felt::from_dec_str(
                    "2805180267536471620369715068237762638204710971142209985448115065526708105983"
                )
                .unwrap(),
                Felt::from_dec_str(
                    "3151312365169595090315724863753927489909436624354740709748557281394568342450"
                )
                .unwrap(),
                Felt::from_dec_str(
                    "2835232394579952276045648147338966184268723952674536708929458753792035266179"
                )
                .unwrap()
            ),
            result
        );
    }

    #[test]
    fn ec_state_add_mul() {
        let result = run_sierra_program(&EC_STATE_ADD_MUL, &[
            Value::EcState(
                Felt::from_dec_str("3151312365169595090315724863753927489909436624354740709748557281394568342450").unwrap(),
                Felt::from_dec_str("2835232394579952276045648147338966184268723952674536708929458753792035266179").unwrap(),
                Felt::from_dec_str("3151312365169595090315724863753927489909436624354740709748557281394568342450").unwrap(),
                Felt::from_dec_str("2835232394579952276045648147338966184268723952674536708929458753792035266179").unwrap()
            ),
            Felt::ONE.into(), // scalar
            Value::EcPoint(
                Felt::from_dec_str("1234").unwrap(),
                Felt::from_dec_str("1301976514684871091717790968549291947487646995000837413367950573852273027507").unwrap()
            )
        ]).return_value;

        assert_eq!(
            Value::EcState(
                Felt::from_dec_str(
                    "763975897824944497806946001227010133599886598340174017198031710397718335159"
                )
                .unwrap(),
                Felt::from_dec_str(
                    "2805180267536471620369715068237762638204710971142209985448115065526708105983"
                )
                .unwrap(),
                Felt::from_dec_str(
                    "3151312365169595090315724863753927489909436624354740709748557281394568342450"
                )
                .unwrap(),
                Felt::from_dec_str(
                    "2835232394579952276045648147338966184268723952674536708929458753792035266179"
                )
                .unwrap()
            ),
            result
        );

        let result = run_sierra_program(&EC_STATE_ADD_MUL, &[
            Value::EcState(
                Felt::from_dec_str("3151312365169595090315724863753927489909436624354740709748557281394568342450").unwrap(),
                Felt::from_dec_str("2835232394579952276045648147338966184268723952674536708929458753792035266179").unwrap(),
                Felt::from_dec_str("3151312365169595090315724863753927489909436624354740709748557281394568342450").unwrap(),
                Felt::from_dec_str("2835232394579952276045648147338966184268723952674536708929458753792035266179").unwrap()
            ),
            Felt::from(2).into(), // scalar
            Value::EcPoint(
                Felt::from_dec_str("1234").unwrap(),
                Felt::from_dec_str("1301976514684871091717790968549291947487646995000837413367950573852273027507").unwrap()
            )
        ]
        ).return_value;

        assert_eq!(
            Value::EcState(
                Felt::from_dec_str(
                    "3016674370847061744386893405108272070153695046160622325692702034987910716850"
                )
                .unwrap(),
                Felt::from_dec_str(
                    "898133181809473419542838028331350248951548889944002871647069130998202992502"
                )
                .unwrap(),
                Felt::from_dec_str(
                    "3151312365169595090315724863753927489909436624354740709748557281394568342450"
                )
                .unwrap(),
                Felt::from_dec_str(
                    "2835232394579952276045648147338966184268723952674536708929458753792035266179"
                )
                .unwrap()
            ),
            result
        )
    }

    #[test]
    fn ec_state_finalize() {
        let result = run_sierra_program(
            &EC_STATE_FINALIZE,
            &[Value::EcState(
                Felt::from_dec_str(
                    "3151312365169595090315724863753927489909436624354740709748557281394568342450",
                )
                .unwrap(),
                Felt::from_dec_str(
                    "2835232394579952276045648147338966184268723952674536708929458753792035266179",
                )
                .unwrap(),
                Felt::from_dec_str(
                    "3151312365169595090315724863753927489909436624354740709748557281394568342450",
                )
                .unwrap(),
                Felt::from_dec_str(
                    "2835232394579952276045648147338966184268723952674536708929458753792035266179",
                )
                .unwrap(),
            )],
        )
        .return_value;

        assert_eq!(jit_enum!(1, jit_struct!()), result);

        let result = run_sierra_program(
            &EC_STATE_FINALIZE,
            &[Value::EcState(
                Felt::from_dec_str(
                    "763975897824944497806946001227010133599886598340174017198031710397718335159",
                )
                .unwrap(),
                Felt::from_dec_str(
                    "2805180267536471620369715068237762638204710971142209985448115065526708105983",
                )
                .unwrap(),
                Felt::from_dec_str(
                    "3151312365169595090315724863753927489909436624354740709748557281394568342450",
                )
                .unwrap(),
                Felt::from_dec_str(
                    "2835232394579952276045648147338966184268723952674536708929458753792035266179",
                )
                .unwrap(),
            )],
        )
        .return_value;

        assert_eq!( jit_enum!(0, Value::EcPoint(
                Felt::from(1234),
                Felt::from_dec_str("1301976514684871091717790968549291947487646995000837413367950573852273027507").unwrap()
            )
        ), result);
    }

    #[test]
    fn ec_state_init() {
        let result = run_sierra_program(&EC_STATE_INIT, &[]);
        // cant match the values because the state init is a random point
        assert!(matches!(result.return_value, Value::EcState(_, _, _, _)));
    }

    #[test]
    fn ec_point_try_new_nz() {
        let result = run_sierra_program(
            &EC_POINT_TRY_NEW_NZ,
            &[
                Felt::from_dec_str("0").unwrap().into(),
                Felt::from_dec_str("0").unwrap().into(),
            ],
        )
        .return_value;

        assert_eq!(jit_enum!(1, jit_struct!()), result);

        let result = run_sierra_program(
            &EC_POINT_TRY_NEW_NZ,
            &[
                Felt::from_dec_str("1234").unwrap().into(),
                Felt::from_dec_str(
                    "1301976514684871091717790968549291947487646995000837413367950573852273027507",
                )
                .unwrap()
                .into(),
            ],
        )
        .return_value;

        assert_eq!(jit_enum!(0, Value::EcPoint(
            Felt::from_dec_str("1234").unwrap(),
            Felt::from_dec_str("1301976514684871091717790968549291947487646995000837413367950573852273027507").unwrap()
        )), result);

        let result = run_sierra_program(
            &EC_POINT_TRY_NEW_NZ,
            &[
                Felt::from_dec_str("1234").unwrap().into(),
                Felt::from_dec_str(
                    "1301976514684871091717790968549291947487646995000837413367950573852273027507",
                )
                .unwrap()
                .neg()
                .into(),
            ],
        )
        .return_value;

        assert_eq!(jit_enum!(0, Value::EcPoint(
            Felt::from_dec_str("1234").unwrap(),
            Felt::from_dec_str("1301976514684871091717790968549291947487646995000837413367950573852273027507").unwrap().neg()
        )), result);
    }

    #[test]
    fn ec_point_unwrap() {
        fn parse(x: &str) -> Felt {
            if let Some(x) = x.strip_prefix('-') {
                Felt::from_dec_str(x).unwrap().neg()
            } else {
                Felt::from_dec_str(x).unwrap()
            }
        }

        #[track_caller]
        fn run(a: &str, b: &str, ea: &str, eb: &str) {
            let result =
                run_sierra_program(&EC_POINT_UNWRAP, &[Value::EcPoint(parse(a), parse(b))])
                    .return_value;

            assert_eq!(jit_struct!(parse(ea).into(), parse(eb).into()), result)
        }

        run("0", "0", "0", "0");
        run("0", "1", "0", "1");
        run("0", "-1", "0", "-1");
        run("1", "0", "1", "0");
        run("1", "1", "1", "1");
        run("1", "-1", "1", "-1");
        run("-1", "0", "-1", "0");
        run("-1", "1", "-1", "1");
        run("-1", "-1", "-1", "-1");
    }

    #[test]
    fn ec_point_zero() {
        let result = run_sierra_program(&EC_POINT_ZERO, &[]).return_value;

        assert_eq!(
            Value::EcPoint(
                Felt::from_dec_str("0").unwrap(),
                Felt::from_dec_str("0").unwrap().neg(),
            ),
            result
        );
    }
}
