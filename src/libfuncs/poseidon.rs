//! # Poseidon hashing libfuncs
//!

use super::LibfuncHelper;
use crate::{
    error::Result,
    metadata::{runtime_bindings::RuntimeBindingsMeta, MetadataStorage},
    utils::{get_integer_layout, BlockExt, ProgramRegistryExt},
};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureOnlyConcreteLibfunc,
        poseidon::PoseidonConcreteLibfunc,
        ConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::ods,
    ir::{r#type::IntegerType, Block, Location},
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
    selector: &PoseidonConcreteLibfunc,
) -> Result<()> {
    match selector {
        PoseidonConcreteLibfunc::HadesPermutation(info) => {
            build_hades_permutation(context, registry, entry, location, helper, metadata, info)
        }
    }
}

pub fn build_hades_permutation<'ctx>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'ctx Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, '_>,
    metadata: &mut MetadataStorage,
    info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.");

    let poseidon_builtin =
        super::increment_builtin_counter(context, entry, location, entry.arg(0)?)?;

    let felt252_ty =
        registry.build_type(context, helper, metadata, &info.param_signatures()[1].ty)?;

    let i256_ty = IntegerType::new(context, 256).into();
    let layout_i256 = get_integer_layout(256);

    let op0 = entry.arg(1)?;
    let op1 = entry.arg(2)?;
    let op2 = entry.arg(3)?;

    // We must extend to i256 because bswap must be an even number of bytes.

    let op0_ptr = helper
        .init_block()
        .alloca1(context, location, i256_ty, layout_i256.align())?;
    let op1_ptr = helper
        .init_block()
        .alloca1(context, location, i256_ty, layout_i256.align())?;
    let op2_ptr = helper
        .init_block()
        .alloca1(context, location, i256_ty, layout_i256.align())?;

    let op0_i256 =
        entry.append_op_result(ods::arith::extui(context, i256_ty, op0, location).into())?;

    let op1_i256 =
        entry.append_op_result(ods::arith::extui(context, i256_ty, op1, location).into())?;
    let op2_i256 =
        entry.append_op_result(ods::arith::extui(context, i256_ty, op2, location).into())?;

    entry.store(context, location, op0_ptr, op0_i256)?;
    entry.store(context, location, op1_ptr, op1_i256)?;
    entry.store(context, location, op2_ptr, op2_i256)?;

    let runtime_bindings = metadata
        .get_mut::<RuntimeBindingsMeta>()
        .expect("Runtime library not available.");

    runtime_bindings
        .libfunc_hades_permutation(context, helper, entry, op0_ptr, op1_ptr, op2_ptr, location)?;

    let op0_i256 = entry.load(context, location, op0_ptr, i256_ty)?;
    let op1_i256 = entry.load(context, location, op1_ptr, i256_ty)?;
    let op2_i256 = entry.load(context, location, op2_ptr, i256_ty)?;

    let op0 = entry.trunci(op0_i256, felt252_ty, location)?;
    let op1 = entry.trunci(op1_i256, felt252_ty, location)?;
    let op2 = entry.trunci(op2_i256, felt252_ty, location)?;

    entry.append_operation(helper.br(0, &[poseidon_builtin, op0, op1, op2], location));

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::utils::test::{jit_struct, run_sierra_program};

    use cairo_lang_sierra::ProgramParser;
    use starknet_types_core::felt::Felt;

    #[test]
    fn run_hades_permutation() {
        // use core::poseidon::hades_permutation;
        // fn run_test(a: felt252, b: felt252, c: felt252) -> (felt252, felt252, felt252) {
        //     hades_permutation(a, b, c)
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
                type [0] = Poseidon [storable: true, drop: false, dup: false, zero_sized: false];
                type [1] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
                type [2] = Struct<ut@Tuple, [1], [1], [1]> [storable: true, drop: true, dup: true, zero_sized: false];

                libfunc [1] = hades_permutation;
                libfunc [0] = struct_construct<[2]>;
                libfunc [3] = store_temp<[0]>;
                libfunc [4] = store_temp<[2]>;

                [1]([0], [1], [2], [3]) -> ([4], [5], [6], [7]); // 0
                [0]([5], [6], [7]) -> ([8]); // 1
                [3]([4]) -> ([4]); // 2
                [4]([8]) -> ([8]); // 3
                return([4], [8]); // 4

                [0]@0([0]: [0], [1]: [1], [2]: [1], [3]: [1]) -> ([0], [2]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let return_value = run_sierra_program(
            &program,
            &[
                Felt::from(2).into(),
                Felt::from(4).into(),
                Felt::from(4).into(),
            ],
        )
        .return_value;

        assert_eq!(
            jit_struct!(
                Felt::from_dec_str(
                    "1627044480024625333712068603977073585655327747658231320998869768849911913066"
                )
                .unwrap()
                .into(),
                Felt::from_dec_str(
                    "2368195581807763724810563135784547417602556730014791322540110420941926079965"
                )
                .unwrap()
                .into(),
                Felt::from_dec_str(
                    "2381325839211954898363395375151559373051496038592329842107874845056395867189"
                )
                .unwrap()
                .into(),
            ),
            return_value
        );
    }
}
