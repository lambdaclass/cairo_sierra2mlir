use super::{LibfuncBuilder, LibfuncHelper};
use crate::{
    metadata::{prime_modulo::PrimeModulo, MetadataStorage},
    types::{felt252::Felt252, TypeBuilder},
};
use cairo_lang_sierra::{
    extensions::{
        felt252::{
            Felt252BinaryOperationConcrete, Felt252BinaryOperator, Felt252Concrete,
            Felt252ConstConcreteLibfunc,
        },
        GenericLibfunc, GenericType,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::arith,
    ir::{r#type::IntegerType, Attribute, Block, Location, Type},
    Context,
};

pub fn build<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    selector: &Felt252Concrete,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    match selector {
        Felt252Concrete::BinaryOperation(info) => {
            build_binary_operation(context, registry, entry, location, helper, metadata, info)
        }
        Felt252Concrete::Const(info) => {
            build_const(context, registry, entry, location, helper, metadata, info)
        }
        Felt252Concrete::IsZero(_) => todo!(),
    }
}

pub fn build_binary_operation<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &Felt252BinaryOperationConcrete,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    let prime = metadata.get::<PrimeModulo<Felt252>>().unwrap().prime();

    let result = match info {
        Felt252BinaryOperationConcrete::WithVar(info) => match info.operator {
            Felt252BinaryOperator::Add => todo!(),
            Felt252BinaryOperator::Sub => todo!(),
            Felt252BinaryOperator::Mul => {
                let double_felt252_ty: Type = IntegerType::new(context, 504).into();

                let op0 = entry.append_operation(arith::extui(
                    entry.argument(0).unwrap().into(),
                    double_felt252_ty,
                    location,
                ));
                let op1 = entry.append_operation(arith::extui(
                    entry.argument(1).unwrap().into(),
                    double_felt252_ty,
                    location,
                ));

                let op2 = entry.append_operation(arith::muli(
                    op0.result(0).unwrap().into(),
                    op1.result(0).unwrap().into(),
                    location,
                ));
                let op3 = entry.append_operation(arith::constant(
                    context,
                    Attribute::parse(context, &format!("{prime} : i504")).unwrap(),
                    location,
                ));
                let op4 = entry.append_operation(arith::remui(
                    op2.result(0).unwrap().into(),
                    op3.result(0).unwrap().into(),
                    location,
                ));

                op4.result(0).unwrap().into()
            }
            Felt252BinaryOperator::Div => todo!(),
        },
        Felt252BinaryOperationConcrete::WithConst(_) => todo!(),
    };

    entry.append_operation(helper.br(0, &[result], location));

    Ok(())
}

pub fn build_const<'ctx, 'this, TType, TLibfunc>(
    context: &'ctx Context,
    registry: &ProgramRegistry<TType, TLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &Felt252ConstConcreteLibfunc,
) -> Result<(), std::convert::Infallible>
where
    TType: GenericType,
    TLibfunc: GenericLibfunc,
    <TType as GenericType>::Concrete: TypeBuilder,
    <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
{
    let value = &info.c;
    let value_ty = registry
        .get_type(&info.signature.branch_signatures[0].vars[0].ty)
        .unwrap()
        .build(context, helper, registry, metadata)
        .unwrap();

    let op0 = entry.append_operation(arith::constant(
        context,
        Attribute::parse(context, &format!("{value} : {value_ty}")).unwrap(),
        location,
    ));
    entry.append_operation(helper.br(0, &[op0.result(0).unwrap().into()], location));

    Ok(())
}
