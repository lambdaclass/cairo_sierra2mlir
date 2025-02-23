//! # Nullable libfuncs
//!
//! Like a Box but it can be null.

use super::LibfuncHelper;
use crate::{error::Result, metadata::MetadataStorage, utils::BlockExt};
use cairo_lang_sierra::{
    extensions::{
        core::{CoreLibfunc, CoreType},
        lib_func::{SignatureAndTypeConcreteLibfunc, SignatureOnlyConcreteLibfunc},
        nullable::NullableConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{cf, llvm::r#type::pointer, ods},
    ir::{
        attribute::IntegerAttribute, operation::OperationBuilder, r#type::IntegerType, Block,
        BlockLike, Identifier, Location,
    },
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
    selector: &NullableConcreteLibfunc,
) -> Result<()> {
    match selector {
        NullableConcreteLibfunc::ForwardSnapshot(info)
        | NullableConcreteLibfunc::NullableFromBox(info) => super::build_noop::<1, true>(
            context,
            registry,
            entry,
            location,
            helper,
            metadata,
            &info.signature.param_signatures,
        ),
        NullableConcreteLibfunc::MatchNullable(info) => {
            build_match_nullable(context, registry, entry, location, helper, metadata, info)
        }
        NullableConcreteLibfunc::Null(info) => {
            build_null(context, registry, entry, location, helper, metadata, info)
        }
    }
}

/// Generate MLIR operations for the `null` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_null<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureOnlyConcreteLibfunc,
) -> Result<()> {
    let value = entry
        .append_op_result(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())?;

    entry.append_operation(helper.br(0, &[value], location));
    Ok(())
}

/// Generate MLIR operations for the `match_nullable` libfunc.
#[allow(clippy::too_many_arguments)]
fn build_match_nullable<'ctx, 'this>(
    context: &'ctx Context,
    _registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    _info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let arg = entry.arg(0)?;

    let nullptr = entry
        .append_op_result(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())?;

    let is_null_ptr = entry.append_op_result(
        OperationBuilder::new("llvm.icmp", location)
            .add_operands(&[arg, nullptr])
            .add_attributes(&[(
                Identifier::new(context, "predicate"),
                IntegerAttribute::new(IntegerType::new(context, 64).into(), 0).into(),
            )])
            .add_results(&[IntegerType::new(context, 1).into()])
            .build()?,
    )?;

    let block_is_null = helper.append_block(Block::new(&[]));
    let block_is_not_null = helper.append_block(Block::new(&[]));

    entry.append_operation(cf::cond_br(
        context,
        is_null_ptr,
        block_is_null,
        block_is_not_null,
        &[],
        &[],
        location,
    ));

    block_is_null.append_operation(helper.br(0, &[], location));
    block_is_not_null.append_operation(helper.br(1, &[arg], location));

    Ok(())
}

#[cfg(test)]
mod test {
    use cairo_lang_sierra::{
        extensions::{
            int::unsigned::Uint8Type,
            nullable::{MatchNullableLibfunc, NullLibfunc, NullableFromBoxLibfunc, NullableType},
        },
        program::GenericArg,
    };

    use crate::{
        utils::{
            sierra_gen::SierraGenerator,
            test::{
                jit_enum, jit_struct, load_cairo, run_program_assert_output, run_sierra_program,
            },
        },
        values::Value,
    };

    #[test]
    fn run_null() {
        let program = {
            let mut generator = SierraGenerator::<NullLibfunc>::default();

            let u8_ty = generator.push_type_declaration::<Uint8Type>(&[]).clone();
            let nullable_ty = generator
                .push_type_declaration::<NullableType>(&[GenericArg::Type(u8_ty)])
                .clone();

            generator.build(&[GenericArg::Type(nullable_ty)])
        };

        let result = run_sierra_program(&program, &[]).return_value;

        assert_eq!(Value::Null, result);
    }

    #[test]
    fn run_null_jit() {
        let program = load_cairo!(
            use nullable::null;
            use nullable::match_nullable;
            use nullable::FromNullableResult;
            use nullable::nullable_from_box;
            use box::BoxTrait;

            fn run_test() -> Nullable<u8> {
                let a: Nullable<u8> = null();
                a
            }
        );

        run_program_assert_output(&program, "run_test", &[], Value::Null);
    }

    #[test]
    fn run_not_null() {
        let program_from_box = {
            let mut generator = SierraGenerator::<NullableFromBoxLibfunc>::default();

            let u8_ty = generator.push_type_declaration::<Uint8Type>(&[]).clone();

            generator.build(&[GenericArg::Type(u8_ty)])
        };
        let program_match = {
            let mut generator = SierraGenerator::<MatchNullableLibfunc>::default();

            let u8_ty = generator.push_type_declaration::<Uint8Type>(&[]).clone();

            generator.build(&[GenericArg::Type(u8_ty)])
        };

        let result = run_sierra_program(&program_from_box, &[4u8.into()]).return_value;
        let result = run_sierra_program(&program_match, &[result]).return_value;

        assert_eq!(jit_enum!(1, 4u8.into()), result);

        let result = run_sierra_program(&program_match, &[Value::Null]).return_value;

        assert_eq!(jit_enum!(0, jit_struct!()), result);
    }

    #[test]
    fn match_nullable() {
        let program_match = {
            let mut generator = SierraGenerator::<MatchNullableLibfunc>::default();

            let u8_ty = generator.push_type_declaration::<Uint8Type>(&[]).clone();

            generator.build(&[GenericArg::Type(u8_ty)])
        };

        println!("{program_match}");

        let result = run_sierra_program(&program_match, &[Value::Null]).return_value;
        assert_eq!(result, jit_enum!(0, jit_struct!()));

        let result = run_sierra_program(&program_match, &[1u8.into()]).return_value;
        assert_eq!(result, jit_enum!(1, 1u8.into()));
    }

    #[test]
    fn match_snapshot_nullable_clone_bug() {
        let program = load_cairo! {
            use core::{NullableTrait, match_nullable, null, nullable::FromNullableResult};

            fn run_test(x: Option<u8>) -> Option<u8> {
                let a = match x {
                    Option::Some(x) => @NullableTrait::new(x),
                    Option::None(_) => @null::<u8>(),
                };
                let b = *a;
                match match_nullable(b) {
                    FromNullableResult::Null(_) => Option::None(()),
                    FromNullableResult::NotNull(x) => Option::Some(x.unbox()),
                }
            }
        };

        run_program_assert_output(
            &program,
            "run_test",
            &[jit_enum!(0, 42u8.into())],
            jit_enum!(0, 42u8.into()),
        );
        run_program_assert_output(
            &program,
            "run_test",
            &[jit_enum!(
                1,
                Value::Struct {
                    fields: Vec::new(),
                    debug_name: None
                }
            )],
            jit_enum!(
                1,
                Value::Struct {
                    fields: Vec::new(),
                    debug_name: None
                }
            ),
        );
    }
}
