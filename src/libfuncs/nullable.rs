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

    helper.br(entry, 0, &[value], location)
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

    helper.br(block_is_null, 0, &[], location)?;
    helper.br(block_is_not_null, 1, &[arg], location)?;

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{
        utils::test::{jit_enum, jit_struct, load_cairo, run_program_assert_output},
        values::Value,
    };

    #[test]
    fn run_null() {
        let program = load_cairo!(
            use nullable::null;
            use nullable::match_nullable;
            use nullable::FromNullableResult;
            use nullable::nullable_from_box;
            use box::BoxTrait;

            fn run_test() {
                let _a: Nullable<u8> = null();
            }
        );

        run_program_assert_output(&program, "run_test", &[], jit_struct!());
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
        let program = load_cairo!(
            use nullable::null;
            use nullable::match_nullable;
            use nullable::FromNullableResult;
            use nullable::nullable_from_box;
            use box::BoxTrait;

            fn run_test(x: u8) -> u8 {
                let b: Box<u8> = BoxTrait::new(x);
                let c = if x == 0 {
                    null()
                } else {
                    nullable_from_box(b)
                };
                let d = match match_nullable(c) {
                    FromNullableResult::Null(_) => 99_u8,
                    FromNullableResult::NotNull(value) => value.unbox()
                };
                d
            }
        );

        run_program_assert_output(&program, "run_test", &[4u8.into()], 4u8.into());
        run_program_assert_output(&program, "run_test", &[0u8.into()], 99u8.into());
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
