//! # Box libfuncs
//!
//! A heap allocated value, which is internally a pointer that can't be null.

use super::{BlockExt, LibfuncHelper};
use crate::{
    error::Result,
    metadata::{realloc_bindings::ReallocBindingsMeta, MetadataStorage},
    types::TypeBuilder,
};
use cairo_lang_sierra::{
    extensions::{
        boxing::BoxConcreteLibfunc,
        core::{CoreLibfunc, CoreType},
        lib_func::SignatureAndTypeConcreteLibfunc,
    },
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{
        arith,
        llvm::{self, r#type::pointer, LoadStoreOptions},
        ods,
    },
    ir::{attribute::IntegerAttribute, r#type::IntegerType, Block, Location},
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
    selector: &BoxConcreteLibfunc,
) -> Result<()> {
    match selector {
        BoxConcreteLibfunc::Into(info) => {
            build_into_box(context, registry, entry, location, helper, metadata, info)
        }
        BoxConcreteLibfunc::Unbox(info) => {
            build_unbox(context, registry, entry, location, helper, metadata, info)
        }
        BoxConcreteLibfunc::ForwardSnapshot(info) => super::build_noop::<1, true>(
            context,
            registry,
            entry,
            location,
            helper,
            metadata,
            &info.signature.param_signatures,
        ),
    }
}

/// Generate MLIR operations for the `into_box` libfunc.
pub fn build_into_box<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, helper));
    }

    let inner_type = registry.get_type(&info.ty)?;
    let inner_layout = inner_type.layout(registry)?;

    let value_len = entry
        .append_operation(arith::constant(
            context,
            IntegerAttribute::new(
                IntegerType::new(context, 64).into(),
                inner_layout.pad_to_align().size().try_into()?,
            )
            .into(),
            location,
        ))
        .result(0)?
        .into();

    let ptr = entry
        .append_operation(ods::llvm::mlir_zero(context, pointer(context, 0), location).into())
        .result(0)?
        .into();
    let ptr = entry
        .append_operation(ReallocBindingsMeta::realloc(
            context, ptr, value_len, location,
        )?)
        .result(0)?
        .into();

    entry.append_operation(llvm::store(
        context,
        entry.arg(0)?,
        ptr,
        location,
        LoadStoreOptions::new().align(Some(IntegerAttribute::new(
            IntegerType::new(context, 64).into(),
            inner_layout.align() as i64,
        ))),
    ));

    entry.append_operation(helper.br(0, &[ptr], location));
    Ok(())
}

/// Generate MLIR operations for the `unbox` libfunc.
pub fn build_unbox<'ctx, 'this>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    metadata: &mut MetadataStorage,
    info: &SignatureAndTypeConcreteLibfunc,
) -> Result<()> {
    let inner_type = registry.get_type(&info.ty)?;
    let inner_ty = inner_type.build(context, helper, registry, metadata, &info.ty)?;
    let inner_layout = inner_type.layout(registry)?;

    if metadata.get::<ReallocBindingsMeta>().is_none() {
        metadata.insert(ReallocBindingsMeta::new(context, helper));
    }

    // Load the boxed value from memory.
    let value = entry
        .append_operation(llvm::load(
            context,
            entry.arg(0)?,
            inner_ty,
            location,
            LoadStoreOptions::new().align(Some(IntegerAttribute::new(
                IntegerType::new(context, 64).into(),
                inner_layout.align() as i64,
            ))),
        ))
        .result(0)?
        .into();

    entry.append_operation(ReallocBindingsMeta::free(context, entry.arg(0)?, location)?);

    entry.append_operation(helper.br(0, &[value], location));
    Ok(())
}

#[cfg(test)]
mod test {
    use cairo_lang_sierra::ProgramParser;

    use crate::{utils::test::run_sierra_program, values::Value};

    #[test]
    fn run_box_unbox() {
        // use box::BoxTrait;
        // use box::BoxImpl;
        // fn run_test() -> u32 {
        //     let x: u32 = 2_u32;
        //     let box_x: Box<u32> = BoxTrait::new(x);
        //     box_x.unbox()
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
            type [1] = Box<[0]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [0] = u32 [storable: true, drop: true, dup: true, zero_sized: false];
            type [2] = Const<[0], 2> [storable: false, drop: false, dup: false, zero_sized: false];

            libfunc [1] = const_as_box<[2], 0>;
            libfunc [0] = unbox<[0]>;
            libfunc [2] = store_temp<[0]>;

            [1]() -> ([0]); // 0
            [0]([0]) -> ([1]); // 1
            [2]([1]) -> ([1]); // 2
            return([1]); // 3

            [0]@0() -> ([0]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let return_value = run_sierra_program(&program, &[]).return_value;

        assert_eq!(Value::Uint32(2), return_value);
    }

    #[test]
    fn run_box() {
        // use box::BoxTrait;
        // use box::BoxImpl;
        // fn run_test() -> Box<u32>  {
        //     let x: u32 = 2_u32;
        //     let box_x: Box<u32> = BoxTrait::new(x);
        //     box_x
        // }
        let program = ProgramParser::new()
            .parse(
                r#"
            type [1] = Box<[0]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [2] = Const<[0], 2> [storable: false, drop: false, dup: false, zero_sized: false];
            type [0] = u32 [storable: true, drop: true, dup: true, zero_sized: false];

            libfunc [0] = const_as_box<[2], 0>;

            [0]() -> ([0]); // 0
            return([0]); // 1

            [0]@0() -> ([1]);
        "#,
            )
            .map_err(|e| e.to_string())
            .unwrap();

        let return_value = run_sierra_program(&program, &[]).return_value;

        assert_eq!(Value::Uint32(2), return_value);
    }

    #[test]
    fn box_unbox_stack_allocated_enum_single() {
        // use core::box::BoxTrait;
        // enum MyEnum {
        //     A: felt252,
        // }
        // fn run_test() -> MyEnum {
        //     let x = BoxTrait::new(MyEnum::A(1234));
        //     x.unbox()
        // }
        let program = ProgramParser::new().parse(r#"
            type [2] = Box<[1]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [0] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = Enum<ut@program::program::MyEnum, [0]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [4] = Const<[1], 0, [3]> [storable: false, drop: false, dup: false, zero_sized: false];
            type [3] = Const<[0], 1234> [storable: false, drop: false, dup: false, zero_sized: false];

            libfunc [1] = const_as_box<[4], 0>;
            libfunc [0] = unbox<[1]>;
            libfunc [2] = store_temp<[1]>;

            [1]() -> ([0]); // 0
            [0]([0]) -> ([1]); // 1
            [2]([1]) -> ([1]); // 2
            return([1]); // 3

            [0]@0() -> ([1]);
        "#).map_err(|e| e.to_string()).unwrap();

        let return_value = run_sierra_program(&program, &[]).return_value;

        assert_eq!(
            return_value,
            Value::Enum {
                tag: 0,
                value: Box::new(Value::Felt252(1234.into())),
                debug_name: None,
            },
        );
    }

    #[test]
    fn box_unbox_stack_allocated_enum_c() {
        // use core::box::BoxTrait;
        // enum MyEnum {
        //     A: (),
        //     B: (),
        // }
        // fn run_test() -> MyEnum {
        //     let x = BoxTrait::new(MyEnum::A);
        //     x.unbox()
        // }
        let program = ProgramParser::new().parse(r#"
            type [2] = Box<[1]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [0] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
            type [1] = Enum<ut@program::program::MyEnum, [0], [0]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [4] = Const<[1], 0, [3]> [storable: false, drop: false, dup: false, zero_sized: false];
            type [3] = Const<[0]> [storable: false, drop: false, dup: false, zero_sized: false];

            libfunc [1] = const_as_box<[4], 0>;
            libfunc [0] = unbox<[1]>;
            libfunc [2] = store_temp<[1]>;

            [1]() -> ([0]); // 0
            [0]([0]) -> ([1]); // 1
            [2]([1]) -> ([1]); // 2
            return([1]); // 3

            [0]@0() -> ([1]);
        "#).map_err(|e| e.to_string()).unwrap();

        let return_value = run_sierra_program(&program, &[]).return_value;

        assert_eq!(
            return_value,
            Value::Enum {
                tag: 0,
                value: Box::new(Value::Struct {
                    fields: Vec::new(),
                    debug_name: None,
                }),
                debug_name: None,
            },
        );
    }

    #[test]
    fn box_unbox_stack_allocated_enum_c2() {
        // use core::box::BoxTrait;
        // enum MyEnum {
        //     A: (),
        //     B: (),
        // }
        // fn run_test() -> MyEnum {
        //     let x = BoxTrait::new(MyEnum::B);
        //     x.unbox()
        // }
        let program = ProgramParser::new().parse(r#"
            type [2] = Box<[1]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [0] = Struct<ut@Tuple> [storable: true, drop: true, dup: true, zero_sized: true];
            type [1] = Enum<ut@program::program::MyEnum, [0], [0]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [4] = Const<[1], 1, [3]> [storable: false, drop: false, dup: false, zero_sized: false];
            type [3] = Const<[0]> [storable: false, drop: false, dup: false, zero_sized: false];

            libfunc [1] = const_as_box<[4], 0>;
            libfunc [0] = unbox<[1]>;
            libfunc [2] = store_temp<[1]>;

            [1]() -> ([0]); // 0
            [0]([0]) -> ([1]); // 1
            [2]([1]) -> ([1]); // 2
            return([1]); // 3

            [0]@0() -> ([1]);
        "#).map_err(|e| e.to_string()).unwrap();

        let return_value = run_sierra_program(&program, &[]).return_value;

        assert_eq!(
            return_value,
            Value::Enum {
                tag: 1,
                value: Box::new(Value::Struct {
                    fields: Vec::new(),
                    debug_name: None,
                }),
                debug_name: None,
            },
        );
    }

    #[test]
    fn box_unbox_stack_allocated_enum() {
        // use core::box::BoxTrait;

        // enum MyEnum {
        //     A: felt252,
        //     B: u128,
        // }

        // fn run_test() -> MyEnum {
        //     let x = BoxTrait::new(MyEnum::A(1234));
        //     x.unbox()
        // }
        let program = ProgramParser::new().parse(r#"
            type [3] = Box<[2]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [0] = felt252 [storable: true, drop: true, dup: true, zero_sized: false];
            type [1] = u128 [storable: true, drop: true, dup: true, zero_sized: false];
            type [2] = Enum<ut@program::program::MyEnum, [0], [1]> [storable: true, drop: true, dup: true, zero_sized: false];
            type [5] = Const<[2], 0, [4]> [storable: false, drop: false, dup: false, zero_sized: false];
            type [4] = Const<[0], 1234> [storable: false, drop: false, dup: false, zero_sized: false];

            libfunc [1] = const_as_box<[5], 0>;
            libfunc [0] = unbox<[2]>;
            libfunc [2] = store_temp<[2]>;

            [1]() -> ([0]); // 0
            [0]([0]) -> ([1]); // 1
            [2]([1]) -> ([1]); // 2
            return([1]); // 3

            [0]@0() -> ([2]);
        "#).map_err(|e| e.to_string()).unwrap();

        let return_value = run_sierra_program(&program, &[]).return_value;

        assert_eq!(
            return_value,
            Value::Enum {
                tag: 0,
                value: Box::new(Value::Felt252(1234.into())),
                debug_name: None,
            },
        );
    }
}
