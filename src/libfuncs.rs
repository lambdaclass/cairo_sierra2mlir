//! # Compiler libfunc infrastructure
//!
//! Contains libfunc generation stuff (aka. the actual instructions).

use crate::{
    error::{panic::ToNativeAssertError, Error as CoreLibfuncBuilderError, Result},
    metadata::MetadataStorage,
    types::TypeBuilder,
    utils::BlockExt,
};
use bumpalo::Bump;
use cairo_lang_sierra::{
    extensions::{
        core::{CoreConcreteLibfunc, CoreLibfunc, CoreType, CoreTypeConcrete},
        int::{
            signed::{Sint16Traits, Sint32Traits, Sint64Traits, Sint8Traits},
            unsigned::{Uint16Traits, Uint32Traits, Uint64Traits, Uint8Traits},
        },
        lib_func::ParamSignature,
        starknet::StarkNetTypeConcrete,
        ConcreteLibfunc,
    },
    ids::FunctionId,
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::{arith, cf},
    ir::{Block, BlockLike, BlockRef, Location, Module, Region, Value},
    Context,
};
use num_bigint::BigInt;
use std::{cell::Cell, error::Error, ops::Deref};

mod array;
mod r#bool;
mod bounded_int;
mod r#box;
mod bytes31;
mod cast;
mod circuit;
mod r#const;
mod coupon;
mod debug;
mod drop;
mod dup;
mod ec;
mod r#enum;
mod felt252;
mod felt252_dict;
mod felt252_dict_entry;
mod function_call;
mod gas;
mod int;
mod int_range;
mod mem;
mod nullable;
mod pedersen;
mod poseidon;
mod starknet;
mod r#struct;
mod uint256;
mod uint512;

/// Generation of MLIR operations from their Sierra counterparts.
///
/// All possible Sierra libfuncs must implement it. It is already implemented for all the core
/// libfuncs, contained in [CoreConcreteLibfunc].
pub trait LibfuncBuilder {
    /// Error type returned by this trait's methods.
    type Error: Error;

    /// Generate the MLIR operations.
    fn build<'ctx, 'this>(
        &self,
        context: &'ctx Context,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        entry: &'this Block<'ctx>,
        location: Location<'ctx>,
        helper: &LibfuncHelper<'ctx, 'this>,
        metadata: &mut MetadataStorage,
    ) -> Result<()>;

    /// Return the target function if the statement is a function call.
    ///
    /// This is used by the compiler to check whether a statement is a function call and apply the
    /// tail recursion logic.
    fn is_function_call(&self) -> Option<&FunctionId>;
}

impl LibfuncBuilder for CoreConcreteLibfunc {
    type Error = CoreLibfuncBuilderError;

    fn build<'ctx, 'this>(
        &self,
        context: &'ctx Context,
        registry: &ProgramRegistry<CoreType, CoreLibfunc>,
        entry: &'this Block<'ctx>,
        location: Location<'ctx>,
        helper: &LibfuncHelper<'ctx, 'this>,
        metadata: &mut MetadataStorage,
    ) -> Result<()> {
        match self {
            Self::ApTracking(_) | Self::BranchAlign(_) | Self::UnconditionalJump(_) => {
                build_noop::<0, true>(
                    context,
                    registry,
                    entry,
                    location,
                    helper,
                    metadata,
                    self.param_signatures(),
                )
            }
            Self::Array(selector) => self::array::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Bool(selector) => self::r#bool::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::BoundedInt(info) => {
                self::bounded_int::build(context, registry, entry, location, helper, metadata, info)
            }
            Self::Box(selector) => self::r#box::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Bytes31(selector) => self::bytes31::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Cast(selector) => self::cast::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Circuit(info) => {
                self::circuit::build(context, registry, entry, location, helper, metadata, info)
            }
            Self::Const(selector) => self::r#const::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Coupon(selector) => self::coupon::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::CouponCall(info) => self::function_call::build(
                context, registry, entry, location, helper, metadata, info,
            ),
            Self::Debug(selector) => self::debug::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Drop(info) => {
                self::drop::build(context, registry, entry, location, helper, metadata, info)
            }
            Self::Dup(info) | Self::SnapshotTake(info) => {
                self::dup::build(context, registry, entry, location, helper, metadata, info)
            }
            Self::Ec(selector) => self::ec::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Enum(selector) => self::r#enum::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Felt252(selector) => self::felt252::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Felt252Dict(selector) => self::felt252_dict::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Felt252DictEntry(selector) => self::felt252_dict_entry::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::FunctionCall(info) => self::function_call::build(
                context, registry, entry, location, helper, metadata, info,
            ),
            Self::Gas(selector) => self::gas::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::IntRange(selector) => self::int_range::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Mem(selector) => self::mem::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Nullable(selector) => self::nullable::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Pedersen(selector) => self::pedersen::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Poseidon(selector) => self::poseidon::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Sint8(selector) => self::int::build_signed::<Sint8Traits>(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Sint16(selector) => self::int::build_signed::<Sint16Traits>(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Sint32(selector) => self::int::build_signed::<Sint32Traits>(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Sint64(selector) => self::int::build_signed::<Sint64Traits>(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Sint128(selector) => self::int::build_i128(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::StarkNet(selector) => self::starknet::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Struct(selector) => self::r#struct::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Uint8(selector) => self::int::build_unsigned::<Uint8Traits>(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Uint16(selector) => self::int::build_unsigned::<Uint16Traits>(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Uint32(selector) => self::int::build_unsigned::<Uint32Traits>(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Uint64(selector) => self::int::build_unsigned::<Uint64Traits>(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Uint128(selector) => self::int::build_u128(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Uint256(selector) => self::uint256::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Uint512(selector) => self::uint512::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::UnwrapNonZero(info) => build_noop::<1, true>(
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

    fn is_function_call(&self) -> Option<&FunctionId> {
        match self {
            CoreConcreteLibfunc::FunctionCall(info) => Some(&info.function.id),
            CoreConcreteLibfunc::CouponCall(info) => Some(&info.function.id),
            _ => None,
        }
    }
}

/// Helper struct which contains logic generation for extra MLIR blocks and branch operations to the
/// next statements.
///
/// Each branch index should be present in exactly one call a branching method (either
/// [`br`](#method.br) or [`cond_br`](#method.cond_br)).
///
/// This helper is necessary because the statement following the current one may not have the same
/// arguments as the results returned by the current statement. Because of that, a direct jump
/// cannot be made and some processing is required.
pub struct LibfuncHelper<'ctx, 'this>
where
    'this: 'ctx,
{
    pub module: &'this Module<'ctx>,
    pub init_block: &'this BlockRef<'ctx, 'this>,

    pub region: &'this Region<'ctx>,
    pub blocks_arena: &'this Bump,
    pub last_block: Cell<&'this BlockRef<'ctx, 'this>>,

    pub branches: Vec<(&'this Block<'ctx>, Vec<BranchArg<'ctx, 'this>>)>,
    pub results: Vec<Vec<Cell<Option<Value<'ctx, 'this>>>>>,

    #[cfg(feature = "with-profiler")]
    pub profiler: Option<(
        crate::metadata::profiler::ProfilerMeta,
        cairo_lang_sierra::program::StatementIdx,
        (Value<'ctx, 'this>, Value<'ctx, 'this>),
    )>,
}

impl<'ctx, 'this> LibfuncHelper<'ctx, 'this>
where
    'this: 'ctx,
{
    #[doc(hidden)]
    pub(crate) fn results(self) -> Result<Vec<Vec<Value<'ctx, 'this>>>> {
        self.results
            .into_iter()
            .enumerate()
            .map(|(branch_idx, x)| {
                x.into_iter()
                    .enumerate()
                    .map(|(arg_idx, x)| {
                        x.into_inner().to_native_assert_error(&format!(
                            "Argument #{arg_idx} of branch {branch_idx} doesn't have a value."
                        ))
                    })
                    .collect()
            })
            .collect()
    }

    /// Return the initialization block.
    ///
    /// The init block is used for `llvm.alloca` instructions. It is guaranteed to not be executed
    /// multiple times on tail-recursive functions. This property allows generating tail-recursive
    /// functions that do not grow the stack.
    pub fn init_block(&self) -> &'this Block<'ctx> {
        self.init_block
    }

    /// Inserts a new block after all the current libfunc's blocks.
    pub fn append_block(&self, block: Block<'ctx>) -> &'this Block<'ctx> {
        let block = self
            .region
            .insert_block_after(*self.last_block.get(), block);

        let block_ref: &'this mut BlockRef<'ctx, 'this> = self.blocks_arena.alloc(block);
        self.last_block.set(block_ref);

        block_ref
    }

    /// Creates an unconditional branching operation out of the libfunc and into the next statement.
    ///
    /// This method will also store the returned values so that they can be moved into the state and
    /// used later on when required.
    fn br(
        &self,
        block: &'this Block<'ctx>,
        branch: usize,
        results: &[Value<'ctx, 'this>],
        location: Location<'ctx>,
    ) -> Result<()> {
        let (successor, operands) = &self.branches[branch];

        for (dst, src) in self.results[branch].iter().zip(results) {
            dst.replace(Some(*src));
        }

        let destination_operands = operands
            .iter()
            .copied()
            .map(|op| match op {
                BranchArg::External(x) => x,
                BranchArg::Returned(i) => results[i],
            })
            .collect::<Vec<_>>();

        #[cfg(feature = "with-profiler")]
        self.push_profiler_frame(unsafe { self.context().to_ref() }, block, location)?;

        block.append_operation(cf::br(successor, &destination_operands, location));
        Ok(())
    }

    /// Creates a conditional binary branching operation, potentially jumping out of the libfunc and
    /// into the next statement.
    ///
    /// While generating a `cond_br` that doesn't jump out of the libfunc is possible, it should be
    /// avoided whenever possible. In those cases just use [melior::dialect::cf::cond_br].
    ///
    /// This method will also store the returned values so that they can be moved into the state and
    /// used later on when required.
    // TODO: Allow one block to be libfunc-internal.
    fn cond_br(
        &self,
        context: &'ctx Context,
        block: &'this Block<'ctx>,
        condition: Value<'ctx, 'this>,
        branches: [usize; 2],
        results: [&[Value<'ctx, 'this>]; 2],
        location: Location<'ctx>,
    ) -> Result<()> {
        let (block_true, args_true) = {
            let (successor, operands) = &self.branches[branches[0]];

            for (dst, src) in self.results[branches[0]].iter().zip(results[0]) {
                dst.replace(Some(*src));
            }

            let destination_operands = operands
                .iter()
                .copied()
                .map(|op| match op {
                    BranchArg::External(x) => x,
                    BranchArg::Returned(i) => results[0][i],
                })
                .collect::<Vec<_>>();

            (*successor, destination_operands)
        };

        let (block_false, args_false) = {
            let (successor, operands) = &self.branches[branches[1]];

            for (dst, src) in self.results[branches[1]].iter().zip(results[1]) {
                dst.replace(Some(*src));
            }

            let destination_operands = operands
                .iter()
                .copied()
                .map(|op| match op {
                    BranchArg::External(x) => x,
                    BranchArg::Returned(i) => results[1][i],
                })
                .collect::<Vec<_>>();

            (*successor, destination_operands)
        };

        #[cfg(feature = "with-profiler")]
        self.push_profiler_frame(context, block, location)?;

        block.append_operation(cf::cond_br(
            context,
            condition,
            block_true,
            block_false,
            &args_true,
            &args_false,
            location,
        ));
        Ok(())
    }

    #[cfg(feature = "with-profiler")]
    fn push_profiler_frame(
        &self,
        context: &'ctx Context,
        block: &'this Block<'ctx>,
        location: Location<'ctx>,
    ) -> Result<()> {
        if let Some((profiler_meta, statement_idx, t0)) = self.profiler.as_ref() {
            let t0 = *t0;
            let t1 = profiler_meta.measure_timestamp(context, block, location)?;

            profiler_meta.push_frame(context, block, statement_idx.0, t0, t1, location)?;
        }

        Ok(())
    }
}

impl<'ctx> Deref for LibfuncHelper<'ctx, '_> {
    type Target = Module<'ctx>;

    fn deref(&self) -> &Self::Target {
        self.module
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BranchArg<'ctx, 'this> {
    External(Value<'ctx, 'this>),
    Returned(usize),
}

fn increment_builtin_counter<'ctx: 'a, 'a>(
    context: &'ctx Context,
    block: &'ctx Block<'ctx>,
    location: Location<'ctx>,
    value: Value<'ctx, '_>,
) -> crate::error::Result<Value<'ctx, 'a>> {
    increment_builtin_counter_by(context, block, location, value, 1)
}

fn increment_builtin_counter_by<'ctx: 'a, 'a>(
    context: &'ctx Context,
    block: &'ctx Block<'ctx>,
    location: Location<'ctx>,
    value: Value<'ctx, '_>,
    amount: impl Into<BigInt>,
) -> crate::error::Result<Value<'ctx, 'a>> {
    block.append_op_result(arith::addi(
        value,
        block.const_int(context, location, amount, 64)?,
        location,
    ))
}

fn build_noop<'ctx, 'this, const N: usize, const PROCESS_BUILTINS: bool>(
    context: &'ctx Context,
    registry: &ProgramRegistry<CoreType, CoreLibfunc>,
    entry: &'this Block<'ctx>,
    location: Location<'ctx>,
    helper: &LibfuncHelper<'ctx, 'this>,
    _metadata: &mut MetadataStorage,
    param_signatures: &[ParamSignature],
) -> Result<()> {
    let mut params = Vec::with_capacity(N);

    #[allow(clippy::needless_range_loop)]
    for i in 0..N {
        let param_ty = registry.get_type(&param_signatures[i].ty)?;
        let mut param_val = entry.argument(i)?.into();

        if PROCESS_BUILTINS
            && param_ty.is_builtin()
            && !matches!(
                param_ty,
                CoreTypeConcrete::BuiltinCosts(_)
                    | CoreTypeConcrete::Coupon(_)
                    | CoreTypeConcrete::GasBuiltin(_)
                    | CoreTypeConcrete::StarkNet(StarkNetTypeConcrete::System(_))
            )
        {
            param_val = increment_builtin_counter(context, entry, location, param_val)?;
        }

        params.push(param_val);
    }

    helper.br(entry, 0, &params, location)
}
