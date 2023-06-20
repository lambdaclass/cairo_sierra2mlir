use crate::{metadata::MetadataStorage, types::TypeBuilder};
use cairo_lang_sierra::{
    extensions::{core::CoreConcreteLibfunc, GenericLibfunc, GenericType},
    ids::FunctionId,
    program_registry::ProgramRegistry,
};
use melior::{
    dialect::cf,
    ir::{Block, BlockRef, Location, Module, Operation, Region, Type, Value, ValueLike},
    Context,
};
use std::{
    borrow::Cow,
    cell::{Cell, RefCell},
    error::Error,
    mem::transmute,
    ops::Deref,
};

pub mod ap_tracking;
pub mod array;
pub mod bitwise;
pub mod r#bool;
pub mod r#box;
pub mod branch_align;
pub mod cast;
pub mod debug;
pub mod drop;
pub mod dup;
pub mod ec;
pub mod r#enum;
pub mod felt252;
pub mod felt252_dict;
pub mod felt252_dict_entry;
pub mod function_call;
pub mod gas;
pub mod mem;
pub mod nullable;
pub mod pedersen;
pub mod poseidon;
pub mod snapshot_take;
pub mod stark_net;
pub mod r#struct;
pub mod uint128;
pub mod uint16;
pub mod uint256;
pub mod uint32;
pub mod uint512;
pub mod uint64;
pub mod uint8;
pub mod unconditional_jump;
pub mod unwrap_non_zero;

pub trait LibfuncBuilder {
    type Error: Error;

    fn build<'ctx, 'this, TType, TLibfunc>(
        &self,
        context: &'ctx Context,
        registry: &ProgramRegistry<TType, TLibfunc>,
        entry: &'this Block<'ctx>,
        location: Location<'ctx>,
        helper: &LibfuncHelper<'ctx, 'this>,
        metadata: &mut MetadataStorage,
    ) -> Result<(), Self::Error>
    where
        TType: GenericType,
        TLibfunc: GenericLibfunc,
        <TType as GenericType>::Concrete: TypeBuilder,
        <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder;

    fn is_function_call(&self) -> Option<&FunctionId>;
}

impl LibfuncBuilder for CoreConcreteLibfunc {
    type Error = std::convert::Infallible;

    fn build<'ctx, 'this, TType, TLibfunc>(
        &self,
        context: &'ctx Context,
        registry: &ProgramRegistry<TType, TLibfunc>,
        entry: &'this Block<'ctx>,
        location: Location<'ctx>,
        helper: &LibfuncHelper<'ctx, 'this>,
        metadata: &mut MetadataStorage,
    ) -> Result<(), Self::Error>
    where
        TType: GenericType,
        TLibfunc: GenericLibfunc,
        <TType as GenericType>::Concrete: TypeBuilder,
        <TLibfunc as GenericLibfunc>::Concrete: LibfuncBuilder,
    {
        match self {
            Self::ApTracking(selector) => self::ap_tracking::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Array(_) => todo!(),
            Self::Bitwise(_) => todo!(),
            Self::BranchAlign(info) => self::branch_align::build(
                context, registry, entry, location, helper, metadata, info,
            ),
            Self::Bool(_) => todo!(),
            Self::Box(_) => todo!(),
            Self::Cast(_) => todo!(),
            Self::Drop(info) => {
                self::drop::build(context, registry, entry, location, helper, metadata, info)
            }
            Self::Dup(info) => {
                self::dup::build(context, registry, entry, location, helper, metadata, info)
            }
            Self::Ec(_) => todo!(),
            Self::Felt252(selector) => self::felt252::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::FunctionCall(info) => self::function_call::build(
                context, registry, entry, location, helper, metadata, info,
            ),
            Self::Gas(selector) => self::gas::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Uint8(selector) => self::uint8::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Uint16(_) => todo!(),
            Self::Uint32(selector) => self::uint32::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Uint64(_) => todo!(),
            Self::Uint128(_) => todo!(),
            Self::Uint256(_) => todo!(),
            Self::Uint512(_) => todo!(),
            Self::Mem(selector) => self::mem::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Nullable(_) => todo!(),
            Self::UnwrapNonZero(_) => todo!(),
            Self::UnconditionalJump(info) => self::unconditional_jump::build(
                context, registry, entry, location, helper, metadata, info,
            ),
            Self::Enum(selector) => self::r#enum::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Struct(selector) => self::r#struct::build(
                context, registry, entry, location, helper, metadata, selector,
            ),
            Self::Felt252Dict(_) => todo!(),
            Self::Felt252DictEntry(_) => todo!(),
            Self::Pedersen(_) => todo!(),
            Self::Poseidon(_) => todo!(),
            Self::StarkNet(_) => todo!(),
            Self::Debug(_) => todo!(),
            Self::SnapshotTake(_) => todo!(),
        }
    }

    fn is_function_call(&self) -> Option<&FunctionId> {
        match self {
            CoreConcreteLibfunc::FunctionCall(info) => Some(&info.function.id),
            _ => None,
        }
    }
}

pub struct LibfuncHelper<'ctx, 'this> {
    pub(crate) module: &'this Module<'ctx>,

    pub(crate) region: &'this Region<'ctx>,
    pub(crate) entry_block: &'this Block<'ctx>,
    pub(crate) extra_blocks: RefCell<Vec<BlockRef<'ctx, 'this>>>,

    pub(crate) branches: Vec<(&'this Block<'ctx>, Vec<BranchArg<'ctx, 'this>>)>,
    pub(crate) results: Vec<Vec<Cell<Option<Value<'ctx, 'this>>>>>,
}

impl<'ctx, 'this> LibfuncHelper<'ctx, 'this> {
    pub(crate) fn results(self) -> impl Iterator<Item = Vec<Value<'ctx, 'this>>> {
        self.results
            .into_iter()
            .map(|x| x.into_iter().map(|x| x.into_inner().unwrap()).collect())
    }

    pub fn append_block(&self, args: &[(Type<'ctx>, Location<'ctx>)]) -> &'this Block<'ctx> {
        let mut extra_blocks = self.extra_blocks.borrow_mut();

        let prev_block = extra_blocks
            .last()
            .copied()
            .unwrap_or_else(|| unsafe { transmute(self.entry_block) });
        extra_blocks.push(self.region.insert_block_after(prev_block, Block::new(args)));

        unsafe { transmute::<&Block, &'this Block<'ctx>>(extra_blocks.last().unwrap()) }
    }

    pub fn br(
        &self,
        branch: usize,
        results: &[Value<'ctx, 'this>],
        location: Location<'ctx>,
    ) -> Operation<'ctx> {
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

        cf::br(successor, &destination_operands, location)
    }

    // TODO: Allow one block to be libfunc-internal.
    pub fn cond_br(
        &self,
        condition: Value<'ctx, 'this>,
        branches: (usize, usize),
        results: &[Value<'ctx, 'this>],
        location: Location<'ctx>,
    ) -> Operation<'ctx> {
        let (block_true, args_true) = {
            let (successor, operands) = &self.branches[branches.0];

            for (dst, src) in self.results[branches.0].iter().zip(results) {
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

            (*successor, destination_operands)
        };

        let (block_false, args_false) = {
            let (successor, operands) = &self.branches[branches.1];

            for (dst, src) in self.results[branches.1].iter().zip(results) {
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

            (*successor, destination_operands)
        };

        cf::cond_br(
            unsafe { location.context().to_ref() },
            condition,
            block_true,
            block_false,
            &args_true,
            &args_false,
            location,
        )
    }

    pub fn switch(
        &self,
        flag: Value<'ctx, 'this>,
        default: (BranchTarget<'ctx, '_>, &[Value<'ctx, 'this>]),
        branches: &[(i64, BranchTarget<'ctx, '_>, &[Value<'ctx, 'this>])],
        location: Location<'ctx>,
    ) -> Operation<'ctx> {
        let default_destination = match default.0 {
            BranchTarget::Jump(x) => (x, Cow::Borrowed(default.1)),
            BranchTarget::Return(i) => {
                let (successor, operands) = &self.branches[i];

                for (dst, src) in self.results[i].iter().zip(default.1) {
                    dst.replace(Some(*src));
                }

                let destination_operands = operands
                    .iter()
                    .copied()
                    .map(|op| match op {
                        BranchArg::External(x) => x,
                        BranchArg::Returned(i) => default.1[i],
                    })
                    .collect::<Vec<_>>();

                (*successor, Cow::Owned(destination_operands))
            }
        };

        let mut case_values = Vec::with_capacity(branches.len());
        let mut case_destinations = Vec::with_capacity(branches.len());
        for (flag, successor, operands) in branches {
            case_values.push(*flag);

            case_destinations.push(match *successor {
                BranchTarget::Jump(x) => (x, Cow::Borrowed(*operands)),
                BranchTarget::Return(i) => {
                    let (successor, operands) = &self.branches[i];

                    for (dst, src) in self.results[i].iter().zip(default.1) {
                        dst.replace(Some(*src));
                    }

                    let destination_operands = operands
                        .iter()
                        .copied()
                        .map(|op| match op {
                            BranchArg::External(x) => x,
                            BranchArg::Returned(i) => default.1[i],
                        })
                        .collect::<Vec<_>>();

                    (*successor, Cow::Owned(destination_operands))
                }
            });
        }

        cf::switch(
            unsafe { location.context().to_ref() },
            &case_values,
            flag,
            flag.r#type(),
            (default_destination.0, &default_destination.1),
            &case_destinations
                .iter()
                .map(|(x, y)| (*x, y.as_ref()))
                .collect::<Vec<_>>(),
            location,
        )
        .unwrap()
    }
}

impl<'ctx, 'this> Deref for LibfuncHelper<'ctx, 'this> {
    type Target = Module<'ctx>;

    fn deref(&self) -> &Self::Target {
        self.module
    }
}

#[derive(Clone, Copy)]
pub enum BranchArg<'ctx, 'this> {
    External(Value<'ctx, 'this>),
    Returned(usize),
}

#[derive(Clone, Copy)]
pub enum BranchTarget<'ctx, 'a> {
    Jump(&'a Block<'ctx>),
    Return(usize),
}
