#![cfg(test)]

use cairo_lang_sierra::{
    extensions::{
        branch_align::BranchAlignLibfunc,
        core::CoreType,
        enm::{EnumInitLibfunc, EnumType},
        lib_func::{SierraApChange, SignatureSpecializationContext},
        structure::{StructConstructLibfunc, StructType},
        type_specialization_context::TypeSpecializationContext,
        types::TypeInfo,
        ConcreteType, GenericLibfunc, GenericType, NamedLibfunc, NamedType,
    },
    ids::{
        ConcreteLibfuncId, ConcreteTypeId, FunctionId, GenericLibfuncId, GenericTypeId, UserTypeId,
        VarId,
    },
    program::{
        BranchInfo, BranchTarget, ConcreteLibfuncLongId, DeclaredTypeInfo, Function,
        FunctionSignature, GenericArg, Invocation, LibfuncDeclaration, Param, Program, Statement,
        StatementIdx, TypeDeclaration,
    },
};
use std::{
    cell::{OnceCell, RefCell},
    iter::once,
    marker::PhantomData,
};

#[derive(Debug)]
pub struct SierraGenerator<T>
where
    T: GenericLibfunc,
{
    program: Program,
    phantom: PhantomData<T>,
}

impl<T> Default for SierraGenerator<T>
where
    T: GenericLibfunc,
{
    fn default() -> Self {
        Self {
            program: Program {
                type_declarations: Vec::new(),
                libfunc_declarations: Vec::new(),
                statements: Vec::new(),
                funcs: Vec::new(),
            },
            phantom: PhantomData,
        }
    }
}

impl<T> SierraGenerator<T>
where
    T: GenericLibfunc,
{
    pub fn build(self, generic_args: impl Into<Vec<GenericArg>>) -> Program {
        match T::supported_ids().as_slice() {
            [generic_id] => self.build_with_generic_id(generic_id.clone(), generic_args.into()),
            _ => panic!("multiple generic ids detected, please use build_with_generic_id directly"),
        }
    }

    pub fn build_with_generic_id(
        mut self,
        generic_id: GenericLibfuncId,
        generic_args: impl Into<Vec<GenericArg>>,
    ) -> Program {
        let generic_args = generic_args.into();
        let libfunc = T::by_id(&generic_id).unwrap();
        let libfunc_signature = libfunc
            .specialize_signature(
                &SierraGeneratorWrapper(RefCell::new(&mut self)),
                &generic_args,
            )
            .unwrap();

        // Push the libfunc declaration.
        let libfunc_id = self
            .push_libfunc_declaration(ConcreteLibfuncLongId {
                generic_id,
                generic_args: generic_args.to_vec(),
            })
            .clone();

        // Generate packed types.
        let num_builtins = libfunc_signature
            .param_signatures
            .iter()
            .take_while(|param_signature| {
                let long_id = &self
                    .program
                    .type_declarations
                    .iter()
                    .find(|type_declaration| type_declaration.id == param_signature.ty)
                    .unwrap()
                    .long_id;

                matches!(
                    long_id.generic_id.0.as_str(),
                    "Bitwise"
                        | "EcOp"
                        | "GasBuiltin"
                        | "BuiltinCosts"
                        | "RangeCheck"
                        | "RangeCheck96"
                        | "Pedersen"
                        | "Poseidon"
                        | "Coupon"
                        | "System"
                        | "SegmentArena"
                        | "AddMod"
                        | "MulMod"
                )
            })
            .count();

        let mut return_types = Vec::with_capacity(libfunc_signature.branch_signatures.len());
        let mut packed_unit_type_id = None;
        for branch_signature in &libfunc_signature.branch_signatures {
            assert!(branch_signature
                .vars
                .iter()
                .zip(libfunc_signature.param_signatures.iter().take(num_builtins))
                .all(|(lhs, rhs)| lhs.ty == rhs.ty));

            return_types.push(match branch_signature.vars.len() - num_builtins {
                0 => match libfunc_signature.branch_signatures.len() {
                    1 => ResultVarType::Empty(None),
                    _ => ResultVarType::Empty(Some(
                        packed_unit_type_id
                            .get_or_insert_with(|| {
                                self.push_type_declaration::<StructType>(&[GenericArg::UserType(
                                    UserTypeId::from_string("Tuple"),
                                )])
                                .clone()
                            })
                            .clone(),
                    )),
                },
                1 => ResultVarType::Single(branch_signature.vars[num_builtins].ty.clone()),
                _ => ResultVarType::Multi(
                    self.push_type_declaration::<StructType>(
                        once(GenericArg::UserType(UserTypeId::from_string("Tuple")))
                            .chain(
                                branch_signature
                                    .vars
                                    .iter()
                                    .skip(num_builtins)
                                    .map(|var_info| GenericArg::Type(var_info.ty.clone())),
                            )
                            .collect::<Vec<_>>(),
                    )
                    .clone(),
                ),
            });
        }

        // Generate switch type.
        let return_type = match return_types.len() {
            1 => match return_types[0].clone() {
                ResultVarType::Empty(ty) => ty.unwrap().clone(),
                ResultVarType::Single(ty) => ty.clone(),
                ResultVarType::Multi(ty) => ty.clone(),
            },
            _ => self
                .push_type_declaration::<EnumType>(
                    once(GenericArg::UserType(UserTypeId::from_string("Tuple")))
                        .chain(return_types.iter().map(|ty| {
                            GenericArg::Type(match ty {
                                ResultVarType::Empty(ty) => ty.clone().unwrap(),
                                ResultVarType::Single(ty) => ty.clone(),
                                ResultVarType::Multi(ty) => ty.clone(),
                            })
                        }))
                        .collect::<Vec<_>>(),
                )
                .clone(),
        };

        // Generate function declaration.
        self.program.funcs.push(Function {
            id: FunctionId::new(0),
            signature: FunctionSignature {
                param_types: libfunc_signature
                    .param_signatures
                    .iter()
                    .map(|param_signature| param_signature.ty.clone())
                    .collect(),
                ret_types: libfunc_signature.param_signatures[..num_builtins]
                    .iter()
                    .map(|param_signature| param_signature.ty.clone())
                    .chain(once(return_type.clone()))
                    .collect(),
            },
            params: libfunc_signature
                .param_signatures
                .iter()
                .enumerate()
                .map(|(idx, param_signature)| Param {
                    id: VarId::new(idx as u64),
                    ty: param_signature.ty.clone(),
                })
                .collect(),
            entry_point: StatementIdx(0),
        });

        // Generate statements.
        let mut libfunc_invocation = Invocation {
            libfunc_id,
            args: libfunc_signature
                .param_signatures
                .iter()
                .enumerate()
                .map(|(idx, _)| VarId::new(idx as u64))
                .collect(),
            branches: Vec::new(),
        };

        let branch_align_libfunc = OnceCell::new();
        let construct_unit_libfunc = packed_unit_type_id.map(|ty| {
            self.push_libfunc_declaration(ConcreteLibfuncLongId {
                generic_id: GenericLibfuncId::from_string(StructConstructLibfunc::STR_ID),
                generic_args: vec![GenericArg::Type(ty)],
            })
            .clone()
        });

        for (branch_index, branch_signature) in
            libfunc_signature.branch_signatures.iter().enumerate()
        {
            if libfunc_signature.branch_signatures.len() > 1 {
                let branch_align_libfunc_id = branch_align_libfunc
                    .get_or_init(|| {
                        self.push_libfunc_declaration(ConcreteLibfuncLongId {
                            generic_id: GenericLibfuncId::from_string(BranchAlignLibfunc::STR_ID),
                            generic_args: Vec::new(),
                        })
                        .clone()
                    })
                    .clone();
                self.program
                    .statements
                    .push(Statement::Invocation(Invocation {
                        libfunc_id: branch_align_libfunc_id,
                        args: Vec::new(),
                        branches: vec![BranchInfo {
                            target: BranchTarget::Fallthrough,
                            results: Vec::new(),
                        }],
                    }));
            }

            let branch_target = match branch_index {
                0 => BranchTarget::Fallthrough,
                _ => {
                    let statement_idx = StatementIdx(self.program.statements.len());
                    BranchTarget::Statement(statement_idx)
                }
            };

            // Maybe pack values.
            match &return_types[branch_index] {
                ResultVarType::Empty(Some(_)) => {
                    self.program
                        .statements
                        .push(Statement::Invocation(Invocation {
                            libfunc_id: construct_unit_libfunc.clone().unwrap(),
                            args: Vec::new(),
                            branches: vec![BranchInfo {
                                target: BranchTarget::Fallthrough,
                                results: vec![VarId::new(num_builtins as u64)],
                            }],
                        }));
                }
                ResultVarType::Multi(type_id) => {
                    let construct_libfunc_id = self
                        .push_libfunc_declaration(ConcreteLibfuncLongId {
                            generic_id: GenericLibfuncId::from_string(
                                StructConstructLibfunc::STR_ID,
                            ),
                            generic_args: vec![GenericArg::Type(type_id.clone())],
                        })
                        .clone();

                    self.program
                        .statements
                        .push(Statement::Invocation(Invocation {
                            libfunc_id: construct_libfunc_id,
                            args: (num_builtins..branch_signature.vars.len())
                                .map(|x| VarId::new(x as u64))
                                .collect(),
                            branches: vec![BranchInfo {
                                target: BranchTarget::Fallthrough,
                                results: vec![VarId::new(num_builtins as u64)],
                            }],
                        }));
                }
                _ => {}
            }

            // Maybe enum values.
            if libfunc_signature.branch_signatures.len() > 1 {
                let enum_libfunc_id = self
                    .push_libfunc_declaration(ConcreteLibfuncLongId {
                        generic_id: GenericLibfuncId::from_string(EnumInitLibfunc::STR_ID),
                        generic_args: vec![
                            GenericArg::Type(return_type.clone()),
                            GenericArg::Value(branch_index.into()),
                        ],
                    })
                    .clone();

                self.program
                    .statements
                    .push(Statement::Invocation(Invocation {
                        libfunc_id: enum_libfunc_id,
                        args: vec![VarId::new(num_builtins as u64)],
                        branches: vec![BranchInfo {
                            target: BranchTarget::Fallthrough,
                            results: vec![VarId::new(num_builtins as u64)],
                        }],
                    }));
            }

            // Return.
            self.program.statements.push(Statement::Return(
                (0..=num_builtins).map(|x| VarId::new(x as u64)).collect(),
            ));

            // Push the branch target.
            libfunc_invocation.branches.push(BranchInfo {
                target: branch_target,
                results: branch_signature
                    .vars
                    .iter()
                    .enumerate()
                    .map(|(idx, _)| VarId::new(idx as u64))
                    .collect(),
            });
        }

        self.program
            .statements
            .insert(0, Statement::Invocation(libfunc_invocation));

        self.program
    }

    pub fn push_type_declaration<U>(
        &mut self,
        generic_args: impl Into<Vec<GenericArg>>,
    ) -> &ConcreteTypeId
    where
        U: NamedType,
    {
        self.push_type_declaration_with_generic_id::<U>(U::ID, generic_args)
    }

    pub fn push_type_declaration_with_generic_id<U>(
        &mut self,
        generic_id: GenericTypeId,
        generic_args: impl Into<Vec<GenericArg>>,
    ) -> &ConcreteTypeId
    where
        U: GenericType,
    {
        let generic_args = generic_args.into();
        let type_info = U::by_id(&generic_id)
            .unwrap()
            .specialize(&SierraGeneratorWrapper(RefCell::new(self)), &generic_args)
            .unwrap()
            .info()
            .clone();
        let current_index = self
            .program
            .type_declarations
            .iter()
            .enumerate()
            .find_map(|(idx, type_decl)| (type_decl.long_id == type_info.long_id).then_some(idx));

        let current_index = current_index.unwrap_or_else(|| {
            let idx = self.program.type_declarations.len();
            self.program.type_declarations.push(TypeDeclaration {
                id: ConcreteTypeId::new(idx as u64),
                long_id: type_info.long_id,
                declared_type_info: Some(DeclaredTypeInfo {
                    storable: type_info.storable,
                    droppable: type_info.droppable,
                    duplicatable: type_info.duplicatable,
                    zero_sized: type_info.zero_sized,
                }),
            });
            idx
        });
        &self.program.type_declarations[current_index].id
    }

    fn push_libfunc_declaration(&mut self, long_id: ConcreteLibfuncLongId) -> &ConcreteLibfuncId {
        let id = ConcreteLibfuncId::new(self.program.libfunc_declarations.len() as u64);
        self.program
            .libfunc_declarations
            .push(LibfuncDeclaration { id, long_id });

        &self.program.libfunc_declarations.last().unwrap().id
    }
}

struct SierraGeneratorWrapper<'a, T>(RefCell<&'a mut SierraGenerator<T>>)
where
    T: GenericLibfunc;

impl<T> SignatureSpecializationContext for SierraGeneratorWrapper<'_, T>
where
    T: GenericLibfunc,
{
    fn try_get_concrete_type(
        &self,
        id: GenericTypeId,
        generic_args: &[GenericArg],
    ) -> Option<ConcreteTypeId> {
        Some(
            self.0
                .borrow_mut()
                .push_type_declaration_with_generic_id::<CoreType>(id, generic_args)
                .clone(),
        )
    }

    fn try_get_function_signature(&self, _function_id: &FunctionId) -> Option<FunctionSignature> {
        todo!()
    }

    fn try_get_function_ap_change(&self, _function_id: &FunctionId) -> Option<SierraApChange> {
        todo!()
    }

    fn as_type_specialization_context(&self) -> &dyn TypeSpecializationContext {
        self
    }
}

impl<T> TypeSpecializationContext for SierraGeneratorWrapper<'_, T>
where
    T: GenericLibfunc,
{
    fn try_get_type_info(&self, id: ConcreteTypeId) -> Option<TypeInfo> {
        self.0
            .borrow()
            .program
            .type_declarations
            .iter()
            .find_map(|type_declaration| {
                (type_declaration.id == id).then(|| {
                    let declared_type_info = type_declaration.declared_type_info.as_ref().unwrap();
                    TypeInfo {
                        long_id: type_declaration.long_id.clone(),
                        storable: declared_type_info.storable,
                        droppable: declared_type_info.droppable,
                        duplicatable: declared_type_info.duplicatable,
                        zero_sized: declared_type_info.zero_sized,
                    }
                })
            })
    }
}

#[derive(Clone)]
enum ResultVarType {
    Empty(Option<ConcreteTypeId>),
    Single(ConcreteTypeId),
    Multi(ConcreteTypeId),
}

#[cfg(test)]
mod test {
    use super::*;
    use cairo_lang_sierra::extensions::{
        array::ArrayNewLibfunc,
        bounded_int::BoundedIntIsZeroLibfunc,
        bytes31::Bytes31FromFelt252Trait,
        int::{
            signed::{Sint8Traits, SintDiffLibfunc},
            unsigned::{Uint32Type, Uint64Traits, Uint8Type},
            unsigned128::U128GuaranteeMulLibfunc,
            IntConstLibfunc,
        },
        try_from_felt252::TryFromFelt252Libfunc,
    };

    #[test]
    fn sierra_generator() {
        let program = SierraGenerator::<IntConstLibfunc<Uint64Traits>>::default()
            .build(&[GenericArg::Value(0.into())]);
        println!("{program}");
    }

    #[test]
    fn sierra_generator_multiret() {
        let program = SierraGenerator::<U128GuaranteeMulLibfunc>::default().build(&[]);
        println!("{program}");
    }

    #[test]
    fn sierra_generator_multibranch() {
        let program = SierraGenerator::<SintDiffLibfunc<Sint8Traits>>::default().build(&[]);
        println!("{program}");
    }

    #[test]
    fn sierra_generator_template() {
        let program = {
            let mut generator = SierraGenerator::<ArrayNewLibfunc>::default();

            let u8_type = generator.push_type_declaration::<Uint8Type>(&[]).clone();

            generator.build(&[GenericArg::Type(u8_type)])
        };
        println!("{program}");
    }

    #[test]
    fn sierra_generator_type_info() {
        let program = {
            let mut generator = SierraGenerator::<BoundedIntIsZeroLibfunc>::default();

            let u32_type = generator.push_type_declaration::<Uint32Type>(&[]).clone();

            generator.build(&[GenericArg::Type(u32_type)])
        };
        println!("{program}");
    }

    #[test]
    fn sierra_generator_branch_align() {
        let program =
            SierraGenerator::<TryFromFelt252Libfunc<Bytes31FromFelt252Trait>>::default().build(&[]);
        println!("{program}");
    }

    #[test]
    fn sierra_generator_type_generation() {
        let mut generator =
            SierraGenerator::<cairo_lang_sierra::extensions::array::ArrayGetLibfunc>::default();
        let u32_ty = generator.push_type_declaration::<Uint32Type>(&[]).clone();
        let array_ty = generator
            .push_type_declaration::<cairo_lang_sierra::extensions::array::ArrayType>(&[
                GenericArg::Type(u32_ty),
            ])
            .clone();

        let program = generator.build(&[GenericArg::Type(array_ty)]);
        println!("{program}");
    }
}
