use std::cmp::Ordering;

use proc_macro::TokenStream;
use quote::{format_ident, quote, ToTokens};
use syn::{
    parse_macro_input,
    punctuated::Punctuated,
    token::{Brace, Bracket, Paren, Pound},
    AngleBracketedGenericArguments, AssocType, Attribute, Block, DataStruct, DeriveInput, Expr,
    ExprArray, ExprCall, ExprMethodCall, ExprPath, FieldPat, FnArg, GenericArgument, GenericParam,
    Generics, Ident, ImplItem, ImplItemFn, ImplItemType, ItemImpl, ItemUse, Local, LocalInit, Pat,
    PatIdent, PatRest, PatStruct, PatTuple, PatTupleStruct, PatType, Path, PathArguments,
    PathSegment, PredicateType, QSelf, ReturnType, Signature, Stmt, Token, TraitBound,
    TraitBoundModifier, Type, TypeImplTrait, TypeParam, TypeParamBound, TypePath, TypeReference,
    WhereClause, WherePredicate,
};

fn derive_unit(ident: Ident) -> TokenStream {
    let fit_entity = fit_model();
    quote! {
        impl<Scalar: ::num_traits::Zero> #fit_entity for #ident {
            type ParamCount = ::typenum::uint::UTerm;

            fn evaluate(&self, _: &Scalar) -> Scalar {
                Scalar::zero()
            }

            fn jacobian(&self, _: &Scalar) -> impl ::core::convert::Into<::generic_array::GenericArray<Scalar, Self::ParamCount>> {
                []
            }

            fn set_params(&mut self, _: ::generic_array::GenericArray<Scalar, Self::ParamCount>) {}

            fn get_params(&self) -> impl ::core::convert::Into<::generic_array::GenericArray<Scalar, Self::ParamCount>> {
                []
            }
        }
    }
    .into()
}

fn id(name: &str) -> Ident {
    Ident::new(name, proc_macro2::Span::call_site())
}

#[inline]
fn scalar_ident() -> Ident {
    id("Scalar")
}

fn simple_abs_path<'l>(parts: impl IntoIterator<Item = &'l str>) -> Path {
    Path {
        leading_colon: Some(<Token![::] as Default>::default()),
        segments: parts
            .into_iter()
            .map(|name| PathSegment {
                ident: id(name),
                arguments: PathArguments::None,
            })
            .collect(),
    }
}

fn ident_type(i: Ident) -> Type {
    Type::Path(TypePath {
        qself: None,
        path: Path {
            leading_colon: None,
            segments: Punctuated::from_iter([PathSegment {
                arguments: PathArguments::None,
                ident: i,
            }]),
        },
    })
}

fn path_arg(arg: Type) -> PathArguments {
    PathArguments::AngleBracketed(AngleBracketedGenericArguments {
        colon2_token: None,
        lt_token: <Token![<] as Default>::default(),
        args: Punctuated::from_iter([GenericArgument::Type(arg)]),
        gt_token: <Token![>] as Default>::default(),
    })
}

#[inline]
fn uterm() -> Type {
    Type::Path(TypePath {
        qself: None,
        path: simple_abs_path(["typenum", "uint", "UTerm"]),
    })
}

fn q_self(ty: Type, position: usize) -> QSelf {
    QSelf {
        lt_token: <Token![<] as Default>::default(),
        ty: Box::new(ty),
        position,
        as_token: Some(<Token![as] as Default>::default()),
        gt_token: <Token![>] as Default>::default(),
    }
}

#[inline]
fn fit_model() -> Path {
    let mut path = simple_abs_path(["nacfahi", "models"]);
    path.segments.push(PathSegment {
        ident: id("FitModel"),
        arguments: path_arg(ident_type(scalar_ident())),
    });
    path
}

fn fit_entity_sub(ty: Type, sub: Ident) -> TypePath {
    let mut entity = fit_model();
    let params_pos = entity.segments.len(); // 5, presumably
    entity.segments.push(PathSegment {
        ident: sub,
        arguments: PathArguments::None,
    });
    TypePath {
        qself: Some(q_self(ty, params_pos)),
        path: entity,
    }
}

fn fit_entity_params(ty: Type) -> Type {
    Type::Path(fit_entity_sub(ty, id("ParamCount")))
}

fn core_op(op: Ident, rhs: Type) -> Path {
    Path {
        leading_colon: Some(<Token![::] as Default>::default()),
        segments: Punctuated::from_iter([
            PathSegment {
                ident: id("core"),
                arguments: syn::PathArguments::None,
            },
            PathSegment {
                ident: id("ops"),
                arguments: syn::PathArguments::None,
            },
            PathSegment {
                ident: op,
                arguments: path_arg(rhs),
            },
        ]),
    }
}

fn core_op_output(op: Ident, lhs: Type, rhs: Type) -> Type {
    let mut core_op = core_op(op, rhs);
    let output_pos = core_op.segments.len(); // 5, presumably
    core_op.segments.push(PathSegment {
        ident: id("Output"),
        arguments: PathArguments::None,
    });
    Type::Path(TypePath {
        qself: Some(q_self(lhs, output_pos)),
        path: core_op,
    })
}

fn type_sum_output(types: &[Type]) -> Type {
    if let Some((t, rest)) = types.split_last() {
        let rest_type = type_sum_output(rest);
        core_op_output(id("Add"), rest_type, fit_entity_params(t.clone()))
    } else {
        uterm()
    }
}

fn type_sum(types: &[Type]) -> Type {
    if let Some((t, rest)) = types.split_last() {
        let rest_type = type_sum(rest);
        let mut path = simple_abs_path(["typenum"]);
        path.segments.push(PathSegment {
            ident: id("Sum"),
            arguments: PathArguments::AngleBracketed(AngleBracketedGenericArguments {
                colon2_token: None,
                lt_token: <Token![<] as Default>::default(),
                args: Punctuated::from_iter([
                    GenericArgument::Type(rest_type),
                    GenericArgument::Type(fit_entity_params(t.clone())),
                ]),
                gt_token: <Token![>] as Default>::default(),
            }),
        });
        Type::Path(TypePath { qself: None, path })
    } else {
        uterm()
    }
}

fn type_sub_output(main_type: Type, types: &[Type]) -> Type {
    if let Some((t, rest)) = types.split_last() {
        let rest_type = type_sub_output(main_type, rest);
        core_op_output(id("Sub"), rest_type, fit_entity_params(t.clone()))
    } else {
        main_type
    }
}

fn fit_entity_bounds(types: &[Type]) -> impl IntoIterator<Item = WherePredicate> + '_ {
    types.iter().map(|t| {
        WherePredicate::Type(PredicateType {
            lifetimes: None,
            bounded_ty: t.clone(),
            colon_token: <Token![:] as Default>::default(),
            bounds: Punctuated::from_iter([TypeParamBound::Trait(TraitBound {
                paren_token: None,
                modifier: syn::TraitBoundModifier::None,
                lifetimes: None,
                path: fit_model(),
            })]),
        })
    })
}

fn add_bounds(types: &[Type], mut consumer: impl FnMut(WherePredicate)) {
    if let Some((t, rest)) = types.split_last() {
        consumer(WherePredicate::Type(PredicateType {
            lifetimes: None,
            bounded_ty: type_sum_output(types),
            colon_token: <Token![:] as Default>::default(),
            bounds: Punctuated::from_iter([TypeParamBound::Trait(TraitBound {
                paren_token: None,
                modifier: syn::TraitBoundModifier::None,
                lifetimes: None,
                path: simple_abs_path(["generic_array", "ArrayLength"]),
            })]),
        }));
        consumer(WherePredicate::Type(PredicateType {
            lifetimes: None,
            bounded_ty: type_sum_output(rest),
            colon_token: <Token![:] as Default>::default(),
            bounds: Punctuated::from_iter([TypeParamBound::Trait(TraitBound {
                paren_token: None,
                modifier: syn::TraitBoundModifier::None,
                lifetimes: None,
                path: core_op(id("Add"), fit_entity_params(t.clone())),
            })]),
        }));
        add_bounds(rest, consumer);
    }
}

fn sub_bounds(main_type: Type, types: &[Type], mut consumer: impl FnMut(WherePredicate)) {
    if let Some((t, rest)) = types.split_last() {
        consumer(WherePredicate::Type(PredicateType {
            lifetimes: None,
            bounded_ty: type_sub_output(main_type.clone(), rest),
            colon_token: <Token![:] as Default>::default(),
            bounds: Punctuated::from_iter([TypeParamBound::Trait(TraitBound {
                paren_token: None,
                modifier: syn::TraitBoundModifier::None,
                lifetimes: None,
                path: core_op(id("Sub"), fit_entity_params(t.clone())),
            })]),
        }));
        consumer(WherePredicate::Type(PredicateType {
            lifetimes: None,
            bounded_ty: type_sub_output(main_type.clone(), types),
            colon_token: <Token![:] as Default>::default(),
            bounds: Punctuated::from_iter([TypeParamBound::Trait(TraitBound {
                paren_token: None,
                modifier: syn::TraitBoundModifier::None,
                lifetimes: None,
                path: simple_abs_path(["generic_array", "ArrayLength"]),
            })]),
        }));
        sub_bounds(main_type, rest, consumer);
    }
}

fn bounds(types: &[Type], where_clause: Option<WhereClause>) -> WhereClause {
    let mut res = where_clause.unwrap_or_else(|| WhereClause {
        where_token: <Token![where] as Default>::default(),
        predicates: Punctuated::new(),
    });

    res.predicates.push(WherePredicate::Type(PredicateType {
        lifetimes: None,
        bounded_ty: ident_type(scalar_ident()),
        colon_token: <Token![:] as Default>::default(),
        bounds: Punctuated::from_iter([TypeParamBound::Trait(TraitBound {
            paren_token: None,
            modifier: syn::TraitBoundModifier::None,
            lifetimes: None,
            path: Path {
                leading_colon: None,
                segments: Punctuated::from_iter([
                    PathSegment {
                        ident: id("core"),
                        arguments: PathArguments::None,
                    },
                    PathSegment {
                        ident: id("ops"),
                        arguments: PathArguments::None,
                    },
                    PathSegment {
                        ident: id("Add"),
                        arguments: PathArguments::AngleBracketed(AngleBracketedGenericArguments {
                            colon2_token: None,
                            lt_token: <Token![<] as Default>::default(),
                            args: Punctuated::from_iter([
                                GenericArgument::Type(ident_type(scalar_ident())),
                                GenericArgument::AssocType(AssocType {
                                    ident: id("Output"),
                                    generics: None,
                                    eq_token: <Token![=] as Default>::default(),
                                    ty: ident_type(scalar_ident()),
                                }),
                            ]),
                            gt_token: <Token![>] as Default>::default(),
                        }),
                    },
                ]),
            },
        })]),
    }));

    res.predicates.extend(fit_entity_bounds(types));

    add_bounds(types, |pred| res.predicates.push(pred));

    let main_type = type_sum_output(types);
    sub_bounds(main_type, types, |pred| res.predicates.push(pred));

    res
}

fn fit_entity_evaluate(ty: Type) -> ExprPath {
    let TypePath { qself, path } = fit_entity_sub(ty, id("evaluate"));
    ExprPath {
        attrs: Vec::new(),
        qself,
        path,
    }
}

fn fit_entity_jacobian(ty: Type) -> ExprPath {
    let TypePath { qself, path } = fit_entity_sub(ty, id("jacobian"));
    ExprPath {
        attrs: Vec::new(),
        qself,
        path,
    }
}

fn fit_entity_set_params(ty: Type) -> ExprPath {
    let TypePath { qself, path } = fit_entity_sub(ty, id("set_params"));
    ExprPath {
        attrs: Vec::new(),
        qself,
        path,
    }
}

fn fit_entity_get_params(ty: Type) -> ExprPath {
    let TypePath { qself, path } = fit_entity_sub(ty, id("get_params"));
    ExprPath {
        attrs: Vec::new(),
        qself,
        path,
    }
}

fn ident_expr(ident: Ident) -> Expr {
    Expr::Path(ExprPath {
        attrs: Vec::new(),
        qself: None,
        path: Path {
            leading_colon: None,
            segments: Punctuated::from_iter([PathSegment {
                ident,
                arguments: PathArguments::None,
            }]),
        },
    })
}

fn call_function(func: ExprPath, args: impl IntoIterator<Item = Ident>) -> ExprCall {
    ExprCall {
        attrs: Vec::new(),
        func: Box::new(Expr::Path(func)),
        paren_token: Paren::default(),
        args: args.into_iter().map(ident_expr).collect(),
    }
}

fn add_function_path(lhs: Type, rhs: Type) -> ExprPath {
    let mut add_path = core_op(id("Add"), rhs);
    let qself_pos = add_path.segments.len();
    add_path.segments.push(PathSegment {
        ident: id("add"),
        arguments: PathArguments::None,
    });
    ExprPath {
        attrs: Vec::new(),
        qself: Some(q_self(lhs, qself_pos)),
        path: add_path,
    }
}

fn evaluate_body(idents: &[Ident], types: &[Type], destruction_syntax: &Local) -> Vec<Stmt> {
    let mut operands = core::iter::zip(idents, types).map(|(ident, ty)| {
        Expr::Call(call_function(
            fit_entity_evaluate(ty.clone()),
            [ident.clone(), id("x")],
        ))
    });
    let Some(first_expr) = operands.next() else {
        return Vec::new();
    };
    let mut stmts = Vec::new();
    stmts.push(Stmt::Local(destruction_syntax.clone()));
    let mut sum = first_expr;
    for operand in operands {
        sum = Expr::Call(ExprCall {
            attrs: Vec::new(),
            func: Box::new(Expr::Path(add_function_path(
                ident_type(scalar_ident()),
                ident_type(scalar_ident()),
            ))),
            paren_token: Paren::default(),
            args: Punctuated::from_iter([sum, operand]),
        });
    }
    stmts.push(Stmt::Expr(sum, None));
    stmts
}

fn generic_array_concat() -> Path {
    simple_abs_path(["generic_array", "GenericArray", "concat"])
}

fn let_local(ident: Ident, expr: Expr) -> Local {
    Local {
        attrs: Vec::new(),
        let_token: <Token![let] as Default>::default(),
        pat: Pat::Ident(PatIdent {
            attrs: Vec::new(),
            by_ref: None,
            mutability: None,
            ident,
            subpat: None,
        }),
        init: Some(LocalInit {
            eq_token: <Token![=] as Default>::default(),
            expr: Box::new(expr),
            diverge: None,
        }),
        semi_token: <Token![;] as Default>::default(),
    }
}

fn into_call(receiver: Expr) -> Expr {
    Expr::MethodCall(ExprMethodCall {
        attrs: Vec::new(),
        receiver: Box::new(receiver),
        dot_token: <Token![.] as Default>::default(),
        method: id("into"),
        turbofish: None,
        paren_token: Paren::default(),
        args: Punctuated::default(),
    })
}

fn jacobian_body(idents: &[Ident], types: &[Type], destruction_syntax: &Local) -> Vec<Stmt> {
    let mut stmts = Vec::new();
    stmts.push(Stmt::Item(syn::Item::Use(ItemUse {
        attrs: Vec::new(),
        vis: syn::Visibility::Inherited,
        use_token: <Token![use] as Default>::default(),
        leading_colon: Some(<Token![::] as Default>::default()),
        tree: syn::UseTree::Path(syn::UsePath {
            ident: id("generic_array"),
            colon2_token: <Token![::] as Default>::default(),
            tree: Box::new(syn::UseTree::Path(syn::UsePath {
                ident: id("sequence"),
                colon2_token: <Token![::] as Default>::default(),
                tree: Box::new(syn::UseTree::Name(syn::UseName {
                    ident: id("Concat"),
                })),
            })),
        }),
        semi_token: <Token![;] as Default>::default(),
    })));
    stmts.push(Stmt::Local(destruction_syntax.clone()));
    stmts.push(Stmt::Local(let_local(
        id("keep"),
        into_call(Expr::Array(ExprArray {
            attrs: Vec::new(),
            bracket_token: Bracket::default(),
            elems: Punctuated::new(),
        })),
    )));

    for (ident, ty) in core::iter::zip(idents, types) {
        stmts.push(Stmt::Local(let_local(
            id("part_jacobian"),
            into_call(Expr::Call(call_function(
                fit_entity_jacobian(ty.clone()),
                [ident.clone(), id("x")],
            ))),
        )));
        stmts.push(Stmt::Local(let_local(
            id("keep"),
            Expr::Call(call_function(
                ExprPath {
                    attrs: Vec::new(),
                    qself: None,
                    path: generic_array_concat(),
                },
                [id("keep"), id("part_jacobian")],
            )),
        )));
    }
    stmts.push(Stmt::Expr(ident_expr(id("keep")), None));
    stmts
}

fn generic_array_split() -> Path {
    simple_abs_path(["generic_array", "GenericArray", "split"])
}

fn set_params_body(idents: &[Ident], types: &[Type], destruction_syntax: &Local) -> Vec<Stmt> {
    let mut stmts = Vec::new();
    stmts.push(Stmt::Item(syn::Item::Use(ItemUse {
        attrs: Vec::new(),
        vis: syn::Visibility::Inherited,
        use_token: <Token![use] as Default>::default(),
        leading_colon: Some(<Token![::] as Default>::default()),
        tree: syn::UseTree::Path(syn::UsePath {
            ident: id("generic_array"),
            colon2_token: <Token![::] as Default>::default(),
            tree: Box::new(syn::UseTree::Path(syn::UsePath {
                ident: id("sequence"),
                colon2_token: <Token![::] as Default>::default(),
                tree: Box::new(syn::UseTree::Name(syn::UseName { ident: id("Split") })),
            })),
        }),
        semi_token: <Token![;] as Default>::default(),
    })));
    stmts.push(Stmt::Local(destruction_syntax.clone()));

    for (ident, ty) in core::iter::zip(idents, types) {
        stmts.push(Stmt::Local(Local {
            attrs: Vec::new(),
            let_token: <Token![let] as Default>::default(),
            pat: Pat::Tuple(PatTuple {
                attrs: Vec::new(),
                paren_token: Paren::default(),
                elems: Punctuated::from_iter([
                    Pat::Ident(PatIdent {
                        attrs: Vec::new(),
                        by_ref: None,
                        mutability: None,
                        ident: id("pars"),
                        subpat: None,
                    }),
                    Pat::Ident(PatIdent {
                        attrs: Vec::new(),
                        by_ref: None,
                        mutability: None,
                        ident: id("keep"),
                        subpat: None,
                    }),
                ]),
            }),
            init: Some(LocalInit {
                eq_token: <Token![=] as Default>::default(),
                expr: Box::new(Expr::Call(call_function(
                    ExprPath {
                        attrs: Vec::new(),
                        qself: None,
                        path: generic_array_split(),
                    },
                    [id("keep")],
                ))),
                diverge: None,
            }),
            semi_token: <Token![;] as Default>::default(),
        }));
        stmts.push(Stmt::Expr(
            Expr::Call(call_function(
                fit_entity_set_params(ty.clone()),
                [ident.clone(), id("pars")],
            )),
            Some(<Token![;] as Default>::default()),
        ));
    }
    stmts
}

fn get_params_body(idents: &[Ident], types: &[Type], destruction_syntax: &Local) -> Vec<Stmt> {
    let mut stmts = Vec::new();
    stmts.push(Stmt::Item(syn::Item::Use(ItemUse {
        attrs: Vec::new(),
        vis: syn::Visibility::Inherited,
        use_token: <Token![use] as Default>::default(),
        leading_colon: Some(<Token![::] as Default>::default()),
        tree: syn::UseTree::Path(syn::UsePath {
            ident: id("generic_array"),
            colon2_token: <Token![::] as Default>::default(),
            tree: Box::new(syn::UseTree::Path(syn::UsePath {
                ident: id("sequence"),
                colon2_token: <Token![::] as Default>::default(),
                tree: Box::new(syn::UseTree::Name(syn::UseName {
                    ident: id("Concat"),
                })),
            })),
        }),
        semi_token: <Token![;] as Default>::default(),
    })));
    stmts.push(Stmt::Local(destruction_syntax.clone()));
    stmts.push(Stmt::Local(let_local(
        id("keep"),
        into_call(Expr::Array(ExprArray {
            attrs: Vec::new(),
            bracket_token: Bracket::default(),
            elems: Punctuated::new(),
        })),
    )));

    for (ident, ty) in core::iter::zip(idents, types) {
        stmts.push(Stmt::Local(let_local(
            id("part_params"),
            into_call(Expr::Call(call_function(
                fit_entity_get_params(ty.clone()),
                [ident.clone()],
            ))),
        )));
        stmts.push(Stmt::Local(let_local(
            id("keep"),
            Expr::Call(call_function(
                ExprPath {
                    attrs: Vec::new(),
                    qself: None,
                    path: generic_array_concat(),
                },
                [id("keep"), id("part_params")],
            )),
        )));
    }
    stmts.push(Stmt::Expr(ident_expr(id("keep")), None));
    stmts
}

fn self_generic_array() -> Type {
    Type::Path(TypePath {
        qself: None,
        path: Path {
            leading_colon: Some(<Token![::] as Default>::default()),
            segments: Punctuated::from_iter([
                PathSegment {
                    ident: id("generic_array"),
                    arguments: PathArguments::None,
                },
                PathSegment {
                    ident: id("GenericArray"),
                    arguments: PathArguments::AngleBracketed(AngleBracketedGenericArguments {
                        colon2_token: None,
                        lt_token: <Token![<] as Default>::default(),
                        args: Punctuated::from_iter([
                            GenericArgument::Type(ident_type(scalar_ident())),
                            GenericArgument::Type(Type::Path(TypePath {
                                qself: None,
                                path: Path {
                                    leading_colon: None,
                                    segments: Punctuated::from_iter([
                                        PathSegment {
                                            ident: id("Self"),
                                            arguments: PathArguments::None,
                                        },
                                        PathSegment {
                                            ident: id("ParamCount"),
                                            arguments: PathArguments::None,
                                        },
                                    ]),
                                },
                            })),
                        ]),
                        gt_token: <Token![>] as Default>::default(),
                    }),
                },
            ]),
        },
    })
}

#[allow(clippy::too_many_lines)]
fn derive_inner(
    struct_ident: Ident,
    mut generics: Generics,
    destruction_syntax: Local,
    field_idents: &[Ident],
    field_types: &[Type],
) -> ItemImpl {
    let type_params: Vec<_> = generics.params.iter().cloned().collect();
    // if there are arguments, use them in type description
    let impl_type_args = if type_params.is_empty() {
        PathArguments::None
    } else {
        PathArguments::AngleBracketed(AngleBracketedGenericArguments {
            colon2_token: None,
            lt_token: <Token![<] as Default>::default(),
            args: type_params
                .into_iter()
                .map(|gp| match gp {
                    GenericParam::Lifetime(lt) => GenericArgument::Lifetime(lt.lifetime),
                    GenericParam::Type(tp) => GenericArgument::Type(ident_type(tp.ident)),
                    GenericParam::Const(cs) => GenericArgument::Const(ident_expr(cs.ident)),
                })
                .collect(),
            gt_token: <Token![>] as Default>::default(),
        })
    };
    // add scalar generic, if there's no parameter with same ident
    if !generics
        .params
        .iter()
        .any(|p| matches!(p, GenericParam::Type(t) if t.ident == scalar_ident()))
    {
        generics.params.insert(
            0,
            GenericParam::Type(TypeParam {
                attrs: Vec::new(),
                ident: scalar_ident(),
                colon_token: None,
                bounds: Punctuated::new(),
                eq_token: None,
                default: None,
            }),
        );
    }
    // add the bounds
    generics.where_clause = Some(bounds(field_types, generics.where_clause.take()));
    // param count
    let param_count = ImplItem::Type(ImplItemType {
        attrs: Vec::new(),
        vis: syn::Visibility::Inherited,
        defaultness: None,
        type_token: <Token![type] as Default>::default(),
        ident: id("ParamCount"),
        generics: Generics::default(),
        eq_token: <Token![=] as Default>::default(),
        ty: type_sum(field_types),
        semi_token: <Token![;] as Default>::default(),
    });
    // create method impls
    let evaluate_impl = ImplItem::Fn(ImplItemFn {
        attrs: vec![Attribute {
            pound_token: Pound::default(),
            style: syn::AttrStyle::Outer,
            bracket_token: Bracket::default(),
            meta: syn::Meta::Path(Path {
                leading_colon: None,
                segments: Punctuated::from_iter([PathSegment {
                    ident: id("inline"),
                    arguments: PathArguments::None,
                }]),
            }),
        }],
        vis: syn::Visibility::Inherited,
        defaultness: None,
        sig: Signature {
            constness: None,
            asyncness: None,
            unsafety: None,
            abi: None,
            fn_token: <Token![fn] as Default>::default(),
            ident: id("evaluate"),
            generics: Generics::default(),
            paren_token: Paren::default(),
            inputs: Punctuated::from_iter([
                FnArg::Typed(PatType {
                    attrs: Vec::new(),
                    pat: Box::new(Pat::Ident(PatIdent {
                        attrs: Vec::new(),
                        by_ref: None,
                        mutability: None,
                        ident: id("self"),
                        subpat: None,
                    })),
                    colon_token: <Token![:] as Default>::default(),
                    ty: Box::new(Type::Reference(TypeReference {
                        and_token: <Token![&] as Default>::default(),
                        lifetime: None,
                        mutability: None,
                        elem: Box::new(ident_type(id("Self"))),
                    })),
                }),
                FnArg::Typed(PatType {
                    attrs: Vec::new(),
                    pat: Box::new(Pat::Ident(PatIdent {
                        attrs: Vec::new(),
                        by_ref: None,
                        mutability: None,
                        ident: id("x"),
                        subpat: None,
                    })),
                    colon_token: <Token![:] as Default>::default(),
                    ty: Box::new(Type::Reference(TypeReference {
                        and_token: <Token![&] as Default>::default(),
                        lifetime: None,
                        mutability: None,
                        elem: Box::new(ident_type(scalar_ident())),
                    })),
                }),
            ]),
            variadic: None,
            output: ReturnType::Type(
                <Token![->] as Default>::default(),
                Box::new(ident_type(scalar_ident())),
            ),
        },
        block: Block {
            brace_token: Brace::default(),
            stmts: evaluate_body(field_idents, field_types, &destruction_syntax),
        },
    });
    let jacobian_impl = ImplItem::Fn(ImplItemFn {
        attrs: vec![Attribute {
            pound_token: Pound::default(),
            style: syn::AttrStyle::Outer,
            bracket_token: Bracket::default(),
            meta: syn::Meta::Path(Path {
                leading_colon: None,
                segments: Punctuated::from_iter([PathSegment {
                    ident: id("inline"),
                    arguments: PathArguments::None,
                }]),
            }),
        }],
        vis: syn::Visibility::Inherited,
        defaultness: None,
        sig: Signature {
            constness: None,
            asyncness: None,
            unsafety: None,
            abi: None,
            fn_token: <Token![fn] as Default>::default(),
            ident: id("jacobian"),
            generics: Generics::default(),
            paren_token: Paren::default(),
            inputs: Punctuated::from_iter([
                FnArg::Typed(PatType {
                    attrs: Vec::new(),
                    pat: Box::new(Pat::Ident(PatIdent {
                        attrs: Vec::new(),
                        by_ref: None,
                        mutability: None,
                        ident: id("self"),
                        subpat: None,
                    })),
                    colon_token: <Token![:] as Default>::default(),
                    ty: Box::new(Type::Reference(TypeReference {
                        and_token: <Token![&] as Default>::default(),
                        lifetime: None,
                        mutability: None,
                        elem: Box::new(ident_type(id("Self"))),
                    })),
                }),
                FnArg::Typed(PatType {
                    attrs: Vec::new(),
                    pat: Box::new(Pat::Ident(PatIdent {
                        attrs: Vec::new(),
                        by_ref: None,
                        mutability: None,
                        ident: id("x"),
                        subpat: None,
                    })),
                    colon_token: <Token![:] as Default>::default(),
                    ty: Box::new(Type::Reference(TypeReference {
                        and_token: <Token![&] as Default>::default(),
                        lifetime: None,
                        mutability: None,
                        elem: Box::new(ident_type(scalar_ident())),
                    })),
                }),
            ]),
            variadic: None,
            output: ReturnType::Type(
                <Token![->] as Default>::default(),
                Box::new(Type::ImplTrait(TypeImplTrait {
                    impl_token: <Token![impl] as Default>::default(),
                    bounds: Punctuated::from_iter([TypeParamBound::Trait(TraitBound {
                        paren_token: None,
                        modifier: TraitBoundModifier::None,
                        lifetimes: None,
                        path: Path {
                            leading_colon: Some(<Token![::] as Default>::default()),
                            segments: Punctuated::from_iter([
                                PathSegment {
                                    ident: id("core"),
                                    arguments: PathArguments::None,
                                },
                                PathSegment {
                                    ident: id("convert"),
                                    arguments: PathArguments::None,
                                },
                                PathSegment {
                                    ident: id("Into"),
                                    arguments: PathArguments::AngleBracketed(
                                        AngleBracketedGenericArguments {
                                            colon2_token: None,
                                            lt_token: <Token![<] as Default>::default(),
                                            args: Punctuated::from_iter([GenericArgument::Type(
                                                self_generic_array(),
                                            )]),
                                            gt_token: <Token![>] as Default>::default(),
                                        },
                                    ),
                                },
                            ]),
                        },
                    })]),
                })),
            ),
        },
        block: Block {
            brace_token: Brace::default(),
            stmts: jacobian_body(field_idents, field_types, &destruction_syntax),
        },
    });
    let set_params_impl = ImplItem::Fn(ImplItemFn {
        attrs: vec![Attribute {
            pound_token: Pound::default(),
            style: syn::AttrStyle::Outer,
            bracket_token: Bracket::default(),
            meta: syn::Meta::Path(Path {
                leading_colon: None,
                segments: Punctuated::from_iter([PathSegment {
                    ident: id("inline"),
                    arguments: PathArguments::None,
                }]),
            }),
        }],
        vis: syn::Visibility::Inherited,
        defaultness: None,
        sig: Signature {
            constness: None,
            asyncness: None,
            unsafety: None,
            abi: None,
            fn_token: <Token![fn] as Default>::default(),
            ident: id("set_params"),
            generics: Generics::default(),
            paren_token: Paren::default(),
            inputs: Punctuated::from_iter([
                FnArg::Typed(PatType {
                    attrs: Vec::new(),
                    pat: Box::new(Pat::Ident(PatIdent {
                        attrs: Vec::new(),
                        by_ref: None,
                        mutability: None,
                        ident: id("self"),
                        subpat: None,
                    })),
                    colon_token: <Token![:] as Default>::default(),
                    ty: Box::new(Type::Reference(TypeReference {
                        and_token: <Token![&] as Default>::default(),
                        lifetime: None,
                        mutability: Some(<Token![mut] as Default>::default()),
                        elem: Box::new(ident_type(id("Self"))),
                    })),
                }),
                FnArg::Typed(PatType {
                    attrs: Vec::new(),
                    pat: Box::new(Pat::Ident(PatIdent {
                        attrs: Vec::new(),
                        by_ref: None,
                        mutability: None,
                        ident: id("keep"),
                        subpat: None,
                    })),
                    colon_token: <Token![:] as Default>::default(),
                    ty: Box::new(self_generic_array()),
                }),
            ]),
            variadic: None,
            output: ReturnType::Default,
        },
        block: Block {
            brace_token: Brace::default(),
            stmts: set_params_body(field_idents, field_types, &destruction_syntax),
        },
    });
    let get_params_impl = ImplItem::Fn(ImplItemFn {
        attrs: vec![Attribute {
            pound_token: Pound::default(),
            style: syn::AttrStyle::Outer,
            bracket_token: Bracket::default(),
            meta: syn::Meta::Path(Path {
                leading_colon: None,
                segments: Punctuated::from_iter([PathSegment {
                    ident: id("inline"),
                    arguments: PathArguments::None,
                }]),
            }),
        }],
        vis: syn::Visibility::Inherited,
        defaultness: None,
        sig: Signature {
            constness: None,
            asyncness: None,
            unsafety: None,
            abi: None,
            fn_token: <Token![fn] as Default>::default(),
            ident: id("get_params"),
            generics: Generics::default(),
            paren_token: Paren::default(),
            inputs: Punctuated::from_iter([FnArg::Typed(PatType {
                attrs: Vec::new(),
                pat: Box::new(Pat::Ident(PatIdent {
                    attrs: Vec::new(),
                    by_ref: None,
                    mutability: None,
                    ident: id("self"),
                    subpat: None,
                })),
                colon_token: <Token![:] as Default>::default(),
                ty: Box::new(Type::Reference(TypeReference {
                    and_token: <Token![&] as Default>::default(),
                    lifetime: None,
                    mutability: None,
                    elem: Box::new(ident_type(id("Self"))),
                })),
            })]),
            variadic: None,
            output: ReturnType::Type(
                <Token![->] as Default>::default(),
                Box::new(Type::ImplTrait(TypeImplTrait {
                    impl_token: <Token![impl] as Default>::default(),
                    bounds: Punctuated::from_iter([TypeParamBound::Trait(TraitBound {
                        paren_token: None,
                        modifier: TraitBoundModifier::None,
                        lifetimes: None,
                        path: Path {
                            leading_colon: Some(<Token![::] as Default>::default()),
                            segments: Punctuated::from_iter([
                                PathSegment {
                                    ident: id("core"),
                                    arguments: PathArguments::None,
                                },
                                PathSegment {
                                    ident: id("convert"),
                                    arguments: PathArguments::None,
                                },
                                PathSegment {
                                    ident: id("Into"),
                                    arguments: PathArguments::AngleBracketed(
                                        AngleBracketedGenericArguments {
                                            colon2_token: None,
                                            lt_token: <Token![<] as Default>::default(),
                                            args: Punctuated::from_iter([GenericArgument::Type(
                                                self_generic_array(),
                                            )]),
                                            gt_token: <Token![>] as Default>::default(),
                                        },
                                    ),
                                },
                            ]),
                        },
                    })]),
                })),
            ),
        },
        block: Block {
            brace_token: Brace::default(),
            stmts: get_params_body(field_idents, field_types, &destruction_syntax),
        },
    });
    ItemImpl {
        attrs: Vec::new(),
        defaultness: None,
        unsafety: None,
        impl_token: <Token![impl] as Default>::default(),
        generics,
        trait_: Some((None, fit_model(), <Token![for] as Default>::default())),
        self_ty: Box::new(Type::Path(TypePath {
            qself: None,
            path: Path {
                leading_colon: None,
                segments: Punctuated::from_iter([PathSegment {
                    ident: struct_ident.clone(),
                    arguments: impl_type_args,
                }]),
            },
        })),
        brace_token: Brace::default(),
        items: vec![
            param_count,
            evaluate_impl,
            jacobian_impl,
            set_params_impl,
            get_params_impl,
        ],
    }
}

fn skip_ident() -> Ident {
    format_ident!("_")
}

fn tuple_ident(i: usize) -> Ident {
    format_ident!("field_{i}")
}

fn tuple_idents_all(inds: impl IntoIterator<Item = usize>) -> impl IntoIterator<Item = Ident> {
    struct It<I> {
        inner: I,
        memo: Option<usize>,
        planned: usize,
    }
    impl<I> It<I> {
        fn new(inner: impl IntoIterator<Item = usize, IntoIter = I>) -> Self {
            Self {
                inner: inner.into_iter(),
                memo: None,
                planned: 0,
            }
        }
    }
    impl<I: Iterator<Item = usize>> Iterator for It<I> {
        type Item = Ident;

        fn next(&mut self) -> Option<Self::Item> {
            let planned = self.planned;
            self.planned += 1;
            // - take memo content
            // - if there's nothing, try getting next iterator element
            // - if there's still nothing, iterator was exhausted
            let next = self.memo.take().or_else(|| self.inner.next())?;

            match next.cmp(&planned) {
                Ordering::Greater => {
                    // should not be yielded yet
                    // - save it
                    // - yield the empty element
                    self.memo = Some(next);
                    Some(skip_ident())
                }
                Ordering::Equal => {
                    // should be yielded (memory was wiped already)
                    Some(tuple_ident(next))
                }
                Ordering::Less => unreachable!(),
            }
        }
    }
    It::new(inds)
}

fn destruct_tuple(struct_ident: &Ident, inds: impl IntoIterator<Item = usize>) -> Local {
    Local {
        attrs: Vec::new(),
        let_token: <Token![let] as Default>::default(),
        pat: Pat::TupleStruct(PatTupleStruct {
            attrs: Vec::new(),
            paren_token: Paren::default(),
            elems: tuple_idents_all(inds)
                .into_iter()
                .map(|ident| {
                    Pat::Ident(PatIdent {
                        attrs: Vec::new(),
                        by_ref: None,
                        mutability: None,
                        ident,
                        subpat: None,
                    })
                })
                .chain(core::iter::once(Pat::Rest(PatRest {
                    attrs: Vec::new(),
                    dot2_token: <Token![..] as Default>::default(),
                })))
                .collect(),
            qself: None,
            path: Path {
                leading_colon: None,
                segments: Punctuated::from_iter([PathSegment {
                    ident: struct_ident.clone(),
                    arguments: PathArguments::None,
                }]),
            },
        }),
        init: Some(LocalInit {
            eq_token: <Token![=] as Default>::default(),
            expr: Box::new(ident_expr(id("self"))),
            diverge: None,
        }),
        semi_token: <Token![;] as Default>::default(),
    }
}

fn derive_tuple(ident: Ident, generics: Generics, types: Vec<(usize, Type)>) -> ItemImpl {
    let indices = types.iter().map(|(i, _)| *i);
    let field_idents: Vec<_> = types.iter().map(|(i, _)| tuple_ident(*i)).collect();
    let field_types: Vec<_> = types.iter().map(|(_, t)| t.clone()).collect();
    derive_inner(
        ident.clone(),
        generics,
        destruct_tuple(&ident, indices),
        field_idents.as_slice(),
        field_types.as_slice(),
    )
}

fn destruct_named(struct_ident: &Ident, field_names: impl IntoIterator<Item = Ident>) -> Local {
    Local {
        attrs: Vec::new(),
        let_token: <Token![let] as Default>::default(),
        pat: Pat::Struct(PatStruct {
            attrs: Vec::new(),
            qself: None,
            path: Path {
                leading_colon: None,
                segments: Punctuated::from_iter([PathSegment {
                    ident: struct_ident.clone(),
                    arguments: PathArguments::None,
                }]),
            },
            brace_token: Brace::default(),
            fields: field_names
                .into_iter()
                .map(|ident| FieldPat {
                    attrs: Vec::new(),
                    member: syn::Member::Named(ident.clone()),
                    colon_token: None,
                    pat: Box::new(Pat::Ident(PatIdent {
                        attrs: Vec::new(),
                        by_ref: None,
                        mutability: None,
                        ident,
                        subpat: None,
                    })),
                })
                .collect(),
            rest: Some(PatRest {
                attrs: Vec::new(),
                dot2_token: <Token![..] as Default>::default(),
            }),
        }),
        init: Some(LocalInit {
            eq_token: <Token![=] as Default>::default(),
            expr: Box::new(ident_expr(id("self"))),
            diverge: None,
        }),
        semi_token: <Token![;] as Default>::default(),
    }
    /*
    let field_names = field_names.into_iter();
    quote! {
        let #struct_ident{ #(#field_names,)* .. } = self;
    }
    */
}

fn derive_named(ident: Ident, generics: Generics, fields: Vec<(Ident, Type)>) -> ItemImpl {
    let field_idents: Vec<_> = fields.iter().map(|(name, _)| name.clone()).collect();
    let field_types: Vec<_> = fields.iter().map(|(_, ty)| ty.clone()).collect();
    let destruction = destruct_named(&ident, field_idents.iter().cloned());
    derive_inner(
        ident,
        generics,
        destruction,
        field_idents.as_slice(),
        field_types.as_slice(),
    )
}

pub fn derive_sum(input: TokenStream) -> TokenStream {
    let DeriveInput {
        ident,
        generics,
        data,
        ..
    } = parse_macro_input!(input as DeriveInput);
    match data {
        syn::Data::Enum(_) => unimplemented!("Derivation for enum is not supported"),
        syn::Data::Union(_) => unimplemented!("Derivation for union is not supported"),
        syn::Data::Struct(DataStruct { fields, .. }) => match fields {
            syn::Fields::Named(named) => {
                let fields = named
                    .named
                    .into_iter()
                    .map(|f| (f.ident.expect("Named fields always have an ident"), f.ty))
                    .collect();
                derive_named(ident, generics, fields)
                    .to_token_stream()
                    .into()
            }
            syn::Fields::Unnamed(unnamed) => {
                let types = unnamed
                    .unnamed
                    .into_iter()
                    .enumerate()
                    .map(|(i, field)| (i, field.ty))
                    .collect();
                derive_tuple(ident, generics, types)
                    .to_token_stream()
                    .into()
            }
            syn::Fields::Unit => derive_unit(ident),
        },
    }
}
