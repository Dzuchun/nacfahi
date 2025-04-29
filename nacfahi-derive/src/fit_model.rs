#![allow(clippy::default_trait_access)]

use core::iter::zip;
use std::{cell::Cell, rc::Rc};

use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{
    Attribute, DataStruct, DeriveInput, ExprPath, FieldsNamed, FieldsUnnamed, GenericParam,
    Generics, Ident, ItemImpl, MacroDelimiter, Meta, MetaList, Path, PathArguments, PathSegment,
    PredicateLifetime, PredicateType, Stmt, Token, Type, TypeParam, TypePath, WhereClause,
    WherePredicate, parse_macro_input, parse_quote, parse_quote_spanned, spanned::Spanned,
};

macro_rules! procmacro_item {
    ($name:ident) => {
        #[inline]
        #[allow(non_snake_case)]
        fn $name() -> Path {
            ::syn::parse_quote! {
                ::nacfahi::__procmacro::$name
            }
        }
    };
}

procmacro_item!(FitModel);
procmacro_item!(Zero);
procmacro_item!(UTerm);
procmacro_item!(GenericArray);
procmacro_item!(ArrayLength);
procmacro_item!(Concat);
procmacro_item!(Split);
procmacro_item!(Conv);

fn derive_unit(ident: Ident, scalar: ScalarType) -> ItemImpl {
    let zero = Zero();
    let scalar = match scalar {
        ScalarType::Generic(_) => {
            panic!("Unit struct has no way to have a generic type parameter.")
        }
        ScalarType::Specified(s) => s,
    };
    let model = FitModel();
    let uterm = UTerm();
    let generic_array = GenericArray();
    parse_quote_spanned! {ident.span() =>
        impl #model for #ident {
            type Scalar = #scalar;
            type ParamCount = #uterm;

            #[inline]
            fn evaluate(&self, _: & #scalar) -> #scalar {
                < #scalar as #zero >::zero()
            }

            #[inline]
            fn jacobian(&self, _: & #scalar) -> impl ::core::convert::Into< #generic_array <#scalar, #uterm>> {
                []
            }

            #[inline]
            fn set_params(&mut self, _: #generic_array <#scalar, #uterm>) {}

            #[inline]
            fn get_params(&self) -> impl ::core::convert::Into< #generic_array <#scalar, #uterm>> {
                []
            }
        }
    }
}

fn model_params(ty: &Type) -> TypePath {
    let model = FitModel();
    parse_quote_spanned! {ty.span() =>
        < #ty as #model >::ParamCount
    }
}

fn type_sum(types: impl IntoIterator<Item = Type>) -> Type {
    let mut types = types.into_iter();
    let first = types.next().unwrap();

    let mut res = parse_quote_spanned! {first.span() => #first };
    for t in types {
        res = parse_quote_spanned! {t.span() => < #res as ::core::ops::Add< #t > >::Output };
    }
    res
}

fn field_bounds(
    field_types: &[Type],
    scalar_type: &ScalarType,
) -> impl IntoIterator<Item = WherePredicate> {
    let model = FitModel();
    field_types
        .iter()
        .map(move |ty| parse_quote_spanned!(ty.span() => #ty: #model < Scalar = #scalar_type > ))
}

fn count_bounds(
    counts: impl IntoIterator<Item = Type>,
) -> impl IntoIterator<Item = WherePredicate> {
    let mut types = counts.into_iter();
    let first = types
        .next()
        .expect("Need at least 1 count to implement any bounds");
    let array_len = ArrayLength();
    let conv = Conv();
    let tfirst: Type = parse_quote_spanned! { first.span() => < #first as #conv >::TNum };
    let inner_type: Rc<Cell<Option<Type>>> = Rc::new(Cell::new(Some(tfirst.clone())));
    let uterm = UTerm();
    core::iter::once(parse_quote_spanned! {first.span() => #tfirst: ::core::ops::Sub< #uterm, Output = #tfirst > })
    .chain({
        let inner_type = Rc::clone(&inner_type);
        let conv = conv.clone();
        types.flat_map(move |t| {
            let it = inner_type.take().unwrap();
            let tconv: Type = parse_quote_spanned! { t.span() => < #t as #conv >::TNum };
            let out = [
                parse_quote_spanned!{it.span() => #t: #conv },
                parse_quote_spanned!{t.span() => #it: ::core::ops::Add< #tconv > },
                parse_quote_spanned!{t.span() => < #it as ::core::ops::Add< #tconv > >::Output: ::core::ops::Sub< #it, Output = #tconv > + #array_len }
            ];

            inner_type.set(Some( parse_quote_spanned! {t.span() => < #it as ::core::ops::Add< #tconv > >::Output }));
            out
        })
    }).chain(core::iter::once_with(move || {
        let inner_type = inner_type.take().unwrap();
        parse_quote_spanned! {inner_type.span() => #inner_type: #conv }
    }))
}

fn bounds(fields: &[Type], scalar: &ScalarType) -> impl IntoIterator<Item = WherePredicate> {
    [
        parse_quote_spanned! {scalar.span() => #scalar: ::core::ops::Add<#scalar, Output = #scalar> },
        // parse_quote_spanned! {param_count.span() => #param_count: #conv}
    ]
    .into_iter()
    .chain(field_bounds(fields, scalar))
    .chain(count_bounds(
        fields.iter().map(model_params).map(Type::from),
    ))
}

fn model_evaluate(ty: &Type) -> ExprPath {
    let model = FitModel();
    parse_quote_spanned! {ty.span()=> < #ty as #model >::evaluate }
}

fn model_jacobian(ty: &Type) -> ExprPath {
    let model = FitModel();
    parse_quote_spanned! {ty.span() => < #ty as #model >::jacobian }
}

fn model_set_params(ty: &Type) -> ExprPath {
    let model = FitModel();
    parse_quote_spanned! {ty.span() => < #ty as #model >::set_params }
}

fn model_get_params(ty: &Type) -> ExprPath {
    let model = FitModel();
    parse_quote_spanned! {ty.span() => < #ty as #model >::get_params }
}

fn evaluate_body(
    idents: &[Ident],
    types: &[Type],
    destruction_syntax: &Stmt,
    scalar: &ScalarType,
) -> impl IntoIterator<Item = Stmt> {
    core::iter::once(destruction_syntax.clone())
        .chain(zip(idents, types).map(move |(id, ty)| {
            let evaluate = model_evaluate(ty);
            parse_quote_spanned! {ty.span() => let #id: #scalar = #evaluate(#id, x); }
        }))
        .chain(core::iter::once(Stmt::Expr(
            ::syn::parse_quote!( #(#idents)+* ),
            None,
        )))
}

fn jacobian_body(
    idents: &[Ident],
    types: &[Type],
    destruction_syntax: &Stmt,
    scalar: &ScalarType,
) -> impl IntoIterator<Item = Stmt> {
    let mut res = TokenStream::new();
    idents[0].to_tokens(&mut res);
    for id in &idents[1..] {
        quote! { .concat(#id) }.to_tokens(&mut res);
    }

    let concat = Concat();
    [
        ::syn::parse_quote! { use #concat; },
        destruction_syntax.clone(),
    ]
    .into_iter()
    .chain(zip(idents, types).map({
        let generic_array = GenericArray();
        let model = FitModel();
        let conv = Conv();
        move |(id, ty)| {
            let model_jacobian = model_jacobian(ty);
            parse_quote_spanned! {ty.span() =>
                let #id: #generic_array < #scalar, < < #ty as #model >::ParamCount as #conv >::TNum >
                = #model_jacobian ( #id, x).into();
            }
        }
    }))
    .chain(core::iter::once(Stmt::Expr(::syn::parse_quote!( #res ), None)))
}

fn set_params_body(
    idents: &[Ident],
    types: &[Type],
    destruction_syntax: &Stmt,
) -> impl IntoIterator<Item = Stmt> {
    let conv = Conv();
    let model_to_tnum = move |ty: &Type| -> Type {
        let params = model_params(ty);
        parse_quote!( <#params as #conv>::TNum )
    };
    let mut rest_counts = {
        let mut types = types.iter();
        let mut rest_counts = Vec::<Type>::with_capacity(types.len());
        let uterm = UTerm();
        rest_counts.push(parse_quote!( #uterm ));
        rest_counts.push(model_to_tnum(types.next().unwrap()));
        for _ in (0..idents.len()).skip(2) {
            let prev = rest_counts.last().unwrap();
            let this = model_to_tnum(types.next().unwrap());
            rest_counts.push(parse_quote!( <#prev as ::core::ops::Add < #this>>::Output ));
        }
        rest_counts
    };
    let generic_array = GenericArray();
    let split = Split();
    [
        ::syn::parse_quote!( use #split; ),
        ::syn::parse_quote!( let rest = new_params; ),
        destruction_syntax.clone(),
    ]
    .into_iter()
    .chain(zip(idents, types).rev().flat_map(move|(id, ty)| {
        let model_set_params = model_set_params(ty);
        let this_count = model_to_tnum(ty);
        let rest_count = rest_counts.pop().unwrap();
        [
            parse_quote_spanned!(ty.span() => let (rest, this): (#generic_array::<_, #rest_count>, #generic_array::<_, #this_count>) = rest.split(); ),
            parse_quote_spanned!(ty.span() => #model_set_params (#id, this); ),
        ]
    }))
    .chain(core::iter::once( parse_quote!(let _ = rest;)))
}

fn get_params_body(
    idents: &[Ident],
    types: &[Type],
    destruction_syntax: &Stmt,
) -> impl IntoIterator<Item = Stmt> {
    let mut idty = zip(idents, types);
    let (first_id, first_ty) = idty.next().expect("Need at least 1 field to build body");
    let first_get_params = model_get_params(first_ty);

    let concat = Concat();
    [
        ::syn::parse_quote!( use #concat; ),
        destruction_syntax.clone(),
        parse_quote_spanned!(first_ty.span() => let res = #first_get_params (#first_id).into(); ),
    ]
    .into_iter()
    .chain(idty.map(|(id, ty)| {
        let model_get_params = model_get_params(ty);
        parse_quote_spanned! {ty.span() => let res = res.concat( #model_get_params (#id).into()); }
    }))
    .chain(core::iter::once(Stmt::Expr(::syn::parse_quote!(res), None)))
}

fn take_param_bound(par: &mut GenericParam) -> Option<WherePredicate> {
    match par {
        GenericParam::Lifetime(lifetime_param) => {
            if lifetime_param.bounds.is_empty() {
                return None;
            }
            lifetime_param.colon_token = None;
            Some(WherePredicate::Lifetime(PredicateLifetime {
                lifetime: lifetime_param.lifetime.clone(),
                colon_token: Default::default(),
                bounds: core::mem::take(&mut lifetime_param.bounds),
            }))
        }
        GenericParam::Type(type_param) => {
            if type_param.bounds.is_empty() {
                return None;
            }
            type_param.default = None;
            type_param.colon_token = None;
            let bound_ty = type_param.ident.clone();
            Some(WherePredicate::Type(PredicateType {
                lifetimes: None, // parameter declaration can't have HOLB
                bounded_ty: parse_quote_spanned!(type_param.span() => #bound_ty),
                colon_token: Default::default(),
                bounds: core::mem::take(&mut type_param.bounds),
            }))
        }
        GenericParam::Const(_const_param) => None, // no bounds for const params in Rust yet
    }
}

fn generic_param_to_decl(par: &GenericParam, stream: &mut TokenStream) {
    match par {
        GenericParam::Lifetime(lifetime_param) => lifetime_param.lifetime.to_tokens(stream),
        GenericParam::Type(type_param) => type_param.ident.to_tokens(stream),
        GenericParam::Const(const_param) => const_param.ident.to_tokens(stream),
    }
}

#[allow(clippy::too_many_lines)]
fn derive_inner(
    struct_ident: Ident,
    mut generics: Generics,
    destruction_syntax: Stmt,
    field_idents: &[Ident],
    field_types: &[Type],
    scalar: ScalarType,
) -> ItemImpl {
    let conv = Conv();
    let param_count = type_sum(
        field_types
            .iter()
            .map(model_params)
            .map(|mp| parse_quote_spanned! { mp.span() => < #mp as #conv >::TNum }),
    );

    // get all the bounds from definition's where clause
    let mut predicates = generics
        .where_clause
        .take()
        .map(|wc| wc.predicates)
        .unwrap_or_default();
    // append all the bounds from type declarations
    predicates.extend(generics.params.iter_mut().filter_map(take_param_bound));
    // append all the bounds emposed by implementation
    predicates.extend(bounds(field_types, &scalar));
    predicates.push(
        parse_quote_spanned! {param_count.span() => #param_count: #conv <TNum = #param_count>},
    );
    let where_cause = WhereClause {
        where_token: Default::default(),
        predicates,
    };
    let decl_params = if generics.params.is_empty() {
        TokenStream::new()
    } else {
        let mut tps = generics.params.iter();
        let mut res = quote! {<};
        generic_param_to_decl(tps.next().expect("must have at least one tp"), &mut res);
        for tp in tps {
            <Token![,]>::default().to_tokens(&mut res);
            generic_param_to_decl(tp, &mut res);
        }
        <Token![>]>::default().to_tokens(&mut res);
        res
    };
    let def_params = if generics.params.is_empty() {
        TokenStream::new()
    } else {
        let mut tps = generics.params.iter();
        let mut res = quote! {<};

        tps.next()
            .expect("must have at least one tp")
            .to_tokens(&mut res);
        for tp in tps {
            <Token![,]>::default().to_tokens(&mut res);
            tp.to_tokens(&mut res);
        }
        <Token![>]>::default().to_tokens(&mut res);
        res
    };

    let model = FitModel();
    let generic_array = GenericArray();
    let conv = Conv();

    let evaluate =
        evaluate_body(field_idents, field_types, &destruction_syntax, &scalar).into_iter();
    let jacobian =
        jacobian_body(field_idents, field_types, &destruction_syntax, &scalar).into_iter();
    let set_params = set_params_body(field_idents, field_types, &destruction_syntax).into_iter();
    let get_params = get_params_body(field_idents, field_types, &destruction_syntax).into_iter();
    parse_quote_spanned! { struct_ident.span() =>
        impl #def_params #model for  #struct_ident #decl_params
            #where_cause
        {
            type ParamCount = #param_count;
            type Scalar = #scalar;

            #[inline]
            fn evaluate(&self, x: & #scalar) -> #scalar {
                #(#evaluate)*
            }

            #[inline]
            fn jacobian(&self, x: & #scalar) -> impl ::core::convert::Into< #generic_array <Self::Scalar, < Self::ParamCount as #conv >::TNum > > {
                #(#jacobian)*
            }

            #[inline]
            #[allow(clippy::type_complexity)]
            fn set_params(&mut self, new_params: #generic_array < Self::Scalar, < Self::ParamCount as #conv >::TNum >) {
                #(#set_params)*
            }

            #[inline]
            fn get_params(&self) -> impl ::core::convert::Into < #generic_array < Self::Scalar, < Self::ParamCount as #conv >::TNum > > {
                #(#get_params)*
            }
        }
    }
}

fn field_idents() -> impl Iterator<Item = Ident> {
    (0..).map(|i| format_ident!("field_{i}"))
}

fn destruct_tuple(field_idents: &[Ident]) -> Stmt {
    ::syn::parse_quote! { let Self ( #(#field_idents),* ) = self; }
}

fn derive_tuple(
    ident: Ident,
    generics: Generics,
    types: Vec<Type>,
    scalar: ScalarType,
) -> ItemImpl {
    let idents: Vec<_> = field_idents().take(types.len()).collect();
    derive_inner(
        ident,
        generics,
        destruct_tuple(&idents),
        &idents,
        &types,
        scalar,
    )
}

fn destruct_named(field_names: &[Ident]) -> Stmt {
    let field_idents = field_idents().take(field_names.len());
    ::syn::parse_quote! { let Self { #(#field_names: #field_idents,)* .. } = self; }
}

fn derive_named(
    ident: Ident,
    generics: Generics,
    field_names: Vec<Ident>,
    field_types: Vec<Type>,
    scalar: ScalarType,
) -> ItemImpl {
    let idents: Vec<_> = field_idents().take(field_types.len()).collect();
    derive_inner(
        ident,
        generics,
        destruct_named(&field_names),
        &idents,
        field_types.as_slice(),
        scalar,
    )
}

#[derive(Clone)]
enum ScalarType {
    Generic(Ident),
    Specified(Type),
}

impl ToTokens for ScalarType {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            ScalarType::Generic(id) => id.to_tokens(tokens),
            ScalarType::Specified(ty) => ty.to_tokens(tokens),
        }
    }
}

fn filter_attr(Attribute { meta, .. }: &Attribute, filter_ident: &str) -> bool {
    let Meta::List(MetaList {
        path: Path {
            leading_colon: None,
            segments,
        },
        delimiter: MacroDelimiter::Paren(_),
        ..
    }) = meta
    else {
        return false;
    };
    let mut segments = segments.into_iter();
    let Some(PathSegment {
        ident,
        arguments: PathArguments::None,
    }) = segments.next()
    else {
        return false;
    };
    if segments.next().is_some() {
        return false;
    }
    if ident != filter_ident {
        return false;
    }
    true
}

const GENERIC_ATTR: &str = "scalar_generic";
const SPECIFIED_ATTR: &str = "scalar_type";

fn parse_scalar(generics: &Generics, attrs: &[Attribute]) -> ScalarType {
    let scalar_generic = generics.params.iter().find_map(|param| match param {
        GenericParam::Type(TypeParam { ident, .. }) if *ident == "Scalar" => Some(ident.clone()),
        _ => None,
    });

    let attr_generic = attrs
        .iter()
        .find(|attr| filter_attr(attr, GENERIC_ATTR))
        .map(|attr| {
            attr.parse_args::<Ident>()
                .expect("scalar_generic attribute should contain a valid ident")
        });

    let specified = attrs
        .iter()
        .find(|attr| filter_attr(attr, SPECIFIED_ATTR))
        .map(syn::Attribute::parse_args::<Type>)
        .transpose()
        .unwrap();

    match (scalar_generic, attr_generic, specified) {
        (None, None, None) => panic!("Please specify scalar type. See documentation for details"),
        (_, None, Some(specified)) => ScalarType::Specified(specified),
        (_, Some(attr_generic), None) => ScalarType::Generic(attr_generic),
        (Some(scalar_generic), None, None) => ScalarType::Generic(scalar_generic),
        (_, Some(attr_generic), Some(_)) => panic!(
            "Scalar type should only be specified once; currently, you specify as generic ({}) and exact (whatever is in {} attribute) at the same time.",
            attr_generic, SPECIFIED_ATTR
        ),
    }
}

pub fn derive_sum(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let DeriveInput {
        ident,
        generics,
        data,
        attrs,
        ..
    } = parse_macro_input!(input as DeriveInput);

    let scalar_type = parse_scalar(&generics, &attrs);

    match data {
        syn::Data::Union(_) => {
            unimplemented!("Derivation for union is not possible. Please consider using enum")
        }
        syn::Data::Enum(_) => unimplemented!("Derivation for enum is not supported yet"),
        syn::Data::Struct(DataStruct { fields, .. }) => match fields {
            syn::Fields::Named(FieldsNamed {
                brace_token: _,
                named,
            }) => {
                let types: Vec<_> = named.iter().map(|f| f.ty.clone()).collect();
                let names: Vec<_> = named.into_iter().map(|f| f.ident.unwrap()).collect();
                if types.is_empty() {
                    derive_unit(ident, scalar_type)
                } else {
                    derive_named(ident, generics, names, types, scalar_type)
                }
            }
            syn::Fields::Unnamed(FieldsUnnamed {
                paren_token: _,
                unnamed,
            }) => {
                let types: Vec<_> = unnamed.into_iter().map(|f| f.ty).collect();
                if types.is_empty() {
                    derive_unit(ident, scalar_type)
                } else {
                    derive_tuple(ident, generics, types, scalar_type)
                }
            }
            syn::Fields::Unit => derive_unit(ident, scalar_type),
        },
    }
    .to_token_stream()
    .into()
}
