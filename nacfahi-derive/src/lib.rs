#![allow(missing_docs, reason = "It's a procmacro crate")]

use proc_macro::TokenStream;

#[doc(hidden)]
mod fit_model;

#[proc_macro_derive(FitModelSum)]
pub fn derive_fit_entity(input: TokenStream) -> TokenStream {
    fit_model::derive_sum(input)
}
