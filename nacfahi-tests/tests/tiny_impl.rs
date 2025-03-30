#![allow(missing_docs, missing_debug_implementations)]
use nacfahi::models::{
    FitModelSum,
    basic::{Constant, Exponent, Gaussian},
};

#[derive(FitModelSum)]
pub struct CustomModel<Scalar, const N: usize> {
    pub cst: Constant<Scalar>,
    pub exp: Exponent<Scalar>,
    pub peaks: [Gaussian<Scalar>; N],
}
