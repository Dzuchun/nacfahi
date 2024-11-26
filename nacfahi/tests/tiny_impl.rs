#![allow(missing_docs)]
use nacfahi::models::basic::{Constant, Exponent, Gaussian};
use nacfahi_derive::FitModelSum;

#[derive(FitModelSum)]
struct CustomModel<Scalar, const N: usize> {
    pub cst: Constant<Scalar>,
    pub exp: Exponent<Scalar>,
    pub peaks: [Gaussian<Scalar>; N],
}
