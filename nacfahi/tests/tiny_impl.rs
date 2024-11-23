#![allow(missing_docs)]
use nacfahi::models::basic::{Constant, Exponent};
use nacfahi_derive::FitModelSum;

#[derive(FitModelSum)]
struct CustomModel {
    pub cst: Constant<f64>,
    pub exp: Exponent<f64>,
}
