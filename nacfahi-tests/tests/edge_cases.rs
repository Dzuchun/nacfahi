#![allow(missing_docs, missing_debug_implementations)]
use nacfahi::models::{FitModelSum, basic::Constant};

#[derive(FitModelSum)]
#[scalar_type(f64)]
pub struct CustomModel {
    pub x: Constant<f64>,
    pub keep: Constant<f64>,
    pub concat: Constant<f64>,
    pub split: Constant<f64>,
    pub res: Constant<f64>,
    pub result: Constant<f64>,
    pub rest: Constant<f64>,
}

#[derive(FitModelSum)]
#[scalar_type(f64)]
pub struct CustomModel2 {}

#[derive(FitModelSum)]
#[scalar_type(f64)]
pub struct CustomModel3();

#[derive(FitModelSum)]
#[scalar_type(f64)]
pub struct CustomModel4;
