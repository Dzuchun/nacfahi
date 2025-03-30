#![allow(missing_docs, missing_debug_implementations)]

use nacfahi::models::{
    FitModel, FitModelSum,
    basic::{Constant, Exponent, Gaussian, Linear},
};
use static_assertions::{assert_impl_all, assert_not_impl_all};

#[cfg(feature = "num-traits")]
use num_traits::{Float, FloatConst};

#[derive(FitModelSum)]
#[scalar_type(f64)]
pub struct CustomModel {
    pub cst: Constant<f64>,
    pub exp: Exponent<f64>,
}

#[derive(FitModelSum)]
pub struct CustomGenericModel<Scalar> {
    pub cst: Constant<Scalar>,
    pub exp: Exponent<Scalar>,
}

assert_impl_all!(CustomModel: FitModel<Scalar =f64>);
assert_not_impl_all!(CustomModel: FitModel<Scalar =f32>);

#[derive(FitModelSum)]
#[scalar_type(f64)]
pub struct CustomTupleModel(Constant<f64>, pub Exponent<f64>);

assert_impl_all!(CustomTupleModel: FitModel<Scalar =f64>);
assert_not_impl_all!(CustomTupleModel: FitModel<Scalar =f32>);

#[derive(FitModelSum)]
pub struct CustomConditionedModel<Scalar: Send>
where
    Scalar: Copy + Sync,
{
    pub cst: Constant<Scalar>,
    pub exp: Exponent<Scalar>,
}

assert_impl_all!(CustomConditionedModel<f64>: FitModel<Scalar =f64>);
assert_impl_all!(CustomConditionedModel<f32>: FitModel<Scalar =f32>);
assert_not_impl_all!(CustomConditionedModel<i32>: FitModel);

#[derive(FitModelSum)]
pub struct WithBackground<Scalar, Signal> {
    signal: Signal,
    bg_linear: Linear<Scalar>,
    bg_exponent: Exponent<Scalar>,
}

assert_impl_all!(WithBackground<f64, Gaussian<f64>>: FitModel<Scalar =f64>);
assert_impl_all!(WithBackground<f32, Exponent<f32>>: FitModel<Scalar =f32>);
assert_not_impl_all!(WithBackground<f32, Exponent<f64>>: FitModel);

#[cfg(feature = "num-traits")]
#[derive(FitModelSum)]
#[scalar_generic(T)]
pub struct Multipeak<T: Float + FloatConst + core::iter::Sum, const N: usize, BG> {
    bg: BG,
    signal: [Gaussian<T>; N],
}

#[cfg(feature = "num-traits")]
assert_impl_all!(Multipeak<f64, 0, Linear<f64>>: FitModel<Scalar =f64>);
#[cfg(feature = "num-traits")]
assert_not_impl_all!(Multipeak<f64, 0, Linear<f32>>: FitModel);

#[cfg(feature = "num-traits")]
assert_impl_all!(Multipeak<f64, 10, Linear<f64>>: FitModel<Scalar =f64>);
#[cfg(feature = "num-traits")]
assert_not_impl_all!(Multipeak<f64, 10, Linear<f32>>: FitModel);

#[cfg(feature = "num-traits")]
#[derive(FitModelSum)]
pub struct BiMulti<
    Scalar: Float + FloatConst + core::iter::Sum,
    const EXPS: usize,
    const PEAKS: usize,
> {
    pub bg: [Exponent<Scalar>; EXPS],
    peaks: [Gaussian<Scalar>; PEAKS],
}

#[cfg(feature = "num-traits")]
assert_impl_all!(BiMulti<f64, 0, 0>: FitModel<Scalar = f64>);
#[cfg(feature = "num-traits")]
assert_impl_all!(BiMulti<f64, 1, 5>: FitModel<Scalar = f64>);
#[cfg(feature = "num-traits")]
assert_impl_all!(BiMulti<f64, 12, 5>: FitModel<Scalar =f64>);

#[derive(FitModelSum)]
#[scalar_type(f64)]
pub struct UnitModel;

assert_impl_all!(UnitModel: FitModel<Scalar = f64>);
