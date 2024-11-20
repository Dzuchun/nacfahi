use nacfahi::models::{
    basic::{Constant, Exponent, Gaussian, Linear},
    FitModel, FitModelSum,
};
use static_assertions::{assert_impl_all, assert_not_impl_all};

#[derive(FitModelSum)]
struct CustomModel {
    pub cst: Constant<f64>,
    pub exp: Exponent<f64>,
}

assert_impl_all!(CustomModel: FitModel<f64>);
assert_not_impl_all!(CustomModel: FitModel<f32>);

#[derive(FitModelSum)]
struct CustomTupleModel(Constant<f64>, pub Exponent<f64>);

assert_impl_all!(CustomTupleModel: FitModel<f64>);
assert_not_impl_all!(CustomTupleModel: FitModel<f32>);

#[derive(FitModelSum)]
struct CustomConditionedModel<Scalar: Send>
where
    Scalar: Copy + Sync,
{
    pub cst: Constant<Scalar>,
    pub exp: Exponent<Scalar>,
}

assert_impl_all!(CustomConditionedModel<f64>: FitModel<f64>);
assert_impl_all!(CustomConditionedModel<f32>: FitModel<f32>);
assert_not_impl_all!(CustomConditionedModel<i32>: FitModel<f64>, FitModel<f32>);

#[derive(FitModelSum)]
struct WithBackground<Scalar, Signal> {
    signal: Signal,
    bg_linear: Linear<Scalar>,
    bg_exponent: Exponent<Scalar>,
}

assert_impl_all!(WithBackground<f64, Gaussian<f64>>: FitModel<f64>);
assert_impl_all!(WithBackground<f32, Exponent<f32>>: FitModel<f32>);
assert_not_impl_all!(WithBackground<f32, Exponent<f64>>: FitModel<f32>, FitModel<f64>);

#[derive(FitModelSum)]
struct Multipeak<Scalar, const N: usize, BG> {
    bg: BG,
    signal: [Gaussian<Scalar>; N],
}

assert_impl_all!(Multipeak<f64, 0, Linear<f64>>: FitModel<f64>);
assert_not_impl_all!(Multipeak<f64, 0, Linear<f32>>: FitModel<f64>, FitModel<f32>);

assert_impl_all!(Multipeak<f64, 10, Linear<f64>>: FitModel<f64>);
assert_not_impl_all!(Multipeak<f64, 10, Linear<f32>>: FitModel<f64>, FitModel<f32>);

#[derive(FitModelSum)]
struct BiMulti<Scalar, const EXPS: usize, const PEAKS: usize> {
    pub bg: [Exponent<Scalar>; EXPS],
    peaks: [Gaussian<Scalar>; PEAKS],
}

assert_impl_all!(BiMulti<f64, 0, 0>: FitModel<f64>);
assert_impl_all!(BiMulti<f64, 1, 5>: FitModel<f64>);
assert_impl_all!(BiMulti<f64, 12, 5>: FitModel<f64>);
