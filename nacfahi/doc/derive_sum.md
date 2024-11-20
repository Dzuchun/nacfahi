Derive for the model defined as sum of it's fields.

**Note**: type parameter named `Scalar` is treated in a special way: this macro will relay this same parameter to [`FitModel`] parameter. In case there's no `Scalar` generic type parameter in your struct, implementation will create it's own `Scalar` parameter.

**Note**: currently, there's no way to opt-out field(s). Receiver struct implements [`FitModel`] *ONLY* when all of it's fields do.

**Note**: Enums are not supported, for now.

*Note*: Following examples use macros from an excellent [`static_assertions`] create to convey and simultaneously check for trait implementations. Macro names are self-explanatory.

Examples:
```rust
# use nacfahi::{models::{FitModel, FitModelSum, basic::{Constant, Exponent}}, *};
# use static_assertions::{assert_impl_all, assert_not_impl_all};
# 
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
```

(any valid struct definition works, definition bounds are respected)

```rust
# use nacfahi::{models::{FitModel, FitModelSum, basic::{Constant, Exponent, Gaussian}}, *};
# use static_assertions::{assert_impl_all, assert_not_impl_all};
# 
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
```

You are free to use type parameters and constants in field types. All of the rules stay the same.

```rust
# use nacfahi::{models::{FitModel, FitModelSum, basic::{Constant, Exponent, Gaussian, Linear}}, *};
# use static_assertions::{assert_impl_all, assert_not_impl_all};
# 
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
```

[`static_assertions`]: https://docs.rs/static_assertions/latest/static_assertions/
[`FitModel`]: trait.FitModel.html
