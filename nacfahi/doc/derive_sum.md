Derive for the model defined as sum of it's fields.

Since [`FitModel`] has `Scalar` as it's associated type, following ways were implemented to specify it:

- In case your model is not generic over scalar type, implementation can't be generic too. For macro to work properly, specify scalar in `scalar_type` attribute, like that:
```rust,ignore
#[scalar_type(f32)]
```
(see some of the examples below)

- In case your struct is generic over scalar type and it is named `Scalar`, it will be used. No additional attribute is required.
- In case your struct is generic over scalar type, but

    - does not have `Scalar` type parameter
    - `Scalar` type parameter has a different meaning

you should specify type parameter intended for scalar type in `scalar_generic` attribute, like that:
```rust,ignore
#[scalar_generic(T)] // use type T for scalar
```
(see some of the examples below)

**Note**: currently, there's no way to opt-out field(s). Receiver struct implements [`FitModel`] *ONLY* when all of it's fields do.

**Note**: Enums are not supported, for now.

*Note*: Following examples use macros from an excellent [`static_assertions`] create to convey and simultaneously check for trait implementations. Macro names are self-explanatory.

Examples:
```rust
# use nacfahi::{models::{FitModel, FitModelSum, basic::{Constant, Exponent}}, *};
# use static_assertions::{assert_impl_all, assert_not_impl_all};
# 
#[derive(FitModelSum)]
#[scalar_type(f64)]
struct CustomModel {
    pub cst: Constant<f64>,
    pub exp: Exponent<f64>,
}

assert_impl_all!(CustomModel: FitModel<Scalar = f64>);

#[derive(FitModelSum)]
#[scalar_type(f64)]
struct CustomTupleModel(Constant<f64>, pub Exponent<f64>);

assert_impl_all!(CustomTupleModel: FitModel<Scalar = f64>);
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

assert_impl_all!(CustomConditionedModel<f64>: FitModel<Scalar = f64>);
assert_impl_all!(CustomConditionedModel<f32>: FitModel<Scalar = f32>);
assert_not_impl_all!(CustomConditionedModel<i32>: FitModel);
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

assert_impl_all!(WithBackground<f64, Gaussian<f64>>: FitModel<Scalar = f64>);
assert_impl_all!(WithBackground<f32, Exponent<f32>>: FitModel<Scalar = f32>);
assert_not_impl_all!(WithBackground<f32, Exponent<f64>>: FitModel);

#[derive(FitModelSum)]
#[scalar_generic(S)]
struct Multipeak<S, const N: usize, BG> {
    bg: BG,
    signal: [Gaussian<S>; N],
}

assert_impl_all!(Multipeak<f64, 0, Linear<f64>>: FitModel<Scalar = f64>);
assert_not_impl_all!(Multipeak<f64, 0, Linear<f32>>: FitModel);

assert_impl_all!(Multipeak<f64, 10, Linear<f64>>: FitModel<Scalar = f64>);
assert_not_impl_all!(Multipeak<f64, 10, Linear<f32>>: FitModel);

#[derive(FitModelSum)]
#[scalar_generic(T)]
struct BiMulti<T, const EXPS: usize, const PEAKS: usize> {
    pub bg: [Exponent<T>; EXPS],
    peaks: [Gaussian<T>; PEAKS],
}

assert_impl_all!(BiMulti<f64, 0, 0>: FitModel);
assert_impl_all!(BiMulti<f64, 1, 5>: FitModel);
assert_impl_all!(BiMulti<f64, 12, 5>: FitModel);
```

[`static_assertions`]: https://docs.rs/static_assertions/latest/static_assertions/
[`FitModel`]: trait.FitModel.html
