![CI status](https://github.com/Dzuchun/nacfahi/actions/workflows/build.yml/badge.svg)
[![Documentation status](https://github.com/Dzuchun/nacfahi/actions/workflows/docs.yml/badge.svg)][docs]

```text
This crate does not require nor std nor alloc ❤️
```

This is my take on a "convenient interface" for a brilliant [`levenberg_marquardt`] crate, aiming to be versatile, flexible and easy-to-understand.

Before you read further, consider giving [`levenberg_marquardt`] itself a look - it's interface is quite abstract, and you might come up with a more efficient use pattern for your problem.

As another shout out, see [`varpro`] crate, which does basically the same thing (provides a high-level interface to [`levenberg_marquardt`]), except it has some more special sauce your application might benefit from.

## Motivation

I've identified following drawbacks in APIs of mentioned crates:

- [`levenberg_marquardt`] uses [`nalgebra`] and requires you to explicitly use it too. This can be quite confusing, especially if you are yet to read [`nalgebra`]'s doc.
- Both [`levenberg_marquardt`] and [`varpro`] unify parameters and data into a single object. This makes sense for internal solving process, but there's no reason to leave it like that in a public API.
- [`varpro`] requires `std` (because reasons, I guess)
- [`levenberg_marquardt`] allows problem to return any sort of [`nalgebra`] matrix, abstracted over storage. [`varpro`], on the other hand, is hard-coded to use `Dyn`-sized storages, meaning that for every single parameters get/set operation, new vector (on the heap!) is allocated. This might be less than ideal, especially if both parameter and point count happen to be statically known.
- [`varpro`] defines it's `SeparableModel` parameters as a [`nalgebra`] vector. This leaves your code prone to getting wrong parameter or non-existing parameter from the vector.

I've come up with the following design:

1. All models have statically-known parameter count. This allows get/set operations to use stack-allocated storage.
2. Model (parameters container) is separate from data. Combination into a single struct is not required to use to the public API.
3. (2) allows definition of an alternative model trait, with no [`nalgebra`] mentioned. Instead, it tangentially mentions [`generic_array`] \(specifically, there's an `Into<GenericArray>` return bound\), which I find easier to use.
4. (2) allows abstraction over data-providing type. Data provider is converted into [`nalgebra`] matrix internally.
5. This crate only heap-allocates in case there is an unknown data point count.
6. Model type is never erased, models expose relevant parameters as fields/methods. Makes impossible to ask for wrong/incorrect parameter, since all parameters are obtained via field access and function calls (instead of opaque [`nalgebra`] matrix element).
7. Crate provides a simple way to compose multiple models into a single, more complex one.

I find described API more intuitive and less error prone. Also it statically prevents some of the [`levenberg_marquardt`]s termination reasons (`User`, `NoParameters`, `NoResiduals`, `WrongDimensions`).

With general idea outlined, here's an example:

# Basic example

```rust
# use approx::assert_ulps_eq;
# use nacfahi::{models::basic::Linear, *};

// some data, presumably 2x + 1
let x = [1.0, 2.0, 3.0, 4.0, 5.0];
let y = [3.0, 5.0, 7.0, 9.0, 11.0];

// fitting model: a*x + b
let mut line = Linear { a: 0.0, b: 0.0 };

// do the fit!
let report = fit!(&mut line, x, y);

// check that approximation is successful
assert!(
    report.termination.was_successful(),
    "Approximation should be successful"
);

// check that model parameters have expected values
assert_ulps_eq!(line.a, 2.0);
assert_ulps_eq!(line.b, 1.0);
```

Looks simple enough? Consider reading the rest, then!

# What's actually going on

There are a couple things to unpack here:

## Data format

Input data can be any [`AsMatrixView`] trait implementor, and core Rust array happens to be one. See trait's documentation for details.

## `fit` macro

That's a pure convenience macro expanding into a [`function@fit`] function call. For details, see [`macro@fit!`].

## Fit models

Model generally refers to [`FitModel`] implementation, internally wired to [`levenberg_marquardt`]'s [`LeastSquaresProblem`] trait.

Most of the public items in this crate **are** models representing various common fitting functions or meta operations.

Additionally, [`FitModel`] is implemented for `&mut `[`FitModel`] - this is actually utilized in the above example to keep ownership of the model after the fit.

Also, core Rust arrays implement [`FitModel`], as a sum of it's element models. So `[Exponent; 2]` would fit with a sum of two independent [`Exponent`](models::basic::Exponent) models, and `[Gaussian; 5]` would fit with 5 [`Gaussian`](models::basic::Gaussian) independent models.

To reiterate: these models contain multiple **independent** instances **of the same model type** that are added up.

### Basic Models

Basic models are representations of elementary functions. You can fit them directly (as does the example above), or compose more complex models with them (see below).

Basic models are located in [`models::basic`] module, see it's items for details.

### Utility models

Utility models are models containing other models, implementing some sort of additional functionality, like range filtering or mapping.

See [`models::utility`] items for details. Here are some examples:

- [`Ranged`](models::utility::Ranged) has a second field `range` defining `x` variable range the model will be nonzero in. Sharp turns can be emulated with this model, for example here is an exponential "ramp", dropping after `0.0`:

```rust
# use core::ops::RangeTo;
# use nacfahi::{models::{basic::Exponent, utility::Ranged}, *};
// (see `utility_models` integration test for details)
type ExpRamp<Scalar> = Ranged<Exponent<Scalar>, RangeTo<Scalar>>;

// for example, this model equals 0 at x > 0:
let _ramp = Ranged {
    inner: Exponent { a: 0.0, b: 0.0 },
    range: ..0.0,
};
```

- [`ModelMap`](models::utility::ModelMap) \(UNTESTED!\) is **supposed** to additionally map the model, allowing fits in mapped spaces. For example, while fitting to a single exponent, you might want to use [`LnMap`](models::utility::LnMap) to do a linear fit:

```rust
# use nacfahi::{models::{basic::Exponent, utility::{model_map, LnMap}}, fit};
# use num_traits::Float;
# use approx::assert_ulps_eq;
# 
// some exponential data
let expected_a = 3.0;
let expected_b = 0.5;
let x = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
let y = x.map(|x| expected_a * (x * expected_b).exp());
let linear_y = y.map(f64::ln);

// exponential model
let mut expo_model = Exponent { a: 1.0, b: 0.0 };
// expolinear (exponential mapped to linear)
let mut expolinear = model_map(&mut expo_model, LnMap);

// fit!
let report = fit!(&mut expolinear, x, linear_y);

# assert!(
#     report.termination.was_successful(),
#     "Fit should be successful {report:?}"
# );
# assert_ulps_eq!(expo_model.a, expected_a);
# assert_ulps_eq!(expo_model.b, expected_b);
```

*Note: this functionality is largely unfinished, and probably should not be used yet*

- [`Composition`](models::utility::Composition) \(UNTESTED!\) is **supposed** to allow model composition. This is similar to [`ModelMap`](models::utility::ModelMap), except "the map" here has it's own parameters and fitting process fits them as well. Here's an example of gaussian-over-exponential model (whatever that would mean):

```text
/* no example hewe, sowwy :( */
```

*Note: this functionality is largely unfinished, and probably should not be used yet*

### Custom models

What if you need a model representing a sum of linear, exponential and three gaussian peaks? Even `[Box<dyn FitModel>; 5]` won't work, since [`FitModel`] is not object-safe...

Well, [`FitModel`] is fully public, and you are free to implement it yourself! In this case, it's a bunch of annoying boilerplate, you can find [here][manual_impl] (you can check it, in case you plan on implementing the trait yourself).

Good news is - there's a `derive` macro for that!
```rust
# use nacfahi::{models::{FitModel, FitModelSum, basic::{Constant, Exponent, Gaussian}}, *};
# use static_assertions::assert_impl_all;
# 
#[derive(FitModelSum)]
#[scalar_type(f64)]
struct ConstExponent {
    linear: Constant<f64>,
    exponent: Exponent<f64>,
    peaks: [Gaussian<f64>; 3],
}
# 
# assert_impl_all!(ConstExponent: FitModel<Scalar = f64>);
```

And it does exactly all of the above, except you can do some stuff that is hard to implement manually.

See [`FitModelSum`](models::FitModelSum) for usage details and more examples.

# Why the name?

Actual intended name is `nacfa'i`, which is a lojban predicate for ["x1 **is solved to find** x2"][1].

(I am not proficient in lojban at all, pwease don't huwt mw :3

[`levenberg_marquardt`]: https://docs.rs/levenberg-marquardt/latest/levenberg_marquardt/
[`LevenbergMarquardt`]: https://docs.rs/levenberg-marquardt/latest/levenberg_marquardt/struct.LevenbergMarquardt.html
[`LeastSquaresProblem`]: https://docs.rs/levenberg-marquardt/latest/levenberg_marquardt/trait.LeastSquaresProblem.html
[`varpro`]: https://docs.rs/varpro/latest/varpro/
[`nalgebra`]: https://docs.rs/nalgebra/latest/nalgebra/
[`generic_array`]: https://docs.rs/generic-array/latest/generic_array/
[1]: https://la-lojban.github.io/sutysisku/lojban/index.html#seskari=fanva&sisku=nacfa%27i&bangu=en&versio=masno
[docs]: https://dzuchun.github.io/nacfahi/nacfahi/index.html
[manual_impl]: https://github.com/Dzuchun/nacfahi/blob/master/nacfahi-tests/tests/manual_impl.rs
