A convenience macro for [`function@fit`] function. You are free to call the function directly, if this macro is confusing to you.

[`function@fit`] function signature:

```rust,no_run,ignore
pub fn fit<Scalar>(Model, X, Y, &LevenbergMarquardt<Scalar>, impl Fn(Scalar, Scalar) -> Scalar) -> MinimizationReport<Scalar>
```

Here you can see two optional arguments:

- `LevenbergMarquardt` object is, well, [`LevenbergMarquardt`] object from [`levenberg_marquardt`] crate, defining some approximation config. If none provided, one is created via associated `new` function (see crate doc).
- `Fn(Scalar, Scalar) -> Scalar` is a weights function, defining weight of the data point based on `x` and `y` value. Defaults to [`One::one`](https://docs.rs/num-traits/latest/num_traits/identities/trait.One.html#tymethod.one), if possible.

**WARN**: `default_weights` ident for your weights **will not** use your own variable of the same name - that's the name of the default weights function at `nacfahi::default_weights`, and it overrides any ident you might have.

If you need to specify any of them, you can do that:

```rust
# use approx::assert_ulps_ne;
# use nacfahi::{models::basic::Exponent, *};
# 
# // some data, presumably 2x + 1
# let x = [1.0, 2.0, 3.0, 4.0, 5.0];
# let y = [3.0, 5.0, 7.0, 9.0, 11.0];
# 
// fitting model: a * exp(-b*x)
let mut line = Exponent { a: 0.0, b: 0.0 };

// custom very impatient lavmar instance
let lavmar = LevenbergMarquardt::new().with_patience(1);

# // patience too low, proceed to fail the fit :3
let report = fit!(&mut line, x, y, minimizer = lavmar);
# 
# // expect to stop because of patience
# assert_eq!(
#     report.termination,
#     TerminationReason::LostPatience,
#     "Approximation should stop because of patience"
# );
# 
# // no reason to assume this, but oh well
# assert_ulps_ne!(line.a, 2.0);
# assert_ulps_ne!(line.b, 1.0);
# 
# // since we use different model, let's check that it's still able to fit, eventually:
# line.a = 0.0;
# line.b = 0.0;
# 
# // do the fit! (again)
# let report = fit!(&mut line, x, y);
# 
# // check that approximation is successful
# assert!(
#     report.termination.was_successful(),
#     "Approximation should be successful"
# );
```

```rust
# use nacfahi::{models::basic::Linear, *};
# use num_traits::Pow;
# 
# // some data, presumably 2x + 1
# let x = [1.0, 2.0, 3.0, 4.0, 5.0];
# let y = [3.0, 5.0, 7.0, 9.0, 11.0];
# 
# // fitting model: a*x + b
# let mut line = Linear { a: 0.0, b: 0.0 };
# 
// custom weight function, really breaking the fit
let weight = |x: f64, y: f64| x.pow(206_265.0) % y / y.ln();

# // proceed to fail the fit, because omg what is this weight function
let report = fit!(&mut line, x, y, weights = weight);
# 
# // expect to stop because of some bad reason, idk
# assert!(
#     !report.termination.was_successful(),
#     "Approximation should fail because of silly weight function"
# );
```

or both (any order will do):

```rust
# use nacfahi::{models::basic::Linear, *};
# use num_traits::Pow;
# 
# // some data, presumably 2x + 1
# let x = [1.0, 2.0, 3.0, 4.0, 5.0];
# let y = [3.0, 5.0, 7.0, 9.0, 11.0];
# 
# // fitting model: a*x + b
# let mut line = Linear { a: 0.0, b: 0.0 };
# 
# // custom very impatient lavmar instance
# let lavmar = LevenbergMarquardt::new().with_patience(1);
# 
# // custom weight function, really breaking the fit
# let weight = |x: f64, y: f64| x.pow(206_265.0) % y / y.ln();
# 
let _report = fit!(&mut line, x, y, weights = weight, minimizer = lavmar);

let _report = fit!(&mut line, x, y, minimizer = lavmar, weights = weight);
```

[`levenberg_marquardt`]: https://docs.rs/levenberg-marquardt/latest/levenberg_marquardt/
