Generates a test to numerically check, if your model has a correct jacobian implementation.

Most of the time, you don't need to do that - all of the exported models do have such tests, and implementations generated by derive macro should generate one too (**NOTE**: this is not the case yet). There's a point in this macro only if you are implementing [`FitModel`] directly.

In case you are creating multiple of these at the same spot, you can specify `isolation_module` ident to *isolate* the test in a *separate module* (helps with name clashing).

Internally, this macro glues [`FitModel`] implementation to [`levenberg_marquardt::differentiate_numerically`](https://docs.rs/levenberg-marquardt/latest/levenberg_marquardt/fn.differentiate_numerically.html)-compatible one, and tests the jacobian on a provided test of points.

### Example
```rust
// Note: for convenience, macro autoimports `nacfahi::models::{basic::*, utility::*}`
nacfahi::test_model_derivative!(
    Linear::<f64>,
    Linear { a: 5.0, b: -4.0 },
    [
        (0.0, -1.0),
        (1.0, -4.0),
        (2.0, -5.0),
        (3.0, 6.0),
        (4.0, 2.0),
        (5.0, 2.5)
    ]
);
```