crate::test_model_derivative!(
    exponent_gaussian,
    Composition::<Exponent<f64>, Gaussian<f64>>,
    Composition {
        inner: Exponent { a: 5.0, b: -0.1 },
        outer: Gaussian::<_> {
            a: -2.0,
            sigma: 0.4,
            x_c: 3.0,
        }
    },
    [
        (0.0, -1.0),
        (1.0, -4.0),
        (2.0, -5.0),
        (3.0, 6.0),
        (4.0, 2.0),
        (5.0, 2.5)
    ]
);

crate::test_model_derivative!(
    gaussian_exponent,
    Composition::<Gaussian<f64>, Exponent<f64>>,
    Composition {
        inner: Gaussian::<_> {
            a: -2.0,
            sigma: 0.4,
            x_c: 3.0,
        },
        outer: Exponent { a: 5.0, b: -0.1 },
    },
    [
        (0.0, -1.0),
        (1.0, -4.0),
        (2.0, -5.0),
        (3.0, 6.0),
        (4.0, 2.0),
        (5.0, 2.5)
    ]
);

crate::test_model_derivative!(
    exponent_constant,
    Composition::<Exponent<f64>, Constant<f64>>,
    Composition {
        inner: Exponent { a: 5.0, b: -0.1 },
        outer: Constant { c: -4.0 },
    },
    [
        (0.0, -1.0),
        (1.0, -4.0),
        (2.0, -5.0),
        (3.0, 6.0),
        (4.0, 2.0),
        (5.0, 2.5)
    ]
);

crate::test_model_derivative!(
    exponent_linear,
    Composition::<Exponent<f64>, Linear<f64>>,
    Composition {
        inner: Exponent { a: -5.0, b: -3.11 },
        outer: Linear { a: 5.0, b: -0.1 },
    },
    [
        (0.0, -1.0),
        (1.0, -4.0),
        (2.0, -5.0),
        (3.0, 6.0),
        (4.0, 2.0),
        (5.0, 2.5)
    ]
);

crate::test_model_derivative!(
    constant_gassian,
    Composition::<Constant<f64>, Gaussian<f64>>,
    Composition {
        inner: Constant { c: 4.0 },
        outer: Gaussian::<_> {
            a: -5.0,
            sigma: 3.4,
            x_c: -2.1213
        },
    },
    [
        (0.0, -1.0),
        (1.0, -4.0),
        (2.0, -5.0),
        (3.0, 6.0),
        (4.0, 2.0),
        (5.0, 2.5)
    ]
);

crate::test_model_derivative!(
    gaussian_linear,
    Composition::<Gaussian<f64>, Linear<f64>>,
    Composition {
        inner: Gaussian::<_> {
            a: -5.0,
            sigma: 3.4,
            x_c: -2.1213
        },
        outer: Linear {
            a: -0.2421,
            b: 4.321
        },
    },
    [
        (0.0, -1.0),
        (1.0, -4.0),
        (2.0, -5.0),
        (3.0, 6.0),
        (4.0, 2.0),
        (5.0, 2.5)
    ]
);
