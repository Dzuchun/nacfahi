crate::test_model_derivative!(
    exponent_offset,
    ModelMap<Exponent<f64>, Addition<f64>>,
    model_map(Exponent { a: 2.0, b: -0.242 }, Addition(-34.2)),
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
    exponent_multiplier,
    ModelMap<Exponent<f64>, Multiplier<f64>>,
    model_map(Exponent { a: 2.0, b: -0.242 }, Multiplier(-3.232)),
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
    exponent_power,
    ModelMap<Exponent<f64>, Power<f64>>,
    model_map(Exponent { a: 2.0, b: -0.242 }, Power(3.0)),
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
    gaussian_power,
    ModelMap<Gaussian<f64>, Power<f64>>,
    model_map(
        Gaussian::<_> {
            a: -3.0,
            sigma: 0.3,
            x_c: 3.0
        },
        Power(3.0)
    ),
    [
        (0.0, -1.0),
        (1.0, -4.0),
        (2.0, -5.0),
        (3.0, 6.0),
        (4.0, 2.0),
        (5.0, 2.5)
    ]
);
