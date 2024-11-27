crate::test_model_derivative!(
    Exponent::<f64>,
    Exponent { a: -2.0, b: -10.0 },
    [
        (0.0, -1.0),
        (1.0, -4.0),
        (2.0, -5.0),
        (3.0, 6.0),
        (4.0, 2.0),
        (5.0, 2.5)
    ]
);
