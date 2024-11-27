crate::test_model_derivative!(
    Constant::<f64>,
    Constant { c: 5.0 },
    [
        (0.0, -1.0),
        (1.0, -4.0),
        (2.0, -5.0),
        (3.0, 6.0),
        (4.0, 2.0),
        (5.0, 2.5)
    ]
);
