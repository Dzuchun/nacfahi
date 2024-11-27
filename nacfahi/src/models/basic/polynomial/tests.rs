crate::test_model_derivative!(
    Polynomial::<5, f64>,
    Polynomial {
        params: [-4.0, -15.0, -2.0, 0.7, -0.01]
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
