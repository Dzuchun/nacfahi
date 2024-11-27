crate::test_model_derivative!(
    range_to,
    Ranged<Gaussian<f64>, core::ops::RangeTo<f64>>,
    Ranged {
        inner: Gaussian::<_> {
            a: -3.0,
            x_c: 1.0,
            sigma: 0.3,
        },
        range: ..2.0,
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
    range_from,
    Ranged<Gaussian<f64>, core::ops::RangeFrom<f64>>,
    Ranged {
        inner: Gaussian::<_> {
            a: -3.0,
            x_c: 1.0,
            sigma: 0.3,
        },
        range: 0.0..,
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
    range,
    Ranged<Gaussian<f64>, core::ops::Range<f64>>,
    Ranged {
        inner: Gaussian::<_> {
            a: -3.0,
            x_c: 1.0,
            sigma: 0.3,
        },
        range: -0.5..3.0,
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
