#![allow(missing_docs, unused_variables)]

use core::f64;
use std::ops::RangeTo;

use approx::{assert_ulps_eq, assert_ulps_ne};
use models::{
    basic::{Exponent, Gaussian, Linear},
    utility::{model_map, CompositionExt, Fixed, LnMap, Ranged},
    FitModel,
};
use nacfahi::*;
use num_traits::Float;
use static_assertions::assert_impl_all;

#[test]
fn sharp_exponent() {
    type SharpExponent<Scalar> = Ranged<Exponent<Scalar>, RangeTo<Scalar>>;
    assert_impl_all!(SharpExponent<f64>: FitModel);

    // this thing will be an exponent only present at x < 0
    let mut chirp: SharpExponent<f64> = Ranged {
        inner: Exponent { a: 0.0, b: 0.0 },
        range: ..0.0,
    };

    // data consisting of exponential data at x < 0, and some garbage at x > 0
    let expected_a = -5.0;
    let expected_b = 2.5;
    let x = [
        -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
    ];
    let mut y = x.map(|x| expected_a * (expected_b * x).exp());
    y[6] = -23.0;
    y[7] = 43.0;
    y[8] = 0.0;
    y[9] = -5.0;
    y[10] = 2.0;
    y[11] = -7.0;

    // fit!
    let report = fit!(&mut chirp, x, y);

    assert!(
        report.termination.was_successful(),
        "Fit must be successful: {report:?}"
    );
    dbg!(report);
    assert_ulps_eq!(chirp.inner.a, expected_a, epsilon = 1e-6);
    assert_ulps_eq!(chirp.inner.b, expected_b, epsilon = 1e-6);

    // now, let's extend the range past all the values:
    chirp.range = ..10.0;
    chirp.inner.a = 0.0;
    chirp.inner.b = 0.0;

    // fit! (again)
    let _report = fit!(&mut chirp, x, y);

    assert_ulps_ne!(chirp.inner.a, expected_a, epsilon = 1e-6);
    assert_ulps_ne!(chirp.inner.b, expected_b, epsilon = 1e-6);
}

#[test]
fn fit_log() {
    // some exponential data
    let expected_a = 3.0;
    let expected_b = 0.5;
    let x = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    let y = x.map(|x| expected_a * (x * expected_b).exp());
    let linear_y = y.map(f64::ln);

    // exponential model
    let mut expo_model = Exponent { a: 1.0, b: 0.0 };
    // expolinear (exponential mapped to linear)
    let expolinear = model_map(&mut expo_model, LnMap);

    // fit!
    let report = fit!(expolinear, x, linear_y);

    assert!(
        report.termination.was_successful(),
        "Fit should be successful {report:?}"
    );
    assert_ulps_eq!(expo_model.a, expected_a);
    assert_ulps_eq!(expo_model.b, expected_b);
}

#[test]
#[ignore = "yet-untested"]
fn gaussian_over_exp() {
    // some data
    let expected_a = 1.0;
    let expected_b = 0.5;

    let expected_s = 1.0;
    let expected_w = 5.0;
    let expected_x_c = 0.5;

    #[allow(clippy::cast_precision_loss, reason = "da;dc")]
    let x = core::array::from_fn::<f64, 100, _>(|i| -2.0 + 0.04 * (i as f64));
    let y = x.map(|x| {
        let e = expected_a * f64::exp(-expected_b * x);
        expected_s
            * f64::sqrt(4.0 * f64::consts::LN_2 / (f64::consts::PI * expected_w * expected_w))
            * f64::exp(
                -4.0 * f64::consts::LN_2 * (e - expected_x_c) * (e - expected_x_c)
                    / (expected_w * expected_w),
            )
    });

    // composed model
    let mut composed = Exponent { a: 0.0, b: 0.0 }.compose(Gaussian {
        a: 0.4,
        s: 0.2,
        x_c: 0.3,
    });

    // fit!
    let report = fit!(&mut composed, x, y);

    assert!(
        report.termination.was_successful(),
        "Fit should be successful {report:?}"
    );
    assert_ulps_eq!(composed.inner.a, expected_a);
    assert_ulps_eq!(composed.inner.b, expected_b);
    assert_ulps_eq!(composed.outer.a, expected_s);
    assert_ulps_eq!(composed.outer.s, expected_w);
    assert_ulps_eq!(composed.outer.s, expected_x_c);
}

#[test]
fn fixed() {
    // some model
    let model = Linear { a: 0.0, b: 0.0 };

    // make it fixed
    let fixed = Fixed(model);

    // fit!(fixed, [1.0, 2.0, 3.0], [-1.0, 5.0, -6.0]); // <-- does not compile!
}
