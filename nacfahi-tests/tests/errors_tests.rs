#![allow(missing_docs)]
#![cfg(feature = "alloc")]

use approx::assert_ulps_eq;
use nacfahi::{FitStat, LevenbergMarquardt, fit_stat, models::basic::Constant};
use rand::{Rng, rng};

#[test]
#[cfg_attr(miri, ignore = "Takes TOO LONG")]
fn constant_stdev() {
    // some data, x can be whatever
    const SAMPLE_SIZE: usize = 10_000;
    #[allow(clippy::large_stack_arrays, reason = "It's a test!")]
    let x = [0.0f64; SAMPLE_SIZE];
    let y = core::array::from_fn::<f64, SAMPLE_SIZE, _>(|_| rng().random_range(0.0..=5.0));

    // compute expected values
    #[allow(clippy::cast_precision_loss)]
    let sample_size = SAMPLE_SIZE as f64;
    let y_mean = y.iter().sum::<f64>() / sample_size;
    let y_err = f64::sqrt(
        y.iter()
            .map(|y_i| y_i - y_mean)
            .map(|y_dev| y_dev * y_dev)
            .sum::<f64>()
            / (sample_size)
            / (sample_size - 1.0),
    );

    // constant model
    let mut model = Constant { c: 0.0 };

    // fit the model
    let FitStat {
        report,
        errors,
        covariance_matrix,
        ..
    } = fit_stat!(
        &mut model,
        x.as_slice(),
        y.as_slice(),
        minimizer = LevenbergMarquardt::new()
    );

    // assert values
    dbg!(y, report, covariance_matrix);
    assert_ulps_eq!(model.c, y_mean, epsilon = 1e-10);
    assert_ulps_eq!(errors.c, y_err, epsilon = 1e-1); // idk what exactly am I doing at this point :death_emoji:
}
