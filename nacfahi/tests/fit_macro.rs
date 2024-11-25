#![allow(missing_docs)]

use nacfahi::{fit, models::basic::Constant};

#[test]
fn custom_lavmar() {
    use approx::assert_ulps_ne;
    use nacfahi::{models::basic::Exponent, *};

    // some data, presumably 2x + 1
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [3.0, 5.0, 7.0, 9.0, 11.0];

    // fitting model: a * exp(-b*x)
    let mut line = Exponent { a: 0.0, b: 0.0 };

    // custom very impatient lavmar instance
    let lavmar = LevenbergMarquardt::new().with_patience(1);

    // patience too low, proceed to fail the fit :3
    let report = fit!(&mut line, x, y, minimizer = lavmar);

    // expect to stop because of patience
    assert_eq!(
        report.termination,
        TerminationReason::LostPatience,
        "Approximation should stop because of patience"
    );

    // no reason to assume this, but oh well
    assert_ulps_ne!(line.a, 2.0);
    assert_ulps_ne!(line.b, 1.0);

    // since we use different model, let's check that it's still able to fit, eventually:
    line.a = 0.0;
    line.b = 0.0;

    // do the fit! (again)
    let report = fit!(&mut line, x, y);

    // check that approximation is successful
    assert!(
        report.termination.was_successful(),
        "Approximation should be successful"
    );
}

#[test]
fn custom_weight() {
    use nacfahi::{models::basic::Linear, *};
    use num_traits::Pow;

    // some data, presumably 2x + 1
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [3.0, 5.0, 7.0, 9.0, 11.0];

    // fitting model: a*x + b
    let mut line = Linear { a: 0.0, b: 0.0 };

    // custom weight function, really breaking the fit
    let weight = |x: f64, y: f64| x.pow(206_265.0) % y / y.ln();

    // proceed to fail the fit, because omg what is this weight function
    let report = fit!(&mut line, x, y, weights = weight);

    // expect to stop because of some bad reason, idk
    assert!(
        !report.termination.was_successful(),
        "Approximation should fail because of silly weight function"
    );
}

#[test]
fn custom_lavmar_weight() {
    use nacfahi::{models::basic::Linear, *};
    use num_traits::Pow;

    // some data, presumably 2x + 1
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [3.0, 5.0, 7.0, 9.0, 11.0];

    // fitting model: a*x + b
    let mut line = Linear { a: 0.0, b: 0.0 };

    // custom very impatient lavmar instance
    let lavmar = LevenbergMarquardt::new().with_patience(1);

    // custom weight function, really breaking the fit
    let weight = |x: f64, y: f64| x.pow(206_265.0) % y / y.ln();

    let _report = fit!(&mut line, x, y, weights = weight, minimizer = lavmar);

    let _report = fit!(&mut line, x, y, minimizer = lavmar, weights = weight);
}

#[test]
#[should_panic = "Weight function panic"]
fn weight_function_respected() {
    let x = [0.0];
    let y = [1.0];
    let mut model = Constant { c: 0.0 };
    let weights = |_: f64, _: f64| panic!("Weight function panic");

    let _ = fit!(&mut model, x, y, weights = weights);
}

#[test]
fn default_weights_unused() {
    let x = [0.0];
    let y = [1.0];
    let mut model = Constant { c: 0.0 };
    #[allow(unused)]
    let default_weights = |_: f64, _: f64| panic!("Weight function panic");

    let _ = fit!(&mut model, x, y, weights = default_weights);
}
