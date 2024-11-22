#![allow(missing_docs)]

use generic_array::GenericArray;

#[test]
fn array() {
    use approx::assert_ulps_eq;
    use nacfahi::{models::basic::Linear, *};

    // some data, presumably 2x + 1
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [3.0, 5.0, 7.0, 9.0, 11.0];

    // fitting model: a*x + b
    let mut line = Linear { a: 0.0, b: 0.0 };

    // do the fit!
    let report = fit!(&mut line, x, y);

    // check that approximation is successful
    assert!(
        report.termination.was_successful(),
        "Approximation should be successful"
    );

    // check that model parameters have expected values
    assert_ulps_eq!(line.a, 2.0);
    assert_ulps_eq!(line.b, 1.0);
}

#[test]
fn slice() {
    use approx::assert_ulps_eq;
    use nacfahi::{models::basic::Linear, *};

    // some data, presumably 2x + 1
    let x = &[1.0, 2.0, 3.0, 4.0, 5.0][..];
    let y = &[3.0, 5.0, 7.0, 9.0, 11.0][..];

    // fitting model: a*x + b
    let mut line = Linear { a: 0.0, b: 0.0 };

    // do the fit!
    let report = fit!(&mut line, x, y);

    // check that approximation is successful
    assert!(
        report.termination.was_successful(),
        "Approximation should be successful"
    );

    // check that model parameters have expected values
    assert_ulps_eq!(line.a, 2.0);
    assert_ulps_eq!(line.b, 1.0);
}

#[test]
fn matrix() {
    use approx::assert_ulps_eq;
    use nacfahi::{models::basic::Linear, *};
    use nalgebra::matrix;

    // some data, presumably 2x + 1
    let x = matrix![1.0; 2.0; 3.0; 4.0; 5.0];
    let y = matrix![3.0; 5.0; 7.0; 9.0; 11.0];

    // fitting model: a*x + b
    let mut line = Linear { a: 0.0, b: 0.0 };

    // do the fit!
    let report = fit!(&mut line, x, y);

    // check that approximation is successful
    assert!(
        report.termination.was_successful(),
        "Approximation should be successful"
    );

    // check that model parameters have expected values
    assert_ulps_eq!(line.a, 2.0);
    assert_ulps_eq!(line.b, 1.0);
}

#[test]
fn matrix_view() {
    use approx::assert_ulps_eq;
    use nacfahi::{models::basic::Linear, *};
    use nalgebra::matrix;

    // some data, presumably 2x + 1
    let x = matrix![1.0; 2.0; 3.0; 4.0; 5.0];
    let y = matrix![3.0; 5.0; 7.0; 9.0; 11.0];

    // fitting model: a*x + b
    let mut line = Linear { a: 0.0, b: 0.0 };

    // do the fit!
    let report = fit!(&mut line, x.column(0), y.column(0));

    // check that approximation is successful
    assert!(
        report.termination.was_successful(),
        "Approximation should be successful"
    );

    // check that model parameters have expected values
    assert_ulps_eq!(line.a, 2.0);
    assert_ulps_eq!(line.b, 1.0);
}

#[test]
fn dyn_matrix() {
    use approx::assert_ulps_eq;
    use nacfahi::{models::basic::Linear, *};
    use nalgebra::{Dyn, Vector};

    // some data, presumably 2x + 1
    let x = Vector::<_, Dyn, _>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = x.map(|x| 2.0 * x + 1.0);

    // fitting model: a*x + b
    let mut line = Linear { a: 0.0, b: 0.0 };

    // do the fit!
    let report = fit!(&mut line, x, y);

    // check that approximation is successful
    assert!(
        report.termination.was_successful(),
        "Approximation should be successful"
    );

    // check that model parameters have expected values
    assert_ulps_eq!(line.a, 2.0);
    assert_ulps_eq!(line.b, 1.0);
}

#[test]
fn generic_array() {
    use approx::assert_ulps_eq;
    use nacfahi::{models::basic::Linear, *};

    // some data, presumably 2x + 1
    let x = GenericArray::from_array([1.0, 2.0, 3.0, 4.0, 5.0]);
    let y = GenericArray::from_array([3.0, 5.0, 7.0, 9.0, 11.0]);

    // fitting model: a*x + b
    let mut line = Linear { a: 0.0, b: 0.0 };

    // do the fit!
    let report = fit!(&mut line, x, y);

    // check that approximation is successful
    assert!(
        report.termination.was_successful(),
        "Approximation should be successful"
    );

    // check that model parameters have expected values
    assert_ulps_eq!(line.a, 2.0);
    assert_ulps_eq!(line.b, 1.0);
}
