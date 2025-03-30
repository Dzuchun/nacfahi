#![allow(missing_docs)]

use nacfahi::{
    AsMatrixView, FitErrBound, FitterUnit, fit_stat,
    models::{FitModelErrors, LevMarModel, basic::Polynomial},
};

#[cfg(feature = "num-traits")]
pub fn fit_err_generic<
    const ORDER: usize,
    Scalar: num_traits::Zero,
    X: AsMatrixView<Scalar = Scalar>,
    Y: AsMatrixView<Scalar = Scalar, Points = X::Points>,
>(
    x: X,
    y: Y,
) where
    Polynomial<ORDER, Scalar>: FitModelErrors + LevMarModel,
    FitterUnit: FitErrBound<Polynomial<ORDER, X::Scalar>, X, Y>,
{
    let mut model = Polynomial {
        params: core::array::from_fn(|_| Scalar::zero()),
    };

    let stat = fit_stat!(&mut model, x, y);

    model.params[0] = Scalar::zero(); // test for "borrowed model"

    core::hint::black_box(stat);
}

pub fn fit_err_generic2<
    const ORDER: usize,
    X: AsMatrixView<Scalar = f64>,
    Y: AsMatrixView<Scalar = f64, Points = X::Points>,
>(
    x: X,
    y: Y,
) where
    Polynomial<ORDER, f64>: FitModelErrors + LevMarModel,
    FitterUnit: FitErrBound<Polynomial<ORDER, X::Scalar>, X, Y>,
{
    let mut model = Polynomial {
        params: [0.0; ORDER],
    };

    let stat = fit_stat!(&mut model, x, y);

    model.params[0] = 0.0; // test for "borrowed model"

    core::hint::black_box(stat);
}
