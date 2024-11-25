#![allow(missing_docs, unused)]

use generic_array::{ArrayLength, GenericArray};
use generic_array_storage::Conv;
use nacfahi::{
    fit_stat,
    models::{basic::Polynomial, FitModelErrors},
    FitBound, FitErrBound, FitterUnit, LevenbergMarquardt,
};
use typenum::Const;

fn fit_err_generic<const ORDER: usize, Len: ArrayLength>(
    x: GenericArray<f64, Len>,
    y: GenericArray<f64, Len>,
) where
    Polynomial<ORDER, f64>: FitModelErrors<Scalar = f64>,
    FitterUnit: FitErrBound<Polynomial<ORDER, f64>, GenericArray<f64, Len>>,
{
    let mut model = Polynomial {
        params: [0.0f64; ORDER],
    };

    let stat = fit_stat!(&mut model, x, y);

    model.params[0] = 0.0; // test for "borrowed model"

    core::hint::black_box(stat);
}
