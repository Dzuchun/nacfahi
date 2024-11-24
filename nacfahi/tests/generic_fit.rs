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
    for<'r> FitterUnit:
        FitErrBound<&'r mut Polynomial<ORDER, f64>, GenericArray<f64, Len>, GenericArray<f64, Len>>,
{
    let mut model = Polynomial {
        params: [0.0f64; ORDER],
    };

    let stat = fit_stat!(&mut model, x, y);
}
