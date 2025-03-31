#![allow(missing_docs, missing_debug_implementations)]

use generic_array::sequence::{Concat, Split};
use nacfahi::models::{
    FitModel,
    basic::{Exponent, Gaussian, Linear},
};

pub struct CustomModel {
    linear: Linear<f64>,
    exponent: Exponent<f64>,
    peaks: [Gaussian<f64>; 3],
}

impl FitModel for CustomModel {
    type Scalar = f64;
    type ParamCount = typenum::U13;

    fn evaluate(&self, x: &f64) -> f64 {
        self.linear.evaluate(x) + self.exponent.evaluate(x) + self.peaks.evaluate(x)
    }

    fn jacobian(&self, x: &f64) -> impl Into<generic_array::GenericArray<f64, Self::ParamCount>> {
        let linear = self.linear.jacobian(x).into();
        let exponent = self.exponent.jacobian(x).into();
        let peaks = self.peaks.jacobian(x).into();
        linear.concat(exponent).concat(peaks)
    }

    fn set_params(&mut self, new_params: generic_array::GenericArray<f64, Self::ParamCount>) {
        let (peaks, rest) = new_params.split();
        let (linear, exponent) = rest.split();

        self.linear.set_params(linear);
        self.exponent.set_params(exponent);
        self.peaks.set_params(peaks);
    }

    fn get_params(&self) -> impl Into<generic_array::GenericArray<f64, Self::ParamCount>> {
        let linear = self.linear.get_params().into();
        let exponent = self.exponent.get_params().into();
        let peaks = self.peaks.get_params().into();
        linear.concat(exponent).concat(peaks)
    }
}
