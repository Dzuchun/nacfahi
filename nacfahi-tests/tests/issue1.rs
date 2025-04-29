//! Based on <https://github.com/Dzuchun/nacfahi/issues/1>
#![allow(
    missing_docs,
    missing_debug_implementations,
    clippy::unreadable_literal
)]

use approx::assert_ulps_eq;
use nacfahi::{
    GenericArray, U, fit,
    models::{
        FitModel, FitModelSum,
        basic::{Constant, Exponent},
    },
};

#[test]
fn issue1() {
    let x = [0.0, 1.0, 2.0, 3.0, 4.0];
    let y = [6.0, 44.2, 810.9, 16210.2, 325513.6];

    let mut model = DerivedModel {
        exponential: Exponent { a: 0.0, b: 0.0 },
        constant: Constant { c: 0.0 },
    };
    let report = fit!(&mut model, x, y);
    assert!(report.termination.was_successful());
    assert_ulps_eq!(model.exponential.a, 2.0, epsilon = 1e-3);
    assert_ulps_eq!(model.exponential.b, 3.0, epsilon = 1e-3);
    assert_ulps_eq!(model.constant.c, 4.023, epsilon = 1e-3);

    let mut model = ManualModel {
        a: 0.0,
        b: 0.0,
        c: 0.0,
    };
    let report = fit!(&mut model, x, y);
    assert!(report.termination.was_successful());
    assert_ulps_eq!(model.a, 2.0, epsilon = 1e-3);
    assert_ulps_eq!(model.b, 3.0, epsilon = 1e-3);
    assert_ulps_eq!(model.c, 4.023, epsilon = 1e-3);
}

#[derive(FitModelSum)]
#[scalar_type(f64)]
struct DerivedModel {
    exponential: Exponent<f64>,
    constant: Constant<f64>,
}

struct ManualModel {
    a: f64,
    b: f64,
    c: f64,
}

impl FitModel for ManualModel {
    type Scalar = f64;
    type ParamCount = U<3>;

    fn evaluate(&self, x: &Self::Scalar) -> Self::Scalar {
        self.a * (self.b * x).exp() + self.c
    }

    fn jacobian(
        &self,
        x: &Self::Scalar,
    ) -> impl Into<GenericArray<Self::Scalar, Self::ParamCount>> {
        [(self.b * x).exp(), self.a * x * (self.b * x).exp(), 1.0]
    }

    fn set_params(&mut self, new_params: GenericArray<Self::Scalar, Self::ParamCount>) {
        let [a, b, c] = new_params.into_array();
        self.a = a;
        self.b = b;
        self.c = c;
    }

    fn get_params(&self) -> impl Into<GenericArray<Self::Scalar, Self::ParamCount>> {
        [self.a, self.b, self.c]
    }
}
