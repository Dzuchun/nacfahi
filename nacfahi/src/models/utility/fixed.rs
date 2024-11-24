use crate::models::{FitModel, FitModelErrors, FitModelXDeriv};

use generic_array::GenericArray;
use generic_array_storage::Conv;
use typenum::U0;

/// Model always having 0 parameters, regardless of model inside.
///
/// Point is to forbid fitting of parameters in a certain part of a model.
///
/// Note, that main interface ([`fit`](function@crate::fit) and alike) require model with at least one parameter, so you can't pass this model directly:
///
/// ```rust,compile_fail
/// # use nacfahi::{models::{basic::Linear, utility::Fixed}, fit};
/// // some model
/// let model = Linear { a: 0.0, b: 0.0 };
///
/// // make it fixed
/// let fixed = Fixed(model);
///
/// fit!(fixed, [1.0, 2.0, 3.0], [-1.0, 5.0, -6.0]); // <-- does not compile!
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Fixed<Model>(pub Model);

impl<Model> FitModel for Fixed<Model>
where
    Model: FitModel,
{
    type Scalar = Model::Scalar;
    type ParamCount = U0;

    fn evaluate(&self, x: &Self::Scalar) -> Self::Scalar {
        self.0.evaluate(x)
    }

    fn jacobian(
        &self,
        _x: &Self::Scalar,
    ) -> impl Into<GenericArray<Self::Scalar, Self::ParamCount>> {
        []
    }

    fn set_params(&mut self, _new_params: GenericArray<Self::Scalar, Self::ParamCount>) {}

    fn get_params(&self) -> impl Into<GenericArray<Self::Scalar, Self::ParamCount>> {
        []
    }
}

impl<Model> FitModelXDeriv for Fixed<Model>
where
    Model: FitModelXDeriv,
{
    fn deriv_x(&self, x: &Self::Scalar) -> Self::Scalar {
        // derivative over parameters might be zero, but technically, there's no reason to zero-out this one
        self.0.deriv_x(x)
    }
}

impl<Model> FitModelErrors for Fixed<Model>
where
    Self: FitModel,
{
    type OwnedModel = ();

    fn with_errors(
        &self,
        _errors: GenericArray<Self::Scalar, <Self::ParamCount as Conv>::TNum>,
    ) -> Self::OwnedModel {
    }
}
