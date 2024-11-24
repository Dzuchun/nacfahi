use crate::models::{FitModel, FitModelErrors, FitModelXDeriv};
use generic_array::GenericArray;
use num_traits::{One, Zero};
use typenum::U1;

/// Model representing a constant, independent of `x`
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Constant<Scalar> {
    /// The constant itself
    pub c: Scalar,
}

impl<Scalar: Clone + One> FitModel for Constant<Scalar> {
    type Scalar = Scalar;
    type ParamCount = U1;

    #[inline]
    fn evaluate(&self, _: &Self::Scalar) -> Self::Scalar {
        self.c.clone()
    }

    #[inline]
    fn jacobian(
        &self,
        _: &Self::Scalar,
    ) -> impl Into<GenericArray<Self::Scalar, Self::ParamCount>> {
        [Scalar::one()]
    }

    #[inline]
    fn set_params(&mut self, new_params: GenericArray<Self::Scalar, Self::ParamCount>) {
        let [new_c] = new_params.into_array();
        self.c = new_c;
    }

    #[inline]
    fn get_params(&self) -> impl Into<GenericArray<Self::Scalar, Self::ParamCount>> {
        [self.c.clone()]
    }
}

impl<Scalar: Clone + Zero + One> FitModelXDeriv for Constant<Scalar> {
    #[inline]
    fn deriv_x(&self, _: &Self::Scalar) -> Scalar {
        Scalar::zero()
    }
}

impl<Scalar> FitModelErrors for Constant<Scalar>
where
    Self: FitModel<Scalar = Scalar, ParamCount = U1>,
{
    type OwnedModel = Self;

    #[inline]
    fn with_errors(
        &self,
        errors: GenericArray<Self::Scalar, Self::ParamCount>,
    ) -> Self::OwnedModel {
        let [c] = errors.into_array();
        Self { c }
    }
}

crate::test_model_derivative!(
    Constant::<f64>,
    Constant { c: 5.0 },
    [
        (0.0, -1.0),
        (1.0, -4.0),
        (2.0, -5.0),
        (3.0, 6.0),
        (4.0, 2.0),
        (5.0, 2.5)
    ]
);
