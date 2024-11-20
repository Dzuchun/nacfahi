use crate::models::{FitModel, FitModelXDeriv};
use generic_array::GenericArray;
use num_traits::{One, Zero};
use typenum::U1;

/// Model representing a constant, independent of `x`
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Constant<Scalar> {
    /// The constant itself
    pub c: Scalar,
}

impl<Scalar: Clone + One> FitModel<Scalar> for Constant<Scalar> {
    type ParamCount = U1;

    #[inline]
    fn evaluate(&self, _: &Scalar) -> Scalar {
        self.c.clone()
    }

    #[inline]
    fn jacobian(&self, _: &Scalar) -> impl Into<GenericArray<Scalar, Self::ParamCount>> {
        [Scalar::one()]
    }

    #[inline]
    fn set_params(&mut self, new_params: GenericArray<Scalar, Self::ParamCount>) {
        let [new_c] = new_params.into_array();
        self.c = new_c;
    }

    #[inline]
    fn get_params(&self) -> impl Into<GenericArray<Scalar, Self::ParamCount>> {
        [self.c.clone()]
    }
}

impl<Scalar: Clone + Zero + One> FitModelXDeriv<Scalar> for Constant<Scalar> {
    #[inline]
    fn deriv_x(&self, _: &Scalar) -> Scalar {
        Scalar::zero()
    }
}
