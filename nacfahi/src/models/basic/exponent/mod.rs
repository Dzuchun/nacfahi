use core::ops::Mul;

use generic_array::GenericArray;
use num_traits::{FloatConst, Pow};
use typenum::U2;

use crate::models::{FitModel, FitModelErrors, FitModelXDeriv};

/// Exponent model $a \cdot \exp(b \cdot x )$
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Exponent<Scalar> {
    /// Constant in front of the exponent
    pub a: Scalar,
    /// Exponent multiplier
    pub b: Scalar,
}

impl<Scalar: Clone + Mul<Output = Scalar> + Pow<Scalar, Output = Scalar> + FloatConst> FitModel
    for Exponent<Scalar>
{
    type Scalar = Scalar;
    type ParamCount = U2;

    #[inline]
    fn evaluate(&self, x: &Self::Scalar) -> Scalar {
        self.a.clone() * (Self::Scalar::E().pow(self.b.clone() * x.clone()))
    }

    #[inline]
    fn jacobian(
        &self,
        x: &Self::Scalar,
    ) -> impl Into<GenericArray<Self::Scalar, Self::ParamCount>> {
        // y = a * exp(bx)
        // - derivative over a is exp(bx)
        // - derivative over b is ax * exp(bx)
        let e_x = Self::Scalar::E().pow(self.b.clone() * x.clone());
        [e_x.clone(), self.a.clone() * x.clone() * e_x]
    }

    #[inline]
    fn set_params(&mut self, new_params: GenericArray<Self::Scalar, Self::ParamCount>) {
        let [new_a, new_b] = new_params.into_array();
        self.a = new_a;
        self.b = new_b;
    }

    #[inline]
    fn get_params(&self) -> impl Into<GenericArray<Self::Scalar, Self::ParamCount>> {
        [self.a.clone(), self.b.clone()]
    }
}

impl<Scalar: Clone + Mul<Output = Scalar> + Pow<Scalar, Output = Scalar> + FloatConst>
    FitModelXDeriv for Exponent<Scalar>
{
    #[inline]
    fn deriv_x(&self, x: &Self::Scalar) -> Self::Scalar {
        // y = a * exp(bx)
        // - derivative over x is ab * exp(bx)
        self.a.clone() * self.b.clone() * Scalar::E().pow(self.b.clone() * x.clone())
    }
}

impl<Scalar: 'static> FitModelErrors for Exponent<Scalar>
where
    Scalar: Clone + Mul<Output = Scalar> + Pow<Scalar, Output = Scalar> + FloatConst,
{
    type OwnedModel = Self;

    #[inline]
    fn with_errors(errors: GenericArray<Self::Scalar, Self::ParamCount>) -> Self::OwnedModel {
        let [a, b] = errors.into_array();
        Exponent { a, b }
    }
}

#[cfg(test)]
mod tests;
