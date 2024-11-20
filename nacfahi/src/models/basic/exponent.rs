use core::ops::Mul;

use num_traits::{FloatConst, Pow};

use crate::models::{FitModel, FitModelXDeriv};

/// Exponent model $a \cdot \exp(b \cdot x )$
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Exponent<Scalar> {
    /// Constant in front of the exponent
    pub a: Scalar,
    /// Exponent multiplier
    pub b: Scalar,
}

impl<Scalar: Clone + Mul<Output = Scalar> + Pow<Scalar, Output = Scalar> + FloatConst>
    FitModel<Scalar> for Exponent<Scalar>
{
    type ParamCount = typenum::U2;

    #[inline]
    fn evaluate(&self, x: &Scalar) -> Scalar {
        self.a.clone() * (Scalar::E().pow(self.b.clone() * x.clone()))
    }

    #[inline]
    fn jacobian(
        &self,
        x: &Scalar,
    ) -> impl Into<generic_array::GenericArray<Scalar, Self::ParamCount>> {
        // y = a * exp(bx)
        // - derivative over a is exp(bx)
        // - derivative over b is ax * exp(bx)
        let e_x = Scalar::E().pow(self.b.clone() * x.clone());
        [e_x.clone(), self.a.clone() * x.clone() * e_x]
    }

    #[inline]
    fn set_params(&mut self, new_params: generic_array::GenericArray<Scalar, Self::ParamCount>) {
        let [new_a, new_b] = new_params.into_array();
        self.a = new_a;
        self.b = new_b;
    }

    #[inline]
    fn get_params(&self) -> impl Into<generic_array::GenericArray<Scalar, Self::ParamCount>> {
        [self.a.clone(), self.b.clone()]
    }
}

impl<Scalar: Clone + Mul<Output = Scalar> + Pow<Scalar, Output = Scalar> + FloatConst>
    FitModelXDeriv<Scalar> for Exponent<Scalar>
{
    #[inline]
    fn deriv_x(&self, x: &Scalar) -> Scalar {
        // y = a * exp(bx)
        // - derivative over x is ab * exp(bx)
        self.a.clone() * self.b.clone() * Scalar::E().pow(self.b.clone() * x.clone())
    }
}
