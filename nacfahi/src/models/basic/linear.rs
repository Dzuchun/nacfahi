use core::ops::{Add, Mul};

use generic_array::GenericArray;
use num_traits::One;

use typenum::U2;

use crate::models::{FitModel, FitModelXDeriv};

/// Line model $a \cdot x + b$
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Linear<Scalar> {
    /// Line tangent, $a$
    pub a: Scalar,
    /// Line offset, $b$
    pub b: Scalar,
}

impl<Scalar: Clone + Add<Output = Scalar> + Mul<Output = Scalar> + One> FitModel<Scalar>
    for Linear<Scalar>
{
    type ParamCount = U2;

    #[inline]
    fn evaluate(&self, x: &Scalar) -> Scalar {
        self.a.clone() * x.clone() + self.b.clone()
    }

    #[inline]
    fn jacobian(&self, x: &Scalar) -> impl Into<GenericArray<Scalar, Self::ParamCount>> {
        // y = a * x + b
        // - derivative over a is x
        // - derivative over b is 1
        [x.clone(), Scalar::one()]
    }

    #[inline]
    fn set_params(&mut self, new_params: GenericArray<Scalar, Self::ParamCount>) {
        let [new_a, new_b] = new_params.into_array();
        self.a = new_a;
        self.b = new_b;
    }

    #[inline]
    fn get_params(&self) -> impl Into<GenericArray<Scalar, Self::ParamCount>> {
        [self.a.clone(), self.b.clone()]
    }
}

impl<Scalar: Clone> FitModelXDeriv<Scalar> for Linear<Scalar> {
    #[inline]
    fn deriv_x(&self, _x: &Scalar) -> Scalar {
        // y = a * x + b
        // - derivative over x is a
        self.a.clone()
    }
}
