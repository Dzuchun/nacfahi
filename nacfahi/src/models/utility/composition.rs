use core::ops::{Add, Mul, Sub};

use generic_array::{
    functional::FunctionalSequence,
    sequence::{Concat, Split},
    ArrayLength, GenericArray,
};
use typenum::Sum;

use crate::models::{FitModel, FitModelXDeriv};

/// A model equal to consequent application of `inner` and `outer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Composition<Inner, Outer> {
    #[allow(missing_docs)]
    pub inner: Inner,
    #[allow(missing_docs)]
    pub outer: Outer,
}

impl<Scalar, Inner, Outer> FitModel<Scalar> for Composition<Inner, Outer>
where
    Scalar: Clone + Mul<Scalar, Output = Scalar>,
    Inner: FitModel<Scalar>,
    Outer: FitModel<Scalar> + FitModelXDeriv<Scalar>,
    Inner::ParamCount: Add<Outer::ParamCount>,
    Sum<Inner::ParamCount, Outer::ParamCount>:
        ArrayLength + Sub<Inner::ParamCount, Output = Outer::ParamCount>,
{
    type ParamCount = Sum<Inner::ParamCount, Outer::ParamCount>;

    #[inline]
    fn evaluate(&self, x: &Scalar) -> Scalar {
        let y = self.inner.evaluate(x);
        self.outer.evaluate(&y)
    }

    #[inline]
    fn jacobian(&self, x: &Scalar) -> impl Into<GenericArray<Scalar, Self::ParamCount>> {
        let y = self.inner.evaluate(x);
        let j_in_x = self.inner.jacobian(x).into();
        let j_out_y = self.outer.jacobian(&y).into();

        let z_y = self.outer.deriv_x(&y);
        let j_out_x = j_in_x.map(|d| z_y.clone() * d);

        GenericArray::concat(j_out_x, j_out_y)
    }

    #[inline]
    fn set_params(&mut self, new_params: GenericArray<Scalar, Self::ParamCount>) {
        let (inner_params, outer_params) = GenericArray::split(new_params);
        self.inner.set_params(inner_params);
        self.outer.set_params(outer_params);
    }

    #[inline]
    fn get_params(&self) -> impl Into<GenericArray<Scalar, Self::ParamCount>> {
        GenericArray::concat(
            self.inner.get_params().into(),
            self.outer.get_params().into(),
        )
    }
}

impl<Scalar: Mul<Output = Scalar>, Inner, Outer> FitModelXDeriv<Scalar>
    for Composition<Inner, Outer>
where
    Inner: FitModel<Scalar> + FitModelXDeriv<Scalar>,
    Outer: FitModelXDeriv<Scalar>,
{
    #[inline]
    fn deriv_x(&self, x: &Scalar) -> Scalar {
        let y = self.inner.evaluate(x);
        self.inner.deriv_x(x) * self.outer.deriv_x(&y)
    }
}

