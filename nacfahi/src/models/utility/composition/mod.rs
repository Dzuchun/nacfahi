use core::ops::{Add, Mul, Sub};

use generic_array::{
    functional::FunctionalSequence,
    sequence::{Concat, Split},
    ArrayLength, GenericArray,
};
use generic_array_storage::Conv;
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

impl<Inner, Outer> FitModel for Composition<Inner, Outer>
where
    Inner: FitModel,
    Outer: FitModelXDeriv<Scalar = Inner::Scalar>,
    Inner::Scalar: Clone + Mul<Inner::Scalar, Output = Inner::Scalar>,
    <Inner::ParamCount as Conv>::TNum: Add<<Outer::ParamCount as Conv>::TNum>,
    Sum<<Inner::ParamCount as Conv>::TNum, <Outer::ParamCount as Conv>::TNum>: Conv<TNum = Sum<<Inner::ParamCount as Conv>::TNum, <Outer::ParamCount as Conv>::TNum>>
        + ArrayLength
        + Sub<<Inner::ParamCount as Conv>::TNum, Output = <Outer::ParamCount as Conv>::TNum>,
{
    type Scalar = Inner::Scalar;
    type ParamCount = Sum<<Inner::ParamCount as Conv>::TNum, <Outer::ParamCount as Conv>::TNum>;

    #[inline]
    fn evaluate(&self, x: &Self::Scalar) -> Self::Scalar {
        let y = self.inner.evaluate(x);
        self.outer.evaluate(&y)
    }

    #[inline]
    fn jacobian(
        &self,
        x: &Self::Scalar,
    ) -> impl Into<GenericArray<Self::Scalar, <Self::ParamCount as Conv>::TNum>> {
        // y = inner(x, p_in)
        // z = outer(y, p_out)
        //
        // z_p_in = z_y * inner_j(z, p_in)
        // z_p_out = outer_j(y, p_out)
        let y = self.inner.evaluate(x);
        let z_y = self.outer.deriv_x(&y);
        let z_p_in = self.inner.jacobian(x).into().map(|v| v * z_y.clone());
        let z_p_out = self.outer.jacobian(&y).into();

        GenericArray::concat(z_p_in, z_p_out)
    }

    #[inline]
    fn set_params(&mut self, new_params: GenericArray<Self::Scalar, Self::ParamCount>) {
        let (inner_params, outer_params) = GenericArray::split(new_params);
        self.inner.set_params(inner_params);
        self.outer.set_params(outer_params);
    }

    #[inline]
    fn get_params(&self) -> impl Into<GenericArray<Self::Scalar, Self::ParamCount>> {
        GenericArray::concat(
            self.inner.get_params().into(),
            self.outer.get_params().into(),
        )
    }
}

impl<Inner, Outer> FitModelXDeriv for Composition<Inner, Outer>
where
    Inner: FitModelXDeriv,
    Outer: FitModelXDeriv<Scalar = Inner::Scalar>,
    Inner::Scalar: Mul<Output = Inner::Scalar>,
    Self: FitModel<Scalar = Inner::Scalar>,
{
    #[inline]
    fn deriv_x(&self, x: &Self::Scalar) -> Self::Scalar {
        let y = self.inner.evaluate(x);
        self.inner.deriv_x(x) * self.outer.deriv_x(&y)
    }
}

/// Convenience trait to construct [`Composition`]. Alternatively, just construct it manually.
pub trait CompositionExt: FitModel + Sized {
    /// Applies second model on top of current one.
    fn compose<Outer: FitModelXDeriv<Scalar = Self::Scalar>>(
        self,
        outer: Outer,
    ) -> Composition<Self, Outer>;
}

impl<Inner: FitModel> CompositionExt for Inner {
    #[inline]
    fn compose<Outer: FitModelXDeriv<Scalar = Self::Scalar>>(
        self,
        outer: Outer,
    ) -> Composition<Self, Outer> {
        Composition { inner: self, outer }
    }
}

#[cfg(test)]
mod tests;
