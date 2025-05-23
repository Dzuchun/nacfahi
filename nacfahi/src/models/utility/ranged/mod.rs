use core::ops::RangeBounds;

use generic_array::{GenericArray, sequence::GenericSequence};
use generic_array_storage::Conv;
use num_traits::Zero;

use crate::models::{FitModel, FitModelXDeriv};

/// Model filtering the `inner` model to a certain argument `range`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ranged<Inner, Range> {
    #[allow(missing_docs)]
    pub inner: Inner,
    #[allow(missing_docs)]
    pub range: Range,
}

impl<Inner, Range> FitModel for Ranged<Inner, Range>
where
    Inner: FitModel,
    Inner::Scalar: PartialOrd + Zero,
    Range: RangeBounds<Inner::Scalar>,
{
    type Scalar = Inner::Scalar;
    type ParamCount = Inner::ParamCount;

    #[inline]
    fn evaluate(&self, x: &Self::Scalar) -> Self::Scalar {
        if self.range.contains(x) {
            self.inner.evaluate(x)
        } else {
            Inner::Scalar::zero()
        }
    }

    #[inline]
    fn jacobian(
        &self,
        x: &Self::Scalar,
    ) -> impl Into<GenericArray<Self::Scalar, <Self::ParamCount as Conv>::TNum>> {
        if self.range.contains(x) {
            self.inner.jacobian(x).into()
        } else {
            GenericArray::generate(|_| Inner::Scalar::zero())
        }
    }

    #[inline]
    fn set_params(
        &mut self,
        new_params: GenericArray<Self::Scalar, <Self::ParamCount as Conv>::TNum>,
    ) {
        self.inner.set_params(new_params);
    }

    #[inline]
    fn get_params(
        &self,
    ) -> impl Into<GenericArray<Self::Scalar, <Self::ParamCount as Conv>::TNum>> {
        self.inner.get_params()
    }
}

impl<Inner, Range> FitModelXDeriv for Ranged<Inner, Range>
where
    Inner: FitModelXDeriv,
    Inner::Scalar: PartialOrd + Zero,
    Range: RangeBounds<Inner::Scalar>,
{
    #[inline]
    fn deriv_x(&self, x: &Self::Scalar) -> Self::Scalar {
        if self.range.contains(x) {
            self.inner.deriv_x(x)
        } else {
            Inner::Scalar::zero()
        }
    }
}

#[cfg(test)]
mod tests;
