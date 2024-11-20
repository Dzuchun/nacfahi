use core::ops::RangeBounds;

use generic_array::{sequence::GenericSequence, GenericArray};
use num_traits::Zero;

use crate::models::{FitModel, FitModelXDeriv};

/// Model filtering the `inner` model to a certain argument `range`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ranged<Range, Inner> {
    #[allow(missing_docs)]
    pub range: Range,
    #[allow(missing_docs)]
    pub inner: Inner,
}

impl<Scalar, Range, Inner> FitModel<Scalar> for Ranged<Range, Inner>
where
    Scalar: PartialOrd + Zero,
    Range: RangeBounds<Scalar>,
    Inner: FitModel<Scalar>,
{
    type ParamCount = Inner::ParamCount;

    #[inline]
    fn evaluate(&self, x: &Scalar) -> Scalar {
        if self.range.contains(x) {
            self.inner.evaluate(x)
        } else {
            Scalar::zero()
        }
    }

    #[inline]
    fn jacobian(
        &self,
        x: &Scalar,
    ) -> impl Into<generic_array::GenericArray<Scalar, Self::ParamCount>> {
        if self.range.contains(x) {
            self.inner.jacobian(x).into()
        } else {
            GenericArray::generate(|_| Scalar::zero())
        }
    }

    #[inline]
    fn set_params(&mut self, new_params: generic_array::GenericArray<Scalar, Self::ParamCount>) {
        self.inner.set_params(new_params);
    }

    #[inline]
    fn get_params(&self) -> impl Into<generic_array::GenericArray<Scalar, Self::ParamCount>> {
        self.inner.get_params()
    }
}

impl<Scalar, Range, Inner> FitModelXDeriv<Scalar> for Ranged<Range, Inner>
where
    Scalar: PartialOrd + Zero,
    Range: RangeBounds<Scalar>,
    Inner: FitModel<Scalar> + FitModelXDeriv<Scalar>,
{
    #[inline]
    fn deriv_x(&self, x: &Scalar) -> Scalar {
        if self.range.contains(x) {
            self.inner.deriv_x(x)
        } else {
            Scalar::zero()
        }
    }
}
