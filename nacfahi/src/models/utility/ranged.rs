use core::ops::RangeBounds;

use generic_array::{sequence::GenericSequence, GenericArray};
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

impl<Scalar, Inner, Range> FitModel<Scalar> for Ranged<Inner, Range>
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
    ) -> impl Into<GenericArray<Scalar, <Self::ParamCount as generic_array_storage::Conv>::TNum>>
    {
        if self.range.contains(x) {
            self.inner.jacobian(x).into()
        } else {
            GenericArray::generate(|_| Scalar::zero())
        }
    }

    #[inline]
    fn set_params(
        &mut self,
        new_params: GenericArray<Scalar, <Self::ParamCount as generic_array_storage::Conv>::TNum>,
    ) {
        self.inner.set_params(new_params);
    }

    #[inline]
    fn get_params(
        &self,
    ) -> impl Into<GenericArray<Scalar, <Self::ParamCount as generic_array_storage::Conv>::TNum>>
    {
        self.inner.get_params()
    }
}

impl<Scalar, Inner, Range> FitModelXDeriv<Scalar> for Ranged<Inner, Range>
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

crate::test_model_derivative!(
    range_to,
    Ranged<Gaussian<f64>, core::ops::RangeTo<f64>>,
    Ranged {
        inner: Gaussian {
            x_c: 1.0,
            a: -3.0,
            s: 0.3,
        },
        range: ..2.0,
    },
    [
        (0.0, -1.0),
        (1.0, -4.0),
        (2.0, -5.0),
        (3.0, 6.0),
        (4.0, 2.0),
        (5.0, 2.5)
    ]
);

crate::test_model_derivative!(
    range_from,
    Ranged<Gaussian<f64>, core::ops::RangeFrom<f64>>,
    Ranged {
        inner: Gaussian {
            x_c: 1.0,
            a: -3.0,
            s: 0.3,
        },
        range: 0.0..,
    },
    [
        (0.0, -1.0),
        (1.0, -4.0),
        (2.0, -5.0),
        (3.0, 6.0),
        (4.0, 2.0),
        (5.0, 2.5)
    ]
);

crate::test_model_derivative!(
    range,
    Ranged<Gaussian<f64>, core::ops::Range<f64>>,
    Ranged {
        inner: Gaussian {
            x_c: 1.0,
            a: -3.0,
            s: 0.3,
        },
        range: -0.5..3.0,
    },
    [
        (0.0, -1.0),
        (1.0, -4.0),
        (2.0, -5.0),
        (3.0, 6.0),
        (4.0, 2.0),
        (5.0, 2.5)
    ]
);
