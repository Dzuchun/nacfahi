use core::ops::{Add, Mul};

use generic_array::{sequence::GenericSequence, GenericArray, IntoArrayLength};
use generic_array_storage::Conv;
use num_traits::{One, Zero};
use typenum::Const;

use crate::models::{FitModel, FitModelErrors, FitModelXDeriv};

/// Polynomial model, $\sum\limits_{i=0}^{order-1} a_{i} \cdot x^{i}$.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Polynomial<const ORDER: usize, Scalar> {
    /// Array of $a_{i}$
    pub params: [Scalar; ORDER],
}

impl<
        const ORDER: usize,
        Scalar: Clone + Zero + One + Add<Output = Scalar> + Mul<Output = Scalar>,
    > FitModel for Polynomial<ORDER, Scalar>
where
    Const<ORDER>: IntoArrayLength,
{
    type Scalar = Scalar;
    type ParamCount = Const<ORDER>;

    fn evaluate(&self, x: &Self::Scalar) -> Self::Scalar {
        let mut res = Scalar::zero();
        let mut pars = self.params.as_slice(); // TODO: make this a static cycle
        while let Some((last, rest)) = pars.split_last() {
            res = res * x.clone() + last.clone();
            pars = rest;
        }
        res
    }

    fn jacobian(
        &self,
        x: &Self::Scalar,
    ) -> impl Into<GenericArray<Self::Scalar, <Self::ParamCount as Conv>::TNum>> {
        let mut res = GenericArray::generate(|_| Scalar::zero());
        let mut pow = Scalar::one();
        for i in 0..ORDER {
            res[i] = pow.clone();
            pow = pow * x.clone();
        }
        res
    }

    fn set_params(
        &mut self,
        new_params: GenericArray<Self::Scalar, <Self::ParamCount as Conv>::TNum>,
    ) {
        self.params = new_params.into_array();
    }

    fn get_params(
        &self,
    ) -> impl Into<GenericArray<Self::Scalar, <Self::ParamCount as Conv>::TNum>> {
        self.params.clone()
    }
}

impl<
        const ORDER: usize,
        Scalar: Clone + Zero + One + Add<Output = Scalar> + Mul<Output = Scalar>,
    > FitModelXDeriv for Polynomial<ORDER, Scalar>
where
    Self: FitModel<Scalar = Scalar, ParamCount = Const<ORDER>>,
{
    fn deriv_x(&self, x: &Self::Scalar) -> Self::Scalar {
        let mut res = Scalar::zero();
        let mut pow = Scalar::one();
        let mut pow_i = Scalar::one();
        for _ in 1..ORDER {
            res = res + pow_i.clone() * pow.clone();
            pow = pow * x.clone();
            pow_i = pow_i + Scalar::one();
        }
        res
    }
}

impl<const ORDER: usize, Scalar: 'static> FitModelErrors for Polynomial<ORDER, Scalar>
where
    Const<ORDER>: IntoArrayLength,
    Self: FitModel<Scalar = Scalar, ParamCount = Const<ORDER>>,
{
    type OwnedModel = Polynomial<ORDER, Scalar>;

    fn with_errors(
        errors: GenericArray<Self::Scalar, <Self::ParamCount as generic_array_storage::Conv>::TNum>,
    ) -> Self::OwnedModel {
        Polynomial {
            params: errors.into_array(),
        }
    }
}

#[cfg(test)]
mod tests;
