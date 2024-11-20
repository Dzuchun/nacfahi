use core::ops::{Add, Mul};

use generic_array::{sequence::GenericSequence, ArrayLength, GenericArray};
use num_traits::{One, Zero};

use crate::models::FitModel;

/// Polynomial model, $\sum\limits_{i=0}^{order-1} a_{i} \cdot x^{i}$.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Polynomial<const ORDER: usize, Scalar> {
    /// Array of $a_{i}$
    pub params: [Scalar; ORDER],
}

impl<
        const ORDER: usize,
        Scalar: Clone + Zero + One + Add<Output = Scalar> + Mul<Output = Scalar>,
    > FitModel<Scalar> for Polynomial<ORDER, Scalar>
where
    typenum::Const<ORDER>: ArrayLength,
{
    type ParamCount = typenum::Const<ORDER>;

    fn evaluate(&self, x: &Scalar) -> Scalar {
        let mut res = Scalar::zero();
        let mut pars = self.params.as_slice(); // TODO: make this a static cycle
        while let Some((last, rest)) = pars.split_last() {
            res = res * x.clone() + last.clone();
            pars = rest;
        }
        res
    }

    fn jacobian(&self, x: &Scalar) -> impl Into<GenericArray<Scalar, Self::ParamCount>> {
        let mut res = GenericArray::generate(|_| Scalar::zero());
        let mut pow = Scalar::one();
        for i in 0..ORDER {
            res[i] = pow.clone();
            pow = pow * x.clone();
        }
        res
    }

    fn set_params(&mut self, new_params: GenericArray<Scalar, Self::ParamCount>) {
        self.params = new_params.into_array();
    }

    fn get_params(&self) -> impl Into<GenericArray<Scalar, Self::ParamCount>> {
        self.params.clone()
    }
}
