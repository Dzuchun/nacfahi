use core::ops::{Add, Mul};

use generic_array::{sequence::GenericSequence, ArrayLength, GenericArray, IntoArrayLength};
use num_traits::{One, Zero};

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
    > FitModel<Scalar> for Polynomial<ORDER, Scalar>
where
    typenum::Const<ORDER>: IntoArrayLength,
{
    type ParamCount = <typenum::Const<ORDER> as IntoArrayLength>::ArrayLength;

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

impl<
        const ORDER: usize,
        Scalar: Clone + Zero + One + Add<Output = Scalar> + Mul<Output = Scalar>,
    > FitModelXDeriv<Scalar> for Polynomial<ORDER, Scalar>
{
    fn deriv_x(&self, x: &Scalar) -> Scalar {
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

impl<const ORDER: usize, Scalar> FitModelErrors<Scalar> for Polynomial<ORDER, Scalar>
where
    typenum::Const<ORDER>: ArrayLength,
    Self: FitModel<Scalar, ParamCount = typenum::Const<ORDER>>,
{
    type OwnedModel = Polynomial<ORDER, Scalar>;

    fn with_errors(&self, errors: GenericArray<Scalar, Self::ParamCount>) -> Self::OwnedModel {
        Polynomial {
            params: errors.into_array(),
        }
    }
}

crate::test_model_derivative!(
    Polynomial::<5, f64>,
    Polynomial {
        params: [-4.0, -15.0, -2.0, 0.7, -0.01]
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
