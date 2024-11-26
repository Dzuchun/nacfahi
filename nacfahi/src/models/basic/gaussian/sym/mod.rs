use errors::{GaussianErrResolve, GaussianErrResolver};
use generic_array::GenericArray;
use typenum::{U2, U3};

use num_traits::{Float, FloatConst};

use crate::{
    for_all_bool,
    models::{FitModel, FitModelErrors, FitModelXDeriv},
};

use super::common::{ff64, gaussian, gaussian_deriv_a, gaussian_deriv_s, gaussian_deriv_x_c};

#[doc(hidden)]
mod errors;

/// Symmetric gaussian model $\dfrac{A}{\sqrt{2 \pi } \sigma} \cdot \exp\left( \dfrac{ (x - x_{c})^2 }{ 2\sigma^2 } \right)$,
///
/// ### Generic constants
///
/// Generic constant defines, if $\sigma$ is fit for. Default is to fit $\sigma$, but not $s_{p}$.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SymmetricGenericGaussian<Scalar, const FIT_SIGMA: bool> {
    /// Area of the gaussian, $A$
    pub a: Scalar,
    /// Center of the gaussian, $x_{c}$
    pub x_c: Scalar,
    /// Standard deviation of the gaussian, $\sigma$
    pub sigma: Scalar,
}

impl<Scalar: Float + FloatConst, const FIT_SIGMA: bool>
    SymmetricGenericGaussian<Scalar, FIT_SIGMA>
{
    /// [FWHM](https://en.wikipedia.org/wiki/Full_width_at_half_maximum), $2 \cdot \sqrt{ 2 \ln(2)} \sigma$
    #[inline]
    pub fn fwhm(&self) -> Scalar {
        ff64::<Scalar>(2.0 * f64::sqrt(2.0 * 2.0.ln())) * self.sigma
    }
}

impl<Scalar: Float + FloatConst> FitModel for SymmetricGenericGaussian<Scalar, false> {
    type Scalar = Scalar;
    type ParamCount = U2;

    #[inline]
    fn evaluate(&self, &x: &Self::Scalar) -> Self::Scalar {
        gaussian(x, self.x_c, self.sigma, self.a)
    }

    #[inline]
    fn jacobian(
        &self,
        &x: &Self::Scalar,
    ) -> impl Into<GenericArray<Self::Scalar, Self::ParamCount>> {
        [
            gaussian_deriv_a(x, self.x_c, self.sigma, self.a),
            gaussian_deriv_x_c(x, self.x_c, self.sigma, self.a),
        ]
    }

    #[inline]
    fn set_params(&mut self, new_params: GenericArray<Self::Scalar, Self::ParamCount>) {
        let [new_a, new_x_c] = new_params.into_array();
        self.a = new_a;
        self.x_c = new_x_c;
    }

    #[inline]
    fn get_params(&self) -> impl Into<GenericArray<Self::Scalar, Self::ParamCount>> {
        [self.a, self.x_c]
    }
}

impl<Scalar: Float + FloatConst> FitModel for SymmetricGenericGaussian<Scalar, true> {
    type Scalar = Scalar;
    type ParamCount = U3;

    #[inline]
    fn evaluate(&self, &x: &Self::Scalar) -> Self::Scalar {
        gaussian(x, self.x_c, self.sigma, self.a)
    }

    #[inline]
    fn jacobian(
        &self,
        &x: &Self::Scalar,
    ) -> impl Into<GenericArray<Self::Scalar, Self::ParamCount>> {
        [
            gaussian_deriv_a(x, self.x_c, self.sigma, self.a),
            gaussian_deriv_x_c(x, self.x_c, self.sigma, self.a),
            gaussian_deriv_s(x, self.x_c, self.sigma, self.a),
        ]
    }

    #[inline]
    fn set_params(&mut self, new_params: GenericArray<Self::Scalar, Self::ParamCount>) {
        let [new_a, new_x_c, new_sigma] = new_params.into_array();
        self.a = new_a;
        self.x_c = new_x_c;
        self.sigma = new_sigma;
    }

    #[inline]
    fn get_params(&self) -> impl Into<GenericArray<Self::Scalar, Self::ParamCount>> {
        [self.a, self.x_c, self.sigma]
    }
}

impl<Scalar: Float + FloatConst, const FIT_SIGMA: bool> FitModelXDeriv
    for SymmetricGenericGaussian<Scalar, FIT_SIGMA>
where
    Self: FitModel<Scalar = Scalar>,
{
    #[inline]
    fn deriv_x(&self, &x: &Self::Scalar) -> Self::Scalar {
        -gaussian_deriv_x_c(x, self.x_c, self.sigma, self.a)
    }
}

pub type GaussianErr<Scalar, const FIT_SIGMA: bool = true> =
    <GaussianErrResolver as GaussianErrResolve<Scalar, FIT_SIGMA>>::T;

for_all_bool! {
    impl_errors,
    [FIT_SIGMA],
    impl<Scalar: Float + FloatConst + 'static> FitModelErrors for SymmetricGenericGaussian<Scalar, FIT_SIGMA> {
        type OwnedModel = GaussianErr<Scalar, FIT_SIGMA>;

        #[inline]
        fn with_errors(errors: GenericArray<Self::Scalar, Self::ParamCount>) -> Self::OwnedModel {
            <GaussianErrResolver as GaussianErrResolve<Scalar, FIT_SIGMA>>::create(errors)
        }
    }
}

#[cfg(test)]
#[doc(hidden)]
mod tests;
