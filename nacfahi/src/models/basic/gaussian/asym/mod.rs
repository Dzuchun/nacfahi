use core::{
    marker::PhantomData,
    ops::{Add, Mul},
};

use errors::{GaussianErrResolve, GaussianErrResolver};
use generic_array::GenericArray;
use typenum::{U2, U3, U4};

use num_traits::{Float, FloatConst, One, Zero};

use crate::{
    for_all_bool,
    models::{FitModel, FitModelErrors, FitModelXDeriv},
};

use super::common::{ff64, gaussian, gaussian_deriv_a, gaussian_deriv_s, gaussian_deriv_x_c};

#[doc(hidden)]
mod errors;

#[doc(hidden)]
struct XCtx<'lt, Scalar> {
    a: Scalar,
    x_c: Scalar,
    sigma: Scalar,
    x: Scalar,
    one_sp: Scalar,
    sigma_deriv_sp: Scalar,
    _phantom: PhantomData<&'lt ()>,
}

impl<Scalar: Float + FloatConst> XCtx<'_, Scalar> {
    #[inline]
    fn eval(&self) -> Scalar {
        gaussian(self.x, self.x_c, self.sigma, self.a)
    }

    #[inline]
    fn deriv_a(&self) -> Scalar {
        gaussian_deriv_a(self.x, self.x_c, self.sigma, self.a)
    }

    #[inline]
    fn deriv_x_c(&self) -> Scalar {
        gaussian_deriv_x_c(self.x, self.x_c, self.sigma, self.a)
    }

    #[inline]
    fn deriv_x(&self) -> Scalar {
        -self.deriv_x_c()
    }

    #[inline]
    fn deriv_sigma(&self) -> Scalar {
        gaussian_deriv_s(self.x, self.x_c, self.sigma, self.a) * self.one_sp
    }

    #[inline]
    fn deriv_s_p(&self) -> Scalar {
        gaussian_deriv_s(self.x, self.x_c, self.sigma, self.a) * self.sigma_deriv_sp
    }
}

/// Asymmetric gaussian model $\dfrac{A}{\sqrt{2 \pi } \sigma(x)} \cdot \exp\left( \dfrac{ (x - x_{c})^2 }{ 2\sigma(x)^2 } \right)$,
///
/// where
/// - $\sigma(x) = \sigma \cdot (1 + s_{p} \cdot \theta(x - x_{c}))$,
/// - $\theta(x)$ - [step (Heaviside) function](https://en.wikipedia.org/wiki/Heaviside_step_function). This is the branching point, so please avoid asymmetric gaussian if possible
///
/// For $s_{p} = 0$ you get a regular symmetric Gaussian. If this is guaranteed to be the case for you, consider using `symmetric` variant
///
/// ### Generic constants
///
/// Generic constants define, if $\sigma$ and $s_{p}$ are fit for. Default is to fit $\sigma$, but not $s_{p}$.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AsymmetricGenericGaussian<Scalar, const FIT_SIGMA: bool, const FIT_S_P: bool> {
    /// Area of the gaussian, $A$
    pub a: Scalar,
    /// Center of the gaussian, $x_{c}$
    pub x_c: Scalar,
    /// Standard deviation of the gaussian, $\sigma$
    pub sigma: Scalar,
    /// Asymmetry coefficient, $s_{p}$
    pub s_p: Scalar,
}

impl<
        Scalar: Clone + One + Zero + Add<Output = Scalar> + Mul<Output = Scalar> + PartialOrd,
        const FIT_SIGMA: bool,
        const FIT_S_P: bool,
    > AsymmetricGenericGaussian<Scalar, FIT_SIGMA, FIT_S_P>
{
    #[inline]
    fn ctx<'lt>(&'lt self, x: &'lt Scalar) -> XCtx<'lt, Scalar> {
        if x >= &self.x_c {
            XCtx {
                a: self.a.clone(),
                x_c: self.x_c.clone(),
                sigma: self.sigma.clone(),
                x: x.clone(),
                one_sp: Scalar::one(),
                sigma_deriv_sp: Scalar::zero(),
                _phantom: PhantomData,
            }
        } else {
            let one_sp = Scalar::one() + self.s_p.clone();
            XCtx {
                a: self.a.clone(),
                x_c: self.x_c.clone(),
                sigma: self.sigma.clone() * one_sp.clone(),
                x: x.clone(),
                one_sp,
                sigma_deriv_sp: self.sigma.clone(),
                _phantom: PhantomData,
            }
        }
    }
}

impl<Scalar: Float + FloatConst, const FIT_SIGMA: bool, const FIT_S_P: bool>
    AsymmetricGenericGaussian<Scalar, FIT_SIGMA, FIT_S_P>
{
    /// [FWHM](https://en.wikipedia.org/wiki/Full_width_at_half_maximum), $2 \cdot \sqrt{ 2 \ln(2)} \sigma$
    #[inline]
    pub fn fwhm(&self) -> Scalar {
        ff64::<Scalar>(2.0 * f64::sqrt(2.0 * 2.0.ln())) * self.sigma
    }
}

impl<Scalar: Float + FloatConst> FitModel for AsymmetricGenericGaussian<Scalar, false, false> {
    type Scalar = Scalar;
    type ParamCount = U2;

    #[inline]
    fn evaluate(&self, x: &Self::Scalar) -> Self::Scalar {
        self.ctx(x).eval()
    }

    #[inline]
    fn jacobian(
        &self,
        x: &Self::Scalar,
    ) -> impl Into<GenericArray<Self::Scalar, Self::ParamCount>> {
        let ctx = self.ctx(x);
        [ctx.deriv_a(), ctx.deriv_x_c()]
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

impl<Scalar: Float + FloatConst> FitModel for AsymmetricGenericGaussian<Scalar, true, false> {
    type Scalar = Scalar;
    type ParamCount = U3;

    #[inline]
    fn evaluate(&self, x: &Self::Scalar) -> Self::Scalar {
        self.ctx(x).eval()
    }

    #[inline]
    fn jacobian(
        &self,
        x: &Self::Scalar,
    ) -> impl Into<GenericArray<Self::Scalar, Self::ParamCount>> {
        let ctx = self.ctx(x);
        [ctx.deriv_a(), ctx.deriv_x_c(), ctx.deriv_sigma()]
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

impl<Scalar: Float + FloatConst> FitModel for AsymmetricGenericGaussian<Scalar, true, true> {
    type Scalar = Scalar;
    type ParamCount = U4;

    #[inline]
    fn evaluate(&self, x: &Self::Scalar) -> Self::Scalar {
        self.ctx(x).eval()
    }

    #[inline]
    fn jacobian(
        &self,
        x: &Self::Scalar,
    ) -> impl Into<GenericArray<Self::Scalar, Self::ParamCount>> {
        let ctx = self.ctx(x);
        [
            ctx.deriv_a(),
            ctx.deriv_x_c(),
            ctx.deriv_sigma(),
            ctx.deriv_s_p(),
        ]
    }

    #[inline]
    fn set_params(&mut self, new_params: GenericArray<Self::Scalar, Self::ParamCount>) {
        let [new_a, new_x_c, new_sigma, new_s_p] = new_params.into_array();
        self.a = new_a;
        self.x_c = new_x_c;
        self.sigma = new_sigma;
        self.s_p = new_s_p;
    }

    #[inline]
    fn get_params(&self) -> impl Into<GenericArray<Self::Scalar, Self::ParamCount>> {
        [self.a, self.x_c, self.sigma, self.s_p]
    }
}

impl<Scalar: Float + FloatConst> FitModel for AsymmetricGenericGaussian<Scalar, false, true> {
    type Scalar = Scalar;
    type ParamCount = U3;

    #[inline]
    fn evaluate(&self, x: &Self::Scalar) -> Self::Scalar {
        self.ctx(x).eval()
    }

    #[inline]
    fn jacobian(
        &self,
        x: &Self::Scalar,
    ) -> impl Into<GenericArray<Self::Scalar, Self::ParamCount>> {
        let ctx = self.ctx(x);
        [ctx.deriv_a(), ctx.deriv_x_c(), ctx.deriv_s_p()]
    }

    #[inline]
    fn set_params(&mut self, new_params: GenericArray<Self::Scalar, Self::ParamCount>) {
        let [new_a, new_x_c, new_s_p] = new_params.into_array();
        self.a = new_a;
        self.x_c = new_x_c;
        self.s_p = new_s_p;
    }

    #[inline]
    fn get_params(&self) -> impl Into<GenericArray<Self::Scalar, Self::ParamCount>> {
        [self.a, self.x_c, self.s_p]
    }
}

impl<Scalar: Float + FloatConst, const FIT_SIGMA: bool, const FIT_S_P: bool> FitModelXDeriv
    for AsymmetricGenericGaussian<Scalar, FIT_SIGMA, FIT_S_P>
where
    Self: FitModel<Scalar = Scalar>,
{
    #[inline]
    fn deriv_x(&self, x: &Self::Scalar) -> Self::Scalar {
        self.ctx(x).deriv_x()
    }
}

pub type GaussianErr<Scalar, const FIT_SIGMA: bool = true, const FIT_S_P: bool = false> =
    <GaussianErrResolver as GaussianErrResolve<Scalar, FIT_SIGMA, FIT_S_P>>::T;

for_all_bool! {
    impl_errors,
    [FIT_SIGMA, FIT_S_P],
    impl<Scalar: Float + FloatConst + 'static> FitModelErrors for AsymmetricGenericGaussian<Scalar, FIT_SIGMA, FIT_S_P> {
        type OwnedModel = GaussianErr<Scalar, FIT_SIGMA, FIT_S_P>;

        #[inline]
        fn with_errors(errors: GenericArray<Self::Scalar, Self::ParamCount>) -> Self::OwnedModel {
            <GaussianErrResolver as GaussianErrResolve<Scalar, FIT_SIGMA, FIT_S_P>>::create(errors)
        }
    }
}

#[cfg(test)]
#[doc(hidden)]
mod tests;
