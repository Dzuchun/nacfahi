use num_traits::{Float, FloatConst, NumCast};
use typenum::{U2, U3};

use generic_array::GenericArray;

use crate::models::{FitModel, FitModelErrors, FitModelXDeriv};

#[inline]
#[doc(hidden)]
fn exp<Scalar: Float>(x: Scalar) -> Scalar {
    x.exp()
}

#[inline]
#[doc(hidden)]
fn sqrt<Scalar: Float>(x: Scalar) -> Scalar {
    x.sqrt()
}

#[inline]
#[doc(hidden)]
fn sr<Scalar: Float>(x: Scalar) -> Scalar {
    x * x
}

#[inline]
#[doc(hidden)]
fn ff64<Scalar: NumCast>(val: f64) -> Scalar {
    <Scalar as NumCast>::from(val).unwrap()
}

#[inline]
#[doc(hidden)]
fn gaussian_sqrt<Scalar: Float + FloatConst>(s: Scalar) -> Scalar {
    Scalar::one() / (sqrt(Scalar::TAU()) * s)
}

#[inline]
#[doc(hidden)]
fn gaussian_exp<Scalar: Float + FloatConst>(x: Scalar, x_c: Scalar, s: Scalar) -> Scalar {
    exp(-sr(x - x_c) / (ff64::<Scalar>(2.0) * sr(s)))
}

#[inline]
#[doc(hidden)]
fn gaussian<Scalar: Float + FloatConst>(x: Scalar, x_c: Scalar, s: Scalar, a: Scalar) -> Scalar {
    a * gaussian_sqrt(s) * gaussian_exp(x, x_c, s)
}

#[inline]
#[doc(hidden)]
fn gaussian_deriv_s<Scalar: Float + FloatConst>(
    x: Scalar,
    x_c: Scalar,
    s: Scalar,
    a: Scalar,
) -> Scalar {
    a * gaussian_sqrt(s)
        * gaussian_exp(x, x_c, s)
        * (
            // derivative of sqrt
            - ff64::<Scalar>(1.0) / s
        +
            // derivative of exp
            sr(x - x_c) / (s * s * s)
        )
}

#[inline]
#[doc(hidden)]
fn gaussian_deriv_x_c<Scalar: Float + FloatConst>(
    x: Scalar,
    x_c: Scalar,
    s: Scalar,
    a: Scalar,
) -> Scalar {
    // derivative of exp
    a * gaussian_sqrt(s) * gaussian_exp(x, x_c, s) * (x - x_c) / sr(s)
}

#[inline]
#[doc(hidden)]
fn gaussian_deriv_a<Scalar: Float + FloatConst>(
    x: Scalar,
    x_c: Scalar,
    s: Scalar,
    _: Scalar,
) -> Scalar {
    // just drop s
    gaussian_sqrt(s) * gaussian_exp(x, x_c, s)
}

/// Gaussian model $\dfrac{A}{\sqrt{2 \pi } \sigma} \cdot \exp\left( \dfrac{ (x - x_{c})^2 }{ 2\sigma^2 } \right)$
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Gaussian<Scalar> {
    /// Area of the gaussian, $A$
    pub a: Scalar,
    /// Standard deviation of the gaussian, $\sigma$
    pub s: Scalar,
    /// Center of the gaussian, $x_{c}$
    pub x_c: Scalar,
}

impl<Scalar: Float + FloatConst> Gaussian<Scalar> {
    /// [FWHM](https://en.wikipedia.org/wiki/Full_width_at_half_maximum), $2 \cdot \sqrt{ 2 \ln(2)} \sigma$
    #[inline]
    pub fn w(&self) -> Scalar {
        ff64::<Scalar>(2.0 * f64::sqrt(2.0 * 2.0.ln())) * self.s
    }
}

impl<Scalar: Float + FloatConst> FitModel for Gaussian<Scalar> {
    type Scalar = Scalar;
    type ParamCount = U3;

    #[inline]
    fn evaluate(&self, x: &Self::Scalar) -> Self::Scalar {
        gaussian(*x, self.x_c, self.s, self.a)
    }

    #[inline]
    fn jacobian(
        &self,
        &x: &Self::Scalar,
    ) -> impl Into<GenericArray<Self::Scalar, Self::ParamCount>> {
        [
            gaussian_deriv_x_c(x, self.x_c, self.s, self.a),
            gaussian_deriv_s(x, self.x_c, self.s, self.a),
            gaussian_deriv_a(x, self.x_c, self.s, self.a),
        ]
    }

    #[inline]
    fn set_params(&mut self, new_params: GenericArray<Self::Scalar, Self::ParamCount>) {
        let [new_x_c, new_w, new_s] = new_params.into_array();
        self.x_c = new_x_c;
        self.s = new_w;
        self.a = new_s;
    }

    #[inline]
    fn get_params(&self) -> impl Into<GenericArray<Self::Scalar, Self::ParamCount>> {
        [self.x_c, self.s, self.a]
    }
}

impl<Scalar: Float + FloatConst> FitModelXDeriv for Gaussian<Scalar> {
    #[inline]
    fn deriv_x(&self, &x: &Self::Scalar) -> Self::Scalar {
        // derivatives over x and over x_c are the same, they are symmetrical
        gaussian_deriv_x_c(self.x_c, x, self.s, self.a)
    }
}

impl<Scalar> FitModelErrors for Gaussian<Scalar>
where
    Self: FitModel<Scalar = Scalar, ParamCount = U3>,
{
    type OwnedModel = Gaussian<Scalar>;

    fn with_errors(
        &self,
        errors: GenericArray<Self::Scalar, Self::ParamCount>,
    ) -> Self::OwnedModel {
        let [x_c, s, a] = errors.into_array();
        Gaussian { a, s, x_c }
    }
}

/// Same as [`Gaussian`], but with fixed $\sigma$ (i.e. it's not being fitted for, it will be left unchanged).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GaussianS<Scalar> {
    /// Area of the gaussian, $A$
    pub a: Scalar,
    /// Standard deviation of the gaussian, $\sigma$
    pub s: Scalar,
    /// Center of the gaussian, $x_{c}$
    pub x_c: Scalar,
}

impl<Scalar: Float + FloatConst> GaussianS<Scalar> {
    /// [FWHM](https://en.wikipedia.org/wiki/Full_width_at_half_maximum), $2 \cdot \sqrt{ 2 \ln(2)} \sigma$
    #[inline]
    pub fn w(&self) -> Scalar {
        ff64::<Scalar>(2.0 * f64::sqrt(2.0 * 2.0.ln())) * self.s
    }
}

impl<Scalar: Float + FloatConst> FitModel for GaussianS<Scalar> {
    type Scalar = Scalar;
    type ParamCount = U2;

    #[inline]
    fn evaluate(&self, x: &Self::Scalar) -> Self::Scalar {
        gaussian(*x, self.x_c, self.s, self.a)
    }

    #[inline]
    fn jacobian(
        &self,
        &x: &Self::Scalar,
    ) -> impl Into<GenericArray<Self::Scalar, Self::ParamCount>> {
        [
            gaussian_deriv_x_c(x, self.x_c, self.s, self.a),
            gaussian_deriv_a(x, self.x_c, self.s, self.a),
        ]
    }

    #[inline]
    fn set_params(&mut self, new_params: GenericArray<Self::Scalar, Self::ParamCount>) {
        let [new_x_c, new_a] = new_params.into_array();
        self.x_c = new_x_c;
        self.a = new_a;
    }

    #[inline]
    fn get_params(&self) -> impl Into<GenericArray<Self::Scalar, Self::ParamCount>> {
        [self.x_c, self.a]
    }
}

impl<Scalar: Float + FloatConst> FitModelXDeriv for GaussianS<Scalar> {
    #[inline]
    fn deriv_x(&self, x: &Self::Scalar) -> Self::Scalar {
        // derivatives over x and over x_c are the same, they are symmetrical
        gaussian_deriv_x_c(*x, self.x_c, self.s, self.a)
    }
}

/// Represents errors of [`GaussianS`]'s parameters.
///
/// It's a separate type, since `s` parameter of [`GaussianS`] is not fitted for, and no error can be computed for it.
#[derive(Debug)]
pub struct GaussianSErr<Scalar> {
    /// Error for an area of the gaussian, $\sigma_{A}$
    pub a_err: Scalar,
    /// Error for a center of the gaussian, $\sigma_{x_{c}}$
    pub x_c_err: Scalar,
}

impl<Scalar> FitModelErrors for GaussianS<Scalar>
where
    Self: FitModel<Scalar = Scalar, ParamCount = U2>,
{
    type OwnedModel = GaussianSErr<Scalar>;

    #[inline]
    fn with_errors(
        &self,
        errors: GenericArray<Self::Scalar, Self::ParamCount>,
    ) -> Self::OwnedModel {
        let [a_err, x_c_err] = errors.into_array();
        GaussianSErr { a_err, x_c_err }
    }
}

crate::test_model_derivative!(
    gaussian_test,
    Gaussian::<f64>,
    Gaussian {
        a: -5.0,
        x_c: 1.0,
        s: 3.0
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
    gaussians_test,
    GaussianS::<f64>,
    GaussianS {
        a: -5.0,
        x_c: 1.0,
        s: 3.0
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
