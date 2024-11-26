use num_traits::{Float, FloatConst, NumCast};

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
pub(super) fn ff64<Scalar: NumCast>(val: f64) -> Scalar {
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
pub(super) fn gaussian<Scalar: Float + FloatConst>(
    x: Scalar,
    x_c: Scalar,
    s: Scalar,
    a: Scalar,
) -> Scalar {
    a * gaussian_sqrt(s) * gaussian_exp(x, x_c, s)
}

#[inline]
#[doc(hidden)]
pub(super) fn gaussian_deriv_s<Scalar: Float + FloatConst>(
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
pub(super) fn gaussian_deriv_x_c<Scalar: Float + FloatConst>(
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
pub(super) fn gaussian_deriv_a<Scalar: Float + FloatConst>(
    x: Scalar,
    x_c: Scalar,
    s: Scalar,
    _a: Scalar,
) -> Scalar {
    // just drop a
    gaussian_sqrt(s) * gaussian_exp(x, x_c, s)
}

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ErrorsNone<Scalar> {
    pub a_err: Scalar,
    pub x_c_err: Scalar,
}

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ErrorsSigma<Scalar> {
    pub a_err: Scalar,
    pub s_err: Scalar,
    pub x_c_err: Scalar,
}

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ErrorsSp<Scalar> {
    pub a_err: Scalar,
    pub x_c_err: Scalar,
    pub s_p_err: Scalar,
}

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ErrorsFull<Scalar> {
    pub a_err: Scalar,
    pub s_err: Scalar,
    pub x_c_err: Scalar,
    pub s_p_err: Scalar,
}

#[macro_export]
#[doc(hidden)]
macro_rules! for_all_bool {
    ($isolation:ident, [$($vars:ident),*], $thing:item) => {
        #[doc(hidden)]
        mod $isolation {
            #[doc(hidden)]
            mod env {
                pub use super::super::*;
            }

            #[doc(hidden)]
            mod thing {
                $crate::for_all_bool!{@ [$($vars),*], $thing}
            }
        }
    };
    (@ [], $thing:item) => {
        use super::env::*;
        $thing
    };
    (@ [$var:ident $(, $vars:ident)*], $thing:item) => {
        #[doc(hidden)]
        mod for_false {
            #[doc(hidden)]
            mod env {
                pub use super::super::super::env::*;
                pub const $var: bool = false;
            }

            #[doc(hidden)]
            pub mod thing {
                $crate::for_all_bool!{@ [$($vars),*], $thing}
            }
        }
        #[allow(unused)]
        pub use for_false::thing::*;

        #[doc(hidden)]
        mod for_true {
            #[doc(hidden)]
            mod env {
                pub use super::super::super::env::*;
                pub const $var: bool = true;
            }

            #[doc(hidden)]
            pub mod thing {
                $crate::for_all_bool!{@ [$($vars),*], $thing}
            }
        }
        #[allow(unused)]
        pub use for_true::thing::*;
    };
}
