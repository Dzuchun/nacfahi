#[doc(hidden)]
mod asym;
#[doc(hidden)]
mod common;
#[doc(hidden)]
mod sym;

pub use common::*;
use num_traits::{Float, FloatConst};

use crate::{for_all_bool, models::FitModel};

#[doc(hidden)]
pub trait GaussianResolve<Scalar, const HAS_S_P: bool, const FIT_SIGMA: bool, const FIT_S_P: bool> {
    type T: FitModel<Scalar = Scalar>;
}

#[doc(hidden)]
#[allow(missing_debug_implementations)]
pub struct GaussianResolver(());

for_all_bool! {
    impl_sym,
    [FIT_SIGMA],
    impl<Scalar: Float + FloatConst> GaussianResolve<Scalar, false, FIT_SIGMA, false> for GaussianResolver {
        type T = sym::SymmetricGenericGaussian<Scalar, FIT_SIGMA>;
    }
}

for_all_bool! {
    impl_asym,
    [FIT_SIGMA, FIT_S_P],
    impl<Scalar: Float + FloatConst> GaussianResolve<Scalar, true, FIT_SIGMA, FIT_S_P> for GaussianResolver {
        type T = asym::AsymmetricGenericGaussian<Scalar, FIT_SIGMA, FIT_S_P>;
    }
}

pub type GenericGaussian<Scalar, const HAS_S_P: bool, const FIT_SIGMA: bool, const FIT_S_P: bool> =
    <GaussianResolver as GaussianResolve<Scalar, HAS_S_P, FIT_SIGMA, FIT_S_P>>::T;

pub type Gaussian<Scalar, const FIT_SIGMA: bool = true> =
    sym::SymmetricGenericGaussian<Scalar, FIT_SIGMA>;
