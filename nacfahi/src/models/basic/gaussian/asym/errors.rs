use generic_array::GenericArray;
use generic_array_storage::Conv;
use num_traits::{Float, FloatConst};

use crate::models::{
    FitModel,
    basic::gaussian::common::{ErrorsFull, ErrorsNone, ErrorsSigma, ErrorsSp},
};

use super::AsymmetricGenericGaussian;

#[doc(hidden)]
#[allow(missing_debug_implementations)]
pub struct GaussianErrResolver(());

#[doc(hidden)]
pub trait GaussianErrResolve<Scalar, const FIT_SIGMA: bool, const FIT_S_P: bool>
where
    AsymmetricGenericGaussian<Scalar, FIT_SIGMA, FIT_S_P>: FitModel,
{
    type T;

    fn create(
        errors: GenericArray<
            Scalar,
            <<AsymmetricGenericGaussian<Scalar, FIT_SIGMA, FIT_S_P> as FitModel>::ParamCount as Conv>::TNum,
        >,
    ) -> Self::T;
}

impl<Scalar: Float + FloatConst> GaussianErrResolve<Scalar, false, false> for GaussianErrResolver {
    type T = ErrorsNone<Scalar>;

    fn create(
        errors: generic_array::GenericArray<
                Scalar,
                <<AsymmetricGenericGaussian<Scalar, false, false> as FitModel>::ParamCount as generic_array_storage::Conv>::TNum,
            >,
    ) -> Self::T {
        let [a_err, x_c_err] = errors.into_array();
        ErrorsNone { a_err, x_c_err }
    }
}

impl<Scalar: Float + FloatConst> GaussianErrResolve<Scalar, true, false> for GaussianErrResolver {
    type T = ErrorsSigma<Scalar>;

    fn create(
        errors: generic_array::GenericArray<
                Scalar,
                <<AsymmetricGenericGaussian<Scalar, true, false> as FitModel>::ParamCount as generic_array_storage::Conv>::TNum,
            >,
    ) -> Self::T {
        let [a_err, s_err, x_c_err] = errors.into_array();
        ErrorsSigma {
            a_err,
            s_err,
            x_c_err,
        }
    }
}

impl<Scalar: Float + FloatConst> GaussianErrResolve<Scalar, false, true> for GaussianErrResolver {
    type T = ErrorsSp<Scalar>;

    fn create(
        errors: generic_array::GenericArray<
                Scalar,
                <<AsymmetricGenericGaussian<Scalar, false, true> as FitModel>::ParamCount as generic_array_storage::Conv>::TNum,
            >,
    ) -> Self::T {
        let [a_err, x_c_err, s_p_err] = errors.into_array();
        ErrorsSp {
            a_err,
            x_c_err,
            s_p_err,
        }
    }
}

impl<Scalar: Float + FloatConst> GaussianErrResolve<Scalar, true, true> for GaussianErrResolver {
    type T = ErrorsFull<Scalar>;

    fn create(
        errors: generic_array::GenericArray<
                Scalar,
                <<AsymmetricGenericGaussian<Scalar, true, true> as FitModel>::ParamCount as generic_array_storage::Conv>::TNum,
            >,
    ) -> Self::T {
        let [a_err, s_err, x_c_err, s_p_err] = errors.into_array();
        ErrorsFull {
            a_err,
            s_err,
            x_c_err,
            s_p_err,
        }
    }
}
