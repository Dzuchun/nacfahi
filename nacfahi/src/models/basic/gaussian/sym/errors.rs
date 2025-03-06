use generic_array::GenericArray;
use generic_array_storage::Conv;
use num_traits::{Float, FloatConst};

use crate::models::{
    FitModel,
    basic::gaussian::common::{ErrorsNone, ErrorsSigma},
};

use super::SymmetricGenericGaussian;

#[doc(hidden)]
#[allow(missing_debug_implementations)]
pub struct GaussianErrResolver(());

#[doc(hidden)]
pub trait GaussianErrResolve<Scalar, const FIT_SIGMA: bool>
where
    SymmetricGenericGaussian<Scalar, FIT_SIGMA>: FitModel,
{
    type T;

    fn create(
        errors: GenericArray<
            Scalar,
            <<SymmetricGenericGaussian<Scalar, FIT_SIGMA> as FitModel>::ParamCount as Conv>::TNum,
        >,
    ) -> Self::T;
}

impl<Scalar: Float + FloatConst> GaussianErrResolve<Scalar, false> for GaussianErrResolver {
    type T = ErrorsNone<Scalar>;

    fn create(
        errors: generic_array::GenericArray<
                Scalar,
                <<SymmetricGenericGaussian<Scalar, false> as FitModel>::ParamCount as generic_array_storage::Conv>::TNum,
            >,
    ) -> Self::T {
        let [a_err, x_c_err] = errors.into_array();
        ErrorsNone { a_err, x_c_err }
    }
}

impl<Scalar: Float + FloatConst> GaussianErrResolve<Scalar, true> for GaussianErrResolver {
    type T = ErrorsSigma<Scalar>;

    fn create(
        errors: generic_array::GenericArray<
                Scalar,
                <<SymmetricGenericGaussian<Scalar, true> as FitModel>::ParamCount as generic_array_storage::Conv>::TNum,
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
