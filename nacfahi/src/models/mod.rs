use core::{
    iter::Sum,
    ops::{Div, Mul},
};

use generic_array::{
    ArrayLength, GenericArray,
    functional::FunctionalSequence,
    sequence::{Flatten, Unflatten},
};
use generic_array_storage::Conv;
use typenum::{Prod, ToUInt};

/// Basic building blocks for the models.
pub mod basic;

/// Utility models for composing more complex models.
pub mod utility;

#[doc = include_str!("../../doc/derive_sum.md")]
pub use nacfahi_derive::FitModelSum;

#[doc(hidden)]
type TNum<const N: usize> = <typenum::Const<N> as ToUInt>::Output;

/// Defines object that can fit to a set of data points.
///
/// Generally, you have no reason to implement this trait, as there are model primitives and derive macro for that. Manual implementation is always an option though - I've left some hints, in case you're unfamiliar with the types.
pub trait FitModel {
    /// Type of `x`, `y`, model parameters and all the derivatives. Different types are not supported (yet)
    type Scalar;

    /// Type representing number of parameters.
    ///
    /// **Hint**: [`typenum`]`::{U1, U2, ..}` types would most likely work for you.
    ///
    /// [`typenum`]: https://docs.rs/typenum/latest/typenum/
    type ParamCount: generic_array_storage::Conv;

    /// Computes model value for supplied `x` value and current parameters.
    fn evaluate(&self, x: &Self::Scalar) -> Self::Scalar;

    /// Computes jacobian (array of derivatives) for supplied `x` value and current parameters.
    ///
    /// **Hint**: return type allows you to return core Rust array, as long as it's size is correct.
    fn jacobian(
        &self,
        x: &Self::Scalar,
    ) -> impl Into<GenericArray<Self::Scalar, <Self::ParamCount as generic_array_storage::Conv>::TNum>>;

    /// Sets model parameters to ones contained in a generic array
    ///
    /// **Hint**: `GenericArray::into_array` is a thing. So if your model has two params, you can extract them as
    /// ```rust
    /// # use generic_array::GenericArray;
    /// let new_params: GenericArray<_, typenum::U2> = /* ... */
    /// # GenericArray::from_array([(), ()]);
    /// let [p1, p2] = new_params.into_array();
    /// ```
    fn set_params(
        &mut self,
        new_params: GenericArray<
            Self::Scalar,
            <Self::ParamCount as generic_array_storage::Conv>::TNum,
        >,
    );

    /// Returns current values of model params.
    ///
    /// **Hint**: return type allows you to return core Rust array, as long as it's size is correct.
    fn get_params(
        &self,
    ) -> impl Into<GenericArray<Self::Scalar, <Self::ParamCount as generic_array_storage::Conv>::TNum>>;
}

/// Defines model having derivative over the `x` variable.
///
/// This trait is meant to extend [`FitModel`] to allow usage of [`Composition`](utility::Composition) model, as it requires derivative over `x` for outer model.
pub trait FitModelXDeriv: FitModel {
    /// Returns derivative over `x` at supplied `x` value.
    fn deriv_x(&self, x: &Self::Scalar) -> Self::Scalar;
}

/// Defines models having a corresponding error-defining type.
///
/// This trait is meant to extend [`FitModel`] to allow usage of [`macro@crate::fit_stat!`].
pub trait FitModelErrors: FitModel {
    /// Type of the error model
    ///
    /// Most of the time, this can be just `Self`.
    type OwnedModel: 'static;

    /// Creates new model representing errors from the error array
    fn with_errors(
        errors: GenericArray<Self::Scalar, <Self::ParamCount as generic_array_storage::Conv>::TNum>,
    ) -> Self::OwnedModel;
}

impl<Model> FitModel for &'_ mut Model
where
    Model: FitModel,
{
    type Scalar = Model::Scalar;
    type ParamCount = Model::ParamCount;

    #[inline]
    fn evaluate(&self, x: &Self::Scalar) -> Self::Scalar {
        let s: &Model = self;
        Model::evaluate(s, x)
    }

    #[inline]
    fn jacobian(
        &self,
        x: &Self::Scalar,
    ) -> impl Into<GenericArray<Self::Scalar, <Self::ParamCount as generic_array_storage::Conv>::TNum>>
    {
        let s: &Model = self;
        Model::jacobian(s, x)
    }

    #[inline]
    fn set_params(
        &mut self,
        new_params: GenericArray<
            Self::Scalar,
            <Self::ParamCount as generic_array_storage::Conv>::TNum,
        >,
    ) {
        let s: &mut Model = self;
        Model::set_params(s, new_params);
    }

    #[inline]
    fn get_params(
        &self,
    ) -> impl Into<GenericArray<Self::Scalar, <Self::ParamCount as generic_array_storage::Conv>::TNum>>
    {
        let s: &Model = self;
        Model::get_params(s)
    }
}

impl<Model: FitModelXDeriv> FitModelXDeriv for &'_ mut Model {
    #[inline]
    fn deriv_x(&self, x: &Model::Scalar) -> Model::Scalar {
        <Model as FitModelXDeriv>::deriv_x(self, x)
    }
}

impl<Model: FitModelErrors> FitModelErrors for &'_ mut Model {
    type OwnedModel = Model::OwnedModel;

    #[inline]
    fn with_errors(
        errors: GenericArray<
            Model::Scalar,
            <Self::ParamCount as generic_array_storage::Conv>::TNum,
        >,
    ) -> Self::OwnedModel {
        <Model as FitModelErrors>::with_errors(errors)
    }
}

#[cfg(test)]
static_assertions::assert_impl_all!([basic::Gaussian<f64>; 1]: FitModel);
#[cfg(test)]
static_assertions::assert_impl_all!([basic::Exponent<f64>; 5]: FitModel);

impl<const N: usize, Model> FitModel for [Model; N]
where
    Model: FitModel,
    Model::Scalar: Sum,
    typenum::Const<N>: ToUInt,
    TNum<N>: ArrayLength,
    <Model::ParamCount as generic_array_storage::Conv>::TNum: Mul<TNum<N>>,
    Prod<<Model::ParamCount as generic_array_storage::Conv>::TNum, TNum<N>>:
        generic_array_storage::Conv<
                TNum = Prod<<Model::ParamCount as generic_array_storage::Conv>::TNum, TNum<N>>,
            > + ArrayLength
            + Div<<Model::ParamCount as generic_array_storage::Conv>::TNum, Output = TNum<N>>,
{
    type Scalar = Model::Scalar;
    type ParamCount = Prod<<Model::ParamCount as generic_array_storage::Conv>::TNum, TNum<N>>;

    #[inline]
    fn evaluate(&self, x: &Model::Scalar) -> Model::Scalar {
        self.iter()
            .map(move |e| e.evaluate(x))
            .sum::<Model::Scalar>()
    }

    #[inline]
    fn jacobian(
        &self,
        x: &Model::Scalar,
    ) -> impl Into<GenericArray<Model::Scalar, <Self::ParamCount as Conv>::TNum>> {
        let jacobian_generic_arr: GenericArray<
            GenericArray<Model::Scalar, <Model::ParamCount as Conv>::TNum>,
            TNum<N>,
        > = GenericArray::from_array(self.each_ref().map(|entity| entity.jacobian(x).into()));
        jacobian_generic_arr.flatten()
    }

    #[inline]
    fn set_params(
        &mut self,
        new_params: GenericArray<Model::Scalar, <Self::ParamCount as Conv>::TNum>,
    ) {
        let unflat: GenericArray<
            GenericArray<_, <<Model as FitModel>::ParamCount as Conv>::TNum>,
            <typenum::Const<N> as ToUInt>::Output,
        > = new_params.unflatten();
        let inners: &mut GenericArray<Model, TNum<N>> =
            GenericArray::from_mut_slice(self.as_mut_slice());
        inners.zip(unflat, Model::set_params);
    }

    #[inline]
    fn get_params(
        &self,
    ) -> impl Into<GenericArray<Model::Scalar, <Self::ParamCount as Conv>::TNum>> {
        let jacobian_generic_arr: GenericArray<
            GenericArray<Model::Scalar, <Model::ParamCount as Conv>::TNum>,
            TNum<N>,
        > = GenericArray::from_array(self.each_ref().map(|entity| entity.get_params().into()));
        jacobian_generic_arr.flatten()
    }
}

impl<const N: usize, Model> FitModelXDeriv for [Model; N]
where
    Self: FitModel,
    Self::Scalar: Sum,
    Model: FitModelXDeriv<Scalar = Self::Scalar>,
{
    #[inline]
    fn deriv_x(&self, x: &Self::Scalar) -> Self::Scalar {
        self.iter().map(|m| m.deriv_x(x)).sum()
    }
}

impl<const N: usize, Model> FitModelErrors for [Model; N]
where
    Self: FitModel,
    Self::Scalar: Sum,
    typenum::Const<N>: ToUInt,
    TNum<N>: ArrayLength,
    Model: FitModelErrors<Scalar = Self::Scalar>,
    <Self::ParamCount as Conv>::TNum: Div<<Model::ParamCount as Conv>::TNum, Output = TNum<N>>,
{
    type OwnedModel = [Model::OwnedModel; N];

    #[inline]
    fn with_errors(
        errors: GenericArray<Self::Scalar, <Self::ParamCount as Conv>::TNum>,
    ) -> Self::OwnedModel {
        let unflat: GenericArray<
            GenericArray<_, <<Model as FitModel>::ParamCount as Conv>::TNum>,
            <typenum::Const<N> as ToUInt>::Output,
        > = errors.unflatten();
        unflat.map(Model::with_errors).into_array()
    }
}

#[cfg(doc)]
#[macro_export]
#[doc = include_str!("../../doc/test_model_derivative.md")]
macro_rules! test_model_derivative {
    (
        $isolation_module:ident,
        $model_type:path,
        $model_construction:expr,
        [$(($x:expr, $y:expr)),*]
    ) => {
        /* ... */
    };
    (
        $model_type:path,
        $model_construction:expr,
        [$(($x:expr, $y:expr)),*]
    ) => {
        /* ... */
    };
}

#[cfg(not(doc))]
#[macro_export]
#[doc = include_str!("../../doc/test_model_derivative.md")]
macro_rules! test_model_derivative {
    ($isolation:ident, $name:path, $model:expr_2021, [$(($x:expr_2021, $y:expr_2021)),*]) => {
        #[cfg(test)]
        #[doc(hidden)]
        mod $isolation {
            #[allow(unused_imports)]
            use super::*;
            $crate::test_model_derivative!($name, $model, [$(($x, $y)), *]);
        }
    };
    ($name:path, $model:expr_2021, [$(($x:expr_2021, $y:expr_2021)),*]) => {
        #[cfg(test)]
        #[test]
        #[cfg_attr(miri, ignore = "Miri gets angry because of stacked borrows rules somewhere inside nalgebra's storage. The thing is, my own storage implementation IS USED NEARBY (generic_array_storage one), so I AM SCARED PWEASE DON'T HUWT MW")]
        #[allow(clippy::too_many_lines, unused_imports)]
        fn model_numeric_test() {
            use ::generic_array_storage::{Conv, GenericMatrixExt, GenericMatrixFromExt};
            use ::nalgebra::{Matrix, Vector};
            use $crate::{AsMatrixView, models::FitModel, models::basic::*, models::utility::*};

            type Opt<'l, F> = $crate::const_problem::ConstOptimizationProblem<
                'l,
                ::typenum::Const<M>,
                $name,
                F,
            >;

            const N: usize = <<<$name as FitModel>::ParamCount as Conv>::TNum as typenum::Unsigned>::USIZE;
            const M: usize = [$($x),*].len();

            struct T<'l, F>(Opt<'l, F>);

            impl<F: Fn(f64, f64) -> f64>
                ::levenberg_marquardt::LeastSquaresProblem<
                    f64,
                    ::nalgebra::Const<M>,
                    ::nalgebra::Const<N>,
                > for T<'_, F>
            where
                Vector<f64, ::nalgebra::Const<N>, ::nalgebra::base::ArrayStorage<f64, N, 1>>:
                    ::generic_array_storage::GenericMatrixFromExt<
                        ::typenum::Const<N>,
                        ::typenum::Const<1>,
                    >,
            {
                type ResidualStorage = ::nalgebra::base::ArrayStorage<f64, M, 1>;
                type JacobianStorage = ::nalgebra::base::ArrayStorage<f64, M, N>;
                type ParameterStorage = ::nalgebra::base::ArrayStorage<f64, N, 1>;

                fn set_params(
                    &mut self,
                    x: &Vector<f64, ::nalgebra::Const<N>, Self::ParameterStorage>,
                ) {
                    let x: Vector<f64, ::nalgebra::Const<N>, Self::ParameterStorage> = x.clone();
                    <Opt<'_, F> as ::levenberg_marquardt::LeastSquaresProblem<
                        f64,
                        ::nalgebra::Const<M>,
                        ::nalgebra::Const<N>,
                    >>::set_params(&mut self.0, &x.into_generic_matrix());
                }

                fn params(&self) -> Vector<f64, ::nalgebra::Const<N>, Self::ParameterStorage> {
                    <Opt<'_, F> as ::levenberg_marquardt::LeastSquaresProblem<
                        f64,
                        ::nalgebra::Const<M>,
                        ::nalgebra::Const<N>,
                    >>::params(&self.0)
                    .into_regular_matrix()
                }

                fn residuals(
                    &self,
                ) -> Option<Vector<f64, ::nalgebra::Const<M>, Self::ResidualStorage>> {
                    Some(
                        <Opt<'_, F> as ::levenberg_marquardt::LeastSquaresProblem<
                            f64,
                            ::nalgebra::Const<M>,
                            ::nalgebra::Const<N>,
                        >>::residuals(&self.0)?
                        .into_regular_matrix()
                    )
                }

                fn jacobian(
                    &self,
                ) -> Option<
                    Matrix<f64, ::nalgebra::Const<M>, ::nalgebra::Const<N>, Self::JacobianStorage>,
                > {
                    Some(
                        <Opt<'_, F> as ::levenberg_marquardt::LeastSquaresProblem<
                            f64,
                            ::nalgebra::Const<M>,
                            ::nalgebra::Const<N>,
                        >>::jacobian(&self.0)?
                        .into_regular_matrix()
                    )
                }
            }

            // arrange
            let model = $model;
            let x = [$($x),*].convert();
            let y = [$($y),*].convert();
            let mut combined = T($crate::const_problem::ConstOptimizationProblem {
                model,
                x,
                y,
                weights: |_, _| 1.0,
            });

            // act
            let analytic =
                <T<'_, _> as ::levenberg_marquardt::LeastSquaresProblem<_, _, _>>::jacobian(
                    &combined,
                )
                .expect("Should be able to compute jacobian analytically");
            let numerical = ::levenberg_marquardt::differentiate_numerically(&mut combined)
                .expect("Should be able to compute jacobian numerically");

            // assert
            if !::approx::ulps_eq!(analytic, numerical, epsilon = 1e-6) {
                panic!(
                    "Different jacobian values:\n\tanalytic={analytic}\n\tnumerical={numerical}"
                );
            }
        }
    };
}
