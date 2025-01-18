#![doc = include_str!("../README.md")]
#![no_std] // <-- see that attr? No shit!

use core::borrow::{Borrow, BorrowMut};
use core::ops::Sub;

#[cfg(feature = "alloc")]
use dyn_problem::DynOptimizationProblem;
use generic_array::{ArrayLength, GenericArray};
use models::{FitModel, FitModelErrors};

use const_problem::ConstOptimizationProblem;
use generic_array_storage::{Conv, GenericArrayStorage, GenericMatrix, GenericMatrixFromExt};
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::allocator::{Allocator, Reallocator};
#[cfg(feature = "alloc")]
use nalgebra::Dyn;
use nalgebra::{
    ComplexField, DefaultAllocator, Dim, DimMax, DimMaximum, DimMin, DimName, MatrixView, RealField,
};

use nalgebra::{Matrix, OMatrix};
use typenum::Unsigned;

use num_traits::{Float, NumCast};

/// Re-export. See [`LevenbergMarquardt`](https://docs.rs/levenberg-marquardt/latest/levenberg_marquardt/struct.LevenbergMarquardt.html)
///
pub use levenberg_marquardt::LevenbergMarquardt;
/// Re-export. See [`MinimizationReport`](https://docs.rs/levenberg-marquardt/latest/levenberg_marquardt/struct.MinimizationReport.html)
///
pub use levenberg_marquardt::MinimizationReport;
/// Re-export. See [`TerminationReason`](https://docs.rs/levenberg-marquardt/latest/levenberg_marquardt/enum.TerminationReason.html)
///
pub use levenberg_marquardt::TerminationReason;

/// Fitting models
pub mod models;

#[doc(hidden)]
mod const_problem;

#[cfg(feature = "alloc")]
#[doc(hidden)]
mod dyn_problem;

#[doc = include_str!("../doc/as_matrix_view.md")]
pub trait AsMatrixView {
    /// Type of the elements
    type Scalar;
    /// Type representing data points count.
    type Points: CreateProblem;

    /// Creates data view from type.
    fn convert(
        &self,
    ) -> MatrixView<'_, Self::Scalar, <Self::Points as CreateProblem>::Nalg, nalgebra::U1>;
}

impl<T: AsMatrixView + ?Sized> AsMatrixView for &T {
    type Scalar = T::Scalar;
    type Points = T::Points;

    fn convert(
        &self,
    ) -> MatrixView<'_, Self::Scalar, <Self::Points as CreateProblem>::Nalg, nalgebra::U1> {
        <T as AsMatrixView>::convert(self)
    }
}

impl<T: AsMatrixView + ?Sized> AsMatrixView for &mut T {
    type Scalar = T::Scalar;
    type Points = T::Points;

    fn convert(
        &self,
    ) -> MatrixView<'_, Self::Scalar, <Self::Points as CreateProblem>::Nalg, nalgebra::U1> {
        <T as AsMatrixView>::convert(self)
    }
}

impl<Scalar: nalgebra::Scalar, const N: usize> AsMatrixView for [Scalar; N]
where
    nalgebra::Const<N>: CreateProblem<Nalg = nalgebra::Const<N>>,
{
    type Scalar = Scalar;
    type Points = nalgebra::Const<N>;

    fn convert(
        &self,
    ) -> MatrixView<'_, Self::Scalar, <Self::Points as CreateProblem>::Nalg, nalgebra::U1> {
        MatrixView::<'_, Self::Scalar, <Self::Points as CreateProblem>::Nalg, nalgebra::U1>::from_slice(self)
    }
}

#[cfg(feature = "alloc")]
impl<Scalar: nalgebra::Scalar> AsMatrixView for [Scalar] {
    type Scalar = Scalar;
    type Points = Dyn;

    fn convert(&self) -> MatrixView<'_, Self::Scalar, Self::Points, nalgebra::U1> {
        MatrixView::<'_, Self::Scalar, Self::Points, nalgebra::U1>::from_slice(self, self.len())
    }
}

impl<Scalar, Points: Dim, S> AsMatrixView for Matrix<Scalar, Points, nalgebra::U1, S>
where
    Points: CreateProblem<Nalg = Points>,
    S: nalgebra::storage::RawStorage<Scalar, Points, RStride = nalgebra::U1, CStride = Points>,
{
    type Scalar = Scalar;
    type Points = Points;

    fn convert(&self) -> MatrixView<'_, Self::Scalar, Self::Points, nalgebra::U1> {
        self.as_view()
    }
}

impl<Scalar: nalgebra::Scalar, Points: ArrayLength + Conv<TNum = Points>> AsMatrixView
    for GenericArray<Scalar, Points>
where
    <Points as Conv>::Nalg: CreateProblem,
    <<Points as Conv>::Nalg as CreateProblem>::Nalg: DimName,
{
    type Scalar = Scalar;
    type Points = <Points as Conv>::Nalg;

    fn convert(
        &self,
    ) -> MatrixView<'_, Self::Scalar, <Self::Points as CreateProblem>::Nalg, nalgebra::U1> {
        MatrixView::<'_, Self::Scalar, <Self::Points as CreateProblem>::Nalg, nalgebra::U1>::from_slice(self)
    }
}

/// Indicates *how exactly* matrix data views should be converted into [`LeastSquaresProblem`](https://docs.rs/levenberg-marquardt/latest/levenberg_marquardt/trait.LeastSquaresProblem.html) suitable for [`levenberg_marquardt`](https://docs.rs/levenberg-marquardt/latest/levenberg_marquardt/index.html) operation.
///
/// In case it seems pointless, there are actually currently two fundamentally different problem types corresponding to const (stack-based) and dyn (alloc-based) problems. They should be hidden from docs, but this trait's implementation points should get you started (in case you're curious, or something).
pub trait CreateProblem {
    /// [`nalgebra`]-facing type, for view size definition
    type Nalg: Dim;

    /// Creates a problem from data views and arbitrary model.
    fn create<'d, Model: FitModel + 'd>(
        x: MatrixView<'d, Model::Scalar, Self::Nalg, nalgebra::U1>,
        y: MatrixView<'d, Model::Scalar, Self::Nalg, nalgebra::U1>,
        model: Model,
        weights: impl Fn(Model::Scalar, Model::Scalar) -> Model::Scalar + 'd,
    ) -> impl LeastSquaresProblem<Model::Scalar, Self::Nalg, <Model::ParamCount as Conv>::Nalg> + 'd
    where
        Model::Scalar: ComplexField + Copy,
        DefaultAllocator: Allocator<<Model::ParamCount as Conv>::Nalg, nalgebra::U1>,
        DefaultAllocator: Allocator<Self::Nalg, nalgebra::U1>,
        DefaultAllocator: Allocator<Self::Nalg, <Model::ParamCount as Conv>::Nalg>;
}

impl<const POINTS: usize> CreateProblem for typenum::Const<POINTS>
where
    Self: Conv<Nalg = nalgebra::Const<POINTS>>,
{
    type Nalg = nalgebra::Const<POINTS>;

    fn create<'d, Model: FitModel + 'd>(
        x: MatrixView<'d, Model::Scalar, Self::Nalg, nalgebra::U1>,
        y: MatrixView<'d, Model::Scalar, Self::Nalg, nalgebra::U1>,
        model: Model,
        weights: impl Fn(Model::Scalar, Model::Scalar) -> Model::Scalar + 'd,
    ) -> impl LeastSquaresProblem<Model::Scalar, Self::Nalg, <Model::ParamCount as Conv>::Nalg> + 'd
    where
        Model::Scalar: ComplexField + Copy,
        DefaultAllocator: Allocator<<Model::ParamCount as Conv>::Nalg, nalgebra::U1>,
        DefaultAllocator: Allocator<Self::Nalg, nalgebra::U1>,
        DefaultAllocator: Allocator<Self::Nalg, <Model::ParamCount as Conv>::Nalg>,
    {
        ConstOptimizationProblem::<'d, Self, Model, _> {
            model,
            x,
            y,
            weights,
        }
    }
}

impl<const POINTS: usize> CreateProblem for nalgebra::Const<POINTS>
where
    Self: Conv<Nalg = Self>,
{
    type Nalg = Self;

    fn create<'d, Model: FitModel + 'd>(
        x: MatrixView<'d, Model::Scalar, Self::Nalg, nalgebra::U1>,
        y: MatrixView<'d, Model::Scalar, Self::Nalg, nalgebra::U1>,
        model: Model,
        weights: impl Fn(Model::Scalar, Model::Scalar) -> Model::Scalar + 'd,
    ) -> impl LeastSquaresProblem<Model::Scalar, Self::Nalg, <Model::ParamCount as Conv>::Nalg> + 'd
    where
        Model::Scalar: ComplexField + Copy,
        DefaultAllocator: Allocator<<Model::ParamCount as Conv>::Nalg, nalgebra::U1>,
        DefaultAllocator: Allocator<Self::Nalg, nalgebra::U1>,
        DefaultAllocator: Allocator<Self::Nalg, <Model::ParamCount as Conv>::Nalg>,
    {
        ConstOptimizationProblem::<'d, Self, Model, _> {
            model,
            x,
            y,
            weights,
        }
    }
}

#[cfg(feature = "alloc")]
impl CreateProblem for Dyn {
    type Nalg = Self;

    fn create<'d, Model: FitModel + 'd>(
        x: MatrixView<'d, Model::Scalar, Self::Nalg, nalgebra::U1>,
        y: MatrixView<'d, Model::Scalar, Self::Nalg, nalgebra::U1>,
        model: Model,
        weights: impl Fn(Model::Scalar, Model::Scalar) -> Model::Scalar + 'd,
    ) -> impl LeastSquaresProblem<Model::Scalar, Self::Nalg, <Model::ParamCount as Conv>::Nalg> + 'd
    where
        Model::Scalar: ComplexField + Copy,
        DefaultAllocator: Allocator<<Model::ParamCount as Conv>::Nalg, nalgebra::U1>,
        DefaultAllocator: Allocator<Self::Nalg, nalgebra::U1>,
        DefaultAllocator: Allocator<Self::Nalg, <Model::ParamCount as Conv>::Nalg>,
    {
        DynOptimizationProblem::<'d, Model, _> {
            model,
            x,
            y,
            weights,
        }
    }
}

/// A helper unit type that is never constructed, and only used in type bounds.
#[doc(hidden)]
#[allow(missing_debug_implementations)]
pub struct FitterUnit(());

type DataPoints<Data> = <<Data as AsMatrixView>::Points as CreateProblem>::Nalg;

/// A helper trait to simplify type bounds for a user. You probably should no see this.
///
/// In case you do get a "type does not implement" type or error with this trait... I'm sorry.
pub trait FitBound<Model: FitModel, X, Y = X>
where
    Model::Scalar: RealField,
{
    #[doc(hidden)]
    type Points: Dim;
    #[doc(hidden)]
    fn fit(
        minimizer: impl Borrow<LevenbergMarquardt<Model::Scalar>>,
        model: &mut Model,
        x: impl Borrow<X>,
        y: impl Borrow<Y>,
        weights: impl Fn(Model::Scalar, Model::Scalar) -> Model::Scalar,
    ) -> MinimizationReport<Model::Scalar>;
}

impl<Model, X, Y> FitBound<Model, X, Y> for FitterUnit
where
    Model: FitModel,
    Model::Scalar: RealField + Float,
    Model::ParamCount: Conv,
    <Model::ParamCount as Conv>::TNum: Sub<typenum::U1>,

    DefaultAllocator: Allocator<<Model::ParamCount as Conv>::Nalg>,
    DefaultAllocator: Allocator<DataPoints<X>>,
    DefaultAllocator: Allocator<DataPoints<X>, <Model::ParamCount as Conv>::Nalg>,

    X: AsMatrixView<Scalar = Model::Scalar>,
    X::Points: CreateProblem,

    Y: AsMatrixView<Scalar = Model::Scalar, Points = X::Points>,

    DataPoints<X>:
        DimMax<<Model::ParamCount as Conv>::Nalg> + DimMin<<Model::ParamCount as Conv>::Nalg>,
    <Model::ParamCount as Conv>::Nalg:
        DimMax<<X::Points as CreateProblem>::Nalg> + DimMin<<X::Points as CreateProblem>::Nalg>,
    DefaultAllocator: Reallocator<
        Model::Scalar,
        DataPoints<X>,
        <Model::ParamCount as Conv>::Nalg,
        DimMaximum<DataPoints<X>, <Model::ParamCount as Conv>::Nalg>,
        <Model::ParamCount as Conv>::Nalg,
    >,
{
    type Points = <<X as AsMatrixView>::Points as CreateProblem>::Nalg;

    #[allow(
        clippy::inline_always,
        reason = "This function is used in a single place, and, in fact, wound not exist unless I wanted to extract the type bounds to a separate trait."
    )]
    #[inline(always)]
    fn fit(
        minimizer: impl Borrow<LevenbergMarquardt<Model::Scalar>>,
        model: &mut Model,
        x: impl Borrow<X>,
        y: impl Borrow<Y>,
        weights: impl Fn(Model::Scalar, Model::Scalar) -> Model::Scalar,
    ) -> MinimizationReport<Model::Scalar> {
        let x = x.borrow().convert();
        let y = y.borrow().convert();

        let problem = X::Points::create::<'_, &mut Model>(x, y, model.borrow_mut(), weights);
        let (_, report) = LevenbergMarquardt::minimize::<
            <Model::ParamCount as Conv>::Nalg,
            DataPoints<X>,
            _,
        >(minimizer.borrow(), problem);
        report
    }
}

/// A helper trait to simplify type bounds for a user. You probably should no see this.
///
/// In case you do get a "type does not implement" type or error with this trait... I'm sorry.
pub trait FitErrBound<Model: FitModelErrors, X, Y = X>: FitBound<Model, X, Y>
where
    Model::Scalar: RealField,
{
    #[doc(hidden)]
    fn produce_stat(
        model: impl Borrow<Model>,
        report: MinimizationReport<Model::Scalar>,
        x: X,
        y: Y,
    ) -> FitStat<Model>;
}

impl<Model, X, Y> FitErrBound<Model, X, Y> for FitterUnit
where
    Self: FitBound<Model, X, Y>,

    Model: FitModelErrors,
    Model::Scalar: RealField + Float,
    Model::ParamCount: Conv,
    <Model::ParamCount as Conv>::TNum: Sub<typenum::U1>,

    X: AsMatrixView<Scalar = Model::Scalar>,
    X::Points: CreateProblem,

    Y: AsMatrixView<Scalar = Model::Scalar, Points = X::Points>,

    DefaultAllocator: Allocator<<Model::ParamCount as Conv>::Nalg>
        + Allocator<DataPoints<X>>
        + Allocator<<Model::ParamCount as Conv>::Nalg, DataPoints<X>>
        + Allocator<DataPoints<X>, <Model::ParamCount as Conv>::Nalg>
        + Allocator<<Model::ParamCount as Conv>::Nalg, <Model::ParamCount as Conv>::Nalg>,
{
    #[allow(
        clippy::inline_always,
        reason = "This function is used in a single place, and, in fact, wound not exist unless I wanted to extract the type bounds to a separate trait."
    )]
    #[inline(always)]
    fn produce_stat(
        model: impl Borrow<Model>,
        report: MinimizationReport<Model::Scalar>,
        x: X,
        y: Y,
    ) -> FitStat<Model>
    where
        Model: FitModelErrors,
    {
        let model = model.borrow();
        let x = x.convert();
        let y = y.convert();

        let points =
            <Model::Scalar as NumCast>::from::<usize>(x.len()).expect("Too many data points");
        let u_params = <<Model::ParamCount as Conv>::TNum as Unsigned>::USIZE;
        let parameters =
            <Model::Scalar as NumCast>::from::<usize>(u_params).expect("Too many parameters");
        // source: https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=3213&context=facpub
        // ch. 2 Estimating Uncertainties
        let s_y_2 = if points > parameters {
            x.zip_map(&y, |xi, yi| {
                let f_x = model.evaluate(&xi);
                let dev = yi - f_x;
                dev * dev
            })
            .sum()
                / (points - parameters)
        } else {
            // WARN: add trace event here, or something
            Model::Scalar::nan()
        };
        let s_y = Float::sqrt(s_y_2);
        // thing below is (J * J^T)^-1
        let jacobian = OMatrix::<Model::Scalar, <Model::ParamCount as Conv>::Nalg, DataPoints<X>>::from_iterator_generic(
            Model::ParamCount::new_nalg(),
            DataPoints::<X>::from_usize(x.len()),
            x.iter().flat_map(|x| model.jacobian(x).into()),
        );
        let jj_t = jacobian.clone() * jacobian.transpose();
        let covariance_matrix = jj_t.try_inverse().map(|jj_x| jj_x * s_y).map_or_else(
            || {
                // WARN: add trace here too, I guess
                let col = core::iter::repeat_n(Model::Scalar::nan(), u_params).collect();
                let arr = core::iter::repeat_n(col, u_params).collect();
                GenericMatrix::from_data(GenericArrayStorage(arr))
            },
            Matrix::into_generic_matrix,
        );

        let param_errors = (0usize..u_params)
            .map(|i| Float::sqrt(covariance_matrix[(i, i)]))
            .collect();
        let errors = Model::with_errors(param_errors);

        FitStat {
            report,
            reduced_chi2: s_y_2,
            errors,
            covariance_matrix,
        }
    }
}

/// Default weights function.
#[doc(hidden)]
pub fn default_weights<Scalar: num_traits::One>(_x: Scalar, _y: Scalar) -> Scalar {
    Scalar::one()
}

#[macro_export]
#[cfg(doc)]
#[doc = include_str!("../doc/fit_macro.md")]
macro_rules! fit {
    (
        $model:expr,
        $x:expr,
        $y:expr
        $(, weights = $wights:expr)?
        $(, minimizer = $minimizer:expr)?
    ) => { ... };
}

#[macro_export]
#[cfg(not(doc))]
#[doc = include_str!("../doc/fit_macro.md")]
macro_rules! fit {
    ($model:expr, $x:expr, $y:expr $(, $par_name:ident = $par_value:expr) *) => {{
        use ::nacfahi::default_weights;
        let mut minimizer = &::nacfahi::LevenbergMarquardt::new();
        let model = $model;
        let x = $x;
        let y = $y;
        ::nacfahi::fit!(@ $($par_name = $par_value),*; model = model, x = x, y = y, minimizer = minimizer, weights = default_weights)
    }};

    (@ minimizer = $new_minimizer:expr $(, $par_name:ident = $par_value:expr) *; model = $model:ident, x = $x:ident, y = $y:ident, minimizer = $minimizer:ident, weights = $weights:ident) => {{
        use ::core::borrow::Borrow;
        let tmp = $new_minimizer;
        $minimizer = tmp.borrow();
        ::nacfahi::fit!(@ $($par_name = $par_value),*; model = $model, x = $x, y = $y, minimizer = $minimizer, weights = $weights)
    }};

    (@ weights = $new_weights:expr $(, $par_name:ident = $par_value:expr) *; model = $model:ident, x = $x:ident, y = $y:ident, minimizer = $minimizer:ident, weights = $weights:ident) => {{
        let weights = $new_weights;
        ::nacfahi::fit!(@ $($par_name = $par_value),*; model = $model, x = $x, y = $y, minimizer = $minimizer, weights = weights)
    }};

    (@; model = $model:ident, x = $x:ident, y = $y:ident, minimizer = $minimizer:ident, weights = $weights:ident) => {
        ::nacfahi::fit($model, $x, $y, $minimizer, $weights)
    };
}

/// Main interface point. For more convenient use (mostly - to omit some of the fields), you might want to look into [`macro@fit!`] macro.
///
/// **TIP**: The [`FitBound`] is an unfortunate outcome to strict type system. In case you deal with generic code, just put the `fit!` statement down, and add the bound you seemingly violate - you **should** be good after that.
#[must_use = "Minimization report is really important to check if approximation happened at all"]
pub fn fit<Model, X, Y>(
    model: &mut Model,
    x: X,
    y: Y,
    minimizer: impl Borrow<LevenbergMarquardt<Model::Scalar>>,
    weights: impl Fn(Model::Scalar, Model::Scalar) -> Model::Scalar,
) -> MinimizationReport<Model::Scalar>
where
    Model: FitModel,
    Model::Scalar: RealField,
    FitterUnit: FitBound<Model, X, Y>,
{
    FitterUnit::fit(minimizer, model, x, y, weights)
}

/// Result of [`function@fit_stat`].
#[derive(Debug)]
pub struct FitStat<Model: FitModelErrors>
where
    Model::Scalar: RealField,
{
    /// Report resulted from the fit
    pub report: MinimizationReport<Model::Scalar>,
    /// $\chi^{2}/\test{dof}$ criteria. Should be about 1 for correct fit.
    pub reduced_chi2: Model::Scalar,
    /// Type defined by model, containing parameter errors.
    ///
    /// This will usually be the model type itself, but there may be exceptions.
    pub errors: Model::OwnedModel,
    /// A parameter covariance matrix. If you don't know what this is, you can safely ignore it.
    pub covariance_matrix: GenericMatrix<Model::Scalar, Model::ParamCount, Model::ParamCount>,
}

#[macro_export]
#[cfg(doc)]
#[doc = include_str!("../doc/fit_stat_macro.md")]
macro_rules! fit_stat {
    (
        $model:expr,
        $x:expr,
        $y:expr
        $(, weights = $wights:expr)?
        $(, minimizer = $minimizer:expr)?
    ) => { ... };
}

#[macro_export]
#[cfg(not(doc))]
#[doc = include_str!("../doc/fit_stat_macro.md")]
macro_rules! fit_stat {
    ($model:expr, $x:expr, $y:expr $(, $par_name:ident = $par_value:expr) *) => {{
        use ::nacfahi::default_weights;
        let mut minimizer = &::nacfahi::LevenbergMarquardt::new();
        let model = $model;
        let x = $x;
        let y = $y;
        ::nacfahi::fit_stat!(@ $($par_name = $par_value),*; model = model, x = x, y = y, minimizer = minimizer, weights = default_weights)
    }};

    (@ minimizer = $new_minimizer:expr $(, $par_name:ident = $par_value:expr) *; model = $model:ident, x = $x:ident, y = $y:ident, minimizer = $minimizer:ident, weights = $weights:ident) => {{
        use ::core::borrow::Borrow;
        let tmp = $new_minimizer;
        $minimizer = tmp.borrow();
        ::nacfahi::fit_stat!(@ $($par_name = $par_value),*; model = $model, x = $x, y = $y, minimizer = $minimizer, weights = $weights)
    }};

    (@ weights = $new_weights:expr $(, $par_name:ident = $par_value:expr) *; model = $model:ident, x = $x:ident, y = $y:ident, minimizer = $minimizer:ident, weights = $weights:ident) => {{
        let weights = $new_weights;
        ::nacfahi::fit_stat!(@ $($par_name = $par_value),*; model = $model, x = $x, y = $y, minimizer = $minimizer, weights = weights)
    }};

    (@; model = $model:ident, x = $x:ident, y = $y:ident, minimizer = $minimizer:ident, weights = $weights:ident) => {
        ::nacfahi::fit_stat($model, $x, $y, $minimizer, $weights)
    };
}

/// Same as [`function@fit`], but outputs a bunch of other stuff alongside [`MinimizationReport`].
///
/// ### Outputs NaN
///
/// - If there are more less or the same number of data points than parameters. In this case, $\chi^{2}/\text{dof}$ is undefined, and consequently - rest of the analysis.
/// - If
///
/// ### Panics
///
/// - If data points count can't be converted to scalar type
///
/// **TIP**: The `FitDimensionsBound` is an unfortunate outcome to strict type system. In case you deal with generic code, just put the `fit!` statement down, and add the bound you seemingly violate - you **should** be good after that.
#[must_use = "Covariance matrix are the only point to call this function specifically"]
pub fn fit_stat<Model, X, Y>(
    model: &mut Model,
    x: X,
    y: Y,
    minimizer: impl Borrow<LevenbergMarquardt<Model::Scalar>>,
    weights: impl Fn(Model::Scalar, Model::Scalar) -> Model::Scalar,
) -> FitStat<Model>
where
    Model: FitModelErrors,
    Model::Scalar: RealField,
    FitterUnit: FitErrBound<Model, X, Y>,
{
    let report = FitterUnit::fit(minimizer, model.borrow_mut(), &x, &y, weights);
    FitterUnit::produce_stat(model, report, x, y)
}
