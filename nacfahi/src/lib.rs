#![doc = include_str!("../../README.md")]
#![no_std] // <-- see that attr? No shit!

use core::borrow::Borrow;
use core::ops::Sub;

use dyn_problem::DynOptimizationProblem;
use generic_array::{ArrayLength, GenericArray};
use models::{FitModel, FitModelErrors};

use const_problem::ConstOptimizationProblem;
use generic_array_storage::{Conv, GenericArrayStorage, GenericMatrix, GenericMatrixFromExt};
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::allocator::{Allocator, Reallocator};
use nalgebra::{
    ComplexField, DefaultAllocator, Dim, DimMax, DimMaximum, DimMin, Dyn, MatrixView, RealField,
};

use nalgebra::{Matrix, OMatrix};
use typenum::Unsigned;

use num_traits::Float;

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

#[doc(hidden)]
mod dyn_problem;

#[doc = include_str!("../doc/as_matrix_view.md")]
pub trait AsMatrixView<Scalar> {
    /// Type representing data points count.
    type Points: Dim;

    /// Creates data view from type.
    fn convert(&self) -> MatrixView<'_, Scalar, Self::Points, nalgebra::U1>;
}

impl<'r, Scalar, T: AsMatrixView<Scalar> + ?Sized> AsMatrixView<Scalar> for &'r T {
    type Points = T::Points;

    fn convert(&self) -> MatrixView<'_, Scalar, Self::Points, nalgebra::U1> {
        <T as AsMatrixView<Scalar>>::convert(self)
    }
}

impl<'r, Scalar, T: AsMatrixView<Scalar> + ?Sized> AsMatrixView<Scalar> for &'r mut T {
    type Points = T::Points;

    fn convert(&self) -> MatrixView<'_, Scalar, Self::Points, nalgebra::U1> {
        <T as AsMatrixView<Scalar>>::convert(self)
    }
}

impl<Scalar: nalgebra::Scalar, const N: usize> AsMatrixView<Scalar> for [Scalar; N] {
    type Points = nalgebra::Const<N>;

    fn convert(&self) -> MatrixView<'_, Scalar, Self::Points, nalgebra::U1> {
        MatrixView::<'_, Scalar, Self::Points, nalgebra::U1>::from_slice(self)
    }
}

impl<Scalar: nalgebra::Scalar> AsMatrixView<Scalar> for [Scalar] {
    type Points = Dyn;

    fn convert(&self) -> MatrixView<'_, Scalar, Self::Points, nalgebra::U1> {
        MatrixView::<'_, Scalar, Self::Points, nalgebra::U1>::from_slice(self, self.len())
    }
}

impl<Scalar, Points: Dim, S> AsMatrixView<Scalar> for Matrix<Scalar, Points, nalgebra::U1, S>
where
    S: nalgebra::storage::RawStorage<Scalar, Points, RStride = nalgebra::U1, CStride = Points>,
{
    type Points = Points;

    fn convert(&self) -> MatrixView<'_, Scalar, Self::Points, nalgebra::U1> {
        self.as_view()
    }
}

impl<Scalar: nalgebra::Scalar, Points: ArrayLength + Conv<TNum = Points>> AsMatrixView<Scalar>
    for GenericArray<Scalar, Points>
{
    type Points = <Points as Conv>::Nalg;

    fn convert(&self) -> MatrixView<'_, Scalar, Self::Points, nalgebra::U1> {
        MatrixView::<'_, Scalar, Self::Points, nalgebra::U1>::from_slice(self)
    }
}

/// Indicates *how exactly* matrix data views should be converted into [`LeastSquaresProblem`](https://docs.rs/levenberg-marquardt/latest/levenberg_marquardt/trait.LeastSquaresProblem.html) suitable for [`levenberg_marquardt`](https://docs.rs/levenberg-marquardt/latest/levenberg_marquardt/index.html) operation.
///
/// In case it seems pointless, there are actually currently two fundamentally different problem types corresponding to const (stack-based) and dyn (alloc-based) problems. They should be hidden from docs, but this trait's implementation points should get you started (in case you're curious, or something).
pub trait CreateProblem<Scalar> {
    /// [`nalgebra`]-facing type, for view size definition
    type Nalg: Dim;

    /// Creates a problem from data views and arbitrary model.
    fn create<'d, Entity: FitModel<Scalar> + 'd>(
        x: MatrixView<'d, Scalar, Self::Nalg, nalgebra::U1>,
        y: MatrixView<'d, Scalar, Self::Nalg, nalgebra::U1>,
        entity: Entity,
        weights: impl Fn(Scalar, Scalar) -> Scalar + 'd,
    ) -> impl LeastSquaresProblem<Scalar, Self::Nalg, <Entity::ParamCount as Conv>::Nalg> + 'd
    where
        Scalar: ComplexField + Copy,
        DefaultAllocator: Allocator<<Entity::ParamCount as Conv>::Nalg, nalgebra::U1>,
        DefaultAllocator: Allocator<Self::Nalg, nalgebra::U1>,
        DefaultAllocator: Allocator<Self::Nalg, <Entity::ParamCount as Conv>::Nalg>;
}

impl<Scalar, const POINTS: usize> CreateProblem<Scalar> for typenum::Const<POINTS>
where
    Self: Conv<Nalg = nalgebra::Const<POINTS>>,
{
    type Nalg = nalgebra::Const<POINTS>;

    fn create<'d, Entity: FitModel<Scalar> + 'd>(
        x: MatrixView<'d, Scalar, Self::Nalg, nalgebra::U1>,
        y: MatrixView<'d, Scalar, Self::Nalg, nalgebra::U1>,
        entity: Entity,
        weights: impl Fn(Scalar, Scalar) -> Scalar + 'd,
    ) -> impl LeastSquaresProblem<Scalar, Self::Nalg, <Entity::ParamCount as Conv>::Nalg> + 'd
    where
        Scalar: ComplexField + Copy,
        DefaultAllocator: Allocator<<Entity::ParamCount as Conv>::Nalg, nalgebra::U1>,
        DefaultAllocator: Allocator<Self::Nalg, nalgebra::U1>,
        DefaultAllocator: Allocator<Self::Nalg, <Entity::ParamCount as Conv>::Nalg>,
    {
        ConstOptimizationProblem::<'d, Scalar, Self, Entity, _> {
            entity,
            x,
            y,
            weights,
        }
    }
}

impl<Scalar, const POINTS: usize> CreateProblem<Scalar> for nalgebra::Const<POINTS>
where
    Self: Conv<Nalg = Self>,
{
    type Nalg = Self;

    fn create<'d, Entity: FitModel<Scalar> + 'd>(
        x: MatrixView<'d, Scalar, Self::Nalg, nalgebra::U1>,
        y: MatrixView<'d, Scalar, Self::Nalg, nalgebra::U1>,
        entity: Entity,
        weights: impl Fn(Scalar, Scalar) -> Scalar + 'd,
    ) -> impl LeastSquaresProblem<Scalar, Self::Nalg, <Entity::ParamCount as Conv>::Nalg> + 'd
    where
        Scalar: ComplexField + Copy,
        DefaultAllocator: Allocator<<Entity::ParamCount as Conv>::Nalg, nalgebra::U1>,
        DefaultAllocator: Allocator<Self::Nalg, nalgebra::U1>,
        DefaultAllocator: Allocator<Self::Nalg, <Entity::ParamCount as Conv>::Nalg>,
    {
        ConstOptimizationProblem::<'d, Scalar, Self, Entity, _> {
            entity,
            x,
            y,
            weights,
        }
    }
}

impl<Scalar> CreateProblem<Scalar> for Dyn {
    type Nalg = Self;

    fn create<'d, Entity: FitModel<Scalar> + 'd>(
        x: MatrixView<'d, Scalar, Self::Nalg, nalgebra::U1>,
        y: MatrixView<'d, Scalar, Self::Nalg, nalgebra::U1>,
        entity: Entity,
        weights: impl Fn(Scalar, Scalar) -> Scalar + 'd,
    ) -> impl LeastSquaresProblem<Scalar, Self::Nalg, <Entity::ParamCount as Conv>::Nalg> + 'd
    where
        Scalar: ComplexField + Copy,
        DefaultAllocator: Allocator<<Entity::ParamCount as Conv>::Nalg, nalgebra::U1>,
        DefaultAllocator: Allocator<Self::Nalg, nalgebra::U1>,
        DefaultAllocator: Allocator<Self::Nalg, <Entity::ParamCount as Conv>::Nalg>,
    {
        DynOptimizationProblem::<'d, Scalar, Entity, _> {
            entity,
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

/// A helper trait to simplify type bounds for a user. You probably should no see this.
///
/// In case you do get a "type does not implement" type or error with this trait... I'm sorry.
#[doc(hidden)]
pub trait FitDimensionsBound<Scalar: RealField, ParamCount, Points: Dim> {
    fn fit<Model: FitModel<Scalar, ParamCount = ParamCount>>(
        minimizer: impl Borrow<LevenbergMarquardt<Scalar>>,
        model: Model,
        x: Matrix<
            Scalar,
            Points,
            nalgebra::Const<1>,
            nalgebra::ViewStorage<
                '_,
                Scalar,
                Points,
                nalgebra::Const<1>,
                nalgebra::Const<1>,
                Points,
            >,
        >,
        y: Matrix<
            Scalar,
            Points,
            nalgebra::Const<1>,
            nalgebra::ViewStorage<
                '_,
                Scalar,
                Points,
                nalgebra::Const<1>,
                nalgebra::Const<1>,
                Points,
            >,
        >,
        weights: impl Fn(Scalar, Scalar) -> Scalar,
    ) -> MinimizationReport<Scalar>;
}

impl<Scalar, ParamCount, Points> FitDimensionsBound<Scalar, ParamCount, Points> for FitterUnit
where
    Scalar: RealField + Float,
    ParamCount: Conv<TNum = ParamCount> + Sub<typenum::U1>,

    DefaultAllocator: Allocator<<ParamCount as Conv>::Nalg>,
    DefaultAllocator: Allocator<Points>,
    DefaultAllocator: Allocator<Points, <ParamCount as Conv>::Nalg>,

    Points: CreateProblem<Scalar, Nalg = Points>,
    Points: DimMax<<ParamCount as Conv>::Nalg>
        + DimMin<<ParamCount as Conv>::Nalg>
        + CreateProblem<Scalar, Nalg = Points>,
    <ParamCount as Conv>::Nalg: DimMax<Points> + DimMin<Points>,
    DefaultAllocator: Reallocator<
        Scalar,
        Points,
        <ParamCount as Conv>::Nalg,
        DimMaximum<Points, <ParamCount as Conv>::Nalg>,
        <ParamCount as Conv>::Nalg,
    >,
{
    #[allow(
        clippy::inline_always,
        reason = "This function is used in a single place, and, in fact, wound not exist unless I wanted to extract the type bounds to a separate trait."
    )]
    #[inline(always)]
    fn fit<Model: FitModel<Scalar, ParamCount = ParamCount>>(
        minimizer: impl Borrow<LevenbergMarquardt<Scalar>>,
        model: Model,
        x: Matrix<
            Scalar,
            Points,
            nalgebra::Const<1>,
            nalgebra::ViewStorage<
                '_,
                Scalar,
                Points,
                nalgebra::Const<1>,
                nalgebra::Const<1>,
                Points,
            >,
        >,
        y: Matrix<
            Scalar,
            Points,
            nalgebra::Const<1>,
            nalgebra::ViewStorage<
                '_,
                Scalar,
                Points,
                nalgebra::Const<1>,
                nalgebra::Const<1>,
                Points,
            >,
        >,
        weights: impl Fn(Scalar, Scalar) -> Scalar,
    ) -> MinimizationReport<Scalar> {
        let problem = <Points as CreateProblem<Scalar>>::create::<'_, Model>(x, y, model, weights);
        let (_, report) = LevenbergMarquardt::minimize::<<ParamCount as Conv>::Nalg, Points, _>(
            minimizer.borrow(),
            problem,
        );
        report
    }
}

/// A helper trait to simplify type bounds for a user. You probably should no see this.
///
/// In case you do get a "type does not implement" type or error with this trait... I'm sorry.
#[doc(hidden)]
pub trait FitErrDimensionsBound<Scalar: RealField, ParamCount: Conv, Points: Dim>:
    FitDimensionsBound<Scalar, ParamCount, Points>
{
    fn produce_stat<Model: FitModel<Scalar, ParamCount = ParamCount> + FitModelErrors<Scalar>>(
        model: Model,
        report: MinimizationReport<Scalar>,
        x: Matrix<
            Scalar,
            Points,
            nalgebra::Const<1>,
            nalgebra::ViewStorage<
                '_,
                Scalar,
                Points,
                nalgebra::Const<1>,
                nalgebra::Const<1>,
                Points,
            >,
        >,
        y: Matrix<
            Scalar,
            Points,
            nalgebra::Const<1>,
            nalgebra::ViewStorage<
                '_,
                Scalar,
                Points,
                nalgebra::Const<1>,
                nalgebra::Const<1>,
                Points,
            >,
        >,
    ) -> FitStat<Scalar, Model>;
}

impl<Scalar: RealField + Float, ParamCount: Conv, Points: Dim>
    FitErrDimensionsBound<Scalar, ParamCount, Points> for FitterUnit
where
    FitterUnit: FitDimensionsBound<Scalar, ParamCount, Points>,
    DefaultAllocator: Allocator<ParamCount::Nalg>
        + Allocator<Points>
        + Allocator<ParamCount::Nalg, Points>
        + Allocator<Points, ParamCount::Nalg>
        + Allocator<ParamCount::Nalg, ParamCount::Nalg>,
{
    #[allow(
        clippy::inline_always,
        reason = "This function is used in a single place, and, in fact, wound not exist unless I wanted to extract the type bounds to a separate trait."
    )]
    #[inline(always)]
    fn produce_stat<Model: FitModel<Scalar, ParamCount = ParamCount> + FitModelErrors<Scalar>>(
        model: Model,
        report: MinimizationReport<Scalar>,
        x: Matrix<
            Scalar,
            Points,
            nalgebra::Const<1>,
            nalgebra::ViewStorage<
                '_,
                Scalar,
                Points,
                nalgebra::Const<1>,
                nalgebra::Const<1>,
                Points,
            >,
        >,
        y: Matrix<
            Scalar,
            Points,
            nalgebra::Const<1>,
            nalgebra::ViewStorage<
                '_,
                Scalar,
                Points,
                nalgebra::Const<1>,
                nalgebra::Const<1>,
                Points,
            >,
        >,
    ) -> FitStat<Scalar, Model> {
        let points = Scalar::from_usize(x.len()).expect("Too many data points");
        let u_params = <<Model::ParamCount as Conv>::TNum as Unsigned>::USIZE;
        let parameters = Scalar::from_usize(u_params).expect("Too many parameters");
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
            Scalar::nan()
        };
        let s_y = Float::sqrt(s_y_2);
        // thing below is (J * J^T)^-1
        let jacobian = OMatrix::<Scalar, ParamCount::Nalg, Points>::from_iterator_generic(
            ParamCount::new_nalg(),
            Points::from_usize(x.len()),
            x.iter().flat_map(|x| model.jacobian(x).into()),
        );
        let jj_t = jacobian.clone() * jacobian.transpose();
        let covariance_matrix = jj_t.try_inverse().map(|jj_x| jj_x * s_y).map_or_else(
            || {
                // WARN: add trace here too, I guess
                let col = core::iter::repeat_n(Scalar::nan(), u_params).collect();
                let arr = core::iter::repeat_n(col, u_params).collect();
                GenericMatrix::from_data(GenericArrayStorage(arr))
            },
            Matrix::into_generic_matrix,
        );

        let param_errors = (0usize..u_params)
            .map(|i| Float::sqrt(covariance_matrix[(i, i)]))
            .collect();
        let errors = model.with_errors(param_errors);

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
/// **TIP**: The [`FitDimensionsBound`] is an unfortunate outcome to strict type system. In case you deal with generic code, just put the `fit!` statement down, and add the bound you seemingly violate - you **should** be good after that.
#[must_use = "Minimization report is really important to check if approximation happened at all"]
pub fn fit<Scalar, Model, X, Y>(
    model: Model,
    x: X,
    y: Y,
    minimizer: impl Borrow<LevenbergMarquardt<Scalar>>,
    weights: impl Fn(Scalar, Scalar) -> Scalar,
) -> MinimizationReport<Scalar>
where
    Scalar: ComplexField + RealField + Float + Copy,
    Model: FitModel<Scalar>,
    <Model::ParamCount as Conv>::TNum: Sub<typenum::U1>,
    X: AsMatrixView<Scalar>,
    Y: AsMatrixView<Scalar, Points = X::Points>,
    FitterUnit: FitDimensionsBound<Scalar, Model::ParamCount, X::Points>,
{
    let x = <X as AsMatrixView<_>>::convert(&x);
    let y = <Y as AsMatrixView<_>>::convert(&y);
    FitterUnit::fit(minimizer, model, x, y, weights)
}

/// Result of [`function@fit_stat`].
#[derive(Debug)]
pub struct FitStat<Scalar: RealField, Model: FitModel<Scalar> + FitModelErrors<Scalar>>
where
    Model::ParamCount: Conv,
{
    /// Report resulted from the fit
    pub report: MinimizationReport<Scalar>,
    /// $\chi^{2}/\test{dof}$ criteria. Should be about 1 for correct fit.
    pub reduced_chi2: Scalar,
    /// Type defined by model, containing parameter errors.
    ///
    /// This will usually be the model type itself, but there may be exceptions.
    pub errors: Model::OwnedModel,
    /// A parameter covariance matrix. If you don't know what this is, you can safely ignore it.
    pub covariance_matrix: GenericMatrix<Scalar, Model::ParamCount, Model::ParamCount>,
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
pub fn fit_stat<Scalar, Model, X, Y>(
    mut model: Model,
    x: X,
    y: Y,
    minimizer: impl Borrow<LevenbergMarquardt<Scalar>>,
    weights: impl Fn(Scalar, Scalar) -> Scalar,
) -> FitStat<Scalar, Model>
where
    Scalar: ComplexField + RealField + Float + Copy,
    Model: FitModel<Scalar> + FitModelErrors<Scalar>,
    <Model::ParamCount as Conv>::TNum: Sub<typenum::U1>,
    X: AsMatrixView<Scalar>,
    Y: AsMatrixView<Scalar, Points = X::Points>,
    FitterUnit: FitDimensionsBound<Scalar, Model::ParamCount, X::Points>
        + FitErrDimensionsBound<Scalar, Model::ParamCount, X::Points>,
{
    let x = x.convert();
    let y = y.convert();
    let report = fit(&mut model, &x, &y, minimizer, weights);

    FitterUnit::produce_stat(model, report, x, y)
}
