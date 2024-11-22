#![doc = include_str!("../../README.md")]
#![no_std] // <-- see that attr? No shit!

use core::borrow::Borrow;
use core::ops::Sub;

use dyn_problem::DynOptimizationProblem;
use generic_array::{ArrayLength, GenericArray};
use models::{FitModel, FitModelErrors};

use const_problem::ConstOptimizationProblem;
use generic_array_storage::Conv;
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{ComplexField, Dim, DimMax, DimMaximum, DimMin, Dyn, MatrixView, RealField};
use nalgebra::{Matrix, OMatrix};

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

impl<Scalar: nalgebra::Scalar, Points: ArrayLength + Conv<ArrLen = Points>> AsMatrixView<Scalar>
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
    fn create<'d, Params: Conv, Entity: FitModel<Scalar, ParamCount = Params::ArrLen> + 'd>(
        x: MatrixView<'d, Scalar, Self::Nalg, nalgebra::U1>,
        y: MatrixView<'d, Scalar, Self::Nalg, nalgebra::U1>,
        entity: Entity,
        weights: impl Fn(Scalar, Scalar) -> Scalar + 'd,
    ) -> impl LeastSquaresProblem<Scalar, Self::Nalg, Params::Nalg> + 'd
    where
        Scalar: ComplexField + Copy,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<Params::Nalg, nalgebra::U1>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<Self::Nalg, nalgebra::U1>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<Self::Nalg, Params::Nalg>;
}

impl<Scalar, const POINTS: usize> CreateProblem<Scalar> for typenum::Const<POINTS>
where
    Self: Conv<Nalg = nalgebra::Const<POINTS>>,
{
    type Nalg = nalgebra::Const<POINTS>;

    fn create<'d, Params: Conv, Entity: FitModel<Scalar, ParamCount = Params::ArrLen> + 'd>(
        x: MatrixView<'d, Scalar, Self::Nalg, nalgebra::U1>,
        y: MatrixView<'d, Scalar, Self::Nalg, nalgebra::U1>,
        entity: Entity,
        weights: impl Fn(Scalar, Scalar) -> Scalar + 'd,
    ) -> impl LeastSquaresProblem<Scalar, Self::Nalg, Params::Nalg> + 'd
    where
        Scalar: ComplexField + Copy,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<Params::Nalg, nalgebra::U1>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<Self::Nalg, nalgebra::U1>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<Self::Nalg, Params::Nalg>,
    {
        ConstOptimizationProblem::<'d, Scalar, Params, Self, Entity, _> {
            entity,
            x,
            y,
            weights,
            param_count: core::marker::PhantomData,
        }
    }
}

impl<Scalar, const POINTS: usize> CreateProblem<Scalar> for nalgebra::Const<POINTS>
where
    Self: Conv<Nalg = Self>,
{
    type Nalg = Self;

    fn create<'d, Params: Conv, Entity: FitModel<Scalar, ParamCount = Params::ArrLen> + 'd>(
        x: MatrixView<'d, Scalar, Self::Nalg, nalgebra::U1>,
        y: MatrixView<'d, Scalar, Self::Nalg, nalgebra::U1>,
        entity: Entity,
        weights: impl Fn(Scalar, Scalar) -> Scalar + 'd,
    ) -> impl LeastSquaresProblem<Scalar, Self::Nalg, Params::Nalg> + 'd
    where
        Scalar: ComplexField + Copy,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<Params::Nalg, nalgebra::U1>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<Self::Nalg, nalgebra::U1>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<Self::Nalg, Params::Nalg>,
    {
        ConstOptimizationProblem::<'d, Scalar, Params, Self, Entity, _> {
            entity,
            x,
            y,
            weights,
            param_count: core::marker::PhantomData,
        }
    }
}

impl<Scalar> CreateProblem<Scalar> for Dyn {
    type Nalg = Self;

    fn create<'d, Params: Conv, Entity: FitModel<Scalar, ParamCount = Params::ArrLen> + 'd>(
        x: MatrixView<'d, Scalar, Self::Nalg, nalgebra::U1>,
        y: MatrixView<'d, Scalar, Self::Nalg, nalgebra::U1>,
        entity: Entity,
        weights: impl Fn(Scalar, Scalar) -> Scalar + 'd,
    ) -> impl LeastSquaresProblem<Scalar, Self::Nalg, Params::Nalg> + 'd
    where
        Scalar: ComplexField + Copy,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<Params::Nalg, nalgebra::U1>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<Self::Nalg, nalgebra::U1>,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<Self::Nalg, Params::Nalg>,
    {
        DynOptimizationProblem::<'d, Scalar, Params, Entity, _> {
            entity,
            x,
            y,
            weights,
            param_count: core::marker::PhantomData,
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
/// P.S.: in case type bounds look terrifying to you - don't worry, they are worse than you can imagine. See examples in documentation for guiding (they are tested on build, ensuring their validity).
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
    Model::ParamCount: Conv<ArrLen = Model::ParamCount> + Sub<typenum::U1>,
    X: AsMatrixView<Scalar>,
    Y: AsMatrixView<Scalar, Points = X::Points>,

    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<<Model::ParamCount as Conv>::Nalg>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<X::Points>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<X::Points, <Model::ParamCount as Conv>::Nalg>,

    X::Points: DimMax<<Model::ParamCount as Conv>::Nalg>
        + DimMin<<Model::ParamCount as Conv>::Nalg>
        + CreateProblem<Scalar, Nalg = X::Points>,
    <Model::ParamCount as Conv>::Nalg: DimMax<X::Points> + DimMin<X::Points>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Reallocator<
        Scalar,
        X::Points,
        <Model::ParamCount as Conv>::Nalg,
        DimMaximum<X::Points, <Model::ParamCount as Conv>::Nalg>,
        <Model::ParamCount as Conv>::Nalg,
    >,
{
    let x = <X as AsMatrixView<_>>::convert(&x);
    let y = <Y as AsMatrixView<_>>::convert(&y);
    let problem = <X::Points as CreateProblem<Scalar>>::create::<'_, Model::ParamCount, _>(
        x, y, model, weights,
    );
    let (_, report) = LevenbergMarquardt::minimize::<<Model::ParamCount as Conv>::Nalg, X::Points, _>(
        minimizer.borrow(),
        problem,
    );
    report
}

#[doc(hidden)]
type ModelNalg<Scalar, Model> = <<Model as FitModel<Scalar>>::ParamCount as Conv>::Nalg;

/// Result of [`function@fit_stat`].
#[derive(Debug)]
pub struct FitStat<Scalar: RealField, Model: FitModel<Scalar> + FitModelErrors<Scalar>>
where
    <Model as FitModel<Scalar>>::ParamCount: Conv,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<
        <Model::ParamCount as Conv>::Nalg,
        <Model::ParamCount as Conv>::Nalg,
    >,
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
    pub covariance_matrix: OMatrix<Scalar, ModelNalg<Scalar, Model>, ModelNalg<Scalar, Model>>,
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
    Model::ParamCount: Conv<ArrLen = Model::ParamCount> + Sub<typenum::U1>,
    X: AsMatrixView<Scalar>,
    Y: AsMatrixView<Scalar, Points = X::Points>,

    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<<Model::ParamCount as Conv>::Nalg>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<X::Points>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<X::Points, <Model::ParamCount as Conv>::Nalg>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<
        <<Model as FitModel<Scalar>>::ParamCount as Conv>::Nalg,
        X::Points,
    >,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<
        <Model::ParamCount as Conv>::Nalg,
        <Model::ParamCount as Conv>::Nalg,
    >,

    X::Points: DimMax<<Model::ParamCount as Conv>::Nalg>
        + DimMin<<Model::ParamCount as Conv>::Nalg>
        + CreateProblem<Scalar, Nalg = X::Points>,
    <Model::ParamCount as Conv>::Nalg: DimMax<X::Points> + DimMin<X::Points>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Reallocator<
        Scalar,
        X::Points,
        <Model::ParamCount as Conv>::Nalg,
        DimMaximum<X::Points, <Model::ParamCount as Conv>::Nalg>,
        <Model::ParamCount as Conv>::Nalg,
    >,
{
    let x = x.convert();
    let y = y.convert();
    let points = Scalar::from_usize(x.len()).expect("Too many data points");
    let parameters = Scalar::from_usize(<Model as FitModel<Scalar>>::ParamCount::NUM)
        .expect("Too many parameters");
    let report = fit(&mut model, &x, &y, minimizer, weights);

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
    let jacobian = OMatrix::<
        Scalar,
        <<Model as FitModel<Scalar>>::ParamCount as Conv>::Nalg,
        X::Points,
    >::from_iterator_generic(
        Model::ParamCount::new_nalg(),
        X::Points::from_usize(x.len()),
        x.iter().flat_map(|x| model.jacobian(x).into()),
    );
    let jj_t = jacobian.clone() * jacobian.transpose();
    let covariance_matrix = jj_t.try_inverse().unwrap_or_else(|| {
        // WARN: add trace here too, I guess
        Matrix::<
            Scalar,
            <<Model as FitModel<Scalar>>::ParamCount as Conv>::Nalg,
            <<Model as FitModel<Scalar>>::ParamCount as Conv>::Nalg,
            <nalgebra::DefaultAllocator as nalgebra::allocator::Allocator<
                <<Model as FitModel<Scalar>>::ParamCount as Conv>::Nalg,
                <<Model as FitModel<Scalar>>::ParamCount as Conv>::Nalg,
            >>::Buffer<Scalar>,
        >::from_element(Scalar::nan())
    }) * s_y;

    let param_errors = covariance_matrix
        .diagonal()
        .into_iter()
        .copied()
        .map(Float::sqrt)
        .collect();
    let errors = model.with_errors(param_errors);

    FitStat {
        report,
        reduced_chi2: s_y_2,
        errors,
        covariance_matrix,
    }
}
