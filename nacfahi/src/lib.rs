#![no_std] // <-- see that attr? No shit!

use core::borrow::Borrow;
use core::ops::Sub;

use dyn_problem::DynOptimizationProblem;
use models::FitModel;

use const_problem::ConstOptimizationProblem;
use generic_array_storage::Conv;
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::Matrix;
use nalgebra::{ComplexField, Dim, DimMax, DimMaximum, DimMin, Dyn, MatrixView, RealField};


use num_traits::Float;

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
/// Main interface point. For more convenient use (mostly - to omit some of the fields), you might want to look into [`macro@fit!`] macro.
///
/// P.S.: in case type bounds look terrifying to you - don't worry, they are worse than you can imagine. See examples in documentation for guiding (they are tested on build, ensuring their validity).
#[must_use = "Minimization report is really important to check if approximation happened at all"]
pub fn fit<Scalar, Entity, X, Y>(
    model: Entity,
    x: X,
    y: Y,
    minimizer: impl Borrow<LevenbergMarquardt<Scalar>>,
    weights: impl Fn(Scalar, Scalar) -> Scalar,
) -> MinimizationReport<Scalar>
where
    Scalar: ComplexField + RealField + Float + Copy,
    Entity: FitModel<Scalar>,
    Entity::ParamCount: Conv<ArrLen = Entity::ParamCount> + Sub<typenum::U1>,
    X: AsMatrixView<Scalar>,
    Y: AsMatrixView<Scalar, Points = X::Points>,

    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<<Entity::ParamCount as Conv>::Nalg>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<X::Points>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<X::Points, <Entity::ParamCount as Conv>::Nalg>,

    X::Points: DimMax<<Entity::ParamCount as Conv>::Nalg>
        + DimMin<<Entity::ParamCount as Conv>::Nalg>
        + CreateProblem<Scalar, Nalg = X::Points>,
    <Entity::ParamCount as Conv>::Nalg: DimMax<X::Points> + DimMin<X::Points>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Reallocator<
        Scalar,
        X::Points,
        <Entity::ParamCount as Conv>::Nalg,
        DimMaximum<X::Points, <Entity::ParamCount as Conv>::Nalg>,
        <Entity::ParamCount as Conv>::Nalg,
    >,
{
    let x = <X as AsMatrixView<_>>::convert(&x);
    let y = <Y as AsMatrixView<_>>::convert(&y);
    let problem = <X::Points as CreateProblem<Scalar>>::create::<'_, Entity::ParamCount, _>(
        x, y, model, weights,
    );
    let (_, report) = LevenbergMarquardt::minimize::<
        <Entity::ParamCount as Conv>::Nalg,
        X::Points,
        _,
    >(minimizer.borrow(), problem);
    report
}
