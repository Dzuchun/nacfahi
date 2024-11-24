use core::ops::{Mul, Sub};

use generic_array::{functional::FunctionalSequence, ArrayLength, GenericArray, IntoArrayLength};
use generic_array_storage::Conv;
use num_traits::{Float, One, Pow};
use typenum::{U0, U1, U2};

use crate::models::{FitModel, FitModelXDeriv};

/// Defines a function that can be used in [`ModelMap`].
///
/// This trait might seem overcomplicated to you, but the intent is to allow pre-computing of derivable parameters used in actual computation.
///
/// Same as [`FitModel`], this trait uses [`GenericArray`](generic_array::GenericArray), but I've made sure to leave hints to all the parts in the doc. To prove how easy it is, here's an actual implementation of [`Power`] map:
/// ```rust
/// # use nacfahi::models::utility::DifferentiableFunction;
/// use core::ops::Sub;
/// use num_traits::{One, Pow};
/// use typenum::{U1, U2};
/// use generic_array::GenericArray;
/// use nacfahi::models::utility::func_pars; // it's a one-liner; you can totally do this yourself
///
/// # pub struct Power<Scalar>(pub Scalar);
///
/// impl<Scalar: Clone + Sub<Scalar, Output = Scalar> + One + Pow<Scalar, Output = Scalar>>
///     DifferentiableFunction<Scalar> for Power<Scalar>
/// {
///     type ValueParams = U1;
///
///     type DerivativeParams = U2;
///
///     fn into_params(
///         self,
///     ) -> (
///         impl Into<GenericArray<Scalar, Self::ValueParams>>,
///         impl Into<GenericArray<Scalar, Self::DerivativeParams>>,
///     ) {
///         (
///             [self.0.clone()],
///             [self.0.clone(), self.0.clone().sub(Scalar::one())],
///         )
///     }
///
///     fn value(params: &GenericArray<Scalar, Self::ValueParams>, x: &Scalar) -> Scalar {
///         let [pow] = func_pars(params);
///         x.clone().pow(pow.clone())
///     }
///
///     fn derivative(params: &GenericArray<Scalar, Self::DerivativeParams>, x: &Scalar) -> Scalar {
///         let [pow, pow1] = func_pars(params);
///         x.clone().pow(pow1.clone()) * pow.clone()
///     }
/// }
/// ```
pub trait DifferentiableFunction<Scalar>: Sized {
    /// Type representing number of precomputed function parameters.
    ///
    /// **Hint**: [`typenum`]`::{U0, U1, U2, ..}` types would most likely work for you.
    ///
    /// [`typenum`]: https://docs.rs/typenum/latest/typenum/
    type ValueParams: ArrayLength;
    /// Type representing number of precomputed function derivative parameters.
    ///
    /// **Hint**: [`typenum`]`::{U0, U1, U2, ..}` types would most likely work for you.
    ///
    /// [`typenum`]: https://docs.rs/typenum/latest/typenum/
    type DerivativeParams: ArrayLength;

    /// Computes and outputs function and it's derivative pre-computed parameters.
    ///
    /// (First element - function params, second element - derivative params)
    ///
    /// **Hint**: core arrays implement `Into<Generic-Whatever>`, so you can return a tuple of two arrays, containing matching element count.
    fn into_params(
        self,
    ) -> (
        impl Into<GenericArray<Scalar, Self::ValueParams>>,
        impl Into<GenericArray<Scalar, Self::DerivativeParams>>,
    );

    /// Computes value of the function, using params previously produced by `into_params`. Note that it's an *associated function*, so you are not intended to manually pass any sort of state here.
    ///
    /// **Hint**: there's a [`func_pars`] function, you can use to transform parameters into core Rust array:
    /// ```rust
    /// use nacfahi::models::utility::func_pars;
    /// /* ... */
    /// type ValueParams = typenum::U3;
    /// /* ... */
    /// # let params = &generic_array::GenericArray::<(), ValueParams>::from_array([(), (), ()]);
    /// let [arg1, arg2, arg3] = func_pars(params);
    /// ```
    ///
    /// You totally can implement it yourself, or use the array directly, though. I just find this style more comforting.
    fn value(params: &GenericArray<Scalar, Self::ValueParams>, x: &Scalar) -> Scalar;

    /// Computes derivative of the function, using params previously produced by `into_params`. Note that it's an *associated function*, so you are not intended to manually pass any sort of state here.
    ///
    /// **Hint**: there's a [`func_pars`] function, you can use to transform parameters into core Rust array:
    /// ```rust
    /// use nacfahi::models::utility::func_pars;
    /// /* ... */
    /// type DerivativeParams = typenum::U2;
    /// /* ... */
    /// # let params = &generic_array::GenericArray::<(), DerivativeParams>::from_array([(), ()]);
    /// let [arg1, arg2] = func_pars(params);
    /// ```
    ///
    /// You totally can implement it yourself, or use the array directly, though. I just find this style more comforting.
    fn derivative(params: &GenericArray<Scalar, Self::DerivativeParams>, x: &Scalar) -> Scalar;
}

/// A helper function to convert `&GenericArray<T, N>` into `[&T; N]`.
#[inline]
pub fn func_pars<T, L: ArrayLength, const N: usize>(array: &GenericArray<T, L>) -> [&T; N]
where
    typenum::Const<N>: IntoArrayLength<ArrayLength = L>,
{
    array.map(core::convert::identity).into_array()
}

/// Function adding a constant to the model output
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Addition<Scalar>(pub Scalar);

impl<Scalar: Clone + core::ops::Add<Scalar, Output = Scalar> + num_traits::One>
    DifferentiableFunction<Scalar> for Addition<Scalar>
{
    type ValueParams = U1;
    type DerivativeParams = U0;

    #[inline]
    fn into_params(
        self,
    ) -> (
        impl Into<GenericArray<Scalar, Self::ValueParams>>,
        impl Into<GenericArray<Scalar, Self::DerivativeParams>>,
    ) {
        ([self.0], [])
    }

    #[inline]
    fn value(params: &GenericArray<Scalar, Self::ValueParams>, x: &Scalar) -> Scalar {
        let [add] = func_pars(params);
        x.clone() + add.clone()
    }

    #[inline]
    fn derivative(params: &GenericArray<Scalar, Self::DerivativeParams>, _: &Scalar) -> Scalar {
        let [] = func_pars(params);
        Scalar::one()
    }
}

/// Function multiplying by a constant model output
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Multiplier<Scalar>(pub Scalar);

impl<Scalar: Clone + core::ops::Mul<Scalar, Output = Scalar>> DifferentiableFunction<Scalar>
    for Multiplier<Scalar>
{
    type ValueParams = U1;

    type DerivativeParams = U1;

    fn into_params(
        self,
    ) -> (
        impl Into<GenericArray<Scalar, Self::ValueParams>>,
        impl Into<GenericArray<Scalar, Self::DerivativeParams>>,
    ) {
        ([self.0.clone()], [self.0])
    }

    fn value(params: &GenericArray<Scalar, Self::ValueParams>, x: &Scalar) -> Scalar {
        let [mul] = func_pars(params);
        mul.clone() * x.clone()
    }

    fn derivative(params: &GenericArray<Scalar, Self::DerivativeParams>, _x: &Scalar) -> Scalar {
        let [mul] = func_pars(params);
        mul.clone()
    }
}

/// Function potentiating model output
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Power<Scalar>(pub Scalar);

impl<Scalar: Clone + Sub<Scalar, Output = Scalar> + One + Pow<Scalar, Output = Scalar>>
    DifferentiableFunction<Scalar> for Power<Scalar>
{
    type ValueParams = U1;

    type DerivativeParams = U2;

    fn into_params(
        self,
    ) -> (
        impl Into<GenericArray<Scalar, Self::ValueParams>>,
        impl Into<GenericArray<Scalar, Self::DerivativeParams>>,
    ) {
        (
            [self.0.clone()],
            [self.0.clone(), self.0.clone().sub(Scalar::one())],
        )
    }

    fn value(params: &GenericArray<Scalar, Self::ValueParams>, x: &Scalar) -> Scalar {
        let [pow] = func_pars(params);
        x.clone().pow(pow.clone())
    }

    fn derivative(params: &GenericArray<Scalar, Self::DerivativeParams>, x: &Scalar) -> Scalar {
        let [pow, pow1] = func_pars(params);
        x.clone().pow(pow1.clone()) * pow.clone()
    }
}

/// Function applying natural logarithm (ln; logarithm base e) to the model output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LnMap;

impl<Scalar: Float> DifferentiableFunction<Scalar> for LnMap {
    type ValueParams = U0;

    type DerivativeParams = U0;

    fn into_params(
        self,
    ) -> (
        impl Into<GenericArray<Scalar, Self::ValueParams>>,
        impl Into<GenericArray<Scalar, Self::DerivativeParams>>,
    ) {
        ([], [])
    }

    fn value(params: &GenericArray<Scalar, Self::ValueParams>, x: &Scalar) -> Scalar {
        let [] = func_pars(params);
        x.ln()
    }

    fn derivative(params: &GenericArray<Scalar, Self::DerivativeParams>, &x: &Scalar) -> Scalar {
        let [] = func_pars(params);
        Scalar::one() / x
    }
}

/// Function exponentiating model output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExpMap;

impl<Scalar: Float> DifferentiableFunction<Scalar> for ExpMap {
    type ValueParams = U0;

    type DerivativeParams = U0;

    fn into_params(
        self,
    ) -> (
        impl Into<GenericArray<Scalar, Self::ValueParams>>,
        impl Into<GenericArray<Scalar, Self::DerivativeParams>>,
    ) {
        ([], [])
    }

    fn value(params: &GenericArray<Scalar, Self::ValueParams>, x: &Scalar) -> Scalar {
        let [] = func_pars(params);
        x.exp()
    }

    fn derivative(params: &GenericArray<Scalar, Self::DerivativeParams>, x: &Scalar) -> Scalar {
        let [] = func_pars(params);
        x.exp()
    }
}

/// Model equal to application of a function on top of a model.
///
/// You **can't** construct this struct manually, please use [`model_map`] for that. This is motivated by it's fields containing two separate objects (`value` and `derivative`), that are actually dependent and should be derived from each other.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModelMap<Inner: FitModel, Map: DifferentiableFunction<Inner::Scalar>> {
    /// Inner model.
    pub inner: Inner,
    /// Parameters of actual map
    value_params: GenericArray<Inner::Scalar, Map::ValueParams>,
    /// Parameters of map's derivative
    derivative_params: GenericArray<Inner::Scalar, Map::DerivativeParams>,
}

/// The only way to construct [`ModelMap`].
///
/// Second argument *simultaneously* defines function and it's derivative used to map the model.
///
/// Output type might seem complicated, but don't worry about that - the whole point is to keep it non-opaque
pub fn model_map<Inner: FitModel, Map: DifferentiableFunction<Inner::Scalar>>(
    inner: Inner,
    map: Map,
) -> ModelMap<Inner, Map> {
    let (value_params, derivative_params) = map.into_params();
    ModelMap {
        inner,
        value_params: value_params.into(),
        derivative_params: derivative_params.into(),
    }
}

impl<Inner, Map: DifferentiableFunction<Inner::Scalar>> FitModel for ModelMap<Inner, Map>
where
    Inner: FitModel,
    Inner::Scalar: Clone + core::ops::Mul<Inner::Scalar, Output = Inner::Scalar>,
{
    type Scalar = Inner::Scalar;
    type ParamCount = Inner::ParamCount;

    #[inline]
    fn evaluate(&self, x: &Self::Scalar) -> Self::Scalar {
        Map::value(&self.value_params, &self.inner.evaluate(x))
    }

    #[inline]
    fn jacobian(
        &self,
        x: &Self::Scalar,
    ) -> impl Into<GenericArray<Self::Scalar, <Self::ParamCount as Conv>::TNum>> {
        let inner_eval = self.inner.evaluate(x);
        let inner_jacobian = self.inner.jacobian(x).into();

        let map_derivative = Map::derivative(&self.derivative_params, &inner_eval);
        inner_jacobian.map(|d| map_derivative.clone() * d)
    }

    #[inline]
    fn set_params(
        &mut self,
        new_params: GenericArray<Self::Scalar, <Self::ParamCount as Conv>::TNum>,
    ) {
        self.inner.set_params(new_params);
    }

    #[inline]
    fn get_params(
        &self,
    ) -> impl Into<GenericArray<Self::Scalar, <Self::ParamCount as Conv>::TNum>> {
        self.inner.get_params()
    }
}

impl<Inner, Map> FitModelXDeriv for ModelMap<Inner, Map>
where
    Inner: FitModelXDeriv,
    Inner::Scalar: Mul<Output = Inner::Scalar>,
    Map: DifferentiableFunction<Inner::Scalar>,
    Self: FitModel<Scalar = Inner::Scalar, ParamCount = Inner::ParamCount>,
{
    #[inline]
    fn deriv_x(&self, x: &Self::Scalar) -> Self::Scalar {
        let y = self.inner.evaluate(x);
        let y_x = self.inner.deriv_x(x);
        let z_y = Map::derivative(&self.derivative_params, &y);
        z_y * y_x
    }
}

crate::test_model_derivative!(
    exponent_offset,
    ModelMap<Exponent<f64>, Addition<f64>>,
    model_map(Exponent { a: 2.0, b: -0.242 }, Addition(-34.2)),
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
    exponent_multiplier,
    ModelMap<Exponent<f64>, Multiplier<f64>>,
    model_map(Exponent { a: 2.0, b: -0.242 }, Multiplier(-3.232)),
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
    exponent_power,
    ModelMap<Exponent<f64>, Power<f64>>,
    model_map(Exponent { a: 2.0, b: -0.242 }, Power(3.0)),
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
    gaussian_power,
    ModelMap<Gaussian<f64>, Power<f64>>,
    model_map(
        Gaussian {
            a: -3.0,
            s: 0.3,
            x_c: 3.0
        },
        Power(3.0)
    ),
    [
        (0.0, -1.0),
        (1.0, -4.0),
        (2.0, -5.0),
        (3.0, 6.0),
        (4.0, 2.0),
        (5.0, 2.5)
    ]
);
