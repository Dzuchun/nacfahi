use core::{
    marker::PhantomData,
    ops::{Mul, Sub},
};

use generic_array::{functional::FunctionalSequence, GenericArray};
use num_traits::{Float, One, Pow};

use crate::models::{FitModel, FitModelXDeriv};

/// Defines a function that can be used in [`ModelMap`].
///
/// This trait might seem overcomplicated to you, but the intent is to allow creation of closures representing value and derivative of a map.
///
/// Contrary to [`FitModel`], this trait is very easy to implement, no [`GenericArray`](generic_array::GenericArray)s required! To prove that, here's actual implementation for power function:
///
/// ```rust
/// # use nacfahi::models::utility::DifferentiableFunction;
/// # use core::ops::Sub;
/// # use num_traits::{One, Pow};
/// pub struct Power<Scalar>(pub Scalar);
///
/// impl<Scalar: Clone + Sub<Scalar, Output = Scalar> + One + Pow<Scalar, Output = Scalar>>
///     DifferentiableFunction<1, 1, Scalar> for Power<Scalar>
/// {
///     fn into_params(self) -> ([Scalar; 1], [Scalar; 1]) {
///         ([self.0.clone()], [self.0.clone().sub(Scalar::one())])
///     }
///
///     fn value([pow]: &[Scalar; 1], x: &Scalar) -> Scalar {
///         x.clone().pow(pow.clone())
///     }
///
///     fn derivative([pow]: &[Scalar; 1], x: &Scalar) -> Scalar {
///         x.clone().pow(pow.clone())
///     }
/// }
/// ```
pub trait DifferentiableFunction<const VALUE_PARAMS: usize, const DERIVATIVE_PARAMS: usize, Scalar>:
    Sized
{
    /// Should produce functions used by `value` and `derivative` functions.
    fn into_params(self) -> ([Scalar; VALUE_PARAMS], [Scalar; DERIVATIVE_PARAMS]);

    /// Computes value of the function, using params previously produced `into_params`. Note that it's an *associated function*, so you are not intended to pass any sort of state here.
    fn value(params: &[Scalar; VALUE_PARAMS], x: &Scalar) -> Scalar;

    /// Computes derivative of the function, using params previously produced `into_params`. Note that it's an *associated function*, so you are not intended to pass any sort of state here.
    fn derivative(params: &[Scalar; DERIVATIVE_PARAMS], x: &Scalar) -> Scalar;
}

/// Function adding a constant to the model output
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Addition<Scalar>(pub Scalar);

impl<Scalar: Clone + core::ops::Add<Scalar, Output = Scalar> + num_traits::One>
    DifferentiableFunction<1, 0, Scalar> for Addition<Scalar>
{
    #[inline]
    fn into_params(self) -> ([Scalar; 1], [Scalar; 0]) {
        ([self.0], [])
    }

    #[inline]
    fn value([add]: &[Scalar; 1], x: &Scalar) -> Scalar {
        x.clone() + add.clone()
    }

    #[inline]
    fn derivative([]: &[Scalar; 0], _: &Scalar) -> Scalar {
        Scalar::one()
    }
}

/// Function multiplying by a constant model output
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Multiplier<Scalar>(pub Scalar);

impl<Scalar: Clone + core::ops::Mul<Scalar, Output = Scalar>> DifferentiableFunction<1, 1, Scalar>
    for Multiplier<Scalar>
{
    #[inline]
    fn into_params(self) -> ([Scalar; 1], [Scalar; 1]) {
        ([self.0.clone()], [self.0])
    }

    #[inline]
    fn value([mul]: &[Scalar; 1], x: &Scalar) -> Scalar {
        mul.clone() * x.clone()
    }

    #[inline]
    fn derivative([mul]: &[Scalar; 1], _: &Scalar) -> Scalar {
        mul.clone()
    }
}

/// Function potentiating model output
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Power<Scalar>(pub Scalar);

impl<Scalar: Clone + Sub<Scalar, Output = Scalar> + One + Pow<Scalar, Output = Scalar>>
    DifferentiableFunction<1, 1, Scalar> for Power<Scalar>
{
    #[inline]
    fn into_params(self) -> ([Scalar; 1], [Scalar; 1]) {
        ([self.0.clone()], [self.0.clone().sub(Scalar::one())])
    }

    #[inline]
    fn value([pow]: &[Scalar; 1], x: &Scalar) -> Scalar {
        x.clone().pow(pow.clone())
    }

    #[inline]
    fn derivative([pow]: &[Scalar; 1], x: &Scalar) -> Scalar {
        x.clone().pow(pow.clone())
    }
}

/// Function applying natural logarithm (ln; logarithm base e) to the model output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LnMap;

impl<Scalar: Float> DifferentiableFunction<0, 0, Scalar> for LnMap {
    #[inline]
    fn into_params(self) -> ([Scalar; 0], [Scalar; 0]) {
        ([], [])
    }

    #[inline]
    fn value([]: &[Scalar; 0], x: &Scalar) -> Scalar {
        x.ln()
    }

    #[inline]
    fn derivative([]: &[Scalar; 0], &x: &Scalar) -> Scalar {
        Scalar::one() / x
    }
}

/// Function exponentiating model output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExpMap;

impl<Scalar: Float> DifferentiableFunction<0, 0, Scalar> for ExpMap {
    #[inline]
    fn into_params(self) -> ([Scalar; 0], [Scalar; 0]) {
        ([], [])
    }

    #[inline]
    fn value([]: &[Scalar; 0], x: &Scalar) -> Scalar {
        x.exp()
    }

    #[inline]
    fn derivative([]: &[Scalar; 0], x: &Scalar) -> Scalar {
        x.exp()
    }
}

/// Model equal to application of a function on top of a model.
///
/// You **can't** construct this struct manually, please use [`model_map`] for that. This is motivated by it's fields containing two separate objects (`value` and `derivative`), that are actually dependent and should be derived from each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModelMap<Inner, Value, Derivative> {
    /// Inner model.
    pub inner: Inner,
    /// Closure defining actual map.
    value: Value,
    /// Closure defining derivative of the map over it's argument.
    derivative: Derivative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[doc(hidden)]
pub struct ParamFunction<const PARAMS: usize, Scalar, Map> {
    params: [Scalar; PARAMS],
    _ph: PhantomData<Map>,
}

/// The only way to construct [`ModelMap`].
///
/// Second argument *simultaneously* defines function and it's derivative used to map the model.
///
/// Output type might seem complicated, but don't worry about that - the whole point is to keep it non-opaque
pub fn model_map<
    Scalar: 'static,
    Inner: FitModel<Scalar>,
    const VALUE_PARAMS: usize,
    const DERIVATIVE_PARAMS: usize,
    Map: DifferentiableFunction<VALUE_PARAMS, DERIVATIVE_PARAMS, Scalar>,
>(
    inner: Inner,
    map: Map,
) -> ModelMap<
    Inner,
    ParamFunction<VALUE_PARAMS, Scalar, Map>,
    ParamFunction<DERIVATIVE_PARAMS, Scalar, Map>,
> {
    let (value_params, derivative_params) = map.into_params();
    ModelMap {
        inner,
        value: ParamFunction {
            params: value_params,
            _ph: PhantomData,
        },
        derivative: ParamFunction {
            params: derivative_params,
            _ph: PhantomData,
        },
    }
}

impl<
        Scalar: 'static,
        Inner,
        const VALUE_PARAMS: usize,
        const DERIVATIVE_PARAMS: usize,
        Map: DifferentiableFunction<VALUE_PARAMS, DERIVATIVE_PARAMS, Scalar>,
    > FitModel<Scalar>
    for ModelMap<
        Inner,
        ParamFunction<VALUE_PARAMS, Scalar, Map>,
        ParamFunction<DERIVATIVE_PARAMS, Scalar, Map>,
    >
where
    Scalar: Clone + core::ops::Mul<Scalar, Output = Scalar>,
    Inner: FitModel<Scalar>,
{
    type ParamCount = Inner::ParamCount;

    #[inline]
    fn evaluate(&self, x: &Scalar) -> Scalar {
        Map::value(&self.value.params, &self.inner.evaluate(x))
    }

    #[inline]
    fn jacobian(&self, x: &Scalar) -> impl Into<GenericArray<Scalar, Self::ParamCount>> {
        let inner_eval = self.inner.evaluate(x);
        let inner_jacobian = self.inner.jacobian(x).into();

        let map_derivative = Map::derivative(&self.derivative.params, &inner_eval);
        inner_jacobian.map(|d| map_derivative.clone() * d)
    }

    #[inline]
    fn set_params(&mut self, new_params: GenericArray<Scalar, Self::ParamCount>) {
        self.inner.set_params(new_params);
    }

    #[inline]
    fn get_params(&self) -> impl Into<GenericArray<Scalar, Self::ParamCount>> {
        self.inner.get_params()
    }
}

impl<Scalar, Inner, Value, Derivative> FitModelXDeriv<Scalar> for ModelMap<Inner, Value, Derivative>
where
    Scalar: Mul<Output = Scalar>,
    Inner: FitModel<Scalar> + FitModelXDeriv<Scalar>,
    Derivative: Fn(&Scalar) -> Scalar,
{
    #[inline]
    fn deriv_x(&self, x: &Scalar) -> Scalar {
        let y = self.inner.evaluate(x);
        let y_x = self.inner.deriv_x(x);
        let z_y = (self.derivative)(&y);
        z_y * y_x
    }
}
