use generic_array::{ArrayLength, GenericArray};
use typenum::{Prod, Quot, ToUInt};

#[doc(hidden)]
type TNum<const N: usize> = <typenum::Const<N> as ToUInt>::Output;

/// Defines object that can fit to a set of data points.
///
/// Generally, you have no reason to implement this trait, as there are model primitives and derive macro for that. Manual implementation is always an option though - I've left some hints, in case you're unfamiliar with the types.
pub trait FitModel<S> {
    /// Type representing number of parameters.
    ///
    /// **Hint**: [`typenum`]`::{U1, U2, ..}` types would most likely work for you.
    ///
    /// [`typenum`]: https://docs.rs/typenum/latest/typenum/
    type ParamCount: ArrayLength;

    /// Computes model value for supplied `x` value and current parameters.
    fn evaluate(&self, x: &S) -> S;

    /// Computes jacobian (array of derivatives) for supplied `x` value and current parameters.
    ///
    /// **Hint**: return type allows you to return core Rust array, as long as it's size is correct.
    fn jacobian(&self, x: &S) -> impl Into<GenericArray<S, Self::ParamCount>>;

    /// Sets model parameters to ones contained in a generic array
    ///
    /// **Hint**: `GenericArray::into_array` is a thing. So if your model has two params, you can extract them as
    /// ```rust
    /// # use generic_array::GenericArray;
    /// let new_params: GenericArray<_, typenum::U2> = /* ... */
    /// # GenericArray::from_array([(), ()]);
    /// let [p1, p2] = new_params.into_array();
    /// ```
    fn set_params(&mut self, new_params: GenericArray<S, Self::ParamCount>);

    /// Returns current values of model params.
    ///
    /// **Hint**: return type allows you to return core Rust array, as long as it's size is correct.
    fn get_params(&self) -> impl Into<GenericArray<S, Self::ParamCount>>;
}

impl<Scalar, Model> FitModel<Scalar> for &'_ mut Model
where
    Model: FitModel<Scalar>,
{
    type ParamCount = Model::ParamCount;

    #[inline]
    fn evaluate(&self, x: &Scalar) -> Scalar {
        let s: &Model = self;
        Model::evaluate(s, x)
    }

    #[inline]
    fn jacobian(&self, x: &Scalar) -> impl Into<GenericArray<Scalar, Self::ParamCount>> {
        let s: &Model = self;
        Model::jacobian(s, x)
    }

    #[inline]
    fn set_params(&mut self, new_params: GenericArray<Scalar, Self::ParamCount>) {
        let s: &mut Model = self;
        Model::set_params(s, new_params);
    }

    #[inline]
    fn get_params(&self) -> impl Into<GenericArray<Scalar, Self::ParamCount>> {
        let s: &Model = self;
        Model::get_params(s)
    }
}

#[inline]
#[doc(hidden)]
fn flatten<T, R, C>(arr: GenericArray<GenericArray<T, R>, C>) -> GenericArray<T, Prod<R, C>>
where
    R: ArrayLength + core::ops::Mul<C>,
    C: ArrayLength,
    Prod<R, C>: ArrayLength,
{
    #[allow(unsafe_code)]
    unsafe {
        generic_array::const_transmute(arr)
    }
}

#[inline]
#[doc(hidden)]
fn unflatten<T, P, R>(arr: GenericArray<T, P>) -> GenericArray<GenericArray<T, R>, Quot<P, R>>
where
    R: ArrayLength,
    P: ArrayLength + core::ops::Div<R>,
    Quot<P, R>: ArrayLength,
{
    #[allow(unsafe_code)]
    unsafe {
        generic_array::const_transmute(arr)
    }
}
