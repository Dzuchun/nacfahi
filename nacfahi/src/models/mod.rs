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
