use core::marker::PhantomData;

use generic_array::{functional::FunctionalSequence, GenericArray};
use generic_array_storage::{Conv, GenericArrayStorage, GenericMatrix};
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{allocator::Allocator, ComplexField, Dyn, OMatrix};

use crate::models::FitModel;

pub(crate) struct DynOptimizationProblem<'data, Scalar, Params: Conv, Entity, Weights> {
    pub entity: Entity,
    pub x: nalgebra::VectorView<'data, Scalar, Dyn>,
    pub y: nalgebra::VectorView<'data, Scalar, Dyn>,
    pub weights: Weights,
    pub param_count: PhantomData<fn(Params) -> Params>,
}

impl<Params: Conv, Scalar, Entity, Weights> LeastSquaresProblem<Scalar, Dyn, Params::Nalg>
    for DynOptimizationProblem<'_, Scalar, Params, Entity, Weights>
where
    nalgebra::DefaultAllocator:
        Allocator<Params::Nalg, nalgebra::U1> + Allocator<Dyn> + Allocator<Dyn, Params::Nalg>,

    Scalar: nalgebra::Scalar + ComplexField + Copy,
    Entity: FitModel<Scalar, ParamCount = Params::ArrLen>,
    Weights: Fn(Scalar, Scalar) -> Scalar,
{
    type ResidualStorage = <nalgebra::DefaultAllocator as nalgebra::allocator::Allocator<
        Dyn,
        nalgebra::U1,
    >>::Buffer<Scalar>;

    type ParameterStorage = GenericArrayStorage<Scalar, Params, typenum::U1>;

    type JacobianStorage = <nalgebra::DefaultAllocator as nalgebra::allocator::Allocator<
        Dyn,
        Params::Nalg,
    >>::Buffer<Scalar>;

    fn set_params(&mut self, x: &GenericMatrix<Scalar, Params, typenum::U1>) {
        let slice: &[Scalar] = x.data.as_ref();
        let arr = GenericArray::<Scalar, Params::ArrLen>::from_slice(slice).clone();
        self.entity.set_params(arr);
    }

    fn params(&self) -> GenericMatrix<Scalar, Params, typenum::U1> {
        let pars: GenericArray<Scalar, Params::ArrLen> = self.entity.get_params().into();
        GenericMatrix::from_data(GenericArrayStorage(GenericArray::from_array([pars])))
    }

    fn residuals(&self) -> Option<OMatrix<Scalar, Dyn, nalgebra::U1>> {
        let mat: OMatrix<Scalar, Dyn, nalgebra::U1> = self.x.zip_map(&self.y, |x, y| {
            (self.weights)(x, y) * (self.entity.evaluate(&x) - y)
        });
        Some(mat)
    }

    fn jacobian(&self) -> Option<OMatrix<Scalar, Dyn, Params::Nalg>> {
        let mut res = OMatrix::<Scalar, Dyn, Params::Nalg>::zeros_generic(
            Dyn(self.x.len()),
            Params::new_nalg(),
        );

        for i_x in 0..self.x.len() {
            let jacobian_x: GenericArray<_, Params::ArrLen> =
                self.entity.jacobian(&self.x[i_x]).into();
            let arr = jacobian_x.map(|v| GenericArray::<_, typenum::U1>::from_array([v]));
            let mat =
                GenericMatrix::<Scalar, nalgebra::U1, Params>::from_data(GenericArrayStorage(arr));
            res.set_row(i_x, &mat.row(0));
        }
        Some(res)
    }
}
