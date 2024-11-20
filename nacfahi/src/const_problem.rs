use core::marker::PhantomData;

use generic_array::{functional::FunctionalSequence, GenericArray};
use generic_array_storage::{Conv, GenericArrayStorage, GenericMatrix, GenericMatrixFromExt};
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{ComplexField, OMatrix};

use crate::models::FitModel;

pub(crate) struct ConstOptimizationProblem<
    'data,
    Scalar,
    Params: Conv,
    Points: Conv,
    Entity,
    Weights,
> {
    pub entity: Entity,
    pub x: nalgebra::VectorView<'data, Scalar, Points::Nalg>,
    pub y: nalgebra::VectorView<'data, Scalar, Points::Nalg>,
    pub weights: Weights,
    pub param_count: PhantomData<fn(Params) -> Params>,
}

impl<Params: Conv, Points: Conv, Scalar, Entity, Weights>
    LeastSquaresProblem<Scalar, Points::Nalg, Params::Nalg>
    for ConstOptimizationProblem<'_, Scalar, Params, Points, Entity, Weights>
where
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<Params::Nalg, nalgebra::U1>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<Points::Nalg, nalgebra::U1>,
    nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<Points::Nalg, Params::Nalg>,

    Scalar: nalgebra::Scalar + ComplexField + Copy,
    Entity: FitModel<Scalar, ParamCount = Params::ArrLen>,
    Weights: Fn(Scalar, Scalar) -> Scalar,
{
    type ResidualStorage = GenericArrayStorage<Scalar, Points, typenum::U1>;

    type ParameterStorage = GenericArrayStorage<Scalar, Params, typenum::U1>;

    type JacobianStorage = GenericArrayStorage<Scalar, Points, Params>;

    fn set_params(&mut self, x: &GenericMatrix<Scalar, Params, typenum::U1>) {
        let slice: &[Scalar] = x.data.as_ref();
        let arr = GenericArray::<Scalar, Params::ArrLen>::from_slice(slice).clone();
        self.entity.set_params(arr);
    }

    fn params(&self) -> GenericMatrix<Scalar, Params, typenum::U1> {
        let pars: GenericArray<Scalar, Params::ArrLen> = self.entity.get_params().into();
        GenericMatrix::from_data(GenericArrayStorage(GenericArray::from_array([pars])))
    }

    fn residuals(&self) -> Option<GenericMatrix<Scalar, Points, typenum::U1>> {
        let mat: GenericMatrix<Scalar, Points, typenum::U1> = self
            .x
            .zip_map(&self.y, |x, y| {
                (self.weights)(x, y) * (self.entity.evaluate(&x) - y)
            })
            .into_generic_matrix();
        Some(mat)
    }

    fn jacobian(&self) -> Option<GenericMatrix<Scalar, Points, Params>> {
        let mut res = OMatrix::<Scalar, Points::Nalg, Params::Nalg>::zeros_generic(
            Points::new_nalg(),
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
        Some(res.into_generic_matrix())
    }
}
