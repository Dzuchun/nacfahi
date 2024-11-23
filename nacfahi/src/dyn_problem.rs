use generic_array::{functional::FunctionalSequence, GenericArray};
use generic_array_storage::{Conv, GenericArrayStorage, GenericMatrix};
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{allocator::Allocator, ComplexField, DefaultAllocator, Dyn, OMatrix};

use crate::models::FitModel;

pub(crate) struct DynOptimizationProblem<'data, Scalar, Entity, Weights> {
    pub entity: Entity,
    pub x: nalgebra::VectorView<'data, Scalar, Dyn>,
    pub y: nalgebra::VectorView<'data, Scalar, Dyn>,
    pub weights: Weights,
}

impl<Scalar, Entity, Weights> LeastSquaresProblem<Scalar, Dyn, <Entity::ParamCount as Conv>::Nalg>
    for DynOptimizationProblem<'_, Scalar, Entity, Weights>
where
    DefaultAllocator: Allocator<<Entity::ParamCount as Conv>::Nalg, nalgebra::U1>
        + Allocator<Dyn>
        + Allocator<Dyn, <Entity::ParamCount as Conv>::Nalg>,

    Scalar: nalgebra::Scalar + ComplexField + Copy,
    Entity: FitModel<Scalar>,
    Weights: Fn(Scalar, Scalar) -> Scalar,
{
    type ResidualStorage = <DefaultAllocator as Allocator<Dyn, nalgebra::U1>>::Buffer<Scalar>;

    type ParameterStorage = GenericArrayStorage<Scalar, Entity::ParamCount, typenum::U1>;

    type JacobianStorage =
        <DefaultAllocator as Allocator<Dyn, <Entity::ParamCount as Conv>::Nalg>>::Buffer<Scalar>;

    fn set_params(&mut self, x: &GenericMatrix<Scalar, Entity::ParamCount, typenum::U1>) {
        let slice: &[Scalar] = x.data.as_ref();
        let arr =
            GenericArray::<Scalar, <Entity::ParamCount as Conv>::TNum>::from_slice(slice).clone();
        self.entity.set_params(arr);
    }

    fn params(&self) -> GenericMatrix<Scalar, Entity::ParamCount, typenum::U1> {
        let pars: GenericArray<Scalar, <Entity::ParamCount as Conv>::TNum> =
            self.entity.get_params().into();
        GenericMatrix::from_data(GenericArrayStorage(GenericArray::from_array([pars])))
    }

    fn residuals(&self) -> Option<OMatrix<Scalar, Dyn, nalgebra::U1>> {
        let mat: OMatrix<Scalar, Dyn, nalgebra::U1> = self.x.zip_map(&self.y, |x, y| {
            (self.weights)(x, y) * (self.entity.evaluate(&x) - y)
        });
        Some(mat)
    }

    fn jacobian(&self) -> Option<OMatrix<Scalar, Dyn, <Entity::ParamCount as Conv>::Nalg>> {
        let mut res = OMatrix::<Scalar, Dyn, <Entity::ParamCount as Conv>::Nalg>::zeros_generic(
            Dyn(self.x.len()),
            Entity::ParamCount::new_nalg(),
        );

        for i_x in 0..self.x.len() {
            let jacobian_x: GenericArray<_, <Entity::ParamCount as Conv>::TNum> =
                self.entity.jacobian(&self.x[i_x]).into();
            let arr = jacobian_x.map(|v| GenericArray::<_, typenum::U1>::from_array([v]));
            let mat = GenericMatrix::<Scalar, nalgebra::U1, Entity::ParamCount>::from_data(
                GenericArrayStorage(arr),
            );
            res.set_row(i_x, &mat.row(0));
        }
        Some(res)
    }
}
