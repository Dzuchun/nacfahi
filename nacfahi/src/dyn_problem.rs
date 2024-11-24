use generic_array::{functional::FunctionalSequence, GenericArray};
use generic_array_storage::{Conv, GenericArrayStorage, GenericMatrix};
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{allocator::Allocator, ComplexField, DefaultAllocator, Dyn, OMatrix};

use crate::models::FitModel;

pub(crate) struct DynOptimizationProblem<'data, Entity: FitModel, Weights> {
    pub entity: Entity,
    pub x: nalgebra::VectorView<'data, Entity::Scalar, Dyn>,
    pub y: nalgebra::VectorView<'data, Entity::Scalar, Dyn>,
    pub weights: Weights,
}

impl<Entity: FitModel, Weights>
    LeastSquaresProblem<Entity::Scalar, Dyn, <Entity::ParamCount as Conv>::Nalg>
    for DynOptimizationProblem<'_, Entity, Weights>
where
    DefaultAllocator: Allocator<<Entity::ParamCount as Conv>::Nalg, nalgebra::U1>
        + Allocator<Dyn>
        + Allocator<Dyn, <Entity::ParamCount as Conv>::Nalg>,

    Entity::Scalar: nalgebra::Scalar + ComplexField + Copy,
    Weights: Fn(Entity::Scalar, Entity::Scalar) -> Entity::Scalar,
{
    type ResidualStorage =
        <DefaultAllocator as Allocator<Dyn, nalgebra::U1>>::Buffer<Entity::Scalar>;

    type ParameterStorage = GenericArrayStorage<Entity::Scalar, Entity::ParamCount, typenum::U1>;

    type JacobianStorage = <DefaultAllocator as Allocator<
        Dyn,
        <Entity::ParamCount as Conv>::Nalg,
    >>::Buffer<Entity::Scalar>;

    fn set_params(&mut self, x: &GenericMatrix<Entity::Scalar, Entity::ParamCount, typenum::U1>) {
        let slice: &[Entity::Scalar] = x.data.as_ref();
        let arr =
            GenericArray::<Entity::Scalar, <Entity::ParamCount as Conv>::TNum>::from_slice(slice)
                .clone();
        self.entity.set_params(arr);
    }

    fn params(&self) -> GenericMatrix<Entity::Scalar, Entity::ParamCount, typenum::U1> {
        let pars: GenericArray<Entity::Scalar, <Entity::ParamCount as Conv>::TNum> =
            self.entity.get_params().into();
        GenericMatrix::from_data(GenericArrayStorage(GenericArray::from_array([pars])))
    }

    fn residuals(&self) -> Option<OMatrix<Entity::Scalar, Dyn, nalgebra::U1>> {
        let mat: OMatrix<Entity::Scalar, Dyn, nalgebra::U1> = self.x.zip_map(&self.y, |x, y| {
            (self.weights)(x, y) * (self.entity.evaluate(&x) - y)
        });
        Some(mat)
    }

    fn jacobian(&self) -> Option<OMatrix<Entity::Scalar, Dyn, <Entity::ParamCount as Conv>::Nalg>> {
        let mut res =
            OMatrix::<Entity::Scalar, Dyn, <Entity::ParamCount as Conv>::Nalg>::zeros_generic(
                Dyn(self.x.len()),
                Entity::ParamCount::new_nalg(),
            );

        for i_x in 0..self.x.len() {
            let jacobian_x: GenericArray<_, <Entity::ParamCount as Conv>::TNum> =
                self.entity.jacobian(&self.x[i_x]).into();
            let arr = jacobian_x.map(|v| GenericArray::<_, typenum::U1>::from_array([v]));
            let mat = GenericMatrix::<Entity::Scalar, nalgebra::U1, Entity::ParamCount>::from_data(
                GenericArrayStorage(arr),
            );
            res.set_row(i_x, &mat.row(0));
        }
        Some(res)
    }
}
