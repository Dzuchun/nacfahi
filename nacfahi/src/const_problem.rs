use generic_array::{functional::FunctionalSequence, GenericArray};
use generic_array_storage::{Conv, GenericArrayStorage, GenericMatrix, GenericMatrixFromExt};
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{allocator::Allocator, ComplexField, DefaultAllocator, OMatrix};

use crate::models::FitModel;

pub(crate) struct ConstOptimizationProblem<'data, Points: Conv, Model: FitModel, Weights> {
    pub entity: Model,
    pub x: nalgebra::VectorView<'data, Model::Scalar, Points::Nalg>,
    pub y: nalgebra::VectorView<'data, Model::Scalar, Points::Nalg>,
    pub weights: Weights,
}

impl<Points: Conv, Model: FitModel, Weights>
    LeastSquaresProblem<Model::Scalar, Points::Nalg, <Model::ParamCount as Conv>::Nalg>
    for ConstOptimizationProblem<'_, Points, Model, Weights>
where
    DefaultAllocator: Allocator<Points::Nalg, nalgebra::U1>,
    DefaultAllocator: Allocator<<Model::ParamCount as Conv>::Nalg, nalgebra::U1>,
    DefaultAllocator: Allocator<Points::Nalg, <Model::ParamCount as Conv>::Nalg>,

    Model::Scalar: nalgebra::Scalar + ComplexField + Copy,
    Model: FitModel,
    Weights: Fn(Model::Scalar, Model::Scalar) -> Model::Scalar,
{
    type ResidualStorage = GenericArrayStorage<Model::Scalar, Points, typenum::U1>;

    type ParameterStorage = GenericArrayStorage<Model::Scalar, Model::ParamCount, typenum::U1>;

    type JacobianStorage = GenericArrayStorage<Model::Scalar, Points, Model::ParamCount>;

    fn set_params(&mut self, x: &GenericMatrix<Model::Scalar, Model::ParamCount, typenum::U1>) {
        let slice: &[Model::Scalar] = x.data.as_ref();
        let arr =
            GenericArray::<Model::Scalar, <Model::ParamCount as Conv>::TNum>::from_slice(slice)
                .clone();
        self.entity.set_params(arr);
    }

    fn params(&self) -> GenericMatrix<Model::Scalar, Model::ParamCount, typenum::U1> {
        let pars: GenericArray<Model::Scalar, <Model::ParamCount as Conv>::TNum> =
            self.entity.get_params().into();
        GenericMatrix::from_data(GenericArrayStorage(GenericArray::from_array([pars])))
    }

    fn residuals(&self) -> Option<GenericMatrix<Model::Scalar, Points, typenum::U1>> {
        let mat: GenericMatrix<Model::Scalar, Points, typenum::U1> = self
            .x
            .zip_map(&self.y, |x, y| {
                (self.weights)(x, y) * (self.entity.evaluate(&x) - y)
            })
            .into_generic_matrix();
        Some(mat)
    }

    fn jacobian(&self) -> Option<GenericMatrix<Model::Scalar, Points, Model::ParamCount>> {
        let mut res =
            OMatrix::<Model::Scalar, Points::Nalg, <Model::ParamCount as Conv>::Nalg>::zeros_generic(
                Points::new_nalg(),
                Model::ParamCount::new_nalg(),
            );

        for i_x in 0..self.x.len() {
            let jacobian_x: GenericArray<_, <Model::ParamCount as Conv>::TNum> =
                self.entity.jacobian(&self.x[i_x]).into();
            let arr = jacobian_x.map(|v| GenericArray::<_, typenum::U1>::from_array([v]));
            let mat = GenericMatrix::<Model::Scalar, nalgebra::U1, Model::ParamCount>::from_data(
                GenericArrayStorage(arr),
            );
            res.set_row(i_x, &mat.row(0));
        }
        Some(res.into_generic_matrix())
    }
}
