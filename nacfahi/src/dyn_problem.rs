use generic_array::{GenericArray, functional::FunctionalSequence};
use generic_array_storage::{Conv, GenericArrayStorage, GenericMatrix};
use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{ComplexField, DefaultAllocator, Dyn, OMatrix, allocator::Allocator};

use crate::models::FitModel;

pub(crate) struct DynOptimizationProblem<'data, Model: FitModel, Weights> {
    pub model: Model,
    pub x: nalgebra::VectorView<'data, Model::Scalar, Dyn>,
    pub y: nalgebra::VectorView<'data, Model::Scalar, Dyn>,
    pub weights: Weights,
}

impl<Model: FitModel, Weights>
    LeastSquaresProblem<Model::Scalar, Dyn, <Model::ParamCount as Conv>::Nalg>
    for DynOptimizationProblem<'_, Model, Weights>
where
    DefaultAllocator: Allocator<<Model::ParamCount as Conv>::Nalg, nalgebra::U1>
        + Allocator<Dyn>
        + Allocator<Dyn, <Model::ParamCount as Conv>::Nalg>,
    Model::Scalar: nalgebra::Scalar + ComplexField + Copy,
    Weights: Fn(Model::Scalar, Model::Scalar) -> Model::Scalar,
{
    type ResidualStorage =
        <DefaultAllocator as Allocator<Dyn, nalgebra::U1>>::Buffer<Model::Scalar>;

    type ParameterStorage = GenericArrayStorage<Model::Scalar, Model::ParamCount, typenum::U1>;

    type JacobianStorage = <DefaultAllocator as Allocator<
        Dyn,
        <Model::ParamCount as Conv>::Nalg,
    >>::Buffer<Model::Scalar>;

    fn set_params(&mut self, x: &GenericMatrix<Model::Scalar, Model::ParamCount, typenum::U1>) {
        let slice: &[Model::Scalar] = x.data.as_ref();
        let arr =
            GenericArray::<Model::Scalar, <Model::ParamCount as Conv>::TNum>::from_slice(slice)
                .clone();
        self.model.set_params(arr);
    }

    fn params(&self) -> GenericMatrix<Model::Scalar, Model::ParamCount, typenum::U1> {
        let pars: GenericArray<Model::Scalar, <Model::ParamCount as Conv>::TNum> =
            self.model.get_params().into();
        GenericMatrix::from_data(GenericArrayStorage(GenericArray::from_array([pars])))
    }

    fn residuals(&self) -> Option<OMatrix<Model::Scalar, Dyn, nalgebra::U1>> {
        let mat: OMatrix<Model::Scalar, Dyn, nalgebra::U1> = self.x.zip_map(&self.y, |x, y| {
            (self.weights)(x, y) * (self.model.evaluate(&x) - y)
        });
        Some(mat)
    }

    fn jacobian(&self) -> Option<OMatrix<Model::Scalar, Dyn, <Model::ParamCount as Conv>::Nalg>> {
        let mut res =
            OMatrix::<Model::Scalar, Dyn, <Model::ParamCount as Conv>::Nalg>::zeros_generic(
                Dyn(self.x.len()),
                Model::ParamCount::new_nalg(),
            );

        for i_x in 0..self.x.len() {
            let jacobian_x: GenericArray<_, <Model::ParamCount as Conv>::TNum> =
                self.model.jacobian(&self.x[i_x]).into();
            let arr = jacobian_x.map(|v| GenericArray::<_, typenum::U1>::from_array([v]));
            let mat = GenericMatrix::<Model::Scalar, nalgebra::U1, Model::ParamCount>::from_data(
                GenericArrayStorage(arr),
            );
            res.set_row(i_x, &mat.row(0));
        }
        Some(res)
    }
}
