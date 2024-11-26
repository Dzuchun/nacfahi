#![allow(unused)]

use super::AsymmetricGenericGaussian;
use crate::for_all_bool;

for_all_bool! {
    for_all,
    [FIT_SIGMA, FIT_S_P],
    crate::test_model_derivative!(
        AsymmetricGenericGaussian::<f64, FIT_SIGMA, FIT_S_P>,
        AsymmetricGenericGaussian {
            a: 15.0,
            x_c: 3.5,
            sigma: 1.5,
            s_p: 0.5,
        },
        [
            (0.0, -1.0),
            (1.5, -4.0),
            (2.0, -5.0),
            (3.0, 6.0),
            (4.0, 2.0),
            (5.0, 2.5)
        ]
    );
}
