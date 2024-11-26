#![allow(unused)]

use super::SymmetricGenericGaussian;
use crate::for_all_bool;

for_all_bool! {
    for_all,
    [FIT_SIGMA],
    crate::test_model_derivative!(
        SymmetricGenericGaussian::<f64, FIT_SIGMA>,
        SymmetricGenericGaussian {
            a: -5.0,
            x_c: 1.0,
            sigma: 3.0,
        },
        [
            (0.0, -1.0),
            (1.0, -4.0),
            (2.0, -5.0),
            (3.0, 6.0),
            (4.0, 2.0),
            (5.0, 2.5)
        ]
    );
}
