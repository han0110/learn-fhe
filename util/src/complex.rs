use crate::complex::f256::F256;
use num_complex::Complex;

pub mod f256;

pub type C256 = Complex<F256>;
pub type C64 = Complex<f64>;
