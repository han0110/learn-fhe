use rand_distr::{Distribution, Standard, WeightedIndex};
use std::f64::consts::SQRT_2;

mod avec;
mod fq;
mod poly;

pub use avec::AVec;
pub use fq::Fq;

pub trait Dot<Rhs = Self> {
    type Output;

    fn dot(&self, rhs: Rhs) -> Self::Output;
}

pub fn zo(rho: f64) -> impl Distribution<i8> {
    assert!(rho <= 1.0);
    Standard.map(move |v: f64| {
        if v <= rho / 2.0 {
            -1
        } else if v <= rho {
            1
        } else {
            0
        }
    })
}

pub fn dg(std_dev: f64, n: usize) -> impl Distribution<i8> {
    // Formula 7.1.26 from Handbook of Mathematical Functions.
    let erf = |x: f64| {
        let p = 0.3275911;
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let t = 1.0 / (1.0 + p * x.abs());
        let positive_erf =
            1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
        if x.is_sign_positive() {
            positive_erf
        } else {
            -positive_erf
        }
    };
    let cdf = |x| (1.0 + erf(x / (std_dev * SQRT_2))) / 2.0;
    let max = (n as f64 * std_dev).floor() as i8;
    let weights = (-max..=max).map(|i| cdf(i as f64 + 0.5) - cdf(i as f64 - 0.5));
    WeightedIndex::new(weights)
        .unwrap()
        .map(move |v| v as i8 - max)
}
