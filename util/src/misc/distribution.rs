use crate::torus::T64;
use core::f64::consts::SQRT_2;
use rand::distributions::{Distribution, Standard, WeightedIndex};
use rand_distr::Normal;

pub fn binary() -> impl Distribution<i64> {
    Standard.map(move |v: f64| if v <= 0.5 { 0 } else { 1 })
}

pub fn zo(rho: f64) -> impl Distribution<i64> {
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

pub fn dg(std_dev: f64, n: usize) -> impl Distribution<i64> {
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
    let max = (n as f64 * std_dev).floor() as i64;
    let weights = (-max..=max).map(|i| cdf(i as f64 + 0.5) - cdf(i as f64 - 0.5));
    WeightedIndex::new(weights)
        .unwrap()
        .map(move |v| v as i64 - max)
}

pub fn tdg(std_dev: f64) -> impl Distribution<T64> {
    Normal::new(0., std_dev).unwrap().map(T64::from)
}
