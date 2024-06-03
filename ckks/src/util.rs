use core::{
    f64::consts::SQRT_2,
    iter::{successors, Sum},
    ops::Mul,
};
use itertools::izip;
use num_bigint::{BigInt, BigUint, ToBigInt};
use num_integer::Integer;
use num_traits::One;
use rand_distr::{Distribution, Standard, WeightedIndex};

pub mod float;
pub mod poly;
pub mod prime;

pub fn rem_center(value: &BigUint, q: &BigUint) -> BigInt {
    let value = value % q;
    if value < q >> 1usize {
        value.to_bigint().unwrap()
    } else {
        value.to_bigint().unwrap() - q.to_bigint().unwrap()
    }
}

pub fn mod_inv(v: u64, q: u64) -> u64 {
    (v as i64).extended_gcd(&(q as i64)).x.rem_euclid(q as i64) as u64
}

pub fn powers<T: One>(base: &T) -> impl Iterator<Item = T> + '_
where
    for<'t> &'t T: Mul<&'t T, Output = T>,
{
    successors(Some(T::one()), move |pow| Some(pow * base))
}

pub fn horner<T: One + Sum>(coeffs: &[T], x: &T) -> T
where
    for<'t> &'t T: Mul<&'t T, Output = T>,
{
    izip!(coeffs, powers(x))
        .map(|(coeff, pow)| coeff * &pow)
        .sum()
}

pub fn bit_reverse<T>(values: &mut [T]) {
    if values.len() > 2 {
        assert!(values.len().is_power_of_two());
        let log_len = values.len().ilog2();
        for i in 0..values.len() {
            let j = i.reverse_bits() >> (usize::BITS - log_len);
            if i < j {
                values.swap(i, j)
            }
        }
    }
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

pub fn dg(std_dev: f64, n: u64) -> impl Distribution<i8> {
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
