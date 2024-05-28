use itertools::izip;
use num_bigint::{BigInt, BigUint, ToBigInt};
use num_bigint_dig::prime::probably_prime;
use num_traits::{One, Signed};
use std::{
    iter::{self, Sum},
    ops::{Add, Mul, Rem, Sub},
};

pub fn rem_euclid<L: Signed, R>(value: &L, q: &R) -> L
where
    for<'t> &'t L: Add<&'t R, Output = L> + Sub<&'t R, Output = L> + Rem<&'t R, Output = L>,
{
    let value = value % q;
    if value.is_negative() {
        &value + q
    } else {
        value
    }
}

pub fn rem_center(value: &BigUint, q: &BigUint) -> BigInt {
    let value = value % q;
    if value < q >> 1usize {
        value.to_bigint().unwrap()
    } else {
        -(q - value).to_bigint().unwrap()
    }
}

pub fn powers<T: One>(base: &T) -> impl Iterator<Item = T> + '_
where
    for<'t> &'t T: Mul<&'t T, Output = T>,
{
    iter::successors(Some(T::one()), move |pow| Some(pow * base))
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

pub fn primes(candidates: impl IntoIterator<Item = u64>) -> impl Iterator<Item = u64> {
    candidates
        .into_iter()
        .filter(|candidate| probably_prime(&(*candidate).into(), 20))
}
