use itertools::{izip, Itertools};
use num_bigint::{BigInt, BigUint, Sign, ToBigInt};
use num_bigint_dig::prime::probably_prime;
use num_integer::Integer;
use num_traits::{One, ToPrimitive};
use rand::Rng;
use rand_distr::{Distribution, Standard, WeightedError, WeightedIndex};
use std::{
    f64::consts::SQRT_2,
    iter::{self, Sum},
    ops::{Deref, Mul},
};

pub fn rem_center(value: &BigUint, q: &BigUint) -> BigInt {
    let value = value % q;
    if value < q >> 1usize {
        value.to_bigint().unwrap()
    } else {
        BigInt::from_biguint(Sign::Minus, q - value)
    }
}

pub fn mod_inv(v: u64, q: u64) -> u64 {
    (v as i64).extended_gcd(&(q as i64)).x.rem_euclid(q as i64) as u64
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

pub fn two_adic_primes(bits: usize, log_n: usize) -> impl Iterator<Item = u64> {
    let (min, max) = (1 << (bits - 1), 1 << bits);
    primes((min..max).rev().map(move |q| (q << log_n) + 1))
}

#[derive(Clone, Debug)]
pub struct SmallPrime {
    q: u64,
    s: u32,
    n_invs: Vec<u64>,
    psi: Vec<u64>,
    psi_inv: Vec<u64>,
}

impl Deref for SmallPrime {
    type Target = u64;

    fn deref(&self) -> &Self::Target {
        &self.q
    }
}

impl PartialEq for SmallPrime {
    fn eq(&self, other: &Self) -> bool {
        self.q == other.q
    }
}

impl Eq for SmallPrime {}

impl SmallPrime {
    pub fn new(q: u64) -> Self {
        assert!(probably_prime(&q.into(), 20));
        assert!(q.leading_zeros() > 1);

        let order = q - 1;
        let s = order.trailing_zeros();
        let mut q = Self {
            q,
            s,
            n_invs: Vec::new(),
            psi: Vec::new(),
            psi_inv: Vec::new(),
        };

        if s > 0 {
            q.n_invs = (0..s as u64).map(|log_n| q.inv(1 << log_n)).collect();
            q.psi = {
                let g = (1..order).find(|g| q.pow(*g, order >> 1) == order).unwrap();
                let rou = q.pow(g, order >> s);
                let mut psi = q.powers(rou).take(1 << (s - 1)).collect_vec();
                bit_reverse(&mut psi);
                psi
            };
            q.psi_inv = q.psi.iter().map(|psi| q.inv(*psi)).collect();
        }

        q
    }

    pub fn neg_ntt(&self, mut a: Vec<u64>) -> Vec<u64> {
        self.neg_ntt_in_place(&mut a);
        a
    }

    // Algorithm 1 in 2016/504.
    pub fn neg_ntt_in_place(&self, a: &mut [u64]) {
        assert!(a.len().is_power_of_two());
        assert!(a.len().ilog2() < self.s);

        for log_m in 0..a.len().ilog2() {
            let m = 1 << log_m;
            let t = a.len() / m;
            izip!(0.., a.chunks_exact_mut(t), &self.psi[m..]).for_each(|(i, a, psi)| {
                let (u, v) = a.split_at_mut(t / 2);
                if m == 0 && i == 0 {
                    izip!(u, v).for_each(|(u, v)| self.twiddle_free_bufferfly(u, v));
                } else {
                    izip!(u, v).for_each(|(u, v)| self.dit_bufferfly(u, v, psi));
                }
            });
        }
    }

    // Algorithm 2 in 2016/504.
    pub fn neg_intt(&self, mut a: Vec<u64>) -> Vec<u64> {
        self.neg_intt_in_place(&mut a);
        a
    }

    pub fn neg_intt_in_place(&self, a: &mut [u64]) {
        assert!(a.len().is_power_of_two());
        assert!(a.len().ilog2() < self.s);

        for log_m in (0..a.len().ilog2()).rev() {
            let m = 1 << log_m;
            let t = a.len() / m;
            izip!(0.., a.chunks_exact_mut(t), &self.psi_inv[m..]).for_each(|(i, a, psi_inv)| {
                let (u, v) = a.split_at_mut(t / 2);
                if m == 0 && i == 0 {
                    izip!(u, v).for_each(|(u, v)| self.twiddle_free_bufferfly(u, v));
                } else {
                    izip!(u, v).for_each(|(u, v)| self.dif_bufferfly(u, v, psi_inv));
                }
            });
        }

        let n_inv = self.n_invs[a.len().ilog2() as usize];
        a.iter_mut().for_each(|a| *a = self.mul(*a, n_inv));
    }

    #[inline(always)]
    pub fn dit_bufferfly(&self, a: &mut u64, b: &mut u64, twiddle: &u64) {
        let tb = self.mul(*b, *twiddle);
        let c = self.add(*a, tb);
        let d = self.sub(*a, tb);
        *a = c;
        *b = d;
    }

    #[inline(always)]
    pub fn dif_bufferfly(&self, a: &mut u64, b: &mut u64, twiddle: &u64) {
        let c = self.add(*a, *b);
        let d = self.mul(self.sub(*a, *b), *twiddle);
        *a = c;
        *b = d;
    }

    #[inline(always)]
    pub fn twiddle_free_bufferfly(&self, a: &mut u64, b: &mut u64) {
        let c = self.add(*a, *b);
        let d = self.sub(*a, *b);
        *a = c;
        *b = d;
    }

    #[inline(always)]
    pub fn neg(&self, a: u64) -> u64 {
        self.q - a
    }

    #[inline(always)]
    pub fn add(&self, a: u64, b: u64) -> u64 {
        let c = a + b;
        if c < self.q {
            c
        } else {
            c - self.q
        }
    }

    #[inline(always)]
    pub fn sub(&self, a: u64, b: u64) -> u64 {
        let c = a.wrapping_sub(b);
        if c < self.q {
            c
        } else {
            c.wrapping_add(self.q)
        }
    }

    #[inline(always)]
    pub fn mul(&self, a: u64, b: u64) -> u64 {
        ((a as u128 * b as u128) % self.q as u128) as u64
    }

    pub fn pow(&self, base: u64, exp: u64) -> u64 {
        BigUint::from(base)
            .modpow(&exp.into(), &self.q.into())
            .to_u64()
            .unwrap()
    }

    pub fn powers(&self, base: u64) -> impl Iterator<Item = u64> + '_ {
        iter::successors(Some(1), move |pow| Some(self.mul(*pow, base)))
    }

    pub fn inv(&self, a: u64) -> u64 {
        mod_inv(a, self.q)
    }

    pub fn from_bigint(&self, a: &BigInt) -> u64 {
        let a = (a % self.q as i64).to_i64().unwrap();
        let a = if a < 0 { a + self.q as i64 } else { a };
        a as u64
    }

    pub fn from_i8(&self, a: &i8) -> u64 {
        let a = *a as i64 % self.q as i64;
        let a = if a < 0 { a + self.q as i64 } else { a };
        a as u64
    }
}

#[derive(Clone, Debug)]
pub struct DiscreteNormal {
    sampler: WeightedIndex<f64>,
    max: i8,
}

impl DiscreteNormal {
    pub fn new(std_dev: f64, n: u64) -> Result<Self, WeightedError> {
        let max = (n as f64 * std_dev).floor() as i8;
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
        let cdf = |x| (1. + erf(x / (std_dev * SQRT_2))) / 2.;
        let sampler =
            WeightedIndex::new((-max..=max).map(|i| cdf(i as f64 + 0.5) - cdf(i as f64 - 0.5)))?;
        Ok(Self { sampler, max })
    }
}

impl Distribution<i8> for DiscreteNormal {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> i8 {
        self.sampler.sample(rng) as i8 - self.max
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
