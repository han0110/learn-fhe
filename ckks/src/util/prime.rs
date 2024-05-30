use crate::util::{bit_reverse, mod_inv};
use core::{iter::successors, ops::Deref};
use itertools::{izip, Itertools};
use num_bigint::{BigInt, BigUint};
use num_bigint_dig::prime::probably_prime;
use num_traits::ToPrimitive;
use rand::Rng;
use rand_distr::{Distribution, Uniform};

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
        successors(Some(1), move |pow| Some(self.mul(*pow, base)))
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

impl Distribution<u64> for SmallPrime {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        Uniform::new(0, self.q).sample(rng)
    }
}
