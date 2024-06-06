use core::{
    borrow::Borrow,
    fmt::{self, Display, Formatter},
    iter::{successors, Product, Sum},
    ops::{AddAssign, MulAssign, Neg, SubAssign},
};
use itertools::Itertools;
use num_bigint::{BigUint, ToBigUint};
use num_bigint_dig::prime::probably_prime;
use num_integer::Integer;
use num_traits::ToPrimitive;
use rand::RngCore;
use rand_distr::{Distribution, Uniform};
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fq {
    q: u64,
    v: u64,
}

impl Fq {
    pub fn q(&self) -> u64 {
        self.q
    }

    pub fn from_u64(q: u64, v: u64) -> Self {
        let v = v % q;
        Self { q, v }
    }

    pub fn from_u128(q: u64, v: u128) -> Self {
        let v = (v % q as u128) as u64;
        Self { q, v }
    }

    pub fn from_i8(q: u64, v: i8) -> Self {
        Self::from_i64(q, v as i64)
    }

    pub fn from_i64(q: u64, v: i64) -> Self {
        let v = v.rem_euclid(q as i64) as u64;
        Self { q, v }
    }

    pub fn from_f64(q: u64, v: f64) -> Self {
        Self::from_i64(q, v.round() as i64)
    }

    pub fn sample_uniform(q: u64, rng: &mut impl RngCore) -> Self {
        Fq::from_u64(q, Uniform::new(0, q).sample(rng))
    }

    pub fn sample_i8(q: u64, dist: &impl Distribution<i8>, rng: &mut impl RngCore) -> Self {
        Fq::from_i8(q, dist.sample(rng))
    }

    pub fn generator(q: u64) -> Self {
        let order = q - 1;
        (1..order)
            .map(|g| Fq::from_u64(q, g))
            .find(|g| g.pow(order >> 1).v == order)
            .unwrap()
    }

    pub fn pow(mut self, exp: impl ToBigUint) -> Self {
        self.v = BigUint::from(self.v)
            .modpow(&exp.to_biguint().unwrap(), &self.q.into())
            .to_u64()
            .unwrap();
        self
    }

    pub fn powers(self) -> impl Iterator<Item = Self> {
        successors(Some(Fq::from_i8(self.q, 1)), move |v| Some(v * self))
    }

    pub fn inv(self) -> Option<Self> {
        (self.v != 0)
            .then(move || Self::from_i64(self.q, (self.v as i64).extended_gcd(&(self.q as i64)).x))
    }
}

impl From<Fq> for u64 {
    fn from(value: Fq) -> Self {
        value.v
    }
}

impl From<&Fq> for u64 {
    fn from(value: &Fq) -> Self {
        value.v
    }
}

impl From<Fq> for f64 {
    fn from(value: Fq) -> Self {
        value.v as f64
    }
}

impl From<&Fq> for f64 {
    fn from(value: &Fq) -> Self {
        value.v as f64
    }
}

impl Display for Fq {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.v)
    }
}

impl Neg for &Fq {
    type Output = Fq;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Fq::from_u64(self.q, self.q - self.v)
    }
}

impl Neg for Fq {
    type Output = Fq;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        -&self
    }
}

impl AddAssign<&u64> for Fq {
    #[inline(always)]
    fn add_assign(&mut self, rhs: &u64) {
        assert!(*rhs < self.q);
        *self += Self::from_u64(self.q, *rhs);
    }
}

impl AddAssign<&Fq> for Fq {
    #[inline(always)]
    fn add_assign(&mut self, rhs: &Fq) {
        assert_eq!(self.q, rhs.q);
        *self = Self::from_u128(self.q, self.v as u128 + rhs.v as u128);
    }
}

impl SubAssign<&Fq> for Fq {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &Fq) {
        assert_eq!(self.q, rhs.q);
        *self += -rhs;
    }
}

impl SubAssign<&u64> for Fq {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &u64) {
        assert!(*rhs < self.q);
        *self -= Self::from_u64(self.q, *rhs);
    }
}

impl MulAssign<&Fq> for Fq {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &Fq) {
        assert_eq!(self.q, rhs.q);
        *self = Self::from_u128(self.q, self.v as u128 * rhs.v as u128);
    }
}

impl MulAssign<&u64> for Fq {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &u64) {
        assert!(*rhs < self.q);
        *self *= Self::from_u64(self.q, *rhs);
    }
}

impl<T: Borrow<Fq>> Sum<T> for Fq {
    fn sum<I: Iterator<Item = T>>(mut iter: I) -> Self {
        let init = *iter.next().unwrap().borrow();
        iter.fold(init, |acc, item| acc + item.borrow())
    }
}

impl<T: Borrow<Fq>> Product<T> for Fq {
    fn product<I: Iterator<Item = T>>(mut iter: I) -> Self {
        let init = *iter.next().unwrap().borrow();
        iter.fold(init, |acc, item| acc * item.borrow())
    }
}

macro_rules! impl_ops {
    (@ impl $trait:ident<$rhs:ty> for $lhs:ty; $lhs_convert:expr) => {
        paste::paste! {
            impl core::ops::$trait<$rhs> for $lhs {
                type Output = Fq;

                #[inline(always)]
                fn [<$trait:lower>](self, rhs: $rhs) -> Fq {
                    let mut lhs = $lhs_convert(self);
                    lhs.[<$trait:lower _assign>](rhs.borrow());
                    lhs
                }
            }
        }
    };
    ($(impl $trait:ident<$rhs:ty> for $lhs:ty),* $(,)?) => {
        $(
            paste::paste! {
                impl core::ops::[<$trait Assign>]<$rhs> for $lhs {
                    #[inline(always)]
                    fn [<$trait:lower _assign>](&mut self, rhs: $rhs) {
                        self.[<$trait:lower _assign>](&rhs);
                    }
                }
            }
            impl_ops!(@ impl $trait<$rhs> for $lhs; core::convert::identity);
            impl_ops!(@ impl $trait<&$rhs> for $lhs; core::convert::identity);
            impl_ops!(@ impl $trait<$rhs> for &$lhs; <_>::clone);
            impl_ops!(@ impl $trait<&$rhs> for &$lhs; <_>::clone);
        )*
    };
}

impl_ops!(
    impl Add<Fq> for Fq,
    impl Sub<Fq> for Fq,
    impl Mul<Fq> for Fq,
    impl Add<u64> for Fq,
    impl Sub<u64> for Fq,
    impl Mul<u64> for Fq,
);

pub(crate) static NEG_NTT_PSI: OnceLock<Mutex<HashMap<u64, [Vec<Fq>; 2]>>> = OnceLock::new();

pub fn two_adic_primes(bits: usize, log_n: usize) -> impl Iterator<Item = u64> {
    assert!(bits > log_n);

    let (min, max) = (1 << (bits - log_n - 1), 1 << (bits - log_n));
    primes((min..max).rev().map(move |q| (q << log_n) + 1)).map(|q| {
        NEG_NTT_PSI
            .get_or_init(Default::default)
            .lock()
            .unwrap()
            .entry(q)
            .or_insert_with(|| neg_ntt_psi(q));
        q
    })
}

fn primes(candidates: impl IntoIterator<Item = u64>) -> impl Iterator<Item = u64> {
    candidates
        .into_iter()
        .filter(|candidate| probably_prime(&(*candidate).into(), 20))
}

fn neg_ntt_psi(q: u64) -> [Vec<Fq>; 2] {
    let order = q - 1;
    let s = order.trailing_zeros();
    let psi = {
        let g = Fq::generator(q);
        let rou = g.pow(order >> s);
        let mut psi = rou.powers().take(1 << (s - 1)).collect_vec();
        bit_reverse(&mut psi);
        psi
    };
    let psi_inv = psi.iter().map(|v| v.inv().unwrap()).collect();
    [psi, psi_inv]
}

fn bit_reverse<T>(values: &mut [T]) {
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
