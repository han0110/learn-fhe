use core::{
    borrow::Borrow,
    cmp::Ordering,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fq {
    q: u64,
    v: u64,
}

impl Fq {
    pub fn q(&self) -> u64 {
        self.q
    }

    pub fn from_u128(q: u64, v: u128) -> Self {
        let v = (v % q as u128) as u64;
        Self { q, v }
    }

    pub fn from_u64(q: u64, v: u64) -> Self {
        let v = v % q;
        Self { q, v }
    }

    pub fn from_i64(q: u64, v: i64) -> Self {
        let v = v.rem_euclid(q as i64) as u64;
        Self { q, v }
    }

    pub fn from_f64(q: u64, v: f64) -> Self {
        Self::from_i64(q, v.round() as i64)
    }

    pub fn from_bool(q: u64, v: bool) -> Self {
        Self::from_u64(q, v as u64)
    }

    pub(crate) fn into_center_signed(self) -> i64 {
        if self.v < self.q >> 1 {
            self.v as i64
        } else {
            self.v as i64 - self.q as i64
        }
    }

    pub(crate) fn into_center_unsigned(self) -> u64 {
        if self.v < self.q >> 1 {
            self.v
        } else {
            !(self.q - self.v) + 1
        }
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

    pub fn mod_switch(self, q_prime: u64) -> Self {
        Self::from_f64(q_prime, (self.v as f64 * q_prime as f64) / self.q as f64)
    }

    pub fn mod_switch_odd(self, q_prime: u64) -> Self {
        let v = (self.v as f64 * q_prime as f64) / self.q as f64;
        let u = v.floor();
        if u == 0.0 {
            Self::from_u64(q_prime, v.round() as u64)
        } else {
            Self::from_u64(q_prime, u as u64 | 1)
        }
    }
}

impl From<&Fq> for u64 {
    fn from(value: &Fq) -> Self {
        value.v
    }
}

impl From<&Fq> for i64 {
    fn from(value: &Fq) -> Self {
        value.into_center_signed()
    }
}

impl From<Fq> for f64 {
    fn from(value: Fq) -> Self {
        value.into_center_signed() as f64
    }
}

impl From<&Fq> for f64 {
    fn from(value: &Fq) -> Self {
        value.into_center_signed() as f64
    }
}

impl Ord for Fq {
    fn cmp(&self, other: &Self) -> Ordering {
        assert_eq!(self.q, other.q);
        self.v.cmp(&other.v)
    }
}

impl PartialOrd for Fq {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
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

impl MulAssign<&Fq> for Fq {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &Fq) {
        assert_eq!(self.q, rhs.q);
        *self = Self::from_u128(self.q, self.v as u128 * rhs.v as u128);
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

macro_rules! impl_op {
    (@ impl $trait:ident<$rhs:ty> for $lhs:ty; $lhs_convert:expr) => {
        paste::paste! {
            impl core::ops::$trait<$rhs> for $lhs {
                type Output = Fq;

                #[inline(always)]
                fn [<$trait:lower>](self, rhs: $rhs) -> Self::Output {
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
            impl_op!(@ impl $trait<$rhs> for $lhs; core::convert::identity);
            impl_op!(@ impl $trait<&$rhs> for $lhs; core::convert::identity);
            impl_op!(@ impl $trait<$rhs> for &$lhs; <_>::clone);
            impl_op!(@ impl $trait<&$rhs> for &$lhs; <_>::clone);
        )*
    };
}

impl_op!(
    impl Add<Fq> for Fq,
    impl Sub<Fq> for Fq,
    impl Mul<Fq> for Fq,
);

macro_rules! impl_op_with_primitive {
    (@ impl $trait:ident<&$p:ty> for Fq) => {
        paste::paste! {
            impl core::ops::$trait<&$p> for Fq {
                #[inline(always)]
                fn [<$trait:snake:lower>](&mut self, rhs: &$p) {
                    self.[<$trait:snake:lower>](Self::[<from_ $p>](self.q, *rhs));
                }
            }
        }
    };
    ($(impl $trait:ident<&$p:ty> for Fq),* $(,)?) => {
        $(impl_op_with_primitive!(@ impl $trait<&$p> for Fq);)*
    };
    ($($p1:ty $(as $p2:ty)?),* $(,)?) => {
        $(
            $(
                paste::paste! {
                    impl Fq {
                        pub fn [<from_ $p1>](q: u64, v: $p1) -> Self {
                            Self::[<from_ $p2>](q, v as $p2)
                        }
                    }
                }
                impl From<&Fq> for $p1 {
                    fn from(value: &Fq) -> $p1 {
                        <$p2>::from(value).try_into().unwrap()
                    }
                }
            )?
            impl From<Fq> for $p1 {
                fn from(value: Fq) -> $p1 {
                    (&value).into()
                }
            }
            impl_op_with_primitive!(
                impl AddAssign<&$p1> for Fq,
                impl SubAssign<&$p1> for Fq,
                impl MulAssign<&$p1> for Fq,
            );
            impl_op!(
                impl Add<$p1> for Fq,
                impl Sub<$p1> for Fq,
                impl Mul<$p1> for Fq,
            );
        )*
    };
}

impl_op_with_primitive!(
    u64,
    i64,
    u32 as u64,
    i32 as i64,
    u16 as u64,
    i16 as i64,
    u8 as u64,
    i8 as i64,
    usize as u64,
    isize as i64,
);

pub static NEG_NTT_PSI: OnceLock<Mutex<HashMap<u64, [Vec<Fq>; 2]>>> = OnceLock::new();

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
