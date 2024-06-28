use crate::bit_reverse;
use core::{
    borrow::Borrow,
    cmp::Ordering,
    fmt::{self, Display, Formatter},
    iter::{successors, Product, Sum},
    ops::{AddAssign, Deref, MulAssign, Neg, SubAssign},
};
use itertools::Itertools;
use num_bigint::{BigInt, BigUint, ToBigUint};
use num_bigint_dig::prime::probably_prime;
use num_integer::Integer;
use num_traits::ToPrimitive;
use rand::RngCore;
use rand_distr::{Distribution, Uniform};
use std::{
    collections::{hash_map::Entry, HashMap},
    sync::{Mutex, MutexGuard, OnceLock},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Zq {
    q: u64,
    v: u64,
}

impl Zq {
    pub fn q(&self) -> u64 {
        self.q
    }

    pub fn from_bigint(q: u64, v: &BigInt) -> Self {
        let v = (v % q).to_i64().unwrap();
        let v = if v < 0 { v + q as i64 } else { v };
        Self { q, v: v as u64 }
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
        Zq::from_u64(q, Uniform::new(0, q).sample(rng))
    }

    pub fn sample_i8(q: u64, dist: impl Distribution<i8>, rng: &mut impl RngCore) -> Self {
        Zq::from_i8(q, dist.sample(rng))
    }

    pub fn generator(q: u64) -> Self {
        let order = q - 1;
        (1..order)
            .map(|g| Zq::from_u64(q, g))
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
        successors(Some(Zq::from_i8(self.q, 1)), move |v| Some(v * self))
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

impl From<&Zq> for u64 {
    fn from(value: &Zq) -> Self {
        value.v
    }
}

impl From<&Zq> for i64 {
    fn from(value: &Zq) -> Self {
        value.into_center_signed()
    }
}

impl From<Zq> for f64 {
    fn from(value: Zq) -> Self {
        value.into_center_signed() as f64
    }
}

impl From<&Zq> for f64 {
    fn from(value: &Zq) -> Self {
        value.into_center_signed() as f64
    }
}

impl Ord for Zq {
    fn cmp(&self, other: &Self) -> Ordering {
        assert_eq!(self.q, other.q);
        self.v.cmp(&other.v)
    }
}

impl PartialOrd for Zq {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Display for Zq {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.v)
    }
}

impl Neg for &Zq {
    type Output = Zq;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Zq::from_u64(self.q, self.q - self.v)
    }
}

impl Neg for Zq {
    type Output = Zq;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        -&self
    }
}

impl AddAssign<&Zq> for Zq {
    #[inline(always)]
    fn add_assign(&mut self, rhs: &Zq) {
        assert_eq!(self.q, rhs.q);
        *self = Self::from_u128(self.q, self.v as u128 + rhs.v as u128);
    }
}

impl SubAssign<&Zq> for Zq {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &Zq) {
        assert_eq!(self.q, rhs.q);
        *self += -rhs;
    }
}

impl MulAssign<&Zq> for Zq {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &Zq) {
        assert_eq!(self.q, rhs.q);
        *self = Self::from_u128(self.q, self.v as u128 * rhs.v as u128);
    }
}

impl<T: Borrow<Zq>> Sum<T> for Zq {
    fn sum<I: Iterator<Item = T>>(mut iter: I) -> Self {
        let init = *iter.next().unwrap().borrow();
        iter.fold(init, |acc, item| acc + item.borrow())
    }
}

impl<T: Borrow<Zq>> Product<T> for Zq {
    fn product<I: Iterator<Item = T>>(mut iter: I) -> Self {
        let init = *iter.next().unwrap().borrow();
        iter.fold(init, |acc, item| acc * item.borrow())
    }
}

macro_rules! impl_rest_op_by_op_assign_ref {
    (@ impl $trait:ident<$rhs:ty> for $lhs:ty; type Output = $out:ty; $lhs_convert:expr) => {
        paste::paste! {
            impl core::ops::$trait<$rhs> for $lhs {
                type Output = $out;

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
            impl_rest_op_by_op_assign_ref!(@ impl $trait<$rhs> for $lhs; type Output = $lhs; core::convert::identity);
            impl_rest_op_by_op_assign_ref!(@ impl $trait<&$rhs> for $lhs; type Output = $lhs; core::convert::identity);
            impl_rest_op_by_op_assign_ref!(@ impl $trait<$rhs> for &$lhs; type Output = $lhs; <_>::clone);
            impl_rest_op_by_op_assign_ref!(@ impl $trait<&$rhs> for &$lhs; type Output = $lhs; <_>::clone);
        )*
    };
}

macro_rules! impl_op_with_primitive {
    (@ impl $trait:ident<&$p:ty> for Zq) => {
        paste::paste! {
            impl core::ops::$trait<&$p> for Zq {
                #[inline(always)]
                fn [<$trait:snake:lower>](&mut self, rhs: &$p) {
                    self.[<$trait:snake:lower>](Self::[<from_ $p>](self.q, *rhs));
                }
            }
        }
    };
    ($(impl $trait:ident<&$p:ty> for Zq),* $(,)?) => {
        $(impl_op_with_primitive!(@ impl $trait<&$p> for Zq);)*
    };
    ($($p1:ty $(as $p2:ty)?),* $(,)?) => {
        $(
            $(
                paste::paste! {
                    impl Zq {
                        pub fn [<from_ $p1>](q: u64, v: $p1) -> Self {
                            Self::[<from_ $p2>](q, v as $p2)
                        }
                    }
                }
                impl From<&Zq> for $p1 {
                    fn from(value: &Zq) -> $p1 {
                        <$p2>::from(value).try_into().unwrap()
                    }
                }
            )?
            impl From<Zq> for $p1 {
                fn from(value: Zq) -> $p1 {
                    (&value).into()
                }
            }
            impl_op_with_primitive!(
                impl AddAssign<&$p1> for Zq,
                impl SubAssign<&$p1> for Zq,
                impl MulAssign<&$p1> for Zq,
            );
            impl_rest_op_by_op_assign_ref!(
                impl Add<$p1> for Zq,
                impl Sub<$p1> for Zq,
                impl Mul<$p1> for Zq,
            );
        )*
    };
}

impl_rest_op_by_op_assign_ref!(
    impl Add<Zq> for Zq,
    impl Sub<Zq> for Zq,
    impl Mul<Zq> for Zq,
);
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

pub(crate) use impl_rest_op_by_op_assign_ref;

pub fn two_adic_primes(bits: usize, log_n: usize) -> impl Iterator<Item = u64> {
    assert!(bits > log_n);
    let (min, max) = (1 << (bits - log_n - 1), 1 << (bits - log_n));
    primes((min..max).rev().map(move |q| (q << log_n) + 1))
}

fn primes(candidates: impl IntoIterator<Item = u64>) -> impl Iterator<Item = u64> {
    candidates
        .into_iter()
        .filter(|candidate| is_prime(*candidate))
}

fn is_prime(q: impl Into<num_bigint_dig::BigUint>) -> bool {
    probably_prime(&q.into(), 20)
}

static TWIDDLE: OnceLock<Mutex<HashMap<u64, [Vec<Zq>; 2]>>> = OnceLock::new();

pub struct Twiddle<'a>(MutexGuard<'a, HashMap<u64, [Vec<Zq>; 2]>>, u64);

impl<'a> Twiddle<'a> {
    pub fn get(&self) -> Option<&[Vec<Zq>; 2]> {
        self.0.get(&self.1)
    }
}

impl<'a> Deref for Twiddle<'a> {
    type Target = [Vec<Zq>; 2];

    fn deref(&self) -> &Self::Target {
        &self.0[&self.1]
    }
}

// Twiddle factors in bit-reversed order.
pub fn twiddle<'a>(q: u64) -> Twiddle<'a> {
    let mut guard = TWIDDLE.get_or_init(Default::default).lock().unwrap();
    if let Entry::Vacant(entry) = guard.entry(q) {
        if is_prime(q) {
            entry.insert(compute_twiddle(q));
        }
    }
    Twiddle(guard, q)
}

fn compute_twiddle(q: u64) -> [Vec<Zq>; 2] {
    let order = q - 1;
    let s = order.trailing_zeros();
    let twiddle = {
        let g = Zq::generator(q);
        let rou = g.pow(order >> s);
        let mut twiddle = rou.powers().take(1 << (s - 1)).collect_vec();
        bit_reverse(&mut twiddle);
        twiddle
    };
    let twiddle_inv = twiddle.iter().map(|v| v.inv().unwrap()).collect();
    [twiddle, twiddle_inv]
}
