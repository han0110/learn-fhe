use core::{
    iter::{repeat_with, Sum},
    num::Wrapping,
    ops::{Add, AddAssign, BitAnd, Deref, DerefMut, Mul, Neg, Rem, RemAssign, Shl, Shr, Sub},
    slice,
};
use itertools::izip;
use num_traits::{WrappingAdd, WrappingMul, WrappingNeg, WrappingShl, WrappingShr, WrappingSub};
use rand::{Rng, RngCore};
use rand_distr::{Distribution, Normal, Standard};
use std::vec::IntoIter;

pub type W64 = Wrapping<u64>;

pub fn uniform_binary(rng: &mut impl RngCore) -> bool {
    Standard.sample(rng)
}

pub fn uniform_binaries(n: usize, rng: &mut impl RngCore) -> Vec<bool> {
    Standard.sample_iter(rng).take(n).collect()
}

pub fn uniform_q(q: W64, rng: &mut impl RngCore) -> W64 {
    Wrapping(rng.next_u64()) % q
}

#[derive(Clone, Copy, Debug)]
pub struct TorusNormal {
    q: u64,
    normal: Normal<f64>,
}

impl TorusNormal {
    pub fn new(log_q: usize, std_dev: f64) -> Self {
        TorusNormal {
            q: 1 << log_q,
            normal: Normal::new(0., std_dev).unwrap(),
        }
    }

    pub fn std_dev(&self) -> f64 {
        self.normal.std_dev()
    }
}

impl Distribution<W64> for TorusNormal {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> W64 {
        let input = self.normal.sample(rng);
        let fractional = input - input.round();
        let numerator = (fractional * self.q as f64).round();
        Wrapping(if numerator.is_sign_negative() {
            self.q - numerator.abs() as u64
        } else {
            numerator as u64
        })
    }
}

pub type Polynomial<T> = AdditiveVec<T>;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AdditiveVec<T>(Vec<T>);

impl AdditiveVec<W64> {
    pub fn uniform_q(q: W64, n: usize, rng: &mut impl RngCore) -> Self {
        repeat_with(|| uniform_q(q, rng)).take(n).collect()
    }

    pub fn constant(c: W64, n: usize) -> Self {
        let mut poly = avec![Wrapping(0); n];
        poly[0] = c;
        poly
    }

    pub fn monomial(i: W64, n: usize) -> Self {
        let i = (i.0 % (2 * n as u64)) as usize;
        let mut poly = avec![Wrapping(0); n];
        if i < n {
            poly[i] = Wrapping(1)
        } else {
            poly[i - n] = -Wrapping(1)
        }
        poly
    }

    pub fn into_split_last(mut self) -> Option<(W64, Self)> {
        let last = self.pop()?;
        Some((last, self))
    }
}

impl<T> From<AdditiveVec<T>> for Vec<T> {
    fn from(value: AdditiveVec<T>) -> Self {
        value.0
    }
}

impl<T> From<Vec<T>> for AdditiveVec<T> {
    fn from(value: Vec<T>) -> Self {
        Self(value)
    }
}

impl<'a, T: Clone> From<&'a [T]> for AdditiveVec<T> {
    fn from(value: &'a [T]) -> Self {
        value.to_vec().into()
    }
}

impl<T> Deref for AdditiveVec<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for AdditiveVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> Extend<T> for AdditiveVec<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.0.extend(iter)
    }
}

impl<T> FromIterator<T> for AdditiveVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<T> IntoIterator for AdditiveVec<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a AdditiveVec<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<T: WrappingNeg> Neg for AdditiveVec<T> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.iter_mut()
            .for_each(|value| *value = value.wrapping_neg());
        self
    }
}

impl<T: WrappingAdd> AddAssign<AdditiveVec<T>> for AdditiveVec<T> {
    fn add_assign(&mut self, rhs: AdditiveVec<T>) {
        *self = self.wrapping_add(&rhs);
    }
}

impl<T: WrappingAdd> Add<&AdditiveVec<T>> for &AdditiveVec<T> {
    type Output = AdditiveVec<T>;

    fn add(self, rhs: &AdditiveVec<T>) -> Self::Output {
        self.wrapping_add(rhs)
    }
}

impl<T: WrappingAdd> Add<AdditiveVec<T>> for &AdditiveVec<T> {
    type Output = AdditiveVec<T>;

    fn add(self, rhs: AdditiveVec<T>) -> Self::Output {
        self + &rhs
    }
}

impl<T: WrappingAdd> Add<&AdditiveVec<T>> for AdditiveVec<T> {
    type Output = AdditiveVec<T>;

    fn add(self, rhs: &AdditiveVec<T>) -> Self::Output {
        &self + rhs
    }
}

impl<T: WrappingAdd> Add<AdditiveVec<T>> for AdditiveVec<T> {
    type Output = AdditiveVec<T>;

    fn add(self, rhs: AdditiveVec<T>) -> Self::Output {
        &self + &rhs
    }
}

impl<T: WrappingAdd> WrappingAdd for AdditiveVec<T> {
    fn wrapping_add(&self, rhs: &Self) -> Self {
        assert_eq!(self.len(), rhs.len());
        izip!(&**self, &**rhs)
            .map(|(lhs, rhs)| lhs.wrapping_add(rhs))
            .collect()
    }
}

impl<T: WrappingSub> Sub<&AdditiveVec<T>> for &AdditiveVec<T> {
    type Output = AdditiveVec<T>;

    fn sub(self, rhs: &AdditiveVec<T>) -> Self::Output {
        self.wrapping_sub(rhs)
    }
}

impl<T: WrappingSub> Sub<AdditiveVec<T>> for &AdditiveVec<T> {
    type Output = AdditiveVec<T>;

    fn sub(self, rhs: AdditiveVec<T>) -> Self::Output {
        self - &rhs
    }
}

impl<T: WrappingSub> Sub<&AdditiveVec<T>> for AdditiveVec<T> {
    type Output = AdditiveVec<T>;

    fn sub(self, rhs: &AdditiveVec<T>) -> Self::Output {
        &self - rhs
    }
}

impl<T: WrappingSub> Sub<AdditiveVec<T>> for AdditiveVec<T> {
    type Output = AdditiveVec<T>;

    fn sub(self, rhs: AdditiveVec<T>) -> Self::Output {
        &self - &rhs
    }
}

impl<T: WrappingSub> WrappingSub for AdditiveVec<T> {
    fn wrapping_sub(&self, rhs: &Self) -> Self {
        assert_eq!(self.len(), rhs.len());
        izip!(&**self, &**rhs)
            .map(|(lhs, rhs)| lhs.wrapping_sub(rhs))
            .collect()
    }
}

impl<T> Mul<AdditiveVec<T>> for AdditiveVec<T>
where
    Self: NegCyclicMul<AdditiveVec<T>>,
{
    type Output = AdditiveVec<T>;

    fn mul(self, rhs: AdditiveVec<T>) -> Self::Output {
        self.negcyclic_mul(&rhs)
    }
}

impl<T> WrappingMul for AdditiveVec<T>
where
    Self: NegCyclicMul<AdditiveVec<T>>,
{
    fn wrapping_mul(&self, rhs: &AdditiveVec<T>) -> Self::Output {
        self.negcyclic_mul(rhs)
    }
}

impl<T: WrappingMul> Mul<T> for AdditiveVec<T> {
    type Output = AdditiveVec<T>;

    fn mul(self, rhs: T) -> Self::Output {
        self.into_iter().map(|lhs| lhs.wrapping_mul(&rhs)).collect()
    }
}

impl<T: WrappingMul> Mul<T> for &AdditiveVec<T> {
    type Output = AdditiveVec<T>;

    fn mul(self, rhs: T) -> Self::Output {
        self.iter().map(|lhs| lhs.wrapping_mul(&rhs)).collect()
    }
}

impl<T: Copy + BitAnd<Output = T>> BitAnd<T> for AdditiveVec<T> {
    type Output = AdditiveVec<T>;

    fn bitand(self, rhs: T) -> Self::Output {
        self.into_iter().map(|lhs| lhs & rhs).collect()
    }
}

impl<T: Copy + BitAnd<Output = T>> BitAnd<T> for &AdditiveVec<T> {
    type Output = AdditiveVec<T>;

    fn bitand(self, rhs: T) -> Self::Output {
        self.iter().map(|lhs| *lhs & rhs).collect()
    }
}

impl<T: RemAssign<R>, R: Copy> RemAssign<R> for AdditiveVec<T> {
    fn rem_assign(&mut self, rhs: R) {
        self.iter_mut().for_each(|value| *value %= rhs);
    }
}

impl<T: RemAssign<R>, R: Copy> Rem<R> for AdditiveVec<T> {
    type Output = AdditiveVec<T>;

    fn rem(mut self, rhs: R) -> Self::Output {
        self %= rhs;
        self
    }
}

impl<T: Clone + RemAssign<R>, R: Copy> Rem<R> for &AdditiveVec<T> {
    type Output = AdditiveVec<T>;

    fn rem(self, rhs: R) -> Self::Output {
        self.clone() % rhs
    }
}

impl<T: WrappingShr> Shr<usize> for &AdditiveVec<T> {
    type Output = AdditiveVec<T>;

    fn shr(self, rhs: usize) -> Self::Output {
        self.wrapping_shr(rhs as u32)
    }
}

impl<T: WrappingShr> Shr<usize> for AdditiveVec<T> {
    type Output = AdditiveVec<T>;

    fn shr(self, rhs: usize) -> Self::Output {
        self.wrapping_shr(rhs as u32)
    }
}

impl<T: WrappingShr> WrappingShr for AdditiveVec<T> {
    fn wrapping_shr(&self, rhs: u32) -> Self {
        self.iter().map(|value| value.wrapping_shr(rhs)).collect()
    }
}

impl<T: WrappingShl> Shl<usize> for &AdditiveVec<T> {
    type Output = AdditiveVec<T>;

    fn shl(self, rhs: usize) -> Self::Output {
        self.wrapping_shl(rhs as u32)
    }
}

impl<T: WrappingShl> Shl<usize> for AdditiveVec<T> {
    type Output = AdditiveVec<T>;

    fn shl(self, rhs: usize) -> Self::Output {
        self.wrapping_shl(rhs as u32)
    }
}

impl<T: WrappingShl> WrappingShl for AdditiveVec<T> {
    fn wrapping_shl(&self, rhs: u32) -> Self {
        self.iter().map(|value| value.wrapping_shl(rhs)).collect()
    }
}

impl<T: WrappingAdd> Sum for AdditiveVec<T> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.into_iter()
            .reduce(|acc, item| acc.wrapping_add(&item))
            .unwrap()
    }
}

pub trait Dot<Rhs> {
    type Output;

    fn dot(&self, rhs: &Rhs) -> Self::Output;
}

impl<T: Clone + Sum> Dot<Vec<bool>> for AdditiveVec<T> {
    type Output = T;

    fn dot(&self, rhs: &Vec<bool>) -> Self::Output {
        assert_eq!(self.len(), rhs.len());
        izip!(&self.0, rhs)
            .filter(|(_, rhs)| **rhs)
            .map(|(lhs, _)| lhs.clone())
            .sum()
    }
}

impl<T> Dot<Vec<Vec<bool>>> for AdditiveVec<Polynomial<T>>
where
    Polynomial<T>: NegCyclicMul<Vec<bool>> + Sum,
{
    type Output = Polynomial<T>;

    fn dot(&self, rhs: &Vec<Vec<bool>>) -> Self::Output {
        assert_eq!(self.len(), rhs.len());
        izip!(&self.0, rhs)
            .map(|(lhs, rhs)| lhs.negcyclic_mul(rhs))
            .sum()
    }
}

impl<T: Clone + WrappingMul + Sum> Dot<AdditiveVec<AdditiveVec<T>>> for AdditiveVec<T> {
    type Output = AdditiveVec<T>;

    fn dot(&self, rhs: &AdditiveVec<AdditiveVec<T>>) -> Self::Output {
        assert_eq!(self.len(), rhs.len());
        (0..rhs[0].len())
            .map(|col| {
                izip!(self, rhs.iter().map(|row| &row[col]))
                    .map(|(lhs, rhs)| lhs.wrapping_mul(rhs))
                    .sum()
            })
            .collect()
    }
}

pub trait NegCyclicMul<Rhs> {
    fn negcyclic_mul(&self, rhs: &Rhs) -> Self;
}

impl<T> NegCyclicMul<Vec<bool>> for Polynomial<T>
where
    T: Clone + Default + WrappingAdd + WrappingSub,
{
    fn negcyclic_mul(&self, rhs: &Vec<bool>) -> Self {
        assert_eq!(self.len(), rhs.len());
        let n = self.len();
        let mut out = vec![T::default(); n];
        izip!(0.., &**self).for_each(|(i, lhs)| {
            izip!(0.., &**rhs).for_each(|(j, rhs)| {
                if *rhs {
                    if i + j < n {
                        out[i + j] = out[i + j].wrapping_add(lhs);
                    } else {
                        out[i + j - n] = out[i + j - n].wrapping_sub(lhs);
                    }
                }
            })
        });
        Self(out)
    }
}

impl<T> NegCyclicMul<Polynomial<T>> for Polynomial<T>
where
    T: Clone + Default + WrappingAdd + WrappingSub + WrappingMul,
{
    fn negcyclic_mul(&self, rhs: &Polynomial<T>) -> Self {
        assert_eq!(self.len(), rhs.len());
        let n = self.len();
        let mut out = vec![T::default(); n];
        izip!(0.., &**self).for_each(|(i, lhs)| {
            izip!(0.., &**rhs).for_each(|(j, rhs)| {
                if i + j < n {
                    out[i + j] = out[i + j].wrapping_add(&lhs.wrapping_mul(rhs));
                } else {
                    out[i + j - n] = out[i + j - n].wrapping_sub(&lhs.wrapping_mul(rhs));
                }
            })
        });
        Self(out)
    }
}

pub trait Round {
    fn round_shr(self, bits: usize) -> Self;

    fn round(self, bits: usize) -> Self;
}

impl Round for W64 {
    fn round_shr(self, bits: usize) -> Self {
        (self + Wrapping((1 << bits) >> 1)) >> bits
    }

    fn round(self, bits: usize) -> Self {
        ((self + Wrapping((1 << bits) >> 1)) >> bits) << bits
    }
}

impl<T: Round> Round for AdditiveVec<T> {
    fn round_shr(self, bits: usize) -> Self {
        self.into_iter()
            .map(|value| value.round_shr(bits))
            .collect()
    }

    fn round(self, bits: usize) -> Self {
        self.into_iter().map(|value| value.round(bits)).collect()
    }
}

pub trait Decompose<T = Self> {
    fn decompose(self, log_q: usize, log_b: usize) -> impl Iterator<Item = T>;
}

impl Decompose for W64 {
    fn decompose(self, log_q: usize, log_b: usize) -> impl Iterator<Item = Self> {
        let mask = Wrapping((1 << log_b) - 1);
        (0..log_q)
            .step_by(log_b)
            .rev()
            .map(move |shift| (self >> shift) & mask)
    }
}

impl Decompose for Polynomial<W64> {
    fn decompose(self, log_q: usize, log_b: usize) -> impl Iterator<Item = Self> {
        let mask = Wrapping((1 << log_b) - 1);
        (0..log_q)
            .step_by(log_b)
            .rev()
            .map(move |shift| (&self >> shift) & mask)
    }
}

macro_rules! avec {
    () => {
        $crate::util::AdditiveVec::from(vec![])
    };
    ($elem:expr; $n:expr) => {
        $crate::util::AdditiveVec::from(vec![$elem; $n])
    };
}

pub(crate) use avec;
