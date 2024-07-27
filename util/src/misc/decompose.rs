use crate::{torus::T64, zq::Zq};
use core::{iter::repeat_with, ops::Mul};
use itertools::{izip, Itertools};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Base2Decomposor<T> {
    log_q: usize,
    log_b: usize,
    d: usize,
    rounding_bits: usize,
    bases: [T; 64],
}

impl<T> Base2Decomposor<T> {
    pub fn log_q(&self) -> usize {
        self.log_q
    }

    pub fn log_b(&self) -> usize {
        self.log_b
    }

    pub fn log_bases(&self) -> impl Iterator<Item = usize> + Clone {
        (self.rounding_bits..).step_by(self.log_b).take(self.d)
    }

    pub fn d(&self) -> usize {
        self.d
    }

    pub fn rounding_bits(&self) -> usize {
        self.rounding_bits
    }

    pub fn power_up<'a, I: 'a, O>(&'a self, v: I) -> impl Iterator<Item = O> + 'a
    where
        for<'t> &'t I: Mul<&'t T, Output = O>,
    {
        self.bases.iter().map(move |base| &v * base).take(self.d)
    }

    pub fn decompose<I: Base2Decomposable>(&self, v: &I) -> impl Iterator<Item = I> {
        v.rounding_shr(self.rounding_bits)
            .decompose(self.log_b)
            .take(self.d)
    }
}

impl Base2Decomposor<Zq> {
    pub fn new(q: u64, log_b: usize, d: usize) -> Self {
        let log_q = q.next_power_of_two().ilog2() as usize;
        let rounding_bits = log_q.saturating_sub(log_b * d);
        let mut bases = [Zq::from_u64(q, 0); 64];
        izip!(&mut bases, (rounding_bits..).step_by(log_b).take(d))
            .for_each(|(base, bits)| *base = Zq::from_u64(q, 1 << bits));
        Self {
            log_q,
            log_b,
            d,
            rounding_bits,
            bases,
        }
    }
}

impl Base2Decomposor<T64> {
    pub fn new(log_b: usize, d: usize) -> Self {
        let log_q = u64::BITS as usize;
        let rounding_bits = log_q.saturating_sub(log_b * d);
        let mut bases = [T64::default(); 64];
        izip!(&mut bases, (rounding_bits..).step_by(log_b).take(d))
            .for_each(|(base, bits)| *base = T64::from(1u64 << bits));
        Self {
            log_q,
            log_b,
            d,
            rounding_bits,
            bases,
        }
    }
}

pub trait Base2Decomposable: Sized {
    fn rounding_shr(&self, bits: usize) -> Self;

    fn round(&self, bits: usize) -> Self;

    fn decompose(self, log_b: usize) -> impl Iterator<Item = Self>;
}

impl Base2Decomposable for Zq {
    fn rounding_shr(&self, bits: usize) -> Self {
        let rounded = self + ((1u64 << bits) >> 1);
        Zq::from_u64(self.q(), rounded.to_u64() >> bits)
    }

    fn round(&self, bits: usize) -> Self {
        Zq::from_u64(self.q(), self.rounding_shr(bits).to_u64() << bits)
    }

    fn decompose(self, log_b: usize) -> impl Iterator<Item = Self> {
        let (b_by_2, mask, neg_b) = (1 << (log_b - 1), (1 << log_b) - 1, self.q() - (1 << log_b));
        let mut v = self.to_center_u64();
        repeat_with(move || {
            let limb = v & mask;
            let carry = (limb + (v & 1) > b_by_2) as u64;
            v >>= log_b;
            v += carry;
            Self::from_u64(self.q(), limb + carry * neg_b)
        })
    }
}

impl Base2Decomposable for T64 {
    fn rounding_shr(&self, bits: usize) -> Self {
        let rounded = self + ((1u64 << bits) >> 1);
        (rounded.to_u64() >> bits).into()
    }

    fn round(&self, bits: usize) -> Self {
        (self.rounding_shr(bits).to_u64() << bits).into()
    }

    fn decompose(self, log_b: usize) -> impl Iterator<Item = Self> {
        let mask = (1 << log_b) - 1;
        let mut v = self.to_u64();
        repeat_with(move || {
            let limb = v & mask;
            v >>= log_b;
            let carry = ((limb.wrapping_sub(1) | v) & limb) >> (log_b - 1);
            v += carry;
            (limb.wrapping_sub(carry << log_b)).into()
        })
    }
}

impl<T> Base2Decomposable for T
where
    T: IntoIterator + FromIterator<T::Item>,
    T::Item: Base2Decomposable,
    for<'t> &'t T: IntoIterator<Item = &'t T::Item>,
{
    fn rounding_shr(&self, bits: usize) -> Self {
        self.into_iter().map(|v| v.rounding_shr(bits)).collect()
    }

    fn round(&self, bits: usize) -> Self {
        self.into_iter().map(|v| v.round(bits)).collect()
    }

    fn decompose(self, log_b: usize) -> impl Iterator<Item = Self> {
        let mut iters = self.into_iter().map(|v| v.decompose(log_b)).collect_vec();
        repeat_with(move || iters.iter_mut().map(|iter| iter.next().unwrap()).collect())
    }
}
