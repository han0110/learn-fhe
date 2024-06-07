use std::iter::repeat_with;

use itertools::Itertools;

use crate::util::Fq;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decomposor {
    q: u64,
    log_b: usize,
    k: usize,
    rounding_bits: usize,
}

impl Decomposor {
    pub fn new(q: u64, log_b: usize, k: usize) -> Self {
        let log_q_ceil = q.next_power_of_two().ilog2() as usize;
        let rounding_bits = log_q_ceil.saturating_sub(log_b * k);
        Self {
            q,
            log_b,
            k,
            rounding_bits,
        }
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn log_bases(&self) -> impl Iterator<Item = usize> {
        (self.rounding_bits..).step_by(self.log_b).take(self.k)
    }

    pub fn bases(&self) -> impl Iterator<Item = Fq> + '_ {
        self.log_bases().map(|bits| Fq::from_u64(self.q, 1 << bits))
    }

    pub fn decompose<T: Decomposable>(&self, value: &T) -> impl Iterator<Item = T> {
        value
            .rounding_shr(self.rounding_bits)
            .decompose(self.log_b)
            .take(self.k)
    }
}

pub trait Decomposable: Sized {
    fn rounding_shr(&self, bits: usize) -> Self;

    fn decompose(self, log_b: usize) -> impl Iterator<Item = Self>;
}

impl Decomposable for Fq {
    fn rounding_shr(&self, bits: usize) -> Self {
        let rounded = self + ((1u64 << bits) >> 1);
        Fq::from_u64(self.q(), u64::from(rounded) >> bits)
    }

    fn decompose(self, log_b: usize) -> impl Iterator<Item = Self> {
        let mask = (1 << log_b) - 1;
        (0..)
            .step_by(log_b)
            .map(move |bits| Self::from_u64(self.q(), (u64::from(self) >> bits) & mask))
    }
}

impl<T> Decomposable for T
where
    T: IntoIterator + FromIterator<T::Item>,
    T::Item: Decomposable,
    for<'t> &'t T: IntoIterator<Item = &'t T::Item>,
{
    fn rounding_shr(&self, bits: usize) -> Self {
        self.into_iter().map(|v| v.rounding_shr(bits)).collect()
    }

    fn decompose(self, log_b: usize) -> impl Iterator<Item = Self> {
        let mut iters = self.into_iter().map(|v| v.decompose(log_b)).collect_vec();
        repeat_with(move || iters.iter_mut().map(|iter| iter.next().unwrap()).collect())
    }
}
