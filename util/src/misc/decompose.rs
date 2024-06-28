use crate::Zq;
use core::iter::repeat_with;
use itertools::Itertools;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decomposor {
    q: u64,
    log_b: usize,
    d: usize,
    rounding_bits: usize,
}

impl Decomposor {
    pub fn new(q: u64, log_b: usize, d: usize) -> Self {
        let log_q_ceil = q.next_power_of_two().ilog2() as usize;
        let rounding_bits = log_q_ceil.saturating_sub(log_b * d);
        Self {
            q,
            log_b,
            d,
            rounding_bits,
        }
    }

    pub fn d(&self) -> usize {
        self.d
    }

    pub fn log_bases(&self) -> impl Iterator<Item = usize> + Clone {
        (self.rounding_bits..).step_by(self.log_b).take(self.d)
    }

    pub fn bases(&self) -> impl Iterator<Item = Zq> + Clone + '_ {
        self.log_bases().map(|bits| Zq::from_u64(self.q, 1 << bits))
    }

    pub fn decompose<T: Decomposable>(&self, value: &T) -> impl Iterator<Item = T> {
        value
            .rounding_shr(self.rounding_bits)
            .decompose(self.log_b)
            .take(self.d)
    }
}

pub trait Decomposable: Sized {
    fn rounding_shr(&self, bits: usize) -> Self;

    fn decompose(self, log_b: usize) -> impl Iterator<Item = Self>;
}

impl Decomposable for Zq {
    fn rounding_shr(&self, bits: usize) -> Self {
        let rounded = self + ((1u64 << bits) >> 1);
        Zq::from_u64(self.q(), u64::from(rounded) >> bits)
    }

    fn decompose(self, log_b: usize) -> impl Iterator<Item = Self> {
        let (b_by_2, mask, neg_b) = (1 << (log_b - 1), (1 << log_b) - 1, self.q() - (1 << log_b));
        let mut v = self.into_center_unsigned();
        repeat_with(move || {
            let limb = v & mask;
            let carry = (limb + (v & 1) > b_by_2) as u64;
            v >>= log_b;
            v += carry;
            Self::from_u64(self.q(), limb + carry * neg_b)
        })
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
