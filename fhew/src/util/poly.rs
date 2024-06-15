use crate::util::{
    avec::{
        impl_element_wise_neg, impl_element_wise_op, impl_element_wise_op_assign,
        impl_mul_assign_element, impl_mul_element, AVec,
    },
    fq::{Fq, NEG_NTT_PSI},
    izip_eq,
};
use core::{
    borrow::Borrow,
    fmt::{self, Debug, Display, Formatter},
    iter::Sum,
    marker::PhantomData,
    ops::{AddAssign, BitXor, Mul, MulAssign, Neg},
    slice,
};
use derive_more::{Deref, DerefMut};
use itertools::izip;
use rand::RngCore;
use rand_distr::Distribution;
use std::vec;

pub trait Basis: Clone + Copy + Debug + PartialEq + Eq {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Coefficient;

impl Basis for Coefficient {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Evaluation;

impl Basis for Evaluation {}

#[derive(Clone, Debug, PartialEq, Eq, Deref, DerefMut)]
pub struct Poly<T, B: Basis = Coefficient>(
    #[deref]
    #[deref_mut]
    AVec<T>,
    PhantomData<B>,
);

impl<T, B: Basis> Poly<T, B> {
    fn new(v: AVec<T>) -> Self {
        assert!(v.len().is_power_of_two());
        Self(v, PhantomData)
    }

    pub fn n(&self) -> usize {
        self.len()
    }

    pub fn sample(n: usize, dist: &impl Distribution<T>, rng: &mut impl RngCore) -> Self {
        Self::new(AVec::sample(n, dist, rng))
    }
}

impl<T: Copy + Neg<Output = T>> Poly<T, Coefficient> {
    pub fn automorphism(&self, t: i64) -> Self {
        let mut poly = self.clone();
        let n = self.n();
        let t = t.rem_euclid(2 * n as i64) as usize;
        (0..n).for_each(|i| {
            let it = (i * t) % (2 * n);
            if it < n {
                poly[it] = self[i]
            } else {
                poly[it - n] = -self[i]
            }
        });
        poly
    }
}

impl Poly<Fq, Coefficient> {
    pub fn zero(n: usize, q: u64) -> Self {
        Self::new(vec![Fq::from_u64(q, 0); n].into())
    }

    pub fn one(n: usize, q: u64) -> Self {
        let mut poly = Self::zero(n, q);
        poly[0] += 1;
        poly
    }

    pub fn constant(n: usize, v: Fq) -> Self {
        let mut poly = Self::zero(n, v.q());
        poly[0] = v;
        poly
    }

    pub fn sample_fq_uniform(n: usize, q: u64, rng: &mut impl RngCore) -> Self {
        Self::new(AVec::sample_fq_uniform(n, q, rng))
    }

    pub fn sample_fq_from_i8(
        n: usize,
        q: u64,
        dist: &impl Distribution<i8>,
        rng: &mut impl RngCore,
    ) -> Self {
        Self::new(AVec::sample_fq_from_i8(n, q, dist, rng))
    }

    pub fn mod_switch(&self, q_prime: u64) -> Self {
        Self::new(self.0.mod_switch(q_prime))
    }

    pub fn mod_switch_odd(&self, q_prime: u64) -> Self {
        Self::new(self.0.mod_switch_odd(q_prime))
    }

    pub fn to_evaluation(&self) -> Poly<Fq, Evaluation> {
        let [psi, _] = &NEG_NTT_PSI.get().unwrap().lock().unwrap()[&self[0].q()];
        let mut a = self.0.clone();
        neg_ntt_in_place(&mut a, psi);
        Poly::new(a)
    }
}

impl Poly<Fq, Evaluation> {
    pub fn to_coefficient(&self) -> Poly<Fq, Coefficient> {
        let [_, psi_inv] = &NEG_NTT_PSI.get().unwrap().lock().unwrap()[&self[0].q()];
        let mut a = self.0.clone();
        neg_intt_in_place(&mut a, psi_inv);
        Poly::new(a)
    }
}

impl<T: Display> Display for Poly<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<T> From<Poly<T>> for Vec<T> {
    fn from(value: Poly<T>) -> Self {
        value.0.into()
    }
}

impl<T> From<Vec<T>> for Poly<T> {
    fn from(value: Vec<T>) -> Self {
        Self::new(value.into())
    }
}

impl<T: Clone> From<&[T]> for Poly<T> {
    fn from(value: &[T]) -> Self {
        Self::new(value.into())
    }
}

impl<T> From<Poly<T>> for AVec<T> {
    fn from(value: Poly<T>) -> Self {
        value.0
    }
}

impl<T> From<AVec<T>> for Poly<T> {
    fn from(value: AVec<T>) -> Self {
        Self::new(value)
    }
}

impl<T, B: Basis> FromIterator<T> for Poly<T, B> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

impl<T, B: Basis> IntoIterator for Poly<T, B> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T, B: Basis> IntoIterator for &'a Poly<T, B> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a, T, B: Basis> IntoIterator for &'a mut Poly<T, B> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

impl MulAssign<&Poly<Fq, Coefficient>> for Poly<Fq, Coefficient> {
    fn mul_assign(&mut self, rhs: &Poly<Fq, Coefficient>) {
        assert_eq!(self.len(), rhs.len());
        match NEG_NTT_PSI.get().unwrap().lock().unwrap().get(&self[0].q()) {
            Some([psi, psi_inv]) if self.n() <= psi.len() => {
                neg_ntt_mul_assign(self, rhs, psi, psi_inv)
            }
            _ => *self = neg_schoolbook_mul(self, rhs),
        }
    }
}

impl MulAssign<&Poly<Fq, Evaluation>> for Poly<Fq, Evaluation> {
    fn mul_assign(&mut self, rhs: &Poly<Fq, Evaluation>) {
        izip_eq!(self, rhs).for_each(|(lhs, rhs)| *lhs *= rhs);
    }
}

impl MulAssign<&Poly<i8>> for Poly<Fq> {
    fn mul_assign(&mut self, rhs: &Poly<i8>) {
        *self *= Self::from_iter(rhs.iter().map(|v| Fq::from_i8(self[0].q(), *v)));
    }
}

impl Mul<&Fq> for &Poly<i8> {
    type Output = Poly<Fq>;

    fn mul(self, rhs: &Fq) -> Self::Output {
        let q = rhs.q();
        self.iter().map(|v| Fq::from_i8(q, *v) * rhs).collect()
    }
}

impl<T, B, Item> Sum<Item> for Poly<T, B>
where
    T: Clone + for<'t> AddAssign<&'t T>,
    B: Basis,
    Item: Borrow<Poly<T, B>>,
{
    fn sum<I: Iterator<Item = Item>>(mut iter: I) -> Self {
        let init = iter.next().unwrap().borrow().0.clone();
        Self::new(iter.fold(init, |mut acc, item| {
            acc += &item.borrow().0;
            acc
        }))
    }
}

impl_element_wise_neg!(
    impl<T> Neg for Poly<T, Coefficient>,
    impl<T> Neg for Poly<T, Evaluation>,
);
impl_element_wise_op_assign!(
    impl<T> AddAssign<Poly<T, Coefficient>> for Poly<T, Coefficient>,
    impl<T> AddAssign<Poly<T, Evaluation>> for Poly<T, Evaluation>,
    impl<T> SubAssign<Poly<T, Coefficient>> for Poly<T, Coefficient>,
    impl<T> SubAssign<Poly<T, Evaluation>> for Poly<T, Evaluation>,
);
impl_element_wise_op!(
    impl<T> Add<Poly<T, Coefficient>> for Poly<T, Coefficient>,
    impl<T> Add<Poly<T, Evaluation>> for Poly<T, Evaluation>,
    impl<T> Sub<Poly<T, Coefficient>> for Poly<T, Coefficient>,
    impl<T> Sub<Poly<T, Evaluation>> for Poly<T, Evaluation>,
);
impl_mul_assign_element!(
    impl<T> MulAssign<T> for Poly<T, Coefficient>,
    impl<T> MulAssign<T> for Poly<T, Evaluation>,
);
impl_mul_element!(
    impl<T> Mul<T> for Poly<T, Coefficient>,
    impl<T> Mul<T> for Poly<T, Evaluation>,
);

macro_rules! impl_poly_fq_mul {
    (@ impl Mul<$rhs:ty> for $lhs:ty; type Output = $out:ty; $lhs_convert:expr) => {
        impl core::ops::Mul<$rhs> for $lhs {
            type Output = $out;

            fn mul(self, rhs: $rhs) -> $out {
                let mut lhs = $lhs_convert(self);
                lhs.mul_assign(rhs.borrow());
                lhs
            }
        }
    };
    ($(impl Mul<$rhs:ty> for $lhs:ty),* $(,)?) => {
        $(
            impl core::ops::MulAssign<$rhs> for $lhs {
                fn mul_assign(&mut self, rhs: $rhs) {
                    *self *= &rhs;
                }
            }
            impl_poly_fq_mul!(@ impl Mul<$rhs> for $lhs; type Output = $lhs; core::convert::identity);
            impl_poly_fq_mul!(@ impl Mul<&$rhs> for $lhs; type Output = $lhs; core::convert::identity);
            impl_poly_fq_mul!(@ impl Mul<$rhs> for &$lhs; type Output = $lhs; <_>::clone);
            impl_poly_fq_mul!(@ impl Mul<&$rhs> for &$lhs; type Output = $lhs; <_>::clone);
        )*
    }
}

macro_rules! impl_poly_i8_mul {
    (@ impl Mul<$rhs:ty> for $lhs:ty) => {
        impl core::ops::Mul<$rhs> for $lhs {
            type Output = Poly<Fq>;

            fn mul(self, rhs: $rhs) -> Poly<Fq> {
                self.borrow().mul(rhs.borrow())
            }
        }
    };
    ($(impl Mul<$rhs:ty> for Poly<i8>),* $(,)?) => {
        $(
            impl_poly_i8_mul!(@ impl Mul<$rhs> for Poly<i8>);
            impl_poly_i8_mul!(@ impl Mul<&$rhs> for Poly<i8>);
            impl_poly_i8_mul!(@ impl Mul<$rhs> for &Poly<i8>);
        )*
    }
}

impl_poly_fq_mul!(
    impl Mul<Poly<Fq, Coefficient>> for Poly<Fq, Coefficient>,
    impl Mul<Poly<Fq, Evaluation>> for Poly<Fq, Evaluation>,
    impl Mul<Poly<i8>> for Poly<Fq>,
    impl Mul<Monomial> for Poly<Fq>,
);
impl_poly_i8_mul!(
    impl Mul<Fq> for Poly<i8>,
);

pub struct Monomial(i64);

impl MulAssign<&Monomial> for Poly<Fq> {
    fn mul_assign(&mut self, rhs: &Monomial) {
        let n = self.n();
        let i = rhs.0.rem_euclid(2 * n as i64) as usize;
        self.rotate_right(i % n);
        if i < n {
            self[..i].iter_mut().for_each(|v| *v = -*v);
        } else {
            self[i - n..].iter_mut().for_each(|v| *v = -*v);
        }
    }
}

pub struct X;

impl BitXor<&i8> for X {
    type Output = Monomial;

    fn bitxor(self, rhs: &i8) -> Self::Output {
        Monomial(*rhs as i64)
    }
}

impl BitXor<Fq> for X {
    type Output = Monomial;

    fn bitxor(self, rhs: Fq) -> Self::Output {
        Monomial(rhs.into())
    }
}

fn neg_schoolbook_mul(a: &Poly<Fq>, b: &Poly<Fq>) -> Poly<Fq> {
    let n = a.len();
    let mut c = Poly::from(vec![Fq::from_u64(a[0].q(), 0); n]);
    izip!(0.., a.iter()).for_each(|(i, a)| {
        izip!(0.., b.iter()).for_each(|(j, b)| {
            if i + j < n {
                c[i + j] += a * b;
            } else {
                c[i + j - n] -= a * b;
            }
        })
    });
    c
}

fn neg_ntt_mul_assign(a: &mut Poly<Fq>, b: &Poly<Fq>, psi: &[Fq], psi_inv: &[Fq]) {
    let b = &mut b.clone();
    neg_ntt_in_place(a, psi);
    neg_ntt_in_place(b, psi);
    izip!(a.iter_mut(), b.iter()).for_each(|(a, b)| *a *= b);
    neg_intt_in_place(a, psi_inv);
}

// Algorithm 1 in 2016/504.
fn neg_ntt_in_place(a: &mut [Fq], psi: &[Fq]) {
    assert!(a.len().is_power_of_two());

    for log_m in 0..a.len().ilog2() {
        let m = 1 << log_m;
        let t = a.len() / m;
        izip!(0.., a.chunks_exact_mut(t), &psi[m..]).for_each(|(i, a, psi)| {
            let (u, v) = a.split_at_mut(t / 2);
            if m == 0 && i == 0 {
                izip!(u, v).for_each(|(u, v)| twiddle_free_bufferfly(u, v));
            } else {
                izip!(u, v).for_each(|(u, v)| dit_bufferfly(u, v, psi));
            }
        });
    }
}

// Algorithm 2 in 2016/504.
fn neg_intt_in_place(a: &mut [Fq], psi_inv: &[Fq]) {
    assert!(a.len().is_power_of_two());

    for log_m in (0..a.len().ilog2()).rev() {
        let m = 1 << log_m;
        let t = a.len() / m;
        izip!(0.., a.chunks_exact_mut(t), &psi_inv[m..]).for_each(|(i, a, psi_inv)| {
            let (u, v) = a.split_at_mut(t / 2);
            if m == 0 && i == 0 {
                izip!(u, v).for_each(|(u, v)| twiddle_free_bufferfly(u, v));
            } else {
                izip!(u, v).for_each(|(u, v)| dif_bufferfly(u, v, psi_inv));
            }
        });
    }

    let n_inv = Fq::from_u64(a[0].q(), a.len() as u64).inv().unwrap();
    a.iter_mut().for_each(|a| *a *= n_inv);
}

#[inline(always)]
fn dit_bufferfly(a: &mut Fq, b: &mut Fq, twiddle: &Fq) {
    let tb = twiddle * *b;
    let c = *a + tb;
    let d = *a - tb;
    *a = c;
    *b = d;
}

#[inline(always)]
fn dif_bufferfly(a: &mut Fq, b: &mut Fq, twiddle: &Fq) {
    let c = *a + *b;
    let d = (*a - *b) * twiddle;
    *a = c;
    *b = d;
}

#[inline(always)]
fn twiddle_free_bufferfly(a: &mut Fq, b: &mut Fq) {
    let c = *a + *b;
    let d = *a - *b;
    *a = c;
    *b = d;
}

#[cfg(test)]
mod test {
    use crate::util::{
        fq::two_adic_primes,
        poly::{neg_schoolbook_mul, Poly},
    };
    use core::array::from_fn;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn neg_mul() {
        let mut rng = StdRng::from_entropy();
        for log_n in 0..10 {
            let n = 1 << log_n;
            for q in two_adic_primes(45, log_n + 1).take(10) {
                let [a, b] = &from_fn(|_| Poly::sample_fq_uniform(n, q, &mut rng));
                assert_eq!(a * b, neg_schoolbook_mul(a, b));
            }
        }
    }
}
