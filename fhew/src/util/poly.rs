use crate::util::{
    avec::{impl_ops, AVec},
    fq::{Fq, NEG_NTT_PSI},
};
use core::{
    borrow::Borrow,
    fmt::{self, Display, Formatter},
    iter::Sum,
    ops::{AddAssign, BitXor, Deref, DerefMut, Mul, MulAssign, Neg, SubAssign},
    slice,
};
use itertools::izip;
use rand::RngCore;
use rand_distr::Distribution;
use std::vec;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poly<T>(AVec<T>);

impl<T> Poly<T> {
    fn new(v: AVec<T>) -> Self {
        assert!(v.len().is_power_of_two());
        Self(v)
    }

    pub fn n(&self) -> usize {
        self.len()
    }

    pub fn sample(n: usize, dist: &impl Distribution<T>, rng: &mut impl RngCore) -> Self {
        Self::new(AVec::sample(n, dist, rng))
    }

    pub fn automorphism(&self, t: i64) -> Self
    where
        T: Copy + Neg<Output = T>,
    {
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

impl Poly<Fq> {
    pub fn zero(n: usize, q: u64) -> Self {
        Self::new(vec![Fq::from_u64(q, 0); n].into())
    }

    pub fn one(n: usize, q: u64) -> Self {
        let mut poly = Self::zero(n, q);
        poly[0] += 1;
        poly
    }

    pub fn monomial(n: usize, q: u64, i: i64) -> Self {
        let mut poly = Self::zero(n, q);
        let i = i.rem_euclid(2 * n as i64) as usize;
        if i < n {
            poly[i] += 1;
        } else {
            poly[i - n] -= 1;
        }
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
}

impl<T> Deref for Poly<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Poly<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
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

impl<'a, T: Clone> From<&'a [T]> for Poly<T> {
    fn from(value: &'a [T]) -> Self {
        Self::new(value.into())
    }
}

impl<T: Clone> From<AVec<T>> for Poly<T> {
    fn from(value: AVec<T>) -> Self {
        Self::new(value)
    }
}

impl<T> Extend<T> for Poly<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.0.extend(iter)
    }
}

impl<T> FromIterator<T> for Poly<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

impl<T> IntoIterator for Poly<T> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a Poly<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Poly<T> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

impl<T> Neg for Poly<T>
where
    for<'t> &'t T: Neg<Output = T>,
{
    type Output = Poly<T>;

    fn neg(self) -> Self::Output {
        Poly(-self.0)
    }
}

impl<T> Neg for &Poly<T>
where
    for<'t> &'t T: Neg<Output = T>,
{
    type Output = Poly<T>;

    fn neg(self) -> Self::Output {
        Poly(-&self.0)
    }
}

impl<T> AddAssign<&Poly<T>> for Poly<T>
where
    for<'t> T: AddAssign<&'t T>,
{
    fn add_assign(&mut self, rhs: &Poly<T>) {
        self.0 += &rhs.0;
    }
}

impl<T> SubAssign<&Poly<T>> for Poly<T>
where
    for<'t> T: SubAssign<&'t T>,
{
    fn sub_assign(&mut self, rhs: &Poly<T>) {
        self.0 -= &rhs.0;
    }
}

impl<T> MulAssign<&T> for Poly<T>
where
    for<'t> T: MulAssign<&'t T>,
{
    fn mul_assign(&mut self, rhs: &T) {
        self.0 *= rhs;
    }
}

impl MulAssign<&Poly<Fq>> for Poly<Fq> {
    fn mul_assign(&mut self, rhs: &Poly<Fq>) {
        assert_eq!(self.len(), rhs.len());

        match NEG_NTT_PSI
            .get_or_init(Default::default)
            .lock()
            .unwrap()
            .get(&self[0].q())
        {
            Some([psi, psi_inv]) if self.n() <= psi.len() => {
                neg_ntt_mul_assign(self, rhs, psi, psi_inv)
            }
            _ => *self = neg_schoolbook_mul(self, rhs),
        }
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

impl<T, Item> Sum<Item> for Poly<T>
where
    T: Clone + for<'t> AddAssign<&'t T>,
    Item: Borrow<Poly<T>>,
{
    fn sum<I: Iterator<Item = Item>>(mut iter: I) -> Self {
        let init = iter.next().unwrap().borrow().0.clone();
        Self(iter.fold(init, |acc, item| acc + &item.borrow().0))
    }
}

impl_ops!(
    impl<T> Add<Poly<T>> for Poly<T>,
    impl<T> Sub<Poly<T>> for Poly<T>,
    impl<T> Mul<T> for Poly<T>,
);

macro_rules! impl_mul_ops {
    (@ impl Mul<$rhs:ty> for $lhs:ty; $lhs_convert:expr) => {
        impl core::ops::Mul<$rhs> for $lhs {
            type Output = Poly<Fq>;

            fn mul(self, rhs: $rhs) -> Poly<Fq> {
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
            impl_mul_ops!(@ impl Mul<$rhs> for $lhs; core::convert::identity);
            impl_mul_ops!(@ impl Mul<&$rhs> for $lhs; core::convert::identity);
            impl_mul_ops!(@ impl Mul<$rhs> for &$lhs; <_>::clone);
            impl_mul_ops!(@ impl Mul<&$rhs> for &$lhs; <_>::clone);
        )*
    }
}

impl_mul_ops!(
    impl Mul<Poly<Fq>> for Poly<Fq>,
    impl Mul<Poly<i8>> for Poly<Fq>,
    impl Mul<Monomial> for Poly<Fq>,
);

macro_rules! impl_mul_ops_from_i8 {
    (@ impl Mul<$rhs:ty> for $lhs:ty) => {
        impl core::ops::Mul<$rhs> for $lhs {
            type Output = Poly<Fq>;

            fn mul(self, rhs: $rhs) -> Poly<Fq> {
                self.borrow().mul(rhs.borrow())
            }
        }
    };
    ($rhs:ty) => {
        impl_mul_ops_from_i8!(@ impl Mul<$rhs> for Poly<i8>);
        impl_mul_ops_from_i8!(@ impl Mul<&$rhs> for Poly<i8>);
        impl_mul_ops_from_i8!(@ impl Mul<$rhs> for &Poly<i8>);
    }
}

impl_mul_ops_from_i8!(Fq);

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
