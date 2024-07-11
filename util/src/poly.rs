use crate::{
    avec::{
        impl_element_wise_neg, impl_element_wise_op, impl_element_wise_op_assign,
        impl_mul_assign_element, impl_mul_element, AVec,
    },
    izip_eq,
    misc::Butterfly,
    zq::{impl_rest_op_by_op_assign_ref, twiddle, Zq},
};
use core::{
    borrow::Borrow,
    fmt::{self, Debug, Display, Formatter},
    iter::Sum,
    marker::PhantomData,
    ops::{AddAssign, BitXor, Mul, MulAssign, Neg, SubAssign},
    slice,
};
use derive_more::{Deref, DerefMut};
use itertools::izip;
use num_bigint::{BigInt, BigUint};
use rand::{distributions::Distribution, RngCore};
use std::vec;

pub mod rns;

pub trait Basis: Clone + Copy + Debug + PartialEq + Eq {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Coefficient;

impl Basis for Coefficient {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Evaluation;

impl Basis for Evaluation {}

pub type Rq<B = Coefficient> = NegaCyclicPoly<Zq, B>;

#[derive(Clone, Debug, PartialEq, Eq, Deref, DerefMut)]
pub struct NegaCyclicPoly<T, B: Basis = Coefficient>(
    #[deref]
    #[deref_mut]
    AVec<T>,
    PhantomData<B>,
);

impl<T, B: Basis> NegaCyclicPoly<T, B> {
    fn new(v: AVec<T>) -> Self {
        assert!(v.len().is_power_of_two());
        Self(v, PhantomData)
    }

    pub fn n(&self) -> usize {
        self.len()
    }

    pub fn sample(n: usize, dist: impl Distribution<T>, rng: &mut impl RngCore) -> Self {
        Self::new(AVec::sample(n, dist, rng))
    }

    pub fn square(&self) -> Self
    where
        for<'t> &'t Self: Mul<&'t Self, Output = Self>,
    {
        self * self
    }
}

impl<T: Copy + Neg<Output = T>> NegaCyclicPoly<T, Coefficient> {
    pub fn automorphism(&self, t: i64) -> Self {
        Self::new(self.0.automorphism(t))
    }
}

impl<B: Basis> NegaCyclicPoly<Zq, B> {
    pub fn zero(q: u64, n: usize) -> Self {
        Self::new(vec![Zq::from_u64(q, 0); n].into())
    }

    pub fn sample_uniform(q: u64, n: usize, rng: &mut impl RngCore) -> Self {
        Self::new(AVec::sample_uniform(q, n, rng))
    }

    pub fn mod_switch(&self, q_prime: u64) -> Self {
        Self::new(self.0.mod_switch(q_prime))
    }

    pub fn mod_switch_odd(&self, q_prime: u64) -> Self {
        Self::new(self.0.mod_switch_odd(q_prime))
    }

    pub fn q(&self) -> u64 {
        self[0].q()
    }

    fn from_bigint(q: u64, v: &[BigInt]) -> Self {
        Self::new(v.iter().map(|v| Zq::from_bigint(q, v)).collect())
    }
}

impl NegaCyclicPoly<Zq, Coefficient> {
    pub fn one(q: u64, n: usize) -> Self {
        let mut poly = Self::zero(q, n);
        poly[0] += 1;
        poly
    }

    pub fn constant(v: Zq, n: usize) -> Self {
        let mut poly = Self::zero(v.q(), n);
        poly[0] = v;
        poly
    }

    pub fn sample_i64(
        q: u64,
        n: usize,
        dist: impl Distribution<i64>,
        rng: &mut impl RngCore,
    ) -> Self {
        Self::from_i64(q, &AVec::sample(n, dist, rng))
    }

    fn from_i64(q: u64, v: &[i64]) -> Self {
        Self::new(v.iter().map(|v| Zq::from_i64(q, *v)).collect())
    }

    pub fn to_evaluation(&self) -> NegaCyclicPoly<Zq, Evaluation> {
        let [psi, _] = &*twiddle(self.q());
        let mut a = self.0.clone();
        nega_cyclic_ntt_in_place(&mut a, psi);
        NegaCyclicPoly::new(a)
    }
}

impl NegaCyclicPoly<Zq, Evaluation> {
    pub fn to_coefficient(&self) -> NegaCyclicPoly<Zq, Coefficient> {
        let [_, psi_inv] = &*twiddle(self.q());
        let mut a = self.0.clone();
        nega_cyclic_intt_in_place(&mut a, psi_inv);
        NegaCyclicPoly::new(a)
    }
}

impl<T: Display> Display for NegaCyclicPoly<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<T> From<NegaCyclicPoly<T>> for Vec<T> {
    fn from(value: NegaCyclicPoly<T>) -> Self {
        value.0.into()
    }
}

impl<T> From<Vec<T>> for NegaCyclicPoly<T> {
    fn from(value: Vec<T>) -> Self {
        Self::new(value.into())
    }
}

impl<T: Clone> From<&[T]> for NegaCyclicPoly<T> {
    fn from(value: &[T]) -> Self {
        Self::new(value.into())
    }
}

impl<T> From<NegaCyclicPoly<T>> for AVec<T> {
    fn from(value: NegaCyclicPoly<T>) -> Self {
        value.0
    }
}

impl<T> From<AVec<T>> for NegaCyclicPoly<T> {
    fn from(value: AVec<T>) -> Self {
        Self::new(value)
    }
}

impl<T, B: Basis> FromIterator<T> for NegaCyclicPoly<T, B> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

impl<T, B: Basis> IntoIterator for NegaCyclicPoly<T, B> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T, B: Basis> IntoIterator for &'a NegaCyclicPoly<T, B> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a, T, B: Basis> IntoIterator for &'a mut NegaCyclicPoly<T, B> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

impl MulAssign<&NegaCyclicPoly<Zq, Coefficient>> for NegaCyclicPoly<Zq, Coefficient> {
    fn mul_assign(&mut self, rhs: &NegaCyclicPoly<Zq, Coefficient>) {
        assert_eq!(self.n(), rhs.n());
        match twiddle(self.q()).get() {
            Some([psi, psi_inv]) if self.n() <= psi.len() => {
                nega_cyclic_ntt_mul_assign(self, rhs, psi, psi_inv)
            }
            _ => *self = nega_cyclic_schoolbook_mul(self, rhs),
        }
    }
}

impl MulAssign<&NegaCyclicPoly<Zq, Evaluation>> for NegaCyclicPoly<Zq, Evaluation> {
    fn mul_assign(&mut self, rhs: &NegaCyclicPoly<Zq, Evaluation>) {
        izip_eq!(self, rhs).for_each(|(lhs, rhs)| *lhs *= rhs);
    }
}

impl MulAssign<&AVec<i64>> for NegaCyclicPoly<Zq, Coefficient> {
    fn mul_assign(&mut self, rhs: &AVec<i64>) {
        *self *= Self::from_i64(self.q(), rhs);
    }
}

impl MulAssign<&AVec<i64>> for NegaCyclicPoly<Zq, Evaluation> {
    fn mul_assign(&mut self, rhs: &AVec<i64>) {
        *self *= NegaCyclicPoly::from_i64(self.q(), rhs).to_evaluation();
    }
}

impl MulAssign<&NegaCyclicPoly<i64, Coefficient>> for NegaCyclicPoly<i64, Coefficient> {
    fn mul_assign(&mut self, rhs: &NegaCyclicPoly<i64, Coefficient>) {
        *self = nega_cyclic_schoolbook_mul(self, rhs);
    }
}

impl<B: Basis> MulAssign<&BigUint> for NegaCyclicPoly<Zq, B>
where
    Self: MulAssign<Zq>,
{
    fn mul_assign(&mut self, rhs: &BigUint) {
        *self *= Zq::from_biguint(self.q(), rhs);
    }
}

impl MulAssign<&Monomial> for NegaCyclicPoly<Zq> {
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

impl<T, B, Item> Sum<Item> for NegaCyclicPoly<T, B>
where
    T: Clone + for<'t> AddAssign<&'t T>,
    B: Basis,
    Item: Borrow<NegaCyclicPoly<T, B>>,
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
    impl<T> Neg for NegaCyclicPoly<T, Coefficient>,
    impl<T> Neg for NegaCyclicPoly<T, Evaluation>,
);
impl_element_wise_op_assign!(
    impl<T> AddAssign<NegaCyclicPoly<T, Coefficient>> for NegaCyclicPoly<T, Coefficient>,
    impl<T> AddAssign<NegaCyclicPoly<T, Evaluation>> for NegaCyclicPoly<T, Evaluation>,
    impl<T> SubAssign<NegaCyclicPoly<T, Coefficient>> for NegaCyclicPoly<T, Coefficient>,
    impl<T> SubAssign<NegaCyclicPoly<T, Evaluation>> for NegaCyclicPoly<T, Evaluation>,
);
impl_element_wise_op!(
    impl<T> Add<NegaCyclicPoly<T, Coefficient>> for NegaCyclicPoly<T, Coefficient>,
    impl<T> Add<NegaCyclicPoly<T, Evaluation>> for NegaCyclicPoly<T, Evaluation>,
    impl<T> Sub<NegaCyclicPoly<T, Coefficient>> for NegaCyclicPoly<T, Coefficient>,
    impl<T> Sub<NegaCyclicPoly<T, Evaluation>> for NegaCyclicPoly<T, Evaluation>,
);
impl_mul_assign_element!(
    impl<T> MulAssign<T> for NegaCyclicPoly<T, Coefficient>,
    impl<T> MulAssign<T> for NegaCyclicPoly<T, Evaluation>,
);
impl_mul_element!(
    impl<T> Mul<T> for NegaCyclicPoly<T, Coefficient>,
    impl<T> Mul<T> for NegaCyclicPoly<T, Evaluation>,
);
impl_rest_op_by_op_assign_ref!(
    impl Mul<NegaCyclicPoly<Zq, Coefficient>> for NegaCyclicPoly<Zq, Coefficient>,
    impl Mul<NegaCyclicPoly<Zq, Evaluation>> for NegaCyclicPoly<Zq, Evaluation>,
    impl Mul<NegaCyclicPoly<i64, Coefficient>> for NegaCyclicPoly<i64, Coefficient>,
    impl Mul<AVec<i64>> for NegaCyclicPoly<Zq>,
    impl Mul<BigUint> for NegaCyclicPoly<Zq>,
    impl Mul<Monomial> for NegaCyclicPoly<Zq>,
);

pub struct Monomial(i64);

pub struct X;

impl BitXor<&i64> for X {
    type Output = Monomial;

    fn bitxor(self, rhs: &i64) -> Self::Output {
        Monomial(*rhs)
    }
}

impl BitXor<Zq> for X {
    type Output = Monomial;

    fn bitxor(self, rhs: Zq) -> Self::Output {
        Monomial(rhs.into())
    }
}

fn nega_cyclic_schoolbook_mul<T>(a: &NegaCyclicPoly<T>, b: &NegaCyclicPoly<T>) -> NegaCyclicPoly<T>
where
    T: AddAssign<T> + SubAssign<T>,
    for<'t> &'t T: Mul<&'t T, Output = T>,
{
    let n = a.n();
    let mut c = a.iter().map(|a| a * &b[0]).collect::<NegaCyclicPoly<_>>();
    izip!(0.., a.iter()).for_each(|(i, a)| {
        izip!(0.., b.iter()).skip(1).for_each(|(j, b)| {
            if i + j < n {
                c[i + j] += a * b;
            } else {
                c[i + j - n] -= a * b;
            }
        })
    });
    c
}

fn nega_cyclic_ntt_mul_assign(
    a: &mut NegaCyclicPoly<Zq>,
    b: &NegaCyclicPoly<Zq>,
    psi: &[Zq],
    psi_inv: &[Zq],
) {
    let b = &mut b.clone();
    nega_cyclic_ntt_in_place(a, psi);
    nega_cyclic_ntt_in_place(b, psi);
    izip!(a.iter_mut(), b.iter()).for_each(|(a, b)| *a *= b);
    nega_cyclic_intt_in_place(a, psi_inv);
}

// Algorithm 1 in 2016/504.
fn nega_cyclic_ntt_in_place(a: &mut [Zq], psi: &[Zq]) {
    assert!(a.len().is_power_of_two());
    for log_m in 0..a.len().ilog2() {
        let m = 1 << log_m;
        let t = a.len() / m;
        izip!(0.., a.chunks_exact_mut(t), &psi[m..]).for_each(|(i, a, psi)| {
            let (u, v) = a.split_at_mut(t / 2);
            if m == 0 && i == 0 {
                izip!(u, v).for_each(|(u, v)| Butterfly::twiddle_free(u, v));
            } else {
                izip!(u, v).for_each(|(u, v)| Butterfly::dit(u, v, psi));
            }
        });
    }
}

// Algorithm 2 in 2016/504.
fn nega_cyclic_intt_in_place(a: &mut [Zq], psi_inv: &[Zq]) {
    assert!(a.len().is_power_of_two());
    for log_m in (0..a.len().ilog2()).rev() {
        let m = 1 << log_m;
        let t = a.len() / m;
        izip!(0.., a.chunks_exact_mut(t), &psi_inv[m..]).for_each(|(i, a, psi_inv)| {
            let (u, v) = a.split_at_mut(t / 2);
            if m == 0 && i == 0 {
                izip!(u, v).for_each(|(u, v)| Butterfly::twiddle_free(u, v));
            } else {
                izip!(u, v).for_each(|(u, v)| Butterfly::dif(u, v, psi_inv));
            }
        });
    }
    let n_inv = Zq::from_u64(a[0].q(), a.len() as u64).inv().unwrap();
    a.iter_mut().for_each(|a| *a *= n_inv);
}

#[cfg(test)]
mod test {
    use crate::{
        poly::{nega_cyclic_schoolbook_mul, NegaCyclicPoly},
        zq::two_adic_primes,
    };
    use core::array::from_fn;
    use rand::thread_rng;

    #[test]
    fn nega_cyclic_mul() {
        let mut rng = thread_rng();
        for log_n in 0..10 {
            let n = 1 << log_n;
            for q in two_adic_primes(45, log_n + 1).take(10) {
                let [a, b] = &from_fn(|_| NegaCyclicPoly::sample_uniform(q, n, &mut rng));
                assert_eq!(a * b, nega_cyclic_schoolbook_mul(a, b));
            }
        }
    }
}
