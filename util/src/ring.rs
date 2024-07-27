use crate::{
    avec::{
        impl_element_wise_neg, impl_element_wise_op, impl_element_wise_op_assign,
        impl_mul_assign_element, impl_mul_element, AVec,
    },
    izip_eq,
    ring::{
        fft::{
            c64::nega_cyclic_fft64_mul_assign_rt,
            zq::{nega_cyclic_intt_in_place, nega_cyclic_ntt_in_place, nega_cyclic_ntt_mul_assign},
        },
        karatsuba::nega_cyclic_karatsuba_mul_assign,
    },
    torus::T64,
    zq::{impl_rest_op_by_op_assign_ref, is_prime, Zq},
};
use core::{
    borrow::Borrow,
    fmt::{self, Debug, Display, Formatter},
    iter::Sum,
    marker::PhantomData,
    ops::{AddAssign, BitXor, Mul, MulAssign, Neg},
    slice,
};
use derive_more::{AsMut, AsRef, Deref, DerefMut};
use itertools::Itertools;
use num_bigint::{BigInt, BigUint};
use rand::{distributions::Distribution, RngCore};
use std::vec;

pub mod fft;
pub mod karatsuba;
pub mod rns;

pub trait Basis: Clone + Copy + Debug + PartialEq + Eq {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Coefficient;

impl Basis for Coefficient {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Evaluation;

impl Basis for Evaluation {}

pub type Rq<B = Coefficient> = NegaCyclicRing<Zq, B>;
pub type Rt<B = Coefficient> = NegaCyclicRing<T64, B>;

#[derive(Clone, Debug, PartialEq, Eq, Deref, DerefMut, AsRef, AsMut)]
pub struct NegaCyclicRing<T, B: Basis = Coefficient>(
    #[deref]
    #[deref_mut]
    #[as_ref(forward)]
    #[as_mut(forward)]
    AVec<T>,
    PhantomData<B>,
);

impl<T, B: Basis> NegaCyclicRing<T, B> {
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

impl<T: Copy + Neg<Output = T>> NegaCyclicRing<T, Coefficient> {
    pub fn automorphism(&self, t: i64) -> Self {
        Self::new(self.0.automorphism(t))
    }
}

impl<B: Basis> NegaCyclicRing<Zq, B> {
    pub fn zero(q: u64, n: usize) -> Self {
        Self::new(vec![Zq::from_u64(q, 0); n].into())
    }

    pub fn sample_uniform(q: u64, n: usize, rng: &mut impl RngCore) -> Self {
        Self::new(AVec::<Zq>::sample_uniform(q, n, rng))
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

impl NegaCyclicRing<Zq, Coefficient> {
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

    pub fn to_evaluation(&self) -> NegaCyclicRing<Zq, Evaluation> {
        let mut a = self.0.clone();
        nega_cyclic_ntt_in_place(&mut a);
        NegaCyclicRing::new(a)
    }
}

impl<B: Basis> NegaCyclicRing<T64, B> {
    pub fn zero(n: usize) -> Self {
        Self::new(vec![T64::zero(); n].into())
    }

    pub fn sample_uniform(n: usize, rng: &mut impl RngCore) -> Self {
        Self::new(AVec::<T64>::sample_uniform(n, rng))
    }
}

impl NegaCyclicRing<T64, Coefficient> {
    pub fn one(n: usize) -> Self {
        let mut poly = Self::zero(n);
        poly[0] += 1;
        poly
    }

    pub fn constant(v: T64, n: usize) -> Self {
        let mut poly = Self::zero(n);
        poly[0] = v;
        poly
    }

    pub fn sample_i64(n: usize, dist: impl Distribution<i64>, rng: &mut impl RngCore) -> Self {
        Self::from_i64(&AVec::sample(n, dist, rng))
    }

    fn from_i64(v: &[i64]) -> Self {
        Self::new(v.iter().copied().map_into().collect())
    }
}

impl NegaCyclicRing<Zq, Evaluation> {
    pub fn to_coefficient(&self) -> NegaCyclicRing<Zq, Coefficient> {
        let mut a = self.0.clone();
        nega_cyclic_intt_in_place(&mut a);
        NegaCyclicRing::new(a)
    }
}

impl<T: Display> Display for NegaCyclicRing<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<T> From<NegaCyclicRing<T>> for Vec<T> {
    fn from(value: NegaCyclicRing<T>) -> Self {
        value.0.into()
    }
}

impl<T> From<Vec<T>> for NegaCyclicRing<T> {
    fn from(value: Vec<T>) -> Self {
        Self::new(value.into())
    }
}

impl<T: Clone> From<&[T]> for NegaCyclicRing<T> {
    fn from(value: &[T]) -> Self {
        Self::new(value.into())
    }
}

impl<T> From<NegaCyclicRing<T>> for AVec<T> {
    fn from(value: NegaCyclicRing<T>) -> Self {
        value.0
    }
}

impl<T> From<AVec<T>> for NegaCyclicRing<T> {
    fn from(value: AVec<T>) -> Self {
        Self::new(value)
    }
}

impl<T, B: Basis> FromIterator<T> for NegaCyclicRing<T, B> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

impl<T, B: Basis> IntoIterator for NegaCyclicRing<T, B> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T, B: Basis> IntoIterator for &'a NegaCyclicRing<T, B> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a, T, B: Basis> IntoIterator for &'a mut NegaCyclicRing<T, B> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

impl MulAssign<&NegaCyclicRing<Zq, Coefficient>> for NegaCyclicRing<Zq, Coefficient> {
    fn mul_assign(&mut self, rhs: &NegaCyclicRing<Zq, Coefficient>) {
        assert_eq!(self.n(), rhs.n());
        match is_prime(self.q()) {
            true => nega_cyclic_ntt_mul_assign(self, rhs),
            false => nega_cyclic_karatsuba_mul_assign::<Zq>(self, rhs),
        }
    }
}

impl MulAssign<&NegaCyclicRing<Zq, Evaluation>> for NegaCyclicRing<Zq, Evaluation> {
    fn mul_assign(&mut self, rhs: &NegaCyclicRing<Zq, Evaluation>) {
        izip_eq!(self, rhs).for_each(|(lhs, rhs)| *lhs *= rhs);
    }
}

impl MulAssign<&AVec<i64>> for NegaCyclicRing<Zq, Coefficient> {
    fn mul_assign(&mut self, rhs: &AVec<i64>) {
        *self *= Self::from_i64(self.q(), rhs);
    }
}

impl MulAssign<&AVec<i64>> for NegaCyclicRing<Zq, Evaluation> {
    fn mul_assign(&mut self, rhs: &AVec<i64>) {
        *self *= NegaCyclicRing::<Zq>::from_i64(self.q(), rhs).to_evaluation();
    }
}

impl MulAssign<&NegaCyclicRing<i64, Coefficient>> for NegaCyclicRing<i64, Coefficient> {
    fn mul_assign(&mut self, rhs: &NegaCyclicRing<i64, Coefficient>) {
        nega_cyclic_karatsuba_mul_assign::<i64>(self, rhs);
    }
}

impl<B: Basis> MulAssign<&BigUint> for NegaCyclicRing<Zq, B>
where
    Self: MulAssign<Zq>,
{
    fn mul_assign(&mut self, rhs: &BigUint) {
        *self *= Zq::from_biguint(self.q(), rhs);
    }
}

impl<T> MulAssign<&Monomial> for NegaCyclicRing<T>
where
    for<'t> &'t T: Neg<Output = T>,
{
    fn mul_assign(&mut self, rhs: &Monomial) {
        let n = self.n();
        let i = rhs.0.rem_euclid(2 * n as i64) as usize;
        self.rotate_right(i % n);
        if i < n {
            self[..i].iter_mut().for_each(|v| *v = -&*v);
        } else {
            self[i - n..].iter_mut().for_each(|v| *v = -&*v);
        }
    }
}

impl MulAssign<&NegaCyclicRing<T64, Coefficient>> for NegaCyclicRing<T64, Coefficient> {
    fn mul_assign(&mut self, rhs: &NegaCyclicRing<T64, Coefficient>) {
        assert_eq!(self.n(), rhs.n());
        nega_cyclic_fft64_mul_assign_rt(self, rhs)
    }
}

impl MulAssign<&AVec<i64>> for NegaCyclicRing<T64, Coefficient> {
    fn mul_assign(&mut self, rhs: &AVec<i64>) {
        *self *= Self::from_i64(rhs);
    }
}

impl<T, B, Item> Sum<Item> for NegaCyclicRing<T, B>
where
    T: Clone + for<'t> AddAssign<&'t T>,
    B: Basis,
    Item: Borrow<NegaCyclicRing<T, B>>,
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
    impl<T> Neg for NegaCyclicRing<T, Coefficient>,
    impl<T> Neg for NegaCyclicRing<T, Evaluation>,
);
impl_element_wise_op_assign!(
    impl<T> AddAssign<NegaCyclicRing<T, Coefficient>> for NegaCyclicRing<T, Coefficient>,
    impl<T> AddAssign<NegaCyclicRing<T, Evaluation>> for NegaCyclicRing<T, Evaluation>,
    impl<T> SubAssign<NegaCyclicRing<T, Coefficient>> for NegaCyclicRing<T, Coefficient>,
    impl<T> SubAssign<NegaCyclicRing<T, Evaluation>> for NegaCyclicRing<T, Evaluation>,
);
impl_element_wise_op!(
    impl<T> Add<NegaCyclicRing<T, Coefficient>> for NegaCyclicRing<T, Coefficient>,
    impl<T> Add<NegaCyclicRing<T, Evaluation>> for NegaCyclicRing<T, Evaluation>,
    impl<T> Sub<NegaCyclicRing<T, Coefficient>> for NegaCyclicRing<T, Coefficient>,
    impl<T> Sub<NegaCyclicRing<T, Evaluation>> for NegaCyclicRing<T, Evaluation>,
);
impl_mul_assign_element!(
    impl<T> MulAssign<T> for NegaCyclicRing<T, Coefficient>,
    impl<T> MulAssign<T> for NegaCyclicRing<T, Evaluation>,
);
impl_mul_element!(
    impl<T> Mul<T> for NegaCyclicRing<T, Coefficient>,
    impl<T> Mul<T> for NegaCyclicRing<T, Evaluation>,
);
impl_rest_op_by_op_assign_ref!(
    impl Mul<NegaCyclicRing<i64, Coefficient>> for NegaCyclicRing<i64, Coefficient>,
    impl Mul<NegaCyclicRing<Zq, Coefficient>> for NegaCyclicRing<Zq, Coefficient>,
    impl Mul<NegaCyclicRing<Zq, Evaluation>> for NegaCyclicRing<Zq, Evaluation>,
    impl Mul<AVec<i64>> for NegaCyclicRing<Zq, Coefficient>,
    impl Mul<AVec<i64>> for NegaCyclicRing<Zq, Evaluation>,
    impl Mul<Monomial> for NegaCyclicRing<Zq>,
    impl Mul<BigUint> for NegaCyclicRing<Zq>,
    impl Mul<NegaCyclicRing<T64, Coefficient>> for NegaCyclicRing<T64, Coefficient>,
    impl Mul<AVec<i64>> for NegaCyclicRing<T64, Coefficient>,
    impl Mul<Monomial> for NegaCyclicRing<T64>,
);

pub struct Monomial(i64);

pub struct X;

impl BitXor<i64> for X {
    type Output = Monomial;

    fn bitxor(self, rhs: i64) -> Self::Output {
        Monomial(rhs)
    }
}

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

#[cfg(test)]
pub mod test {
    use crate::{
        ring::{Coefficient, Rq},
        zq::two_adic_primes,
    };
    use core::{
        array::from_fn,
        ops::{AddAssign, Mul, SubAssign},
    };
    use itertools::izip;
    use rand::thread_rng;

    pub fn nega_cyclic_schoolbook_mul<T, V>(a: &V, b: &V) -> V
    where
        T: AddAssign<T> + SubAssign<T>,
        for<'t> &'t T: Mul<&'t T, Output = T>,
        V: AsRef<[T]> + AsMut<[T]> + FromIterator<T>,
    {
        let (a, b) = (a.as_ref(), b.as_ref());
        let n = a.len();
        let mut c = a.iter().map(|a| a * &b[0]).collect::<V>();
        izip!(0.., a.iter()).for_each(|(i, a)| {
            izip!(0.., b.iter()).skip(1).for_each(|(j, b)| {
                if i + j < n {
                    c.as_mut()[i + j] += a * b;
                } else {
                    c.as_mut()[i + j - n] -= a * b;
                }
            })
        });
        c
    }

    #[test]
    fn nega_cyclic_mul() {
        let mut rng = thread_rng();
        for log_n in 0..10 {
            let n = 1 << log_n;
            for q in two_adic_primes(45, log_n + 1).take(10) {
                let [a, b] = &from_fn(|_| Rq::<Coefficient>::sample_uniform(q, n, &mut rng));
                assert_eq!(a * b, nega_cyclic_schoolbook_mul(a, b));
            }
        }
    }
}
