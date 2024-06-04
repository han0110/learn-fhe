use crate::util::{
    avec::{impl_ops, AVec},
    fq::Fq,
};
use core::{
    borrow::Borrow,
    fmt::{self, Display, Formatter},
    iter::Sum,
    ops::{AddAssign, Deref, DerefMut, MulAssign, Neg, ShlAssign, ShrAssign, SubAssign},
    slice,
};
use itertools::izip;
use rand::RngCore;
use rand_distr::Distribution;
use std::vec;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poly<T>(AVec<T>);

impl<T> Poly<T> {
    pub fn sample(n: usize, dist: &impl Distribution<T>, rng: &mut impl RngCore) -> Self {
        assert!(n.is_power_of_two());
        Self(AVec::sample(n, dist, rng))
    }

    pub fn n(&self) -> usize {
        self.len()
    }
}

impl Poly<Fq> {
    pub fn sample_uniform(n: usize, q: u64, rng: &mut impl RngCore) -> Self {
        assert!(n.is_power_of_two());
        Self(AVec::sample_uniform(n, q, rng))
    }

    pub fn sample_i8(
        n: usize,
        q: u64,
        dist: &impl Distribution<i8>,
        rng: &mut impl RngCore,
    ) -> Self {
        assert!(n.is_power_of_two());
        Self(AVec::sample_i8(n, q, dist, rng))
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
        Self(value.into())
    }
}

impl<'a, T: Clone> From<&'a [T]> for Poly<T> {
    fn from(value: &'a [T]) -> Self {
        Self(value.into())
    }
}

impl<T> FromIterator<T> for Poly<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
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

impl<T: Clone> MulAssign<&Poly<T>> for Poly<T>
where
    for<'t> T: AddAssign<&'t T> + SubAssign<&'t T> + MulAssign<&'t T>,
{
    fn mul_assign(&mut self, rhs: &Poly<T>) {
        assert_eq!(self.len(), rhs.len());
        let mut out = &self.0 * &rhs[0];
        izip!(0.., self.iter()).for_each(|(i, lhs)| {
            izip!(0.., rhs.iter()).skip(1).for_each(|(j, rhs)| {
                let mut t = lhs.clone();
                t *= rhs;
                if i + j < self.n() {
                    out[i + j] += &t;
                } else {
                    out[i + j - self.n()] -= &t;
                }
            })
        });
        self.0 = out;
    }
}

impl<T: Clone> MulAssign<Poly<T>> for Poly<T>
where
    for<'t> T: AddAssign<&'t T> + SubAssign<&'t T> + MulAssign<&'t T>,
{
    fn mul_assign(&mut self, rhs: Poly<T>) {
        *self *= &rhs;
    }
}

impl<T> ShlAssign<&usize> for Poly<T>
where
    for<'t> T: ShlAssign<&'t usize>,
{
    fn shl_assign(&mut self, rhs: &usize) {
        self.0 <<= rhs;
    }
}

impl<T> ShrAssign<&usize> for Poly<T>
where
    for<'t> T: ShrAssign<&'t usize>,
{
    fn shr_assign(&mut self, rhs: &usize) {
        self.0 >>= rhs;
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
    impl<T> Add<Poly<T>> for Poly<T>; T,
    impl<T> Sub<Poly<T>> for Poly<T>; T,
    impl<T> Mul<T> for Poly<T>; T,
    impl<T> Shl<usize> for Poly<T>; usize,
    impl<T> Shr<usize> for Poly<T>; usize,
);

macro_rules! impl_mul_ops {
    (impl<T> Mul<$rhs:ty> for $lhs:ty; $lhs_convert:expr) => {
        impl<T: Clone> core::ops::Mul<$rhs> for $lhs
        where
            for<'t> T: AddAssign<&'t T> + SubAssign<&'t T> + MulAssign<&'t T>,
        {
            type Output = Poly<T>;

            fn mul(self, rhs: $rhs) -> Poly<T> {
                let mut lhs = $lhs_convert(self);
                lhs.mul_assign(rhs.borrow());
                lhs
            }
        }
    };
}

impl_mul_ops!(impl<T> Mul<Poly<T>> for Poly<T>; core::convert::identity);
impl_mul_ops!(impl<T> Mul<&Poly<T>> for Poly<T>; core::convert::identity);
impl_mul_ops!(impl<T> Mul<Poly<T>> for &Poly<T>; <_>::clone);
impl_mul_ops!(impl<T> Mul<&Poly<T>> for &Poly<T>; <_>::clone);
