use crate::util::{fq::Fq, Dot};
use core::{
    borrow::Borrow,
    fmt::{self, Display, Formatter},
    iter::{repeat_with, Sum},
    ops::{AddAssign, Deref, DerefMut, Mul, MulAssign, Neg, SubAssign},
    slice,
};
use itertools::{izip, Itertools};
use rand::RngCore;
use rand_distr::Distribution;
use std::vec;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AVec<T>(Vec<T>);

impl<T> AVec<T> {
    pub fn sample(n: usize, dist: &impl Distribution<T>, rng: &mut impl RngCore) -> Self {
        repeat_with(|| dist.sample(rng)).take(n).collect()
    }
}

impl AVec<Fq> {
    pub fn sample_fq_uniform(n: usize, q: u64, rng: &mut impl RngCore) -> Self {
        repeat_with(|| Fq::sample_uniform(q, rng)).take(n).collect()
    }

    pub fn sample_fq_from_i8(
        n: usize,
        q: u64,
        dist: &impl Distribution<i8>,
        rng: &mut impl RngCore,
    ) -> Self {
        repeat_with(|| Fq::sample_i8(q, dist, rng))
            .take(n)
            .collect()
    }

    pub fn round_shr(&self, bits: usize) -> Self {
        self.iter().map(|v| v.round_shr(bits)).collect()
    }

    pub fn decompose(&self, log_b: usize, k: usize) -> impl Iterator<Item = Self> {
        let mut iters = self.iter().map(|v| v.decompose(log_b, k)).collect_vec();
        (0..k).map(move |_| iters.iter_mut().map(|iter| iter.next().unwrap()).collect())
    }
}

impl<T> Deref for AVec<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for AVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: Display> Display for AVec<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        if let Some((first, rest)) = self.split_first() {
            write!(f, "{first}")?;
            rest.iter().try_for_each(|value| write!(f, ", {}", value))?;
        }
        write!(f, "]")
    }
}

impl<T> From<AVec<T>> for Vec<T> {
    fn from(value: AVec<T>) -> Self {
        value.0
    }
}

impl<T> From<Vec<T>> for AVec<T> {
    fn from(value: Vec<T>) -> Self {
        Self(value)
    }
}

impl<'a, T: Clone> From<&'a [T]> for AVec<T> {
    fn from(value: &'a [T]) -> Self {
        Self(value.into())
    }
}

impl<T> FromIterator<T> for AVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<T> IntoIterator for AVec<T> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a AVec<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut AVec<T> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

impl<T> Neg for AVec<T>
where
    for<'t> &'t T: Neg<Output = T>,
{
    type Output = AVec<T>;

    fn neg(mut self) -> Self::Output {
        self.iter_mut().for_each(|value| *value = value.neg());
        self
    }
}

impl<T> Neg for &AVec<T>
where
    for<'t> &'t T: Neg<Output = T>,
{
    type Output = AVec<T>;

    fn neg(self) -> Self::Output {
        self.iter().map(|value| value.neg()).collect()
    }
}

impl<T> AddAssign<&AVec<T>> for AVec<T>
where
    for<'t> T: AddAssign<&'t T>,
{
    fn add_assign(&mut self, rhs: &AVec<T>) {
        assert_eq!(self.len(), rhs.len());
        izip!(&mut self.0, &rhs.0).for_each(|(lhs, rhs)| *lhs += rhs);
    }
}

impl<T> SubAssign<&AVec<T>> for AVec<T>
where
    for<'t> T: SubAssign<&'t T>,
{
    fn sub_assign(&mut self, rhs: &AVec<T>) {
        assert_eq!(self.len(), rhs.len());
        izip!(&mut self.0, &rhs.0).for_each(|(lhs, rhs)| *lhs -= rhs);
    }
}

impl<T> MulAssign<&T> for AVec<T>
where
    for<'t> T: MulAssign<&'t T>,
{
    fn mul_assign(&mut self, rhs: &T) {
        self.iter_mut().for_each(|value| *value *= rhs);
    }
}

impl<T, Item> Sum<Item> for AVec<T>
where
    T: Clone + for<'t> AddAssign<&'t T>,
    Item: Borrow<AVec<T>>,
{
    fn sum<I: Iterator<Item = Item>>(mut iter: I) -> Self {
        let init = iter.next().unwrap().borrow().clone();
        iter.fold(init, |acc, item| acc + item.borrow())
    }
}

impl<'a, T, I> Dot<I> for AVec<T>
where
    T: 'a + Sum,
    for<'t> &'t T: Mul<&'t T, Output = T>,
    I: IntoIterator<Item = &'a T>,
{
    type Output = T;

    fn dot(&self, rhs: I) -> Self::Output {
        izip!(&self.0, rhs).map(|(lhs, rhs)| lhs * rhs).sum()
    }
}

macro_rules! impl_ops {
    (@ impl<T$(: $generic:ty)?> $trait:ident<$rhs:ty> for $lhs:ty; type Output = $out:ty; $lhs_convert:expr) => {
        paste::paste! {
            impl<T $(: $generic)?> core::ops::$trait<$rhs> for $lhs
            where
                for<'t> T: [<$trait Assign>]<&'t T>,
            {
                type Output = $out;

                fn [<$trait:lower>](self, rhs: $rhs) -> $out {
                    let mut lhs = $lhs_convert(self);
                    lhs.[<$trait:lower _assign>](rhs.borrow());
                    lhs
                }
            }
        }
    };
    ($(impl<T> $trait:ident<$rhs:ty> for $lhs:ty),* $(,)?) => {
        $(
            paste::paste! {
                impl<T> core::ops::[<$trait Assign>]<$rhs> for $lhs
                where
                    for<'t> T: [<$trait Assign>]<&'t T>,
                {
                    fn [<$trait:lower _assign>](&mut self, rhs: $rhs) {
                        self.[<$trait:lower _assign>](&rhs);
                    }
                }
            }
            impl_ops!(@ impl<T> $trait<$rhs> for $lhs; type Output = $lhs; core::convert::identity);
            impl_ops!(@ impl<T> $trait<&$rhs> for $lhs; type Output = $lhs; core::convert::identity);
            impl_ops!(@ impl<T: Clone> $trait<$rhs> for &$lhs; type Output = $lhs; <_>::clone);
            impl_ops!(@ impl<T: Clone> $trait<&$rhs> for &$lhs; type Output = $lhs; <_>::clone);
        )*
    };
}

impl_ops!(
    impl<T> Add<AVec<T>> for AVec<T>,
    impl<T> Sub<AVec<T>> for AVec<T>,
    impl<T> Mul<T> for AVec<T>,
);

pub(crate) use impl_ops;
