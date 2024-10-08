use crate::{complex::C256, zq::Zq, T64};
use core::{
    borrow::Borrow,
    fmt::{self, Display, Formatter},
    iter::{repeat, repeat_with, Sum},
    ops::{AddAssign, Neg},
    slice,
};
use derive_more::{AsMut, AsRef, Deref, DerefMut, From, Into};
use itertools::chain;
use rand::{distributions::Distribution, RngCore};
use std::vec;

#[derive(Clone, Debug, PartialEq, Eq, Deref, DerefMut, From, Into, AsRef, AsMut)]
#[as_ref(forward)]
#[as_mut(forward)]
pub struct AVec<T>(Vec<T>);

impl<T> AVec<T> {
    pub fn broadcast(n: usize, v: impl Clone + IntoIterator<Item = T>) -> Self {
        repeat(v).flatten().take(n).collect()
    }

    pub fn sample(n: usize, dist: impl Distribution<T>, rng: &mut impl RngCore) -> Self {
        repeat_with(|| dist.sample(rng)).take(n).collect()
    }

    pub fn rot_iter(&self, j: i64) -> impl Iterator<Item = &T> {
        let (left, right) = self.split_at(j.rem_euclid(self.len() as _) as _);
        chain![right, left]
    }
}

impl<T: Clone + Neg<Output = T>> AVec<T> {
    pub fn automorphism(&self, t: i64) -> Self {
        assert!(self.len().is_power_of_two());
        let mut v = self.clone();
        let n = self.len();
        let t = t.rem_euclid(2 * n as i64) as usize;
        (0..n).for_each(|i| {
            let it = (i * t) % (2 * n);
            if it < n {
                v[it] = self[i].clone();
            } else {
                v[it - n] = -self[i].clone();
            }
        });
        v
    }
}

impl AVec<Zq> {
    pub fn zero(q: u64, n: usize) -> Self {
        Self(vec![Zq::from_u64(q, 0); n])
    }

    pub fn sample_uniform(q: u64, n: usize, rng: &mut impl RngCore) -> Self {
        repeat_with(|| Zq::sample_uniform(q, rng)).take(n).collect()
    }

    pub fn mod_switch(&self, q_prime: u64) -> Self {
        self.iter().map(|v| v.mod_switch(q_prime)).collect()
    }

    pub fn mod_switch_odd(&self, q_prime: u64) -> Self {
        self.iter().map(|v| v.mod_switch_odd(q_prime)).collect()
    }

    pub fn q(&self) -> u64 {
        self[0].q()
    }
}

impl AVec<T64> {
    pub fn zero(n: usize) -> Self {
        Self(vec![T64::default(); n])
    }

    pub fn sample_uniform(n: usize, rng: &mut impl RngCore) -> Self {
        repeat_with(|| T64::sample_uniform(rng)).take(n).collect()
    }
}

impl AVec<C256> {
    pub fn conjugate(&self) -> Self {
        self.iter().map(C256::conj).collect()
    }
}

impl<T> Default for AVec<T> {
    fn default() -> Self {
        Self(Vec::new())
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

impl<T: Clone> From<&[T]> for AVec<T> {
    fn from(value: &[T]) -> Self {
        Self(value.into())
    }
}

impl<T> Extend<T> for AVec<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.0.extend(iter)
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

impl<T, Item> Sum<Item> for AVec<T>
where
    T: Clone + for<'t> AddAssign<&'t T>,
    Item: Borrow<AVec<T>>,
{
    fn sum<I: Iterator<Item = Item>>(mut iter: I) -> Self {
        let init = iter.next().unwrap().borrow().clone();
        iter.fold(init, |mut acc, item| {
            acc += item.borrow();
            acc
        })
    }
}

macro_rules! impl_element_wise_op_assign {
    (@ impl<T> $trait:ident<$rhs:ty> for $lhs:ty; for<$life:lifetime> $t:ty: $constraint:tt) => {
        paste::paste! {
            impl<T> core::ops::$trait<$rhs> for $lhs
            where
                for<$life> $t: $constraint,
            {
                fn [<$trait:snake:lower>](&mut self, rhs: $rhs) {
                    $crate::izip_eq!(self, rhs).for_each(|(lhs, rhs)| lhs.[<$trait:snake:lower>](rhs));
                }
            }
        }
    };
    ($(impl<T> $trait:ident<$rhs:ty> for $lhs:ty),* $(,)?) => {
        $(
            impl_element_wise_op_assign!(@ impl<T> $trait<&$rhs> for $lhs; for<'t> T: (core::ops::$trait<&'t T>));
            impl_element_wise_op_assign!(@ impl<T> $trait<$rhs> for $lhs; for<'t> T: (core::ops::$trait<T>));
        )*
    }
}

macro_rules! impl_element_wise_neg {
    (@ impl<T> Neg for $lhs:ty; type Output = $out:ty; for<$life:lifetime> $t:ty: $constraint:tt) => {
        impl<T> core::ops::Neg for $lhs
        where
            for<$life> $t: $constraint,
        {
            type Output = $out;

            fn neg(self) -> Self::Output {
                self.into_iter().map(|value| -value).collect()
            }
        }
    };
    ($(impl<T> Neg for $lhs:ty),* $(,)?) => {
        $(
            impl_element_wise_neg!(@ impl<T> Neg for $lhs; type Output = $lhs; for<'t> T: (core::ops::Neg<Output = T>));
            impl_element_wise_neg!(@ impl<T> Neg for &$lhs; type Output = $lhs; for<'t> &'t T: (core::ops::Neg<Output = T>));
        )*
    }
}

macro_rules! impl_element_wise_op {
    (@ impl<T> $trait:ident<$rhs:ty> for $lhs:ty; type Output = $out:ty; for<$life:lifetime> $t:ty: $constraint:tt) => {
        paste::paste! {
            impl<T> core::ops::$trait<$rhs> for $lhs
            where
                for<$life> $t: $constraint,
            {
                type Output = $out;

                fn [<$trait:lower>](self, rhs: $rhs) -> Self::Output {
                    $crate::izip_eq!(self, rhs).map(|(lhs, rhs)| lhs.[<$trait:lower>](rhs)).collect()
                }
            }
        }
    };
    ($(impl<T> $trait:ident<$rhs:ty> for $lhs:ty),* $(,)?) => {
        $(
            impl_element_wise_op!(@ impl<T> $trait<&$rhs> for $lhs; type Output = $lhs; for<'t> T: (core::ops::$trait<&'t T, Output = T>));
            impl_element_wise_op!(@ impl<T> $trait<$rhs> for $lhs; type Output = $lhs; for<'t> T: (core::ops::$trait<T, Output = T>));
            impl_element_wise_op!(@ impl<T> $trait<&$rhs> for &$lhs; type Output = $lhs; for<'t> &'t T: (core::ops::$trait<&'t T, Output = T>));
            impl_element_wise_op!(@ impl<T> $trait<$rhs> for &$lhs; type Output = $lhs; for<'t> &'t T: (core::ops::$trait<T, Output = T>));
        )*
    }
}

macro_rules! impl_mul_assign_element {
    (@ impl<T> MulAssign<$rhs:ty> for $lhs:ty; for<$life:lifetime> $t:ty: $constraint:tt) => {
        impl<T> core::ops::MulAssign<$rhs> for $lhs
        where
            for<$life> $t: $constraint,
        {
            fn mul_assign(&mut self, rhs: $rhs) {
                self.iter_mut().for_each(|lhs| lhs.mul_assign(&rhs));
            }
        }
    };
    ($(impl<T> MulAssign<$rhs:ty> for $lhs:ty),* $(,)?) => {
        $(
            impl_mul_assign_element!(@ impl<T> MulAssign<$rhs> for $lhs; for<'t> T: (core::ops::MulAssign<&'t T>));
            impl_mul_assign_element!(@ impl<T> MulAssign<&$rhs> for $lhs; for<'t> T: (core::ops::MulAssign<&'t T>));
        )*
    }
}

macro_rules! impl_mul_element {
    (@ impl<T> Mul<$rhs:ty> for $lhs:ty; type Output = $out:ty; for<$life:lifetime> $t:ty: $constraint:tt) => {
        impl<T> core::ops::Mul<$rhs> for $lhs
        where
            for<$life> $t: $constraint,
        {
            type Output = $out;

            fn mul(self, rhs: $rhs) -> Self::Output {
                self.into_iter().map(|lhs| lhs.mul(&rhs)).collect()
            }
        }
    };
    ($(impl<T> Mul<$rhs:ty> for $lhs:ty),* $(,)?) => {
        $(
            impl_mul_element!(@ impl<T> Mul<$rhs> for $lhs; type Output = $lhs; for<'t> T: (core::ops::Mul<&'t T, Output = T>));
            impl_mul_element!(@ impl<T> Mul<&$rhs> for $lhs; type Output = $lhs; for<'t> T: (core::ops::Mul<&'t T, Output = T>));
            impl_mul_element!(@ impl<T> Mul<$rhs> for &$lhs; type Output = $lhs; for<'t> &'t T: (core::ops::Mul<&'t T, Output = T>));
            impl_mul_element!(@ impl<T> Mul<&$rhs> for &$lhs; type Output = $lhs; for<'t> &'t T: (core::ops::Mul<&'t T, Output = T>));
        )*
    }
}

impl_element_wise_neg!(
    impl<T> Neg for AVec<T>,
);
impl_element_wise_op_assign!(
    impl<T> AddAssign<AVec<T>> for AVec<T>,
    impl<T> SubAssign<AVec<T>> for AVec<T>,
);
impl_element_wise_op!(
    impl<T> Add<AVec<T>> for AVec<T>,
    impl<T> Sub<AVec<T>> for AVec<T>,
);
impl_mul_assign_element!(
    impl<T> MulAssign<T> for AVec<T>,
);
impl_mul_element!(
    impl<T> Mul<T> for AVec<T>,
);

pub(crate) use {
    impl_element_wise_neg, impl_element_wise_op, impl_element_wise_op_assign,
    impl_mul_assign_element, impl_mul_element,
};

macro_rules! impl_avec_i64_mul {
    (@ impl Mul<$rhs:ty> for $lhs:ty; type Output = $out:ty) => {
        impl core::ops::Mul<$rhs> for $lhs {
            type Output = $out;

            fn mul(self, rhs: $rhs) -> Self::Output {
                self.iter().map(|v| rhs * v).collect()
            }
        }
    };
    ($(impl Mul<$rhs:ty> for $lhs:ty),* $(,)?) => {
        $(
            impl_avec_i64_mul!(@ impl Mul<$rhs> for $lhs; type Output = AVec<$rhs>);
            impl_avec_i64_mul!(@ impl Mul<&$rhs> for $lhs; type Output = AVec<$rhs>);
            impl_avec_i64_mul!(@ impl Mul<$rhs> for &$lhs; type Output = AVec<$rhs>);
            impl_avec_i64_mul!(@ impl Mul<&$rhs> for &$lhs; type Output = AVec<$rhs>);
        )*
    };
}

impl_avec_i64_mul!(
    impl Mul<Zq> for AVec<i64>,
    impl Mul<T64> for AVec<i64>,
);
