use crate::zq::impl_rest_op_by_op_assign_ref;
use core::{
    borrow::Borrow,
    iter::{Product, Sum},
    ops::{AddAssign, MulAssign, Neg, SubAssign},
};
use derive_more::Display;
use rand::RngCore;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Display)]
#[display(fmt = "{}", .0.0)]
pub struct T64(u64);

impl T64 {
    pub fn zero() -> Self {
        Self(0)
    }

    pub fn sample_uniform(rng: &mut impl RngCore) -> Self {
        Self(rng.next_u64())
    }

    pub fn to_f64(&self) -> f64 {
        self.to_i64() as _
    }
}

impl From<f64> for T64 {
    #[inline(always)]
    fn from(value: f64) -> Self {
        Self::from(value.round() as i64)
    }
}

impl From<&T64> for f64 {
    #[inline(always)]
    fn from(value: &T64) -> Self {
        value.to_f64()
    }
}

impl From<T64> for f64 {
    #[inline(always)]
    fn from(value: T64) -> Self {
        value.to_f64()
    }
}

impl Neg for &T64 {
    type Output = T64;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        T64(self.0.wrapping_neg())
    }
}

impl Neg for T64 {
    type Output = T64;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        -&self
    }
}

impl AddAssign<&T64> for T64 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: &T64) {
        self.0 = self.0.wrapping_add(rhs.0);
    }
}

impl SubAssign<&T64> for T64 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &T64) {
        self.0 = self.0.wrapping_sub(rhs.0);
    }
}

impl MulAssign<&T64> for T64 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &T64) {
        self.0 = self.0.wrapping_mul(rhs.0);
    }
}

impl<T: Borrow<T64>> Sum<T> for T64 {
    fn sum<I: Iterator<Item = T>>(mut iter: I) -> Self {
        let init = *iter.next().unwrap().borrow();
        iter.fold(init, |acc, item| acc + item.borrow())
    }
}

impl<T: Borrow<T64>> Product<T> for T64 {
    fn product<I: Iterator<Item = T>>(mut iter: I) -> Self {
        let init = *iter.next().unwrap().borrow();
        iter.fold(init, |acc, item| acc * item.borrow())
    }
}

macro_rules! impl_from_to_primitive {
    (@ $p:ty) => {
        impl From<$p> for T64 {
            #[inline(always)]
            fn from(value: $p) -> Self {
                T64(value as _)
            }
        }

        impl From<T64> for $p {
            #[inline(always)]
            fn from(value: T64) -> Self {
                value.0 as _
            }
        }

        paste::paste! {
            impl T64 {
                #[inline(always)]
                pub fn [<to_ $p>](self) -> $p {
                    self.into()
                }
            }
        }
    };
    ($($p:ty),* $(,)?) => {
        $(impl_from_to_primitive!(@ $p);)*
    }
}

macro_rules! impl_op_with_primitive {
    (@ impl $trait:ident<&$p:ty> for T64) => {
        paste::paste! {
            impl core::ops::$trait<&$p> for T64 {
                #[inline(always)]
                fn [<$trait:snake:lower>](&mut self, rhs: &$p) {
                    self.[<$trait:snake:lower>](Self::from(*rhs));
                }
            }
        }
    };
    ($(impl $trait:ident<&$p:ty> for T64),* $(,)?) => {
        $(impl_op_with_primitive!(@ impl $trait<&$p> for T64);)*
    };
    ($($p:ty),* $(,)?) => {
        $(
            impl_op_with_primitive!(
                impl AddAssign<&$p> for T64,
                impl SubAssign<&$p> for T64,
                impl MulAssign<&$p> for T64,
            );
            impl_rest_op_by_op_assign_ref!(
                impl Add<$p> for T64,
                impl Sub<$p> for T64,
                impl Mul<$p> for T64,
            );
        )*
    };
}

impl_rest_op_by_op_assign_ref!(
    impl Add<T64> for T64,
    impl Sub<T64> for T64,
    impl Mul<T64> for T64,
);

impl_from_to_primitive!(u64, i64, u32, i32, u16, i16, u8, i8, usize, isize);
impl_op_with_primitive!(u64, i64, u32, i32, u16, i16, u8, i8, usize, isize);
