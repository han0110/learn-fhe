use core::{
    borrow::Borrow,
    fmt::{self, Display, Formatter},
    iter::{Product, Sum},
    ops::{AddAssign, Deref, MulAssign, Neg, ShlAssign, ShrAssign, SubAssign},
};
use rand::RngCore;
use rand_distr::Distribution;

#[derive(Clone, Copy, Debug)]
pub struct Fq {
    q: u64,
    v: u64,
}

impl Deref for Fq {
    type Target = u64;

    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl Display for Fq {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.v)
    }
}

impl From<Fq> for u64 {
    fn from(value: Fq) -> Self {
        value.v
    }
}

impl Fq {
    pub fn from_u64(q: u64, v: u64) -> Self {
        let v = v % q;
        Self { q, v }
    }

    pub fn from_u128(q: u64, v: u128) -> Self {
        let v = (v % q as u128) as u64;
        Self { q, v }
    }

    pub fn from_i8(q: u64, v: i8) -> Self {
        let v = (v as i64 + q as i64) as u64 % q;
        Self { q, v }
    }

    pub fn sample_i8(q: u64, dist: &impl Distribution<i8>, rng: &mut impl RngCore) -> Self {
        Fq::from_i8(q, dist.sample(rng))
    }

    pub fn round(mut self, bits: usize) -> Self {
        self >>= bits;
        self <<= bits;
        self
    }
}

impl Neg for &Fq {
    type Output = Fq;

    #[inline]
    fn neg(self) -> Self::Output {
        Fq::from_u64(self.q, self.q - self.v)
    }
}

impl Neg for Fq {
    type Output = Fq;

    #[inline]
    fn neg(self) -> Self::Output {
        -&self
    }
}

impl AddAssign<&Fq> for Fq {
    #[inline]
    fn add_assign(&mut self, rhs: &Fq) {
        assert_eq!(self.q, rhs.q);
        *self = Self::from_u128(self.q, self.v as u128 + rhs.v as u128);
    }
}

impl SubAssign<&Fq> for Fq {
    #[inline]
    fn sub_assign(&mut self, rhs: &Fq) {
        assert_eq!(self.q, rhs.q);
        *self += -rhs;
    }
}

impl MulAssign<&Fq> for Fq {
    #[inline]
    fn mul_assign(&mut self, rhs: &Fq) {
        assert_eq!(self.q, rhs.q);
        *self = Self::from_u128(self.q, self.v as u128 * rhs.v as u128);
    }
}

impl ShlAssign<&usize> for Fq {
    #[inline]
    fn shl_assign(&mut self, rhs: &usize) {
        let v = (self.v as u128) << rhs;
        assert!(v <= self.q as u128);
        self.v = v as u64 % self.q;
    }
}

impl ShrAssign<&usize> for Fq {
    #[inline]
    fn shr_assign(&mut self, rhs: &usize) {
        self.v = (self.v + ((1 << rhs) >> 1)) >> rhs;
    }
}

impl<T: Borrow<Fq>> Sum<T> for Fq {
    fn sum<I: Iterator<Item = T>>(mut iter: I) -> Self {
        let init = *iter.next().unwrap().borrow();
        iter.fold(init, |acc, item| acc + item.borrow())
    }
}

impl<T: Borrow<Fq>> Product<T> for Fq {
    fn product<I: Iterator<Item = T>>(mut iter: I) -> Self {
        let init = *iter.next().unwrap().borrow();
        iter.fold(init, |acc, item| acc * item.borrow())
    }
}

macro_rules! impl_ops {
    (@ impl $trait:ident<$rhs:ty> for $lhs:ty; $lhs_convert:expr) => {
        paste::paste! {
            impl core::ops::$trait<$rhs> for $lhs {
                type Output = Fq;

                #[inline]
                fn [<$trait:lower>](self, rhs: $rhs) -> Fq {
                    let mut lhs = $lhs_convert(self);
                    lhs.[<$trait:lower _assign>](rhs.borrow());
                    lhs
                }
            }
        }
    };
    ($(impl $trait:ident<$rhs:ty> for $lhs:ty),* $(,)?) => {
        $(
            paste::paste! {
                impl core::ops::[<$trait Assign>]<$rhs> for $lhs {
                    #[inline]
                    fn [<$trait:lower _assign>](&mut self, rhs: $rhs) {
                        self.[<$trait:lower _assign>](&rhs);
                    }
                }
            }
            impl_ops!(@ impl $trait<$rhs> for $lhs; core::convert::identity);
            impl_ops!(@ impl $trait<&$rhs> for $lhs; core::convert::identity);
            impl_ops!(@ impl $trait<$rhs> for &$lhs; <_>::clone);
            impl_ops!(@ impl $trait<&$rhs> for &$lhs; <_>::clone);
        )*
    };
}

impl_ops!(
    impl Add<Fq> for Fq,
    impl Sub<Fq> for Fq,
    impl Mul<Fq> for Fq,
    impl Shl<usize> for Fq,
    impl Shr<usize> for Fq,
);
