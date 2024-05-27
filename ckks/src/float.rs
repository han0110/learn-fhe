use astro_float::{self, Consts, Radix, RoundingMode};
use num_traits::{Num, One, Zero};
use rand::Rng;
use rand_distr::{Distribution, Standard, Uniform};
use std::{
    fmt::{self, Display, Formatter},
    ops::{
        Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub,
        SubAssign,
    },
};

pub type Complex<T = BigFloat> = num_complex::Complex<T>;

const PRECISION: usize = 256;
const RM: RoundingMode = RoundingMode::None;

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd)]
pub struct BigFloat(astro_float::BigFloat);

impl BigFloat {
    pub fn pi() -> Self {
        Self(Consts::new().unwrap().pi(PRECISION, RM))
    }

    pub fn cos(&self) -> Self {
        BigFloat(self.0.cos(PRECISION, RM, &mut Consts::new().unwrap()))
    }

    pub fn sin(&self) -> Self {
        BigFloat(self.0.sin(PRECISION, RM, &mut Consts::new().unwrap()))
    }
}

impl Deref for BigFloat {
    type Target = astro_float::BigFloat;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for BigFloat {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Display for BigFloat {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

macro_rules! forward_impl_from {
    ($($primitive:ty),*) => {
        $(
            paste::paste! {
                impl From<$primitive> for BigFloat {
                    fn from(value: $primitive) -> BigFloat {
                        BigFloat(astro_float::BigFloat::[<from_$primitive>](value, PRECISION))
                    }
                }
            }
        )*
    };
}

forward_impl_from!(i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, f32, f64);

impl From<usize> for BigFloat {
    fn from(value: usize) -> Self {
        Self::from(value as u64)
    }
}

impl From<isize> for BigFloat {
    fn from(value: isize) -> Self {
        Self::from(value as i64)
    }
}

macro_rules! forward_arithmetic_ops {
    ($(impl Rem<$rhs:ty> for $lhs:ty),*) => {
        $(
            impl Rem<$rhs> for $lhs {
                type Output = BigFloat;

                #[inline]
                fn rem(self, other: $rhs) -> BigFloat {
                    BigFloat(self.0.rem(&other.0))
                }
            }
        )*
    };
    ($(impl $trait:ident<$rhs:ty> for $lhs:ty),*) => {
        $(
            paste::paste! {
                impl $trait<$rhs> for $lhs {
                    type Output = BigFloat;

                    #[inline]
                    fn [<$trait:lower>](self, other: $rhs) -> BigFloat {
                        BigFloat(self.0.[<$trait:lower>](&other.0, PRECISION, RM))
                    }
                }
            }
        )*
    };
    ($(impl $trait:ident),*) => {
        $(
            forward_arithmetic_ops!(
                impl $trait<BigFloat> for BigFloat,
                impl $trait<&BigFloat> for BigFloat,
                impl $trait<BigFloat> for &BigFloat,
                impl $trait<&BigFloat> for &BigFloat
            );
        )*
    };
}

forward_arithmetic_ops!(impl Add, impl Sub, impl Mul, impl Div, impl Rem);

macro_rules! forward_arithmetic_assign_ops {
    ($(impl $trait:ident<$rhs:ty>),*) => {
        $(
            paste::paste! {
                impl [<$trait Assign>]<$rhs> for BigFloat {
                    fn [<$trait:lower _assign>](&mut self, rhs: $rhs) {
                        *self = (&*self).[<$trait:lower>](rhs);
                    }
                }
            }
        )*
    };
    ($(impl $trait:ident),*) => {
        $(
            forward_arithmetic_assign_ops!(impl $trait<&BigFloat>, impl $trait<BigFloat>);
        )*
    };
}

forward_arithmetic_assign_ops!(impl Add, impl Sub, impl Mul, impl Div, impl Rem);

impl Neg for BigFloat {
    type Output = BigFloat;

    fn neg(self) -> BigFloat {
        BigFloat(self.0.neg())
    }
}

impl Neg for &BigFloat {
    type Output = BigFloat;

    fn neg(self) -> BigFloat {
        BigFloat(self.0.clone().neg())
    }
}

impl Zero for BigFloat {
    fn is_zero(&self) -> bool {
        self == &Self::from(0)
    }

    fn zero() -> Self {
        Self::from(0)
    }
}

impl One for BigFloat {
    fn is_one(&self) -> bool
    where
        Self: PartialEq,
    {
        self == &Self::from(1)
    }

    fn one() -> Self {
        Self::from(1)
    }
}

impl Num for BigFloat {
    type FromStrRadixErr = &'static str;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let radix = match radix {
            2 => Radix::Bin,
            8 => Radix::Oct,
            10 => Radix::Dec,
            16 => Radix::Hex,
            _ => return Err("Unsupported radix"),
        };
        Ok(Self(astro_float::BigFloat::parse(
            str,
            radix,
            PRECISION,
            RM,
            &mut Consts::new().unwrap(),
        )))
    }
}

impl Distribution<BigFloat> for Uniform<f32> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> BigFloat {
        rng.sample::<f32, _>(self).into()
    }
}

impl Distribution<BigFloat> for Uniform<f64> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> BigFloat {
        rng.sample::<f64, _>(self).into()
    }
}

impl Distribution<BigFloat> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> BigFloat {
        rng.sample::<f64, _>(self).into()
    }
}

#[macro_export]
macro_rules! assert_eq_float {
    (@ $lhs:expr, $rhs:expr $(,$field:literal)?) => {{
        let (lhs, rhs) = ($lhs, $rhs);
        let diff = lhs - rhs;
        let diff = if diff.is_negative() { -diff } else { diff };
        assert!(
            diff < $crate::float::BigFloat::from(1.0e-70),
            concat!(
                "assertion `left",
                $(".", $field,)?
                " == right",
                $(".", $field,)?
                "` failed\n  left: {}\n right: {}"
            ),
            lhs,
            rhs,
        );
    }};
    ($lhs:expr, $rhs:expr $(,)?) => {
        $crate::assert_eq_float!(@ $lhs, $rhs);
    };
}

#[macro_export]
macro_rules! assert_eq_complex {
    ($lhs:expr, $rhs:expr $(,)?) => {{
        let (lhs, rhs) = (&$lhs, &$rhs);
        $crate::assert_eq_float!(@ &lhs.re, &rhs.re, "re");
        $crate::assert_eq_float!(@ &lhs.im, &rhs.im, "im");
    }};
}
