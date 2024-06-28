use crate::zq::impl_rest_op_by_op_assign_ref;
use astro_float::{Consts, Radix, RoundingMode, WORD_BIT_SIZE};
use core::{
    borrow::Borrow,
    ops::{Neg, ShlAssign, ShrAssign},
};
use derive_more::{Display, Neg};
use num_bigint::BigInt;
use num_traits::{Num, One, Zero};
use rand::Rng;
use rand_distr::{Distribution, Standard, Uniform};

pub type Complex<T = BigFloat> = num_complex::Complex<T>;

const PRECISION: usize = 256;
const RM: RoundingMode = RoundingMode::None;

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Display, Neg)]
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

    pub fn abs(&self) -> Self {
        Self(self.0.abs())
    }
}

macro_rules! impl_from {
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

impl_from!(u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, f32, f64);

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

impl Neg for &BigFloat {
    type Output = BigFloat;

    fn neg(self) -> BigFloat {
        -self.clone()
    }
}

impl ShlAssign<&usize> for BigFloat {
    #[allow(clippy::suspicious_op_assign_impl)]
    fn shl_assign(&mut self, rhs: &usize) {
        let e = self.0.exponent().unwrap() + *rhs as i32;
        self.0.set_exponent(e);
    }
}

impl ShrAssign<&usize> for BigFloat {
    #[allow(clippy::suspicious_op_assign_impl)]
    fn shr_assign(&mut self, rhs: &usize) {
        let e = self.0.exponent().unwrap() - *rhs as i32;
        self.0.set_exponent(e);
    }
}

macro_rules! impl_arithmetic_op {
    (@ impl $trait:ident<$rhs:ty> for $lhs:ty) => {
        paste::paste! {
            impl core::ops::$trait<$rhs> for $lhs {
                type Output = BigFloat;

                fn [<$trait:lower>](self, other: $rhs) -> BigFloat {
                    BigFloat(self.0.[<$trait:lower>](&other.0, PRECISION, RM))
                }
            }
        }
    };
    ($(impl $trait:ident for BigFloat),* $(,)?) => {
        $(
            impl_arithmetic_op!(@ impl $trait<BigFloat> for BigFloat);
            impl_arithmetic_op!(@ impl $trait<&BigFloat> for BigFloat);
            impl_arithmetic_op!(@ impl $trait<BigFloat> for &BigFloat);
            impl_arithmetic_op!(@ impl $trait<&BigFloat> for &BigFloat);
        )*
    };
}

macro_rules! impl_rem {
    ($(impl Rem<$rhs:ty> for $lhs:ty),* $(,)?) => {
        $(
            impl core::ops::Rem<$rhs> for $lhs {
                type Output = BigFloat;

                fn rem(self, other: $rhs) -> BigFloat {
                    BigFloat(self.0.rem(&other.0))
                }
            }
        )*
    };
}

macro_rules! impl_arithmetic_op_assign {
    (@ impl $trait:ident<$rhs:ty> for BigFloat) => {
        paste::paste! {
            impl core::ops::[<$trait Assign>]<$rhs> for BigFloat {
                fn [<$trait:lower _assign>](&mut self, rhs: $rhs) {
                    *self = core::ops::$trait::[<$trait:lower>](&*self, rhs);
                }
            }
        }
    };
    ($(impl $trait:ident for BigFloat),* $(,)?) => {
        $(
            impl_arithmetic_op_assign!(@ impl $trait<BigFloat> for BigFloat);
            impl_arithmetic_op_assign!(@ impl $trait<&BigFloat> for BigFloat);
        )*
    };
}

impl_arithmetic_op!(
    impl Add for BigFloat,
    impl Sub for BigFloat,
    impl Mul for BigFloat,
    impl Div for BigFloat,
);

impl_rem!(
    impl Rem<BigFloat> for BigFloat,
    impl Rem<&BigFloat> for BigFloat,
    impl Rem<BigFloat> for &BigFloat,
    impl Rem<&BigFloat> for &BigFloat,
);

impl_arithmetic_op_assign!(
    impl Add for BigFloat,
    impl Sub for BigFloat,
    impl Mul for BigFloat,
    impl Div for BigFloat,
    impl Rem for BigFloat,
);

impl_rest_op_by_op_assign_ref!(
    impl Shl<usize> for BigFloat,
    impl Shr<usize> for BigFloat,
);

impl Zero for BigFloat {
    fn zero() -> Self {
        Self::from(0)
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl One for BigFloat {
    fn one() -> Self {
        Self::from(1)
    }

    fn is_one(&self) -> bool {
        self == &Self::from(1)
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

impl From<&BigFloat> for BigInt {
    fn from(value: &BigFloat) -> BigInt {
        let (m, n, s, e, _) = value.0.as_raw_parts().unwrap();

        if n == 0 {
            return BigInt::ZERO;
        }

        let (m_last, m) = m.split_last().unwrap();
        let mut v = m.iter().rev().fold(BigInt::from(*m_last), |acc, word| {
            (acc << WORD_BIT_SIZE) + word
        });

        let shift = PRECISION as i32 - e;
        if shift < 0 {
            v <<= shift.abs();
        } else {
            v >>= shift;
        }

        if s.is_negative() {
            -v
        } else {
            v
        }
    }
}

impl From<BigFloat> for BigInt {
    fn from(value: BigFloat) -> Self {
        Self::from(&value)
    }
}

impl From<&BigInt> for BigFloat {
    fn from(value: &BigInt) -> Self {
        let (sign, digits) = value.to_radix_be(10);
        let sign = if matches!(sign, num_bigint::Sign::Minus) {
            astro_float::Sign::Neg
        } else {
            astro_float::Sign::Pos
        };
        BigFloat(astro_float::BigFloat::convert_from_radix(
            sign,
            &digits,
            digits.len() as _,
            astro_float::Radix::Dec,
            PRECISION,
            RM,
            &mut Consts::new().unwrap(),
        ))
    }
}

impl From<BigInt> for BigFloat {
    fn from(value: BigInt) -> Self {
        Self::from(&value)
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
    (@ $precision:expr, $lhs:expr, $rhs:expr $(,$field:literal)?) => {{
        let (lhs, rhs) = ($lhs, $rhs);
        let diff = (lhs - rhs).abs();
        assert!(
            diff < (<$crate::BigFloat as num_traits::One>::one() >> $precision),
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
    ($lhs:expr, $rhs:expr, $precision:expr $(,)?) => {
        $crate::assert_eq_float!(@ $precision, $lhs, $rhs);
    };
    ($lhs:expr, $rhs:expr $(,)?) => {
        $crate::assert_eq_float!($lhs, $rhs, 200);
    };
}

#[macro_export]
macro_rules! assert_eq_complex {
    ($lhs:expr, $rhs:expr, $precision:expr $(,)?) => {{
        let (lhs, rhs) = (&$lhs, &$rhs);
        $crate::assert_eq_float!(@ $precision, &lhs.re, &rhs.re, "re");
        $crate::assert_eq_float!(@ $precision, &lhs.im, &rhs.im, "im");
    }};
    ($lhs:expr, $rhs:expr $(,)?) => {{
        $crate::assert_eq_complex!($lhs, $rhs, 200);
    }};
}
