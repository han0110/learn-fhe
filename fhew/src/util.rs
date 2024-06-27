use core::{f64::consts::SQRT_2, iter::Sum, ops::Mul};
use rand_distr::{Distribution, Standard, WeightedIndex};

mod avec;
mod decompose;
mod poly;
mod zq;

pub use avec::AVec;
pub use decompose::{Decomposable, Decomposor};
pub use poly::{NegaCyclicPoly, Rq, X};
pub use zq::{two_adic_primes, Zq};

pub trait Dot<Rhs> {
    type Output;

    fn dot(self, rhs: Rhs) -> Self::Output;
}

impl<'a, L, R, IR, IL> Dot<IR> for IL
where
    IL: IntoIterator<Item = &'a L>,
    IR: IntoIterator<Item = &'a R>,
    L: 'a + Sum,
    R: 'a,
    &'a L: Mul<&'a R, Output = L>,
{
    type Output = L;

    fn dot(self, rhs: IR) -> Self::Output {
        L::sum(izip_eq!(self, rhs).map(|(lhs, rhs)| lhs * rhs))
    }
}

pub fn zo(rho: f64) -> impl Distribution<i8> {
    assert!(rho <= 1.0);
    Standard.map(move |v: f64| {
        if v <= rho / 2.0 {
            -1
        } else if v <= rho {
            1
        } else {
            0
        }
    })
}

pub fn dg(std_dev: f64, n: usize) -> impl Distribution<i8> {
    // Formula 7.1.26 from Handbook of Mathematical Functions.
    let erf = |x: f64| {
        let p = 0.3275911;
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let t = 1.0 / (1.0 + p * x.abs());
        let positive_erf =
            1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
        if x.is_sign_positive() {
            positive_erf
        } else {
            -positive_erf
        }
    };
    let cdf = |x| (1.0 + erf(x / (std_dev * SQRT_2))) / 2.0;
    let max = (n as f64 * std_dev).floor() as i8;
    let weights = (-max..=max).map(|i| cdf(i as f64 + 0.5) - cdf(i as f64 - 0.5));
    WeightedIndex::new(weights)
        .unwrap()
        .map(move |v| v as i8 - max)
}

#[macro_export]
macro_rules! izip_eq {
    (@closure $p:pat => $tup:expr) => {
        |$p| $tup
    };
    (@closure $p:pat => ($($tup:tt)*) , $_iter:expr $(, $tail:expr)*) => {
        $crate::util::izip_eq!(@closure ($p, b) => ($($tup)*, b) $(, $tail)*)
    };
    ($first:expr $(,)*) => {
        itertools::__std_iter::IntoIterator::into_iter($first)
    };
    ($first:expr, $second:expr $(,)*) => {
        itertools::Itertools::zip_eq($crate::util::izip_eq!($first), $second)
    };
    ($first:expr $(, $rest:expr)* $(,)*) => {{
        let t = $crate::util::izip_eq!($first);
        $(let t = $crate::util::izip_eq!(t, $rest);)*
        t.map($crate::util::izip_eq!(@closure a => (a) $(, $rest)*))
    }};
}

#[macro_export]
macro_rules! zipstar {
    ($iters:expr $(, $field:tt)?) => {{
        let mut iters = $iters
            .into_iter()
            .map(|iter| iter$(.$field)?.into_iter())
            .collect::<Vec<_>>();
        let size_hint = itertools::Itertools::unique(iters.iter().map(|iter| iter.size_hint()))
            .collect::<Vec<_>>();
        assert_eq!(size_hint.len(), 1);
        core::iter::repeat_with(move || {
            iters
                .iter_mut()
                .map(|iter| iter.next().unwrap())
                .collect::<Vec<_>>()
        })
        .take(size_hint[0].1.unwrap())
    }};
}

#[macro_export]
macro_rules! cartesian {
    (@closure $p:pat => $tup:expr) => {
        |$p| $tup
    };
    (@closure $p:pat => ($($tup:tt)*) , $_iter:expr $(, $tail:expr)*) => {
        $crate::util::cartesian!(@closure ($p, b) => ($($tup)*, b) $(, $tail)*)
    };
    ($first:expr $(,)*) => {
        itertools::__std_iter::IntoIterator::into_iter($first)
    };
    ($first:expr, $second:expr $(,)*) => {
        itertools::Itertools::cartesian_product($crate::util::cartesian!($first), $second)
    };
    ($first:expr $(, $rest:expr)* $(,)*) => {
        let t = $crate::util::cartesian_product!($first);
        $(let t = $crate::util::cartesian_product!(t, $rest);)*
        t.map($crate::util::cartesian_product!(@closure a => (a) $(, $rest)*))
    };
}

pub use {cartesian, izip_eq, zipstar};
