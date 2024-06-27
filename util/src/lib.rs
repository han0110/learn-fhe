use core::{iter::Sum, ops::Mul};

mod avec;
mod decompose;
mod distribution;
mod poly;
mod zq;

pub use avec::AVec;
pub use decompose::{Decomposable, Decomposor};
pub use distribution::{dg, zo};
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

#[macro_export]
macro_rules! izip_eq {
    (@closure $p:pat => $tup:expr) => {
        |$p| $tup
    };
    (@closure $p:pat => ($($tup:tt)*) , $_iter:expr $(, $tail:expr)*) => {
        $crate::izip_eq!(@closure ($p, b) => ($($tup)*, b) $(, $tail)*)
    };
    ($first:expr $(,)*) => {
        itertools::__std_iter::IntoIterator::into_iter($first)
    };
    ($first:expr, $second:expr $(,)*) => {
        itertools::Itertools::zip_eq($crate::izip_eq!($first), $second)
    };
    ($first:expr $(, $rest:expr)* $(,)*) => {{
        let t = $crate::izip_eq!($first);
        $(let t = $crate::izip_eq!(t, $rest);)*
        t.map($crate::izip_eq!(@closure a => (a) $(, $rest)*))
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
        $crate::cartesian!(@closure ($p, b) => ($($tup)*, b) $(, $tail)*)
    };
    ($first:expr $(,)*) => {
        itertools::__std_iter::IntoIterator::into_iter($first)
    };
    ($first:expr, $second:expr $(,)*) => {
        itertools::Itertools::cartesian_product($crate::cartesian!($first), $second)
    };
    ($first:expr $(, $rest:expr)* $(,)*) => {
        let t = $crate::cartesian_product!($first);
        $(let t = $crate::cartesian_product!(t, $rest);)*
        t.map($crate::cartesian_product!(@closure a => (a) $(, $rest)*))
    };
}
