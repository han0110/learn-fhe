use core::{
    iter::{successors, Sum},
    ops::{Add, Mul},
};
use num_traits::One;

pub mod decompose;
pub mod distribution;

pub fn powers<T: One>(base: &T) -> impl Iterator<Item = T> + '_
where
    for<'t> &'t T: Mul<&'t T, Output = T>,
{
    successors(Some(T::one()), move |pow| Some(pow * base))
}

pub fn horner<L, R>(coeffs: &[L], x: &R) -> L
where
    for<'t> L: Clone + Mul<&'t R, Output = L> + Add<&'t L, Output = L>,
{
    let (init, coeffs) = coeffs.split_last().unwrap();
    coeffs.iter().rev().fold(init.clone(), |acc, c| acc * x + c)
}

pub fn bit_reverse<T>(values: &mut [T]) {
    if values.len() > 2 {
        assert!(values.len().is_power_of_two());
        let log_len = values.len().ilog2();
        for i in 0..values.len() {
            let j = i.reverse_bits() >> (usize::BITS - log_len);
            if i < j {
                values.swap(i, j)
            }
        }
    }
}

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
        L::sum(crate::izip_eq!(self, rhs).map(|(lhs, rhs)| lhs * rhs))
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
