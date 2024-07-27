use crate::izip_eq;
use core::{
    iter::{successors, Sum},
    ops::{Add, Mul},
};
use num_traits::One;

pub mod decompose;
pub mod distribution;
pub mod matrix;

pub fn powers<T: One>(base: &T) -> impl Iterator<Item = T> + '_
where
    for<'t> &'t T: Mul<&'t T, Output = T>,
{
    successors(Some(T::one()), move |pow| Some(pow * base))
}

pub fn horner<'a, L, R, IL>(coeffs: IL, x: &'a R) -> L
where
    L: 'a + Clone + Mul<&'a R, Output = L> + Add<&'a L, Output = L>,
    IL: IntoIterator<Item = &'a L, IntoIter: DoubleEndedIterator>,
{
    let mut coeffs = coeffs.into_iter().rev();
    let init = coeffs.next().unwrap();
    coeffs.fold(init.clone(), |acc, c| acc * x + c)
}

pub fn bit_reverse<T, V: AsMut<[T]>>(mut values: V) -> V {
    let n = values.as_mut().len();
    if n > 2 {
        assert!(n.is_power_of_two());
        let log_len = n.ilog2();
        for i in 0..n {
            let j = i.reverse_bits() >> (usize::BITS - log_len);
            if i < j {
                values.as_mut().swap(i, j)
            }
        }
    }
    values
}

pub trait Dot<Rhs> {
    type Output;

    fn dot(self, rhs: Rhs) -> Self::Output;
}

impl<'a, L, R, IL, IR> Dot<IR> for IL
where
    IL: IntoIterator<Item = &'a L>,
    IR: IntoIterator<Item = R>,
    L: 'a + Sum,
    for<'t> &'t L: Mul<R, Output = L>,
{
    type Output = L;

    fn dot(self, rhs: IR) -> Self::Output {
        L::sum(izip_eq!(self, rhs).map(|(lhs, rhs)| lhs * rhs))
    }
}

pub trait HadamardMul<Rhs> {
    type Output;

    fn hada_mul(self, rhs: Rhs) -> Self::Output;
}

impl<'a, L, R, IL, IR> HadamardMul<IR> for &'a IL
where
    &'a IL: IntoIterator<Item = &'a L>,
    IL: FromIterator<L>,
    IR: IntoIterator<Item = &'a R>,
    L: 'a,
    R: 'a,
    for<'t> &'t L: Mul<&'t R, Output = L>,
{
    type Output = IL;

    fn hada_mul(self, rhs: IR) -> Self::Output {
        izip_eq!(self, rhs).map(|(lhs, rhs)| lhs * rhs).collect()
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
    ($first:expr $(, $rest:expr)* $(,)*) => {{
        let t = $crate::cartesian!($first);
        $(let t = $crate::cartesian!(t, $rest);)*
        t.map($crate::cartesian!(@closure a => (a) $(, $rest)*))
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
macro_rules! vec_with {
    [|| $f:expr; $n:expr] => {
        core::iter::repeat_with(|| $f).take($n).collect::<Vec<_>>()
    };
    [|$v:ident| $f:expr; $vs:expr] => {
        $vs.into_iter().map(|$v| $f).collect::<Vec<_>>()
    };
}
