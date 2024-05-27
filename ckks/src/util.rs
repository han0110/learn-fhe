use itertools::izip;
use num_traits::One;
use std::{
    iter::{self, Sum},
    ops::Mul,
};

pub fn powers<T: One>(base: &T) -> impl Iterator<Item = T> + '_
where
    for<'t> &'t T: Mul<&'t T, Output = T>,
{
    iter::successors(Some(T::one()), move |pow| Some(pow * base))
}

pub fn horner<T: One + Sum>(coeffs: &[T], x: &T) -> T
where
    for<'t> &'t T: Mul<&'t T, Output = T>,
{
    izip!(coeffs, powers(x))
        .map(|(coeff, pow)| coeff * &pow)
        .sum()
}
