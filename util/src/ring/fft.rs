use core::ops::{Add, Mul, MulAssign, Sub};
use itertools::izip;

pub mod c64;
pub mod zq;

// Given normal order input and bit-reversed order twiddle factors,
// compute bit-reversed order output in place.
fn fft_in_place<T: Butterfly>(a: &mut [T], twiddle_bo: &[T]) {
    assert!(a.len().is_power_of_two());
    for layer in (0..a.len().ilog2()).rev() {
        let size = 1 << layer;
        izip!(a.chunks_mut(2 * size), twiddle_bo).for_each(|(values, twiddle)| {
            let (a, b) = values.split_at_mut(size);
            izip!(a, b).for_each(|(a, b): (_, &mut _)| Butterfly::dit(a, b, twiddle));
        });
    }
}

// Given bit-reversed order input and bit-reversed order twiddle factors,
// compute normal order output in place.
fn ifft_in_place<T, S>(a: &mut [T], twiddle_inv_bo: &[T], n_inv: &S)
where
    T: Butterfly + for<'t> MulAssign<&'t S>,
{
    assert!(a.len().is_power_of_two());
    for layer in 0..a.len().ilog2() {
        let size = 1 << layer;
        izip!(a.chunks_mut(2 * size), twiddle_inv_bo).for_each(|(values, twiddle)| {
            let (a, b) = values.split_at_mut(size);
            izip!(a, b).for_each(|(a, b): (_, &mut _)| Butterfly::dif(a, b, twiddle));
        });
    }
    a.iter_mut().for_each(|a| *a *= n_inv);
}

// Algorithm 1 in 2016/504.
// Given normal order input and bit-reversed order twiddle factors,
// compute bit-reversed order output in place.
fn nega_cyclic_fft_in_place<T: Butterfly>(a: &mut [T], twiddle_bo: &[T]) {
    assert!(a.len().is_power_of_two());
    let log_n = a.len().ilog2();
    for layer in 0..log_n {
        let (m, size) = (1 << layer, 1 << (log_n - layer - 1));
        izip!(0.., a.chunks_exact_mut(2 * size), &twiddle_bo[m..]).for_each(|(i, a, t)| {
            let (u, v) = a.split_at_mut(size);
            if m == 0 && i == 0 {
                izip!(u, v).for_each(|(u, v)| Butterfly::twiddle_free(u, v));
            } else {
                izip!(u, v).for_each(|(u, v)| Butterfly::dit(u, v, t));
            }
        });
    }
}

// Algorithm 2 in 2016/504.
// Given bit-reversed order input and bit-reversed order twiddle factors,
// compute normal order output in place.
fn nega_cyclic_ifft_in_place<T>(a: &mut [T], twiddle_inv_bo: &[T], n_inv: &T)
where
    T: Butterfly + for<'t> MulAssign<&'t T>,
{
    assert!(a.len().is_power_of_two());
    let log_n = a.len().ilog2();
    for layer in (0..log_n).rev() {
        let (m, size) = (1 << layer, 1 << (log_n - layer - 1));
        izip!(0.., a.chunks_exact_mut(2 * size), &twiddle_inv_bo[m..]).for_each(|(i, a, t)| {
            let (u, v) = a.split_at_mut(size);
            if m == 0 && i == 0 {
                izip!(u, v).for_each(|(u, v)| Butterfly::twiddle_free(u, v));
            } else {
                izip!(u, v).for_each(|(u, v)| Butterfly::dif(u, v, t));
            }
        });
    }
    a.iter_mut().for_each(|a| *a *= n_inv);
}

pub trait Butterfly {
    fn dit(a: &mut Self, b: &mut Self, t: &Self);

    fn dif(a: &mut Self, b: &mut Self, t: &Self);

    fn twiddle_free(a: &mut Self, b: &mut Self);
}

impl<T> Butterfly for T
where
    for<'t> &'t T: Mul<&'t T, Output = T> + Add<&'t T, Output = T> + Sub<&'t T, Output = T>,
{
    #[inline(always)]
    fn dit(a: &mut Self, b: &mut Self, t: &Self) {
        let tb = t * b;
        let c = &*a + &tb;
        let d = &*a - &tb;
        *a = c;
        *b = d;
    }

    #[inline(always)]
    fn dif(a: &mut Self, b: &mut Self, t: &Self) {
        let c = &*a + b;
        let d = &(&*a - b) * t;
        *a = c;
        *b = d;
    }

    #[inline(always)]
    fn twiddle_free(a: &mut Self, b: &mut Self) {
        let c = &*a + b;
        let d = &*a - b;
        *a = c;
        *b = d;
    }
}
