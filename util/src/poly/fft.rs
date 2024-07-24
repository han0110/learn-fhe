use core::ops::{Add, Mul, MulAssign, Sub};
use itertools::izip;

pub mod c64;
pub mod zq;

// Algorithm 1 in 2016/504.
fn nega_cyclic_fft_in_place<T: Butterfly>(a: &mut [T], twiddle: &[T]) {
    assert!(a.len().is_power_of_two());
    for log_m in 0..a.len().ilog2() {
        let m = 1 << log_m;
        let t = a.len() / m;
        izip!(0.., a.chunks_exact_mut(t), &twiddle[m..]).for_each(|(i, a, twiddle)| {
            let (u, v) = a.split_at_mut(t / 2);
            if m == 0 && i == 0 {
                izip!(u, v).for_each(|(u, v)| Butterfly::twiddle_free(u, v));
            } else {
                izip!(u, v).for_each(|(u, v)| Butterfly::dit(u, v, twiddle));
            }
        });
    }
}

// Algorithm 2 in 2016/504.
fn nega_cyclic_ifft_in_place<T>(a: &mut [T], twiddle_inv: &[T], n_inv: &T)
where
    T: Butterfly + for<'t> MulAssign<&'t T>,
{
    assert!(a.len().is_power_of_two());
    for log_m in (0..a.len().ilog2()).rev() {
        let m = 1 << log_m;
        let t = a.len() / m;
        izip!(0.., a.chunks_exact_mut(t), &twiddle_inv[m..]).for_each(|(i, a, twiddle_inv)| {
            let (u, v) = a.split_at_mut(t / 2);
            if m == 0 && i == 0 {
                izip!(u, v).for_each(|(u, v)| Butterfly::twiddle_free(u, v));
            } else {
                izip!(u, v).for_each(|(u, v)| Butterfly::dif(u, v, twiddle_inv));
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
