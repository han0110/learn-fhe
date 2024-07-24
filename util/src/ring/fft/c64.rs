use crate::{
    avec::AVec,
    complex::C64,
    misc::bit_reverse,
    ring::fft::{nega_cyclic_fft_in_place, nega_cyclic_ifft_in_place},
    zq::Zq,
};
use core::f64::consts::PI;
use itertools::{izip, Itertools};
use std::sync::{Mutex, MutexGuard, OnceLock};

#[allow(dead_code)]
pub fn nega_cyclic_fft64_mul_assign(a: &mut [Zq], b: &[Zq]) {
    let mut t = zq_to_c64(a);
    let mut b = zq_to_c64(b);
    nega_cyclic_fft64_in_place(&mut t);
    nega_cyclic_fft64_in_place(&mut b);
    izip!(t.iter_mut(), b.iter()).for_each(|(a, b)| *a *= b);
    nega_cyclic_ifft64_in_place(&mut t);
    izip!(a, t).for_each(|(a, t)| *a = Zq::from_f64(a.q(), t.re));
}

fn zq_to_c64(v: &[Zq]) -> AVec<C64> {
    v.iter().map(|v| C64::new(v.into(), 0.)).collect()
}

#[allow(dead_code)]
pub fn nega_cyclic_fft64_in_place(a: &mut [C64]) {
    let [twiddle, _] = &*twiddle(a.len());
    nega_cyclic_fft_in_place(a, twiddle)
}

#[allow(dead_code)]
pub fn nega_cyclic_ifft64_in_place(a: &mut [C64]) {
    let [_, twiddle_inv] = &*twiddle(a.len());
    let n_inv = C64::new(1.0 / a.len() as f64, 0.);
    nega_cyclic_ifft_in_place(a, twiddle_inv, &n_inv)
}

// Twiddle factors in bit-reversed order.
fn twiddle<'a>(n: usize) -> MutexGuard<'a, [Vec<C64>; 2]> {
    static TWIDDLE: OnceLock<Mutex<[Vec<C64>; 2]>> = OnceLock::new();
    let mut twiddle = TWIDDLE.get_or_init(Default::default).lock().unwrap();
    if twiddle[0].len() < n {
        *twiddle = compute_twiddle(n);
    }
    twiddle
}

fn compute_twiddle(n: usize) -> [Vec<C64>; 2] {
    let twiddle = (0..n)
        .map(|i| C64::cis((i as f64 * PI) / n as f64))
        .collect_vec();
    let twiddle_inv = twiddle.iter().map(C64::conj).collect();
    [bit_reverse(twiddle), bit_reverse(twiddle_inv)]
}

#[cfg(test)]
mod test {
    use crate::{
        avec::AVec,
        complex::C64,
        ring::{
            fft::c64::{
                nega_cyclic_fft64_in_place, nega_cyclic_fft64_mul_assign,
                nega_cyclic_ifft64_in_place, zq_to_c64,
            },
            nega_cyclic_schoolbook_mul,
        },
        zq::Zq,
    };
    use core::array::from_fn;
    use rand::thread_rng;

    fn nega_cyclic_fft64(a: &[C64]) -> AVec<C64> {
        let mut a = AVec::from(a);
        nega_cyclic_fft64_in_place(&mut a);
        a
    }

    fn nega_cyclic_ifft64(a: &[C64]) -> AVec<C64> {
        let mut a = AVec::from(a);
        nega_cyclic_ifft64_in_place(&mut a);
        a
    }

    fn nega_cyclic_fft64_mul(a: &[Zq], b: &[Zq]) -> AVec<Zq> {
        let mut a = AVec::from(a);
        nega_cyclic_fft64_mul_assign(&mut a, b);
        a
    }

    #[test]
    fn round_trip() {
        let round_trip = |a: &AVec<Zq>| -> AVec<Zq> {
            let t = zq_to_c64(a);
            let t = nega_cyclic_ifft64(&nega_cyclic_fft64(&t));
            t.iter().map(|t| Zq::from_f64(a.q(), t.re)).collect()
        };
        let mut rng = thread_rng();
        for log_n in 0..10 {
            let q = 1 << (f64::MANTISSA_DIGITS - log_n);
            let n = 1 << log_n;
            for _ in 0..100 {
                let a = AVec::sample_uniform(q, n, &mut rng);
                assert_eq!(round_trip(&a), a);
            }
        }
    }

    #[test]
    fn nega_cyclic_mul() {
        let mut rng = thread_rng();
        for log_n in 0..10 {
            let n = 1 << log_n;
            let q = 1 << ((f64::MANTISSA_DIGITS - log_n) / 2);
            for _ in 0..100 {
                let [a, b] = &from_fn(|_| AVec::sample_uniform(q, n, &mut rng));
                assert_eq!(
                    nega_cyclic_fft64_mul(a, b),
                    nega_cyclic_schoolbook_mul(a, b)
                );
            }
        }
    }
}
