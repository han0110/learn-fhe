use crate::{
    complex::C64,
    misc::bit_reverse,
    ring::fft::{nega_cyclic_fft_in_place, nega_cyclic_ifft_in_place},
    torus::T64,
};
use core::f64::consts::PI;
use itertools::{izip, Itertools};
use std::sync::{Mutex, MutexGuard, OnceLock};

pub fn nega_cyclic_fft64_mul_assign_rt(a: &mut [T64], b: &[T64]) {
    let torus_to_c64 = |a: &T64| a.to_f64().into();
    let integ_to_c64 = |b: &T64| (b.to_i64() as f64).into();
    let c64_to_torus = |c: C64| c.re.into();
    nega_cyclic_fft64_mul_assign(a, b, torus_to_c64, integ_to_c64, c64_to_torus);
}

pub fn nega_cyclic_fft64_mul_assign<T>(
    a: &mut [T],
    b: &[T],
    a_to_c64: impl Fn(&T) -> C64,
    b_to_c64: impl Fn(&T) -> C64,
    from_c64: impl Fn(C64) -> T,
) {
    let mut ca = a.iter().map(a_to_c64).collect_vec();
    let mut cb = b.iter().map(b_to_c64).collect_vec();
    nega_cyclic_fft64_in_place(&mut ca);
    nega_cyclic_fft64_in_place(&mut cb);
    izip!(ca.iter_mut(), cb.iter()).for_each(|(a, b)| *a *= b);
    nega_cyclic_ifft64_in_place(&mut ca);
    izip!(a, ca).for_each(|(a, ca)| *a = from_c64(ca));
}

pub fn nega_cyclic_fft64_in_place(a: &mut [C64]) {
    let [twiddle, _] = &*twiddle(a.len());
    nega_cyclic_fft_in_place(a, twiddle)
}

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
        cartesian,
        complex::C64,
        ring::{
            fft::c64::{
                nega_cyclic_fft64_in_place, nega_cyclic_fft64_mul_assign_rt,
                nega_cyclic_ifft64_in_place,
            },
            test::nega_cyclic_schoolbook_mul,
        },
        torus::T64,
    };
    use core::array::from_fn;
    use itertools::{izip, Itertools};
    use rand::{
        distributions::{uniform::Uniform, Distribution},
        thread_rng,
    };

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

    fn nega_cyclic_fft64_mul_rt(a: &[T64], b: &[T64]) -> AVec<T64> {
        let mut a = AVec::from(a);
        nega_cyclic_fft64_mul_assign_rt(&mut a, b);
        a
    }

    #[test]
    fn round_trip() {
        let round_trip = |a: &AVec<f64>| -> AVec<f64> {
            let a = a.iter().copied().map_into().collect_vec();
            let a = nega_cyclic_ifft64(&nega_cyclic_fft64(&a));
            a.iter().map(|a| a.re.round()).collect()
        };
        let mut rng = thread_rng();
        for log_n in 0..10 {
            let n = 1 << log_n;
            let q = 1u64 << (f64::MANTISSA_DIGITS - 1 - log_n);
            let uniform = Uniform::new(0, q).map(|v| v as f64);
            for _ in 0..1000 {
                let a = AVec::sample(n, &uniform, &mut rng);
                assert_eq!(round_trip(&a), a);
            }
        }
    }

    #[test]
    fn nega_cyclic_mul_rt() {
        let mut rng = thread_rng();
        for log_n in 0..10 {
            let n = 1 << log_n;
            let q = 1u64 << ((f64::MANTISSA_DIGITS - 2 - log_n) / 2);
            let uniform = Uniform::new(0, q).map(T64::from);
            for _ in 0..1000 {
                let [a, b] = &from_fn(|_| AVec::sample(n, &uniform, &mut rng));
                assert_eq!(
                    nega_cyclic_fft64_mul_rt(a, b),
                    nega_cyclic_schoolbook_mul(a, b)
                );
            }
        }
    }

    #[test]
    fn precision() {
        let mut rng = thread_rng();
        for (log_n, log_b) in cartesian!(8..12, 12..18) {
            let precision_loss = u64::BITS + log_b + log_n - f64::MANTISSA_DIGITS;
            let n = 1 << log_n;
            let b = 1 << log_b;
            let uniform_b = &Uniform::new(0, b).map(T64::from);
            let sample = |_| {
                let a = &AVec::<T64>::sample_uniform(n, &mut rng);
                let b = &AVec::sample(n, uniform_b, &mut rng);
                izip!(
                    nega_cyclic_fft64_mul_rt(a, b),
                    nega_cyclic_schoolbook_mul(a, b)
                )
                .map(|(a, b)| (a - b).to_i64().abs())
                .max()
            };
            let diff = (0..100).flat_map(sample).max().unwrap_or_default();
            let log_diff = diff.checked_ilog2().unwrap_or_default();
            assert!(log_diff <= precision_loss);
        }
    }
}
