use crate::{
    complex::C64,
    misc::bit_reverse,
    ring::fft::{fft_in_place, ifft_in_place},
    torus::T64,
};
use core::f64::consts::PI;
use itertools::{izip, Itertools};
use std::sync::{Mutex, MutexGuard, OnceLock};

pub fn nega_cyclic_fft64_mul_assign_rt(a: &mut [T64], b: &[T64]) {
    if a.len() == 1 {
        a[0] *= b[0];
        return;
    }
    nega_cyclic_fft64_mul_assign(a, b, to_c64_twisted, assign_from_c64_twisted);
}

// Formula 8 in 2021/480.
fn to_c64_twisted(a: &[T64]) -> Vec<C64> {
    let n = a.len();
    let [twiddle, _, _, _] = &*twiddle(n);
    let (lo, hi) = a.split_at(n / 2);
    let t = twiddle.iter().step_by(twiddle.len() / n);
    izip!(lo, hi, t)
        .map(|(lo, hi, t)| C64::new(lo.to_i64() as _, hi.to_i64() as _) * t)
        .collect()
}

// Formula 10 in 2021/480.
fn assign_from_c64_twisted(a: &mut [T64], c: Vec<C64>) {
    let n = a.len();
    let [_, twiddle_inv, _, _] = &*twiddle(n);
    let (lo, hi) = a.split_at_mut(n / 2);
    let t = twiddle_inv.iter().step_by(twiddle_inv.len() / n);
    izip!(lo, hi, t, c).for_each(|(lo, hi, t, mut c): (_, &mut _, _, _)| {
        c *= t;
        *lo = T64::from(f64_mod_u64(c.re));
        *hi = T64::from(f64_mod_u64(c.im));
    });
}

pub fn nega_cyclic_fft64_mul_assign<T>(
    a: &mut [T],
    b: &[T],
    to_c64: impl Fn(&[T]) -> Vec<C64>,
    assign_from_c64: impl Fn(&mut [T], Vec<C64>),
) {
    let mut ca = to_c64(a);
    let mut cb = to_c64(b);
    nega_cyclic_fft64_in_place(&mut ca);
    nega_cyclic_fft64_in_place(&mut cb);
    izip!(ca.iter_mut(), cb.iter()).for_each(|(a, b)| *a *= b);
    nega_cyclic_ifft64_in_place(&mut ca);
    assign_from_c64(a, ca);
}

pub fn nega_cyclic_fft64_in_place(a: &mut [C64]) {
    let [_, _, twiddle_bo, _] = &*twiddle(a.len());
    fft_in_place(a, twiddle_bo)
}

pub fn nega_cyclic_ifft64_in_place(a: &mut [C64]) {
    let [_, _, _, twiddle_inv_bo] = &*twiddle(a.len());
    let n_inv = 1f64 / a.len() as f64;
    ifft_in_place(a, twiddle_inv_bo, &n_inv)
}

#[inline(always)]
fn f64_mod_u64(v: f64) -> u64 {
    let bits = v.to_bits();
    let sign = bits >> 63;
    let exponent = (bits >> 52) & 0x7ff;
    let mantissa = (bits << 11) | 0x8000000000000000;
    let value = match 1086 - exponent as i64 {
        shift @ -63..=0 => mantissa << -shift,
        shift @ 1..=64 => ((mantissa >> (shift - 1)).wrapping_add(1)) >> 1,
        _ => 0,
    };
    if sign == 0 {
        value
    } else {
        value.wrapping_neg()
    }
}

// Twiddle factors in normal and bit-reversed order.
fn twiddle<'a>(n: usize) -> MutexGuard<'a, [Vec<C64>; 4]> {
    static TWIDDLE: OnceLock<Mutex<[Vec<C64>; 4]>> = OnceLock::new();
    let mut twiddle = TWIDDLE.get_or_init(Default::default).lock().unwrap();
    if twiddle[0].len() < n {
        *twiddle = compute_twiddle(n);
    }
    twiddle
}

fn compute_twiddle(n: usize) -> [Vec<C64>; 4] {
    let twiddle = (0..n)
        .map(|i| C64::cis((i as f64 * PI) / n as f64))
        .collect_vec();
    let twiddle_inv = twiddle.iter().map(C64::conj).collect_vec();
    [
        twiddle.clone(),
        twiddle_inv.clone(),
        bit_reverse(twiddle),
        bit_reverse(twiddle_inv),
    ]
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
            let q = 1u64 << (f64::MANTISSA_DIGITS - log_n);
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
            let q = 1u64 << ((f64::MANTISSA_DIGITS - 3 - log_n) / 2);
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
