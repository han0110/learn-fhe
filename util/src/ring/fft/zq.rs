use crate::{
    avec::AVec,
    misc::bit_reverse,
    ring::fft::{nega_cyclic_fft_in_place, nega_cyclic_ifft_in_place},
    zq::{is_prime, Zq},
};
use core::ops::Deref;
use itertools::{izip, Itertools};
use std::{
    collections::HashMap,
    sync::{Mutex, MutexGuard, OnceLock},
};

pub fn nega_cyclic_ntt_mul_assign(a: &mut [Zq], b: &[Zq]) {
    nega_cyclic_ntt_in_place(a);
    let b = nega_cyclic_ntt(b);
    izip!(a.iter_mut(), b.iter()).for_each(|(a, b)| *a *= b);
    nega_cyclic_intt_in_place(a);
}

fn nega_cyclic_ntt(a: &[Zq]) -> AVec<Zq> {
    let mut a = AVec::from(a);
    nega_cyclic_ntt_in_place(&mut a);
    a
}

pub fn nega_cyclic_ntt_in_place(a: &mut [Zq]) {
    let [twiddle, _] = &*twiddle(a[0].q());
    nega_cyclic_fft_in_place(a, twiddle)
}

pub fn nega_cyclic_intt_in_place(a: &mut [Zq]) {
    let [_, twiddle_inv] = &*twiddle(a[0].q());
    let n_inv = Zq::from_u64(a[0].q(), a.len() as u64).inv().unwrap();
    nega_cyclic_ifft_in_place(a, twiddle_inv, &n_inv)
}

struct Twiddle<'a>(MutexGuard<'a, HashMap<u64, [Vec<Zq>; 2]>>, u64);

impl<'a> Deref for Twiddle<'a> {
    type Target = [Vec<Zq>; 2];

    fn deref(&self) -> &Self::Target {
        &self.0[&self.1]
    }
}

// Twiddle factors in bit-reversed order.
fn twiddle<'a>(q: u64) -> Twiddle<'a> {
    static TWIDDLE: OnceLock<Mutex<HashMap<u64, [Vec<Zq>; 2]>>> = OnceLock::new();
    let mut map = TWIDDLE.get_or_init(Default::default).lock().unwrap();
    if is_prime(q) {
        map.entry(q).or_insert_with(|| compute_twiddle(q));
    }
    Twiddle(map, q)
}

fn compute_twiddle(q: u64) -> [Vec<Zq>; 2] {
    let order = q - 1;
    let s = order.trailing_zeros() as _;
    let twiddle = Zq::two_adic_generator(q, s)
        .powers()
        .take(1 << (s - 1))
        .collect_vec();
    let twiddle_inv = twiddle.iter().map(|v| v.inv().unwrap()).collect();
    [bit_reverse(twiddle), bit_reverse(twiddle_inv)]
}

#[cfg(test)]
mod test {
    use crate::{
        avec::AVec,
        ring::{
            fft::zq::{nega_cyclic_intt_in_place, nega_cyclic_ntt, nega_cyclic_ntt_mul_assign},
            test::nega_cyclic_schoolbook_mul,
        },
        zq::{two_adic_primes, Zq},
    };
    use core::array::from_fn;
    use rand::thread_rng;

    fn nega_cyclic_intt(a: &[Zq]) -> AVec<Zq> {
        let mut a = AVec::from(a);
        nega_cyclic_intt_in_place(&mut a);
        a
    }

    fn nega_cyclic_ntt_mul(a: &[Zq], b: &[Zq]) -> AVec<Zq> {
        let mut a = AVec::from(a);
        nega_cyclic_ntt_mul_assign(&mut a, b);
        a
    }

    #[test]
    fn round_trip() {
        let mut rng = thread_rng();
        for log_n in 0..10 {
            let n = 1 << log_n;
            for q in two_adic_primes(45, log_n + 1).take(10) {
                let a = AVec::<Zq>::sample_uniform(q, n, &mut rng);
                assert_eq!(nega_cyclic_intt(&nega_cyclic_ntt(&a)), a);
            }
        }
    }

    #[test]
    fn nega_cyclic_mul() {
        let mut rng = thread_rng();
        for log_n in 0..10 {
            let n = 1 << log_n;
            for q in two_adic_primes(45, log_n + 1).take(10) {
                let [a, b] = &from_fn(|_| AVec::<Zq>::sample_uniform(q, n, &mut rng));
                assert_eq!(nega_cyclic_ntt_mul(a, b), nega_cyclic_schoolbook_mul(a, b));
            }
        }
    }
}
