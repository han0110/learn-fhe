use core::{iter::repeat, ops::Deref};
use itertools::{chain, izip, Itertools};
use std::sync::{Mutex, MutexGuard, OnceLock};
use util::{bit_reverse, powers, AVec, Butterfly, DiagSparseMatrix, Zq, C256, F256};

// Algorithm 1 in 2018/1043
pub(crate) fn sfft(mut z: AVec<C256>) -> AVec<C256> {
    assert!(z.len().is_power_of_two());
    bit_reverse(&mut z);
    for log_m in 0..z.len().ilog2() {
        let m = 1 << log_m;
        let w = w(2 * m);
        z.chunks_exact_mut(2 * m).for_each(|chunk| {
            let (a, b) = chunk.split_at_mut(m);
            izip!(a, b, w.iter()).for_each(|(a, b, t)| Butterfly::dit(a, b as &mut _, t))
        });
    }
    z
}

pub(crate) fn sifft(mut z: AVec<C256>) -> AVec<C256> {
    assert!(z.len().is_power_of_two());
    for log_m in (0..z.len().ilog2()).rev() {
        let m = 1 << log_m;
        let w = w(2 * m).conj();
        z.chunks_exact_mut(2 * m).for_each(|chunk| {
            let (a, b) = chunk.split_at_mut(m);
            izip!(a, b, w.iter()).for_each(|(a, b, t)| Butterfly::dif(a, b as &mut _, t))
        });
    }
    bit_reverse(&mut z);
    let n = F256::from(z.len());
    z.iter_mut().for_each(|z| *z /= &n);
    z
}

static POW5_TWIDDLE: OnceLock<Mutex<(Vec<usize>, Vec<C256>)>> = OnceLock::new();

struct W<'a>(MutexGuard<'a, (Vec<usize>, Vec<C256>)>, usize, bool);

impl<'a> W<'a> {
    fn conj(self) -> Self {
        Self(self.0, self.1, !self.2)
    }

    fn iter(&self) -> impl Iterator<Item = &C256> + Clone {
        let (pow5, twiddle) = self.0.deref();
        let (n, conj, step) = (self.1, self.2, pow5.len() / self.1);
        pow5.iter().take(n / 2).map(move |j| {
            let j = Zq::from_usize(4 * n as u64, *j);
            let j = if conj { -j } else { j };
            &twiddle[j.to_usize() * step]
        })
    }
}

// Twiddle factors in powers-of-5-mod-4n order.
fn w<'a>(n: usize) -> W<'a> {
    let mut guard = POW5_TWIDDLE.get_or_init(Default::default).lock().unwrap();
    if guard.0.len() < n {
        guard.0 = {
            let five = Zq::from_u64(4 * n as u64, 5);
            five.powers().take(n).map_into().collect()
        };
        guard.1 = {
            let phase = F256::pi() / F256::from(2 * n);
            let cis = C256::new(phase.cos(), phase.sin());
            powers(&cis).take(4 * n).collect()
        };
    }
    W(guard, n, false)
}

// V_0 in 2018/1073
pub(crate) fn sfft_fmats(n: usize) -> Vec<DiagSparseMatrix<C256>> {
    assert!(n.is_power_of_two());
    let log_n = n.ilog2();
    (0..log_n)
        .map(|log_k| {
            let m = 1 << (log_n - 1 - log_k);
            let [zero, one] = [0, 1].map(|v| repeat(C256::new(v.into(), 0.into())).take(m));
            let w = w(2 * m);
            let diag_zero = AVec::broadcast(n, chain![one.clone(), w.iter().map(|w| -w)]);
            if log_k == 0 {
                let diag_neg = AVec::broadcast(n, chain![w.iter().cloned(), one.clone()]);
                DiagSparseMatrix::new([(0, diag_zero), (n - m, diag_neg)])
            } else {
                let diag_neg = AVec::broadcast(n, chain![zero.clone(), one.clone()]);
                let diag_pos = AVec::broadcast(n, chain![w.iter().cloned(), zero.clone()]);
                DiagSparseMatrix::new([(0, diag_zero), (n - m, diag_neg), (m, diag_pos)])
            }
        })
        .collect()
}

// V_0^-1 in 2018/1073
pub(crate) fn sifft_fmats(n: usize) -> Vec<DiagSparseMatrix<C256>> {
    sfft_fmats(n).iter().rev().map(|mat| mat.inv()).collect()
}

#[cfg(test)]
mod test {
    use crate::sfft::{sfft, sfft_fmats, sifft, w};
    use itertools::{chain, izip, Itertools};
    use rand::{distributions::Standard, rngs::StdRng, SeedableRng};
    use util::{
        assert_eq_complex, bit_reverse, horner, izip_eq, powers, vec_with, AVec, DiagSparseMatrix,
    };

    #[test]
    fn sifft_sfft() {
        let rng = &mut StdRng::from_entropy();
        for log_n in 1..10 {
            let n = 1 << log_n;
            let evals = AVec::sample(n, Standard, rng);
            let coeffs = sifft(evals.clone());
            let w = w(n).iter().cloned().collect_vec();
            izip_eq!(chain![w.iter().cloned(), w.iter().map(|t| -t)], &evals)
                .for_each(|(t, eval)| assert_eq_complex!(horner(&coeffs, &t), eval));
            izip_eq!(evals, sfft(coeffs)).for_each(|(a, b)| assert_eq_complex!(a, b));
        }
    }

    #[test]
    fn sfft_mat_factorization() {
        for log_n in 1..10 {
            let n = 1 << log_n;
            let lhs = sfft_fmats(n).into_iter().product::<DiagSparseMatrix<_>>();
            let rhs = vec_with![|t| bit_reverse(powers(t).take(n).collect_vec()); w(n).iter()];
            izip!(lhs.to_dense(), rhs).for_each(|(lhs, rhs)| {
                izip!(lhs, rhs).for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs))
            });
        }
    }
}
