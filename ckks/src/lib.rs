#![allow(dead_code)]

use crate::{
    float::{BigFloat, Complex},
    mat::Matrix,
    util::powers,
};
use std::iter;

pub mod float;
pub mod mat;
pub mod util;

#[derive(Clone, Debug)]
pub struct Ckks;

#[derive(Clone, Debug)]
pub struct CkksParam {
    n: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum CkksEncoding {
    Coefficient,
    Evaluation,
}

#[derive(Clone, Debug)]
pub struct CkksCleartext(Vec<Complex>);

#[derive(Clone, Debug)]
pub struct CkksPlaintext(Matrix<u64>);

impl Ckks {
    pub fn param_gen(n: usize) -> CkksParam {
        CkksParam { n }
    }

    pub fn encode(_param: &CkksParam, _m: CkksCleartext, _enc: CkksEncoding) -> CkksPlaintext {
        todo!()
    }

    pub fn decode(_param: &CkksParam, _pt: CkksPlaintext, _enc: CkksEncoding) -> CkksCleartext {
        todo!()
    }
}

pub fn special_fft(mut w: Vec<Complex>, psis: &[Complex], pow5s: &[usize]) -> Vec<Complex> {
    assert!(w.len().is_power_of_two());
    assert_eq!(w.len() * 4, psis.len());
    assert_eq!(pow5s.len(), psis.len());

    bit_reverse(&mut w);

    let l = w.len();
    let mut m = 2;
    while m <= l {
        for i in (0..l).step_by(m) {
            for j in 0..m / 2 {
                let k = (pow5s[j] % (4 * m)) * l / m;
                let u = w[i + j].clone();
                let v = &w[i + j + m / 2] * &psis[k];
                w[i + j] = &u + &v;
                w[i + j + m / 2] = &u - &v;
            }
        }
        m *= 2;
    }

    w
}

pub fn special_ifft(mut w: Vec<Complex>, psis: &[Complex], pow5s: &[usize]) -> Vec<Complex> {
    assert!(w.len().is_power_of_two());
    assert_eq!(w.len() * 4, psis.len());
    assert_eq!(pow5s.len(), psis.len());

    let l = w.len();
    let mut m = l;
    while m >= 2 {
        for i in (0..l).step_by(m) {
            for j in 0..m / 2 {
                let k = (4 * m - pow5s[j] % (4 * m)) * l / m;
                let u = &w[i + j] + &w[i + j + m / 2];
                let v = (&w[i + j] - &w[i + j + m / 2]) * &psis[k];
                w[i + j] = u;
                w[i + j + m / 2] = v;
            }
        }
        m /= 2;
    }

    bit_reverse(&mut w);
    w.iter_mut().for_each(|w| *w /= BigFloat::from(l));

    w
}

pub fn psis(m: usize) -> Vec<Complex> {
    assert!(m >= 4 && m.is_power_of_two());

    let phase = BigFloat::from(2) * BigFloat::pi() / BigFloat::from(m);
    let cis = Complex::new(phase.cos(), phase.sin());
    powers(&cis).take(m).collect()
}

pub fn pow5s(m: usize) -> Vec<usize> {
    assert!(m >= 4 && m.is_power_of_two());

    iter::successors(Some(1), |pow| ((pow * 5) % m).into())
        .take(m)
        .collect()
}

pub fn bit_reverse<T>(values: &mut [T]) {
    if values.len() < 2 {
        return;
    }

    assert!(values.len().is_power_of_two());
    let log_len = values.len().ilog2();
    for i in 0..values.len() {
        let j = i.reverse_bits() >> (usize::BITS - log_len);
        if i < j {
            values.swap(i, j)
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{assert_eq_complex, pow5s, psis, special_fft, special_ifft, util::horner};
    use itertools::{izip, Itertools};
    use rand::{
        rngs::{OsRng, StdRng},
        Rng, RngCore, SeedableRng,
    };
    use rand_distr::Standard;

    #[test]
    fn special_ifft_fft() {
        let rng = &mut StdRng::seed_from_u64(OsRng.next_u64());
        for l in (0..10).map(|log_l| 1 << log_l) {
            let psis = psis(4 * l);
            let pow5s = pow5s(4 * l);
            let evals = rng.sample_iter(Standard).take(l).collect_vec();
            let coeffs = special_ifft(evals.clone(), &psis, &pow5s);
            izip!(&pow5s, &evals)
                .for_each(|(k, eval)| assert_eq_complex!(horner(&coeffs, &psis[*k]), eval));
            izip!(evals, special_fft(coeffs, &psis, &pow5s))
                .for_each(|(a, b)| assert_eq_complex!(a, b));
        }
    }
}
