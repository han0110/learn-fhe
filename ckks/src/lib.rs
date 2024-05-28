#![allow(dead_code)]

use crate::{
    float::{BigFloat, Complex},
    mat::Matrix,
    util::{bit_reverse, powers},
};
use itertools::{chain, izip, Itertools};
use num_bigint::{BigInt, BigUint};
use num_traits::ToPrimitive;
use util::{rem_center, rem_euclid};

use std::iter;

pub mod float;
pub mod mat;
pub mod util;

#[derive(Clone, Debug)]
pub struct Ckks;

#[derive(Clone, Debug)]
pub struct CkksParam {
    q: BigUint,
    qs: Vec<u64>,
    q_hats: Vec<BigUint>,
    q_hat_invs: Vec<BigUint>,
    log_n: usize,
    log_scale: usize,
    psi: Vec<Complex>,
    pow5: Vec<usize>,
}

impl CkksParam {
    pub fn q(&self) -> &BigUint {
        &self.q
    }

    pub fn qs(&self) -> &[u64] {
        &self.qs
    }

    pub fn q_hats(&self) -> &[BigUint] {
        &self.q_hats
    }

    pub fn q_hat_invs(&self) -> &[BigUint] {
        &self.q_hat_invs
    }

    pub fn m(&self) -> usize {
        1 << (self.log_n + 1)
    }

    pub fn n(&self) -> usize {
        1 << self.log_n
    }

    pub fn l(&self) -> usize {
        1 << (self.log_n - 1)
    }

    pub fn psi(&self) -> &[Complex] {
        &self.psi
    }

    pub fn pow5(&self) -> &[usize] {
        &self.pow5
    }
}

#[derive(Clone, Debug)]
pub struct CkksCleartext(Vec<Complex>);

#[derive(Clone, Debug)]
pub struct CkksPlaintext(Matrix<u64>);

impl Ckks {
    pub fn param_gen(qs: Vec<u64>, log_n: usize, log_scale: usize) -> CkksParam {
        assert!(log_n >= 1);
        let n = 1 << log_n;

        let q = qs.iter().product::<BigUint>();
        let q_hats = qs.iter().map(|qi| &q / qi).collect_vec();
        let q_hat_invs = izip!(&qs, &q_hats)
            .map(|(qi, qi_hat)| qi_hat.modinv(&BigUint::from(*qi)).unwrap())
            .collect_vec();

        let psi = {
            let phase = BigFloat::from(2) * BigFloat::pi() / BigFloat::from(2 * n);
            let cis = Complex::new(phase.cos(), phase.sin());
            powers(&cis).take(2 * n).collect()
        };

        let pow5 = iter::successors(Some(1), |pow| ((pow * 5) % (2 * n)).into())
            .take(2 * n)
            .collect();

        CkksParam {
            q,
            qs,
            q_hats,
            q_hat_invs,
            log_n,
            log_scale,
            psi,
            pow5,
        }
    }

    pub fn encode(param: &CkksParam, m: CkksCleartext) -> CkksPlaintext {
        assert_eq!(m.0.len(), param.l());

        let z = special_ifft(param, m.0);

        let z_scaled = chain![z.iter().map(|z| &z.re), z.iter().map(|z| &z.im)]
            .map(|z| BigInt::from(z << param.log_scale))
            .collect_vec();

        let mut pt = Matrix::new(param.qs().len(), param.n());
        izip!(pt.cols_mut(), param.qs()).for_each(|(col, qi)| {
            izip!(col, &z_scaled)
                .for_each(|(cell, z)| *cell = (rem_euclid(z, qi)).to_u64().unwrap());
        });

        CkksPlaintext(pt)
    }

    pub fn decode(param: &CkksParam, pt: CkksPlaintext) -> CkksCleartext {
        assert_eq!(pt.0.height(), param.n());

        let z_scaled =
            pt.0.rows()
                .map(|row| {
                    let z = izip!(param.q_hats(), param.q_hat_invs(), row)
                        .map(|(qi_hat, qi_hat_inv, cell)| qi_hat * qi_hat_inv * cell)
                        .sum::<BigUint>();
                    rem_center(&z, param.q())
                })
                .collect_vec();

        let z = izip!(&z_scaled[..param.l()], &z_scaled[param.l()..])
            .map(|(re, im)| {
                let [re, im] = [re, im].map(|z| BigFloat::from(z) >> param.log_scale);
                Complex::new(re, im)
            })
            .collect_vec();

        CkksCleartext(special_fft(param, z))
    }
}

fn special_fft(param: &CkksParam, mut w: Vec<Complex>) -> Vec<Complex> {
    assert_eq!(w.len(), param.l());
    let (pow5, psi) = (param.pow5(), param.psi());

    bit_reverse(&mut w);

    let l = w.len();
    let mut m = 2;
    while m <= l {
        for i in (0..l).step_by(m) {
            for j in 0..m / 2 {
                let k = (pow5[j] % (4 * m)) * l / m;
                let u = w[i + j].clone();
                let v = &w[i + j + m / 2] * &psi[k];
                w[i + j] = &u + &v;
                w[i + j + m / 2] = &u - &v;
            }
        }
        m *= 2;
    }

    w
}

fn special_ifft(param: &CkksParam, mut w: Vec<Complex>) -> Vec<Complex> {
    assert_eq!(w.len(), param.l());
    let (pow5, psi) = (param.pow5(), param.psi());

    let l = w.len();
    let mut m = l;
    while m >= 2 {
        for i in (0..l).step_by(m) {
            for j in 0..m / 2 {
                let k = (4 * m - pow5[j] % (4 * m)) * l / m;
                let u = &w[i + j] + &w[i + j + m / 2];
                let v = (&w[i + j] - &w[i + j + m / 2]) * &psi[k];
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

#[cfg(test)]
mod test {
    use crate::{
        assert_eq_complex, special_fft, special_ifft,
        util::{horner, primes},
        Ckks, CkksCleartext,
    };
    use itertools::{izip, Itertools};
    use rand::{
        rngs::{OsRng, StdRng},
        Rng, RngCore, SeedableRng,
    };
    use rand_distr::Standard;

    #[test]
    fn encrypt_decrypt() {
        let rng = &mut StdRng::seed_from_u64(OsRng.next_u64());
        let qs = primes((0..1 << 50).rev()).take(4).collect_vec();
        let log_scale = 40;
        for log_n in 1..10 {
            let param = Ckks::param_gen(qs.clone(), log_n, log_scale);
            let m = CkksCleartext(rng.sample_iter(Standard).take(param.l()).collect_vec());
            izip!(m.clone().0, Ckks::decode(&param, Ckks::encode(&param, m)).0)
                .for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs, 1.0e-10));
        }
    }

    #[test]
    fn special_ifft_fft() {
        crate::util::rem_euclid(&10, &10);
        let rng = &mut StdRng::seed_from_u64(OsRng.next_u64());
        for log_n in 1..10 {
            let param = Ckks::param_gen(Vec::new(), log_n, 40);
            let evals = rng.sample_iter(Standard).take(param.l()).collect_vec();
            let coeffs = special_ifft(&param, evals.clone());
            izip!(param.pow5(), &evals)
                .for_each(|(k, eval)| assert_eq_complex!(horner(&coeffs, &param.psi()[*k]), eval));
            izip!(evals, special_fft(&param, coeffs)).for_each(|(a, b)| assert_eq_complex!(a, b));
        }
    }
}
