use crate::util::{
    bit_reverse, dg,
    float::{BigFloat, Complex},
    poly::CrtPoly,
    powers,
    prime::{two_adic_primes, SmallPrime},
    zo,
};
use core::iter::successors;
use itertools::{chain, izip, Itertools};
use num_bigint::BigInt;
use rand::RngCore;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub struct Ckks;

#[derive(Clone, Debug)]
pub struct CkksParam {
    log_n: usize,
    pow5: Vec<usize>,
    psi: Vec<Complex>,
    qs: Vec<Rc<SmallPrime>>,
    scale: BigFloat,
}

impl CkksParam {
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

    pub fn qs(&self) -> &[Rc<SmallPrime>] {
        &self.qs
    }

    pub fn scale(&self) -> &BigFloat {
        &self.scale
    }
}

#[derive(Clone, Debug)]
pub struct CkksSecretKey(CrtPoly);

#[derive(Clone, Debug)]
pub struct CkksPublicKey(CrtPoly, CrtPoly);

#[derive(Clone, Debug)]
pub struct CkksEvaluationKey;

#[derive(Clone, Debug)]
pub struct CkksCleartext(Vec<Complex>);

#[derive(Clone, Debug)]
pub struct CkksPlaintext(CrtPoly);

#[derive(Clone, Debug)]
pub struct CkksCiphertext(CrtPoly, CrtPoly, Option<CrtPoly>);

impl Ckks {
    pub fn param_gen(log_n: usize, log_qi: usize, num_qs: usize) -> CkksParam {
        assert!(log_n >= 1);

        let n = 1 << log_n;

        let psi = {
            let phase = BigFloat::from(2) * BigFloat::pi() / BigFloat::from(2 * n);
            let cis = Complex::new(phase.cos(), phase.sin());
            powers(&cis).take(2 * n).collect()
        };

        let pow5 = successors(Some(1), |pow| ((pow * 5) % (2 * n)).into())
            .take(2 * n)
            .collect();

        let qs = two_adic_primes(log_qi, log_n + 1)
            .map(|qi| Rc::new(SmallPrime::new(qi)))
            .take(num_qs)
            .collect_vec();

        let scale = BigFloat::from(***qs.last().unwrap());

        CkksParam {
            log_n,
            psi,
            pow5,
            qs,
            scale,
        }
    }

    pub fn key_gen(
        param: &CkksParam,
        rng: &mut impl RngCore,
    ) -> (CkksSecretKey, CkksPublicKey, CkksEvaluationKey) {
        let sk = CkksSecretKey(CrtPoly::sample_small(param.n(), param.qs(), &zo(0.5), rng));
        let pk = {
            let a = CrtPoly::sample_uniform(param.n(), param.qs(), rng);
            let e = CrtPoly::sample_small(param.n(), param.qs(), &dg(3.2, 6), rng);
            let b = -(&a * &sk.0) + e;
            CkksPublicKey(b, a)
        };
        let ek = CkksEvaluationKey;
        (sk, pk, ek)
    }

    pub fn encode(param: &CkksParam, m: CkksCleartext) -> CkksPlaintext {
        assert_eq!(m.0.len(), param.l());

        let z = special_ifft(param, m.0);

        let z_scaled = chain![z.iter().map(|z| &z.re), z.iter().map(|z| &z.im)]
            .map(|z| BigInt::from(z * param.scale()))
            .collect_vec();

        let pt = CrtPoly::from_bigint(z_scaled, param.qs());

        CkksPlaintext(pt)
    }

    pub fn decode(param: &CkksParam, pt: CkksPlaintext) -> CkksCleartext {
        assert_eq!(pt.0.n(), param.n());

        let z_scaled = pt.0.into_bigint();

        let z = izip!(&z_scaled[..param.l()], &z_scaled[param.l()..])
            .map(|(re, im)| {
                let [re, im] = [re, im].map(|z| BigFloat::from(z) / param.scale());
                Complex::new(re, im)
            })
            .collect_vec();

        let m = special_fft(param, z);

        CkksCleartext(m)
    }

    pub fn encrypt(
        param: &CkksParam,
        pk: &CkksPublicKey,
        pt: CkksPlaintext,
        rng: &mut impl RngCore,
    ) -> CkksCiphertext {
        let u = &CrtPoly::sample_small(param.n(), param.qs(), &zo(0.5), rng);
        let e0 = &CrtPoly::sample_small(param.n(), param.qs(), &dg(3.2, 6), &mut *rng);
        let e1 = &CrtPoly::sample_small(param.n(), param.qs(), &dg(3.2, 6), &mut *rng);
        let c0 = &pk.0 * u + e0 + pt.0;
        let c1 = &pk.1 * u + e1;
        CkksCiphertext(c0, c1, None)
    }

    pub fn decrypt(_: &CkksParam, sk: &CkksSecretKey, ct: CkksCiphertext) -> CkksPlaintext {
        let pt = if let Some(c2) = ct.2 {
            ct.0 + ct.1 * &sk.0 + c2 * &sk.0 * &sk.0
        } else {
            ct.0 + ct.1 * &sk.0
        };
        CkksPlaintext(pt)
    }

    pub fn eval_add(
        _: &CkksParam,
        _: &CkksEvaluationKey,
        ct0: CkksCiphertext,
        ct1: CkksCiphertext,
    ) -> CkksCiphertext {
        let c0 = ct0.0 + ct1.0;
        let c1 = ct0.1 + ct1.1;
        let c2 = match (ct0.2, ct1.2) {
            (Some(lhs), Some(rhs)) => Some(lhs + rhs),
            (Some(c2), None) | (None, Some(c2)) => Some(c2),
            (None, None) => None,
        };
        CkksCiphertext(c0, c1, c2)
    }

    pub fn eval_mul(
        _: &CkksParam,
        _: &CkksEvaluationKey,
        ct0: CkksCiphertext,
        ct1: CkksCiphertext,
    ) -> CkksCiphertext {
        assert!(ct0.2.is_none());
        assert!(ct1.2.is_none());

        let [c0, c1, c2] = [
            &ct0.0 * &ct1.0,
            &ct0.0 * &ct1.1 + &ct0.1 * &ct1.0,
            &ct0.1 * &ct1.1,
        ]
        .map(|c| c.rescale());
        CkksCiphertext(c0, c1, Some(c2))
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
        assert_eq_complex,
        ckks::{special_fft, special_ifft, Ckks, CkksCleartext},
        util::horner,
    };
    use itertools::{izip, Itertools};
    use rand::{
        rngs::{OsRng, StdRng},
        Rng, RngCore, SeedableRng,
    };
    use rand_distr::Standard;

    #[test]
    fn special_ifft_fft() {
        let rng = &mut StdRng::seed_from_u64(OsRng.next_u64());
        let (log_qi, num_qs) = (45, 8);
        for log_n in 1..10 {
            let param = Ckks::param_gen(log_n, log_qi, num_qs);
            let evals = rng.sample_iter(Standard).take(param.l()).collect_vec();
            let coeffs = special_ifft(&param, evals.clone());
            izip!(param.pow5(), &evals)
                .for_each(|(k, eval)| assert_eq_complex!(horner(&coeffs, &param.psi()[*k]), eval));
            izip!(evals, special_fft(&param, coeffs)).for_each(|(a, b)| assert_eq_complex!(a, b));
        }
    }

    #[test]
    fn encrypt_decrypt() {
        let rng = &mut StdRng::seed_from_u64(OsRng.next_u64());
        let (log_qi, num_qs) = (45, 8);
        for log_n in 1..10 {
            let param = Ckks::param_gen(log_n, log_qi, num_qs);
            let (sk, pk, _) = Ckks::key_gen(&param, rng);

            let m = CkksCleartext(rng.sample_iter(Standard).take(param.l()).collect_vec());
            let pt = Ckks::encode(&param, m.clone());
            let ct = Ckks::encrypt(&param, &pk, pt, rng);

            izip!(m.0, Ckks::decode(&param, Ckks::decrypt(&param, &sk, ct)).0)
                .for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs, 40));
        }
    }

    #[test]
    fn eval_add() {
        let rng = &mut StdRng::seed_from_u64(OsRng.next_u64());
        let (log_qi, num_qs) = (45, 8);
        for log_n in 1..10 {
            let param = Ckks::param_gen(log_n, log_qi, num_qs);
            let (sk, pk, ek) = Ckks::key_gen(&param, rng);

            let m0 = CkksCleartext(rng.sample_iter(Standard).take(param.l()).collect_vec());
            let m1 = CkksCleartext(rng.sample_iter(Standard).take(param.l()).collect_vec());
            let m2 = izip!(&m0.0, &m1.0).map(|(m0, m1)| m0 + m1).collect_vec();
            let ct0 = Ckks::encrypt(&param, &pk, Ckks::encode(&param, m0), rng);
            let ct1 = Ckks::encrypt(&param, &pk, Ckks::encode(&param, m1), rng);
            let ct2 = Ckks::eval_add(&param, &ek, ct0, ct1);

            izip!(m2, Ckks::decode(&param, Ckks::decrypt(&param, &sk, ct2)).0)
                .for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs, 40));
        }
    }

    #[test]
    fn eval_mul() {
        let rng = &mut StdRng::seed_from_u64(OsRng.next_u64());
        let (log_qi, num_qs) = (45, 8);
        for log_n in 1..10 {
            let param = Ckks::param_gen(log_n, log_qi, num_qs);
            let (sk, pk, ek) = Ckks::key_gen(&param, rng);

            let m0 = CkksCleartext(rng.sample_iter(Standard).take(param.l()).collect_vec());
            let m1 = CkksCleartext(rng.sample_iter(Standard).take(param.l()).collect_vec());
            let m2 = izip!(&m0.0, &m1.0).map(|(m0, m1)| m0 * m1).collect_vec();
            let ct0 = Ckks::encrypt(&param, &pk, Ckks::encode(&param, m0), rng);
            let ct1 = Ckks::encrypt(&param, &pk, Ckks::encode(&param, m1), rng);
            let ct2 = Ckks::eval_mul(&param, &ek, ct0, ct1);

            izip!(m2, Ckks::decode(&param, Ckks::decrypt(&param, &sk, ct2)).0)
                .for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs, 40));
        }
    }
}
