use derive_more::{Add, Sub};
use itertools::{chain, izip, Itertools};
use num_bigint::BigInt;
use rand::RngCore;
use util::{
    bit_reverse, dg, horner, powers, two_adic_primes, zo, AVec, BigFloat, Complex, CrtRq, Zq,
};

#[derive(Clone, Debug)]
pub struct Ckks;

#[derive(Clone, Debug)]
pub struct CkksParam {
    log_n: usize,
    pow5: Vec<usize>,
    psi: Vec<Complex>,
    qs: Vec<u64>,
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

    pub fn qs(&self) -> &[u64] {
        &self.qs
    }

    pub fn scale(&self) -> &BigFloat {
        &self.scale
    }
}

#[derive(Clone, Debug)]
pub struct CkksSecretKey(AVec<i8>);

#[derive(Clone, Debug)]
pub struct CkksPublicKey(CrtRq, CrtRq);

#[derive(Clone, Debug)]
pub struct CkksEvaluationKey;

#[derive(Clone, Debug)]
pub struct CkksCleartext(AVec<Complex>);

#[derive(Clone, Debug)]
pub struct CkksPlaintext(CrtRq);

#[derive(Clone, Debug, Add, Sub)]
pub struct CkksCiphertext(AVec<CrtRq>);

impl Ckks {
    pub fn param_gen(log_n: usize, log_qi: usize, num_qs: usize) -> CkksParam {
        assert!(log_n >= 1);

        let n = 1 << log_n;

        let psi = {
            let phase = BigFloat::pi() / BigFloat::from(n);
            let cis = Complex::new(phase.cos(), phase.sin());
            powers(&cis).take(2 * n).collect()
        };

        let pow5 = {
            let five = Zq::from_u64(2 * n as u64, 5);
            five.powers().take(2 * n).map_into().collect()
        };

        let qs = two_adic_primes(log_qi, log_n + 1)
            .take(num_qs)
            .collect_vec();

        let scale = BigFloat::from(*qs.last().unwrap());

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
        let sk = AVec::sample(param.n(), zo(0.5), rng);
        let pk = {
            let a = CrtRq::sample_uniform(param.n(), param.qs(), rng) as CrtRq;
            let e = CrtRq::sample_i8(param.n(), param.qs(), dg(3.2, 6), rng);
            let b = -(&a * &sk) + e;
            CkksPublicKey(b, a)
        };
        let ek = CkksEvaluationKey;
        (CkksSecretKey(sk), pk, ek)
    }

    pub fn encode(param: &CkksParam, CkksCleartext(m): CkksCleartext) -> CkksPlaintext {
        assert_eq!(m.len(), param.l());

        let z = special_ifft(param, m);

        let z_scaled = chain![z.iter().map(|z| &z.re), z.iter().map(|z| &z.im)]
            .map(|z| BigInt::from(z * param.scale()))
            .collect_vec();

        let pt = CrtRq::from_bigint(z_scaled, param.qs());

        CkksPlaintext(pt)
    }

    pub fn decode(param: &CkksParam, CkksPlaintext(pt): CkksPlaintext) -> CkksCleartext {
        assert_eq!(pt.n(), param.n());

        let z_scaled = pt.into_bigint();

        let z = izip!(&z_scaled[..param.l()], &z_scaled[param.l()..])
            .map(|(re, im)| {
                let [re, im] = [re, im].map(|z| BigFloat::from(z) / param.scale());
                Complex::new(re, im)
            })
            .collect();

        let m = special_fft(param, z);

        CkksCleartext(m)
    }

    pub fn encrypt(
        param: &CkksParam,
        CkksPublicKey(pk0, pk1): &CkksPublicKey,
        CkksPlaintext(pt): CkksPlaintext,
        rng: &mut impl RngCore,
    ) -> CkksCiphertext {
        let u = &CrtRq::sample_i8(param.n(), param.qs(), zo(0.5), rng);
        let e0 = &CrtRq::sample_i8(param.n(), param.qs(), dg(3.2, 6), &mut *rng);
        let e1 = &CrtRq::sample_i8(param.n(), param.qs(), dg(3.2, 6), &mut *rng);
        let c0 = pk0 * u + e0 + pt;
        let c1 = pk1 * u + e1;
        CkksCiphertext(AVec::from_iter([c0, c1]))
    }

    pub fn decrypt(
        _: &CkksParam,
        CkksSecretKey(sk): &CkksSecretKey,
        CkksCiphertext(ct): CkksCiphertext,
    ) -> CkksPlaintext {
        let pt = horner(&ct, sk);
        CkksPlaintext(pt)
    }

    pub fn eval_mul(
        _: &CkksParam,
        _: &CkksEvaluationKey,
        CkksCiphertext(ct0): CkksCiphertext,
        CkksCiphertext(ct1): CkksCiphertext,
    ) -> CkksCiphertext {
        assert_eq!(ct0.len(), 2);
        assert_eq!(ct1.len(), 2);
        let ct = [
            &ct0[0] * &ct1[0],
            &ct0[0] * &ct1[1] + &ct0[1] * &ct1[0],
            &ct0[1] * &ct1[1],
        ]
        .map(|c| c.rescale());
        CkksCiphertext(AVec::from_iter(ct))
    }
}

// Algorithm 1 in 2018/1043
fn special_fft(param: &CkksParam, mut w: AVec<Complex>) -> AVec<Complex> {
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

fn special_ifft(param: &CkksParam, mut w: AVec<Complex>) -> AVec<Complex> {
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
    use crate::ckks::{special_fft, special_ifft, Ckks, CkksCleartext};
    use core::array::from_fn;
    use itertools::izip;
    use rand::{
        rngs::{OsRng, StdRng},
        RngCore, SeedableRng,
    };
    use rand_distr::Standard;
    use util::{assert_eq_complex, horner, AVec, Complex};

    #[test]
    fn special_ifft_fft() {
        let rng = &mut StdRng::seed_from_u64(OsRng.next_u64());
        let (log_qi, num_qs) = (55, 8);
        for log_n in 1..10 {
            let param = Ckks::param_gen(log_n, log_qi, num_qs);
            let evals = AVec::sample(param.l(), Standard, rng);
            let coeffs = special_ifft(&param, evals.clone());
            izip!(param.pow5(), &evals)
                .for_each(|(k, eval)| assert_eq_complex!(horner(&coeffs, &param.psi()[*k]), eval));
            izip!(evals, special_fft(&param, coeffs)).for_each(|(a, b)| assert_eq_complex!(a, b));
        }
    }

    #[test]
    fn encrypt_decrypt() {
        let rng = &mut StdRng::seed_from_u64(OsRng.next_u64());
        let (log_qi, num_qs) = (55, 8);
        for log_n in 1..10 {
            let param = Ckks::param_gen(log_n, log_qi, num_qs);
            let (sk, pk, _) = Ckks::key_gen(&param, rng);
            let m = AVec::sample(param.l(), Standard, rng);
            let pt = Ckks::encode(&param, CkksCleartext(m.clone()));
            let ct = Ckks::encrypt(&param, &pk, pt, rng);
            izip!(m, Ckks::decode(&param, Ckks::decrypt(&param, &sk, ct)).0)
                .for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs, 40));
        }
    }

    #[test]
    fn add_sub() {
        let rng = &mut StdRng::seed_from_u64(OsRng.next_u64());
        let (log_qi, num_qs) = (55, 8);
        for log_n in 1..10 {
            let param = Ckks::param_gen(log_n, log_qi, num_qs);
            let (sk, pk, _) = Ckks::key_gen(&param, rng);
            let [m0, m1] = &from_fn(|_| AVec::<Complex>::sample(param.l(), Standard, rng));
            let [pt0, pt1] = [m0, m1].map(|m| Ckks::encode(&param, CkksCleartext(m.clone())));
            let [ct0, ct1] = [pt0, pt1].map(|pt| Ckks::encrypt(&param, &pk, pt, rng));
            let (m2, ct2) = (m0 + m1, ct0.clone() + ct1.clone());
            let (m3, ct3) = (m0 - m1, ct0.clone() - ct1.clone());
            izip!(m2, Ckks::decode(&param, Ckks::decrypt(&param, &sk, ct2)).0)
                .for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs, 40));
            izip!(m3, Ckks::decode(&param, Ckks::decrypt(&param, &sk, ct3)).0)
                .for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs, 40));
        }
    }

    #[test]
    fn eval_mul() {
        let rng = &mut StdRng::seed_from_u64(OsRng.next_u64());
        let (log_qi, num_qs) = (55, 8);
        for log_n in 1..10 {
            let param = Ckks::param_gen(log_n, log_qi, num_qs);
            let (sk, pk, ek) = Ckks::key_gen(&param, rng);
            let [m0, m1] = &from_fn(|_| AVec::<Complex>::sample(param.l(), Standard, rng));
            let [pt0, pt1] = [m0, m1].map(|m| Ckks::encode(&param, CkksCleartext(m.clone())));
            let [ct0, ct1] = [pt0, pt1].map(|pt| Ckks::encrypt(&param, &pk, pt, rng));
            let (m2, ct2) = (m0.ew_mul(m1), Ckks::eval_mul(&param, &ek, ct0, ct1));
            izip!(m2, Ckks::decode(&param, Ckks::decrypt(&param, &sk, ct2)).0)
                .for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs, 40));
        }
    }
}
