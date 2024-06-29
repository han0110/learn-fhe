use derive_more::{Add, Sub};
use itertools::{chain, izip, Itertools};
use num_bigint::{BigInt, BigUint};
use rand::RngCore;
use util::{bit_reverse, dg, powers, two_adic_primes, zo, AVec, BigFloat, Complex, CrtRq, Zq};

#[derive(Clone, Debug)]
pub struct Ckks;

#[derive(Clone, Debug)]
pub struct CkksParam {
    log_n: usize,
    pow5: Vec<usize>,
    psi: Vec<Complex>,
    big_l: usize,
    qs: Vec<u64>,
    ps: Vec<u64>,
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

    pub fn big_l(&self) -> usize {
        self.big_l
    }

    pub fn qs(&self) -> &[u64] {
        &self.qs
    }

    pub fn ps(&self) -> &[u64] {
        &self.ps
    }

    pub fn qps(&self) -> Vec<u64> {
        chain![self.qs(), self.ps()].copied().collect()
    }

    pub fn p(&self) -> BigUint {
        self.ps.iter().product()
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
pub struct CkksEvaluationKey {
    mul: CkksKeySwitchingKey,
}

#[derive(Clone, Debug)]
pub struct CkksKeySwitchingKey(CrtRq, CrtRq);

impl CkksKeySwitchingKey {
    pub fn a(&self) -> &CrtRq {
        &self.1
    }

    pub fn b(&self) -> &CrtRq {
        &self.0
    }
}

#[derive(Clone, Debug)]
pub struct CkksCleartext(AVec<Complex>);

#[derive(Clone, Debug)]
pub struct CkksPlaintext(CrtRq);

#[derive(Clone, Debug, Add, Sub)]
pub struct CkksCiphertext(CrtRq, CrtRq);

impl CkksCiphertext {
    pub fn a(&self) -> &CrtRq {
        &self.1
    }

    pub fn b(&self) -> &CrtRq {
        &self.0
    }
}

impl Ckks {
    pub fn param_gen(log_n: usize, log_qi: usize, big_l: usize) -> CkksParam {
        assert!(log_n >= 1);
        assert!(big_l > 1);

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

        let mut primes = two_adic_primes(log_qi, log_n + 1);
        let qs = primes.by_ref().take(big_l).collect_vec();
        let ps = primes.by_ref().take(big_l).collect_vec();

        let scale = BigFloat::from(*qs.last().unwrap());

        CkksParam {
            log_n,
            psi,
            pow5,
            big_l,
            qs,
            ps,
            scale,
        }
    }

    pub fn key_gen(
        param: &CkksParam,
        rng: &mut impl RngCore,
    ) -> (CkksSecretKey, CkksPublicKey, CkksEvaluationKey) {
        let sk = CkksSecretKey(AVec::sample(param.n(), zo(0.5), rng));
        let pk = {
            let a = CrtRq::sample_uniform(param.n(), param.qs(), rng) as CrtRq;
            let e = CrtRq::sample_i8(param.n(), param.qs(), dg(3.2, 6), rng);
            let b = -(&a * &sk.0) + e;
            CkksPublicKey(b, a)
        };
        let ek = {
            let mul = {
                let s_prime = CrtRq::from_i8(&sk.0, &param.qps()).square();
                Self::ksk_gen(param, &sk, &s_prime, rng)
            };
            CkksEvaluationKey { mul }
        };
        (sk, pk, ek)
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
        let e0 = &CrtRq::sample_i8(param.n(), param.qs(), dg(3.2, 6), rng);
        let e1 = &CrtRq::sample_i8(param.n(), param.qs(), dg(3.2, 6), rng);
        let c0 = pk0 * u + e0 + pt;
        let c1 = pk1 * u + e1;
        CkksCiphertext(c0, c1)
    }

    pub fn decrypt(
        _: &CkksParam,
        CkksSecretKey(sk): &CkksSecretKey,
        ct: CkksCiphertext,
    ) -> CkksPlaintext {
        let pt = ct.b() + ct.a() * sk;
        CkksPlaintext(pt)
    }

    pub fn mul(
        param: &CkksParam,
        ek: &CkksEvaluationKey,
        ct0: CkksCiphertext,
        ct1: CkksCiphertext,
    ) -> CkksCiphertext {
        let [d0, d1, d2] = [
            ct0.b() * ct1.b(),
            ct0.b() * ct1.a() + ct0.a() * ct1.b(),
            ct0.a() * ct1.a(),
        ];
        let d2 = &d2.extend_bases(param.ps());
        let c0 = (d0 + (d2 * ek.mul.b()).rescale_k(param.big_l())).rescale();
        let c1 = (d1 + (d2 * ek.mul.a()).rescale_k(param.big_l())).rescale();
        CkksCiphertext(c0, c1)
    }

    pub fn ksk_gen(
        param: &CkksParam,
        CkksSecretKey(sk): &CkksSecretKey,
        s_prime: &CrtRq,
        rng: &mut impl RngCore,
    ) -> CkksKeySwitchingKey {
        let a = CrtRq::sample_uniform(param.n(), &param.qps(), rng) as CrtRq;
        let e = CrtRq::sample_i8(param.n(), &param.qps(), dg(3.2, 6), rng);
        let b = -(&a * sk) + e + (s_prime * param.p());
        CkksKeySwitchingKey(b, a)
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
    use rand::{rngs::StdRng, SeedableRng};
    use rand_distr::Standard;
    use util::{assert_eq_complex, horner, vec_with, AVec, Complex};

    #[test]
    fn special_ifft_fft() {
        let rng = &mut StdRng::from_entropy();
        let (log_qi, big_l) = (55, 8);
        for log_n in 1..10 {
            let param = Ckks::param_gen(log_n, log_qi, big_l);
            let evals = AVec::sample(param.l(), Standard, rng);
            let coeffs = special_ifft(&param, evals.clone());
            izip!(param.pow5(), &evals)
                .for_each(|(k, eval)| assert_eq_complex!(horner(&coeffs, &param.psi()[*k]), eval));
            izip!(evals, special_fft(&param, coeffs)).for_each(|(a, b)| assert_eq_complex!(a, b));
        }
    }

    #[test]
    fn encrypt_decrypt() {
        let rng = &mut StdRng::from_entropy();
        let (log_qi, big_l) = (55, 8);
        for log_n in 1..10 {
            let param = Ckks::param_gen(log_n, log_qi, big_l);
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
        let rng = &mut StdRng::from_entropy();
        let (log_qi, big_l) = (55, 8);
        for log_n in 1..10 {
            let param = Ckks::param_gen(log_n, log_qi, big_l);
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
    fn mul() {
        let rng = &mut StdRng::from_entropy();
        let (log_qi, big_l) = (55, 8);
        for log_n in 1..10 {
            let param = Ckks::param_gen(log_n, log_qi, big_l);
            let (sk, pk, ek) = Ckks::key_gen(&param, rng);
            let mul_m = |m0, m1| AVec::ew_mul(&m0, &m1);
            let mul_ct = |ct0, ct1| Ckks::mul(&param, &ek, ct0, ct1);
            let ms = vec_with![|| AVec::<Complex>::sample(param.l(), Standard, rng); big_l - 1];
            let pts = vec_with!(|m| Ckks::encode(&param, CkksCleartext(m.clone())); &ms);
            let cts = vec_with!(|pt| Ckks::encrypt(&param, &pk, pt, rng); pts);
            let m = ms.into_iter().reduce(mul_m).unwrap();
            let ct = cts.into_iter().reduce(mul_ct).unwrap();
            izip!(m, Ckks::decode(&param, Ckks::decrypt(&param, &sk, ct)).0)
                .for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs, 32));
        }
    }
}
