use crate::{
    rlwe::{Rlwe, RlweCiphertext, RlweParam, RlwePlaintext, RlwePublicKey, RlweSecretKey},
    util::{AVec, Dot, Fq, Poly},
};
use core::{iter::repeat_with, ops::Deref};
use itertools::{chain, izip, Itertools};
use rand::RngCore;

pub struct Rgsw;

#[derive(Clone, Copy, Debug)]
pub struct RgswParam {
    rlwe: RlweParam,
    log_b: usize,
    k: usize,
}

impl Deref for RgswParam {
    type Target = RlweParam;

    fn deref(&self) -> &Self::Target {
        &self.rlwe
    }
}

impl RgswParam {
    pub fn round_bits(&self) -> usize {
        let log_q_ceil = self.q().next_power_of_two().ilog2() as usize;
        log_q_ceil.saturating_sub(self.log_b * self.k)
    }

    pub fn log_bs(&self) -> impl Iterator<Item = usize> {
        (self.round_bits()..).step_by(self.log_b).take(self.k)
    }

    pub fn bs(&self) -> impl Iterator<Item = Fq> + '_ {
        self.log_bs().map(|bits| Fq::from_u64(self.q(), 1 << bits))
    }
}

pub type RgswSecretKey = RlweSecretKey;

pub type RgswPublicKey = RlwePublicKey;

pub type RgswPlaintext = RlwePlaintext;

pub struct RgswCiphertext(Vec<RlweCiphertext>);

impl Rgsw {
    pub fn param_gen(rlwe: RlweParam, log_b: usize, k: usize) -> RgswParam {
        RgswParam { rlwe, log_b, k }
    }

    pub fn key_gen(param: &RgswParam, rng: &mut impl RngCore) -> (RgswSecretKey, RgswPublicKey) {
        Rlwe::key_gen(param, rng)
    }

    pub fn encode(param: &RgswParam, m: &Poly<Fq>) -> RgswPlaintext {
        assert_eq!(m.n(), param.n());
        assert!(m.iter().all(|m| m.q() == param.p()));

        let to_fq = |m: &Fq| Fq::from_u64(param.q(), m.into());
        RlwePlaintext(m.iter().map(to_fq).collect())
    }

    pub fn decode(param: &RgswParam, pt: &RgswPlaintext) -> Poly<Fq> {
        let to_fp = |m: &Fq| Fq::from_u64(param.p(), m.into());
        pt.0.iter().map(to_fp).collect()
    }

    pub fn encrypt(
        param: &RgswParam,
        pk: &RgswPublicKey,
        pt: &RgswPlaintext,
        rng: &mut impl RngCore,
    ) -> RgswCiphertext {
        let zero = RlwePlaintext(Poly::zero(param.n(), param.q()));
        let mut cts = repeat_with(|| Rlwe::encrypt(param, pk, &zero, rng))
            .take(2 * param.k)
            .collect_vec();
        izip!(&mut cts[..param.k], param.bs()).for_each(|(ct, bi)| ct.0 += &pt.0 * bi);
        izip!(&mut cts[param.k..], param.bs()).for_each(|(ct, bi)| ct.1 += &pt.0 * bi);
        RgswCiphertext(cts)
    }

    pub fn decrypt(param: &RgswParam, sk: &RgswSecretKey, ct: &RgswCiphertext) -> RgswPlaintext {
        let pt = Rlwe::decrypt(param, sk, ct.0.last().unwrap());
        RlwePlaintext(pt.0.round_shr(param.log_bs().last().unwrap()))
    }

    pub fn eval_add(
        param: &RgswParam,
        ct0: &RgswCiphertext,
        ct1: &RgswCiphertext,
    ) -> RgswCiphertext {
        let eval_add = |(ct0, ct1)| Rlwe::eval_add(param, ct0, ct1);
        RgswCiphertext(izip!(&ct0.0, &ct1.0).map(eval_add).collect())
    }

    pub fn eval_sub(
        param: &RgswParam,
        ct0: &RgswCiphertext,
        ct1: &RgswCiphertext,
    ) -> RgswCiphertext {
        let eval_sub = |(ct0, ct1)| Rlwe::eval_sub(param, ct0, ct1);
        RgswCiphertext(izip!(&ct0.0, &ct1.0).map(eval_sub).collect())
    }

    pub fn external_product(
        param: &RgswParam,
        ct0: &RgswCiphertext,
        ct1: &RlweCiphertext,
    ) -> RlweCiphertext {
        let ct1 = chain![[ct1.a(), ct1.b()]]
            .map(|v| v.round_shr(param.round_bits()))
            .flat_map(|v| v.decompose(param.log_b, param.k))
            .collect::<AVec<_>>();
        let a = ct1.dot(ct0.0.iter().map(|ct| ct.a()));
        let b = ct1.dot(ct0.0.iter().map(|ct| ct.b()));
        RlweCiphertext(a, b)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        rgsw::Rgsw,
        rlwe::Rlwe,
        util::{two_adic_primes, Poly},
    };
    use core::array::from_fn;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn encrypt_decrypt() {
        let mut rng = StdRng::from_entropy();
        let (log_q, log_p, log_b, k) = (45, 4, 5, 9);
        for log_n in 0..10 {
            let (n, p) = (1 << log_n, 1 << log_p);
            for q in two_adic_primes(log_q, log_n + 1).take(10) {
                let param = Rgsw::param_gen(Rlwe::param_gen(q, p, n), log_b, k);
                let (sk, pk) = Rgsw::key_gen(&param, &mut rng);
                let m = Poly::sample_fq_uniform(n, p, &mut rng);
                let pt = Rgsw::encode(&param, &m);
                let ct = Rgsw::encrypt(&param, &pk, &pt, &mut rng);
                assert_eq!(m, Rgsw::decode(&param, &Rgsw::decrypt(&param, &sk, &ct)));
            }
        }
    }

    #[test]
    fn eval_add() {
        let mut rng = StdRng::from_entropy();
        let (log_q, log_p, log_b, k) = (45, 4, 5, 9);
        for log_n in 0..10 {
            let (n, p) = (1 << log_n, 1 << log_p);
            for q in two_adic_primes(log_q, log_n + 1).take(10) {
                let param = Rgsw::param_gen(Rlwe::param_gen(q, p, n), log_b, k);
                let (sk, pk) = Rgsw::key_gen(&param, &mut rng);
                let [m0, m1] = &from_fn(|_| Poly::sample_fq_uniform(n, p, &mut rng));
                let [pt0, pt1] = [m0, m1].map(|m| Rgsw::encode(&param, m));
                let [ct0, ct1] = [pt0, pt1].map(|pt| Rgsw::encrypt(&param, &pk, &pt, &mut rng));
                let ct2 = Rgsw::eval_add(&param, &ct0, &ct1);
                let m2 = m0 + m1;
                assert_eq!(m2, Rgsw::decode(&param, &Rgsw::decrypt(&param, &sk, &ct2)));
            }
        }
    }

    #[test]
    fn eval_sub() {
        let mut rng = StdRng::from_entropy();
        let (log_q, log_p, log_b, k) = (45, 4, 5, 9);
        for log_n in 0..10 {
            let (n, p) = (1 << log_n, 1 << log_p);
            for q in two_adic_primes(log_q, log_n + 1).take(10) {
                let param = Rgsw::param_gen(Rlwe::param_gen(q, p, n), log_b, k);
                let (sk, pk) = Rgsw::key_gen(&param, &mut rng);
                let [m0, m1] = &from_fn(|_| Poly::sample_fq_uniform(n, p, &mut rng));
                let [pt0, pt1] = [m0, m1].map(|m| Rgsw::encode(&param, m));
                let [ct0, ct1] = [pt0, pt1].map(|pt| Rgsw::encrypt(&param, &pk, &pt, &mut rng));
                let ct2 = Rgsw::eval_sub(&param, &ct0, &ct1);
                let m2 = m0 - m1;
                assert_eq!(m2, Rgsw::decode(&param, &Rgsw::decrypt(&param, &sk, &ct2)));
            }
        }
    }

    #[test]
    fn external_product() {
        let mut rng = StdRng::from_entropy();
        let (log_q, log_p, log_b, k) = (45, 4, 5, 9);
        for log_n in 0..10 {
            let (n, p) = (1 << log_n, 1 << log_p);
            for q in two_adic_primes(log_q, log_n + 1).take(10) {
                let param = Rgsw::param_gen(Rlwe::param_gen(q, p, n), log_b, k);
                let (sk, pk) = Rgsw::key_gen(&param, &mut rng);
                let [m0, m1] = &from_fn(|_| Poly::sample_fq_uniform(n, p, &mut rng));
                let ct0 = Rgsw::encrypt(&param, &pk, &Rgsw::encode(&param, m0), &mut rng);
                let ct1 = Rlwe::encrypt(&param, &pk, &Rlwe::encode(&param, m1), &mut rng);
                let ct2 = Rgsw::external_product(&param, &ct0, &ct1);
                let m2 = m0 * m1;
                assert_eq!(m2, Rlwe::decode(&param, &Rlwe::decrypt(&param, &sk, &ct2)));
            }
        }
    }
}
