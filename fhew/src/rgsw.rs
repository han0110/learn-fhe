use crate::{
    rlwe::{Rlwe, RlweCiphertext, RlweParam, RlwePlaintext, RlwePublicKey, RlweSecretKey},
    util::{AVec, Decomposable, Decomposor, Dot, Fq, Poly},
};
use core::{iter::repeat_with, ops::Deref};
use itertools::{chain, izip, Either, Itertools};
use rand::RngCore;

#[derive(Debug)]
pub struct Rgsw;

#[derive(Clone, Copy, Debug)]
pub struct RgswParam {
    rlwe: RlweParam,
    decomposor: Decomposor,
}

impl Deref for RgswParam {
    type Target = RlweParam;

    fn deref(&self) -> &Self::Target {
        &self.rlwe
    }
}

impl RgswParam {
    pub fn new(rlwe: RlweParam, log_b: usize, d: usize) -> Self {
        let decomposor = Decomposor::new(rlwe.q(), log_b, d);
        Self { rlwe, decomposor }
    }
}

pub type RgswSecretKey = RlweSecretKey;

pub type RgswPublicKey = RlwePublicKey;

#[derive(Debug)]
pub struct RgswPlaintext(pub(crate) Poly<Fq>);

#[derive(Debug)]
pub struct RgswCiphertext(Vec<RlweCiphertext>);

impl RgswCiphertext {
    pub fn a(&self) -> impl Iterator<Item = &Poly<Fq>> {
        self.0.iter().map(|ct| ct.a())
    }

    pub fn b(&self) -> impl Iterator<Item = &Poly<Fq>> {
        self.0.iter().map(|ct| ct.b())
    }
}

impl Rgsw {
    pub fn key_gen(param: &RgswParam, rng: &mut impl RngCore) -> (RgswSecretKey, RgswPublicKey) {
        Rlwe::key_gen(param, rng)
    }

    pub fn encode(param: &RgswParam, m: &Poly<Fq>) -> RgswPlaintext {
        assert_eq!(m.n(), param.n());
        assert!(m.iter().all(|m| m.q() == param.p()));

        let to_fq = |m: &Fq| Fq::from_u64(param.q(), m.into());
        RgswPlaintext(m.iter().map(to_fq).collect())
    }

    pub fn decode(param: &RgswParam, pt: &RgswPlaintext) -> Poly<Fq> {
        let to_fp = |m: &Fq| Fq::from_u64(param.p(), m.into());
        pt.0.iter().map(to_fp).collect()
    }

    pub fn sk_encrypt(
        param: &RgswParam,
        sk: &RgswSecretKey,
        pt: &RgswPlaintext,
        rng: &mut impl RngCore,
    ) -> RgswCiphertext {
        Self::encrypt(param, Either::Left(sk), pt, rng)
    }

    pub fn pk_encrypt(
        param: &RgswParam,
        pk: &RgswPublicKey,
        pt: &RgswPlaintext,
        rng: &mut impl RngCore,
    ) -> RgswCiphertext {
        Self::encrypt(param, Either::Right(pk), pt, rng)
    }

    fn encrypt(
        param: &RgswParam,
        key: Either<&RgswSecretKey, &RgswPublicKey>,
        pt: &RgswPlaintext,
        rng: &mut impl RngCore,
    ) -> RgswCiphertext {
        let zero = RlwePlaintext(Poly::zero(param.n(), param.q()));
        let rlwe_encrypt_zero = || match key {
            Either::Left(sk) => Rlwe::sk_encrypt(param, sk, &zero, rng),
            Either::Right(pk) => Rlwe::pk_encrypt(param, pk, &zero, rng),
        };
        let mut cts = repeat_with(rlwe_encrypt_zero)
            .take(2 * param.decomposor.d())
            .collect_vec();
        let (c0, c1) = cts.split_at_mut(param.decomposor.d());
        izip!(c0, param.decomposor.bases()).for_each(|(ct, bi)| ct.0 += &pt.0 * bi);
        izip!(c1, param.decomposor.bases()).for_each(|(ct, bi)| ct.1 += &pt.0 * bi);
        RgswCiphertext(cts)
    }

    pub fn decrypt(param: &RgswParam, sk: &RgswSecretKey, ct: &RgswCiphertext) -> RgswPlaintext {
        let pt = Rlwe::decrypt(param, sk, ct.0.last().unwrap());
        RgswPlaintext(pt.0.rounding_shr(param.decomposor.log_bases().last().unwrap()))
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
        let ct1_limbs = chain![[ct1.a(), ct1.b()]]
            .flat_map(|v| param.decomposor.decompose(v))
            .collect::<AVec<_>>();
        let a = ct0.a().dot(&ct1_limbs);
        let b = ct0.b().dot(&ct1_limbs);
        RlweCiphertext(a, b)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        rgsw::{Rgsw, RgswParam},
        rlwe::{test::testing_n_q, Rlwe, RlweParam},
        util::Poly,
    };
    use core::array::from_fn;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn encrypt_decrypt() {
        let mut rng = StdRng::from_entropy();
        let (log_n_range, log_q, p, log_b, d) = (0..10, 45, 1 << 4, 5, 9);
        for (log_n, q) in testing_n_q(log_n_range, log_q) {
            let param = RgswParam::new(RlweParam::new(q, p, log_n), log_b, d);
            let (sk, pk) = Rgsw::key_gen(&param, &mut rng);
            let m = Poly::sample_fq_uniform(param.n(), p, &mut rng);
            let pt = Rgsw::encode(&param, &m);
            let ct0 = Rgsw::sk_encrypt(&param, &sk, &pt, &mut rng);
            let ct1 = Rgsw::pk_encrypt(&param, &pk, &pt, &mut rng);
            assert_eq!(m, Rgsw::decode(&param, &Rgsw::decrypt(&param, &sk, &ct0)));
            assert_eq!(m, Rgsw::decode(&param, &Rgsw::decrypt(&param, &sk, &ct1)));
        }
    }

    #[test]
    fn eval_add() {
        let mut rng = StdRng::from_entropy();
        let (log_n_range, log_q, p, log_b, d) = (0..10, 45, 1 << 4, 5, 9);
        for (log_n, q) in testing_n_q(log_n_range, log_q) {
            let param = RgswParam::new(RlweParam::new(q, p, log_n), log_b, d);
            let (sk, pk) = Rgsw::key_gen(&param, &mut rng);
            let [m0, m1] = &from_fn(|_| Poly::sample_fq_uniform(param.n(), p, &mut rng));
            let [pt0, pt1] = [m0, m1].map(|m| Rgsw::encode(&param, m));
            let [ct0, ct1] = [pt0, pt1].map(|pt| Rgsw::pk_encrypt(&param, &pk, &pt, &mut rng));
            let ct2 = Rgsw::eval_add(&param, &ct0, &ct1);
            let m2 = m0 + m1;
            assert_eq!(m2, Rgsw::decode(&param, &Rgsw::decrypt(&param, &sk, &ct2)));
        }
    }

    #[test]
    fn eval_sub() {
        let mut rng = StdRng::from_entropy();
        let (log_n_range, log_q, p, log_b, d) = (0..10, 45, 1 << 4, 5, 9);
        for (log_n, q) in testing_n_q(log_n_range, log_q) {
            let param = RgswParam::new(RlweParam::new(q, p, log_n), log_b, d);
            let (sk, pk) = Rgsw::key_gen(&param, &mut rng);
            let [m0, m1] = &from_fn(|_| Poly::sample_fq_uniform(param.n(), p, &mut rng));
            let [pt0, pt1] = [m0, m1].map(|m| Rgsw::encode(&param, m));
            let [ct0, ct1] = [pt0, pt1].map(|pt| Rgsw::pk_encrypt(&param, &pk, &pt, &mut rng));
            let ct2 = Rgsw::eval_sub(&param, &ct0, &ct1);
            let m2 = m0 - m1;
            assert_eq!(m2, Rgsw::decode(&param, &Rgsw::decrypt(&param, &sk, &ct2)));
        }
    }

    #[test]
    fn external_product() {
        let mut rng = StdRng::from_entropy();
        let (log_n_range, log_q, p, log_b, d) = (0..10, 45, 1 << 4, 5, 9);
        for (log_n, q) in testing_n_q(log_n_range, log_q) {
            let param = RgswParam::new(RlweParam::new(q, p, log_n), log_b, d);
            let (sk, pk) = Rgsw::key_gen(&param, &mut rng);
            let [m0, m1] = &from_fn(|_| Poly::sample_fq_uniform(param.n(), p, &mut rng));
            let ct0 = Rgsw::pk_encrypt(&param, &pk, &Rgsw::encode(&param, m0), &mut rng);
            let ct1 = Rlwe::pk_encrypt(&param, &pk, &Rlwe::encode(&param, m1), &mut rng);
            let ct2 = Rgsw::external_product(&param, &ct0, &ct1);
            let m2 = m0 * m1;
            assert_eq!(m2, Rlwe::decode(&param, &Rlwe::decrypt(&param, &sk, &ct2)));
        }
    }
}
