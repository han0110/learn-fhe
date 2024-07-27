use crate::rlwe::{Rlwe, RlweCiphertext, RlweParam, RlwePlaintext, RlwePublicKey, RlweSecretKey};
use core::{borrow::Borrow, iter::repeat_with};
use derive_more::{Add, AddAssign, Deref, Sub, SubAssign};
use itertools::{chain, izip, Either, Itertools};
use rand::RngCore;
use util::{AVec, Base2Decomposable, Base2Decomposor, Dot, Rq, Zq};

#[derive(Debug)]
pub struct Rgsw;

#[derive(Clone, Copy, Debug, Deref)]
pub struct RgswParam {
    #[deref]
    rlwe: RlweParam,
    decomposor: Base2Decomposor<Zq>,
}

impl RgswParam {
    pub fn new(rlwe: RlweParam, log_b: usize, d: usize) -> Self {
        let decomposor = Base2Decomposor::<Zq>::new(rlwe.q(), log_b, d);
        Self { rlwe, decomposor }
    }

    pub fn decomposor(&self) -> &Base2Decomposor<Zq> {
        &self.decomposor
    }
}

pub type RgswSecretKey = RlweSecretKey;

pub type RgswPublicKey = RlwePublicKey;

#[derive(Clone, Debug)]
pub struct RgswPlaintext(pub(crate) Rq);

#[derive(Clone, Debug, Add, Sub, AddAssign, SubAssign)]
pub struct RgswCiphertext(AVec<RlweCiphertext>);

impl RgswCiphertext {
    pub fn a(&self) -> impl Iterator<Item = &Rq> {
        self.0.iter().map(RlweCiphertext::a)
    }

    pub fn b(&self) -> impl Iterator<Item = &Rq> {
        self.0.iter().map(RlweCiphertext::b)
    }
}

impl Rgsw {
    pub fn key_gen(param: &RgswParam, rng: &mut impl RngCore) -> (RgswSecretKey, RgswPublicKey) {
        Rlwe::key_gen(param, rng)
    }

    pub fn encode(param: &RgswParam, m: Rq) -> RgswPlaintext {
        assert_eq!(m.n(), param.n());
        assert!(m.iter().all(|m| m.q() == param.p()));
        let to_zq = |m: Zq| Zq::from_u64(param.q(), m.into());
        RgswPlaintext(m.into_iter().map(to_zq).collect())
    }

    pub fn decode(param: &RgswParam, pt: RgswPlaintext) -> Rq {
        let to_zp = |m: Zq| Zq::from_u64(param.p(), m.into());
        pt.0.into_iter().map(to_zp).collect()
    }

    pub fn sk_encrypt(
        param: &RgswParam,
        sk: &RgswSecretKey,
        pt: RgswPlaintext,
        rng: &mut impl RngCore,
    ) -> RgswCiphertext {
        Rgsw::encrypt(param, Either::Left(sk), pt, rng)
    }

    pub fn pk_encrypt(
        param: &RgswParam,
        pk: &RgswPublicKey,
        pt: RgswPlaintext,
        rng: &mut impl RngCore,
    ) -> RgswCiphertext {
        Rgsw::encrypt(param, Either::Right(pk), pt, rng)
    }

    fn encrypt(
        param: &RgswParam,
        key: Either<&RgswSecretKey, &RgswPublicKey>,
        RgswPlaintext(pt): RgswPlaintext,
        rng: &mut impl RngCore,
    ) -> RgswCiphertext {
        let pt = param.decomposor().power_up(pt).collect_vec();
        let mut ct = {
            let zero = RlwePlaintext(Rq::zero(param.q(), param.n()));
            let rlwe_encrypt_zero = || match key {
                Either::Left(sk) => Rlwe::sk_encrypt(param, sk, zero.clone(), rng),
                Either::Right(pk) => Rlwe::pk_encrypt(param, pk, zero.clone(), rng),
            };
            repeat_with(rlwe_encrypt_zero)
                .take(2 * param.decomposor().d())
                .collect::<AVec<_>>()
        };
        let (c0, c1) = ct.split_at_mut(param.decomposor().d());
        izip!(c0, &pt).for_each(|(ct, pt)| ct.0 += pt);
        izip!(c1, &pt).for_each(|(ct, pt)| ct.1 += pt);
        RgswCiphertext(ct)
    }

    pub fn decrypt(
        param: &RgswParam,
        sk: &RgswSecretKey,
        RgswCiphertext(ct): RgswCiphertext,
    ) -> RgswPlaintext {
        let pt = Rlwe::decrypt(param, sk, ct.into_iter().last().unwrap()).0;
        RgswPlaintext(pt.rounding_shr(param.decomposor().log_bases().last().unwrap()))
    }

    pub fn external_product(
        param: &RgswParam,
        ct0: impl Borrow<RgswCiphertext>,
        ct1: impl Borrow<RlweCiphertext>,
    ) -> RlweCiphertext {
        let (ct0, ct1) = (ct0.borrow(), ct1.borrow());
        let ct1_limbs = chain![[ct1.a(), ct1.b()]]
            .flat_map(|v| param.decomposor().decompose(v))
            .collect_vec();
        let a = ct0.a().dot(&ct1_limbs);
        let b = ct0.b().dot(&ct1_limbs);
        RlweCiphertext(a, b)
    }

    pub fn internal_product(
        param: &RgswParam,
        ct0: impl Borrow<RgswCiphertext>,
        ct1: impl Borrow<RgswCiphertext>,
    ) -> RgswCiphertext {
        let (ct0_a, ct0_b) = chain![&ct0.borrow().0]
            .map(|c| (c.a().to_evaluation(), c.b().to_evaluation()))
            .unzip::<_, _, Vec<_>, Vec<_>>();
        let ct = chain![&ct1.borrow().0]
            .map(|ct1| {
                let ct1_limbs = chain![[ct1.a(), ct1.b()]]
                    .flat_map(|v| param.decomposor.decompose(v))
                    .map(|p| p.to_evaluation())
                    .collect::<AVec<_>>();
                let a = ct0_a.dot(&ct1_limbs).to_coefficient();
                let b = ct0_b.dot(&ct1_limbs).to_coefficient();
                RlweCiphertext(a, b)
            })
            .collect();
        RgswCiphertext(ct)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        rgsw::{Rgsw, RgswParam},
        rlwe::{test::testing_n_q, Rlwe, RlweParam},
    };
    use core::array::from_fn;
    use rand::thread_rng;
    use util::Rq;

    #[test]
    fn encrypt_decrypt() {
        let mut rng = thread_rng();
        let (log_n_range, log_q, p, log_b, d) = (0..10, 45, 1 << 4, 5, 9);
        for (log_n, q) in testing_n_q(log_n_range, log_q) {
            let param = RgswParam::new(RlweParam::new(q, p, log_n), log_b, d);
            let (sk, pk) = Rgsw::key_gen(&param, &mut rng);
            let m = Rq::sample_uniform(p, param.n(), &mut rng);
            let pt0 = Rgsw::encode(&param, m.clone());
            let pt1 = Rgsw::encode(&param, m.clone());
            let ct0 = Rgsw::sk_encrypt(&param, &sk, pt0, &mut rng);
            let ct1 = Rgsw::pk_encrypt(&param, &pk, pt1, &mut rng);
            assert_eq!(m, Rgsw::decode(&param, Rgsw::decrypt(&param, &sk, ct0)));
            assert_eq!(m, Rgsw::decode(&param, Rgsw::decrypt(&param, &sk, ct1)));
        }
    }

    #[test]
    fn add_sub() {
        let mut rng = thread_rng();
        let (log_n_range, log_q, p, log_b, d) = (0..10, 45, 1 << 4, 5, 9);
        for (log_n, q) in testing_n_q(log_n_range, log_q) {
            let param = RgswParam::new(RlweParam::new(q, p, log_n), log_b, d);
            let (sk, pk) = Rgsw::key_gen(&param, &mut rng);
            let [m0, m1] = &from_fn(|_| Rq::sample_uniform(p, param.n(), &mut rng));
            let [pt0, pt1] = [m0, m1].map(|m| Rgsw::encode(&param, m.clone()));
            let [ct0, ct1] = [pt0, pt1].map(|pt| Rgsw::pk_encrypt(&param, &pk, pt, &mut rng));
            let (m2, ct2) = (m0 + m1, ct0.clone() + ct1.clone());
            let (m3, ct3) = (m0 - m1, ct0.clone() - ct1.clone());
            assert_eq!(m2, Rgsw::decode(&param, Rgsw::decrypt(&param, &sk, ct2)));
            assert_eq!(m3, Rgsw::decode(&param, Rgsw::decrypt(&param, &sk, ct3)));
        }
    }

    #[test]
    fn external_product() {
        let mut rng = thread_rng();
        let (log_n_range, log_q, p, log_b, d) = (0..10, 45, 1 << 4, 5, 9);
        for (log_n, q) in testing_n_q(log_n_range, log_q) {
            let param = RgswParam::new(RlweParam::new(q, p, log_n), log_b, d);
            let (sk, pk) = Rgsw::key_gen(&param, &mut rng);
            let [m0, m1] = from_fn(|_| Rq::sample_uniform(p, param.n(), &mut rng));
            let ct0 = Rgsw::pk_encrypt(&param, &pk, Rgsw::encode(&param, m0.clone()), &mut rng);
            let ct1 = Rlwe::pk_encrypt(&param, &pk, Rlwe::encode(&param, m1.clone()), &mut rng);
            let ct2 = Rgsw::external_product(&param, ct0, ct1);
            let m2 = m0 * m1;
            assert_eq!(m2, Rlwe::decode(&param, Rlwe::decrypt(&param, &sk, ct2)));
        }
    }

    #[test]
    fn internal_product() {
        let mut rng = thread_rng();
        let (log_n_range, log_q, p, log_b, d) = (0..10, 45, 1 << 4, 5, 9);
        for (log_n, q) in testing_n_q(log_n_range, log_q) {
            let param = RgswParam::new(RlweParam::new(q, p, log_n), log_b, d);
            let (sk, pk) = Rgsw::key_gen(&param, &mut rng);
            let [m0, m1] = &from_fn(|_| Rq::sample_uniform(p, param.n(), &mut rng));
            let [pt0, pt1] = [m0, m1].map(|m| Rgsw::encode(&param, m.clone()));
            let [ct0, ct1] = [pt0, pt1].map(|pt| Rgsw::pk_encrypt(&param, &pk, pt, &mut rng));
            let ct2 = Rgsw::internal_product(&param, ct0, ct1);
            let m2 = m0 * m1;
            assert_eq!(m2, Rgsw::decode(&param, Rgsw::decrypt(&param, &sk, ct2)));
        }
    }
}
