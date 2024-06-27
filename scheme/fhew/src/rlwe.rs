use crate::lwe::{LweCiphertext, LweParam, LweSecretKey};
use derive_more::{Add, AddAssign, Deref, Sub, SubAssign};
use itertools::chain;
use rand::RngCore;
use util::{dg, izip_eq, zipstar, zo, AVec, Dot, Rq, Zq};

#[derive(Debug)]
pub struct Rlwe;

#[derive(Clone, Copy, Debug, Deref)]
pub struct RlweParam(LweParam);

impl RlweParam {
    pub fn new(q: u64, p: u64, log_n: usize) -> Self {
        Self(LweParam::new(q, p, 1 << log_n))
    }

    pub fn with_decomposor(self, log_b: usize, d: usize) -> Self {
        Self(self.0.with_decomposor(log_b, d))
    }

    pub fn log_n(&self) -> usize {
        self.n().ilog2() as usize
    }
}

#[derive(Clone, Debug, Deref)]
pub struct RlweSecretKey(LweSecretKey);

impl RlweSecretKey {
    fn as_avec(&self) -> &AVec<i8> {
        &self.0 .0
    }

    fn automorphism(&self, t: i64) -> Self {
        RlweSecretKey(LweSecretKey(self.as_avec().automorphism(t)))
    }
}

#[derive(Clone, Debug)]
pub struct RlwePublicKey(Rq, Rq);

impl RlwePublicKey {
    pub fn a(&self) -> &Rq {
        &self.0
    }

    pub fn b(&self) -> &Rq {
        &self.1
    }
}

#[derive(Clone, Debug)]
pub struct RlweKeySwitchingKey(AVec<RlweCiphertext>);

impl RlweKeySwitchingKey {
    pub fn a(&self) -> impl Iterator<Item = &Rq> {
        self.0.iter().map(|ct| ct.a())
    }

    pub fn b(&self) -> impl Iterator<Item = &Rq> {
        self.0.iter().map(|ct| ct.b())
    }
}

#[derive(Clone, Debug, Deref)]
pub struct RlweAutoKey(i64, #[deref] RlweKeySwitchingKey);

impl RlweAutoKey {
    pub fn t(&self) -> i64 {
        self.0
    }
}

#[derive(Clone, Debug)]
pub struct RlwePlaintext(pub(crate) Rq);

#[derive(Clone, Debug, Add, Sub, AddAssign, SubAssign)]
pub struct RlweCiphertext(pub(crate) Rq, pub(crate) Rq);

impl RlweCiphertext {
    pub fn a(&self) -> &Rq {
        &self.0
    }

    pub fn b(&self) -> &Rq {
        &self.1
    }

    pub fn automorphism(&self, k: i64) -> Self {
        Self(self.a().automorphism(k), self.b().automorphism(k))
    }
}

impl From<RlwePlaintext> for RlweCiphertext {
    fn from(RlwePlaintext(b): RlwePlaintext) -> Self {
        Self(Rq::zero(b.n(), b[0].q()), b)
    }
}

impl Rlwe {
    pub const AUTO_G: i64 = 5;

    pub fn sk_gen(param: &RlweParam, rng: &mut impl RngCore) -> RlweSecretKey {
        RlweSecretKey(LweSecretKey(AVec::sample(param.n(), &dg(3.2, 6), rng)))
    }

    pub fn pk_gen(param: &RlweParam, sk: &RlweSecretKey, rng: &mut impl RngCore) -> RlwePublicKey {
        let zero = RlwePlaintext(Rq::zero(param.n(), param.q()));
        let RlweCiphertext(a, b) = Rlwe::sk_encrypt(param, sk, zero, rng);
        RlwePublicKey(a, b)
    }

    pub fn key_gen(param: &RlweParam, rng: &mut impl RngCore) -> (RlweSecretKey, RlwePublicKey) {
        let sk = Rlwe::sk_gen(param, rng);
        let pk = Rlwe::pk_gen(param, &sk, rng);
        (sk, pk)
    }

    pub fn ksk_gen(
        param: &RlweParam,
        sk0: &RlweSecretKey,
        sk1: &RlweSecretKey,
        rng: &mut impl RngCore,
    ) -> RlweKeySwitchingKey {
        let pt = param.decomposor().bases().map(|bi| -sk1.as_avec() * bi);
        let ksk = pt
            .map(|pt| Rlwe::sk_encrypt(param, sk0, RlwePlaintext(pt.into()), rng))
            .collect();
        RlweKeySwitchingKey(ksk)
    }

    pub fn ak_gen(
        param: &RlweParam,
        t: i64,
        sk: &RlweSecretKey,
        rng: &mut impl RngCore,
    ) -> RlweAutoKey {
        let sk_auto = sk.automorphism(t);
        let ksk = Rlwe::ksk_gen(param, sk, &sk_auto, rng);
        RlweAutoKey(t, ksk)
    }

    pub fn encode(param: &RlweParam, m: Rq) -> RlwePlaintext {
        assert_eq!(m.n(), param.n());
        assert!(m.iter().all(|m| m.q() == param.p()));
        let scale_up = |m| Zq::from_f64(param.q(), f64::from(m) * param.delta());
        RlwePlaintext(m.iter().map(scale_up).collect())
    }

    pub fn decode(param: &RlweParam, pt: RlwePlaintext) -> Rq {
        let scale_down = |m| Zq::from_f64(param.p(), f64::from(m) / param.delta());
        pt.0.iter().map(scale_down).collect()
    }

    pub fn sk_encrypt(
        param: &RlweParam,
        sk: &RlweSecretKey,
        pt: RlwePlaintext,
        rng: &mut impl RngCore,
    ) -> RlweCiphertext {
        let a = Rq::sample_zq_uniform(param.n(), param.q(), rng);
        let e = Rq::sample_zq_from_i8(param.n(), param.q(), &dg(3.2, 6), rng);
        let b = &a * sk.as_avec() + e + pt.0;
        RlweCiphertext(a, b)
    }

    pub fn pk_encrypt(
        param: &RlweParam,
        pk: &RlwePublicKey,
        pt: RlwePlaintext,
        rng: &mut impl RngCore,
    ) -> RlweCiphertext {
        let u = &Rq::sample_zq_from_i8(param.n(), param.q(), &zo(0.5), rng);
        let e0 = Rq::sample_zq_from_i8(param.n(), param.q(), &dg(3.2, 6), rng);
        let e1 = Rq::sample_zq_from_i8(param.n(), param.q(), &dg(3.2, 6), rng);
        let a = pk.a() * u + e0;
        let b = pk.b() * u + e1 + pt.0;
        RlweCiphertext(a, b)
    }

    pub fn decrypt(_: &RlweParam, sk: &RlweSecretKey, ct: RlweCiphertext) -> RlwePlaintext {
        let pt = ct.b() - ct.a() * sk.as_avec();
        RlwePlaintext(pt)
    }

    pub fn key_switch(
        param: &RlweParam,
        ksk: &RlweKeySwitchingKey,
        ct: RlweCiphertext,
    ) -> RlweCiphertext {
        let ct_a_limbs = param.decomposor().decompose(ct.a()).collect::<AVec<_>>();
        let a = ksk.a().dot(&ct_a_limbs);
        let b = ksk.b().dot(&ct_a_limbs) + ct.b();
        RlweCiphertext(a, b)
    }

    pub fn automorphism(param: &RlweParam, ak: &RlweAutoKey, ct: RlweCiphertext) -> RlweCiphertext {
        let ct_auto = ct.automorphism(ak.t());
        Rlwe::key_switch(param, ak, ct_auto)
    }

    pub fn sample_extract(param: &RlweParam, ct: RlweCiphertext, i: usize) -> LweCiphertext {
        assert!(i < param.n());
        let a = chain![
            ct.a()[..i + 1].iter().rev().copied(),
            ct.a()[i + 1..].iter().rev().map(|v| -v)
        ]
        .collect();
        let b = ct.b()[i];
        LweCiphertext(a, b)
    }
}

#[derive(Clone, Debug)]
pub struct RlwePublicKeyShare(Rq);

#[derive(Clone, Debug)]
pub struct RlweEncryptionShare(Rq);

#[derive(Clone, Debug)]
pub struct RlweDecryptionShare(Rq);

#[derive(Clone, Debug)]
pub struct RlweKeySwitchingKeyShare(Vec<RlweEncryptionShare>);

pub type RlweAutoKeyShare = RlweKeySwitchingKeyShare;

impl Rlwe {
    pub fn pk_share_gen(
        param: &RlweParam,
        a: &Rq,
        sk: &RlweSecretKey,
        rng: &mut impl RngCore,
    ) -> RlwePublicKeyShare {
        let zero = RlwePlaintext(Rq::zero(param.n(), param.q()));
        let RlweEncryptionShare(b) = Rlwe::share_encrypt(param, a, sk, zero, rng);
        RlwePublicKeyShare(b)
    }

    pub fn pk_share_merge(
        _: &RlweParam,
        a: Rq,
        shares: impl IntoIterator<Item = RlwePublicKeyShare>,
    ) -> RlwePublicKey {
        let b = shares.into_iter().map(|share| share.0).sum();
        RlwePublicKey(a, b)
    }

    pub fn share_encrypt(
        param: &RlweParam,
        a: &Rq,
        sk: &RlweSecretKey,
        pt: RlwePlaintext,
        rng: &mut impl RngCore,
    ) -> RlweEncryptionShare {
        let e = Rq::sample_zq_from_i8(param.n(), param.q(), &dg(3.2, 6), rng);
        let b = a * sk.as_avec() + e + pt.0;
        RlweEncryptionShare(b)
    }

    pub fn encryption_share_merge(
        _: &RlweParam,
        a: Rq,
        shares: impl IntoIterator<Item = RlweEncryptionShare>,
    ) -> RlweCiphertext {
        let b = shares.into_iter().map(|share| share.0).sum();
        RlweCiphertext(a, b)
    }

    pub fn share_decrypt(
        param: &RlweParam,
        sk: &RlweSecretKey,
        a: &Rq,
        rng: &mut impl RngCore,
    ) -> RlweDecryptionShare {
        let e = Rq::sample_zq_from_i8(param.n(), param.q(), &dg(3.2, 6), rng);
        let share = a * sk.as_avec() + e;
        RlweDecryptionShare(share)
    }

    pub fn decryption_share_merge(
        _: &RlweParam,
        b: &Rq,
        shares: impl IntoIterator<Item = RlweDecryptionShare>,
    ) -> RlwePlaintext {
        let pt = b - shares.into_iter().map(|share| share.0).sum::<Rq<_>>();
        RlwePlaintext(pt)
    }

    pub fn ksk_share_gen(
        param: &RlweParam,
        crs: &[Rq],
        sk0: &RlweSecretKey,
        sk1: &RlweSecretKey,
        rng: &mut impl RngCore,
    ) -> RlweKeySwitchingKeyShare {
        let pt = param.decomposor().bases().map(|bi| -sk1.as_avec() * bi);
        let ksk = izip_eq!(crs, pt)
            .map(|(a, pt)| Rlwe::share_encrypt(param, a, sk0, RlwePlaintext(pt.into()), rng))
            .collect();
        RlweKeySwitchingKeyShare(ksk)
    }

    pub fn ksk_share_merge(
        param: &RlweParam,
        crs: Vec<Rq>,
        shares: impl IntoIterator<Item = RlweKeySwitchingKeyShare>,
    ) -> RlweKeySwitchingKey {
        let ksk = izip_eq!(crs, zipstar!(shares, 0))
            .map(|(crs, shares)| Rlwe::encryption_share_merge(param, crs, shares))
            .collect();
        RlweKeySwitchingKey(ksk)
    }

    pub fn ak_share_gen(
        param: &RlweParam,
        t: i64,
        crs: &[Rq],
        sk: &RlweSecretKey,
        rng: &mut impl RngCore,
    ) -> RlweAutoKeyShare {
        let sk_auto = sk.automorphism(t);
        Rlwe::ksk_share_gen(param, crs, sk, &sk_auto, rng)
    }

    pub fn ak_share_merge(
        param: &RlweParam,
        t: i64,
        crs: Vec<Rq>,
        shares: impl IntoIterator<Item = RlweAutoKeyShare>,
    ) -> RlweAutoKey {
        RlweAutoKey(t, Rlwe::ksk_share_merge(param, crs, shares))
    }
}

#[cfg(test)]
pub(crate) mod test {
    use crate::{
        lwe::Lwe,
        rlwe::{Rlwe, RlweParam},
    };
    use core::{array::from_fn, ops::Range};
    use itertools::izip;
    use rand::thread_rng;
    use util::{two_adic_primes, Rq};

    pub(crate) fn testing_n_q(
        log_n_range: Range<usize>,
        log_q: usize,
    ) -> impl Iterator<Item = (usize, u64)> {
        log_n_range.flat_map(move |log_n| izip!([log_n; 10], two_adic_primes(log_q, log_n + 1)))
    }

    #[test]
    fn encrypt_decrypt() {
        let mut rng = thread_rng();
        let (log_n_range, log_q, p) = (0..10, 45, 1 << 4);
        for (log_n, q) in testing_n_q(log_n_range, log_q) {
            let param = RlweParam::new(q, p, log_n);
            let (sk, pk) = Rlwe::key_gen(&param, &mut rng);
            let m = Rq::sample_zq_uniform(param.n(), p, &mut rng);
            let pt0 = Rlwe::encode(&param, m.clone());
            let pt1 = Rlwe::encode(&param, m.clone());
            let ct0 = Rlwe::sk_encrypt(&param, &sk, pt0, &mut rng);
            let ct1 = Rlwe::pk_encrypt(&param, &pk, pt1, &mut rng);
            assert_eq!(m, Rlwe::decode(&param, Rlwe::decrypt(&param, &sk, ct0)));
            assert_eq!(m, Rlwe::decode(&param, Rlwe::decrypt(&param, &sk, ct1)));
        }
    }

    #[test]
    fn add_sub() {
        let mut rng = thread_rng();
        let (log_n_range, log_q, p) = (0..10, 45, 1 << 4);
        for (log_n, q) in testing_n_q(log_n_range, log_q) {
            let param = RlweParam::new(q, p, log_n);
            let (sk, pk) = Rlwe::key_gen(&param, &mut rng);
            let [m0, m1] = &from_fn(|_| Rq::sample_zq_uniform(param.n(), p, &mut rng));
            let [pt0, pt1] = [m0, m1].map(|m| Rlwe::encode(&param, m.clone()));
            let [ct0, ct1] = [pt0, pt1].map(|pt| Rlwe::pk_encrypt(&param, &pk, pt, &mut rng));
            let (m2, ct2) = (m0 + m1, ct0.clone() + ct1.clone());
            let (m3, ct3) = (m0 - m1, ct0.clone() - ct1.clone());
            assert_eq!(m2, Rlwe::decode(&param, Rlwe::decrypt(&param, &sk, ct2)));
            assert_eq!(m3, Rlwe::decode(&param, Rlwe::decrypt(&param, &sk, ct3)));
        }
    }

    #[test]
    fn key_switch() {
        let mut rng = thread_rng();
        let (log_n_range, log_q, p, log_b, d) = (0..10, 45, 1 << 4, 5, 9);
        for (log_n, q) in testing_n_q(log_n_range, log_q) {
            let param0 = RlweParam::new(q, p, log_n);
            let param1 = RlweParam::new(q, p, log_n).with_decomposor(log_b, d);
            let (sk0, pk0) = Rlwe::key_gen(&param0, &mut rng);
            let (sk1, _) = Rlwe::key_gen(&param1, &mut rng);
            let ksk = Rlwe::ksk_gen(&param1, &sk1, &sk0, &mut rng);
            let m = Rq::sample_zq_uniform(param0.n(), p, &mut rng);
            let pt = Rlwe::encode(&param0, m.clone());
            let ct0 = Rlwe::pk_encrypt(&param0, &pk0, pt, &mut rng);
            let ct1 = Rlwe::key_switch(&param1, &ksk, ct0);
            assert_eq!(m, Rlwe::decode(&param1, Rlwe::decrypt(&param1, &sk1, ct1)));
        }
    }

    #[test]
    fn automorphism() {
        let mut rng = thread_rng();
        let (log_n_range, log_q, p, log_b, d) = (0..10, 45, 1 << 4, 5, 9);
        for (log_n, q) in testing_n_q(log_n_range, log_q) {
            for t in [-Rlwe::AUTO_G, Rlwe::AUTO_G] {
                let param = RlweParam::new(q, p, log_n).with_decomposor(log_b, d);
                let (sk, pk) = Rlwe::key_gen(&param, &mut rng);
                let ak = Rlwe::ak_gen(&param, t, &sk, &mut rng);
                let m = Rq::sample_zq_uniform(param.n(), p, &mut rng);
                let pt = Rlwe::encode(&param, m.clone());
                let ct0 = Rlwe::pk_encrypt(&param, &pk, pt, &mut rng);
                let ct1 = Rlwe::automorphism(&param, &ak, ct0);
                assert_eq!(
                    m.automorphism(ak.t()),
                    Rlwe::decode(&param, Rlwe::decrypt(&param, &sk, ct1))
                );
            }
        }
    }

    #[test]
    fn sample_extract() {
        let mut rng = thread_rng();
        let (log_n_range, log_q, p) = (0..10, 45, 1 << 4);
        for (log_n, q) in testing_n_q(log_n_range, log_q) {
            let param = RlweParam::new(q, p, log_n);
            let (sk, pk) = Rlwe::key_gen(&param, &mut rng);
            let m = Rq::sample_zq_uniform(param.n(), p, &mut rng);
            let pt = Rlwe::encode(&param, m.clone());
            let ct = Rlwe::pk_encrypt(&param, &pk, pt, &mut rng);
            for i in 0..param.n() {
                let ct = Rlwe::sample_extract(&param, ct.clone(), i);
                assert_eq!(m[i], Lwe::decode(&param, Lwe::decrypt(&param, &sk, ct)));
            }
        }
    }

    #[test]
    fn multi_key_encrypt_decrypt() {
        const N: usize = 3;

        let mut rng = thread_rng();
        let (log_n_range, log_q, p) = (0..10, 45, 1 << 4);
        for (log_n, q) in testing_n_q(log_n_range, log_q) {
            let param = RlweParam::new(q, p, log_n);
            let a = Rq::sample_zq_uniform(param.n(), param.q(), &mut rng);
            let sk_shares: [_; N] = from_fn(|_| Rlwe::sk_gen(&param, &mut rng));
            let pk = {
                let pk_share_gen = |sk| Rlwe::pk_share_gen(&param, &a, sk, &mut rng);
                let pk_shares = sk_shares.each_ref().map(pk_share_gen);
                Rlwe::pk_share_merge(&param, a, pk_shares)
            };
            let m = Rq::sample_zq_uniform(param.n(), p, &mut rng);
            let ct = Rlwe::pk_encrypt(&param, &pk, Rlwe::encode(&param, m.clone()), &mut rng);
            let pt = {
                let d_shares = sk_shares
                    .each_ref()
                    .map(|sk| Rlwe::share_decrypt(&param, sk, ct.a(), &mut rng));
                Rlwe::decryption_share_merge(&param, ct.b(), d_shares)
            };
            assert_eq!(m, Rlwe::decode(&param, pt));
        }
    }
}
