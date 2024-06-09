use crate::{
    lwe::{LweCiphertext, LweParam, LweSecretKey},
    util::{dg, zo, AVec, Dot, Fq, Poly},
};
use derive_more::{Add, Deref, Sub};
use itertools::chain;
use rand::RngCore;

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

#[derive(Clone, Debug)]
pub struct RlweSecretKey(pub(crate) Poly<i8>);

impl From<&LweSecretKey> for RlweSecretKey {
    fn from(value: &LweSecretKey) -> Self {
        RlweSecretKey(value.0.clone().into())
    }
}

#[derive(Clone, Debug)]
pub struct RlwePublicKey(Poly<Fq>, Poly<Fq>);

impl RlwePublicKey {
    pub fn a(&self) -> &Poly<Fq> {
        &self.0
    }

    pub fn b(&self) -> &Poly<Fq> {
        &self.1
    }
}

#[derive(Clone, Debug)]
pub struct RlweKeySwitchingKey(Vec<RlweCiphertext>);

impl RlweKeySwitchingKey {
    pub fn a(&self) -> impl Iterator<Item = &Poly<Fq>> {
        self.0.iter().map(|ct| ct.a())
    }

    pub fn b(&self) -> impl Iterator<Item = &Poly<Fq>> {
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
pub struct RlwePlaintext(pub(crate) Poly<Fq>);

#[derive(Clone, Debug, Add, Sub)]
pub struct RlweCiphertext(pub(crate) Poly<Fq>, pub(crate) Poly<Fq>);

impl RlweCiphertext {
    pub fn a(&self) -> &Poly<Fq> {
        &self.0
    }

    pub fn b(&self) -> &Poly<Fq> {
        &self.1
    }

    pub fn automorphism(&self, k: i64) -> Self {
        Self(self.a().automorphism(k), self.b().automorphism(k))
    }
}

impl From<RlwePlaintext> for RlweCiphertext {
    fn from(RlwePlaintext(b): RlwePlaintext) -> Self {
        Self(Poly::zero(b.n(), b[0].q()), b)
    }
}

impl Rlwe {
    pub const AUTO_G: i64 = 5;

    pub fn key_gen(param: &RlweParam, rng: &mut impl RngCore) -> (RlweSecretKey, RlwePublicKey) {
        let sk = RlweSecretKey(Poly::sample(param.n(), &dg(3.2, 6), rng));
        let pk = {
            let a = Poly::sample_fq_uniform(param.n(), param.q(), rng);
            let e = Poly::sample_fq_from_i8(param.n(), param.q(), &dg(3.2, 6), rng);
            let b = &a * &sk.0 + e;
            RlwePublicKey(a, b)
        };
        (sk, pk)
    }

    pub fn ksk_gen(
        param: &RlweParam,
        sk0: &RlweSecretKey,
        sk1: &RlweSecretKey,
        rng: &mut impl RngCore,
    ) -> RlweKeySwitchingKey {
        let ksk = chain![param.decomposor().bases()]
            .map(move |bi| -&sk1.0 * bi)
            .map(|m| Rlwe::sk_encrypt(param, sk0, RlwePlaintext(m), rng))
            .collect();
        RlweKeySwitchingKey(ksk)
    }

    pub fn ak_gen(
        param: &RlweParam,
        sk: &RlweSecretKey,
        t: i64,
        rng: &mut impl RngCore,
    ) -> RlweAutoKey {
        let sk_auto = sk.0.automorphism(t);
        let ksk = Rlwe::ksk_gen(param, sk, &RlweSecretKey(sk_auto), rng);
        RlweAutoKey(t, ksk)
    }

    pub fn encode(param: &RlweParam, m: Poly<Fq>) -> RlwePlaintext {
        assert_eq!(m.n(), param.n());
        assert!(m.iter().all(|m| m.q() == param.p()));

        let scale_up = |m| Fq::from_f64(param.q(), f64::from(m) * param.delta());
        RlwePlaintext(m.iter().map(scale_up).collect())
    }

    pub fn decode(param: &RlweParam, pt: RlwePlaintext) -> Poly<Fq> {
        let scale_down = |m| Fq::from_f64(param.p(), f64::from(m) / param.delta());
        pt.0.iter().map(scale_down).collect()
    }

    pub fn sk_encrypt(
        param: &RlweParam,
        sk: &RlweSecretKey,
        pt: RlwePlaintext,
        rng: &mut impl RngCore,
    ) -> RlweCiphertext {
        let a = Poly::sample_fq_uniform(param.n(), param.q(), rng);
        let e = Poly::sample_fq_from_i8(param.n(), param.q(), &dg(3.2, 6), rng);
        let b = &a * &sk.0 + e + pt.0;
        RlweCiphertext(a, b)
    }

    pub fn pk_encrypt(
        param: &RlweParam,
        pk: &RlwePublicKey,
        pt: RlwePlaintext,
        rng: &mut impl RngCore,
    ) -> RlweCiphertext {
        let u = &Poly::sample_fq_from_i8(param.n(), param.q(), &zo(0.5), rng);
        let e0 = Poly::sample_fq_from_i8(param.n(), param.q(), &dg(3.2, 6), rng);
        let e1 = Poly::sample_fq_from_i8(param.n(), param.q(), &dg(3.2, 6), rng);
        let a = pk.a() * u + e0;
        let b = pk.b() * u + e1 + pt.0;
        RlweCiphertext(a, b)
    }

    pub fn decrypt(_: &RlweParam, sk: &RlweSecretKey, ct: RlweCiphertext) -> RlwePlaintext {
        let pt = ct.b() - ct.a() * &sk.0;
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

#[cfg(test)]
pub(crate) mod test {
    use crate::{
        lwe::Lwe,
        rlwe::{Rlwe, RlweParam},
        util::{two_adic_primes, Poly},
    };
    use core::{array::from_fn, ops::Range};
    use itertools::izip;
    use rand::{rngs::StdRng, SeedableRng};

    pub(crate) fn testing_n_q(
        log_n_range: Range<usize>,
        log_q: usize,
    ) -> impl Iterator<Item = (usize, u64)> {
        log_n_range.flat_map(move |log_n| izip!([log_n; 10], two_adic_primes(log_q, log_n + 1)))
    }

    #[test]
    fn encrypt_decrypt() {
        let mut rng = StdRng::from_entropy();
        let (log_n_range, log_q, p) = (0..10, 45, 1 << 4);
        for (log_n, q) in testing_n_q(log_n_range, log_q) {
            let param = RlweParam::new(q, p, log_n);
            let (sk, pk) = Rlwe::key_gen(&param, &mut rng);
            let m = Poly::sample_fq_uniform(param.n(), p, &mut rng);
            let pt0 = Rlwe::encode(&param, m.clone());
            let pt1 = Rlwe::encode(&param, m.clone());
            let ct0 = Rlwe::sk_encrypt(&param, &sk, pt0, &mut rng);
            let ct1 = Rlwe::pk_encrypt(&param, &pk, pt1, &mut rng);
            assert_eq!(m, Rlwe::decode(&param, Rlwe::decrypt(&param, &sk, ct0)));
            assert_eq!(m, Rlwe::decode(&param, Rlwe::decrypt(&param, &sk, ct1)));
        }
    }

    #[test]
    fn eval_add() {
        let mut rng = StdRng::from_entropy();
        let (log_n_range, log_q, p) = (0..10, 45, 1 << 4);
        for (log_n, q) in testing_n_q(log_n_range, log_q) {
            let param = RlweParam::new(q, p, log_n);
            let (sk, pk) = Rlwe::key_gen(&param, &mut rng);
            let [m0, m1] = &from_fn(|_| Poly::sample_fq_uniform(param.n(), p, &mut rng));
            let [pt0, pt1] = [m0, m1].map(|m| Rlwe::encode(&param, m.clone()));
            let [ct0, ct1] = [pt0, pt1].map(|pt| Rlwe::pk_encrypt(&param, &pk, pt, &mut rng));
            let ct2 = ct0 + ct1;
            let m2 = m0 + m1;
            assert_eq!(m2, Rlwe::decode(&param, Rlwe::decrypt(&param, &sk, ct2)));
        }
    }

    #[test]
    fn eval_sub() {
        let mut rng = StdRng::from_entropy();
        let (log_n_range, log_q, p) = (0..10, 45, 1 << 4);
        for (log_n, q) in testing_n_q(log_n_range, log_q) {
            let param = RlweParam::new(q, p, log_n);
            let (sk, pk) = Rlwe::key_gen(&param, &mut rng);
            let [m0, m1] = &from_fn(|_| Poly::sample_fq_uniform(param.n(), p, &mut rng));
            let [pt0, pt1] = [m0, m1].map(|m| Rlwe::encode(&param, m.clone()));
            let [ct0, ct1] = [pt0, pt1].map(|pt| Rlwe::pk_encrypt(&param, &pk, pt, &mut rng));
            let ct2 = ct0 - ct1;
            let m2 = m0 - m1;
            assert_eq!(m2, Rlwe::decode(&param, Rlwe::decrypt(&param, &sk, ct2)));
        }
    }

    #[test]
    fn key_switch() {
        let mut rng = StdRng::from_entropy();
        let (log_n_range, log_q, p, log_b, d) = (0..10, 45, 1 << 4, 5, 9);
        for (log_n, q) in testing_n_q(log_n_range, log_q) {
            let param0 = RlweParam::new(q, p, log_n);
            let param1 = RlweParam::new(q, p, log_n).with_decomposor(log_b, d);
            let (sk0, pk0) = Rlwe::key_gen(&param0, &mut rng);
            let (sk1, _) = Rlwe::key_gen(&param1, &mut rng);
            let ksk = Rlwe::ksk_gen(&param1, &sk1, &sk0, &mut rng);
            let m = Poly::sample_fq_uniform(param0.n(), p, &mut rng);
            let pt = Rlwe::encode(&param0, m.clone());
            let ct0 = Rlwe::pk_encrypt(&param0, &pk0, pt, &mut rng);
            let ct1 = Rlwe::key_switch(&param1, &ksk, ct0);
            assert_eq!(m, Rlwe::decode(&param1, Rlwe::decrypt(&param1, &sk1, ct1)));
        }
    }

    #[test]
    fn automorphism() {
        let mut rng = StdRng::from_entropy();
        let (log_n_range, log_q, p, log_b, d) = (0..10, 45, 1 << 4, 5, 9);
        for (log_n, q) in testing_n_q(log_n_range, log_q) {
            for t in [-Rlwe::AUTO_G, Rlwe::AUTO_G] {
                let param = RlweParam::new(q, p, log_n).with_decomposor(log_b, d);
                let (sk, pk) = Rlwe::key_gen(&param, &mut rng);
                let ak = Rlwe::ak_gen(&param, &sk, t, &mut rng);
                let m = Poly::sample_fq_uniform(param.n(), p, &mut rng);
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
        let mut rng = StdRng::from_entropy();
        let (log_n_range, log_q, p) = (0..10, 45, 1 << 4);
        for (log_n, q) in testing_n_q(log_n_range, log_q) {
            let param = RlweParam::new(q, p, log_n);
            let (sk, pk) = Rlwe::key_gen(&param, &mut rng);
            let sk = (&sk).into();
            let m = Poly::sample_fq_uniform(param.n(), p, &mut rng);
            let pt = Rlwe::encode(&param, m.clone());
            let ct = Rlwe::pk_encrypt(&param, &pk, pt, &mut rng);
            for i in 0..param.n() {
                let ct = Rlwe::sample_extract(&param, ct.clone(), i);
                assert_eq!(m[i], Lwe::decode(&param, Lwe::decrypt(&param, &sk, ct)));
            }
        }
    }
}
