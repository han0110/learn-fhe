use crate::{
    rlwe::RlweSecretKey,
    util::{cartesian, dg, izip_eq, zipstar, AVec, Decomposor, Dot, Fq},
};
use derive_more::{Add, AddAssign, Sub, SubAssign};
use itertools::chain;
use rand::RngCore;

#[derive(Debug)]
pub struct Lwe;

#[derive(Clone, Copy, Debug)]
pub struct LweParam {
    q: u64,
    p: u64,
    n: usize,
    decomposor: Option<Decomposor>,
}

impl LweParam {
    pub fn new(q: u64, p: u64, n: usize) -> Self {
        assert!(q > p);

        Self {
            q,
            p,
            n,
            decomposor: None,
        }
    }

    pub fn with_decomposor(mut self, log_b: usize, d: usize) -> Self {
        self.decomposor = Some(Decomposor::new(self.q(), log_b, d));
        self
    }

    pub fn q(&self) -> u64 {
        self.q
    }

    pub fn p(&self) -> u64 {
        self.p
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn delta(&self) -> f64 {
        self.q as f64 / self.p as f64
    }

    pub fn decomposor(&self) -> &Decomposor {
        self.decomposor.as_ref().unwrap()
    }
}

#[derive(Clone, Debug)]
pub struct LweSecretKey(pub(crate) AVec<i8>);

impl From<&RlweSecretKey> for LweSecretKey {
    fn from(value: &RlweSecretKey) -> Self {
        LweSecretKey(value.0.clone().into())
    }
}

#[derive(Clone, Debug, Add, Sub, AddAssign, SubAssign)]
pub struct LweKeySwitchingKey(pub(crate) AVec<LweCiphertext>);

impl LweKeySwitchingKey {
    pub fn a(&self) -> impl Iterator<Item = &AVec<Fq>> {
        self.0.iter().map(|ct| ct.a())
    }

    pub fn b(&self) -> impl Iterator<Item = &Fq> {
        self.0.iter().map(|ct| ct.b())
    }
}

#[derive(Clone, Debug)]
pub struct LwePlaintext(pub(crate) Fq);

#[derive(Clone, Debug, Add, Sub, AddAssign, SubAssign)]
pub struct LweCiphertext(pub(crate) AVec<Fq>, pub(crate) Fq);

impl LweCiphertext {
    pub fn a(&self) -> &AVec<Fq> {
        &self.0
    }

    pub fn b(&self) -> &Fq {
        &self.1
    }

    pub fn double(self) -> Self {
        self.clone() + self
    }

    pub fn mod_switch(&self, q_prime: u64) -> Self {
        Self(self.a().mod_switch(q_prime), self.b().mod_switch(q_prime))
    }

    pub fn mod_switch_odd(&self, q_prime: u64) -> Self {
        Self(
            self.a().mod_switch_odd(q_prime),
            self.b().mod_switch_odd(q_prime),
        )
    }
}

impl Lwe {
    pub fn sk_gen(param: &LweParam, rng: &mut impl RngCore) -> LweSecretKey {
        let sk = AVec::sample(param.n, &dg(3.2, 6), rng);
        LweSecretKey(sk)
    }

    pub fn ksk_gen(
        param: &LweParam,
        sk0: &LweSecretKey,
        sk1: &LweSecretKey,
        rng: &mut impl RngCore,
    ) -> LweKeySwitchingKey {
        let pt = cartesian!(&sk1.0, param.decomposor().bases()).map(|(sk1j, bi)| -bi * sk1j);
        let ksk = pt
            .map(|pt| Lwe::sk_encrypt(param, sk0, LwePlaintext(pt), rng))
            .collect();
        LweKeySwitchingKey(ksk)
    }

    pub fn encode(param: &LweParam, m: Fq) -> LwePlaintext {
        assert_eq!(m.q(), param.p());
        LwePlaintext(Fq::from_f64(param.q(), f64::from(m) * param.delta()))
    }

    pub fn decode(param: &LweParam, pt: LwePlaintext) -> Fq {
        Fq::from_f64(param.p(), f64::from(pt.0) / param.delta())
    }

    pub fn sk_encrypt(
        param: &LweParam,
        sk: &LweSecretKey,
        pt: LwePlaintext,
        rng: &mut impl RngCore,
    ) -> LweCiphertext {
        let a = AVec::sample_fq_uniform(param.n, param.q(), rng);
        let e = Fq::sample_i8(param.q(), &dg(3.2, 6), rng);
        let b = a.dot(&sk.0) + pt.0 + e;
        LweCiphertext(a, b)
    }

    pub fn decrypt(_: &LweParam, sk: &LweSecretKey, ct: LweCiphertext) -> LwePlaintext {
        let pt = ct.b() - ct.a().dot(&sk.0);
        LwePlaintext(pt)
    }

    pub fn key_switch(
        param: &LweParam,
        ksk: &LweKeySwitchingKey,
        ct: LweCiphertext,
    ) -> LweCiphertext {
        let ct_a_limbs = chain![ct.a()]
            .flat_map(|a| param.decomposor().decompose(a))
            .collect::<AVec<_>>();
        let a = ksk.a().dot(&ct_a_limbs);
        let b = ksk.b().dot(&ct_a_limbs) + ct.b();
        LweCiphertext(a, b)
    }
}

#[derive(Clone, Debug)]
pub struct LweEncryptionShare(Fq);

#[derive(Clone, Debug)]
pub struct LweDecryptionShare(Fq);

#[derive(Clone, Debug)]
pub struct LweKeySwitchingKeyShare(Vec<LweEncryptionShare>);

impl Lwe {
    pub fn sk_share_encrypt(
        param: &LweParam,
        a: &AVec<Fq>,
        sk: &LweSecretKey,
        pt: LwePlaintext,
        rng: &mut impl RngCore,
    ) -> LweEncryptionShare {
        let e = Fq::sample_i8(param.q(), &dg(3.2, 6), rng);
        let b = a.dot(&sk.0) + pt.0 + e;
        LweEncryptionShare(b)
    }

    pub fn encryption_share_merge(
        _: &LweParam,
        a: AVec<Fq>,
        shares: impl IntoIterator<Item = LweEncryptionShare>,
    ) -> LweCiphertext {
        let b = shares.into_iter().map(|share| share.0).sum();
        LweCiphertext(a, b)
    }

    pub fn share_decrypt(
        param: &LweParam,
        sk: &LweSecretKey,
        a: &AVec<Fq>,
        rng: &mut impl RngCore,
    ) -> LweDecryptionShare {
        let e = Fq::sample_i8(param.q(), &dg(3.2, 6), rng);
        let share = a.dot(&sk.0) + e;
        LweDecryptionShare(share)
    }

    pub fn decryption_share_merge(
        _: &LweParam,
        b: Fq,
        shares: impl IntoIterator<Item = LweDecryptionShare>,
    ) -> LwePlaintext {
        let pt = b - shares.into_iter().map(|share| share.0).sum::<Fq>();
        LwePlaintext(pt)
    }

    pub fn ksk_share_gen(
        param: &LweParam,
        crs: &[AVec<Fq>],
        sk0: &LweSecretKey,
        sk1: &LweSecretKey,
        rng: &mut impl RngCore,
    ) -> LweKeySwitchingKeyShare {
        let pt = cartesian!(&sk1.0, param.decomposor().bases()).map(|(sk1j, bi)| -bi * sk1j);
        let ksk = izip_eq!(crs, pt)
            .map(|(a, pt)| Lwe::sk_share_encrypt(param, a, sk0, LwePlaintext(pt), rng))
            .collect();
        LweKeySwitchingKeyShare(ksk)
    }

    pub fn ksk_share_merge(
        param: &LweParam,
        crs: Vec<AVec<Fq>>,
        shares: impl IntoIterator<Item = LweKeySwitchingKeyShare>,
    ) -> LweKeySwitchingKey {
        let ksk = izip_eq!(crs, zipstar!(shares, 0))
            .map(|(crs, share)| Lwe::encryption_share_merge(param, crs, share))
            .collect();
        LweKeySwitchingKey(ksk)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        lwe::{Lwe, LweParam},
        util::Fq,
    };
    use itertools::Itertools;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn encrypt_decrypt() {
        let mut rng = StdRng::from_entropy();
        let (q, p, n) = (1 << 16, 1 << 4, 1024);
        let param = LweParam::new(q, p, n);
        let sk = Lwe::sk_gen(&param, &mut rng);
        for m in 0..param.p() {
            let m = Fq::from_u64(param.p(), m);
            let pt = Lwe::encode(&param, m);
            let ct = Lwe::sk_encrypt(&param, &sk, pt, &mut rng);
            assert_eq!(m, Lwe::decode(&param, Lwe::decrypt(&param, &sk, ct)));
        }
    }

    #[test]
    fn add_sub() {
        let mut rng = StdRng::from_entropy();
        let (q, p, n) = (1 << 16, 1 << 4, 1024);
        let param = LweParam::new(q, p, n);
        let sk = Lwe::sk_gen(&param, &mut rng);
        for (m0, m1) in (0..param.p()).cartesian_product(0..param.p()) {
            let [m0, m1] = [m0, m1].map(|m| Fq::from_u64(param.p(), m));
            let [pt0, pt1] = [m0, m1].map(|m| Lwe::encode(&param, m));
            let [ct0, ct1] = [pt0, pt1].map(|pt| Lwe::sk_encrypt(&param, &sk, pt, &mut rng));
            let (m2, ct2) = (m0 + m1, ct0.clone() + ct1.clone());
            let (m3, ct3) = (m0 - m1, ct0.clone() - ct1.clone());
            assert_eq!(m2, Lwe::decode(&param, Lwe::decrypt(&param, &sk, ct2)));
            assert_eq!(m3, Lwe::decode(&param, Lwe::decrypt(&param, &sk, ct3)));
        }
    }

    #[test]
    fn key_switch() {
        let mut rng = StdRng::from_entropy();
        let (q, p, n0, n1, log_b, d) = (1 << 16, 1 << 4, 1024, 512, 2, 8);
        let param0 = LweParam::new(q, p, n0);
        let param1 = LweParam::new(q, p, n1).with_decomposor(log_b, d);
        let sk0 = Lwe::sk_gen(&param0, &mut rng);
        let sk1 = Lwe::sk_gen(&param1, &mut rng);
        let ksk = Lwe::ksk_gen(&param1, &sk1, &sk0, &mut rng);
        for m in 0..param0.p() {
            let m = Fq::from_u64(param0.p(), m);
            let pt = Lwe::encode(&param0, m);
            let ct0 = Lwe::sk_encrypt(&param0, &sk0, pt, &mut rng);
            let ct1 = Lwe::key_switch(&param1, &ksk, ct0);
            assert_eq!(m, Lwe::decode(&param1, Lwe::decrypt(&param1, &sk1, ct1)));
        }
    }
}
