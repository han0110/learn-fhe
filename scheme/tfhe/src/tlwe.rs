use derive_more::{Add, Sub};
use itertools::Itertools;
use rand::{distributions::Distribution, RngCore};
use util::{binary, tdg, AVec, Base2Decomposable, Base2Decomposor, Dot, Zq, T64};

#[derive(Debug)]
pub struct Tlwe;

#[derive(Clone, Copy, Debug)]
pub struct TlweParam {
    log_p: usize,
    padding: usize,
    n: usize,
    std_dev: f64,
    decomposor: Option<Base2Decomposor<T64>>,
}

impl TlweParam {
    pub fn new(log_p: usize, padding: usize, n: usize, std_dev: f64) -> Self {
        TlweParam {
            log_p,
            padding,
            n,
            std_dev,
            decomposor: None,
        }
    }

    pub fn with_decomposor(mut self, log_b: usize, d: usize) -> Self {
        self.decomposor = Some(Base2Decomposor::<T64>::new(log_b, d));
        self
    }

    pub fn log_p(&self) -> usize {
        self.log_p
    }

    pub fn p(&self) -> u64 {
        1 << self.log_p()
    }

    pub fn padding(&self) -> usize {
        self.padding
    }

    pub fn log_delta(&self) -> usize {
        u64::BITS as usize - (self.log_p + self.padding)
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn std_dev(&self) -> f64 {
        self.std_dev
    }

    pub fn decomposor(&self) -> &Base2Decomposor<T64> {
        self.decomposor.as_ref().unwrap()
    }
}

#[derive(Clone, Debug)]
pub struct TlweSecretKey(pub(crate) AVec<i64>);

#[derive(Clone, Debug)]
pub struct TlwePlaintext(pub(crate) T64);

#[derive(Clone, Debug, Add, Sub)]
pub struct TlweCiphertext(pub(crate) AVec<T64>, pub(crate) T64);

impl TlweCiphertext {
    pub fn a(&self) -> &AVec<T64> {
        &self.0
    }

    pub fn b(&self) -> &T64 {
        &self.1
    }
}

impl TlweKeySwitchingKey {
    pub fn a(&self) -> impl Iterator<Item = &AVec<T64>> {
        self.0.iter().map(TlweCiphertext::a)
    }

    pub fn b(&self) -> impl Iterator<Item = &T64> {
        self.0.iter().map(TlweCiphertext::b)
    }
}

#[derive(Clone, Debug)]
pub struct TlweKeySwitchingKey(AVec<TlweCiphertext>);

impl Tlwe {
    pub fn sk_gen(param: &TlweParam, rng: &mut impl RngCore) -> TlweSecretKey {
        TlweSecretKey(AVec::sample(param.n(), binary(), rng))
    }

    pub fn ksk_gen(
        param: &TlweParam,
        sk0: &TlweSecretKey,
        TlweSecretKey(sk1): &TlweSecretKey,
        rng: &mut impl RngCore,
    ) -> TlweKeySwitchingKey {
        let pt = param.decomposor().power_up(-sk1).flatten();
        let ksk = pt
            .map(|pt| Tlwe::sk_encrypt(param, sk0, TlwePlaintext(pt), rng))
            .collect();
        TlweKeySwitchingKey(ksk)
    }

    pub fn encode(param: &TlweParam, m: Zq) -> TlwePlaintext {
        assert!(m.q() == param.p());
        TlwePlaintext((m.to_u64() << param.log_delta()).into())
    }

    pub fn decode(param: &TlweParam, TlwePlaintext(pt): TlwePlaintext) -> Zq {
        Zq::from_u64(param.p(), pt.to_u64() >> param.log_delta())
    }

    pub fn sk_encrypt(
        param: &TlweParam,
        TlweSecretKey(sk): &TlweSecretKey,
        TlwePlaintext(pt): TlwePlaintext,
        rng: &mut impl RngCore,
    ) -> TlweCiphertext {
        let a = AVec::<T64>::sample_uniform(param.n(), rng);
        let e = tdg(param.std_dev()).sample(rng);
        let b = a.dot(sk) + e + pt;
        TlweCiphertext(a, b)
    }

    pub fn decrypt(
        param: &TlweParam,
        TlweSecretKey(sk): &TlweSecretKey,
        TlweCiphertext(a, b): TlweCiphertext,
    ) -> TlwePlaintext {
        let mu_star = b - a.dot(sk);
        let mu = mu_star.round(param.log_delta());
        TlwePlaintext(mu)
    }

    pub fn key_switch(
        param: &TlweParam,
        ksk: &TlweKeySwitchingKey,
        ct: TlweCiphertext,
    ) -> TlweCiphertext {
        let ct_a_limbs = param.decomposor().decompose(ct.a()).flatten().collect_vec();
        let a = ksk.a().dot(&ct_a_limbs);
        let b = ksk.b().dot(&ct_a_limbs) + ct.b();
        TlweCiphertext(a, b)
    }
}

#[cfg(test)]
mod test {
    use crate::tlwe::{Tlwe, TlweParam};
    use rand::thread_rng;
    use util::Zq;

    #[test]
    fn encrypt_decrypt() {
        let mut rng = thread_rng();
        let (log_p, padding, n, std_dev) = (8, 1, 256, 1.0e-8);
        let param = TlweParam::new(log_p, padding, n, std_dev);
        let sk = Tlwe::sk_gen(&param, &mut rng);
        for m in 0..param.p() {
            let m = Zq::from_u64(param.p(), m);
            let pt = Tlwe::encode(&param, m);
            let ct = Tlwe::sk_encrypt(&param, &sk, pt.clone(), &mut rng);
            assert_eq!(m, Tlwe::decode(&param, Tlwe::decrypt(&param, &sk, ct)));
        }
    }

    #[test]
    fn key_switch() {
        let mut rng = thread_rng();
        let (log_p, padding, n, std_dev, log_b, d) = (8, 1, 256, 1.0e-8, 8, 8);
        let param0 = TlweParam::new(log_p, padding, n, std_dev);
        let param1 = TlweParam::new(log_p, padding, n, std_dev).with_decomposor(log_b, d);
        let sk0 = Tlwe::sk_gen(&param0, &mut rng);
        let sk1 = Tlwe::sk_gen(&param1, &mut rng);
        let ksk = Tlwe::ksk_gen(&param1, &sk1, &sk0, &mut rng);
        for m in 0..param0.p() {
            let m = Zq::from_u64(param0.p(), m);
            let pt = Tlwe::encode(&param0, m);
            let ct0 = Tlwe::sk_encrypt(&param0, &sk0, pt, &mut rng);
            let ct1 = Tlwe::key_switch(&param1, &ksk, ct0);
            assert_eq!(m, Tlwe::decode(&param1, Tlwe::decrypt(&param1, &sk1, ct1)));
        }
    }
}
