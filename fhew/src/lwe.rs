use crate::util::{dg, zo, AVec, Decomposor, Dot, Fq};
use itertools::chain;
use rand::RngCore;

pub struct Lwe;

#[derive(Clone, Copy, Debug)]
pub struct LweParam {
    log_q: usize,
    log_p: usize,
    n: usize,
    decomposor: Option<Decomposor>,
}

impl LweParam {
    pub fn new(log_q: usize, log_p: usize, n: usize) -> Self {
        assert!(log_q > log_p);

        Self {
            log_q,
            log_p,
            n,
            decomposor: None,
        }
    }

    pub fn with_decomposor(mut self, log_b: usize, k: usize) -> Self {
        self.decomposor = Some(Decomposor::new(self.q(), log_b, k));
        self
    }

    pub fn q(&self) -> u64 {
        1 << self.log_q
    }

    pub fn p(&self) -> u64 {
        1 << self.log_p
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn delta(&self) -> u64 {
        1 << (self.log_q - self.log_p)
    }
}

pub struct LweSecretKey(AVec<Fq>);

pub struct LweKeySwitchingKey(Vec<LweCiphertext>);

impl LweKeySwitchingKey {
    pub fn a(&self) -> impl Iterator<Item = &AVec<Fq>> {
        self.0.iter().map(|ct| ct.a())
    }

    pub fn b(&self) -> impl Iterator<Item = &Fq> {
        self.0.iter().map(|ct| ct.b())
    }
}

pub struct LwePlaintext(Fq);

pub struct LweCiphertext(AVec<Fq>, Fq);

impl LweCiphertext {
    pub fn a(&self) -> &AVec<Fq> {
        &self.0
    }

    pub fn b(&self) -> &Fq {
        &self.1
    }
}

impl Lwe {
    pub fn key_gen(param: &LweParam, rng: &mut impl RngCore) -> LweSecretKey {
        let sk = AVec::sample_fq_from_i8(param.n, param.q(), &zo(0.5), rng);
        LweSecretKey(sk)
    }

    pub fn ksk_gen(
        param: &LweParam,
        sk0: &LweSecretKey,
        sk1: &LweSecretKey,
        rng: &mut impl RngCore,
    ) -> LweKeySwitchingKey {
        let decomposor = param.decomposor.as_ref().unwrap();
        let ksk = chain![&sk1.0]
            .flat_map(|sk1i| decomposor.bases().map(move |bi| -sk1i * bi))
            .map(|m| Lwe::sk_encrypt(param, sk0, &LwePlaintext(m), rng))
            .collect();
        LweKeySwitchingKey(ksk)
    }

    pub fn encode(param: &LweParam, m: &Fq) -> LwePlaintext {
        assert_eq!(m.q(), param.p());
        LwePlaintext(Fq::from_u64(param.q(), u64::from(m) * param.delta()))
    }

    pub fn decode(param: &LweParam, pt: &LwePlaintext) -> Fq {
        Fq::from_f64(param.p(), f64::from(pt.0) / param.delta() as f64)
    }

    pub fn sk_encrypt(
        param: &LweParam,
        sk: &LweSecretKey,
        pt: &LwePlaintext,
        rng: &mut impl RngCore,
    ) -> LweCiphertext {
        let a = AVec::sample_fq_uniform(param.n, param.q(), rng);
        let e = Fq::sample_i8(param.q(), &dg(3.2, 6), rng);
        let b = a.dot(&sk.0) + pt.0 + e;
        LweCiphertext(a, b)
    }

    pub fn decrypt(_: &LweParam, sk: &LweSecretKey, ct: &LweCiphertext) -> LwePlaintext {
        let pt = ct.b() - ct.a().dot(&sk.0);
        LwePlaintext(pt)
    }

    pub fn eval_add(_: &LweParam, ct0: &LweCiphertext, ct1: &LweCiphertext) -> LweCiphertext {
        LweCiphertext(ct0.a() + ct1.a(), ct0.b() + ct1.b())
    }

    pub fn eval_sub(_: &LweParam, ct0: &LweCiphertext, ct1: &LweCiphertext) -> LweCiphertext {
        LweCiphertext(ct0.a() - ct1.a(), ct0.b() - ct1.b())
    }

    pub fn key_switch(
        param: &LweParam,
        ksk: &LweKeySwitchingKey,
        ct: &LweCiphertext,
    ) -> LweCiphertext {
        let decomposor = param.decomposor.as_ref().unwrap();
        let ct_a_limbs = chain![ct.a()]
            .flat_map(|a| decomposor.decompose(a))
            .collect::<AVec<_>>();
        let a = ksk.a().dot(&ct_a_limbs);
        let b = ksk.b().dot(&ct_a_limbs) + ct.b();
        LweCiphertext(a, b)
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
        let (log_q, log_p, n) = (16, 4, 1024);
        let param = LweParam::new(log_q, log_p, n);
        let sk = Lwe::key_gen(&param, &mut rng);
        for m in 0..param.p() {
            let m = Fq::from_u64(param.p(), m);
            let pt = Lwe::encode(&param, &m);
            let ct = Lwe::sk_encrypt(&param, &sk, &pt, &mut rng);
            assert_eq!(m, Lwe::decode(&param, &Lwe::decrypt(&param, &sk, &ct)));
        }
    }

    #[test]
    fn eval_add() {
        let mut rng = StdRng::from_entropy();
        let (log_q, log_p, n) = (16, 4, 1024);
        let param = LweParam::new(log_q, log_p, n);
        let sk = Lwe::key_gen(&param, &mut rng);
        for (m0, m1) in (0..param.p()).cartesian_product(0..param.p()) {
            let [m0, m1] = [m0, m1].map(|m| Fq::from_u64(param.p(), m));
            let [pt0, pt1] = [m0, m1].map(|m| Lwe::encode(&param, &m));
            let [ct0, ct1] = [pt0, pt1].map(|pt| Lwe::sk_encrypt(&param, &sk, &pt, &mut rng));
            let ct2 = Lwe::eval_add(&param, &ct0, &ct1);
            let m2 = m0 + m1;
            assert_eq!(m2, Lwe::decode(&param, &Lwe::decrypt(&param, &sk, &ct2)));
        }
    }

    #[test]
    fn eval_sub() {
        let mut rng = StdRng::from_entropy();
        let (log_q, log_p, n) = (16, 4, 1024);
        let param = LweParam::new(log_q, log_p, n);
        let sk = Lwe::key_gen(&param, &mut rng);
        for (m0, m1) in (0..param.p()).cartesian_product(0..param.p()) {
            let [m0, m1] = [m0, m1].map(|m| Fq::from_u64(param.p(), m));
            let [pt0, pt1] = [m0, m1].map(|m| Lwe::encode(&param, &m));
            let [ct0, ct1] = [pt0, pt1].map(|pt| Lwe::sk_encrypt(&param, &sk, &pt, &mut rng));
            let ct2 = Lwe::eval_sub(&param, &ct0, &ct1);
            let m2 = m0 - m1;
            assert_eq!(m2, Lwe::decode(&param, &Lwe::decrypt(&param, &sk, &ct2)));
        }
    }

    #[test]
    fn key_switch() {
        let mut rng = StdRng::from_entropy();
        let (log_q, log_p, n0, n1, log_b, k) = (16, 4, 1024, 512, 2, 8);
        let param0 = LweParam::new(log_q, log_p, n0);
        let param1 = LweParam::new(log_q, log_p, n1).with_decomposor(log_b, k);
        let sk0 = Lwe::key_gen(&param0, &mut rng);
        let sk1 = Lwe::key_gen(&param1, &mut rng);
        let ksk = Lwe::ksk_gen(&param1, &sk1, &sk0, &mut rng);
        for m in 0..param0.p() {
            let m = Fq::from_u64(param0.p(), m);
            let pt = Lwe::encode(&param0, &m);
            let ct0 = Lwe::sk_encrypt(&param0, &sk0, &pt, &mut rng);
            let ct1 = Lwe::key_switch(&param1, &ksk, &ct0);
            assert_eq!(m, Lwe::decode(&param1, &Lwe::decrypt(&param1, &sk1, &ct1)));
        }
    }
}
