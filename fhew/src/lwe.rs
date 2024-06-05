use crate::util::{dg, zo, AVec, Dot, Fq};
use rand::RngCore;

pub struct Lwe;

#[derive(Clone, Copy, Debug)]
pub struct LweParam {
    pub(crate) log_q: usize,
    pub(crate) log_p: usize,
    pub(crate) n: usize,
}

impl LweParam {
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

pub struct LwePlaintext(Fq);

pub struct LweCiphertext(Fq, AVec<Fq>);

impl LweCiphertext {
    pub fn b(&self) -> Fq {
        self.0
    }

    pub fn a(&self) -> &AVec<Fq> {
        &self.1
    }
}

impl Lwe {
    pub fn param_gen(log_q: usize, log_p: usize, n: usize) -> LweParam {
        LweParam { log_q, log_p, n }
    }

    pub fn key_gen(param: &LweParam, rng: &mut impl RngCore) -> LweSecretKey {
        let sk = AVec::sample_fq_from_i8(param.n, param.q(), &zo(0.5), rng);
        LweSecretKey(sk)
    }

    pub fn encode(param: &LweParam, m: &Fq) -> LwePlaintext {
        assert_eq!(m.q(), param.p());
        LwePlaintext(Fq::from_u64(param.q(), u64::from(m) * param.delta()))
    }

    pub fn decode(param: &LweParam, pt: &LwePlaintext) -> Fq {
        Fq::from_f64(param.p(), f64::from(pt.0) / param.delta() as f64)
    }

    pub fn encrypt(
        param: &LweParam,
        sk: &LweSecretKey,
        pt: &LwePlaintext,
        rng: &mut impl RngCore,
    ) -> LweCiphertext {
        let a = AVec::sample_fq_uniform(param.n, param.q(), rng);
        let e = Fq::sample_i8(param.q(), &dg(3.2, 6), rng);
        let b = a.dot(&sk.0) + pt.0 + e;
        LweCiphertext(b, a)
    }

    pub fn decrypt(_: &LweParam, sk: &LweSecretKey, ct: &LweCiphertext) -> LwePlaintext {
        let LweCiphertext(b, a) = ct;
        let pt = b - a.dot(&sk.0);
        LwePlaintext(pt)
    }

    pub fn eval_add(_: &LweParam, ct0: &LweCiphertext, ct1: &LweCiphertext) -> LweCiphertext {
        LweCiphertext(ct0.b() + ct1.b(), ct0.a() + ct1.a())
    }

    pub fn eval_sub(_: &LweParam, ct0: &LweCiphertext, ct1: &LweCiphertext) -> LweCiphertext {
        LweCiphertext(ct0.b() - ct1.b(), ct0.a() - ct1.a())
    }
}

#[cfg(test)]
mod test {
    use crate::{lwe::Lwe, util::Fq};
    use itertools::Itertools;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn encrypt_decrypt() {
        let mut rng = StdRng::from_entropy();
        let (log_q, log_p, n) = (16, 4, 1024);
        let param = Lwe::param_gen(log_q, log_p, n);
        let sk = Lwe::key_gen(&param, &mut rng);
        for m in 0..param.p() {
            let m = Fq::from_u64(param.p(), m);
            let pt = Lwe::encode(&param, &m);
            let ct = Lwe::encrypt(&param, &sk, &pt, &mut rng);
            assert_eq!(m, Lwe::decode(&param, &Lwe::decrypt(&param, &sk, &ct)));
        }
    }

    #[test]
    fn eval_add() {
        let mut rng = StdRng::from_entropy();
        let (log_q, log_p, n) = (16, 4, 1024);
        let param = Lwe::param_gen(log_q, log_p, n);
        let sk = Lwe::key_gen(&param, &mut rng);
        for (m0, m1) in (0..param.p()).cartesian_product(0..param.p()) {
            let [m0, m1] = [m0, m1].map(|m| Fq::from_u64(param.p(), m));
            let [pt0, pt1] = [m0, m1].map(|m| Lwe::encode(&param, &m));
            let [ct0, ct1] = [pt0, pt1].map(|pt| Lwe::encrypt(&param, &sk, &pt, &mut rng));
            let ct2 = Lwe::eval_add(&param, &ct0, &ct1);
            let m2 = m0 + m1;
            assert_eq!(m2, Lwe::decode(&param, &Lwe::decrypt(&param, &sk, &ct2)));
        }
    }

    #[test]
    fn eval_sub() {
        let mut rng = StdRng::from_entropy();
        let (log_q, log_p, n) = (16, 4, 1024);
        let param = Lwe::param_gen(log_q, log_p, n);
        let sk = Lwe::key_gen(&param, &mut rng);
        for (m0, m1) in (0..param.p()).cartesian_product(0..param.p()) {
            let [m0, m1] = [m0, m1].map(|m| Fq::from_u64(param.p(), m));
            let [pt0, pt1] = [m0, m1].map(|m| Lwe::encode(&param, &m));
            let [ct0, ct1] = [pt0, pt1].map(|pt| Lwe::encrypt(&param, &sk, &pt, &mut rng));
            let ct2 = Lwe::eval_sub(&param, &ct0, &ct1);
            let m2 = m0 - m1;
            assert_eq!(m2, Lwe::decode(&param, &Lwe::decrypt(&param, &sk, &ct2)));
        }
    }
}
