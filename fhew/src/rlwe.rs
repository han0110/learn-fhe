use crate::util::{dg, zo, Fq, Poly};
use rand::RngCore;

pub struct Rlwe;

#[derive(Clone, Copy, Debug)]
pub struct RlweParam {
    q: u64,
    p: u64,
    n: usize,
}

impl RlweParam {
    pub fn new(q: u64, p: u64, n: usize) -> Self {
        assert!(q > p);
        assert!(n.is_power_of_two());

        Self { q, p, n }
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
}

pub struct RlweSecretKey(Poly<Fq>);

pub struct RlwePublicKey(Poly<Fq>, Poly<Fq>);

impl RlwePublicKey {
    pub fn a(&self) -> &Poly<Fq> {
        &self.0
    }

    pub fn b(&self) -> &Poly<Fq> {
        &self.1
    }
}

pub struct RlwePlaintext(pub(crate) Poly<Fq>);

pub struct RlweCiphertext(pub(crate) Poly<Fq>, pub(crate) Poly<Fq>);

impl RlweCiphertext {
    pub fn a(&self) -> &Poly<Fq> {
        &self.0
    }

    pub fn b(&self) -> &Poly<Fq> {
        &self.1
    }
}

impl Rlwe {
    pub fn key_gen(param: &RlweParam, rng: &mut impl RngCore) -> (RlweSecretKey, RlwePublicKey) {
        let sk = RlweSecretKey(Poly::sample_fq_from_i8(param.n(), param.q(), &zo(0.5), rng));
        let pk = {
            let a = Poly::sample_fq_uniform(param.n(), param.q(), rng);
            let e = Poly::sample_fq_from_i8(param.n(), param.q(), &dg(3.2, 6), rng);
            let b = &a * &sk.0 + e;
            RlwePublicKey(a, b)
        };
        (sk, pk)
    }

    pub fn encode(param: &RlweParam, m: &Poly<Fq>) -> RlwePlaintext {
        assert_eq!(m.n(), param.n());
        assert!(m.iter().all(|m| m.q() == param.p()));

        let scale_up = |m| Fq::from_f64(param.q(), f64::from(m) * param.delta());
        RlwePlaintext(m.iter().map(scale_up).collect())
    }

    pub fn decode(param: &RlweParam, pt: &RlwePlaintext) -> Poly<Fq> {
        let scale_down = |m| Fq::from_f64(param.p(), f64::from(m) / param.delta());
        pt.0.iter().map(scale_down).collect()
    }

    pub fn encrypt(
        param: &RlweParam,
        pk: &RlwePublicKey,
        pt: &RlwePlaintext,
        rng: &mut impl RngCore,
    ) -> RlweCiphertext {
        let u = &Poly::sample_fq_from_i8(param.n(), param.q(), &zo(0.5), rng);
        let e0 = Poly::sample_fq_from_i8(param.n(), param.q(), &dg(3.2, 6), rng);
        let e1 = Poly::sample_fq_from_i8(param.n(), param.q(), &dg(3.2, 6), rng);
        let a = pk.a() * u + e0;
        let b = pk.b() * u + e1 + &pt.0;
        RlweCiphertext(a, b)
    }

    pub fn decrypt(_: &RlweParam, sk: &RlweSecretKey, ct: &RlweCiphertext) -> RlwePlaintext {
        let pt = ct.b() - ct.a() * &sk.0;
        RlwePlaintext(pt)
    }

    pub fn eval_add(_: &RlweParam, ct0: &RlweCiphertext, ct1: &RlweCiphertext) -> RlweCiphertext {
        RlweCiphertext(ct0.a() + ct1.a(), ct0.b() + ct1.b())
    }

    pub fn eval_sub(_: &RlweParam, ct0: &RlweCiphertext, ct1: &RlweCiphertext) -> RlweCiphertext {
        RlweCiphertext(ct0.a() - ct1.a(), ct0.b() - ct1.b())
    }
}

#[cfg(test)]
mod test {
    use crate::{
        rlwe::{Rlwe, RlweParam},
        util::{two_adic_primes, Poly},
    };
    use core::array::from_fn;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn encrypt_decrypt() {
        let mut rng = StdRng::from_entropy();
        let (log_q, log_p) = (45, 4);
        for log_n in 0..10 {
            let (n, p) = (1 << log_n, 1 << log_p);
            for q in two_adic_primes(log_q, log_n + 1).take(10) {
                let param = RlweParam::new(q, p, n);
                let (sk, pk) = Rlwe::key_gen(&param, &mut rng);
                let m = Poly::sample_fq_uniform(n, p, &mut rng);
                let pt = Rlwe::encode(&param, &m);
                let ct = Rlwe::encrypt(&param, &pk, &pt, &mut rng);
                assert_eq!(m, Rlwe::decode(&param, &Rlwe::decrypt(&param, &sk, &ct)));
            }
        }
    }

    #[test]
    fn eval_add() {
        let mut rng = StdRng::from_entropy();
        let (log_q, log_p) = (45, 4);
        for log_n in 0..10 {
            let (n, p) = (1 << log_n, 1 << log_p);
            for q in two_adic_primes(log_q, log_n + 1).take(10) {
                let param = RlweParam::new(q, p, n);
                let (sk, pk) = Rlwe::key_gen(&param, &mut rng);
                let [m0, m1] = &from_fn(|_| Poly::sample_fq_uniform(n, p, &mut rng));
                let [pt0, pt1] = [m0, m1].map(|m| Rlwe::encode(&param, m));
                let [ct0, ct1] = [pt0, pt1].map(|pt| Rlwe::encrypt(&param, &pk, &pt, &mut rng));
                let ct2 = Rlwe::eval_add(&param, &ct0, &ct1);
                let m2 = m0 + m1;
                assert_eq!(m2, Rlwe::decode(&param, &Rlwe::decrypt(&param, &sk, &ct2)));
            }
        }
    }

    #[test]
    fn eval_sub() {
        let mut rng = StdRng::from_entropy();
        let (log_q, log_p) = (45, 4);
        for log_n in 0..10 {
            let (n, p) = (1 << log_n, 1 << log_p);
            for q in two_adic_primes(log_q, log_n + 1).take(10) {
                let param = RlweParam::new(q, p, n);
                let (sk, pk) = Rlwe::key_gen(&param, &mut rng);
                let [m0, m1] = &from_fn(|_| Poly::sample_fq_uniform(n, p, &mut rng));
                let [pt0, pt1] = [m0, m1].map(|m| Rlwe::encode(&param, m));
                let [ct0, ct1] = [pt0, pt1].map(|pt| Rlwe::encrypt(&param, &pk, &pt, &mut rng));
                let ct2 = Rlwe::eval_sub(&param, &ct0, &ct1);
                let m2 = m0 - m1;
                assert_eq!(m2, Rlwe::decode(&param, &Rlwe::decrypt(&param, &sk, &ct2)));
            }
        }
    }
}
