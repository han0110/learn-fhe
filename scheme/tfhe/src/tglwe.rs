use crate::{
    tlwe::{TlweCiphertext, TlweParam, TlweSecretKey},
    util::{
        avec, uniform_binaries, AdditiveVec, Dot, NegCyclicMul, Polynomial, Round, TorusNormal, W64,
    },
};
use core::{iter::repeat_with, num::Wrapping};
use itertools::{chain, Itertools};
use rand::RngCore;
use rand_distr::Distribution;

pub struct Tglwe;

#[derive(Clone, Copy, Debug)]
pub struct TglweParam {
    pub(crate) log_q: usize,
    pub(crate) log_p: usize,
    pub(crate) padding: usize,
    pub(crate) k: usize,
    pub(crate) n: usize,
    pub(crate) m: usize,
    pub(crate) t_normal: TorusNormal,
}

impl TglweParam {
    pub fn q(&self) -> W64 {
        Wrapping(1 << self.log_q)
    }

    pub fn p(&self) -> W64 {
        Wrapping(1 << self.log_p)
    }

    pub fn log_delta(&self) -> usize {
        self.log_q - (self.log_p + self.padding)
    }

    pub fn delta(&self) -> W64 {
        Wrapping(1 << self.log_delta())
    }
}

pub struct TglweSecretKey(pub(crate) Vec<Vec<bool>>);

pub struct TglwePublicKey(
    AdditiveVec<AdditiveVec<Polynomial<W64>>>,
    AdditiveVec<Polynomial<W64>>,
);

impl TglwePublicKey {
    pub fn a(&self) -> &AdditiveVec<AdditiveVec<Polynomial<W64>>> {
        &self.0
    }

    pub fn b(&self) -> &AdditiveVec<Polynomial<W64>> {
        &self.1
    }
}

pub struct TglwePlaintext(pub(crate) Polynomial<W64>);

pub struct TglweCiphertext(
    pub(crate) AdditiveVec<Polynomial<W64>>,
    pub(crate) Polynomial<W64>,
);

impl TglweCiphertext {
    pub fn a(&self) -> &AdditiveVec<Polynomial<W64>> {
        &self.0
    }

    pub fn b(&self) -> &Polynomial<W64> {
        &self.1
    }
}

impl Tglwe {
    pub fn param_gen(
        log_q: usize,
        log_p: usize,
        padding: usize,
        k: usize,
        n: usize,
        m: usize,
        std_dev: f64,
    ) -> TglweParam {
        TglweParam {
            log_q,
            log_p,
            padding,
            k,
            n,
            m,
            t_normal: TorusNormal::new(log_q, std_dev),
        }
    }

    pub fn key_gen(param: &TglweParam, rng: &mut impl RngCore) -> (TglweSecretKey, TglwePublicKey) {
        let sk = repeat_with(|| uniform_binaries(param.n, rng))
            .take(param.k)
            .collect_vec();
        let pk = {
            repeat_with(|| {
                let a = repeat_with(|| Polynomial::uniform_q(param.q(), param.n, rng))
                    .take(param.k)
                    .collect::<AdditiveVec<_>>();
                let e = repeat_with(|| param.t_normal.sample(rng))
                    .take(param.n)
                    .collect::<AdditiveVec<_>>();
                let b = (a.dot(&sk) + e) % param.q();
                (a, b)
            })
            .take(param.m)
            .unzip()
        };
        (TglweSecretKey(sk), TglwePublicKey(pk.0, pk.1))
    }

    pub fn encode(param: &TglweParam, m: &Polynomial<W64>) -> TglwePlaintext {
        assert_eq!(m.len(), param.n);
        assert!(m.iter().all(|m| *m < param.p()));
        TglwePlaintext(m * param.delta())
    }

    pub fn decode(param: &TglweParam, pt: &TglwePlaintext) -> Polynomial<W64> {
        ((&pt.0) >> param.log_delta()) % param.p()
    }

    pub fn recode(param: &TglweParam, sk: &TglweSecretKey) -> (TlweParam, TlweSecretKey) {
        let param = TlweParam {
            log_q: param.log_q,
            log_p: param.log_p,
            padding: param.padding,
            n: param.k * param.n,
            m: 1,
            t_normal: TorusNormal::new(0, 0.),
        };
        let sk = TlweSecretKey(sk.0.concat());
        (param, sk)
    }

    pub fn encrypt(
        param: &TglweParam,
        pk: &TglwePublicKey,
        pt: &TglwePlaintext,
        rng: &mut impl RngCore,
    ) -> TglweCiphertext {
        let r = uniform_binaries(param.m, rng);
        let a = (pk.a().dot(&r)) % param.q();
        let b = (pk.b().dot(&r) + &pt.0) % param.q();
        TglweCiphertext(a, b)
    }

    pub fn encrypt_const(param: &TglweParam, pt: &TglwePlaintext) -> TglweCiphertext {
        TglweCiphertext(avec![avec![Wrapping(0); param.n]; param.k], pt.0.clone())
    }

    pub fn decrypt(
        param: &TglweParam,
        sk: &TglweSecretKey,
        ct: &TglweCiphertext,
    ) -> TglwePlaintext {
        let TglweCiphertext(a, b) = ct;
        let mu_star = (b - a.dot(&sk.0)) % param.q();
        let mu = mu_star.round(param.log_delta());
        TglwePlaintext(mu)
    }

    pub fn eval_add(
        param: &TglweParam,
        ct1: &TglweCiphertext,
        ct2: &TglweCiphertext,
    ) -> TglweCiphertext {
        TglweCiphertext(
            (ct1.a() + ct2.a()) % param.q(),
            (ct1.b() + ct2.b()) % param.q(),
        )
    }

    pub fn eval_sub(
        param: &TglweParam,
        ct1: &TglweCiphertext,
        ct2: &TglweCiphertext,
    ) -> TglweCiphertext {
        TglweCiphertext(
            (ct1.a() - ct2.a()) % param.q(),
            (ct1.b() - ct2.b()) % param.q(),
        )
    }

    pub fn rotate(param: &TglweParam, ct: &TglweCiphertext, i: W64) -> TglweCiphertext {
        let p = Polynomial::monomial(i, param.n);
        TglweCiphertext(
            ct.a().iter().map(|a| a.negcyclic_mul(&p)).collect(),
            ct.b().negcyclic_mul(&p),
        )
    }

    pub fn sample_extract(param: &TglweParam, ct: &TglweCiphertext, i: usize) -> TlweCiphertext {
        assert!(i < param.n);
        let a = ct
            .a()
            .iter()
            .flat_map(|poly| {
                let (lo, hi) = poly.split_at(i + 1);
                chain![
                    lo.iter().rev().copied(),
                    hi.iter().rev().map(|value| (-value) % param.q())
                ]
            })
            .collect();
        let b = ct.b()[i];
        TlweCiphertext(a, b)
    }
}

#[cfg(test)]
mod test {
    use crate::{tglwe::Tglwe, tlwe::Tlwe, util::Polynomial};
    use rand::{
        rngs::{OsRng, StdRng},
        RngCore, SeedableRng,
    };

    #[test]
    fn encrypt_decrypt() {
        let mut rng = StdRng::seed_from_u64(OsRng.next_u64());
        let (log_q, log_p, padding, k, n, m, std_dev) = (32, 8, 1, 2, 256, 32, 1.0e-8);
        let param = Tglwe::param_gen(log_q, log_p, padding, k, n, m, std_dev);
        let (sk, pk) = Tglwe::key_gen(&param, &mut rng);
        for _ in 0..1 << log_p {
            let m = Polynomial::uniform_q(param.p(), n, &mut rng);
            let pt = Tglwe::encode(&param, &m);
            let ct = Tglwe::encrypt(&param, &pk, &pt, &mut rng);
            assert_eq!(m, Tglwe::decode(&param, &Tglwe::decrypt(&param, &sk, &ct)));
        }
    }

    #[test]
    fn sample_extract() {
        let mut rng = StdRng::seed_from_u64(OsRng.next_u64());
        let (log_q, log_p, padding, k, n, m, std_dev) = (32, 8, 1, 2, 256, 32, 1.0e-8);
        let param1 = Tglwe::param_gen(log_q, log_p, padding, k, n, m, std_dev);
        let (sk1, pk1) = Tglwe::key_gen(&param1, &mut rng);
        let (param2, sk2) = Tglwe::recode(&param1, &sk1);
        for i in 0..n {
            let m = Polynomial::uniform_q(param1.p(), n, &mut rng);
            let pt = Tglwe::encode(&param1, &m);
            let ct1 = Tglwe::encrypt(&param1, &pk1, &pt, &mut rng);
            let ct2 = Tglwe::sample_extract(&param1, &ct1, i);
            assert_eq!(
                m[i],
                Tlwe::decode(&param2, &Tlwe::decrypt(&param2, &sk2, &ct2))
            )
        }
    }
}
