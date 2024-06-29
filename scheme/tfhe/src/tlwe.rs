use crate::util::{uniform_binaries, AdditiveVec, Dot, Round, TorusNormal, W64};
use core::{iter::repeat_with, num::Wrapping};
use rand::{distributions::Distribution, RngCore};

pub struct Tlwe;

#[derive(Clone, Copy, Debug)]
pub struct TlweParam {
    pub(crate) log_q: usize,
    pub(crate) log_p: usize,
    pub(crate) padding: usize,
    pub(crate) n: usize,
    pub(crate) m: usize,
    pub(crate) t_normal: TorusNormal,
}

impl TlweParam {
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

pub struct TlweSecretKey(pub(crate) Vec<bool>);

pub struct TlwePublicKey(AdditiveVec<AdditiveVec<W64>>, AdditiveVec<W64>);

impl TlwePublicKey {
    pub fn a(&self) -> &AdditiveVec<AdditiveVec<W64>> {
        &self.0
    }

    pub fn b(&self) -> &AdditiveVec<W64> {
        &self.1
    }
}

pub struct TlwePlaintext(pub(crate) W64);

pub struct TlweCiphertext(pub(crate) AdditiveVec<W64>, pub(crate) W64);

impl TlweCiphertext {
    pub fn a(&self) -> &AdditiveVec<W64> {
        &self.0
    }

    pub fn b(&self) -> &W64 {
        &self.1
    }
}

impl Tlwe {
    pub fn param_gen(
        log_q: usize,
        log_p: usize,
        padding: usize,
        n: usize,
        m: usize,
        std_dev: f64,
    ) -> TlweParam {
        TlweParam {
            log_q,
            log_p,
            padding,
            n,
            m,
            t_normal: TorusNormal::new(log_q, std_dev),
        }
    }

    pub fn key_gen(param: &TlweParam, rng: &mut impl RngCore) -> (TlweSecretKey, TlwePublicKey) {
        let sk = uniform_binaries(param.n, rng);
        let pk = {
            repeat_with(|| {
                let a = AdditiveVec::uniform_q(param.q(), param.n, rng);
                let e = param.t_normal.sample(rng);
                let b = (a.dot(&sk) + e) % param.q();
                (a, b)
            })
            .take(param.m)
            .unzip()
        };
        (TlweSecretKey(sk), TlwePublicKey(pk.0, pk.1))
    }

    pub fn encode(param: &TlweParam, m: &W64) -> TlwePlaintext {
        assert!(*m < param.p());
        TlwePlaintext(m << param.log_delta())
    }

    pub fn decode(param: &TlweParam, pt: &TlwePlaintext) -> W64 {
        (pt.0 >> param.log_delta()) % param.p()
    }

    pub fn encrypt(
        param: &TlweParam,
        pk: &TlwePublicKey,
        pt: &TlwePlaintext,
        rng: &mut impl RngCore,
    ) -> TlweCiphertext {
        let r = uniform_binaries(param.m, rng);
        let a = (pk.a().dot(&r)) % param.q();
        let b = (pk.b().dot(&r) + pt.0) % param.q();
        TlweCiphertext(a, b)
    }

    pub fn decrypt(param: &TlweParam, sk: &TlweSecretKey, ct: &TlweCiphertext) -> TlwePlaintext {
        let TlweCiphertext(a, b) = ct;
        let mu_star = (b - a.dot(&sk.0)) % param.q();
        let mu = mu_star.round(param.log_delta());
        TlwePlaintext(mu)
    }

    pub fn eval_add(
        param: &TlweParam,
        ct1: &TlweCiphertext,
        ct2: &TlweCiphertext,
    ) -> TlweCiphertext {
        TlweCiphertext(
            (ct1.a() + ct2.a()) % param.q(),
            (ct1.b() + ct2.b()) % param.q(),
        )
    }

    pub fn eval_sub(
        param: &TlweParam,
        ct1: &TlweCiphertext,
        ct2: &TlweCiphertext,
    ) -> TlweCiphertext {
        TlweCiphertext(
            (ct1.a() - ct2.a()) % param.q(),
            (ct1.b() - ct2.b()) % param.q(),
        )
    }
}

#[cfg(test)]
mod test {
    use crate::tlwe::Tlwe;
    use core::num::Wrapping;
    use rand::{
        rngs::{OsRng, StdRng},
        RngCore, SeedableRng,
    };

    #[test]
    fn encrypt_decrypt() {
        let mut rng = StdRng::seed_from_u64(OsRng.next_u64());
        let (log_q, log_p, padding, n, m, std_dev) = (32, 8, 1, 256, 32, 1.0e-08);
        let param = Tlwe::param_gen(log_q, log_p, padding, n, m, std_dev);
        let (sk, pk) = Tlwe::key_gen(&param, &mut rng);
        for m in (0..1 << log_p).map(Wrapping) {
            let pt = Tlwe::encode(&param, &m);
            let ct = Tlwe::encrypt(&param, &pk, &pt, &mut rng);
            assert_eq!(m, Tlwe::decode(&param, &Tlwe::decrypt(&param, &sk, &ct)));
        }
    }
}
