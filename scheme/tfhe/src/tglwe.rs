use crate::tlwe::{Tlwe, TlweCiphertext, TlweParam, TlwePlaintext, TlweSecretKey};
use core::iter::repeat_with;
use derive_more::{Add, Deref, Sub};
use itertools::{chain, Itertools};
use rand::RngCore;
use util::{tdg, AVec, Base2Decomposable, Dot, Rq, Rt, X};

#[derive(Debug)]
pub struct Tglwe;

#[derive(Clone, Copy, Debug, Deref)]
pub struct TglweParam {
    #[deref]
    tlwe: TlweParam,
    big_n: usize,
    n: usize,
}

impl TglweParam {
    pub fn new(log_p: usize, padding: usize, big_n: usize, n: usize, std_dev: f64) -> Self {
        Self {
            tlwe: TlweParam::new(log_p, padding, big_n * n, std_dev),
            big_n,
            n,
        }
    }

    pub fn big_n(&self) -> usize {
        self.big_n
    }

    pub fn n(&self) -> usize {
        self.n
    }
}

#[derive(Clone, Debug, Deref)]
pub struct TglweSecretKey(TlweSecretKey);

impl TglweSecretKey {
    fn as_rings(&self, big_n: usize) -> impl Iterator<Item = AVec<i64>> + '_ {
        self.0 .0.chunks(big_n).map_into()
    }
}

#[derive(Clone, Debug)]
pub struct TglwePlaintext(pub(crate) Rt);

#[derive(Clone, Debug, Add, Sub)]
pub struct TglweCiphertext(pub(crate) AVec<Rt>, pub(crate) Rt);

impl TglweCiphertext {
    pub fn a(&self) -> &AVec<Rt> {
        &self.0
    }

    pub fn b(&self) -> &Rt {
        &self.1
    }

    pub fn rotate(&self, i: i64) -> Self {
        Self(
            self.a().iter().map(|a| a * (X ^ i)).collect(),
            self.b() * (X ^ i),
        )
    }
}

impl From<(usize, TglwePlaintext)> for TglweCiphertext {
    fn from((n, TglwePlaintext(b)): (usize, TglwePlaintext)) -> Self {
        Self(vec![Rt::zero(b.n()); n].into(), b)
    }
}

impl Tglwe {
    pub fn sk_gen(param: &TglweParam, rng: &mut impl RngCore) -> TglweSecretKey {
        TglweSecretKey(Tlwe::sk_gen(param, rng))
    }

    pub fn encode(param: &TglweParam, m: Rq) -> TglwePlaintext {
        assert_eq!(m.n(), param.big_n());
        let encode = |m| Tlwe::encode(param, m).0;
        TglwePlaintext(m.into_iter().map(encode).collect())
    }

    pub fn decode(param: &TglweParam, TglwePlaintext(pt): TglwePlaintext) -> Rq {
        let decode = |pt| Tlwe::decode(param, TlwePlaintext(pt));
        pt.into_iter().map(decode).collect()
    }

    pub fn sk_encrypt(
        param: &TglweParam,
        sk: &TglweSecretKey,
        TglwePlaintext(pt): TglwePlaintext,
        rng: &mut impl RngCore,
    ) -> TglweCiphertext {
        let a = repeat_with(|| Rt::sample_uniform(param.big_n(), rng))
            .take(param.n())
            .collect::<AVec<_>>();
        let e = Rt::sample(param.big_n(), tdg(param.std_dev()), rng);
        let b = a.dot(sk.as_rings(param.big_n())) + e + pt;
        TglweCiphertext(a, b)
    }

    pub fn decrypt(
        param: &TglweParam,
        sk: &TglweSecretKey,
        TglweCiphertext(a, b): TglweCiphertext,
    ) -> TglwePlaintext {
        let mu_star = b - a.dot(sk.as_rings(param.big_n()));
        let mu = mu_star.round(param.log_delta());
        TglwePlaintext(mu)
    }

    pub fn sample_extract(param: &TglweParam, ct: TglweCiphertext, i: usize) -> TlweCiphertext {
        assert!(i < param.big_n());
        let a = chain![ct.a()]
            .flat_map(|a| {
                chain![
                    a[..i + 1].iter().rev().copied(),
                    a[i + 1..].iter().rev().map(|v| -v)
                ]
            })
            .collect();
        let b = ct.b()[i];
        TlweCiphertext(a, b)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        tglwe::{Tglwe, TglweParam},
        tlwe::Tlwe,
    };
    use rand::thread_rng;
    use util::Rq;

    #[test]
    fn encrypt_decrypt() {
        let mut rng = thread_rng();
        let (log_p, padding, big_n, n, std_dev) = (8, 1, 256, 2, 1.0e-8);
        let param = TglweParam::new(log_p, padding, big_n, n, std_dev);
        let sk = Tglwe::sk_gen(&param, &mut rng);
        for _ in 0..100 {
            let m = Rq::sample_uniform(param.p(), param.big_n(), &mut rng);
            let pt = Tglwe::encode(&param, m.clone());
            let ct = Tglwe::sk_encrypt(&param, &sk, pt, &mut rng);
            assert_eq!(m, Tglwe::decode(&param, Tglwe::decrypt(&param, &sk, ct)));
        }
    }

    #[test]
    fn sample_extract() {
        let mut rng = thread_rng();
        let (log_p, padding, big_n, n, std_dev) = (8, 1, 256, 2, 1.0e-8);
        let param = TglweParam::new(log_p, padding, big_n, n, std_dev);
        let sk = Tglwe::sk_gen(&param, &mut rng);
        for i in 0..param.big_n() {
            let m = Rq::sample_uniform(param.p(), param.big_n(), &mut rng);
            let pt = Tglwe::encode(&param, m.clone());
            let ct0 = Tglwe::sk_encrypt(&param, &sk, pt, &mut rng);
            let ct1 = Tglwe::sample_extract(&param, ct0, i);
            assert_eq!(m[i], Tlwe::decode(&param, Tlwe::decrypt(&param, &sk, ct1)))
        }
    }
}
