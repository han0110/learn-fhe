use crate::tlwe::{Tlwe, TlweCiphertext, TlweParam, TlwePlaintext, TlweSecretKey};
use core::iter::repeat_with;
use derive_more::{Add, Deref, Sub};
use itertools::{chain, izip, Itertools};
use rand::RngCore;
use util::{AVec, Base2Decomposable, Base2Decomposor, Dot, Zq, T64};

#[derive(Debug)]
pub struct Tgsw;

#[derive(Clone, Copy, Debug, Deref)]
pub struct TgswParam {
    #[deref]
    tlwe: TlweParam,
    decomposor: Base2Decomposor<T64>,
}

impl TgswParam {
    pub fn new(
        log_p: usize,
        padding: usize,
        n: usize,
        std_dev: f64,
        log_b: usize,
        d: usize,
    ) -> Self {
        Self {
            tlwe: TlweParam::new(log_p, padding, n, std_dev),
            decomposor: Base2Decomposor::<T64>::new(log_b, d),
        }
    }

    fn decomposor(&self) -> &Base2Decomposor<T64> {
        &self.decomposor
    }
}

pub type TgswSecretKey = TlweSecretKey;

#[derive(Clone, Debug)]
pub struct TgswPlaintext(T64);

#[derive(Clone, Debug, Add, Sub)]
pub struct TgswCiphertext(AVec<TlweCiphertext>);

impl TgswCiphertext {
    pub fn a(&self) -> impl Iterator<Item = &AVec<T64>> {
        self.0.iter().map(TlweCiphertext::a)
    }

    pub fn b(&self) -> impl Iterator<Item = &T64> {
        self.0.iter().map(TlweCiphertext::b)
    }
}

impl Tgsw {
    pub fn sk_gen(param: &TgswParam, rng: &mut impl RngCore) -> TgswSecretKey {
        Tlwe::sk_gen(param, rng)
    }

    pub fn encode(param: &TgswParam, m: Zq) -> TgswPlaintext {
        assert_eq!(m.q(), param.p());
        TgswPlaintext(m.to_u64().into())
    }

    pub fn decode(param: &TgswParam, TgswPlaintext(pt): TgswPlaintext) -> Zq {
        Zq::from_u64(param.p(), pt.into())
    }

    pub fn sk_encrypt(
        param: &TgswParam,
        sk: &TgswSecretKey,
        TgswPlaintext(pt): TgswPlaintext,
        rng: &mut impl RngCore,
    ) -> TgswCiphertext {
        let pt = param.decomposor().power_up(pt).collect_vec();
        let zero = TlwePlaintext(0.into());
        let mut ct = repeat_with(|| Tlwe::sk_encrypt(param, sk, zero.clone(), rng))
            .take((param.n() + 1) * param.decomposor().d())
            .collect::<AVec<_>>();
        let (c0, c1) = ct.split_at_mut(param.n() * param.decomposor().d());
        izip!(0.., c0.chunks_mut(param.decomposor().d()))
            .for_each(|(j, c0)| izip!(c0, &pt).for_each(|(ct, pt)| ct.0[j] += pt));
        izip!(c1, &pt).for_each(|(ct, pt)| ct.1 += pt);
        TgswCiphertext(ct)
    }

    pub fn decrypt(
        param: &TgswParam,
        sk: &TgswSecretKey,
        TgswCiphertext(ct): TgswCiphertext,
    ) -> TgswPlaintext {
        let pt = Tlwe::decrypt(param, sk, ct.into_iter().last().unwrap()).0;
        TgswPlaintext(pt.rounding_shr(param.decomposor().log_bases().last().unwrap()))
    }

    pub fn external_product(
        param: &TgswParam,
        ct0: TgswCiphertext,
        ct1: TlweCiphertext,
    ) -> TlweCiphertext {
        let ct1_limbs = chain![ct1.a(), [ct1.b()]]
            .flat_map(|v| param.decomposor().decompose(v))
            .collect_vec();
        let a = ct0.a().dot(&ct1_limbs);
        let b = ct0.b().dot(&ct1_limbs);
        TlweCiphertext(a, b)
    }

    pub fn cmux(
        param: &TgswParam,
        b: TgswCiphertext,
        ct0: TlweCiphertext,
        ct1: TlweCiphertext,
    ) -> TlweCiphertext {
        ct0.clone() + Tgsw::external_product(param, b, ct1 - ct0)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        tgsw::{Tgsw, TgswParam},
        tlwe::Tlwe,
    };
    use core::array::from_fn;
    use rand::thread_rng;
    use util::Zq;

    #[test]
    fn encrypt_decrypt() {
        let mut rng = thread_rng();
        let (log_p, padding, n, std_dev, log_b, d) = (8, 1, 256, 1.0e-8, 8, 8);
        let param = TgswParam::new(log_p, padding, n, std_dev, log_b, d);
        let sk = Tgsw::sk_gen(&param, &mut rng);
        for m in 0..param.p() {
            let m = Zq::from_u64(param.p(), m);
            let pt = Tgsw::encode(&param, m);
            let ct = Tgsw::sk_encrypt(&param, &sk, pt.clone(), &mut rng);
            assert_eq!(m, Tgsw::decode(&param, Tgsw::decrypt(&param, &sk, ct)));
        }
    }

    #[test]
    fn external_product() {
        let mut rng = thread_rng();
        let (log_p, padding, n, std_dev, log_b, d) = (8, 1, 256, 1.0e-8, 8, 8);
        let param = TgswParam::new(log_p, padding, n, std_dev, log_b, d);
        let sk = Tgsw::sk_gen(&param, &mut rng);
        for _ in 0..100 {
            let [m0, m1] = from_fn(|_| Zq::sample_uniform(param.p(), &mut rng));
            let m2 = m0 * m1;
            let ct0 = Tgsw::sk_encrypt(&param, &sk, Tgsw::encode(&param, m0), &mut rng);
            let ct1 = Tlwe::sk_encrypt(&param, &sk, Tlwe::encode(&param, m1), &mut rng);
            let ct2 = Tgsw::external_product(&param, ct0, ct1);
            assert_eq!(m2, Tlwe::decode(&param, Tlwe::decrypt(&param, &sk, ct2)));
        }
    }

    #[test]
    fn cmux() {
        let mut rng = thread_rng();
        let (log_p, padding, n, std_dev, log_b, d) = (8, 1, 256, 1.0e-8, 8, 8);
        let param = TgswParam::new(log_p, padding, n, std_dev, log_b, d);
        let sk = Tgsw::sk_gen(&param, &mut rng);
        for _ in 0..100 {
            let [m0, m1] = from_fn(|_| Zq::sample_uniform(param.p(), &mut rng));
            for (b, m2) in [(0, m0), (1, m1)] {
                let b = Zq::from_usize(param.p(), b);
                let b = Tgsw::sk_encrypt(&param, &sk, Tgsw::encode(&param, b), &mut rng);
                let pt = [m0, m1].map(|m| Tlwe::encode(&param, m));
                let [ct0, ct1] = pt.map(|pt| Tlwe::sk_encrypt(&param, &sk, pt, &mut rng));
                let ct2 = Tgsw::cmux(&param, b, ct0, ct1);
                assert_eq!(m2, Tlwe::decode(&param, Tlwe::decrypt(&param, &sk, ct2)));
            }
        }
    }
}
