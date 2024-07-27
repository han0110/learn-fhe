use crate::tglwe::{Tglwe, TglweCiphertext, TglweParam, TglwePlaintext, TglweSecretKey};
use core::{borrow::Borrow, iter::repeat_with};
use derive_more::{Add, Deref, Sub};
use itertools::{chain, izip, Itertools};
use rand::RngCore;
use util::{AVec, Base2Decomposable, Base2Decomposor, Dot, Rq, Rt, Zq, T64};

#[derive(Debug)]
pub struct Tggsw;

#[derive(Clone, Copy, Debug, Deref)]
pub struct TggswParam {
    #[deref]
    tglwe: TglweParam,
    decomposor: Base2Decomposor<T64>,
}

impl TggswParam {
    pub fn new(
        log_p: usize,
        padding: usize,
        big_n: usize,
        n: usize,
        std_dev: f64,
        log_b: usize,
        d: usize,
    ) -> Self {
        Self {
            tglwe: TglweParam::new(log_p, padding, big_n, n, std_dev),
            decomposor: Base2Decomposor::<T64>::new(log_b, d),
        }
    }

    fn decomposor(&self) -> &Base2Decomposor<T64> {
        &self.decomposor
    }
}

pub type TggswSecretKey = TglweSecretKey;

#[derive(Clone, Debug)]
pub struct TggswPlaintext(pub(crate) Rt);

#[derive(Clone, Debug, Add, Sub)]
pub struct TggswCiphertext(AVec<TglweCiphertext>);

impl TggswCiphertext {
    pub fn a(&self) -> impl Iterator<Item = &AVec<Rt>> {
        self.0.iter().map(TglweCiphertext::a)
    }

    pub fn b(&self) -> impl Iterator<Item = &Rt> {
        self.0.iter().map(TglweCiphertext::b)
    }
}

impl Tggsw {
    pub fn sk_gen(param: &TggswParam, rng: &mut impl RngCore) -> TggswSecretKey {
        Tglwe::sk_gen(param, rng)
    }

    pub fn encode(param: &TggswParam, m: Rq) -> TggswPlaintext {
        assert_eq!(m.n(), param.big_n());
        let encode = |m: Zq| m.to_u64().into();
        TggswPlaintext(m.into_iter().map(encode).collect())
    }

    pub fn decode(param: &TggswParam, TggswPlaintext(pt): TggswPlaintext) -> Rq {
        let decode = |pt: T64| Zq::from_u64(param.p(), pt.into());
        pt.into_iter().map(decode).collect()
    }

    pub fn sk_encrypt(
        param: &TggswParam,
        sk: &TggswSecretKey,
        TggswPlaintext(pt): TggswPlaintext,
        rng: &mut impl RngCore,
    ) -> TggswCiphertext {
        let pt = param.decomposor().power_up(pt).collect_vec();
        let zero = TglwePlaintext(Rt::zero(param.big_n()));
        let mut ct = repeat_with(|| Tglwe::sk_encrypt(param, sk, zero.clone(), rng))
            .take((param.n() + 1) * param.decomposor().d())
            .collect::<AVec<_>>();
        let (c0, c1) = ct.split_at_mut(param.n() * param.decomposor().d());
        izip!(0.., c0.chunks_mut(param.decomposor().d()))
            .for_each(|(j, c0)| izip!(c0, &pt).for_each(|(ct, pt)| ct.0[j] += pt));
        izip!(c1, &pt).for_each(|(ct, pt)| ct.1 += pt);
        TggswCiphertext(ct)
    }

    pub fn decrypt(
        param: &TggswParam,
        sk: &TggswSecretKey,
        TggswCiphertext(ct): TggswCiphertext,
    ) -> TggswPlaintext {
        let pt = Tglwe::decrypt(param, sk, ct.into_iter().last().unwrap()).0;
        TggswPlaintext(pt.rounding_shr(param.decomposor().log_bases().last().unwrap()))
    }

    pub fn external_product(
        param: &TggswParam,
        ct0: impl Borrow<TggswCiphertext>,
        ct1: impl Borrow<TglweCiphertext>,
    ) -> TglweCiphertext {
        let (ct0, ct1) = (ct0.borrow(), ct1.borrow());
        let ct1_limbs = chain![ct1.a(), [ct1.b()]]
            .flat_map(|v| param.decomposor().decompose(v))
            .collect_vec();
        let a = ct0.a().dot(&ct1_limbs);
        let b = ct0.b().dot(&ct1_limbs);
        TglweCiphertext(a, b)
    }

    pub fn cmux(
        param: &TggswParam,
        b: impl Borrow<TggswCiphertext>,
        ct0: TglweCiphertext,
        ct1: TglweCiphertext,
    ) -> TglweCiphertext {
        ct0.clone() + Tggsw::external_product(param, b, ct1 - ct0)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        tggsw::{Tggsw, TggswParam},
        tglwe::Tglwe,
    };
    use core::array::from_fn;
    use rand::thread_rng;
    use util::{Rq, Zq};

    #[test]
    fn encrypt_decrypt() {
        let mut rng = thread_rng();
        let (log_p, padding, big_n, n, std_dev, log_b, d) = (8, 1, 256, 2, 1.0e-8, 8, 8);
        let param = TggswParam::new(log_p, padding, big_n, n, std_dev, log_b, d);
        let sk = Tggsw::sk_gen(&param, &mut rng);
        for _ in 0..100 {
            let m = Rq::sample_uniform(param.p(), param.big_n(), &mut rng);
            let pt = Tggsw::encode(&param, m.clone());
            let ct = Tggsw::sk_encrypt(&param, &sk, pt.clone(), &mut rng);
            assert_eq!(m, Tggsw::decode(&param, Tggsw::decrypt(&param, &sk, ct)));
        }
    }

    #[test]
    fn external_product() {
        let mut rng = thread_rng();
        let (log_p, padding, big_n, n, std_dev, log_b, d) = (8, 1, 256, 2, 1.0e-8, 8, 8);
        let param = TggswParam::new(log_p, padding, big_n, n, std_dev, log_b, d);
        let sk = Tggsw::sk_gen(&param, &mut rng);
        for _ in 0..100 {
            let [m0, m1] = &from_fn(|_| Rq::sample_uniform(param.p(), param.big_n(), &mut rng));
            let m2 = m0 * m1;
            let ct0 = Tggsw::sk_encrypt(&param, &sk, Tggsw::encode(&param, m0.clone()), &mut rng);
            let ct1 = Tglwe::sk_encrypt(&param, &sk, Tglwe::encode(&param, m1.clone()), &mut rng);
            let ct2 = Tggsw::external_product(&param, ct0, ct1);
            assert_eq!(m2, Tglwe::decode(&param, Tglwe::decrypt(&param, &sk, ct2)));
        }
    }

    #[test]
    fn cmux() {
        let mut rng = thread_rng();
        let (log_p, padding, big_n, n, std_dev, log_b, d) = (8, 1, 256, 2, 1.0e-8, 8, 8);
        let param = TggswParam::new(log_p, padding, big_n, n, std_dev, log_b, d);
        let sk = Tggsw::sk_gen(&param, &mut rng);
        for _ in 0..100 {
            let [m0, m1] = &from_fn(|_| Rq::sample_uniform(param.p(), param.big_n(), &mut rng));
            for (b, m2) in [(0, m0), (1, m1)] {
                let b = Rq::constant(Zq::from_usize(param.p(), b), param.big_n());
                let b = Tggsw::sk_encrypt(&param, &sk, Tggsw::encode(&param, b), &mut rng);
                let pt = [m0, m1].map(|m| Tglwe::encode(&param, m.clone()));
                let [ct0, ct1] = pt.map(|pt| Tglwe::sk_encrypt(&param, &sk, pt, &mut rng));
                let ct2 = Tggsw::cmux(&param, b, ct0, ct1);
                assert_eq!(m2, &Tglwe::decode(&param, Tglwe::decrypt(&param, &sk, ct2)));
            }
        }
    }
}
