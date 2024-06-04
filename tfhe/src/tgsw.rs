use crate::{
    tlwe::{Tlwe, TlweCiphertext, TlweParam, TlwePublicKey, TlweSecretKey},
    util::{AdditiveVec, Decompose, Dot, Round, W64},
};
use core::{iter::repeat_with, num::Wrapping, ops::Deref};
use itertools::chain;
use rand::RngCore;

pub struct Tgsw;

#[derive(Clone, Copy, Debug)]
pub struct TgswParam {
    pub tlwe: TlweParam,
    pub log_b: usize,
    pub ell: usize,
}

impl Deref for TgswParam {
    type Target = TlweParam;

    fn deref(&self) -> &Self::Target {
        &self.tlwe
    }
}

pub type TgswSecretKey = TlweSecretKey;

pub type TgswPublicKey = TlwePublicKey;

pub struct TgswPlaintext(W64);

pub struct TgswCiphertext(AdditiveVec<AdditiveVec<W64>>);

impl Tgsw {
    pub fn param_gen(tlwe: TlweParam, log_b: usize, ell: usize) -> TgswParam {
        assert!(tlwe.log_q >= log_b * ell);
        TgswParam { tlwe, log_b, ell }
    }

    pub fn key_gen(param: &TgswParam, rng: &mut impl RngCore) -> (TgswSecretKey, TgswPublicKey) {
        Tlwe::key_gen(param, rng)
    }

    pub fn encode(param: &TgswParam, m: &W64) -> TgswPlaintext {
        assert!(*m < param.p());
        TgswPlaintext(*m)
    }

    pub fn decode(param: &TgswParam, pt: &TgswPlaintext) -> W64 {
        pt.0 % param.p()
    }

    pub fn encrypt(
        param: &TgswParam,
        pk: &TgswPublicKey,
        pt: &TgswPlaintext,
        rng: &mut impl RngCore,
    ) -> TgswCiphertext {
        let zero = Tlwe::encode(param, &Wrapping(0));
        let mut ct = repeat_with(|| {
            let ct = Tlwe::encrypt(param, pk, &zero, rng);
            chain![ct.a(), [ct.b()]].copied().collect()
        })
        .take((param.n + 1) * param.ell)
        .collect::<AdditiveVec<AdditiveVec<_>>>();
        for col in 0..param.n + 1 {
            for i in 0..param.ell {
                ct[col * param.ell + i][col] += pt.0 << (param.log_q - (i + 1) * param.log_b);
                ct[col * param.ell + i][col] %= param.q();
            }
        }
        TgswCiphertext(ct)
    }

    pub fn decrypt(param: &TgswParam, sk: &TgswSecretKey, ct: &TgswCiphertext) -> TgswPlaintext {
        let (b, a) = ct.0.last().unwrap().split_last().unwrap();
        let ct = TlweCiphertext(a.into(), *b);
        let pt = Tlwe::decrypt(param, sk, &ct).0;
        TgswPlaintext(pt >> (param.log_q - param.ell * param.log_b))
    }

    pub fn external_product(
        param: &TgswParam,
        ct1: &TgswCiphertext,
        ct2: &TlweCiphertext,
    ) -> TlweCiphertext {
        let g_inv_ct2 = chain![ct2.a(), [ct2.b()]]
            .map(|value| value.round(param.log_q - param.log_b * param.ell))
            .flat_map(|value| value.decompose(param.log_q, param.log_b).take(param.ell))
            .collect::<AdditiveVec<_>>();
        let ct3 = g_inv_ct2.dot(&ct1.0) % param.q();
        let (b, a) = ct3.split_last().unwrap();
        TlweCiphertext(a.into(), *b)
    }

    pub fn cmux(
        param: &TgswParam,
        b: &TgswCiphertext,
        ct1: &TlweCiphertext,
        ct2: &TlweCiphertext,
    ) -> TlweCiphertext {
        let d = Tgsw::external_product(param, b, &Tlwe::eval_sub(param, ct2, ct1));
        Tlwe::eval_add(param, &d, ct1)
    }
}

#[cfg(test)]
mod test {
    use crate::{tgsw::Tgsw, tlwe::Tlwe};
    use core::{array::from_fn, num::Wrapping};
    use rand::{
        rngs::{OsRng, StdRng},
        RngCore, SeedableRng,
    };

    #[test]
    fn encrypt_decrypt() {
        let mut rng = StdRng::seed_from_u64(OsRng.next_u64());
        let (log_q, log_p, padding, n, m, std_dev, log_b, ell) = (32, 8, 1, 256, 32, 1.0e-8, 4, 2);
        let param = Tgsw::param_gen(
            Tlwe::param_gen(log_q, log_p, padding, n, m, std_dev),
            log_b,
            ell,
        );
        let (sk, pk) = Tgsw::key_gen(&param, &mut rng);
        for m in (0..1 << log_p).map(Wrapping) {
            let pt = Tgsw::encode(&param, &m);
            let ct = Tgsw::encrypt(&param, &pk, &pt, &mut rng);
            assert_eq!(m, Tgsw::decode(&param, &Tgsw::decrypt(&param, &sk, &ct)));
        }
    }

    #[test]
    fn external_product() {
        let mut rng = StdRng::seed_from_u64(OsRng.next_u64());
        let (log_q, log_p, padding, n, m, std_dev, log_b, ell) = (32, 8, 1, 256, 32, 1.0e-8, 4, 8);
        let param = Tgsw::param_gen(
            Tlwe::param_gen(log_q, log_p, padding, n, m, std_dev),
            log_b,
            ell,
        );
        let (sk, pk) = Tgsw::key_gen(&param, &mut rng);
        for _ in 0..1 << log_p {
            let [m1, m2] = from_fn(|_| Wrapping(rng.next_u64()) % param.p());
            let m3 = (m1 * m2) % param.p();
            let ct1 = Tgsw::encrypt(&param, &pk, &Tgsw::encode(&param, &m1), &mut rng);
            let ct2 = Tlwe::encrypt(&param, &pk, &Tlwe::encode(&param, &m2), &mut rng);
            let ct3 = Tgsw::external_product(&param, &ct1, &ct2);
            assert_eq!(m3, Tlwe::decode(&param, &Tlwe::decrypt(&param, &sk, &ct3)));
        }
    }

    #[test]
    fn cmux() {
        let mut rng = StdRng::seed_from_u64(OsRng.next_u64());
        let (log_q, log_p, padding, n, m, std_dev, log_b, ell) = (32, 8, 1, 256, 32, 1.0e-8, 4, 8);
        let param = Tgsw::param_gen(
            Tlwe::param_gen(log_q, log_p, padding, n, m, std_dev),
            log_b,
            ell,
        );
        let (sk, pk) = Tgsw::key_gen(&param, &mut rng);
        for _ in 0..1 << log_p {
            let [m1, m2] = from_fn(|_| Wrapping(rng.next_u64()) % param.p());
            let ct1 = Tlwe::encrypt(&param, &pk, &Tlwe::encode(&param, &m1), &mut rng);
            let ct2 = Tlwe::encrypt(&param, &pk, &Tlwe::encode(&param, &m2), &mut rng);
            for (b, m) in [(Wrapping(0), m1), (Wrapping(1), m2)] {
                let b = Tgsw::encrypt(&param, &pk, &Tgsw::encode(&param, &b), &mut rng);
                let ct3 = Tgsw::cmux(&param, &b, &ct1, &ct2);
                assert_eq!(m, Tlwe::decode(&param, &Tlwe::decrypt(&param, &sk, &ct3)));
            }
        }
    }
}
