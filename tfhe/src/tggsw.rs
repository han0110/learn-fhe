use crate::{
    tglwe::{Tglwe, TglweCiphertext, TglweParam, TglwePublicKey, TglweSecretKey},
    tlwe::TlweCiphertext,
    util::{AdditiveVec, Decompose, Dot, Polynomial, Round, W64},
};
use core::{
    iter::{repeat, repeat_with},
    num::Wrapping,
    ops::Deref,
};
use itertools::{chain, izip};
use rand::RngCore;

pub struct Tggsw;

#[derive(Clone, Copy, Debug)]
pub struct TggswParam {
    pub tglwe: TglweParam,
    pub log_b: usize,
    pub ell: usize,
}

impl Deref for TggswParam {
    type Target = TglweParam;

    fn deref(&self) -> &Self::Target {
        &self.tglwe
    }
}

pub type TggswSecretKey = TglweSecretKey;

pub type TggswPublicKey = TglwePublicKey;

pub struct TggswPlaintext(Polynomial<W64>);

pub struct TggswCiphertext(AdditiveVec<AdditiveVec<Polynomial<W64>>>);

impl Tggsw {
    pub fn param_gen(tglwe: TglweParam, log_b: usize, ell: usize) -> TggswParam {
        assert!(tglwe.log_q >= log_b * ell);
        TggswParam { tglwe, log_b, ell }
    }

    pub fn key_gen(param: &TggswParam, rng: &mut impl RngCore) -> (TggswSecretKey, TggswPublicKey) {
        Tglwe::key_gen(param, rng)
    }

    pub fn encode(param: &TggswParam, m: &Polynomial<W64>) -> TggswPlaintext {
        assert_eq!(m.len(), param.n);
        assert!(m.iter().all(|m| *m < param.p()));
        TggswPlaintext(m.clone())
    }

    pub fn decode(param: &TggswParam, pt: &TggswPlaintext) -> Polynomial<W64> {
        (&pt.0) % param.p()
    }

    pub fn encrypt(
        param: &TggswParam,
        pk: &TggswPublicKey,
        pt: &TggswPlaintext,
        rng: &mut impl RngCore,
    ) -> TggswCiphertext {
        let zero = Tglwe::encode(param, &repeat(Wrapping(0)).take(param.n).collect());
        let mut ct = repeat_with(|| {
            let ct = Tglwe::encrypt(param, pk, &zero, rng);
            chain![ct.a(), [ct.b()]].cloned().collect()
        })
        .take((param.k + 1) * param.ell)
        .collect::<AdditiveVec<AdditiveVec<_>>>();
        for col in 0..param.k + 1 {
            for i in 0..param.ell {
                ct[col * param.ell + i][col] += (&pt.0) << (param.log_q - (i + 1) * param.log_b);
                ct[col * param.ell + i][col] %= param.q();
            }
        }
        TggswCiphertext(ct)
    }

    pub fn decrypt(param: &TggswParam, sk: &TggswSecretKey, ct: &TggswCiphertext) -> TggswPlaintext {
        let (b, a) = ct.0.last().unwrap().split_last().unwrap();
        let ct = TglweCiphertext(a.into(), b.clone());
        let pt = Tglwe::decrypt(param, sk, &ct).0;
        TggswPlaintext(pt >> (param.log_q - param.ell * param.log_b))
    }

    pub fn external_product(param: &TggswParam, ct1: &TggswCiphertext, ct2: &TglweCiphertext) -> TglweCiphertext {
        let g_inv_ct2 = chain![ct2.a(), [ct2.b()]]
            .map(|value| value.clone().round(param.log_q - param.log_b * param.ell))
            .flat_map(|value| value.decompose(param.log_q, param.log_b).take(param.ell))
            .collect::<AdditiveVec<_>>();
        let ct3 = g_inv_ct2.dot(&ct1.0) % param.q();
        let (b, a) = ct3.split_last().unwrap();
        TglweCiphertext(a.into(), b.clone())
    }

    pub fn cmux(
        param: &TggswParam,
        b: &TggswCiphertext,
        ct1: &TglweCiphertext,
        ct2: &TglweCiphertext,
    ) -> TglweCiphertext {
        let d = Tggsw::external_product(param, b, &Tglwe::eval_sub(param, ct2, ct1));
        Tglwe::eval_add(param, &d, ct1)
    }

    pub fn blind_rotate(
        param: &TggswParam,
        bs: &[TggswCiphertext],
        v: &Polynomial<W64>,
        ct: &TlweCiphertext,
    ) -> TglweCiphertext {
        let v = Tglwe::encrypt_const(param, &Tglwe::encode(param, v));
        let ct = modulus_switch(param.log_q, 1 + param.n.ilog2() as usize, ct);
        izip!(bs, ct.a()).fold(Tglwe::rotate(param, &v, -ct.b()), |c, (b, a)| {
            Tggsw::cmux(param, b, &c, &Tglwe::rotate(param, &c, *a))
        })
    }
}

fn modulus_switch(log_from: usize, log_to: usize, ct: &TlweCiphertext) -> TlweCiphertext {
    TlweCiphertext(
        ct.a().iter().map(|a| a.round_shr(log_from - log_to)).collect(),
        ct.b().round_shr(log_from - log_to),
    )
}

#[cfg(test)]
mod test {
    use crate::{
        tggsw::Tggsw,
        tglwe::Tglwe,
        util::{NegCyclicMul, Polynomial},
    };
    use core::{array::from_fn, num::Wrapping};
    use rand::{
        rngs::{OsRng, StdRng},
        RngCore, SeedableRng,
    };

    #[test]
    fn encrypt_decrypt() {
        let mut rng = StdRng::seed_from_u64(OsRng.next_u64());
        let (log_q, log_p, padding, k, n, m, std_dev, log_b, ell) = (32, 8, 1, 2, 256, 32, 1.0e-8, 4, 2);
        let param = Tggsw::param_gen(Tglwe::param_gen(log_q, log_p, padding, k, n, m, std_dev), log_b, ell);
        let (sk, pk) = Tggsw::key_gen(&param, &mut rng);
        for _ in 0..1 << log_p {
            let m = Polynomial::uniform_q(param.p(), n, &mut rng);
            let pt = Tggsw::encode(&param, &m);
            let ct = Tggsw::encrypt(&param, &pk, &pt, &mut rng);
            assert_eq!(m, Tggsw::decode(&param, &Tggsw::decrypt(&param, &sk, &ct)));
        }
    }

    #[test]
    fn external_product() {
        let mut rng = StdRng::seed_from_u64(OsRng.next_u64());
        let (log_q, log_p, padding, k, n, m, std_dev, log_b, ell) = (32, 8, 1, 2, 256, 32, 1.0e-8, 4, 8);
        let param = Tggsw::param_gen(Tglwe::param_gen(log_q, log_p, padding, k, n, m, std_dev), log_b, ell);
        let (sk, pk) = Tggsw::key_gen(&param, &mut rng);
        for _ in 0..1 << log_p {
            let [m1, m2] = from_fn(|_| Polynomial::uniform_q(param.p(), n, &mut rng));
            let m3 = (m1.negcyclic_mul(&m2)) % param.p();
            let ct1 = Tggsw::encrypt(&param, &pk, &Tggsw::encode(&param, &m1), &mut rng);
            let ct2 = Tglwe::encrypt(&param, &pk, &Tglwe::encode(&param, &m2), &mut rng);
            let ct3 = Tggsw::external_product(&param, &ct1, &ct2);
            assert_eq!(m3, Tglwe::decode(&param, &Tglwe::decrypt(&param, &sk, &ct3)));
        }
    }

    #[test]
    fn cmux() {
        let mut rng = StdRng::seed_from_u64(OsRng.next_u64());
        let (log_q, log_p, padding, k, n, m, std_dev, log_b, ell) = (32, 8, 1, 2, 256, 32, 1.0e-8, 4, 8);
        let param = Tggsw::param_gen(Tglwe::param_gen(log_q, log_p, padding, k, n, m, std_dev), log_b, ell);
        let (sk, pk) = Tggsw::key_gen(&param, &mut rng);
        for _ in 0..1 << log_p {
            let [b1, b2] = from_fn(|b| Polynomial::constant(Wrapping(b as u64), n));
            let [m1, m2] = from_fn(|_| Polynomial::uniform_q(param.p(), n, &mut rng));
            let ct1 = Tglwe::encrypt(&param, &pk, &Tglwe::encode(&param, &m1), &mut rng);
            let ct2 = Tglwe::encrypt(&param, &pk, &Tglwe::encode(&param, &m2), &mut rng);
            for (b, m) in [(b1, m1), (b2, m2)] {
                let b = Tggsw::encrypt(&param, &pk, &Tggsw::encode(&param, &b), &mut rng);
                let ct3 = Tggsw::cmux(&param, &b, &ct1, &ct2);
                assert_eq!(m, Tglwe::decode(&param, &Tglwe::decrypt(&param, &sk, &ct3)));
            }
        }
    }
}
