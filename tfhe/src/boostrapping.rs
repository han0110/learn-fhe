use crate::{
    tggsw::{Tggsw, TggswCiphertext, TggswParam, TggswPublicKey, TggswSecretKey},
    tglwe::Tglwe,
    tlwe::{Tlwe, TlweCiphertext, TlweParam, TlwePlaintext, TlwePublicKey, TlweSecretKey},
    util::{AdditiveVec, Decompose, Dot, Polynomial, Round, W64},
};
use core::num::Wrapping;
use itertools::chain;
use rand::RngCore;

pub struct Boostrapping;

pub struct BoostrappingParam {
    tlwe: TlweParam,
    tggsw: TggswParam,
    log_b: usize,
    ell: usize,
}

pub struct BoostrappingKey(Vec<TggswCiphertext>, AdditiveVec<AdditiveVec<W64>>);

impl Boostrapping {
    pub fn param_gen(tlwe: TlweParam, tggsw: TggswParam, log_b: usize, ell: usize) -> BoostrappingParam {
        BoostrappingParam {
            tlwe,
            tggsw,
            log_b,
            ell,
        }
    }

    pub fn key_gen(
        param: &BoostrappingParam,
        sk1: &TlweSecretKey,
        pk1: &TlwePublicKey,
        sk2: &TggswSecretKey,
        pk2: &TggswPublicKey,
        rng: &mut impl RngCore,
    ) -> BoostrappingKey {
        let k1 = {
            let param = &param.tggsw;
            sk1.0
                .iter()
                .map(|b| {
                    let b = Polynomial::constant(Wrapping(*b as u64), param.n);
                    Tggsw::encrypt(param, pk2, &Tggsw::encode(param, &b), rng)
                })
                .collect()
        };
        let k2 = {
            let (log_b, ell) = (param.log_b, param.ell);
            let param = &param.tlwe;
            sk2.0
                .iter()
                .flatten()
                .flat_map(|b| (0..ell).map(|i| (*b as u64) << (param.log_q - (i + 1) * log_b)))
                .map(|b| {
                    let ct = Tlwe::encrypt(param, pk1, &TlwePlaintext(Wrapping(b)), rng);
                    chain![ct.a(), [ct.b()]].copied().collect()
                })
                .collect()
        };
        BoostrappingKey(k1, k2)
    }

    pub fn boostrap(
        param: &BoostrappingParam,
        bsk: &BoostrappingKey,
        v: &Polynomial<W64>,
        ct: &TlweCiphertext,
    ) -> TlweCiphertext {
        let ct = {
            let param = &param.tggsw;
            Tggsw::blind_rotate(param, &bsk.0, v, ct)
        };
        let ct = {
            let param = &param.tggsw;
            Tglwe::sample_extract(param, &ct, 0)
        };
        let ct = {
            let (log_b, ell) = (param.log_b, param.ell);
            let param = &param.tlwe;
            let g_inv_a = ct
                .a()
                .iter()
                .map(|a| a.round(param.log_q - log_b * ell))
                .flat_map(|a| a.decompose(param.log_q, log_b).take(ell))
                .collect::<AdditiveVec<_>>();
            let composition = g_inv_a.dot(&bsk.1);
            let (b, a) = composition.into_split_last().unwrap();
            TlweCiphertext((-a) % param.q(), (ct.b() - b) % param.q())
        };
        ct
    }
}

#[cfg(test)]
mod test {
    use crate::{
        boostrapping::Boostrapping,
        tggsw::Tggsw,
        tglwe::Tglwe,
        tlwe::Tlwe,
        util::{Polynomial, W64},
    };
    use core::{convert::identity, iter::repeat, num::Wrapping, ops::Neg};
    use rand::{
        rngs::{OsRng, StdRng},
        RngCore, SeedableRng,
    };

    fn programmable_poly(log_p: usize, n: usize, f: &impl Fn(W64) -> W64) -> Polynomial<W64> {
        let p = Wrapping(1 << log_p);
        let reps = n >> log_p;
        let mut v = (0..1 << log_p)
            .map(Wrapping)
            .flat_map(|i| repeat(f(i) % p).take(reps))
            .collect::<Polynomial<_>>();
        v[0..reps >> 1].iter_mut().for_each(|v| *v = v.neg() % p);
        v.rotate_left(reps >> 1);
        v
    }

    fn plus_3(i: W64) -> W64 {
        i + Wrapping(3)
    }

    fn double(i: W64) -> W64 {
        i << 1
    }

    fn parity(i: W64) -> W64 {
        i % Wrapping(2)
    }

    #[test]
    fn boostrap() {
        let mut rng = StdRng::seed_from_u64(OsRng.next_u64());
        let (log_q, log_p, padding, k, n, m, std_dev, log_b, ell) = (32, 3, 1, 1, 256, 32, 1.0e-8, 4, 8);
        let param1 = Tlwe::param_gen(log_q, log_p, padding, n, m, std_dev);
        let (sk1, pk1) = Tlwe::key_gen(&param1, &mut rng);
        let param2 = Tggsw::param_gen(Tglwe::param_gen(log_q, log_p, padding, k, n, m, std_dev), log_b, ell);
        let (sk2, pk2) = Tggsw::key_gen(&param2, &mut rng);
        let param3 = Boostrapping::param_gen(param1, param2, log_b, ell);
        let bsk = Boostrapping::key_gen(&param3, &sk1, &pk1, &sk2, &pk2, &mut rng);
        for f in [identity, plus_3, double, parity] {
            let v = programmable_poly(log_p, n, &f);
            for m in (0..1 << log_p).map(Wrapping) {
                let ct1 = Tlwe::encrypt(&param1, &pk1, &Tlwe::encode(&param1, &m), &mut rng);
                let ct2 = Boostrapping::boostrap(&param3, &bsk, &v, &ct1);
                assert_eq!(
                    f(m) % param1.p(),
                    Tlwe::decode(&param1, &Tlwe::decrypt(&param1, &sk1, &ct2)),
                )
            }
        }
    }
}
