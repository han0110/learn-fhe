use crate::{
    tggsw::{Tggsw, TggswCiphertext, TggswParam, TggswPlaintext},
    tglwe::{Tglwe, TglweCiphertext, TglweParam},
    tlwe::{Tlwe, TlweCiphertext, TlweKeySwitchingKey, TlweParam, TlweSecretKey},
};
use derive_more::Deref;
use itertools::Itertools;
use rand::RngCore;
use util::{izip_eq, AVec, Base2Decomposable, Rq, Rt};

#[derive(Debug)]
pub struct Bootstrapping;

#[derive(Clone, Copy, Debug, Deref)]
pub struct BootstrappingParam {
    #[deref]
    tlwe: TlweParam,
    tggsw: TggswParam,
}

impl BootstrappingParam {
    pub fn new(tlwe: TlweParam, tggsw: TggswParam) -> Self {
        assert_eq!(tlwe.p(), tggsw.p());
        Self { tlwe, tggsw }
    }

    pub fn tggsw(&self) -> &TggswParam {
        &self.tggsw
    }

    pub fn tglwe(&self) -> &TglweParam {
        self.tggsw()
    }

    pub fn big_n(&self) -> usize {
        self.tggsw().big_n()
    }
}

#[derive(Clone, Debug, Deref)]
pub struct BootstrappingKey {
    #[deref]
    param: BootstrappingParam,
    brk: AVec<TggswCiphertext>,
    ksk: TlweKeySwitchingKey,
}

impl BootstrappingKey {
    fn brk(&self) -> &[TggswCiphertext] {
        &self.brk
    }

    fn ksk(&self) -> &TlweKeySwitchingKey {
        &self.ksk
    }
}

impl Bootstrapping {
    pub fn key_gen(
        param: &BootstrappingParam,
        z: &TlweSecretKey,
        rng: &mut impl RngCore,
    ) -> BootstrappingKey {
        let s = Tglwe::sk_gen(param.tggsw(), rng);
        let brk =
            z.0.iter()
                .map(|&zi| Rt::constant(zi.into(), param.big_n()))
                .map(|pt| Tggsw::sk_encrypt(param.tggsw(), &s, TggswPlaintext(pt), rng))
                .collect();
        let ksk = Tlwe::ksk_gen(param, z, &s, rng);
        BootstrappingKey {
            param: *param,
            brk,
            ksk,
        }
    }

    pub fn bootstrap(bsk: &BootstrappingKey, v: &Rq, ct: TlweCiphertext) -> TlweCiphertext {
        let ct = Bootstrapping::blind_rotate(bsk, bsk.brk(), v, ct);
        let ct = Tglwe::sample_extract(bsk.tggsw(), ct, 0);
        Tlwe::key_switch(bsk, bsk.ksk(), ct)
    }

    fn blind_rotate(
        param: &BootstrappingParam,
        brk: &[TggswCiphertext],
        v: &Rq,
        ct: TlweCiphertext,
    ) -> TglweCiphertext {
        let v = Tglwe::encode(param.tglwe(), v.clone());
        let acc = TglweCiphertext::from((param.tglwe().n(), v));
        let (a, b) = mod_switch(ct, param.big_n());
        izip_eq!(brk, a).fold(acc.rotate(-b), |acc, (brk, a)| {
            Tggsw::cmux(param.tggsw(), brk, acc.clone(), acc.rotate(a))
        })
    }
}

fn mod_switch(ct: TlweCiphertext, big_n: usize) -> (AVec<i64>, i64) {
    let rounding_bits = u64::BITS as usize - (2 * big_n).ilog2() as usize;
    let a = ct.a().rounding_shr(rounding_bits);
    let b = ct.b().rounding_shr(rounding_bits);
    (a.into_iter().map_into().collect(), b.into())
}

#[cfg(test)]
mod test {
    use crate::{
        bootstrapping::{Bootstrapping, BootstrappingParam},
        tggsw::TggswParam,
        tlwe::{Tlwe, TlweParam},
    };
    use core::{convert::identity, iter::repeat};
    use itertools::{chain, Itertools};
    use rand::thread_rng;
    use util::{Rq, Zq};

    fn table(log_p: usize, big_n: usize, f: impl Fn(Zq) -> Zq) -> Rq {
        let p = 1 << log_p;
        let m = big_n >> log_p;
        let table = (0..p).map(|v| f(Zq::from_u64(p, v))).collect_vec();
        chain![
            repeat(table[0]).take(m / 2),
            table[1..].iter().flat_map(|table| repeat(*table).take(m)),
            repeat(-table[0]).take(m / 2),
        ]
        .collect()
    }

    fn double(i: Zq) -> Zq {
        i * 2
    }

    fn parity(i: Zq) -> Zq {
        Zq::from_u64(i.q(), i.to_u64() % 2)
    }

    #[test]
    fn bootstrap() {
        let mut rng = thread_rng();
        let param = {
            let (log_p, padding) = (4, 1);
            let tlwe = {
                let (n, std_dev, log_b, d) = (1024, 1.339775301998614e-7, 4, 5);
                TlweParam::new(log_p, padding, n, std_dev).with_decomposor(log_b, d)
            };
            let tggsw = {
                let (big_n, n, std_dev, log_b, d) = (2048, 1, 2.845267479601915e-15, 23, 1);
                TggswParam::new(log_p, padding, big_n, n, std_dev, log_b, d)
            };
            BootstrappingParam::new(tlwe, tggsw)
        };
        let z = Tlwe::sk_gen(&param, &mut rng);
        let brk = Bootstrapping::key_gen(&param, &z, &mut rng);
        for f in [identity, double, parity] {
            let v = table(param.log_p(), param.big_n(), f);
            for m in 0..param.p() {
                let m = Zq::from_u64(param.p(), m);
                let pt = Tlwe::encode(&param, m);
                let ct0 = Tlwe::sk_encrypt(&param, &z, pt, &mut rng);
                let ct1 = Bootstrapping::bootstrap(&brk, &v, ct0);
                assert_eq!(f(m), Tlwe::decode(&param, Tlwe::decrypt(&param, &z, ct1)),)
            }
        }
    }
}
