//! Implementation of [\[LMKCDEY\]](https://eprint.iacr.org/2022/198.pdf).

use crate::{
    lwe::{Lwe, LweCiphertext, LweKeySwitchingKey, LweParam, LweSecretKey},
    rgsw::{Rgsw, RgswCiphertext, RgswParam, RgswPlaintext},
    rlwe::{Rlwe, RlweAutoKey, RlweCiphertext, RlwePlaintext},
    util::{AVec, Fq, Poly, X},
};
use itertools::{chain, izip};
use rand::RngCore;
use std::collections::HashMap;

#[derive(Debug)]
pub struct Bootstrapping;

#[derive(Clone, Copy, Debug)]
pub struct BootstrappingParam {
    rgsw: RgswParam,
    lwe_z: LweParam,
    lwe_s: LweParam,
    w: usize,
}

impl BootstrappingParam {
    pub fn new(rgsw: RgswParam, lwe_s: LweParam, w: usize) -> Self {
        assert_eq!(rgsw.p(), lwe_s.p());
        let lwe_z = LweParam::new(rgsw.q(), rgsw.p(), rgsw.n());
        Self {
            rgsw,
            lwe_z,
            lwe_s,
            w,
        }
    }

    pub fn rgsw(&self) -> &RgswParam {
        &self.rgsw
    }

    pub fn lwe_s(&self) -> &LweParam {
        &self.lwe_s
    }

    pub fn lwe_z(&self) -> &LweParam {
        &self.lwe_z
    }

    pub fn p(&self) -> u64 {
        self.lwe_z.p()
    }

    pub fn n(&self) -> usize {
        self.rgsw().n()
    }

    pub fn big_q(&self) -> u64 {
        self.rgsw().q()
    }

    pub fn big_q_by_8(&self) -> Fq {
        Fq::from_f64(self.big_q(), self.big_q() as f64 / 8.0)
    }

    pub fn big_q_by_4(&self) -> Fq {
        Fq::from_f64(self.big_q(), self.big_q() as f64 / 4.0)
    }

    pub fn big_q_ks(&self) -> u64 {
        self.lwe_s().q()
    }

    pub fn q(&self) -> u64 {
        2 * self.n() as u64
    }

    pub fn q_by_8(&self) -> usize {
        self.q() as usize / 8
    }

    pub fn w(&self) -> usize {
        self.w
    }
}

#[derive(Clone, Debug)]
pub struct BootstrappingKey {
    ksk: LweKeySwitchingKey,
    brk: Vec<RgswCiphertext>,
    ak: AVec<RlweAutoKey>,
}

impl Bootstrapping {
    pub fn key_gen(
        param: &BootstrappingParam,
        z: &LweSecretKey,
        rng: &mut impl RngCore,
    ) -> BootstrappingKey {
        let s = Lwe::key_gen(param.lwe_s(), rng);
        let ksk = Lwe::ksk_gen(param.lwe_s(), &s, z, rng);
        let brk = {
            let one = &Poly::one(param.n(), param.big_q());
            s.0.iter()
                .map(|sj| one * (X ^ sj))
                .map(|pt| Rgsw::sk_encrypt(param.rgsw(), &z.into(), RgswPlaintext(pt), rng))
                .collect()
        };
        let ak = {
            let g = Fq::from_i64(2 * param.n() as u64, Rlwe::AUTO_G);
            chain![[-g], g.powers().skip(1).take(param.w())]
                .map(|t| Rlwe::ak_gen(param.rgsw(), &z.into(), t.into(), rng))
                .collect()
        };
        BootstrappingKey { ksk, brk, ak }
    }

    // Figure 2 in 2022/198.
    pub fn bootstrap(
        param: &BootstrappingParam,
        bk: &BootstrappingKey,
        f: &Poly<Fq>,
        ct: LweCiphertext,
    ) -> LweCiphertext {
        let ct = ct.mod_switch(param.big_q_ks());
        let ct = Lwe::key_switch(param.lwe_s(), &bk.ksk, ct);
        let ct = ct.mod_switch_odd(param.q());
        let ct = Bootstrapping::blind_rotate(param, &bk.brk, &bk.ak, f, ct);
        Rlwe::sample_extract(param.rgsw(), ct, 0)
    }

    // Algorithm 7 in 2022/198.
    fn blind_rotate(
        param: &BootstrappingParam,
        brk: &[RgswCiphertext],
        ak: &[RlweAutoKey],
        f: &Poly<Fq>,
        LweCiphertext(a, b): LweCiphertext,
    ) -> RlweCiphertext {
        let g = Rlwe::AUTO_G;
        let f_prime = f.automorphism(-g) * (X ^ (b * g));
        let acc = RlwePlaintext(f_prime).into();
        Bootstrapping::blind_rotate_core(param, brk, ak, a, acc)
    }

    // Algorithm 3 in 2022/198.
    fn blind_rotate_core(
        param: &BootstrappingParam,
        brk: &[RgswCiphertext],
        ak: &[RlweAutoKey],
        a: AVec<Fq>,
        mut acc: RlweCiphertext,
    ) -> RlweCiphertext {
        let (i_minus, i_plus) = i_minus_i_plus(param.n(), &a);
        let mut v = 0;
        for l in (1..i_minus.len()).rev() {
            for j in &i_minus[l] {
                acc = Rgsw::external_product(param.rgsw(), &brk[*j], acc);
            }
            v += 1;
            if !i_minus[l - 1].is_empty() || v == param.w || l == 1 {
                acc = Rlwe::automorphism(param.rgsw(), &ak[v], acc);
                v = 0
            }
        }
        for j in &i_minus[0] {
            acc = Rgsw::external_product(param.rgsw(), &brk[*j], acc);
        }
        acc = Rlwe::automorphism(param.rgsw(), &ak[0], acc);
        for l in (1..i_plus.len()).rev() {
            for j in &i_plus[l] {
                acc = Rgsw::external_product(param.rgsw(), &brk[*j], acc);
            }
            v += 1;
            if !i_plus[l - 1].is_empty() || v == param.w || l == 1 {
                acc = Rlwe::automorphism(param.rgsw(), &ak[v], acc);
                v = 0
            }
        }
        for j in &i_plus[0] {
            acc = Rgsw::external_product(param.rgsw(), &brk[*j], acc);
        }
        acc
    }
}

fn i_minus_i_plus(n: usize, a: &AVec<Fq>) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let (log_map_g_minus, log_map_g_plus) = (log_g_map(n, -1), log_g_map(n, 1));
    izip!(0.., a).fold(
        (vec![vec![]; n / 2], vec![vec![]; n / 2]),
        |(mut i_minus, mut i_plus), (i, ai)| {
            match (log_map_g_minus.get(ai), log_map_g_plus.get(ai)) {
                (Some(l), None) => i_minus[*l].push(i),
                (None, Some(l)) => i_plus[*l].push(i),
                _ if u64::from(ai) == 0 => {}
                _ => unreachable!(),
            }
            (i_minus, i_plus)
        },
    )
}

fn log_g_map(n: usize, sign: i64) -> HashMap<Fq, usize> {
    let g = Fq::from_i64(2 * n as u64, Rlwe::AUTO_G);
    izip!(g.powers().map(|g| g * sign), 0..n / 2).collect()
}
