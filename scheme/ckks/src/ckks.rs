use crate::sfft::{sfft, sifft};
use derive_more::{Add, Deref, Sub};
use itertools::{chain, izip, Itertools};
use rand::RngCore;
use util::{
    dg, two_adic_primes, zo, AVec, BigFloat, BigInt, BigUint, Complex, NegaCyclicPoly, RnsRq, Zq,
};

#[derive(Clone, Debug)]
pub struct Ckks;

#[derive(Clone, Debug)]
pub struct CkksParam {
    log_n: usize,
    qs: Vec<u64>,
    ps: Vec<u64>,
    scale: BigFloat,
}

impl CkksParam {
    pub fn new(log_n: usize, log_qi: usize, big_l: usize) -> Self {
        assert!(log_n >= 1);
        assert!(big_l > 1);

        let mut primes = two_adic_primes(log_qi, log_n + 1);
        let qs = primes.by_ref().take(big_l).collect_vec();
        let ps = primes.by_ref().take(big_l).collect_vec();
        let scale = BigFloat::from(*qs.last().unwrap());

        Self {
            log_n,
            qs,
            ps,
            scale,
        }
    }

    pub fn m(&self) -> usize {
        1 << (self.log_n + 1)
    }

    pub fn n(&self) -> usize {
        1 << self.log_n
    }

    pub fn l(&self) -> usize {
        1 << (self.log_n - 1)
    }

    pub fn pow5(&self, j: usize) -> usize {
        Zq::from_usize(2 * self.n() as u64, 5).pow(j).into()
    }

    pub fn qs(&self) -> &[u64] {
        &self.qs
    }

    pub fn ps(&self) -> &[u64] {
        &self.ps
    }

    pub fn qps(&self) -> Vec<u64> {
        chain![self.qs(), self.ps()].copied().collect()
    }

    pub fn p(&self) -> BigUint {
        self.ps.iter().product()
    }

    pub fn scale(&self) -> &BigFloat {
        &self.scale
    }
}

#[derive(Clone, Debug)]
pub struct CkksSecretKey(AVec<i64>);

impl CkksSecretKey {
    fn square(&self) -> Self {
        Self(NegaCyclicPoly::from(self.0.clone()).square().into())
    }

    fn automorphism(&self, t: i64) -> Self {
        CkksSecretKey(self.0.automorphism(t))
    }
}

#[derive(Clone, Debug, Deref)]
pub struct CkksPublicKey(CkksCiphertext);

#[derive(Clone, Debug, Deref)]
pub struct CkksKeySwitchingKey(CkksCiphertext);

#[derive(Clone, Debug, Deref)]
pub struct CkksRelinKey(CkksKeySwitchingKey);

#[derive(Clone, Debug, Deref)]
pub struct CkksConjKey(CkksKeySwitchingKey);

#[derive(Clone, Debug, Deref)]
pub struct CkksRotKey(usize, #[deref] CkksKeySwitchingKey);

impl CkksRotKey {
    pub fn j(&self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug)]
pub struct CkksPlaintext(RnsRq);

#[derive(Clone, Debug, Add, Sub)]
pub struct CkksCiphertext(RnsRq, RnsRq);

impl CkksCiphertext {
    pub fn a(&self) -> &RnsRq {
        &self.1
    }

    pub fn b(&self) -> &RnsRq {
        &self.0
    }

    fn rescale(self) -> CkksCiphertext {
        CkksCiphertext(self.0.rescale(), self.1.rescale())
    }

    fn automorphism(self, t: i64) -> CkksCiphertext {
        CkksCiphertext(self.0.automorphism(t), self.1.automorphism(t))
    }
}

impl Ckks {
    pub fn sk_gen(param: &CkksParam, rng: &mut impl RngCore) -> CkksSecretKey {
        CkksSecretKey(AVec::sample(param.n(), zo(0.5), rng))
    }

    pub fn pk_gen(param: &CkksParam, sk: &CkksSecretKey, rng: &mut impl RngCore) -> CkksPublicKey {
        let zero = CkksPlaintext(RnsRq::zero(param.qs(), param.n()));
        CkksPublicKey(Ckks::sk_encrypt(param, sk, zero, rng))
    }

    pub fn key_gen(param: &CkksParam, rng: &mut impl RngCore) -> (CkksSecretKey, CkksPublicKey) {
        let sk = Ckks::sk_gen(param, rng);
        let pk = Ckks::pk_gen(param, &sk, rng);
        (sk, pk)
    }

    pub fn ksk_gen(
        param: &CkksParam,
        sk: &CkksSecretKey,
        CkksSecretKey(sk_prime): CkksSecretKey,
        rng: &mut impl RngCore,
    ) -> CkksKeySwitchingKey {
        let pt = RnsRq::from_i64(param.qps(), &sk_prime) * param.p();
        CkksKeySwitchingKey(Ckks::sk_encrypt(param, sk, CkksPlaintext(pt), rng))
    }

    pub fn rlk_gen(param: &CkksParam, sk: &CkksSecretKey, rng: &mut impl RngCore) -> CkksRelinKey {
        let sk_square = sk.square();
        CkksRelinKey(Ckks::ksk_gen(param, sk, sk_square, rng))
    }

    pub fn cjk_gen(param: &CkksParam, sk: &CkksSecretKey, rng: &mut impl RngCore) -> CkksConjKey {
        let sk_conj = sk.automorphism(-1);
        CkksConjKey(Ckks::ksk_gen(param, sk, sk_conj, rng))
    }

    pub fn rtk_gen(
        param: &CkksParam,
        sk: &CkksSecretKey,
        j: i64,
        rng: &mut impl RngCore,
    ) -> CkksRotKey {
        let j = j.rem_euclid(param.l() as _) as _;
        let sk_rot = sk.automorphism(param.pow5(j) as _);
        CkksRotKey(j, Ckks::ksk_gen(param, sk, sk_rot, rng))
    }

    pub fn encode(param: &CkksParam, m: AVec<Complex>) -> CkksPlaintext {
        assert_eq!(m.len(), param.l());

        let z = sifft(m);

        let z_scaled = chain![z.iter().map(|z| &z.re), z.iter().map(|z| &z.im)]
            .map(|z| BigInt::from(z * param.scale()))
            .collect_vec();

        let pt = RnsRq::from_bigint(param.qs(), &z_scaled);

        CkksPlaintext(pt)
    }

    pub fn decode(param: &CkksParam, CkksPlaintext(pt): CkksPlaintext) -> AVec<Complex> {
        assert_eq!(pt.n(), param.n());

        let z_scaled = pt.into_bigint();

        let z = izip!(&z_scaled[..param.l()], &z_scaled[param.l()..])
            .map(|(re, im)| {
                let [re, im] = [re, im].map(|z| BigFloat::from(z) / param.scale());
                Complex::new(re, im)
            })
            .collect();

        sfft(z)
    }

    pub fn sk_encrypt(
        _: &CkksParam,
        CkksSecretKey(sk): &CkksSecretKey,
        CkksPlaintext(pt): CkksPlaintext,
        rng: &mut impl RngCore,
    ) -> CkksCiphertext {
        let a = RnsRq::sample_uniform(pt.qs(), pt.n(), rng) as RnsRq;
        let e = RnsRq::sample_i64(pt.qs(), pt.n(), dg(3.2, 6), rng);
        let b = -(&a * sk) + e + pt;
        CkksCiphertext(b, a)
    }

    pub fn pk_encrypt(
        param: &CkksParam,
        pk: &CkksPublicKey,
        CkksPlaintext(pt): CkksPlaintext,
        rng: &mut impl RngCore,
    ) -> CkksCiphertext {
        let u = &RnsRq::sample_i64(param.qs(), param.n(), zo(0.5), rng);
        let e0 = RnsRq::sample_i64(param.qs(), param.n(), dg(3.2, 6), rng);
        let e1 = RnsRq::sample_i64(param.qs(), param.n(), dg(3.2, 6), rng);
        let a = pk.a() * u + e0;
        let b = pk.b() * u + e1 + pt;
        CkksCiphertext(b, a)
    }

    pub fn decrypt(
        _: &CkksParam,
        CkksSecretKey(sk): &CkksSecretKey,
        ct: CkksCiphertext,
    ) -> CkksPlaintext {
        let pt = ct.b() + ct.a() * sk;
        CkksPlaintext(pt)
    }

    pub fn mul_constant(param: &CkksParam, m: AVec<Complex>, ct: CkksCiphertext) -> CkksCiphertext {
        let CkksPlaintext(pt) = &Ckks::encode(param, m);
        CkksCiphertext(pt * ct.b(), pt * ct.a()).rescale()
    }

    pub fn mul(
        param: &CkksParam,
        rlk: &CkksRelinKey,
        ct0: CkksCiphertext,
        ct1: CkksCiphertext,
    ) -> CkksCiphertext {
        let [d0, d1, d2] = [
            ct0.b() * ct1.b(),
            ct0.b() * ct1.a() + ct0.a() * ct1.b(),
            ct0.a() * ct1.a(),
        ];
        (CkksCiphertext(d0, d1) + Ckks::relinearize(param, rlk, d2)).rescale()
    }

    fn relinearize(param: &CkksParam, rlk: &CkksRelinKey, d2: RnsRq) -> CkksCiphertext {
        let ct_quad = CkksCiphertext(RnsRq::zero(d2.qs(), d2.n()), d2);
        Ckks::key_switch(param, rlk, ct_quad)
    }

    pub fn conjugate(param: &CkksParam, cjk: &CkksConjKey, ct: CkksCiphertext) -> CkksCiphertext {
        let ct_conj = ct.automorphism(-1);
        Ckks::key_switch(param, cjk, ct_conj)
    }

    pub fn rotate(param: &CkksParam, rtk: &CkksRotKey, ct: CkksCiphertext) -> CkksCiphertext {
        let ct_rot = ct.automorphism(param.pow5(rtk.j()) as _);
        Ckks::key_switch(param, rtk, ct_rot)
    }

    pub fn key_switch(
        param: &CkksParam,
        ksk: &CkksKeySwitchingKey,
        CkksCiphertext(ct_b, ct_a): CkksCiphertext,
    ) -> CkksCiphertext {
        let ct_a = &ct_a.extend_bases(param.ps());
        let b = (ksk.b() * ct_a).rescale_k(param.ps().len()) + ct_b;
        let a = (ksk.a() * ct_a).rescale_k(param.ps().len());
        CkksCiphertext(b, a)
    }
}

#[cfg(test)]
mod test {
    use crate::ckks::{Ckks, CkksParam};
    use core::array::from_fn;
    use rand::{distributions::Standard, rngs::StdRng, Rng, SeedableRng};
    use util::{assert_eq_complex, izip_eq, vec_with, AVec, Complex, HadamardMul};

    #[test]
    fn encrypt_decrypt() {
        let rng = &mut StdRng::from_entropy();
        let (log_qi, big_l) = (55, 8);
        for log_n in 1..10 {
            let param = CkksParam::new(log_n, log_qi, big_l);
            let (sk, pk) = Ckks::key_gen(&param, rng);
            let m = &AVec::sample(param.l(), Standard, rng);
            let pt = Ckks::encode(&param, m.clone());
            let ct0 = Ckks::sk_encrypt(&param, &sk, pt.clone(), rng);
            let ct1 = Ckks::pk_encrypt(&param, &pk, pt.clone(), rng);
            izip_eq!(m, Ckks::decode(&param, Ckks::decrypt(&param, &sk, ct0)))
                .for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs, 40));
            izip_eq!(m, Ckks::decode(&param, Ckks::decrypt(&param, &sk, ct1)))
                .for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs, 40));
        }
    }

    #[test]
    fn add_sub() {
        let rng = &mut StdRng::from_entropy();
        let (log_qi, big_l) = (55, 8);
        for log_n in 1..10 {
            let param = CkksParam::new(log_n, log_qi, big_l);
            let (sk, pk) = Ckks::key_gen(&param, rng);
            let [m0, m1] = &from_fn(|_| AVec::<Complex>::sample(param.l(), Standard, rng));
            let [pt0, pt1] = [m0, m1].map(|m| Ckks::encode(&param, m.clone()));
            let [ct0, ct1] = [pt0, pt1].map(|pt| Ckks::pk_encrypt(&param, &pk, pt, rng));
            let (m2, ct2) = (m0 + m1, ct0.clone() + ct1.clone());
            let (m3, ct3) = (m0 - m1, ct0.clone() - ct1.clone());
            izip_eq!(m2, Ckks::decode(&param, Ckks::decrypt(&param, &sk, ct2)))
                .for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs, 40));
            izip_eq!(m3, Ckks::decode(&param, Ckks::decrypt(&param, &sk, ct3)))
                .for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs, 40));
        }
    }

    #[test]
    fn mul_constant() {
        let rng = &mut StdRng::from_entropy();
        let (log_qi, big_l) = (55, 8);
        for log_n in 1..10 {
            let param = CkksParam::new(log_n, log_qi, big_l);
            let (sk, pk) = Ckks::key_gen(&param, rng);
            let mul_m = |a, b| HadamardMul::hada_mul(&a, &b);
            let mul_ct = |ct, m| Ckks::mul_constant(&param, m, ct);
            let ms = vec_with![|| AVec::<Complex>::sample(param.l(), Standard, rng); big_l - 1];
            let pt = Ckks::encode(&param, ms[0].clone());
            let ct = Ckks::pk_encrypt(&param, &pk, pt, rng);
            let m = ms.clone().into_iter().reduce(mul_m).unwrap();
            let ct = ms.into_iter().skip(1).fold(ct, mul_ct);
            izip_eq!(m, Ckks::decode(&param, Ckks::decrypt(&param, &sk, ct)))
                .for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs, 32));
        }
    }

    #[test]
    fn mul() {
        let rng = &mut StdRng::from_entropy();
        let (log_qi, big_l) = (55, 8);
        for log_n in 1..10 {
            let param = CkksParam::new(log_n, log_qi, big_l);
            let (sk, pk) = Ckks::key_gen(&param, rng);
            let rlk = Ckks::rlk_gen(&param, &sk, rng);
            let mul_m = |a, b| HadamardMul::hada_mul(&a, &b);
            let mul_ct = |a, b| Ckks::mul(&param, &rlk, a, b);
            let ms = vec_with![|| AVec::<Complex>::sample(param.l(), Standard, rng); big_l - 1];
            let pts = vec_with![|m| Ckks::encode(&param, m.clone()); &ms];
            let cts = vec_with![|pt| Ckks::pk_encrypt(&param, &pk, pt, rng); pts];
            let m = ms.into_iter().reduce(mul_m).unwrap();
            let ct = cts.into_iter().reduce(mul_ct).unwrap();
            izip_eq!(m, Ckks::decode(&param, Ckks::decrypt(&param, &sk, ct)))
                .for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs, 32));
        }
    }

    #[test]
    fn rotate() {
        let rng = &mut StdRng::from_entropy();
        let (log_qi, big_l) = (55, 8);
        for log_n in 1..10 {
            let param = CkksParam::new(log_n, log_qi, big_l);
            let (sk, pk) = Ckks::key_gen(&param, rng);
            let m0 = AVec::sample(param.l(), Standard, rng);
            let pt0 = Ckks::encode(&param, m0.clone());
            let ct0 = Ckks::pk_encrypt(&param, &pk, pt0, rng);
            for _ in 0..10 {
                let rtk = Ckks::rtk_gen(&param, &sk, rng.gen(), rng);
                let m1 = m0.rot_iter(rtk.j() as _);
                let ct1 = Ckks::rotate(&param, &rtk, ct0.clone());
                izip_eq!(m1, Ckks::decode(&param, Ckks::decrypt(&param, &sk, ct1)))
                    .for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs, 40));
            }
        }
    }

    #[test]
    fn conjugate() {
        let rng = &mut StdRng::from_entropy();
        let (log_qi, big_l) = (55, 8);
        for log_n in 1..10 {
            let param = CkksParam::new(log_n, log_qi, big_l);
            let (sk, pk) = Ckks::key_gen(&param, rng);
            let cjk = Ckks::cjk_gen(&param, &sk, rng);
            let m0 = AVec::sample(param.l(), Standard, rng);
            let m1 = m0.conjugate();
            let pt0 = Ckks::encode(&param, m0);
            let ct0 = Ckks::pk_encrypt(&param, &pk, pt0, rng);
            let ct1 = Ckks::conjugate(&param, &cjk, ct0);
            izip_eq!(m1, Ckks::decode(&param, Ckks::decrypt(&param, &sk, ct1)))
                .for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs, 40));
        }
    }
}
