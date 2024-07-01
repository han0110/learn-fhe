use derive_more::{Add, Deref, Sub};
use itertools::{chain, izip, Itertools};
use rand::RngCore;
use util::{
    bit_reverse, dg, powers, two_adic_primes, zo, AVec, BigFloat, BigInt, BigUint, Complex, CrtRq,
    NegaCyclicPoly, Zq,
};

#[derive(Clone, Debug)]
pub struct Ckks;

#[derive(Clone, Debug)]
pub struct CkksParam {
    log_n: usize,
    pow5: Vec<usize>,
    psi: Vec<Complex>,
    qs: Vec<u64>,
    ps: Vec<u64>,
    scale: BigFloat,
    d_num: usize,
}

impl CkksParam {
    pub fn m(&self) -> usize {
        1 << (self.log_n + 1)
    }

    pub fn n(&self) -> usize {
        1 << self.log_n
    }

    pub fn l(&self) -> usize {
        1 << (self.log_n - 1)
    }

    pub fn psi(&self) -> &[Complex] {
        &self.psi
    }

    pub fn pow5(&self) -> &[usize] {
        &self.pow5
    }

    pub fn big_l(&self) -> usize {
        self.qs.len()
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

    pub fn d_num(&self) -> usize {
        self.d_num
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
pub struct CkksCleartext(AVec<Complex>);

#[derive(Clone, Debug)]
pub struct CkksPlaintext(CrtRq);

#[derive(Clone, Debug, Add, Sub)]
pub struct CkksCiphertext(CrtRq, CrtRq);

impl CkksCiphertext {
    pub fn a(&self) -> &CrtRq {
        &self.1
    }

    pub fn b(&self) -> &CrtRq {
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
    pub fn param_gen(log_n: usize, log_qi: usize, big_l: usize, d_num: usize) -> CkksParam {
        assert!(log_n >= 1);
        assert!(big_l > 1);

        let n = 1 << log_n;

        let psi = {
            let phase = BigFloat::pi() / BigFloat::from(n);
            let cis = Complex::new(phase.cos(), phase.sin());
            powers(&cis).take(2 * n).collect()
        };

        let pow5 = {
            let five = Zq::from_u64(2 * n as u64, 5);
            five.powers().take(n / 2).map_into().collect()
        };

        let mut primes = two_adic_primes(log_qi, log_n + 1);
        let qs = primes.by_ref().take(big_l).collect_vec();
        let ps = primes.by_ref().take(big_l).collect_vec();

        let scale = BigFloat::from(*qs.last().unwrap());

        CkksParam {
            log_n,
            psi,
            pow5,
            qs,
            ps,
            scale,
            d_num,
        }
    }

    pub fn sk_gen(param: &CkksParam, rng: &mut impl RngCore) -> CkksSecretKey {
        CkksSecretKey(AVec::sample(param.n(), zo(0.5), rng))
    }

    pub fn pk_gen(param: &CkksParam, sk: &CkksSecretKey, rng: &mut impl RngCore) -> CkksPublicKey {
        let zero = CkksPlaintext(CrtRq::zero(param.n(), param.qs()));
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
        let pt = CrtRq::from_i64(&sk_prime, &param.qps()) * param.p();
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
        let sk_rot = sk.automorphism(param.pow5()[j] as _);
        CkksRotKey(j, Ckks::ksk_gen(param, sk, sk_rot, rng))
    }

    pub fn encode(param: &CkksParam, CkksCleartext(m): CkksCleartext) -> CkksPlaintext {
        assert_eq!(m.len(), param.l());

        let z = special_ifft(param, m);

        let z_scaled = chain![z.iter().map(|z| &z.re), z.iter().map(|z| &z.im)]
            .map(|z| BigInt::from(z * param.scale()))
            .collect_vec();

        let pt = CrtRq::from_bigint(z_scaled, param.qs());

        CkksPlaintext(pt)
    }

    pub fn decode(param: &CkksParam, CkksPlaintext(pt): CkksPlaintext) -> CkksCleartext {
        assert_eq!(pt.n(), param.n());

        let z_scaled = pt.into_bigint();

        let z = izip!(&z_scaled[..param.l()], &z_scaled[param.l()..])
            .map(|(re, im)| {
                let [re, im] = [re, im].map(|z| BigFloat::from(z) / param.scale());
                Complex::new(re, im)
            })
            .collect();

        let m = special_fft(param, z);

        CkksCleartext(m)
    }

    pub fn sk_encrypt(
        _: &CkksParam,
        CkksSecretKey(sk): &CkksSecretKey,
        CkksPlaintext(pt): CkksPlaintext,
        rng: &mut impl RngCore,
    ) -> CkksCiphertext {
        let a = CrtRq::sample_uniform(pt.n(), &pt.qs(), rng) as CrtRq;
        let e = CrtRq::sample_i64(pt.n(), &pt.qs(), dg(3.2, 6), rng);
        let b = -(&a * sk) + e + pt;
        CkksCiphertext(b, a)
    }

    pub fn pk_encrypt(
        param: &CkksParam,
        pk: &CkksPublicKey,
        CkksPlaintext(pt): CkksPlaintext,
        rng: &mut impl RngCore,
    ) -> CkksCiphertext {
        let u = &CrtRq::sample_i64(param.n(), param.qs(), zo(0.5), rng);
        let e0 = CrtRq::sample_i64(param.n(), param.qs(), dg(3.2, 6), rng);
        let e1 = CrtRq::sample_i64(param.n(), param.qs(), dg(3.2, 6), rng);
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

    fn relinearize(param: &CkksParam, rlk: &CkksRelinKey, d2: CrtRq) -> CkksCiphertext {
        let ct_quad = CkksCiphertext(CrtRq::zero(d2.n(), &d2.qs()), d2);
        Ckks::key_switch(param, rlk, ct_quad)
    }

    pub fn conjugate(param: &CkksParam, cjk: &CkksConjKey, ct: CkksCiphertext) -> CkksCiphertext {
        let ct_conj = ct.automorphism(-1);
        Ckks::key_switch(param, cjk, ct_conj)
    }

    pub fn rotate(param: &CkksParam, rtk: &CkksRotKey, ct: CkksCiphertext) -> CkksCiphertext {
        let ct_rot = ct.automorphism(param.pow5()[rtk.j()] as i64);
        Ckks::key_switch(param, rtk, ct_rot)
    }

    pub fn key_switch(
        param: &CkksParam,
        ksk: &CkksKeySwitchingKey,
        CkksCiphertext(ct_b, ct_a): CkksCiphertext,
    ) -> CkksCiphertext {
        let ct_a = &ct_a.extend_bases(param.ps());
        let b = (ksk.b() * ct_a).rescale_k(param.big_l()) + ct_b;
        let a = (ksk.a() * ct_a).rescale_k(param.big_l());
        CkksCiphertext(b, a)
    }
}

// Algorithm 1 in 2018/1043
fn special_fft(param: &CkksParam, mut w: AVec<Complex>) -> AVec<Complex> {
    assert_eq!(w.len(), param.l());
    let (pow5, psi) = (param.pow5(), param.psi());

    bit_reverse(&mut w);

    let l = w.len();
    let mut m = 2;
    while m <= l {
        for i in (0..l).step_by(m) {
            for j in 0..m / 2 {
                let k = (pow5[j] % (4 * m)) * l / m;
                let u = w[i + j].clone();
                let v = &w[i + j + m / 2] * &psi[k];
                w[i + j] = &u + &v;
                w[i + j + m / 2] = &u - &v;
            }
        }
        m *= 2;
    }

    w
}

fn special_ifft(param: &CkksParam, mut w: AVec<Complex>) -> AVec<Complex> {
    assert_eq!(w.len(), param.l());
    let (pow5, psi) = (param.pow5(), param.psi());

    let l = w.len();
    let mut m = l;
    while m >= 2 {
        for i in (0..l).step_by(m) {
            for j in 0..m / 2 {
                let k = (4 * m - pow5[j] % (4 * m)) * l / m;
                let u = &w[i + j] + &w[i + j + m / 2];
                let v = (&w[i + j] - &w[i + j + m / 2]) * &psi[k];
                w[i + j] = u;
                w[i + j + m / 2] = v;
            }
        }
        m /= 2;
    }

    bit_reverse(&mut w);
    w.iter_mut().for_each(|w| *w /= BigFloat::from(l));

    w
}

#[cfg(test)]
mod test {
    use crate::ckks::{special_fft, special_ifft, Ckks, CkksCleartext};
    use core::array::from_fn;
    use rand::{distributions::Standard, rngs::StdRng, Rng, SeedableRng};
    use util::{assert_eq_complex, horner, izip_eq, vec_with, AVec, Complex};

    #[test]
    fn special_ifft_fft() {
        let rng = &mut StdRng::from_entropy();
        let (log_qi, big_l) = (55, 8);
        for log_n in 1..10 {
            let param = Ckks::param_gen(log_n, log_qi, big_l, 0);
            let evals = AVec::sample(param.l(), Standard, rng);
            let coeffs = special_ifft(&param, evals.clone());
            izip_eq!(param.pow5(), &evals)
                .for_each(|(k, eval)| assert_eq_complex!(horner(&coeffs, &param.psi()[*k]), eval));
            izip_eq!(evals, special_fft(&param, coeffs))
                .for_each(|(a, b)| assert_eq_complex!(a, b));
        }
    }

    #[test]
    fn encrypt_decrypt() {
        let rng = &mut StdRng::from_entropy();
        let (log_qi, big_l) = (55, 8);
        for log_n in 1..10 {
            let param = Ckks::param_gen(log_n, log_qi, big_l, 0);
            let (sk, pk) = Ckks::key_gen(&param, rng);
            let m = &AVec::sample(param.l(), Standard, rng);
            let pt = Ckks::encode(&param, CkksCleartext(m.clone()));
            let ct0 = Ckks::sk_encrypt(&param, &sk, pt.clone(), rng);
            let ct1 = Ckks::pk_encrypt(&param, &pk, pt.clone(), rng);
            izip_eq!(m, Ckks::decode(&param, Ckks::decrypt(&param, &sk, ct0)).0)
                .for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs, 40));
            izip_eq!(m, Ckks::decode(&param, Ckks::decrypt(&param, &sk, ct1)).0)
                .for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs, 40));
        }
    }

    #[test]
    fn add_sub() {
        let rng = &mut StdRng::from_entropy();
        let (log_qi, big_l) = (55, 8);
        for log_n in 1..10 {
            let param = Ckks::param_gen(log_n, log_qi, big_l, 0);
            let (sk, pk) = Ckks::key_gen(&param, rng);
            let [m0, m1] = &from_fn(|_| AVec::<Complex>::sample(param.l(), Standard, rng));
            let [pt0, pt1] = [m0, m1].map(|m| Ckks::encode(&param, CkksCleartext(m.clone())));
            let [ct0, ct1] = [pt0, pt1].map(|pt| Ckks::pk_encrypt(&param, &pk, pt, rng));
            let (m2, ct2) = (m0 + m1, ct0.clone() + ct1.clone());
            let (m3, ct3) = (m0 - m1, ct0.clone() - ct1.clone());
            izip_eq!(m2, Ckks::decode(&param, Ckks::decrypt(&param, &sk, ct2)).0)
                .for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs, 40));
            izip_eq!(m3, Ckks::decode(&param, Ckks::decrypt(&param, &sk, ct3)).0)
                .for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs, 40));
        }
    }

    #[test]
    fn mul() {
        let rng = &mut StdRng::from_entropy();
        let (log_qi, big_l, d_num) = (55, 8, 8);
        for log_n in 1..10 {
            let param = Ckks::param_gen(log_n, log_qi, big_l, d_num);
            let (sk, pk) = Ckks::key_gen(&param, rng);
            let rlk = Ckks::rlk_gen(&param, &sk, rng);
            let mul_m = |m0, m1| AVec::ew_mul(&m0, &m1);
            let mul_ct = |ct0, ct1| Ckks::mul(&param, &rlk, ct0, ct1);
            let ms = vec_with![|| AVec::<Complex>::sample(param.l(), Standard, rng); big_l - 1];
            let pts = vec_with!(|m| Ckks::encode(&param, CkksCleartext(m.clone())); &ms);
            let cts = vec_with!(|pt| Ckks::pk_encrypt(&param, &pk, pt, rng); pts);
            let m = ms.into_iter().reduce(mul_m).unwrap();
            let ct = cts.into_iter().reduce(mul_ct).unwrap();
            izip_eq!(m, Ckks::decode(&param, Ckks::decrypt(&param, &sk, ct)).0)
                .for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs, 32));
        }
    }

    #[test]
    fn rotate() {
        let rng = &mut StdRng::from_entropy();
        let (log_qi, big_l, d_num) = (55, 8, 8);
        for log_n in 1..10 {
            let param = Ckks::param_gen(log_n, log_qi, big_l, d_num);
            let (sk, pk) = Ckks::key_gen(&param, rng);
            let m0 = AVec::sample(param.l(), Standard, rng);
            let pt0 = Ckks::encode(&param, CkksCleartext(m0.clone()));
            let ct0 = Ckks::pk_encrypt(&param, &pk, pt0, rng);
            for _ in 0..10 {
                let rtk = Ckks::rtk_gen(&param, &sk, rng.gen(), rng);
                let m1 = m0.clone().rotate(rtk.j() as _);
                let ct1 = Ckks::rotate(&param, &rtk, ct0.clone());
                izip_eq!(m1, Ckks::decode(&param, Ckks::decrypt(&param, &sk, ct1)).0)
                    .for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs, 40));
            }
        }
    }

    #[test]
    fn conjugate() {
        let rng = &mut StdRng::from_entropy();
        let (log_qi, big_l, d_num) = (55, 8, 8);
        for log_n in 1..10 {
            let param = Ckks::param_gen(log_n, log_qi, big_l, d_num);
            let (sk, pk) = Ckks::key_gen(&param, rng);
            let cjk = Ckks::cjk_gen(&param, &sk, rng);
            let m0 = AVec::sample(param.l(), Standard, rng);
            let m1 = m0.conjugate();
            let pt0 = Ckks::encode(&param, CkksCleartext(m0));
            let ct0 = Ckks::pk_encrypt(&param, &pk, pt0, rng);
            let ct1 = Ckks::conjugate(&param, &cjk, ct0);
            izip_eq!(m1, Ckks::decode(&param, Ckks::decrypt(&param, &sk, ct1)).0)
                .for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs, 40));
        }
    }
}
