//! Implementation of [\[MP21\]](https://eprint.iacr.org/2020/086.pdf) with
//! [\[LMKCDEY\]](https://eprint.iacr.org/2022/198.pdf) bootstrapping.

use crate::{
    boostrapping::{Boostraping, BoostrapingKey, BoostrapingParam},
    lwe::{Lwe, LweCiphertext, LwePlaintext, LweSecretKey},
    util::{Fq, Poly},
};
use core::iter::repeat;
use rand::RngCore;

#[derive(Debug)]
pub struct Fhew;

#[derive(Debug)]
pub struct FhewBit(LweCiphertext);

#[derive(Clone, Copy, Debug)]
enum FhewBitOps {
    And,
    Nand,
    Or,
    Nor,
    Xor,
    Xnor,
    Majority,
}

impl Fhew {
    fn encode(param: &BoostrapingParam, m: bool) -> LwePlaintext {
        Lwe::encode(param.lwe_z(), &Fq::from_i8(param.p(), m as i8))
    }

    fn decode(param: &BoostrapingParam, pt: &LwePlaintext) -> bool {
        let m = u64::from(Lwe::decode(param.lwe_z(), pt));
        assert!(m == 0 || m == 1);
        m == 1
    }

    pub fn sk_encrypt(
        param: &BoostrapingParam,
        sk: &LweSecretKey,
        m: bool,
        rng: &mut impl RngCore,
    ) -> FhewBit {
        let pt = Fhew::encode(param, m);
        let ct = Lwe::sk_encrypt(param.lwe_z(), sk, &pt, rng);
        FhewBit(ct)
    }

    pub fn decrypt(param: &BoostrapingParam, sk: &LweSecretKey, ct: &FhewBit) -> bool {
        let pt = Lwe::decrypt(param.lwe_z(), sk, &ct.0);
        Fhew::decode(param, &pt)
    }

    // Table 1 in 2020/086.
    fn f(param: &BoostrapingParam, op: FhewBitOps) -> Poly<Fq> {
        let map = [-param.big_q_by_8(), param.big_q_by_8()];
        let truth_table = match op {
            FhewBitOps::And => [0, 0, 0, 1],
            FhewBitOps::Nand => [1, 1, 1, 0],
            FhewBitOps::Or => [0, 1, 1, 1],
            FhewBitOps::Nor => [1, 0, 0, 0],
            FhewBitOps::Xor => [0, 1, 1, 1],
            FhewBitOps::Xnor => [1, 0, 0, 0],
            FhewBitOps::Majority => [0, 0, 0, 1],
        };
        truth_table
            .into_iter()
            .flat_map(|out| repeat(map[out]).take(param.q_by_8()))
            .collect()
    }

    fn op(
        param: &BoostrapingParam,
        bk: &BoostrapingKey,
        op: FhewBitOps,
        ct: LweCiphertext,
    ) -> FhewBit {
        let f = Fhew::f(param, op);
        let ct = Boostraping::boostrap(param, bk, &f, &ct);
        let big_q_by_8 = Lwe::trivial_encrypt(param.lwe_z(), &LwePlaintext(param.big_q_by_8()));
        FhewBit(Lwe::eval_add(param.lwe_z(), &ct, &big_q_by_8))
    }

    pub fn and(
        param: &BoostrapingParam,
        bk: &BoostrapingKey,
        FhewBit(ct0): &FhewBit,
        FhewBit(ct1): &FhewBit,
    ) -> FhewBit {
        let ct = Lwe::eval_add(param.lwe_z(), ct0, ct1);
        Self::op(param, bk, FhewBitOps::And, ct)
    }

    pub fn nand(
        param: &BoostrapingParam,
        bk: &BoostrapingKey,
        FhewBit(ct0): &FhewBit,
        FhewBit(ct1): &FhewBit,
    ) -> FhewBit {
        let ct = Lwe::eval_add(param.lwe_z(), ct0, ct1);
        Self::op(param, bk, FhewBitOps::Nand, ct)
    }

    pub fn or(
        param: &BoostrapingParam,
        bk: &BoostrapingKey,
        FhewBit(ct0): &FhewBit,
        FhewBit(ct1): &FhewBit,
    ) -> FhewBit {
        let ct = Lwe::eval_add(param.lwe_z(), ct0, ct1);
        Self::op(param, bk, FhewBitOps::Or, ct)
    }

    pub fn nor(
        param: &BoostrapingParam,
        bk: &BoostrapingKey,
        FhewBit(ct0): &FhewBit,
        FhewBit(ct1): &FhewBit,
    ) -> FhewBit {
        let ct = Lwe::eval_add(param.lwe_z(), ct0, ct1);
        Self::op(param, bk, FhewBitOps::Nor, ct)
    }

    pub fn xor(
        param: &BoostrapingParam,
        bk: &BoostrapingKey,
        FhewBit(ct0): &FhewBit,
        FhewBit(ct1): &FhewBit,
    ) -> FhewBit {
        let ct = Lwe::eval_double(param.lwe_z(), &Lwe::eval_sub(param.lwe_z(), ct0, ct1));
        Self::op(param, bk, FhewBitOps::Xor, ct)
    }

    pub fn xnor(
        param: &BoostrapingParam,
        bk: &BoostrapingKey,
        FhewBit(ct0): &FhewBit,
        FhewBit(ct1): &FhewBit,
    ) -> FhewBit {
        let ct = Lwe::eval_double(param.lwe_z(), &Lwe::eval_sub(param.lwe_z(), ct0, ct1));
        Self::op(param, bk, FhewBitOps::Xnor, ct)
    }

    pub fn majority(
        param: &BoostrapingParam,
        bk: &BoostrapingKey,
        FhewBit(ct0): &FhewBit,
        FhewBit(ct1): &FhewBit,
        FhewBit(ct2): &FhewBit,
    ) -> FhewBit {
        let ct = Lwe::eval_add(param.lwe_z(), &Lwe::eval_add(param.lwe_z(), ct0, ct1), ct2);
        Self::op(param, bk, FhewBitOps::Majority, ct)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        boostrapping::{Boostraping, BoostrapingParam},
        fhew::Fhew,
        lwe::{Lwe, LweParam},
        rgsw::RgswParam,
        rlwe::RlweParam,
        util::two_adic_primes,
    };
    use core::array::from_fn;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn fhew_bitwise_ops() {
        let mut rng = StdRng::from_entropy();
        let param = {
            let p = 4;
            let rgsw = {
                let (log_q, log_n, log_b, d) = (28, 10, 7, 4);
                let q = two_adic_primes(log_q, log_n + 1).next().unwrap();
                let rlwe = RlweParam::new(q, p, log_n).with_decomposor(log_b, d);
                RgswParam::new(rlwe, log_b, d)
            };
            let lwe = {
                let (n, q, log_b, d) = (458, 1 << 14, 3, 5);
                LweParam::new(q, p, n).with_decomposor(log_b, d)
            };
            let w = 10;
            BoostrapingParam::new(rgsw, lwe, w)
        };
        let z = Lwe::key_gen(param.lwe_z(), &mut rng);
        let bk = {
            let s = Lwe::key_gen(param.lwe_s(), &mut rng);
            Boostraping::key_gen(&param, &z, &s, &mut rng)
        };

        macro_rules! assert_correct_op {
            ($op:ident($ct0:ident, $ct1:ident $(, $ct2:ident)?) == $output:expr) => {
                assert_eq!(
                    $output,
                    Fhew::decrypt(&param, &z, &Fhew::$op(&param, &bk, &$ct0, &$ct1 $(, &$ct2)?))
                )
            };
        }

        for m in 0..1 << 2 {
            let [m0, m1] = from_fn(|i| (m >> i) & 1 == 1);
            let [ct0, ct1] = [m0, m1].map(|m| Fhew::sk_encrypt(&param, &z, m, &mut rng));
            assert_correct_op!(and(ct0, ct1) == m0 & m1);
            assert_correct_op!(nand(ct0, ct1) == !(m0 & m1));
            assert_correct_op!(or(ct0, ct1) == m0 | m1);
            assert_correct_op!(nor(ct0, ct1) == !(m0 | m1));
            assert_correct_op!(xor(ct0, ct1) == m0 ^ m1);
            assert_correct_op!(xnor(ct0, ct1) == !(m0 ^ m1));
        }
        for m in 0..1 << 3 {
            let [m0, m1, m2] = from_fn(|i| (m >> i) & 1 == 1);
            let [ct0, ct1, ct2] = [m0, m1, m2].map(|m| Fhew::sk_encrypt(&param, &z, m, &mut rng));
            assert_correct_op!(majority(ct0, ct1, ct2) == (m0 & m1) | (m1 & m2) | (m2 & m0));
        }
    }
}
