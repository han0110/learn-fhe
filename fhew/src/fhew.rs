//! Implementation of [\[MP21\]](https://eprint.iacr.org/2020/086.pdf) with
//! [\[LMKCDEY\]](https://eprint.iacr.org/2022/198.pdf) bootstrapping.

use crate::{
    boostrapping::{Boostraping, BoostrapingKey, BoostrapingParam},
    lwe::{Lwe, LweCiphertext, LwePlaintext, LweSecretKey},
    util::Fq,
};
use core::iter::repeat;
use rand::RngCore;

#[derive(Debug)]
pub struct Fhew;

#[derive(Clone, Debug)]
pub struct FhewBit(LweCiphertext);

impl Fhew {
    fn encode(param: &BoostrapingParam, m: bool) -> LwePlaintext {
        assert_eq!(param.p(), 4);

        Lwe::encode(param.lwe_z(), Fq::from_i8(param.p(), m as i8))
    }

    fn decode(param: &BoostrapingParam, pt: LwePlaintext) -> bool {
        assert_eq!(param.p(), 4);

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
        let ct = Lwe::sk_encrypt(param.lwe_z(), sk, pt, rng);
        FhewBit(ct)
    }

    pub fn decrypt(param: &BoostrapingParam, sk: &LweSecretKey, ct: FhewBit) -> bool {
        let pt = Lwe::decrypt(param.lwe_z(), sk, ct.0);
        Fhew::decode(param, pt)
    }

    pub fn not(
        param: &BoostrapingParam,
        _: &BoostrapingKey,
        FhewBit(LweCiphertext(a, b)): FhewBit,
    ) -> FhewBit {
        FhewBit(LweCiphertext(-a, -b + param.big_q_by_4()))
    }

    fn op(
        param: &BoostrapingParam,
        bk: &BoostrapingKey,
        table: [usize; 4],
        ct: LweCiphertext,
    ) -> FhewBit {
        let map = [-param.big_q_by_8(), param.big_q_by_8()];
        let f = table
            .into_iter()
            .flat_map(|out| repeat(map[out]).take(param.q_by_8()))
            .collect();
        let LweCiphertext(a, b) = Boostraping::boostrap(param, bk, &f, ct);
        FhewBit(LweCiphertext(a, b + param.big_q_by_8()))
    }
}

macro_rules! impl_op {
    (@ $op:ident, $table:expr, |$ct0:ident, $ct1:ident $(, $ct2:ident)?| $lin:expr) => {
        impl Fhew {
            pub fn $op(
                param: &BoostrapingParam,
                bk: &BoostrapingKey,
                FhewBit($ct0): FhewBit,
                FhewBit($ct1): FhewBit,
                $(FhewBit($ct2): FhewBit,)?
            ) -> FhewBit {
                Fhew::op(param, bk, $table, $lin)
            }
        }
    };
    ($($op:ident, $table:expr, |$ct0:ident, $ct1:ident $(, $ct2:ident)?| $lin:expr);* $(;)?) => {
        $(impl_op!(@ $op, $table, |$ct0, $ct1 $(, $ct2)?| $lin);)*
    }
}

// Table 1 in 2020/086.
impl_op!(
         and, [0, 0, 0, 1], |ct0, ct1| ct0 + ct1;
        nand, [1, 1, 1, 0], |ct0, ct1| ct0 + ct1;
          or, [0, 1, 1, 1], |ct0, ct1| ct0 + ct1;
         nor, [1, 0, 0, 0], |ct0, ct1| ct0 + ct1;
         xor, [0, 1, 1, 1], |ct0, ct1| (ct0 - ct1).double();
        xnor, [1, 0, 0, 0], |ct0, ct1| (ct0 - ct1).double();
    majority, [0, 0, 0, 1], |ct0, ct1, ct2| ct0 + ct1 + ct2;
);

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
    fn ops() {
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
                let (n, q, log_b, d) = (458, 1 << 14, 2, 7);
                LweParam::new(q, p, n).with_decomposor(log_b, d)
            };
            let w = 10;
            BoostrapingParam::new(rgsw, lwe, w)
        };
        let sk = Lwe::key_gen(param.lwe_z(), &mut rng);
        let bk = Boostraping::key_gen(&param, &sk, &mut rng);

        macro_rules! assert_correct_op {
            ($op:ident($ct0:ident $(, $ct1:ident $(, $ct2:ident)?)?) == $output:expr) => {
                assert_eq!(
                    $output,
                    Fhew::decrypt(&param, &sk, Fhew::$op(&param, &bk, $ct0.clone() $(, $ct1.clone() $(, $ct2.clone())?)?))
                )
            };
        }

        for m in 0..1 << 1 {
            let m = m == 1;
            let ct = Fhew::sk_encrypt(&param, &sk, m, &mut rng);
            assert_correct_op!(not(ct) == !m);
        }
        for m in 0..1 << 2 {
            let [m0, m1] = from_fn(|i| (m >> i) & 1 == 1);
            let [ct0, ct1] = [m0, m1].map(|m| Fhew::sk_encrypt(&param, &sk, m, &mut rng));
            assert_correct_op!(and(ct0, ct1) == m0 & m1);
            assert_correct_op!(nand(ct0, ct1) == !(m0 & m1));
            assert_correct_op!(or(ct0, ct1) == m0 | m1);
            assert_correct_op!(nor(ct0, ct1) == !(m0 | m1));
            assert_correct_op!(xor(ct0, ct1) == m0 ^ m1);
            assert_correct_op!(xnor(ct0, ct1) == !(m0 ^ m1));
        }
        for m in 0..1 << 3 {
            let [m0, m1, m2] = from_fn(|i| (m >> i) & 1 == 1);
            let [ct0, ct1, ct2] = [m0, m1, m2].map(|m| Fhew::sk_encrypt(&param, &sk, m, &mut rng));
            assert_correct_op!(majority(ct0, ct1, ct2) == (m0 & m1) | (m1 & m2) | (m2 & m0));
        }
    }
}
