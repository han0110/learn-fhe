//! Implementation of [\[MP21\]](https://eprint.iacr.org/2020/086.pdf) with
//! [\[LMKCDEY\]](https://eprint.iacr.org/2022/198.pdf) bootstrapping.

use crate::{
    bootstrapping::{Bootstrapping, BootstrappingKey, BootstrappingParam},
    lwe::{Lwe, LweCiphertext, LweDecryptionShare, LwePlaintext, LweSecretKey},
    rlwe::{Rlwe, RlwePublicKey},
    util::{Fq, Poly},
};
use core::iter::repeat;
use rand::RngCore;

#[derive(Debug)]
pub struct Fhew;

#[derive(Clone, Debug)]
pub struct FhewBit(LweCiphertext);

impl Fhew {
    fn decode(param: &BootstrappingParam, pt: LwePlaintext) -> bool {
        assert_eq!(param.p(), 4);
        let m = u64::from(Lwe::decode(param.lwe_z(), pt));
        assert!(m == 0 || m == 1);
        m == 1
    }

    pub fn sk_encrypt(
        param: &BootstrappingParam,
        sk: &LweSecretKey,
        m: bool,
        rng: &mut impl RngCore,
    ) -> FhewBit {
        assert_eq!(param.p(), 4);
        let m = Fq::from_bool(param.p(), m);
        let pt = Lwe::encode(param.lwe_z(), m);
        let ct = Lwe::sk_encrypt(param.lwe_z(), sk, pt, rng);
        FhewBit(ct)
    }

    pub fn pk_encrypt(
        param: &BootstrappingParam,
        pk: &RlwePublicKey,
        m: bool,
        rng: &mut impl RngCore,
    ) -> FhewBit {
        assert_eq!(param.p(), 4);
        let m = Poly::constant(param.n(), Fq::from_bool(param.p(), m));
        let pt = Rlwe::encode(param.rgsw(), m);
        let ct = Rlwe::pk_encrypt(param.rgsw(), pk, pt, rng);
        let ct = Rlwe::sample_extract(param.rgsw(), ct, 0);
        FhewBit(ct)
    }

    pub fn decrypt(param: &BootstrappingParam, sk: &LweSecretKey, ct: FhewBit) -> bool {
        let pt = Lwe::decrypt(param.lwe_z(), sk, ct.0);
        Fhew::decode(param, pt)
    }

    pub fn not(
        param: &BootstrappingParam,
        _: &BootstrappingKey,
        FhewBit(LweCiphertext(a, b)): FhewBit,
    ) -> FhewBit {
        FhewBit(LweCiphertext(-a, -b + param.big_q_by_4()))
    }

    fn op(
        param: &BootstrappingParam,
        bk: &BootstrappingKey,
        table: [usize; 4],
        ct: LweCiphertext,
    ) -> FhewBit {
        let map = [-param.big_q_by_8(), param.big_q_by_8()];
        let f = table
            .into_iter()
            .flat_map(|out| repeat(map[out]).take(param.q_by_8()))
            .collect();
        let LweCiphertext(a, b) = Bootstrapping::bootstrap(param, bk, &f, ct);
        FhewBit(LweCiphertext(a, b + param.big_q_by_8()))
    }
}

macro_rules! impl_op {
    (@ $op:ident, $table:expr, |$($ct:ident),+| $lin:expr) => {
        impl Fhew {
            pub fn $op(
                param: &BootstrappingParam,
                bk: &BootstrappingKey,
                $(FhewBit($ct): FhewBit,)+
            ) -> FhewBit {
                Fhew::op(param, bk, $table, $lin)
            }
        }
    };
    ($($op:ident, $table:expr, |$($ct:ident),+| $lin:expr);* $(;)?) => {
        $(impl_op!(@ $op, $table, |$($ct),+| $lin);)*
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

#[derive(Clone, Debug)]
pub struct FhewBitDecryptionShare(LweDecryptionShare);

impl Fhew {
    pub fn share_decrypt(
        param: &BootstrappingParam,
        sk: &LweSecretKey,
        FhewBit(ct): &FhewBit,
        rng: &mut impl RngCore,
    ) -> FhewBitDecryptionShare {
        FhewBitDecryptionShare(Lwe::share_decrypt(param.lwe_z(), sk, ct.a(), rng))
    }

    pub fn decryption_share_merge(
        param: &BootstrappingParam,
        FhewBit(ct): &FhewBit,
        shares: impl IntoIterator<Item = FhewBitDecryptionShare>,
    ) -> bool {
        let shares = shares.into_iter().map(|share| share.0);
        let pt = Lwe::decryption_share_merge(param.lwe_z(), *ct.b(), shares);
        Fhew::decode(param, pt)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        bootstrapping::{Bootstrapping, BootstrappingParam},
        fhew::Fhew,
        lwe::{Lwe, LweParam},
        rgsw::RgswParam,
        rlwe::{Rlwe, RlweParam},
        util::two_adic_primes,
    };
    use core::array::from_fn;
    use rand::{rngs::StdRng, SeedableRng};

    fn single_key_testing_param() -> BootstrappingParam {
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
        BootstrappingParam::new(rgsw, lwe, w)
    }

    #[test]
    fn encrypt_decrypt() {
        let mut rng = StdRng::from_entropy();
        let param = single_key_testing_param();
        let sk = Lwe::sk_gen(param.lwe_z(), &mut rng);
        let pk = Rlwe::pk_gen(param.rgsw(), &(&sk).into(), &mut rng);
        for m in 0..1 << 1 {
            let m = m == 1;
            let ct0 = Fhew::sk_encrypt(&param, &sk, m, &mut rng);
            let ct1 = Fhew::pk_encrypt(&param, &pk, m, &mut rng);
            assert_eq!(m, Fhew::decrypt(&param, &sk, ct0));
            assert_eq!(m, Fhew::decrypt(&param, &sk, ct1));
        }
    }

    #[test]
    fn op() {
        let mut rng = StdRng::from_entropy();
        let param = single_key_testing_param();
        let sk = Lwe::sk_gen(param.lwe_z(), &mut rng);
        let bk = Bootstrapping::key_gen(&param, &sk, &mut rng);

        macro_rules! assert_decrypted_to {
            ($op:ident($($ct:ident),+), $m:expr) => {
                let ct = Fhew::$op(&param, &bk, $($ct.clone()),+);
                assert_eq!(Fhew::decrypt(&param, &sk, ct), $m);
            };
        }

        for m in 0..1 << 1 {
            let m = m == 1;
            let ct = Fhew::sk_encrypt(&param, &sk, m, &mut rng);
            assert_decrypted_to!(not(ct), !m);
        }
        for m in 0..1 << 2 {
            let [m0, m1] = from_fn(|i| (m >> i) & 1 == 1);
            let [ct0, ct1] = [m0, m1].map(|m| Fhew::sk_encrypt(&param, &sk, m, &mut rng));
            assert_decrypted_to!(and(ct0, ct1), m0 & m1);
            assert_decrypted_to!(nand(ct0, ct1), !(m0 & m1));
            assert_decrypted_to!(or(ct0, ct1), m0 | m1);
            assert_decrypted_to!(nor(ct0, ct1), !(m0 | m1));
            assert_decrypted_to!(xor(ct0, ct1), m0 ^ m1);
            assert_decrypted_to!(xnor(ct0, ct1), !(m0 ^ m1));
        }
        for m in 0..1 << 3 {
            let [m0, m1, m2] = from_fn(|i| (m >> i) & 1 == 1);
            let [ct0, ct1, ct2] = [m0, m1, m2].map(|m| Fhew::sk_encrypt(&param, &sk, m, &mut rng));
            assert_decrypted_to!(majority(ct0, ct1, ct2), (m0 & m1) | (m1 & m2) | (m2 & m0));
        }
    }

    fn multi_key_testing_param() -> BootstrappingParam {
        let p = 4;
        let rgsw = {
            let (log_q, log_n, log_b, d) = (60, 11, 6, 10);
            let q = two_adic_primes(log_q, log_n + 1).next().unwrap();
            let rlwe = RlweParam::new(q, p, log_n).with_decomposor(log_b, d);
            RgswParam::new(rlwe, log_b, d)
        };
        let lwe = {
            let (n, q, log_b, d) = (500, 1 << 20, 4, 5);
            LweParam::new(q, p, n).with_decomposor(log_b, d)
        };
        let w = 10;
        BootstrappingParam::new(rgsw, lwe, w)
    }

    #[test]
    fn multi_key_op() {
        const N: usize = 3;

        let mut rng = StdRng::from_entropy();
        let param = multi_key_testing_param();
        let crs = Bootstrapping::crs_gen(&param, &mut rng);
        let sk_shares: [_; N] = from_fn(|_| Lwe::sk_gen(param.lwe_z(), &mut rng));
        let pk = {
            let pk_shares = sk_shares
                .each_ref()
                .map(|sk| Rlwe::pk_share_gen(param.rgsw(), crs.pk(), &sk.into(), &mut rng));
            Rlwe::pk_share_merge(param.rgsw(), crs.pk().clone(), pk_shares)
        };
        let bk = {
            let bk_shares = sk_shares
                .each_ref()
                .map(|sk| Bootstrapping::key_share_gen(&param, &crs, sk, &pk, &mut rng));
            Bootstrapping::key_share_merge(&param, crs, bk_shares)
        };

        macro_rules! assert_decrypted_to {
            ($op:ident($($ct:ident),+), $m:expr) => {
                let ct = Fhew::$op(&param, &bk, $($ct.clone()),+);
                let d_shares = sk_shares
                    .each_ref()
                    .map(|sk| Fhew::share_decrypt(&param, sk, &ct, &mut rng));
                assert_eq!(Fhew::decryption_share_merge(&param, &ct, d_shares), $m);
            };
        }

        for m in 0..1 << 1 {
            let m = m == 1;
            let ct = Fhew::pk_encrypt(&param, &pk, m, &mut rng);
            assert_decrypted_to!(not(ct), !m);
        }
        for m in 0..1 << 2 {
            let [m0, m1] = from_fn(|i| (m >> i) & 1 == 1);
            let [ct0, ct1] = [m0, m1].map(|m| Fhew::pk_encrypt(&param, &pk, m, &mut rng));
            assert_decrypted_to!(and(ct0, ct1), m0 & m1);
            assert_decrypted_to!(nand(ct0, ct1), !(m0 & m1));
            assert_decrypted_to!(or(ct0, ct1), m0 | m1);
            assert_decrypted_to!(nor(ct0, ct1), !(m0 | m1));
            assert_decrypted_to!(xor(ct0, ct1), m0 ^ m1);
            assert_decrypted_to!(xnor(ct0, ct1), !(m0 ^ m1));
        }
        for m in 0..1 << 3 {
            let [m0, m1, m2] = from_fn(|i| (m >> i) & 1 == 1);
            let [ct0, ct1, ct2] = [m0, m1, m2].map(|m| Fhew::pk_encrypt(&param, &pk, m, &mut rng));
            assert_decrypted_to!(majority(ct0, ct1, ct2), (m0 & m1) | (m1 & m2) | (m2 & m0));
        }
    }
}
