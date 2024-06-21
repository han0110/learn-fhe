use crate::{
    bootstrapping::{BootstrappingKey, BootstrappingParam},
    fhew::Fhew,
    lwe::{Lwe, LweCiphertext, LweDecryptionShare, LweSecretKey},
    rlwe::{Rlwe, RlwePublicKey},
    util::{Fq, Poly},
};
use core::{borrow::Borrow, ops::Not};
use rand::RngCore;

#[derive(Clone, Debug)]
pub struct FhewBool<T> {
    ct: LweCiphertext,
    bk: T,
}

impl<T: Borrow<BootstrappingParam>> FhewBool<T> {
    pub fn sk_encrypt(bk: T, sk: &LweSecretKey, m: bool, rng: &mut impl RngCore) -> Self {
        let param = bk.borrow();
        assert_eq!(param.p(), 4);
        let m = Fq::from_bool(param.p(), m);
        let pt = Lwe::encode(param.lwe_z(), m);
        let ct = Lwe::sk_encrypt(param.lwe_z(), sk, pt, rng);
        Self { ct, bk }
    }

    pub fn pk_encrypt(bk: T, pk: &RlwePublicKey, m: bool, rng: &mut impl RngCore) -> Self {
        let param = bk.borrow();
        assert_eq!(param.p(), 4);
        let m = Poly::constant(param.n(), Fq::from_bool(param.p(), m));
        let pt = Rlwe::encode(param.rgsw(), m);
        let ct = Rlwe::pk_encrypt(param.rgsw(), pk, pt, rng);
        let ct = Rlwe::sample_extract(param.rgsw(), ct, 0);
        Self { ct, bk }
    }

    pub fn decrypt(self, sk: &LweSecretKey) -> bool {
        let param = self.bk.borrow();
        let pt = Lwe::decrypt(param.lwe_z(), sk, self.ct);
        Fhew::decode(param, pt)
    }
}

macro_rules! impl_op {
    ($(fn $op:ident(&self $(,)? $($other:ident),*));* $(;)?) => {
        $(
            paste::paste! {
                impl<T: Borrow<BootstrappingKey> + Copy> FhewBool<T> {
                    pub fn [<bit $op _assign>](&mut self, $($other: &Self),*) {
                        self.ct = Fhew::$op(self.bk.borrow(), self.ct.clone(), $($other.ct.clone()),*);
                    }

                    pub fn [<bit $op>](&self, $($other: &Self),*) -> Self {
                        let mut lhs = self.clone();
                        lhs.[<bit $op _assign>]($($other),*);
                        lhs
                    }
                }
            }
        )*
    };
}

impl_op!(
    fn not(&self);
    fn and(&self, other);
    fn nand(&self, other);
    fn or(&self, other);
    fn nor(&self, other);
    fn xor(&self, other);
    fn xnor(&self, other);
    fn majority(&self, another, the_other);
);

impl<T: Borrow<BootstrappingKey> + Copy> Not for FhewBool<T> {
    type Output = FhewBool<T>;

    fn not(mut self) -> Self::Output {
        self.bitnot_assign();
        self
    }
}

impl<T: Borrow<BootstrappingKey> + Copy> Not for &FhewBool<T> {
    type Output = FhewBool<T>;

    fn not(self) -> Self::Output {
        self.bitnot()
    }
}

macro_rules! impl_core_op {
    (@ impl<T> $trait:ident<$rhs:ty> for $lhs:ty; $lhs_convert:expr) => {
        paste::paste! {
            impl<T: Borrow<BootstrappingKey> + Copy> core::ops::$trait<$rhs> for $lhs {
                type Output = FhewBool<T>;

                fn [<$trait:lower>](self, rhs: $rhs) -> Self::Output {
                    let mut lhs = $lhs_convert(self);
                    lhs.[<$trait:lower _assign>](&rhs);
                    lhs
                }
            }
        }
    };
    ($(impl<T> $trait:ident<$rhs:ty> for $lhs:ty),* $(,)?) => {
        $(
            paste::paste! {
                impl<T: Borrow<BootstrappingKey> + Copy> core::ops::[<$trait Assign>]<$rhs> for $lhs {
                    fn [<$trait:lower _assign>](&mut self, rhs: $rhs) {
                        self.[<$trait:lower _assign>](&rhs);
                    }
                }
                impl<T: Borrow<BootstrappingKey> + Copy> core::ops::[<$trait Assign>]<&$rhs> for $lhs {
                    fn [<$trait:lower _assign>](&mut self, rhs: &$rhs) {
                        self.[<$trait:lower _assign>](rhs);
                    }
                }
            }
            impl_core_op!(@ impl<T> $trait<$rhs> for $lhs; core::convert::identity);
            impl_core_op!(@ impl<T> $trait<&$rhs> for $lhs; core::convert::identity);
            impl_core_op!(@ impl<T> $trait<$rhs> for &$lhs; <_>::clone);
            impl_core_op!(@ impl<T> $trait<&$rhs> for &$lhs; <_>::clone);
        )*
    }
}

impl_core_op!(
    impl<T> BitAnd<FhewBool<T>> for FhewBool<T>,
    impl<T> BitOr<FhewBool<T>> for FhewBool<T>,
    impl<T> BitXor<FhewBool<T>> for FhewBool<T>,
);

impl<T: Borrow<BootstrappingKey> + Copy> FhewBool<T> {
    pub fn select(&self, f: &Self, t: &Self) -> Self {
        (!self & f) | (self & t)
    }

    pub fn overflowing_add(&self, rhs: &Self) -> (Self, Self) {
        let sum = self ^ rhs;
        let carry = self & rhs;
        (sum, carry)
    }

    pub fn carrying_add(&self, rhs: &Self, carry: &Self) -> (Self, Self) {
        let t = &(self ^ rhs);
        let sum = t ^ carry;
        let carry = (self & rhs) | (t & carry);
        (sum, carry)
    }

    pub fn overflowing_sub(&self, rhs: &Self) -> (Self, Self) {
        let diff = self ^ rhs;
        let borrow = !self & rhs;
        (diff, borrow)
    }

    pub fn borrowing_sub(&self, rhs: &Self, borrow: &Self) -> (Self, Self) {
        let t = &(self ^ rhs);
        let diff = t ^ borrow;
        let borrow = (!self & rhs) | (!t & borrow);
        (diff, borrow)
    }
}

impl<T: Borrow<BootstrappingKey> + Copy> FhewBool<T> {
    pub(crate) fn carrying_add_assign(&mut self, rhs: &Self, carry: &mut Self) {
        (*self, *carry) = self.carrying_add(rhs, carry);
    }

    pub(crate) fn overflowing_add_assign(&mut self, rhs: &Self) -> Self {
        let (sum, carry) = self.overflowing_add(rhs);
        *self = sum;
        carry
    }
}

#[derive(Clone, Debug)]
pub struct FhewBoolDecryptionShare(LweDecryptionShare);

impl<T: Borrow<BootstrappingParam>> FhewBool<T> {
    pub fn share_decrypt(
        &self,
        sk: &LweSecretKey,
        rng: &mut impl RngCore,
    ) -> FhewBoolDecryptionShare {
        let param = self.bk.borrow();
        FhewBoolDecryptionShare(Lwe::share_decrypt(param.lwe_z(), sk, self.ct.a(), rng))
    }

    pub fn decryption_share_merge(
        &self,
        shares: impl IntoIterator<Item = FhewBoolDecryptionShare>,
    ) -> bool {
        let param = self.bk.borrow();
        let shares = shares.into_iter().map(|share| share.0);
        let pt = Lwe::decryption_share_merge(param.lwe_z(), *self.ct.b(), shares);
        Fhew::decode(param, pt)
    }
}

#[cfg(test)]
pub(crate) mod test {
    use crate::{
        bootstrapping::{Bootstrapping, BootstrappingParam},
        fhew::FhewBool,
        lwe::{Lwe, LweParam},
        rgsw::RgswParam,
        rlwe::{Rlwe, RlweParam},
        util::two_adic_primes,
    };
    use core::array::from_fn;
    use rand::thread_rng;

    #[rustfmt::skip]
    mod tt {
        const F: bool = false;
        const T: bool = true;
        pub const OVERFLOWING_ADD: [(bool, bool); 4] = [(F, F), (T, F), (T, F), (F, T)];
        pub const OVERFLOWING_SUB: [(bool, bool); 4] = [(F, F), (T, F), (T, T), (F, F)];
        pub const CARRYING_ADD:    [(bool, bool); 8] = [(F, F), (T, F), (T, F), (F, T), (T, F), (F, T), (F, T), (T, T)];
        pub const BORROWING_SUB:   [(bool, bool); 8] = [(F, F), (T, F), (T, T), (F, F), (T, T), (F, F), (F, T), (T, T)];
    }

    pub(crate) fn single_key_testing_param() -> BootstrappingParam {
        let p = 4;
        let rgsw = {
            let (log_q, log_n, log_b, d) = (28, 9, 7, 4);
            let q = two_adic_primes(log_q, log_n + 1).next().unwrap();
            let rlwe = RlweParam::new(q, p, log_n).with_decomposor(log_b, d);
            RgswParam::new(rlwe, log_b, d)
        };
        let lwe = {
            let (n, q, log_b, d) = (100, 1 << 16, 4, 4);
            LweParam::new(q, p, n).with_decomposor(log_b, d)
        };
        let w = 10;
        BootstrappingParam::new(rgsw, lwe, w)
    }

    #[test]
    fn encrypt_decrypt() {
        let mut rng = thread_rng();
        let param = single_key_testing_param();
        let sk = Lwe::sk_gen(param.lwe_z(), &mut rng);
        let pk = Rlwe::pk_gen(param.rgsw(), &(&sk).into(), &mut rng);
        for m in 0..1 << 1 {
            let m = m == 1;
            let ct0 = FhewBool::sk_encrypt(&param, &sk, m, &mut rng);
            let ct1 = FhewBool::pk_encrypt(&param, &pk, m, &mut rng);
            assert_eq!(m, ct0.decrypt(&sk));
            assert_eq!(m, ct1.decrypt(&sk));
        }
    }

    #[test]
    fn op() {
        let mut rng = thread_rng();
        let param = single_key_testing_param();
        let sk = Lwe::sk_gen(param.lwe_z(), &mut rng);
        let bk = Bootstrapping::key_gen(&param, &sk, &mut rng);
        let encrypt = |m| FhewBool::sk_encrypt(&bk, &sk, m, &mut thread_rng());

        macro_rules! assert_decrypted_to {
            ($ct:expr, $m:expr) => {
                assert_eq!($ct.decrypt(&sk), $m);
            };
        }

        for m in 0..1 << 1 {
            let m = m == 1;
            let ct = encrypt(m);
            assert_decrypted_to!(ct.bitnot(), !m);
        }
        for m in 0..1 << 2 {
            let [m0, m1] = from_fn(|i| (m >> i) & 1 == 1);
            let [ct0, ct1] = &[m0, m1].map(encrypt);
            assert_decrypted_to!(ct0.bitand(ct1), m0 & m1);
            assert_decrypted_to!(ct0.bitnand(ct1), !(m0 & m1));
            assert_decrypted_to!(ct0.bitor(ct1), m0 | m1);
            assert_decrypted_to!(ct0.bitnor(ct1), !(m0 | m1));
            assert_decrypted_to!(ct0.bitxor(ct1), m0 ^ m1);
            assert_decrypted_to!(ct0.bitxnor(ct1), !(m0 ^ m1));
        }
        for m in 0..1 << 3 {
            let [m0, m1, m2] = from_fn(|i| (m >> i) & 1 == 1);
            let [ct0, ct1, ct2] = &[m0, m1, m2].map(encrypt);
            assert_decrypted_to!(ct0.bitmajority(ct1, ct2), (m0 & m1) | (m1 & m2) | (m2 & m0));
        }
    }

    #[test]
    fn add_sub() {
        let mut rng = thread_rng();
        let param = single_key_testing_param();
        let sk = Lwe::sk_gen(param.lwe_z(), &mut rng);
        let bk = Bootstrapping::key_gen(&param, &sk, &mut rng);
        let encrypt = |m| FhewBool::sk_encrypt(&bk, &sk, m, &mut thread_rng());

        macro_rules! assert_decrypted_to {
            ($ct:expr, $m:expr) => {
                let (t0, t1) = $ct;
                assert_eq!((t0.decrypt(&sk), t1.decrypt(&sk)), $m);
            };
        }

        for m in 0..1 << 2 {
            let [m0, m1] = from_fn(|i| (m >> i) & 1 == 1);
            let [ct0, ct1] = &[m0, m1].map(encrypt);
            assert_decrypted_to!(ct0.overflowing_add(ct1), tt::OVERFLOWING_ADD[m]);
            assert_decrypted_to!(ct0.overflowing_sub(ct1), tt::OVERFLOWING_SUB[m]);
        }
        for m in 0..1 << 3 {
            let [m0, m1, m2] = from_fn(|i| (m >> i) & 1 == 1);
            let [ct0, ct1, ct2] = &[m0, m1, m2].map(encrypt);
            assert_decrypted_to!(ct0.carrying_add(ct1, ct2), tt::CARRYING_ADD[m]);
            assert_decrypted_to!(ct0.borrowing_sub(ct1, ct2), tt::BORROWING_SUB[m]);
        }
    }

    pub(crate) fn multi_key_testing_param() -> BootstrappingParam {
        let p = 4;
        let rgsw = {
            let (log_q, log_n, log_b, d) = (54, 9, 6, 9);
            let q = two_adic_primes(log_q, log_n + 1).next().unwrap();
            let rlwe = RlweParam::new(q, p, log_n).with_decomposor(log_b, d);
            RgswParam::new(rlwe, log_b, d)
        };
        let lwe = {
            let (n, q, log_b, d) = (100, 1 << 16, 4, 4);
            LweParam::new(q, p, n).with_decomposor(log_b, d)
        };
        let w = 10;
        BootstrappingParam::new(rgsw, lwe, w)
    }

    #[test]
    fn multi_key_op() {
        const N: usize = 3;

        let mut rng = thread_rng();
        let param = multi_key_testing_param();
        let crs = Bootstrapping::crs_gen(&param, &mut rng);
        let sk_shares: [_; N] = from_fn(|_| Lwe::sk_gen(param.lwe_z(), &mut rng));
        let pk = {
            let pk_share_gen = |sk| Rlwe::pk_share_gen(param.rgsw(), crs.pk(), &sk, &mut rng);
            let pk_shares = sk_shares.each_ref().map(|sk| sk.into()).map(pk_share_gen);
            Rlwe::pk_share_merge(param.rgsw(), crs.pk().clone(), pk_shares)
        };
        let bk = {
            let bk_share_gen = |sk| Bootstrapping::key_share_gen(&param, &crs, sk, &pk, &mut rng);
            let bk_shares = sk_shares.each_ref().map(bk_share_gen);
            Bootstrapping::key_share_merge(&param, crs, bk_shares)
        };
        let encrypt = |m| FhewBool::pk_encrypt(&bk, &pk, m, &mut thread_rng());

        macro_rules! assert_decrypted_to {
            ($ct:expr, $m:expr) => {{
                let ct = $ct;
                let share_decrypt = |sk| ct.share_decrypt(sk, &mut rng);
                let d_shares = sk_shares.each_ref().map(share_decrypt);
                assert_eq!(ct.decryption_share_merge(d_shares), $m);
            }};
        }

        for m in 0..1 << 1 {
            let m = m == 1;
            let ct = encrypt(m);
            assert_decrypted_to!(ct.bitnot(), !m);
        }
        for m in 0..1 << 2 {
            let [m0, m1] = from_fn(|i| (m >> i) & 1 == 1);
            let [ct0, ct1] = &[m0, m1].map(encrypt);
            assert_decrypted_to!(ct0.bitand(ct1), m0 & m1);
            assert_decrypted_to!(ct0.bitnand(ct1), !(m0 & m1));
            assert_decrypted_to!(ct0.bitor(ct1), m0 | m1);
            assert_decrypted_to!(ct0.bitnor(ct1), !(m0 | m1));
            assert_decrypted_to!(ct0.bitxor(ct1), m0 ^ m1);
            assert_decrypted_to!(ct0.bitxnor(ct1), !(m0 ^ m1));
        }
        for m in 0..1 << 3 {
            let [m0, m1, m2] = from_fn(|i| (m >> i) & 1 == 1);
            let [ct0, ct1, ct2] = &[m0, m1, m2].map(encrypt);
            assert_decrypted_to!(ct0.bitmajority(ct1, ct2), (m0 & m1) | (m1 & m2) | (m2 & m0));
        }
    }
}
