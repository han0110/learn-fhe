use crate::{
    bootstrapping::{BootstrappingKey, BootstrappingParam},
    fhew::boolean::{FhewBool, FhewBoolDecryptionShare},
    lwe::LweSecretKey,
    rlwe::RlwePublicKey,
    zipstar,
};
use core::{array::from_fn, borrow::Borrow, ops::Not};
use itertools::{izip, Itertools};
use rand::RngCore;
use std::borrow::Cow;

#[derive(Clone, Debug)]
pub struct FhewU8<T>([FhewBool<T>; 8]);

impl<T: Borrow<BootstrappingParam> + Copy> FhewU8<T> {
    pub fn sk_encrypt(bk: T, sk: &LweSecretKey, v: u8, rng: &mut impl RngCore) -> Self {
        let le_bits = from_fn(|i| FhewBool::sk_encrypt(bk, sk, (v >> i & 1) == 1, rng));
        Self(le_bits)
    }

    pub fn pk_encrypt(bk: T, pk: &RlwePublicKey, v: u8, rng: &mut impl RngCore) -> Self {
        let le_bits = from_fn(|i| FhewBool::pk_encrypt(bk, pk, (v >> i & 1) == 1, rng));
        Self(le_bits)
    }

    pub fn decrypt(self, sk: &LweSecretKey) -> u8 {
        let le_bits = self.0.into_iter().map(|bit| bit.decrypt(sk));
        u8_from_le_bits(le_bits)
    }
}

impl<T: Borrow<BootstrappingKey> + Copy> Not for FhewU8<T> {
    type Output = FhewU8<T>;

    fn not(mut self) -> Self::Output {
        self.0.each_mut().map(|bit| bit.bitnot_assign());
        self
    }
}

impl<T: Borrow<BootstrappingKey> + Copy> Not for &FhewU8<T> {
    type Output = FhewU8<T>;

    fn not(self) -> Self::Output {
        FhewU8(self.0.each_ref().map(|bit| bit.bitnot()))
    }
}

impl<T: Borrow<BootstrappingKey> + Copy> FhewU8<T> {
    pub fn overflowing_add(&self, rhs: &Self) -> (Self, FhewBool<T>) {
        let (lhs, rhs, mut carry) = (self.0.each_ref(), rhs.0.each_ref(), None);
        let sum = from_fn(|i| {
            let (sum, carry_out) = match carry.take() {
                Some(carry) => lhs[i].carrying_add(rhs[i], &carry),
                None => lhs[i].overflowing_add(rhs[i]),
            };
            carry = Some(carry_out);
            sum
        });
        (Self(sum), carry.unwrap())
    }

    pub fn carrying_add(&self, rhs: &Self, carry: &FhewBool<T>) -> (Self, FhewBool<T>) {
        let (lhs, rhs, mut carry) = (self.0.each_ref(), rhs.0.each_ref(), Cow::Borrowed(carry));
        let sum = from_fn(|i| {
            let (sum, carry_out) = lhs[i].carrying_add(rhs[i], &carry);
            carry = Cow::Owned(carry_out);
            sum
        });
        (Self(sum), carry.into_owned())
    }

    pub fn wrapping_add(&self, rhs: &Self) -> Self {
        self.overflowing_add(rhs).0
    }

    pub fn overflowing_sub(&self, rhs: &Self) -> (Self, FhewBool<T>) {
        let (lhs, rhs, mut borrow) = (self.0.each_ref(), rhs.0.each_ref(), None);
        let sum = from_fn(|i| {
            let (sum, borrow_out) = match borrow.take() {
                Some(borrow) => lhs[i].borrowing_sub(rhs[i], &borrow),
                None => lhs[i].overflowing_sub(rhs[i]),
            };
            borrow = Some(borrow_out);
            sum
        });
        (Self(sum), borrow.unwrap())
    }

    pub fn borrowing_sub(&self, rhs: &Self, borrow: &FhewBool<T>) -> (Self, FhewBool<T>) {
        let (lhs, rhs, mut borrow) = (self.0.each_ref(), rhs.0.each_ref(), Cow::Borrowed(borrow));
        let diff = from_fn(|i| {
            let (diff, borrow_out) = lhs[i].borrowing_sub(rhs[i], &borrow);
            borrow = Cow::Owned(borrow_out);
            diff
        });
        (Self(diff), borrow.into_owned())
    }

    pub fn wrapping_sub(&self, rhs: &Self) -> Self {
        self.overflowing_sub(rhs).0
    }

    pub fn wrapping_mul(&self, rhs: &Self) -> Self {
        let (lhs, rhs, mut carries) = (self.0.each_ref(), rhs.0.each_ref(), [const { None }; 7]);
        let product = from_fn(|i| {
            let mut t = (0..=i).map(|j| lhs[j] & rhs[i - j]);
            let mut sum = t.next().unwrap();
            izip!(t, &mut carries).for_each(|(tj, carry)| match carry {
                Some(carry) => sum.carrying_add_assign(&tj, carry),
                _ => *carry = Some(sum.overflowing_add_assign(&tj)),
            });
            sum
        });
        Self(product)
    }
}

macro_rules! impl_core_op {
    (@ impl<T> $trait:ident<$rhs:ty> for $lhs:ty) => {
        paste::paste! {
            impl<T: Borrow<BootstrappingKey> + Copy> core::ops::$trait<$rhs> for $lhs {
                type Output = FhewU8<T>;

                fn [<$trait:lower>](self, rhs: $rhs) -> Self::Output {
                    self.[<wrapping_ $trait:lower>](rhs.borrow())
                }
            }
        }
    };
    ($(impl<T> $trait:ident<$rhs:ty> for $lhs:ty),* $(,)?) => {
        $(
            paste::paste! {
                impl<T: Borrow<BootstrappingKey> + Copy> core::ops::[<$trait Assign>]<&$rhs> for $lhs {
                    fn [<$trait:lower _assign>](&mut self, rhs: &$rhs) {
                        *self = self.[<wrapping_ $trait:lower>](rhs);
                    }
                }
                impl<T: Borrow<BootstrappingKey> + Copy> core::ops::[<$trait Assign>]<$rhs> for $lhs {
                    fn [<$trait:lower _assign>](&mut self, rhs: $rhs) {
                        *self = self.[<wrapping_ $trait:lower>](&rhs);
                    }
                }
            }
            impl_core_op!(@ impl<T> $trait<$rhs> for $lhs);
            impl_core_op!(@ impl<T> $trait<&$rhs> for $lhs);
            impl_core_op!(@ impl<T> $trait<$rhs> for &$lhs);
            impl_core_op!(@ impl<T> $trait<&$rhs> for &$lhs);
        )*
    }
}

impl_core_op!(
    impl<T> Add<FhewU8<T>> for FhewU8<T>,
    impl<T> Sub<FhewU8<T>> for FhewU8<T>,
    impl<T> Mul<FhewU8<T>> for FhewU8<T>,
);

macro_rules! impl_wrapping_op {
    ($(impl<T> $trait:ident for FhewU8<T>),* $(,)?) => {
        $(
            paste::paste! {
                impl<T: Borrow<BootstrappingKey> + Copy> num_traits::$trait for FhewU8<T> {
                    fn [<$trait:snake>](&self, rhs: &Self) -> Self {
                        self.[<$trait:snake>](rhs)
                    }
                }
            }
        )*
    }
}

impl_wrapping_op!(
    impl<T> WrappingAdd for FhewU8<T>,
    impl<T> WrappingSub for FhewU8<T>,
    impl<T> WrappingMul for FhewU8<T>,
);

#[derive(Clone, Debug)]
pub struct FhewU8DecryptionShare([FhewBoolDecryptionShare; 8]);

impl<T: Borrow<BootstrappingParam>> FhewU8<T> {
    pub fn share_decrypt(
        &self,
        sk: &LweSecretKey,
        rng: &mut impl RngCore,
    ) -> FhewU8DecryptionShare {
        FhewU8DecryptionShare(self.0.each_ref().map(|bit| bit.share_decrypt(sk, rng)))
    }

    pub fn decryption_share_merge(
        &self,
        shares: impl IntoIterator<Item = FhewU8DecryptionShare>,
    ) -> u8 {
        let le_bits = izip!(&self.0, zipstar!(shares, 0))
            .map(|(bit, shares)| bit.decryption_share_merge(shares))
            .collect_vec();
        u8_from_le_bits(le_bits)
    }
}

fn u8_from_le_bits(le_bits: impl IntoIterator<Item = bool, IntoIter: DoubleEndedIterator>) -> u8 {
    let be_bits = le_bits.into_iter().rev();
    be_bits.fold(0, |acc, bit| (acc << 1) | bit as u8)
}

#[cfg(test)]
#[allow(unstable_name_collisions)]
mod test {
    use crate::{
        bootstrapping::Bootstrapping,
        fhew::{boolean::test::single_key_testing_param, uint8::FhewU8, FhewBool},
        lwe::Lwe,
        rlwe::Rlwe,
    };
    use rand::{thread_rng, Rng};

    trait CarryingAdd: Sized {
        fn carrying_add(self, rhs: Self, carry: bool) -> (Self, bool);
    }

    impl CarryingAdd for u8 {
        fn carrying_add(self, rhs: Self, carry: bool) -> (Self, bool) {
            match self.overflowing_add(rhs) {
                (sum, true) => (sum + carry as u8, true),
                (sum, false) => sum.overflowing_add(carry as u8),
            }
        }
    }

    trait BorrowingSub: Sized {
        fn borrowing_sub(self, rhs: Self, borrow: bool) -> (Self, bool);
    }

    impl BorrowingSub for u8 {
        fn borrowing_sub(self, rhs: Self, borrow: bool) -> (Self, bool) {
            match self.overflowing_sub(rhs) {
                (diff, true) => (diff - borrow as u8, true),
                (diff, false) => diff.overflowing_sub(borrow as u8),
            }
        }
    }

    #[test]
    fn encrypt_decrypt() {
        let mut rng = thread_rng();
        let param = single_key_testing_param();
        let sk = Lwe::sk_gen(param.lwe_z(), &mut rng);
        let pk = Rlwe::pk_gen(param.rgsw(), &(&sk).into(), &mut rng);
        for m in 0..u8::MAX {
            let ct0 = FhewU8::sk_encrypt(&param, &sk, m, &mut rng);
            let ct1 = FhewU8::pk_encrypt(&param, &pk, m, &mut rng);
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
        let encrypt_bool = |m| FhewBool::sk_encrypt(&bk, &sk, m, &mut thread_rng());
        let encrypt_u8 = |m| FhewU8::sk_encrypt(&bk, &sk, m, &mut thread_rng());

        macro_rules! assert_decrypted_to {
            ($ct0:ident $op:tt $ct1:ident, $m:expr) => {
                assert_eq!(($ct0 $op $ct1).decrypt(&sk), $m);
            };
            ($ct:expr, $m:expr) => {
                let (t0, t1) = $ct;
                assert_eq!((t0.decrypt(&sk), t1.decrypt(&sk)), $m);
            };
        }

        for m in 0..u8::MAX {
            let ct = encrypt_u8(m);
            assert_eq!((!ct).decrypt(&sk), !m);
        }
        for _ in 0..4 {
            let (m0, m1, m2) = (rng.gen(), rng.gen(), rng.gen());
            let (ct0, ct1, ct2) = &(encrypt_u8(m0), encrypt_u8(m1), encrypt_bool(m2));
            assert_decrypted_to!(ct0.overflowing_add(ct1), m0.overflowing_add(m1));
            assert_decrypted_to!(ct0.carrying_add(ct1, ct2), m0.carrying_add(m1, m2));
            assert_decrypted_to!(ct0 + ct1, m0.wrapping_add(m1));
            assert_decrypted_to!(ct0.overflowing_sub(ct1), m0.overflowing_sub(m1));
            assert_decrypted_to!(ct0.borrowing_sub(ct1, ct2), m0.borrowing_sub(m1, m2));
            assert_decrypted_to!(ct0 - ct1, m0.wrapping_sub(m1));
            assert_decrypted_to!(ct0 * ct1, m0.wrapping_mul(m1));
        }
    }
}
