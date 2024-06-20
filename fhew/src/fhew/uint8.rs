use crate::{
    bootstrapping::{BootstrappingKey, BootstrappingParam},
    fhew::boolean::FhewBool,
    lwe::LweSecretKey,
    rlwe::RlwePublicKey,
};
use core::{array::from_fn, borrow::Borrow, ops::Not};
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
        le_bits.rev().fold(0, |acc, bit| (acc << 1) | bit as u8)
    }
}

impl<T: Borrow<BootstrappingKey>> Not for FhewU8<T> {
    type Output = FhewU8<T>;

    fn not(mut self) -> Self::Output {
        self.0.each_mut().map(|bit| bit.bitnot_assign());
        self
    }
}

impl<T: Borrow<BootstrappingKey> + Clone> Not for &FhewU8<T> {
    type Output = FhewU8<T>;

    fn not(self) -> Self::Output {
        FhewU8(self.0.each_ref().map(|bit| bit.bitnot()))
    }
}

impl<T: Borrow<BootstrappingKey> + Clone> FhewU8<T> {
    pub fn overflowing_add(&self, rhs: &Self) -> (Self, FhewBool<T>) {
        let (lhs, rhs, mut carry_in) = (&self.0, &rhs.0, None);
        let sum = from_fn(|i| {
            let (sum, carry_out) = match carry_in.take() {
                Some(carry_in) => lhs[i].carrying_add(&rhs[i], &carry_in),
                None => lhs[i].overflowing_add(&rhs[i]),
            };
            carry_in = Some(carry_out);
            sum
        });
        (Self(sum), carry_in.unwrap())
    }

    pub fn carrying_add(&self, rhs: &Self, carry: &FhewBool<T>) -> (Self, FhewBool<T>) {
        let (lhs, rhs, mut carry_in) = (&self.0, &rhs.0, Cow::Borrowed(carry));
        let sum = from_fn(|i| {
            let (sum, carry_out) = lhs[i].carrying_add(&rhs[i], &carry_in);
            carry_in = Cow::Owned(carry_out);
            sum
        });
        (Self(sum), carry_in.into_owned())
    }

    pub fn wrapping_add(&self, rhs: &Self) -> Self {
        self.overflowing_add(rhs).0
    }

    pub fn overflowing_sub(&self, rhs: &Self) -> (Self, FhewBool<T>) {
        let (lhs, rhs, mut borrow_in) = (&self.0, &rhs.0, None);
        let sum = from_fn(|i| {
            let (sum, borrow_out) = match borrow_in.take() {
                Some(borrow_in) => lhs[i].borrowing_sub(&rhs[i], &borrow_in),
                None => lhs[i].overflowing_sub(&rhs[i]),
            };
            borrow_in = Some(borrow_out);
            sum
        });
        (Self(sum), borrow_in.unwrap())
    }

    pub fn borrowing_sub(&self, rhs: &Self, borrow: &FhewBool<T>) -> (Self, FhewBool<T>) {
        let (lhs, rhs, mut borrow_in) = (&self.0, &rhs.0, Cow::Borrowed(borrow));
        let diff = from_fn(|i| {
            let (diff, borrow_out) = lhs[i].borrowing_sub(&rhs[i], &borrow_in);
            borrow_in = Cow::Owned(borrow_out);
            diff
        });
        (Self(diff), borrow_in.into_owned())
    }

    pub fn wrapping_sub(&self, rhs: &Self) -> Self {
        self.overflowing_sub(rhs).0
    }

    pub fn wrapping_mul(&self, rhs: &Self) -> Self {
        let (lhs, rhs, mut carries) = (&self.0, &rhs.0, Vec::with_capacity(7));
        let product = from_fn(|i| {
            let t0 = &lhs[0] & &rhs[i];
            (1..=i).fold(t0, |mut sum, j| {
                let tj = &lhs[j] & &rhs[i - j];
                if j != i {
                    (sum, carries[j - 1]) = sum.carrying_add(&tj, &carries[j - 1]);
                } else {
                    let carry_out;
                    (sum, carry_out) = sum.overflowing_add(&tj);
                    carries.push(carry_out);
                }
                sum
            })
        });
        Self(product)
    }
}

macro_rules! impl_core_op {
    (@ impl<T> $trait:ident<$rhs:ty> for $lhs:ty) => {
        paste::paste! {
            impl<T: Borrow<BootstrappingKey> + Clone> core::ops::$trait<$rhs> for $lhs {
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
                impl<T: Borrow<BootstrappingKey> + Clone> core::ops::[<$trait Assign>]<&$rhs> for $lhs {
                    fn [<$trait:lower _assign>](&mut self, rhs: &$rhs) {
                        *self = self.[<wrapping_ $trait:lower>](rhs);
                    }
                }
                impl<T: Borrow<BootstrappingKey> + Clone> core::ops::[<$trait Assign>]<$rhs> for $lhs {
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

#[cfg(test)]
#[allow(unstable_name_collisions)]
mod test {
    use crate::{
        bootstrapping::Bootstrapping,
        fhew::{boolean::test::single_key_testing_param, uint8::FhewU8, FhewBool},
        lwe::Lwe,
        rlwe::Rlwe,
    };
    use core::array::from_fn;
    use rand::{rngs::StdRng, Rng, SeedableRng};

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
        let mut rng = StdRng::from_entropy();
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
    fn not() {
        let mut rng = StdRng::from_entropy();
        let param = single_key_testing_param();
        let sk = Lwe::sk_gen(param.lwe_z(), &mut rng);
        let bk = Bootstrapping::key_gen(&param, &sk, &mut rng);

        for m in 0..u8::MAX {
            let ct = FhewU8::sk_encrypt(&bk, &sk, m, &mut rng);
            assert_eq!((!ct).decrypt(&sk), !m);
        }
    }

    #[test]
    fn op() {
        let mut rng = StdRng::from_entropy();
        let param = single_key_testing_param();
        let sk = Lwe::sk_gen(param.lwe_z(), &mut rng);
        let bk = Bootstrapping::key_gen(&param, &sk, &mut rng);

        macro_rules! assert_decrypted_to {
            ($ct0:ident $op:tt $ct1:ident, $m:expr) => {
                assert_eq!(($ct0 $op $ct1).decrypt(&sk), $m);
            };
            ($ct:expr, $m:expr) => {
                let (t0, t1) = $ct;
                assert_eq!((t0.decrypt(&sk), t1.decrypt(&sk)), $m);
            };
        }

        for _ in 0..4 {
            let [m0, m1] = from_fn(|_| rng.gen());
            let [ct0, ct1] = &[m0, m1].map(|m| FhewU8::sk_encrypt(&bk, &sk, m, &mut rng));
            let m2 = rng.gen();
            let ct2 = &FhewBool::sk_encrypt(&bk, &sk, m2, &mut rng);
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
