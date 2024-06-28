use crate::{
    poly::{Basis, Coefficient, Evaluation},
    zipstar,
    zq::impl_rest_op_by_op_assign_ref,
    AVec, Rq, Zq,
};
use core::{borrow::Borrow, ops::MulAssign};
use itertools::{izip, Itertools};
use num_bigint::{BigInt, BigUint, ToBigInt};
use num_traits::ToPrimitive;
use rand::RngCore;
use rand_distr::Distribution;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CrtRq<B: Basis = Coefficient>(AVec<Rq<B>>);

impl<B: Basis> CrtRq<B> {
    pub fn zero(n: usize, qs: &[u64]) -> Self {
        assert!(qs.iter().all_unique());
        CrtRq(qs.iter().copied().map(|qi| Rq::zero(n, qi)).collect())
    }

    pub fn n(&self) -> usize {
        self.0[0].len()
    }

    pub fn qs(&self) -> Vec<u64> {
        self.0.iter().map(Rq::q).collect()
    }

    pub fn sample_uniform(n: usize, qs: &[u64], rng: &mut impl RngCore) -> Self {
        assert!(qs.iter().all_unique());
        let sample = |qi| Rq::sample_uniform(n, qi, rng);
        CrtRq(qs.iter().copied().map(sample).collect())
    }
}

impl CrtRq<Coefficient> {
    pub fn sample_i8(
        n: usize,
        qs: &[u64],
        dist: impl Distribution<i8>,
        rng: &mut impl RngCore,
    ) -> Self {
        Self::from_i8(&AVec::sample(n, dist, rng), qs)
    }

    fn from_i8(v: &[i8], qs: &[u64]) -> Self {
        assert!(qs.iter().all_unique());
        CrtRq(qs.iter().copied().map(|qi| Rq::from_i8(v, qi)).collect())
    }

    pub fn from_bigint(v: Vec<BigInt>, qs: &[u64]) -> Self {
        assert!(qs.iter().all_unique());
        let to_rq = |qi| Rq::from_bigint(&v, qi);
        CrtRq(qs.iter().copied().map(to_rq).collect())
    }

    pub fn into_bigint(self) -> Vec<BigInt> {
        let helper = CrtHelper::new(self.qs());
        let to_biging = |rems| helper.rems_to_bigint(rems);
        zipstar!(self.0).map(to_biging).collect()
    }

    pub fn rescale(mut self) -> Self {
        let ql = self.qs().pop().unwrap();
        let rql = self.0.pop().unwrap();
        let rql = rql.into_iter().map(|v| u64::from(v + ql / 2)).collect_vec();
        self.0.iter_mut().for_each(|rqi| {
            let ql_inv = Zq::from_u64(rqi.q(), ql).inv().unwrap();
            izip!(rqi, &rql).for_each(|(vi, vl)| *vi = ql_inv * (*vi + ql / 2 - vl));
        });
        self
    }
}

impl MulAssign<&CrtRq<Coefficient>> for CrtRq<Coefficient> {
    fn mul_assign(&mut self, rhs: &CrtRq<Coefficient>) {
        izip!(&mut self.0, &rhs.0).for_each(|(lhs, rhs)| *lhs *= rhs);
    }
}

impl MulAssign<&CrtRq<Evaluation>> for CrtRq<Evaluation> {
    fn mul_assign(&mut self, rhs: &CrtRq<Evaluation>) {
        izip!(&mut self.0, &rhs.0).for_each(|(lhs, rhs)| *lhs *= rhs);
    }
}

impl MulAssign<&AVec<i8>> for CrtRq<Coefficient> {
    fn mul_assign(&mut self, rhs: &AVec<i8>) {
        self.0.iter_mut().for_each(|lhs| *lhs *= rhs);
    }
}

impl MulAssign<&AVec<i8>> for CrtRq<Evaluation> {
    fn mul_assign(&mut self, rhs: &AVec<i8>) {
        self.0.iter_mut().for_each(|lhs| *lhs *= rhs);
    }
}

macro_rules! impl_neg_by_forwarding {
    (@ impl Neg for $lhs:ty; type Output = $out:ty) => {
        paste::paste! {
            impl core::ops::Neg for $lhs {
                type Output = $out;

                fn neg(self) -> Self::Output {
                    CrtRq(-&self.0)
                }
            }
        }
    };
    ($(impl Neg for $lhs:ty),* $(,)?) => {
        $(
            impl_neg_by_forwarding!(@ impl Neg for $lhs; type Output = $lhs);
            impl_neg_by_forwarding!(@ impl Neg for &$lhs; type Output = $lhs);
        )*
    };
}

macro_rules! impl_op_assign_by_forwarding {
    (@ impl $trait:ident<$rhs:ty> for $lhs:ty) => {
        paste::paste! {
            impl core::ops::$trait<$rhs> for $lhs {
                fn [<$trait:snake>](&mut self, rhs: $rhs) {
                    core::ops::$trait::[<$trait:snake>](&mut self.0, &rhs.0);
                }
            }
        }
    };
    ($(impl $trait:ident<$rhs:ty> for $lhs:ty),* $(,)?) => {
        $(
            impl_op_assign_by_forwarding!(@ impl $trait<$rhs> for $lhs);
            impl_op_assign_by_forwarding!(@ impl $trait<&$rhs> for $lhs);
        )*
    };
}

macro_rules! impl_op_by_forwarding {
    (@ impl $trait:ident<$rhs:ty> for $lhs:ty; type Output = $out:ty) => {
        paste::paste! {
            impl core::ops::$trait<$rhs> for $lhs {
                type Output = $out;

                fn [<$trait:lower>](self, rhs: $rhs) -> Self::Output {
                    CrtRq(core::ops::$trait::[<$trait:lower>](&self.0, &rhs.0))
                }
            }
        }
    };
    ($(impl $trait:ident<$rhs:ty> for $lhs:ty),* $(,)?) => {
        $(
            impl_op_by_forwarding!(@ impl $trait<$rhs> for $lhs; type Output = $lhs);
            impl_op_by_forwarding!(@ impl $trait<&$rhs> for $lhs; type Output = $lhs);
            impl_op_by_forwarding!(@ impl $trait<$rhs> for &$lhs; type Output = $lhs);
            impl_op_by_forwarding!(@ impl $trait<&$rhs> for &$lhs; type Output = $lhs);
        )*
    };
}

impl_neg_by_forwarding!(
    impl Neg for CrtRq<Coefficient>,
    impl Neg for CrtRq<Evaluation>,
);
impl_op_assign_by_forwarding!(
    impl AddAssign<CrtRq<Coefficient>> for CrtRq<Coefficient>,
    impl AddAssign<CrtRq<Evaluation>> for CrtRq<Evaluation>,
    impl SubAssign<CrtRq<Coefficient>> for CrtRq<Coefficient>,
    impl SubAssign<CrtRq<Evaluation>> for CrtRq<Evaluation>,
);
impl_op_by_forwarding!(
    impl Add<CrtRq<Coefficient>> for CrtRq<Coefficient>,
    impl Add<CrtRq<Evaluation>> for CrtRq<Evaluation>,
    impl Sub<CrtRq<Coefficient>> for CrtRq<Coefficient>,
    impl Sub<CrtRq<Evaluation>> for CrtRq<Evaluation>,
);
impl_rest_op_by_op_assign_ref!(
    impl Mul<CrtRq<Coefficient>> for CrtRq<Coefficient>,
    impl Mul<CrtRq<Evaluation>> for CrtRq<Evaluation>,
    impl Mul<AVec<i8>> for CrtRq<Coefficient>,
    impl Mul<AVec<i8>> for CrtRq<Evaluation>,
);

struct CrtHelper {
    q: BigUint,
    q_hats: Vec<BigUint>,
    q_hats_inv_qs: Vec<u64>,
}

impl CrtHelper {
    fn new(qs: Vec<u64>) -> Self {
        let q = qs.iter().product::<BigUint>();
        let q_hats = qs.iter().map(|qi| &q / qi).collect_vec();
        let q_hats_inv_qs = izip!(qs, &q_hats)
            .map(|(qi, qi_hat)| qi_hat.modinv(&qi.into()).unwrap().to_u64().unwrap())
            .collect_vec();
        Self {
            q,
            q_hats,
            q_hats_inv_qs,
        }
    }

    fn rems_to_bigint(&self, rems: impl IntoIterator<Item = Zq>) -> BigInt {
        let v = izip!(&self.q_hats, &self.q_hats_inv_qs, rems)
            .map(|(qi_hat, qi_hat_inv, rem)| qi_hat * qi_hat_inv * u64::from(rem))
            .sum::<BigUint>();
        v.centering_rem(&self.q)
    }
}

trait CenteringRem<Rhs = Self> {
    type Output;

    fn centering_rem(self, rhs: Rhs) -> Self::Output;
}

impl CenteringRem<&BigUint> for &BigUint {
    type Output = BigInt;

    fn centering_rem(self, rhs: &BigUint) -> Self::Output {
        let value = self % rhs;
        if value < rhs >> 1usize {
            value.to_bigint().unwrap()
        } else {
            value.to_bigint().unwrap() - rhs.to_bigint().unwrap()
        }
    }
}
