use crate::{
    izip_eq,
    poly::{Basis, Coefficient, Evaluation},
    zipstar,
    zq::impl_rest_op_by_op_assign_ref,
    AVec, Dot, Rq, Zq,
};
use core::{borrow::Borrow, ops::MulAssign};
use itertools::{chain, Itertools};
use num_bigint::{BigInt, BigUint, ToBigInt};
use num_traits::ToPrimitive;
use rand::{distributions::Distribution, RngCore};
use std::collections::HashSet;

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

    pub fn from_i8(v: &[i8], qs: &[u64]) -> Self {
        assert!(qs.iter().all_unique());
        CrtRq(qs.iter().copied().map(|qi| Rq::from_i8(v, qi)).collect())
    }

    pub fn from_bigint(v: Vec<BigInt>, qs: &[u64]) -> Self {
        assert!(qs.iter().all_unique());
        let to_rq = |qi| Rq::from_bigint(&v, qi);
        CrtRq(qs.iter().copied().map(to_rq).collect())
    }

    pub fn into_bigint(self) -> Vec<BigInt> {
        let crt = Crt::new(&self.qs());
        zipstar!(self.0).map(|rems| crt.reconstruct(rems)).collect()
    }

    pub fn square(self) -> Self {
        &self * &self
    }

    pub fn extend_bases(mut self, ps: &[u64]) -> Self {
        assert!(chain![self.qs(), ps.iter().copied()].all_unique());
        let crt = Crt::new(&self.qs()).with_ps(ps);
        let mut rps = ps.iter().map(|p| Rq::zero(self.n(), *p)).collect_vec();
        izip_eq!(zipstar!(&self.0), zipstar!(&mut rps))
            .for_each(|(vqs, vps)| crt.extend_bases(vqs, vps));
        self.0.extend(rps);
        self
    }

    pub fn switch_bases(self, ps: &[u64]) -> Self {
        assert!(chain![self.qs(), ps.iter().copied()].all_unique());
        let n = self.qs().len();
        self.extend_bases(ps).split_off(n)
    }

    pub fn rescale(self) -> Self {
        self.rescale_k(1)
    }

    pub fn rescale_k(mut self, k: usize) -> Self {
        assert!(k > 0);
        let mut qs = self.qs();
        let ps = qs.split_off(self.0.len() - k);
        let p = ps.iter().product();
        self.round(&p);
        if k == 1 {
            let rp = self.0.pop().unwrap();
            self.each_mut(|rqi| izip_eq!(rqi, &rp).for_each(|(vq, vp)| *vq -= vp.to_u64()));
        } else {
            let rps = self.split_off(qs.len());
            self -= rps.switch_bases(&qs);
        }
        self.div(&p);
        self
    }

    fn round(&mut self, p: &BigUint) {
        let p_half = p >> 1;
        self.each_mut(|rqi| {
            let p_half = Zq::from_biguint(rqi.q(), &p_half);
            rqi.iter_mut().for_each(|v| *v += p_half)
        });
    }

    fn div(&mut self, p: &BigUint) {
        self.each_mut(|rqi| {
            let p_inv = Zq::from_biguint(rqi.q(), p).inv().unwrap();
            *rqi *= p_inv;
        });
    }

    fn each_mut(&mut self, f: impl FnMut(&mut Rq)) {
        self.0.iter_mut().for_each(f)
    }

    fn split_off(&mut self, at: usize) -> Self {
        Self(self.0.split_off(at).into())
    }
}

fn intersection(lhs: Vec<u64>, rhs: Vec<u64>) -> HashSet<u64> {
    let [lhs, rhs] = [lhs, rhs].map(HashSet::<_>::from_iter);
    lhs.intersection(&rhs).copied().collect()
}

impl MulAssign<&CrtRq<Coefficient>> for CrtRq<Coefficient> {
    fn mul_assign(&mut self, rhs: &CrtRq<Coefficient>) {
        let qs = intersection(self.qs(), rhs.qs());
        self.0.retain(|rqi| qs.contains(&rqi.q()));
        izip_eq!(&mut self.0, rhs.0.iter().filter(|p| qs.contains(&p.q())))
            .for_each(|(lhs, rhs)| *lhs *= rhs);
    }
}

impl MulAssign<&CrtRq<Evaluation>> for CrtRq<Evaluation> {
    fn mul_assign(&mut self, rhs: &CrtRq<Evaluation>) {
        let qs = intersection(self.qs(), rhs.qs());
        self.0.retain(|rqi| qs.contains(&rqi.q()));
        izip_eq!(&mut self.0, rhs.0.iter().filter(|p| qs.contains(&p.q())))
            .for_each(|(lhs, rhs)| *lhs *= rhs);
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

impl MulAssign<&BigUint> for CrtRq<Coefficient> {
    fn mul_assign(&mut self, rhs: &BigUint) {
        self.0
            .iter_mut()
            .for_each(|lhs| *lhs *= Zq::from_biguint(lhs.q(), rhs));
    }
}

impl MulAssign<&BigUint> for CrtRq<Evaluation> {
    fn mul_assign(&mut self, rhs: &BigUint) {
        self.0
            .iter_mut()
            .for_each(|lhs| *lhs *= Zq::from_biguint(lhs.q(), rhs));
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
    impl Mul<BigUint> for CrtRq<Coefficient>,
    impl Mul<BigUint> for CrtRq<Evaluation>,
);

struct Crt {
    q: BigUint,
    q_hats: Vec<BigUint>,
    q_hats_inv_qs: Vec<u64>,
    q_fracs: Vec<f64>,
    q_hats_ps: Vec<Vec<Zq>>,
    uq_ps: Vec<Vec<Zq>>,
}

impl Crt {
    fn new(qs: &[u64]) -> Self {
        let q = qs.iter().product::<BigUint>();
        let q_hats = qs.iter().map(|qi| &q / qi).collect_vec();
        let q_hats_inv_qs = izip_eq!(qs, &q_hats)
            .map(|(qi, qi_hat)| qi_hat.modinv(&(*qi).into()).unwrap().to_u64().unwrap())
            .collect_vec();
        let q_fracs = qs.iter().map(|qi| 1.0 / *qi as f64).collect_vec();
        Self {
            q,
            q_hats,
            q_hats_inv_qs,
            q_fracs,
            q_hats_ps: Vec::new(),
            uq_ps: Vec::new(),
        }
    }

    fn with_ps(mut self, ps: &[u64]) -> Self {
        self.q_hats_ps = ps
            .iter()
            .map(|pi| {
                self.q_hats
                    .iter()
                    .map(|qi_hat| Zq::from_biguint(*pi, qi_hat))
                    .collect()
            })
            .collect();
        self.uq_ps = {
            let uq = (0..=self.q_hats.len()).map(|u| &self.q * u).collect_vec();
            ps.iter()
                .map(|pi| uq.iter().map(|uq| Zq::from_biguint(*pi, uq)).collect())
                .collect()
        };
        self
    }

    fn reconstruct(&self, rems: impl IntoIterator<Item = Zq>) -> BigInt {
        let v = izip_eq!(&self.q_hats, &self.q_hats_inv_qs, rems)
            .map(|(qi_hat, qi_hat_inv, rem)| qi_hat * qi_hat_inv * rem.to_u64())
            .sum::<BigUint>();
        v.centering_rem(&self.q)
    }

    fn extend_bases<'t>(
        &self,
        vqs: impl IntoIterator<Item = &'t Zq>,
        vps: impl IntoIterator<Item = &'t mut Zq>,
    ) {
        let vs = izip_eq!(vqs, &self.q_hats_inv_qs)
            .map(|(vqi, qi_hats_inv)| (vqi * qi_hats_inv).to_u64())
            .collect_vec();
        let u = izip_eq!(&self.q_fracs, &vs)
            .map(|(qi_frac, vi)| qi_frac * *vi as f64)
            .sum::<f64>()
            .round() as usize;
        izip_eq!(vps, &self.q_hats_ps, &self.uq_ps)
            .for_each(|(vpi, q_hats_pi, uq_pi)| *vpi = q_hats_pi.dot(&vs) - uq_pi[u]);
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
        if value < rhs >> 1 {
            value.to_bigint().unwrap()
        } else {
            value.to_bigint().unwrap() - rhs.to_bigint().unwrap()
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{two_adic_primes, CrtRq};
    use itertools::Itertools;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn extend_bases() {
        let rng = &mut StdRng::from_entropy();
        for log_n in 0..10 {
            let mut primes = two_adic_primes(55, log_n + 1);
            let qs = primes.by_ref().take(8).collect_vec();
            let ps = primes.by_ref().take(8).collect_vec();
            let poly = CrtRq::sample_uniform(1 << log_n, &qs, rng);
            assert_eq!(
                poly.clone().into_bigint(),
                poly.extend_bases(&ps).into_bigint()
            );
        }
    }
}
