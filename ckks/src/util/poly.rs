use crate::util::{prime::SmallPrime, rem_center};
use core::{
    convert::identity,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use itertools::{izip, Itertools};
use num_bigint::{BigInt, BigUint};
use rand::RngCore;
use rand_distr::Distribution;
use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
    rc::Rc,
};

#[derive(Clone, Debug)]
struct Matrix<T, S = Vec<T>> {
    data: S,
    height: usize,
    _marker: PhantomData<T>,
}

type MatrixMut<'a, T> = Matrix<T, &'a mut [T]>;

impl<T> Matrix<T> {
    fn empty(height: usize, width: usize) -> Self
    where
        T: Clone + Default,
    {
        Self::new(vec![T::default(); width * height], height)
    }
}

impl<T, S: Borrow<[T]>> Matrix<T, S> {
    fn new(data: S, height: usize) -> Self {
        assert_eq!(data.borrow().len() % height, 0);

        Self {
            data,
            height,
            _marker: PhantomData,
        }
    }

    fn height(&self) -> usize {
        self.height
    }

    fn cols(&self) -> impl Iterator<Item = &[T]> {
        self.data.borrow().chunks_exact(self.height)
    }

    fn cols_mut(&mut self) -> impl Iterator<Item = &mut [T]>
    where
        S: BorrowMut<[T]>,
    {
        self.data.borrow_mut().chunks_exact_mut(self.height)
    }

    fn rows(&self) -> impl Iterator<Item = impl Iterator<Item = &T>> {
        (0..self.height).map(|idx| self.data.borrow()[idx..].iter().step_by(self.height))
    }

    fn split_last_col_mut(&mut self) -> (&mut [T], MatrixMut<T>)
    where
        S: BorrowMut<[T]>,
    {
        let mid = self.data.borrow().len() - self.height;
        let (left, right) = self.data.borrow_mut().split_at_mut(mid);
        (right, Matrix::new(left, self.height))
    }
}

impl<T> Matrix<T, Vec<T>> {
    fn resize_cols(&mut self, n: usize)
    where
        T: Default,
    {
        self.data.resize_with(self.height * n, T::default);
    }
}

#[derive(Clone, Debug)]
pub struct CrtPoly {
    mat: Matrix<u64>,
    qs: Vec<Rc<SmallPrime>>,
}

impl CrtPoly {
    pub fn new(n: usize, qs: &[Rc<SmallPrime>]) -> Self {
        assert_eq!(qs.iter().map(|qi| ***qi).unique().count(), qs.len());

        Self {
            mat: Matrix::empty(n, qs.len()),
            qs: qs.to_vec(),
        }
    }

    pub fn n(&self) -> usize {
        self.mat.height()
    }

    pub fn sample_uniform(n: usize, qs: &[Rc<SmallPrime>], rng: &mut impl RngCore) -> Self {
        let mut poly = Self::new(n, qs);
        izip!(poly.mat.cols_mut(), qs)
            .for_each(|(col, qi)| col.iter_mut().for_each(|cell| *cell = qi.sample(rng)));
        poly
    }

    pub fn sample_small(
        n: usize,
        qs: &[Rc<SmallPrime>],
        dist: &impl Distribution<i8>,
        rng: &mut impl RngCore,
    ) -> Self {
        Self::from_i8(dist.sample_iter(rng).take(n).collect_vec(), qs)
    }

    pub fn from_i8(z: Vec<i8>, qs: &[Rc<SmallPrime>]) -> Self {
        let mut poly = Self::new(z.len(), qs);
        izip!(poly.mat.cols_mut(), qs).for_each(|(col, qi)| {
            izip!(col.iter_mut(), &z).for_each(|(cell, z)| *cell = qi.from_i8(z));
            qi.neg_ntt_in_place(col);
        });
        poly
    }

    pub fn from_bigint(z: Vec<BigInt>, qs: &[Rc<SmallPrime>]) -> Self {
        let mut poly = Self::new(z.len(), qs);
        izip!(poly.mat.cols_mut(), qs).for_each(|(col, qi)| {
            izip!(col.iter_mut(), &z).for_each(|(cell, z)| *cell = qi.from_bigint(z));
            qi.neg_ntt_in_place(col);
        });
        poly
    }

    pub fn into_bigint(mut self) -> Vec<BigInt> {
        izip!(self.mat.cols_mut(), &self.qs).for_each(|(col, qi)| qi.neg_intt_in_place(col));
        let helper = CrtHelper::new(&self.qs);
        self.mat
            .rows()
            .map(|row| helper.rems_to_bigint(row))
            .collect()
    }

    pub fn rescale(mut self) -> Self {
        let (q_k, qs) = self.qs.split_last().unwrap();
        let (q_k_col, mut mat) = self.mat.split_last_col_mut();
        let q_k_half = ***q_k / 2;

        q_k.neg_intt_in_place(q_k_col);
        q_k_col
            .iter_mut()
            .for_each(|cell| *cell = q_k.add(*cell, q_k_half));

        izip!(mat.cols_mut(), qs).for_each(|(col, qi)| {
            let q_k_inv = qi.inv(***q_k);
            let q_k_col = qi.neg_ntt(q_k_col.iter().map(|cell| qi.sub(*cell, q_k_half)).collect());
            izip!(col, q_k_col)
                .for_each(|(cell, q_k_cell)| *cell = qi.mul(q_k_inv, qi.sub(*cell, q_k_cell)));
        });

        self.mat.resize_cols(qs.len());
        self.qs.pop();
        self
    }
}

impl Neg for CrtPoly {
    type Output = CrtPoly;

    fn neg(mut self) -> Self::Output {
        izip!(self.mat.cols_mut(), &self.qs)
            .for_each(|(col, qi)| col.iter_mut().for_each(|cell| *cell = qi.neg(*cell)));
        self
    }
}

impl AddAssign<&CrtPoly> for CrtPoly {
    fn add_assign(&mut self, rhs: &CrtPoly) {
        let level = self.qs.len().min(rhs.qs.len());
        assert_eq!(self.qs[..level], rhs.qs[..level]);

        izip!(self.mat.cols_mut(), rhs.mat.cols(), &rhs.qs).for_each(|(lhs, rhs, qi)| {
            izip!(lhs, rhs.iter()).for_each(|(lhs, rhs)| *lhs = qi.add(*lhs, *rhs))
        });
    }
}

impl SubAssign<&CrtPoly> for CrtPoly {
    fn sub_assign(&mut self, rhs: &CrtPoly) {
        let level = self.qs.len().min(rhs.qs.len());
        assert_eq!(self.qs[..level], rhs.qs[..level]);

        izip!(self.mat.cols_mut(), rhs.mat.cols(), &rhs.qs).for_each(|(lhs, rhs, qi)| {
            izip!(lhs, rhs.iter()).for_each(|(lhs, rhs)| *lhs = qi.sub(*lhs, *rhs))
        });
    }
}

impl MulAssign<&CrtPoly> for CrtPoly {
    fn mul_assign(&mut self, rhs: &CrtPoly) {
        let level = self.qs.len().min(rhs.qs.len());
        assert_eq!(self.qs[..level], rhs.qs[..level]);

        izip!(self.mat.cols_mut(), rhs.mat.cols(), &rhs.qs).for_each(|(lhs, rhs, qi)| {
            izip!(lhs, rhs.iter()).for_each(|(lhs, rhs)| *lhs = qi.mul(*lhs, *rhs))
        });
    }
}

macro_rules! impl_arithmetic_ops {
    (@ impl $trait:ident<$rhs:ty> for $lhs:ty; $lhs_convert:expr) => {
        paste::paste! {
            impl $trait<$rhs> for $lhs {
                type Output = CrtPoly;

                fn [<$trait:lower>](self, rhs: $rhs) -> CrtPoly {
                    let mut lhs = $lhs_convert(self);
                    lhs.[<$trait:lower _assign>](rhs.borrow());
                    lhs
                }
            }
        }
    };
    ($(impl $trait:ident<$rhs:ty> for $lhs:ty),* $(,)?) => {
        $(
            impl_arithmetic_ops!(@ impl $trait<$rhs> for $lhs; identity);
            impl_arithmetic_ops!(@ impl $trait<&$rhs> for $lhs; identity);
            impl_arithmetic_ops!(@ impl $trait<$rhs> for &$lhs; <_>::clone);
            impl_arithmetic_ops!(@ impl $trait<&$rhs> for &$lhs; <_>::clone);
        )*
    };
}

impl_arithmetic_ops!(
    impl Add<CrtPoly> for CrtPoly,
    impl Sub<CrtPoly> for CrtPoly,
    impl Mul<CrtPoly> for CrtPoly,
);

struct CrtHelper {
    q: BigUint,
    q_hats: Vec<BigUint>,
    q_hats_inv_qs: Vec<u64>,
}

impl CrtHelper {
    fn new(qs: &[Rc<SmallPrime>]) -> Self {
        let q = qs.iter().map(|q| ***q).product::<BigUint>();
        let q_hats = qs.iter().map(|qi| &q / ***qi).collect_vec();
        let q_hats_inv_qs = izip!(qs, &q_hats)
            .map(|(qi, qi_hat)| qi.inv_biguint(qi_hat))
            .collect_vec();
        Self {
            q,
            q_hats,
            q_hats_inv_qs,
        }
    }

    fn rems_to_bigint<'t>(&self, rems: impl Iterator<Item = &'t u64>) -> BigInt {
        let v = izip!(&self.q_hats, &self.q_hats_inv_qs, rems)
            .map(|(qi_hat, qi_hat_inv, rem)| qi_hat * qi_hat_inv * rem)
            .sum();
        rem_center(&v, &self.q)
    }
}
