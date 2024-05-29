use crate::util::{rem_center, SmallPrime};
use itertools::{izip, Itertools};
use num_bigint::{BigInt, BigUint};
use rand::RngCore;
use rand_distr::{Distribution, Uniform};
use std::{
    ops::{Add, AddAssign, Deref, DerefMut, Mul, MulAssign, Neg},
    rc::Rc,
};

#[derive(Clone, Debug)]
pub struct Matrix<T> {
    data: Vec<T>,
    height: usize,
}

impl<T> Matrix<T> {
    pub fn new(height: usize, width: usize) -> Self
    where
        T: Clone + Default,
    {
        Self {
            data: vec![T::default(); width * height],
            height,
        }
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn width(&self) -> usize {
        self.data.len() / self.height
    }

    pub fn cols(&self) -> impl Iterator<Item = &[T]> {
        self.data.chunks_exact(self.height)
    }

    pub fn cols_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        self.data.chunks_exact_mut(self.height)
    }

    pub fn rows(&self) -> impl Iterator<Item = impl Iterator<Item = &T>> {
        (0..self.height()).map(|idx| self.data[idx..].iter().step_by(self.height()))
    }
}

#[derive(Clone, Debug)]
pub struct CrtPoly {
    mat: Matrix<u64>,
    qs: Vec<Rc<SmallPrime>>,
}

impl Deref for CrtPoly {
    type Target = Matrix<u64>;

    fn deref(&self) -> &Self::Target {
        &self.mat
    }
}

impl DerefMut for CrtPoly {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.mat
    }
}

impl CrtPoly {
    pub fn new(n: usize, qs: &[Rc<SmallPrime>]) -> Self {
        Self {
            mat: Matrix::new(n, qs.len()),
            qs: qs.to_vec(),
        }
    }

    pub fn sample_uniform(n: usize, qs: &[Rc<SmallPrime>], rng: &mut impl RngCore) -> Self {
        let mut poly = Self::new(n, qs);
        izip!(poly.cols_mut(), qs).for_each(|(col, qi)| {
            let uniform = Uniform::new(0, ***qi);
            col.iter_mut().for_each(|cell| *cell = uniform.sample(rng))
        });
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
        izip!(poly.cols_mut(), qs).for_each(|(col, qi)| {
            izip!(col.iter_mut(), &z).for_each(|(cell, z)| *cell = qi.from_i8(z));
            qi.neg_ntt_in_place(col);
        });
        poly
    }

    pub fn from_bigint(z: Vec<BigInt>, qs: &[Rc<SmallPrime>]) -> Self {
        let mut poly = Self::new(z.len(), qs);
        izip!(poly.cols_mut(), qs).for_each(|(col, qi)| {
            izip!(col.iter_mut(), &z).for_each(|(cell, z)| *cell = qi.from_bigint(z));
            qi.neg_ntt_in_place(col);
        });
        poly
    }

    pub fn into_bigint(mut self) -> Vec<BigInt> {
        izip!(self.mat.cols_mut(), &self.qs).for_each(|(col, qi)| qi.neg_intt_in_place(col));
        let q = &self.qs.iter().map(|q| ***q).product::<BigUint>();
        let q_hats = &self.qs.iter().map(|qi| q / ***qi).collect_vec();
        let q_hat_invs = &izip!(&self.qs, q_hats)
            .map(|(qi, qi_hat)| qi_hat.modinv(&(***qi).into()).unwrap())
            .collect_vec();
        self.rows()
            .map(|row| {
                let z = izip!(q_hats, q_hat_invs, row)
                    .map(|(qi_hat, qi_hat_inv, cell)| qi_hat * qi_hat_inv * cell)
                    .sum();
                rem_center(&z, q)
            })
            .collect()
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
        assert_eq!(self.qs, rhs.qs);

        izip!(self.cols_mut(), rhs.cols(), &rhs.qs).for_each(|(lhs, rhs, qi)| {
            izip!(lhs, rhs.iter()).for_each(|(lhs, rhs)| *lhs = qi.add(*lhs, *rhs))
        });
    }
}

impl MulAssign<&CrtPoly> for CrtPoly {
    fn mul_assign(&mut self, rhs: &CrtPoly) {
        assert_eq!(self.qs, rhs.qs);

        izip!(self.cols_mut(), rhs.cols(), &rhs.qs).for_each(|(lhs, rhs, qi)| {
            izip!(lhs, rhs.iter()).for_each(|(lhs, rhs)| *lhs = qi.mul(*lhs, *rhs))
        });
    }
}

macro_rules! impl_arithmetic_ops {
    ($(impl $trait:ident<$rhs:ty> for $lhs:ty),* $(,)?) => {
        $(
            paste::paste! {
                impl $trait<$rhs> for $lhs {
                    type Output = $lhs;

                    fn [<$trait:lower>](mut self, other: $rhs) -> $lhs {
                        self.[<$trait:lower _assign>](&other);
                        self
                    }
                }

                impl $trait<&$rhs> for $lhs {
                    type Output = $lhs;

                    fn [<$trait:lower>](mut self, other: &$rhs) -> $lhs {
                        self.[<$trait:lower _assign>](other);
                        self
                    }
                }

                impl $trait<$rhs> for &$lhs {
                    type Output = $lhs;

                    fn [<$trait:lower>](self, other: $rhs) -> $lhs {
                        let mut lhs = self.clone();
                        lhs.[<$trait:lower _assign>](&other);
                        lhs
                    }
                }

                impl $trait<&$rhs> for &$lhs {
                    type Output = $lhs;

                    fn [<$trait:lower>](self, other: &$rhs) -> $lhs {
                        let mut lhs = self.clone();
                        lhs.[<$trait:lower _assign>](other);
                        lhs
                    }
                }
            }
        )*
    };
}

impl_arithmetic_ops!(
    impl Add<CrtPoly> for CrtPoly,
    impl Mul<CrtPoly> for CrtPoly,
);
