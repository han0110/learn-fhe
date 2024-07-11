use crate::{
    avec::AVec,
    cartesian,
    float::{BigFloat, Complex},
    misc::HadamardMul,
    zq::Zq,
};
use core::{
    borrow::Borrow,
    iter::{Product, Sum},
    ops::{Mul, MulAssign},
};
use derive_more::IntoIterator;
use itertools::{chain, izip, Itertools};
use num_traits::Zero;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug)]
pub struct DiagSparseMatrix<T> {
    n: usize,
    diags: BTreeMap<usize, AVec<T>>,
}

impl<T> DiagSparseMatrix<T> {
    pub fn new(diags: impl IntoIterator<Item = (usize, AVec<T>)>) -> Self {
        let diags = diags.into_iter().collect::<BTreeMap<_, _>>();
        let n = diags.first_key_value().unwrap().1.len();
        assert!(diags.iter().map(|diag| diag.0).all_unique());
        assert!(!diags.iter().any(|diag| diag.1.len() != n));
        Self { n, diags }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn diags(&self) -> &BTreeMap<usize, AVec<T>> {
        &self.diags
    }

    pub fn diag(&self, j: usize) -> &AVec<T> {
        &self.diags()[&j]
    }

    pub fn bsgs(&self) -> BabyStepGiantStep {
        let js = self.diags().keys().cloned();
        let max_j = js.clone().max().unwrap_or_default();
        (1..=max_j)
            .map(|k| BabyStepGiantStep::new(js.clone(), k))
            .min_by_key(|bsgs| bsgs.ijs().filter(|j| *j != 0).count())
            .unwrap()
    }
}

impl DiagSparseMatrix<Zq> {
    pub fn to_dense(&self) -> Vec<Vec<Zq>> {
        let q = self.diags().first_key_value().unwrap().1[0].q();
        let mut dense = vec![vec![Zq::from_u64(q, 0); self.n()]; self.n()];
        self.clone_into_dense(&mut dense);
        dense
    }
}

impl DiagSparseMatrix<Complex> {
    pub fn to_dense(&self) -> Vec<Vec<Complex>> {
        let mut dense = vec![vec![Complex::zero(); self.n()]; self.n()];
        self.clone_into_dense(&mut dense);
        dense
    }

    pub fn inv(&self) -> Self {
        let two = &BigFloat::from(2);
        let diags = self
            .diags()
            .iter()
            .map(|(j, diag)| {
                let k = self.n() - j;
                let diag = diag.rot_iter(k as _).map(|v| v.conj() / two).collect();
                (k, diag)
            })
            .collect();
        Self { n: self.n(), diags }
    }
}

impl<T: Clone> DiagSparseMatrix<T> {
    pub fn clone_into_dense(&self, dense: &mut [Vec<T>]) {
        self.diags().iter().for_each(|(j, v)| {
            izip!(0.., v).for_each(|(i, v)| dense[i][(j + i) % self.n()] = v.clone())
        })
    }
}

impl<T> MulAssign<&DiagSparseMatrix<T>> for DiagSparseMatrix<T>
where
    AVec<T>: Sum,
    for<'t> &'t T: Mul<&'t T, Output = T>,
{
    fn mul_assign(&mut self, rhs: &DiagSparseMatrix<T>) {
        assert_eq!(self.n(), rhs.n());
        self.diags = cartesian!(self.diags(), rhs.diags())
            .map(|((i, a), (j, b))| ((i + j) % self.n(), a.hada_mul(b.rot_iter(*i as _))))
            .into_group_map()
            .into_iter()
            .map(|(i, vs)| (i, vs.into_iter().sum()))
            .collect();
    }
}

impl<T, Item> Product<Item> for DiagSparseMatrix<T>
where
    T: Clone,
    Item: Borrow<DiagSparseMatrix<T>>,
    for<'t> DiagSparseMatrix<T>: MulAssign<&'t DiagSparseMatrix<T>>,
{
    fn product<I: Iterator<Item = Item>>(mut iter: I) -> Self {
        let init = iter.next().unwrap().borrow().clone();
        iter.fold(init, |mut acc, item| {
            acc *= item.borrow();
            acc
        })
    }
}

#[derive(IntoIterator)]
pub struct BabyStepGiantStep(BTreeMap<usize, BTreeSet<usize>>);

impl BabyStepGiantStep {
    fn new(indices: impl Iterator<Item = usize>, k: usize) -> Self {
        let mut bsgs = BTreeMap::<_, BTreeSet<_>>::new();
        for idx in indices {
            let i = (idx / k) * k;
            let j = idx % k;
            bsgs.entry(i).or_default().insert(j);
        }
        Self(bsgs)
    }

    pub fn is(&self) -> impl Iterator<Item = usize> {
        self.0.keys().copied().collect_vec().into_iter()
    }

    pub fn js(&self) -> impl Iterator<Item = usize> {
        BTreeSet::from_iter(self.0.values().flatten().copied()).into_iter()
    }

    pub fn ijs(&self) -> impl Iterator<Item = usize> {
        BTreeSet::from_iter(chain![self.is(), self.js()]).into_iter()
    }
}
