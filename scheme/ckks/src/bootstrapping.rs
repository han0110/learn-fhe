use crate::{
    ckks::{Ckks, CkksCiphertext, CkksParam, CkksRotKey, CkksSecretKey},
    sfft::{sfft_fmats, sifft_fmats},
};
use derive_more::Deref;
use itertools::{chain, izip, Itertools};
use rand::RngCore;
use std::{collections::BTreeMap, iter::repeat};
use util::{vec_with, Complex, DiagSparseMatrix};

#[derive(Debug)]
pub struct Bootstrapping;

#[derive(Clone, Debug, Deref)]
pub struct BootstrappingParam {
    #[deref]
    param: CkksParam,
    sfft_fmats: Vec<DiagSparseMatrix<Complex>>,
    sifft_fmats: Vec<DiagSparseMatrix<Complex>>,
}

impl BootstrappingParam {
    pub fn new(param: CkksParam, r: usize) -> Self {
        let [sfft_fmats, sifft_fmats] = [sfft_fmats, sifft_fmats]
            .map(|f| vec_with![|mats| mats.iter().product(); f(param.l()).chunks(r)]);
        Self {
            param,
            sfft_fmats,
            sifft_fmats,
        }
    }

    pub fn sfft_fmats(&self) -> &[DiagSparseMatrix<Complex>] {
        &self.sfft_fmats
    }

    pub fn sifft_fmats(&self) -> &[DiagSparseMatrix<Complex>] {
        &self.sifft_fmats
    }
}

#[derive(Clone, Debug, Deref)]
pub struct BootstrappingKey {
    #[deref]
    param: BootstrappingParam,
    rtk: BTreeMap<usize, CkksRotKey>,
}

impl BootstrappingKey {
    fn rtk(&self, j: usize) -> &CkksRotKey {
        &self.rtk[&j]
    }
}

impl Bootstrapping {
    pub fn key_gen(
        param: &BootstrappingParam,
        sk: &CkksSecretKey,
        rng: &mut impl RngCore,
    ) -> BootstrappingKey {
        let rtk = chain![param.sfft_fmats(), param.sifft_fmats()]
            .flat_map(|mat| mat.bsgs().ijs())
            .filter(|j| *j != 0)
            .unique()
            .map(|j| (j, Ckks::rtk_gen(param, sk, j as _, rng)))
            .collect();
        BootstrappingKey {
            param: param.clone(),
            rtk,
        }
    }

    pub fn slot_to_coeff(bk: &BootstrappingKey, ct: CkksCiphertext) -> CkksCiphertext {
        Self::mul_mats(bk, bk.sfft_fmats(), ct)
    }

    pub fn coeff_to_slot(bk: &BootstrappingKey, ct: CkksCiphertext) -> CkksCiphertext {
        Self::mul_mats(bk, bk.sifft_fmats(), ct)
    }

    fn mul_mats(
        bk: &BootstrappingKey,
        mats: &[DiagSparseMatrix<Complex>],
        ct: CkksCiphertext,
    ) -> CkksCiphertext {
        let mul_mat = |ct, mat| Self::mul_mat(bk, mat, ct);
        mats.iter().rev().fold(ct, mul_mat)
    }

    fn mul_mat(
        bk: &BootstrappingKey,
        mat: &DiagSparseMatrix<Complex>,
        ct: CkksCiphertext,
    ) -> CkksCiphertext {
        let rotate = |j, ct| match j {
            0 => ct,
            _ => Ckks::rotate(bk, bk.rtk(j), ct),
        };
        let bsgs = mat.bsgs();
        let ct_rot = BTreeMap::from_iter(bsgs.js().map(|j| (j, rotate(j, ct.clone()))));
        let diag_rot = |i, j| mat.diag(i + j).rot_iter(-(i as i64)).cloned().collect();
        let baby_step = |(i, j)| Ckks::mul_constant(bk, diag_rot(i, j), ct_rot[&j].clone());
        let giant_step = |(i, ct)| rotate(i, ct);
        bsgs.into_iter()
            .map(|(i, js)| (i, izip!(repeat(i), js).map(baby_step).sum()))
            .map(giant_step)
            .sum()
    }
}

#[cfg(test)]
mod test {
    use crate::{
        bootstrapping::{Bootstrapping, BootstrappingParam},
        ckks::{Ckks, CkksParam},
        sfft::{sfft, sifft},
    };
    use rand::{distributions::Standard, rngs::StdRng, SeedableRng};
    use util::{assert_eq_complex, bit_reverse, izip_eq, AVec};

    #[test]
    fn coeff_to_slot_to_coeff() {
        let rng = &mut StdRng::from_entropy();
        let (log_qi, big_l, r) = (55, 8, 3);
        for log_n in 1..10 {
            let param = BootstrappingParam::new(CkksParam::new(log_n, log_qi, big_l), r);
            let (sk, pk) = Ckks::key_gen(&param, rng);
            let bk = Bootstrapping::key_gen(&param, &sk, rng);
            let m0 = &AVec::sample(param.l(), Standard, rng);
            let m1 = &sfft(bit_reverse(m0.clone()));
            let m2 = &bit_reverse(sifft(m1.clone()));
            let ct0 = Ckks::pk_encrypt(&param, &pk, Ckks::encode(&param, m0.clone()), rng);
            let ct1 = Bootstrapping::slot_to_coeff(&bk, ct0);
            let ct2 = Bootstrapping::coeff_to_slot(&bk, ct1.clone());
            izip_eq!(m0, m2).for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs));
            izip_eq!(m1, Ckks::decode(&param, Ckks::decrypt(&param, &sk, ct1)))
                .for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs, 30));
            izip_eq!(m2, Ckks::decode(&param, Ckks::decrypt(&param, &sk, ct2)))
                .for_each(|(lhs, rhs)| assert_eq_complex!(lhs, rhs, 30));
        }
    }
}
