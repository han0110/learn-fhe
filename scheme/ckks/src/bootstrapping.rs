use crate::{
    ckks::{Ckks, CkksCiphertext, CkksParam, CkksRotKey, CkksSecretKey},
    sfft::{sfft_factors, sifft_factors},
};
use core::ops::Add;
use derive_more::Deref;
use itertools::{chain, Itertools};
use rand::RngCore;
use std::collections::BTreeMap;
use util::{vec_with, Complex, DiagSparseMatrix};

#[derive(Debug)]
pub struct Bootstrapping;

#[derive(Clone, Debug, Deref)]
pub struct BootstrappingParam {
    #[deref]
    param: CkksParam,
    sfft_factors: Vec<DiagSparseMatrix<Complex>>,
    sifft_factors: Vec<DiagSparseMatrix<Complex>>,
}

impl BootstrappingParam {
    pub fn new(param: CkksParam, r: usize) -> Self {
        let [sfft_factors, sifft_factors] = [sfft_factors, sifft_factors]
            .map(|f| vec_with![|factors| factors.iter().product(); f(param.l()).chunks(r)]);
        Self {
            param,
            sfft_factors,
            sifft_factors,
        }
    }

    pub fn sfft_factors(&self) -> &[DiagSparseMatrix<Complex>] {
        &self.sfft_factors
    }

    pub fn sifft_factors(&self) -> &[DiagSparseMatrix<Complex>] {
        &self.sifft_factors
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
        let rtk = {
            let js = chain![param.sfft_factors(), param.sifft_factors()]
                .flat_map(DiagSparseMatrix::diags)
                .filter_map(|(j, _)| (j != 0).then_some(j))
                .unique();
            js.map(|j| (j, Ckks::rtk_gen(param, sk, j as _, rng)))
                .collect()
        };
        BootstrappingKey {
            param: param.clone(),
            rtk,
        }
    }

    pub fn slot_to_coeff(bk: &BootstrappingKey, ct: CkksCiphertext) -> CkksCiphertext {
        Bootstrapping::mul_mat(bk, bk.sfft_factors(), ct)
    }

    pub fn coeff_to_slot(bk: &BootstrappingKey, ct: CkksCiphertext) -> CkksCiphertext {
        Bootstrapping::mul_mat(bk, bk.sifft_factors(), ct)
    }

    fn mul_mat(
        bk: &BootstrappingKey,
        factors: &[DiagSparseMatrix<Complex>],
        ct: CkksCiphertext,
    ) -> CkksCiphertext {
        let rot = |ct, j| match j {
            0 => ct,
            _ => Ckks::rotate(bk, bk.rtk(j), ct),
        };
        factors.iter().rev().fold(ct, |ct, factor| {
            factor
                .diags()
                .map(|(j, diag)| Ckks::mul_constant(bk, diag.clone(), rot(ct.clone(), j)))
                .reduce(Add::add)
                .unwrap()
        })
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
        let (log_qi, big_l, r) = (55, 16, 3);
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
