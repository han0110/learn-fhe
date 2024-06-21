use core::{array::from_fn, num::Wrapping};
use fhew::{
    bootstrapping::{Bootstrapping, BootstrappingParam},
    fhew::FhewU8,
    lwe::{Lwe, LweParam},
    rgsw::RgswParam,
    rlwe::{Rlwe, RlweParam},
    util::two_adic_primes,
};
use num_traits::NumOps;
use rand::{rngs::OsRng, Rng};

const N: usize = 2;

fn multi_key_param() -> BootstrappingParam {
    let p = 4;
    let rgsw = {
        let (log_q, log_n, log_b, d) = (55, 11, 11, 5);
        let q = two_adic_primes(log_q, log_n + 1).next().unwrap();
        let rlwe = RlweParam::new(q, p, log_n).with_decomposor(log_b, d);
        RgswParam::new(rlwe, log_b, d)
    };
    let lwe = {
        let (n, q, log_b, d) = (600, 1 << 20, 4, 5);
        LweParam::new(q, p, n).with_decomposor(log_b, d)
    };
    let w = 10;
    BootstrappingParam::new(rgsw, lwe, w)
}

fn foo<T: Clone + NumOps>(a: T, b: T) -> T {
    (a.clone() + b.clone()) * a / b
}

fn main() {
    let mut rng = OsRng;
    let param = multi_key_param();
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
    let decrypt = |ct: FhewU8<_>| {
        let d_shares = sk_shares.iter().map(|sk| ct.share_decrypt(sk, &mut OsRng));
        ct.decryption_share_merge(d_shares)
    };

    let [m0, m1] = from_fn(|_| Wrapping(rng.gen_range(1..=u8::MAX)));
    let [ct0, ct1] = [m0, m1].map(|m| FhewU8::pk_encrypt(&bk, &pk, m.0, &mut rng));

    assert_eq!(foo(m0, m1).0, decrypt(foo(ct0, ct1)));
}
