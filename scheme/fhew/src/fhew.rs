//! Implementation of [\[MP21\]](https://eprint.iacr.org/2020/086.pdf) with
//! [\[LMKCDEY\]](https://eprint.iacr.org/2022/198.pdf) bootstrapping.

use crate::{
    bootstrapping::{Bootstrapping, BootstrappingKey, BootstrappingParam},
    lwe::{Lwe, LweCiphertext, LwePlaintext},
};
use core::iter::repeat;

mod boolean;
mod uint8;

pub use boolean::{FhewBool, FhewBoolDecryptionShare};
pub use uint8::FhewU8;

#[derive(Debug)]
pub struct Fhew;

impl Fhew {
    fn decode(param: &BootstrappingParam, pt: LwePlaintext) -> bool {
        assert_eq!(param.p(), 4);
        let m = Lwe::decode(param.lwe_z(), pt).to_u64();
        assert!(m == 0 || m == 1);
        m == 1
    }

    pub fn not(param: &BootstrappingParam, LweCiphertext(a, b): LweCiphertext) -> LweCiphertext {
        LweCiphertext(-a, -b + param.big_q_by_4())
    }

    fn op(bk: &BootstrappingKey, table: [usize; 4], ct: LweCiphertext) -> LweCiphertext {
        let map = [-bk.big_q_by_8(), bk.big_q_by_8()];
        let f = table
            .into_iter()
            .flat_map(|out| repeat(map[out]).take(bk.q_by_8()))
            .collect();
        let LweCiphertext(a, b) = Bootstrapping::bootstrap(bk, &f, ct);
        LweCiphertext(a, b + bk.big_q_by_8())
    }
}

macro_rules! impl_op {
    (@ $op:ident, $table:expr, |$($ct:ident),+| $lin:expr) => {
        impl Fhew {
            pub fn $op(
                bk: &BootstrappingKey,
                $($ct: LweCiphertext,)+
            ) -> LweCiphertext {
                Fhew::op(bk, $table, $lin)
            }
        }
    };
    ($($op:ident, $table:expr, |$($ct:ident),+| $lin:expr);* $(;)?) => {
        $(impl_op!(@ $op, $table, |$($ct),+| $lin);)*
    }
}

// Table 1 in 2020/086.
impl_op!(
         and, [0, 0, 0, 1], |ct0, ct1| ct0 + ct1;
        nand, [1, 1, 1, 0], |ct0, ct1| ct0 + ct1;
          or, [0, 1, 1, 1], |ct0, ct1| ct0 + ct1;
         nor, [1, 0, 0, 0], |ct0, ct1| ct0 + ct1;
         xor, [0, 1, 1, 1], |ct0, ct1| (ct0 - ct1).double();
        xnor, [1, 0, 0, 0], |ct0, ct1| (ct0 - ct1).double();
    majority, [0, 0, 0, 1], |ct0, ct1, ct2| ct0 + ct1 + ct2;
);
