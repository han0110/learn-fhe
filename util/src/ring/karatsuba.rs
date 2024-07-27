use core::ops::{Add, AddAssign, Mul, Sub, SubAssign};
use itertools::{izip, Itertools};

pub fn nega_cyclic_karatsuba_mul_assign<T>(a: &mut [T], b: &[T])
where
    T: for<'t> AddAssign<&'t T> + for<'t> SubAssign<&'t T> + Clone,
    for<'t> &'t T: Add<&'t T, Output = T> + Sub<&'t T, Output = T> + Mul<&'t T, Output = T>,
{
    assert!(a.len().is_power_of_two());
    assert_eq!(a.len(), b.len());

    if a.len() == 1 {
        a[0] = &a[0] * &b[0];
        return;
    }

    #[allow(clippy::eq_op)]
    let zero = &a[0] - &a[0];

    let n = a.len();
    let m = n / 2;
    let (al, ar) = a.split_at(m);
    let (bl, br) = b.split_at(m);

    let mut t0 = vec![zero.clone(); n];
    let mut t1 = vec![zero.clone(); n];
    let mut t2 = vec![zero.clone(); n];
    let alr = &izip!(al, ar).map(|(l, r)| l + r).collect_vec();
    let blr = &izip!(bl, br).map(|(l, r)| l + r).collect_vec();

    karatsuba::<T>(&mut t0, al, bl);
    karatsuba::<T>(&mut t1, ar, br);
    karatsuba::<T>(&mut t2, alr, blr);

    izip!(&mut a[..], &t0, &t1).for_each(|(c, a, b)| *c = a - b);
    izip!(&mut a[..m], &t0[m..]).for_each(|(c, a)| *c += a);
    izip!(&mut a[..m], &t1[m..]).for_each(|(c, a)| *c += a);
    izip!(&mut a[..m], &t2[m..]).for_each(|(c, a)| *c -= a);
    izip!(&mut a[m..], &t0[..m]).for_each(|(c, a)| *c -= a);
    izip!(&mut a[m..], &t1[..m]).for_each(|(c, a)| *c -= a);
    izip!(&mut a[m..], &t2[..m]).for_each(|(c, a)| *c += a);
}

fn karatsuba<T>(c: &mut [T], a: &[T], b: &[T])
where
    T: for<'t> AddAssign<&'t T> + for<'t> SubAssign<&'t T> + Clone,
    for<'t> &'t T: Add<&'t T, Output = T> + Sub<&'t T, Output = T> + Mul<&'t T, Output = T>,
{
    if a.len() <= 64 {
        izip!(0.., a).for_each(|(i, a)| izip!(&mut c[i..], b).for_each(|(c, b)| *c += &(a * b)))
    } else {
        #[allow(clippy::eq_op)]
        let zero = &a[0] - &a[0];

        let n = c.len();
        let m = n / 2;
        let q = n / 4;
        let (al, ar) = a.split_at(q);
        let (bl, br) = b.split_at(q);

        let mut t0 = vec![zero.clone(); m];
        let mut t1 = vec![zero.clone(); m];
        let mut t2 = vec![zero.clone(); m];
        let alr = &izip!(al, ar).map(|(l, r)| l + r).collect_vec();
        let blr = &izip!(bl, br).map(|(l, r)| l + r).collect_vec();

        karatsuba::<T>(&mut t0, al, bl);
        karatsuba::<T>(&mut t1, ar, br);
        karatsuba::<T>(&mut t2, alr, blr);

        izip!(&mut c[q..m + q], &t2, &t0).for_each(|(c, a, b)| *c = a - b);
        izip!(&mut c[q..m + q], &t1).for_each(|(c, a)| *c -= a);
        izip!(&mut c[..m], &t0).for_each(|(c, a)| *c += a);
        izip!(&mut c[m..], &t1).for_each(|(c, a)| *c += a);
    }
}

#[cfg(test)]
mod test {
    use crate::{
        avec::AVec,
        ring::{karatsuba::nega_cyclic_karatsuba_mul_assign, test::nega_cyclic_schoolbook_mul},
    };
    use core::{
        array::from_fn,
        ops::{Add, AddAssign, Mul, Sub, SubAssign},
    };
    use rand::{distributions::Uniform, thread_rng};

    fn nega_cyclic_karatsuba_mul<T>(a: &[T], b: &[T]) -> AVec<T>
    where
        T: for<'t> AddAssign<&'t T> + for<'t> SubAssign<&'t T> + Clone,
        for<'t> &'t T: Add<&'t T, Output = T> + Sub<&'t T, Output = T> + Mul<&'t T, Output = T>,
    {
        let mut a = AVec::from(a);
        nega_cyclic_karatsuba_mul_assign::<T>(&mut a, b);
        a
    }

    #[test]
    fn nega_cyclic_mul_i64() {
        let mut rng = thread_rng();
        for log_n in 0..10 {
            let n = 1 << log_n;
            let [a, b] = &from_fn(|_| AVec::sample(n, Uniform::new(-128, 128), &mut rng));
            assert_eq!(
                nega_cyclic_karatsuba_mul::<i64>(a, b),
                nega_cyclic_schoolbook_mul(a, b)
            );
        }
    }
}
