#![feature(test)]
extern crate test;

use std::time::Instant;
use packed_simd::*;
//use crossbeam::thread;

use serde::{Serialize, Deserialize};
use std::fs::{File};
use std::io::Write;


#[derive(Serialize, Deserialize, Debug)]
struct Data {
    n: usize,
    k: usize,
    alpha: f32,
    a: Vec<f32>,
    lda: usize,
    b: Vec<f32>,
    ldb: usize,
    c: Vec<f32>,
    c_new: Vec<f32>,
    ldc: usize,
}


#[no_mangle]
pub extern "C" fn gemm_nn_rust_safe(n: usize, k: usize, alpha: f32,
                                    a: *const f32, lda: usize,
                                    b: *const f32, ldb: usize,
                                    c: *const f32, ldc: usize){

    let size_a = lda + k;
    let a_n;
    unsafe {
        a_n = std::slice::from_raw_parts(a as *const f32, size_a);
    }

    let size_b = k*ldb + n;
    let b_n;
    unsafe {
        b_n = std::slice::from_raw_parts(b as *const f32, size_b);
    }

    let size_c = ldc + n;
    let c_n;
    unsafe {
        c_n = std::slice::from_raw_parts_mut(c as *mut f32, size_c);
    }


    let i = 0;
    for k_index in 0..k {
        let a_part: f32 = alpha * a_n[i * (lda) + k_index];
        let mut j = 0;
        while j < n {
            c_n[i * (ldc) + j] += a_part * (b_n[k_index * (ldb) + j]);
            j = j+1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bench() {
        println!("Start");

        let file_path = "./test.bin";

        let file = File::open(file_path).unwrap();
        let a: Data = bincode::deserialize_from(&file).unwrap();
        println!("Loaded");
        let nb_iter = 1000;

        let start = Instant::now();
        for _i in 0..nb_iter{
            gemm_nn_rust_unsafe(a.n, a.k, a.alpha, a.a.as_ptr(), a.lda, a.b.as_ptr(),
                                a.ldb, a.c.as_ptr(), a.ldc);
        }
        let single = start.elapsed().as_millis();
        let start = Instant::now();
        for _i in 0..nb_iter {
            gemm_nn_rust_simd(a.n, a.k, a.alpha, a.a.as_ptr(), a.lda, a.b.as_ptr(),
                              a.ldb, a.c.as_ptr(), a.ldc);
        }
        let simd = start.elapsed().as_millis();
        println!("Single: {}; SIMD: {}", single, simd);
    }
}
#[no_mangle]
pub extern "C" fn gemm_nn_rust_unsafe(n: usize, k: usize, alpha: f32,
                                      a: *const f32, lda: usize,
                                      b: *const f32, ldb: usize,
                                      c: *const f32, ldc: usize){
    let size_a = lda + k;
    let a_n;
    unsafe {
        a_n = std::slice::from_raw_parts(a as *const f32, size_a);
    }

    let size_b = k*ldb + n;
    let b_n;
    unsafe {
        b_n = std::slice::from_raw_parts(b as *const f32, size_b);
    }

    let size_c = ldc + n;
    let c_n;
    unsafe {
        c_n = std::slice::from_raw_parts_mut(c as *mut f32, size_c);
    }
    unsafe {
        let i = 0;
        for k_index in 0..k {
            let a_part: f32 = alpha * *a_n.get_unchecked(i * (lda) + k_index);
            let mut j = 0;
            while j < n {
                *c_n.get_unchecked_mut(i * (ldc) + j) += a_part * (*b_n.get_unchecked(k_index * (ldb) + j));
                j = j+1;
            }

        }
    }
}

#[no_mangle]
pub extern "C" fn gemm_nn_rust_simd(n: usize, k: usize, alpha: f32,
                                    a: *const f32, lda: usize,
                                    b: *const f32, ldb: usize,
                                    c: *const f32, ldc: usize)
{
    let size_a = lda + k;
    let a_n;
    unsafe {
        a_n = std::slice::from_raw_parts(a as *const f32, size_a);
    }

    let size_b = k*ldb + n;
    let b_n;
    unsafe {
        b_n = std::slice::from_raw_parts(b as *const f32, size_b);
    }

    let size_c = ldc + n;
    let c_n;
    unsafe {
        c_n = std::slice::from_raw_parts_mut(c as *mut f32, size_c);
    }
    let chunks = 16 as usize;
    let integer = n/chunks;
    unsafe {
        let i = 0;
        for k_index in 0..k {
            let a_part: f32 = alpha * *a_n.get_unchecked(i * (lda) + k_index);
            let a_part_simd = f32x16::splat(a_part);
            let mut j = 0;
            let c_ind = i * (ldc);
            let b_ind = k_index * (ldb);
            while j < chunks * integer {
                let c_ind_inner = c_ind + j;
                let c_var = f32x16::from_slice_unaligned_unchecked(&c_n[c_ind_inner ..]);
                let b_var = f32x16::from_slice_unaligned_unchecked(&b_n[(b_ind + j) ..]);

                let res = c_var + a_part_simd * b_var;
                res.write_to_slice_unaligned_unchecked(&mut c_n[c_ind_inner ..]);
                j = j + chunks;
            }

            while j < n {
                *c_n.get_unchecked_mut(c_ind + j) += a_part * (*b_n.get_unchecked(b_ind + j));
                j = j+1;
            }

        }
    }

}


/*
C ARM
real	5m36.687s
user	5m12.121s
sys	0m1.680s

C ARM OpenMP
real	1m44.369s
user	5m26.866s
sys	0m1.541s


Rust ARM Neon
real	2m26.723s
user	2m13.472s
sys	0m1.630s


Rust ARM Neon OpenMP
real	1m1.510s
user	2m40.456s
sys	0m1.469s
*/



/*
void gemm_nn(int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float *C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            PUT_IN_REGISTER float A_PART = ALPHA * A[i * lda + k];
            for (j = 0; j < N; ++j) {
                C[i*ldc + j] += A_PART*B[k*ldb + j];
            }
        }
    }
}
*/



#[no_mangle]
pub extern "C" fn gemm_nn_rust_safe_save_to_file_n_panic(n: usize, k: usize, alpha: f32,
                                                         a: *const f32, lda: usize,
                                                         b: *const f32, ldb: usize,
                                                         c: *const f32, ldc: usize){

    let size_a = lda + k;
    let a_n;
    unsafe {
        a_n = std::slice::from_raw_parts(a as *const f32, size_a);
    }

    let size_b = k*ldb + n;
    let b_n;
    unsafe {
        b_n = std::slice::from_raw_parts(b as *const f32, size_b);
    }

    let size_c = ldc + n;
    let c_n;
    unsafe {
        c_n = std::slice::from_raw_parts_mut(c as *mut f32, size_c);
    }

    let old_c = c_n.to_vec().clone();

    gemm_nn_rust_safe(n, k, alpha,a, lda,b, ldb,c, ldc);



    let new = Data{
        n,
        k,
        alpha,
        a: a_n.to_vec(),
        lda,
        b: b_n.to_vec(),
        ldb,
        c: old_c,
        c_new: c_n.to_vec(),
        ldc
    };

    let encoded: Vec<u8> = bincode::serialize(&new).unwrap();

    let file_path = "./test.bin";

    let mut file = File::create(file_path).unwrap();
    file.write_all(&encoded).unwrap();
    panic!();

}