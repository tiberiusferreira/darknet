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
        let non_simd = start.elapsed().as_millis();
        println!("{}    {}", a.c[100], a.c_new[100]);
//        assert_eq!(a.c[100], a.c_new[100]);

        let start = Instant::now();
        for _i in 0..nb_iter {
            unsafe{
                gemm_nn_rust_unsafe_reddit(a.a.as_ptr(), a.lda, 1,a.k, a.b.as_ptr(), a.ldb, a.k,
                                           a.n, a.c.as_ptr(), a.ldc, 1, a.n, a.alpha);
            }
        }
        let reddit = start.elapsed().as_millis();

        println!("Time for {} iterations. Non SIMD: {}ms; Reddit: {}ms", nb_iter, non_simd, reddit);
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
            let a_part: f32 = alpha * *a_n.get_unchecked( k_index);
            let mut j = 0;
            while j < n {           // stride a = 1 // stride b = ldb // stride c = 1
                // rows a = 1
                *c_n.get_unchecked_mut( j) += a_part * (*b_n.get_unchecked(k_index * (ldb) + j));
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
    let chunks = 4 as usize;
    let integer = n/chunks;
    unsafe {
        let i = 0;
        for k_index in 0..k {
            let a_part: f32 = alpha * *a_n.get_unchecked(i * (lda) + k_index);
            let a_part_simd = f32x4::splat(a_part);
            let mut j = 0;
            let c_ind = i * (ldc);
            let b_ind = k_index * (ldb);
            while j < chunks * integer {
                let c_ind_inner = c_ind + j;
                let c_var = f32x4::from_slice_unaligned_unchecked(&c_n[c_ind_inner ..]);
                let b_var = f32x4::from_slice_unaligned_unchecked(&b_n[(b_ind + j) ..]);

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


#[no_mangle]
pub extern "C" fn gemm_nn_rust_unsafe_reddit_c(m: usize, n: usize, k: usize, alpha: f32,
                                               a: *const f32, lda: usize,
                                               b: *const f32, ldb: usize,
                                               c: *const f32, ldc: usize){
    unsafe{
        gemm_nn_rust_unsafe_reddit(a, lda, m,k, b, ldb, k,
                                   n, c, ldc, m, n, alpha);
    }
}

#[no_mangle]
#[allow(clippy::style)]
pub unsafe extern "C" fn gemm_nn_rust_unsafe_reddit(
    a: *const f32,
    a_row_stride: usize,
    nb_rows_a: usize,
    nb_cols_a: usize,
    b: *const f32,
    b_row_stride: usize,
    nb_rows_b: usize,
    nb_cols_b: usize,
    c: *const f32,
    c_row_stride: usize,
    nb_rows_c: usize,
    nb_cols_c: usize,
    alpha: f32,
) {
    // len_x = lda
    // len_z = k
    // a_stride = 1
    let size_a = nb_rows_a * a_row_stride;
    let a_n = std::slice::from_raw_parts(a as *const f32, size_a);
    let a_n = Matrix {
        buf: a_n,
        row_to_row_stride: a_row_stride,
        n_rows: nb_rows_a,
        n_cols: nb_cols_a,
    };

    // b_stride = ldb
    // len_y = n
    let size_b = nb_rows_b * b_row_stride;
    let b_n = std::slice::from_raw_parts(b as *const f32, size_b);
    let b_n = Matrix {
        buf: b_n,
        row_to_row_stride: b_row_stride,
        n_rows: nb_rows_b,
        n_cols: nb_cols_b,
    };


    let size_c = nb_rows_c * c_row_stride;
    let c_n = std::slice::from_raw_parts_mut(c as *mut f32, size_c);
    let mut c_n = MatrixMut {
        buf: c_n,
        row_to_row_stride: c_row_stride,
        n_rows: nb_rows_c,
        n_cols: nb_cols_c,
    };

    gemm_nn_rust(alpha, &a_n, &b_n, &mut c_n);
}

pub struct Matrix<'a> {
    buf: &'a [f32],
    row_to_row_stride: usize,
    n_rows: usize,
    n_cols: usize,
}

pub struct MatrixMut<'a> {
    buf: &'a mut [f32],
    row_to_row_stride: usize,
    n_rows: usize,
    n_cols: usize,
}

impl<'a> Matrix<'a> {
    fn rows(&self) -> impl Iterator<Item = &[f32]> {
        let n_cols = self.n_cols;
        let buf = self.buf;
        buf.chunks(self.row_to_row_stride)
            .take(self.n_rows)
            .map(move |r| &r[..n_cols])
    }
}

impl<'a> MatrixMut<'a> {
    fn rows(&mut self) -> impl Iterator<Item = &mut [f32]> {
        let n_cols = self.n_cols;
        self.buf
            .chunks_mut(self.row_to_row_stride)
            .take(self.n_rows)
            .map(move |r| &mut r[..n_cols])
    }
}

// c stride is n
// ldc stride is n
use rayon::prelude::*;

pub fn gemm_nn_rust(alpha: f32, a_n: &Matrix<'_>, b_n: &Matrix<'_>, c_n: &mut MatrixMut<'_>) {
    for (a_row, c_row) in a_n.rows().into_iter().zip(c_n.rows()) {
        for (&a_val, b_row) in a_row.iter().zip(b_n.rows()) {
            let a_part = alpha * a_val;

            for (c_cell, &b_val) in c_row.iter_mut().zip(b_row.iter()) {
                *c_cell += a_part * b_val;
            }
        }
    }
}


/*


pub fn gemm_nn_rust(alpha: f32, a_n: &Matrix<'_>, b_n: &Matrix<'_>, c_n: &mut MatrixMut<'_>) {
    for (a_row, c_row) in a_n.rows().zip(c_n.rows()) {
        for (&a_val, b_row) in a_row.iter().zip(b_n.rows()) {
            let a_part = alpha * a_val;

            for (c_cell, &b_val) in c_row.iter_mut().zip(b_row.iter()) {
                *c_cell += a_part * b_val;
            }
        }
    }
}

*/