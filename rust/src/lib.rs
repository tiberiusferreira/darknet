use std::time::Instant;
use packed_simd::*;
#[no_mangle]
pub extern "C" fn hello_rust() -> *const u8 {
    "Hello, world!\0".as_ptr()
}

#[no_mangle]
pub extern "C" fn gemm_nn_rust(n: usize, k: usize, alpha: f32,
                               a: *const f32, lda: usize,
                               b: *const f32, ldb: usize,
                               c: *const f32, ldc: usize)
{

//    let start = Instant::now();
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
            let a_part_simd = f32x16::splat(a_part);
//            for j in 0..n {
            let mut j = 0;
            while j < n {
//                *c_n.get_unchecked_mut(i * (ldc) + j) += a_part * (*b_n.get_unchecked(k_index * (ldb) + j));



                let c_var = f32x16::from_slice_unaligned_unchecked(&c_n[(i * (ldc) + j ) ..]);
                let b_var = f32x16::from_slice_unaligned_unchecked(&b_n[(k_index * (ldb) + j) ..]);

                let res = c_var + a_part_simd * b_var;
                res.write_to_slice_unaligned_unchecked(&mut c_n[(i * (ldc) + j) ..]);

//                *c_n.get_unchecked_mut(i * (ldc) + j) += a_part * (*b_n.get_unchecked(k_index * (ldb) + j));

//                println!("{:?}", res.extract(0));
//                println!("{:?}", *c_n.get_unchecked_mut(i * (ldc) + j));
                j = j+16;
            }

        }
    }



//    println!("Took {}ms", start.elapsed().as_millis());
//    println!("Done");
    /*
     for (i = 0; i < M; ++i) {
            for (k = 0; k < K; ++k) {
                PUT_IN_REGISTER float A_PART = ALPHA * A[i * lda + k];
                for (j = 0; j < N; ++j) {
                    C[i*ldc + j] += A_PART*B[k*ldb + j];
                }
            }
        }
        */
}