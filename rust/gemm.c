//
// Created by Tiberio D A R Ferreira on 01/11/19.
//

#ifdef __cplusplus
#define PUT_IN_REGISTER
#else
#define PUT_IN_REGISTER register
#endif

void gemm_nn_fast_raw_c(int M, int N, int K, float ALPHA,
                        float *A, int lda,
                        float *B, int ldb,
                        float *C, int ldc)
{
    int i, j, k;
#pragma omp parallel for
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            float A_PART = ALPHA*A[i*lda + k];
            for (j = 0; j < N; ++j) {
                C[i*ldc + j] += A_PART*B[k*ldb + j];
            }
        }
    }
}