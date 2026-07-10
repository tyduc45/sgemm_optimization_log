#include "cuda_define.cuh"

void cublas_matmul_row_major(cublasHandle_t handle, const float* A, const float* B,
                             float* C, std::size_t M, std::size_t N, std::size_t K) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                             &alpha, B, static_cast<int>(N), A, static_cast<int>(K),
                             &beta, C, static_cast<int>(N)));
}
