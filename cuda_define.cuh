#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cstdlib>

#define CHECK_CUDA(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA Error: "                           \
                  << cudaGetErrorString(err)                  \
                  << " at " << __LINE__ << std::endl;         \
        exit(1);                                              \
    }                                                         \
} while (0)

#define CHECK_CUBLAS(call)                                      \
do {                                                          \
    cublasStatus_t err = call;                                   \
    if (err != CUBLAS_STATUS_SUCCESS) {                                 \
        std::cerr << "CUBLAS Error: "                           \
                  << " at " << __LINE__ << std::endl;         \
        exit(1);                                              \
    }                                                         \
} while (0)

constexpr int WARMUP_ITERS = 3;
constexpr int BENCH_ITERS = 10;

inline void cublas_matmul_row_major(
    cublasHandle_t handle,
    const float *A,
    const float *B,
    float *C,
    size_t M,
    size_t N,
    size_t K
) {
    float alpha = 1.0f;
    float beta = 0.0f;

    CHECK_CUBLAS(cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        static_cast<int>(N),
        static_cast<int>(M),
        static_cast<int>(K),
        &alpha,
        B, static_cast<int>(N),
        A, static_cast<int>(K),
        &beta,
        C, static_cast<int>(N)
    ));
}
