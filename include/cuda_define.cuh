#pragma once

#include <cstddef>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ':' << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t err = (call); \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error at " << __FILE__ << ':' << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

inline constexpr int WARMUP_ITERS = 3;
inline constexpr int BENCH_ITERS = 10;

void cublas_matmul_row_major(cublasHandle_t handle, const float* A, const float* B,
                             float* C, std::size_t M, std::size_t N, std::size_t K);
