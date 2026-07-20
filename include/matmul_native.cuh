#pragma once

#include <cstddef>

#include "Timer.cuh"
#include "cuda_define.cuh"
#include "matmul_verify.cuh"

namespace {
__global__ void native_matmul(
    const float* A,
    const float* B,
    float* C,
    std::size_t M,
    std::size_t N,
    std::size_t K,
    float alpha,
    float beta)
{
    const std::size_t row =  blockIdx.y * blockDim.y + threadIdx.y;
    const std::size_t col =  blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        for (std::size_t i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }

        C[row * N + col] =
            alpha * sum + beta * C[row * N + col];
    }
}
}

template<std::size_t M, std::size_t N,std::size_t K>
void native_matmul_entry(
    const float* A,
    const float* B,
    float* C)
{
    Timer timer;

    const dim3 block(32, 32);
    const dim3 grid(
        (N + 31) / 32,
        (M + 31) / 32
    );

    native_matmul<<<grid, block>>>(
        A, B, C, M, N, K, 1.0f, 0.0f
    );

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    if (!verify_matmul_with_cublas(
            "native", A, B, C, M, N, K)) {
        std::exit(EXIT_FAILURE);
    }

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        native_matmul<<<grid, block>>>(
            A, B, C, M, N, K, 1.0f, 0.0f
        );
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    timer.tic();

    for (int i = 0; i < BENCH_ITERS; ++i) {
        native_matmul<<<grid, block>>>(
            A, B, C, M, N, K, 1.0f, 0.0f
        );
    }

    CHECK_CUDA(cudaGetLastError());

    const float avg_ms = timer.toc() / BENCH_ITERS;
    const double gflops =
        2.0 * M * N * K / (avg_ms * 1e6);

    std::cout
        << "native: " << avg_ms
        << " ms, GFLOPS: " << gflops
        << '\n';
}
