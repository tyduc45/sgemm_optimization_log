#pragma once

#include <cstddef>

#include "Timer.cuh"
#include "cuda_define.cuh"
#include "matmul_verify.cuh"

namespace{
    template<std::size_t TILE , std::size_t K_>
    __global__ void sm_matmul(const float* A, const float* B, float* C,
                     std::size_t M, std::size_t N, std::size_t K,
                     float alpha , float beta){
       const std::size_t row = blockDim.y * blockIdx.y + threadIdx.y;
       const std::size_t col = blockDim.x * blockIdx.x + threadIdx.x;
       const float *A_start = A + blockDim.y * blockIdx.y * K;
       const float *B_start = B + blockDim.x *blockIdx.x;

       __shared__ float As[TILE][K_];
       __shared__ float Bs[K_][TILE];

       #pragma unroll
       for(int tile = 0 ; tile < K; tile += TILE){
            As[threadIdx.y][threadIdx.x + tile] = A_start[threadIdx.y * K + threadIdx.x + tile];
            Bs[threadIdx.y + tile][threadIdx.x] = B_start[(threadIdx.y + tile) * N + threadIdx.x];
       }
       __syncthreads();

       float sum = 0.0f;
       #pragma unroll
       for(int k = 0 ; k < K; k++){
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
       }
        __syncthreads();
       C[row * N + col] =  C[row * N + col] * beta + sum * alpha;
    }
}

template<std::size_t M, std::size_t N, std::size_t K>
void sm_matmul_entry(const float* A, const float* B, float* C)
{
    constexpr std::size_t TILE = 32;

    Timer timer;
    const dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    sm_matmul<TILE, K><<<grid, block>>>(A, B, C, M, N, K, 1.0f, 0.0f);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    if (!verify_matmul_with_cublas("shared-memory", A, B, C, M, N, K)) {
        std::exit(EXIT_FAILURE);
    }
    for (int i = 0; i < WARMUP_ITERS; ++i){
        sm_matmul<TILE, K><<<grid, block>>>(A, B, C, M, N, K, 1.0f, 0.0f);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.tic();
    for (int i = 0; i < BENCH_ITERS; ++i) {
        sm_matmul<TILE, K><<<grid, block>>>(A, B, C, M, N, K, 1.0f, 0.0f);
    }
    CHECK_CUDA(cudaGetLastError());
    const float avg_ms = timer.toc() / BENCH_ITERS;
    std::cout << "shared-memory: " << avg_ms << " ms, GFLOPS: " << 2.0 * M * N * K / (avg_ms * 1e6) << '\n';
}
