#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <vector>

#include "Timer.cuh"
#include "matmul_verify.cuh"

// A： M*K , B： K*N, C： M*N
// C[M×N] = alpha * A[M×K] * B[K×N] + beta * C[M×N]
__global__ void native_matmul(
    const float *A,
    const float *B,
    float *C,
    size_t M ,size_t N ,size_t K,
    float alpha , float beta
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N){
        float sum = 0.0f;
        for(size_t i = 0 ; i < K ; i++){
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}


void native_matmul_entry(
     const float *A,
     const float *B,
     float *C,
     size_t M ,size_t N ,size_t K
)
{
    Timer timer;
    float alpha = 1.0f;
    float beta = 0.0f;

    dim3 block(32,32);
    dim3 grid((N+31) / 32 , (M+31) / 32);

    native_matmul<<<grid , block>>>(A , B , C , M , N , K, alpha, beta);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    if(!verify_matmul_with_cublas("native", A, B, C, M, N, K)){
        exit(1);
    }

    // wram up
    for(int i = 0; i < WARMUP_ITERS; i++){
        native_matmul<<<grid , block>>>(A , B , C , M , N , K, alpha, beta);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // time tick
    timer.tic();
    for(int i = 0; i < BENCH_ITERS; i++){
        native_matmul<<<grid , block>>>(A , B , C , M , N , K, alpha, beta);
    }
    CHECK_CUDA(cudaGetLastError());
    float ms1 = timer.toc();
    float avg_ms = ms1 / BENCH_ITERS;
    std::cout << avg_ms << "\n";
    std::cout << "GFLOPS:" << 2 * M * N * K / (avg_ms * 1e6) << "\n";
}

