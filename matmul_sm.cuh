#pragma once
#include "cuda_define.cuh"
#include "Timer.cuh"
#include "matmul_verify.cuh"

constexpr int TILE = 32;

__global__ void sm_matmul(
    const float *A,
    const float *B,
    float *C,
    size_t M ,size_t N ,size_t K,
    float alpha , float beta
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (K + TILE - 1) / TILE;
    // move data to shared memory
    for(int i = 0 ; i < numTiles ; i++){
        int aCol = i * TILE + threadIdx.x;
        int bRow = i * TILE + threadIdx.y;
        
       As[threadIdx.y][threadIdx.x] = 
        (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
       Bs[threadIdx.y][threadIdx.x] = 
        (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for(int k = 0; k < TILE; k++){
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if(row < M && col < N){
         C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}


void sm_matmul_entry(
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

    sm_matmul<<<grid , block>>>(A , B , C , M , N , K, alpha, beta);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    if(!verify_matmul_with_cublas("sm", A, B, C, M, N, K)){
        exit(1);
    }

    for(int i = 0; i < WARMUP_ITERS; i++){
        sm_matmul<<<grid , block>>>(A , B , C , M , N , K, alpha, beta);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    timer.tic();
    for(int i = 0; i < BENCH_ITERS; i++){
        sm_matmul<<<grid , block>>>(A , B , C , M , N , K, alpha, beta);
    }
    CHECK_CUDA(cudaGetLastError());
    float ms1 = timer.toc();
    float avg_ms = ms1 / BENCH_ITERS;
    std::cout << avg_ms << "\n";
    std::cout << "GFLOPS:" << 2 * M * N * K / (avg_ms * 1e6) << "\n";
}
