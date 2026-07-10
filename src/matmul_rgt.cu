#include "matmul_rgt.cuh"

#include "Timer.cuh"
#include "cuda_define.cuh"
#include "matmul_verify.cuh"

namespace {
constexpr int TILE = 32;
__global__ void register_tiling_matmul(const float* A, const float* B, float* C,
                                       std::size_t M, std::size_t N, std::size_t K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    const std::size_t row = blockIdx.y * TILE + threadIdx.y;
    const std::size_t col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (std::size_t tile = 0; tile < (K + TILE - 1) / TILE; ++tile) {
        const std::size_t a_col = tile * TILE + threadIdx.x;
        const std::size_t b_row = tile * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = row < M && a_col < K ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = b_row < K && col < N ? B[b_row * N + col] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k) sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}
}

void register_tiling_matmul_entry(const float* A, const float* B, float* C,
                                  std::size_t M, std::size_t N, std::size_t K) {
    Timer timer;
    const dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    register_tiling_matmul<<<grid, block>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize());
    if (!verify_matmul_with_cublas("register-tiling", A, B, C, M, N, K)) std::exit(EXIT_FAILURE);
    for (int i = 0; i < WARMUP_ITERS; ++i) register_tiling_matmul<<<grid, block>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize());
    timer.tic();
    for (int i = 0; i < BENCH_ITERS; ++i) register_tiling_matmul<<<grid, block>>>(A, B, C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    const float avg_ms = timer.toc() / BENCH_ITERS;
    std::cout << "register-tiling: " << avg_ms << " ms, GFLOPS: " << 2.0 * M * N * K / (avg_ms * 1e6) << '\n';
}
