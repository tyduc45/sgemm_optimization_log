#include <random>
#include <vector>
#include <functional>
#include "cuda_define.cuh"
#include "matmul_native.cuh"
#include "matmul_sm.cuh"
#include "matmul_sm_sliding.cuh"

template<std::size_t M, std::size_t N, std::size_t K>
using matmul_type = std::function<void(const float*, const float*, float*)>;

int main() {
    constexpr std::size_t M = 512, K = 512, N = 512;
    std::vector<float> h_A(M * K), h_B(K * N);
    std::mt19937 random_engine(42);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    for (float& value : h_A) value = distribution(random_engine);
    for (float& value : h_B) value = distribution(random_engine);

    float *A = nullptr, *B = nullptr, *C = nullptr;
    CHECK_CUDA(cudaMalloc(&A, h_A.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&B, h_B.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&C, M * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(C, 0, M * N * sizeof(float)));

    matmul_type<M,N,K> matmul_entry;
    //matmul_entry = native_matmul_entry<M,N,K>;
    //matmul_entry = sm_matmul_entry<M,N,K>;
    matmul_entry = sm_sliding_matmul_entry<M,N,K>;

    matmul_entry(A, B, C);

    CHECK_CUDA(cudaFree(A)); CHECK_CUDA(cudaFree(B)); CHECK_CUDA(cudaFree(C));
    return 0;
}
