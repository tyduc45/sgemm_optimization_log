#include <random>
#include <cstdlib>
#include <vector>


#include "matmul_native.cuh"
#include "matmul_sm.cuh"

//#define CUBLAS_BASELINE
//#define NATIVE_TEST
#define SM_TEST


void cublas_matmul(
    const float *A,
    const float *B,
    float *C,
    size_t M ,size_t N ,size_t K
) {
    Timer timer;
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    for(int i = 0; i < WARMUP_ITERS; i++){
        cublas_matmul_row_major(handle, A, B, C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    timer.tic();
    for(int i = 0; i < BENCH_ITERS; i++){
        cublas_matmul_row_major(handle, A, B, C, M, N, K);
    }
    float ms = timer.toc();
    float avg_ms = ms / BENCH_ITERS;
    std::cout << avg_ms << "\n";
    std::cout << "GFLOPS:" << 2 * M * N * K / (avg_ms * 1e6) << "\n";
    CHECK_CUBLAS(cublasDestroy(handle));
}



int main()
{
    Timer timer;
    size_t M = 2048;
    size_t K = 2048;
    size_t N = 2048;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;

    // data prepare
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);

    std::random_device rd;
    std::mt19937 rnd(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    for(int i = 0; i < M * K;i++) h_A[i] = dis(rnd);
    for(int i = 0; i < K * N;i++) h_B[i] = dis(rnd);
    for(int i = 0; i < M * N;i++) h_C[i] = 0.0f;

    CHECK_CUDA(cudaMalloc(&A, sizeA));
    CHECK_CUDA(cudaMalloc(&B, sizeB));
    CHECK_CUDA(cudaMalloc(&C, sizeC));

    CHECK_CUDA(cudaMemcpy(A, h_A.data(), sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B, h_B.data(), sizeB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(C, h_C.data(), sizeC, cudaMemcpyHostToDevice));

#ifdef CUBLAS_BASELINE
    cublas_matmul(A , B , C , M , N , K);
#endif

#ifdef NATIVE_TEST
    native_matmul_entry(A , B , C , M , N , K);
#endif

#ifdef SM_TEST
    sm_matmul_entry(A , B , C , M , N , K);
#endif
  
    CHECK_CUDA(cudaFree(A));
    CHECK_CUDA(cudaFree(B));
    CHECK_CUDA(cudaFree(C));
    

    return 0;
}
