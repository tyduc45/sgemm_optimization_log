#pragma once
#include <cmath>
#include "cuda_define.cuh"

struct VerifyStats{
    unsigned int mismatches;
    unsigned int max_abs_bits;
    unsigned int max_rel_bits;
};

__device__ unsigned int float_to_ordered_bits(float value)
{
    return __float_as_uint(value);
}

__global__ void compare_matmul_kernel(
    const float *actual,
    const float *expected,
    size_t count,
    float abs_tol,
    float rel_tol,
    VerifyStats *stats
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for(size_t i = idx; i < count; i += stride){
        float a = actual[i];
        float e = expected[i];
        float abs_err = fabsf(a - e);
        float rel_err = abs_err / fmaxf(fabsf(e), 1.0f);

        atomicMax(&stats->max_abs_bits, float_to_ordered_bits(abs_err));
        atomicMax(&stats->max_rel_bits, float_to_ordered_bits(rel_err));

        if(abs_err > abs_tol && rel_err > rel_tol){
            atomicAdd(&stats->mismatches, 1u);
        }
    }
}

inline bool compare_matmul_on_gpu(
    const float *actual,
    const float *expected,
    size_t count,
    const char *name,
    float abs_tol = 1e-2f,
    float rel_tol = 1e-4f
) {
    VerifyStats *stats = nullptr;
    CHECK_CUDA(cudaMallocManaged(&stats, sizeof(VerifyStats)));
    stats->mismatches = 0;
    stats->max_abs_bits = 0;
    stats->max_rel_bits = 0;

    int block = 256;
    int grid = static_cast<int>((count + block - 1) / block);
    if(grid > 4096){
        grid = 4096;
    }

    compare_matmul_kernel<<<grid, block>>>(actual, expected, count, abs_tol, rel_tol, stats);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    union {
        unsigned int bits;
        float value;
    } max_abs_converter, max_rel_converter;
    max_abs_converter.bits = stats->max_abs_bits;
    max_rel_converter.bits = stats->max_rel_bits;
    float max_abs = max_abs_converter.value;
    float max_rel = max_rel_converter.value;
    bool ok = stats->mismatches == 0;

    if(ok){
        std::cout << name << " check ok! max_abs=" << max_abs
                  << " max_rel=" << max_rel << "\n";
    }else{
        std::cerr << name << " check failed! mismatches=" << stats->mismatches
                  << " max_abs=" << max_abs
                  << " max_rel=" << max_rel << "\n";
    }

    CHECK_CUDA(cudaFree(stats));
    return ok;
}

inline bool verify_matmul_with_cublas(
    const char *name,
    const float *A,
    const float *B,
    float *candidate_C,
    size_t M,
    size_t N,
    size_t K
) {
    float *reference_C = nullptr;
    CHECK_CUDA(cudaMalloc(&reference_C, M * N * sizeof(float)));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    cublas_matmul_row_major(handle, A, B, reference_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUBLAS(cublasDestroy(handle));

    bool ok = compare_matmul_on_gpu(candidate_C, reference_C, M * N, name);
    CHECK_CUDA(cudaFree(reference_C));
    return ok;
}
