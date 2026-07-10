#include "matmul_verify.cuh"

#include "cuda_define.cuh"

namespace {
__device__ unsigned int float_to_ordered_bits(float value) { return __float_as_uint(value); }

__global__ void compare_matmul_kernel(const float* actual, const float* expected,
                                      std::size_t count, float abs_tol, float rel_tol,
                                      VerifyStats* stats) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = blockDim.x * gridDim.x;
    for (std::size_t i = idx; i < count; i += stride) {
        const float abs_err = fabsf(actual[i] - expected[i]);
        const float rel_err = abs_err / fmaxf(fabsf(expected[i]), 1.0f);
        atomicMax(&stats->max_abs_bits, float_to_ordered_bits(abs_err));
        atomicMax(&stats->max_rel_bits, float_to_ordered_bits(rel_err));
        if (abs_err > abs_tol && rel_err > rel_tol) atomicAdd(&stats->mismatches, 1u);
    }
}
}

bool compare_matmul_on_gpu(const float* actual, const float* expected,
                           std::size_t count, const char* name,
                           float abs_tol, float rel_tol) {
    VerifyStats* stats = nullptr;
    CHECK_CUDA(cudaMallocManaged(&stats, sizeof(VerifyStats)));
    *stats = {};
    const int block = 256;
    int grid = static_cast<int>((count + block - 1) / block);
    if (grid > 4096) grid = 4096;
    compare_matmul_kernel<<<grid, block>>>(actual, expected, count, abs_tol, rel_tol, stats);
    CHECK_CUDA(cudaGetLastError()); CHECK_CUDA(cudaDeviceSynchronize());
    union Converter { unsigned int bits; float value; } max_abs{}, max_rel{};
    max_abs.bits = stats->max_abs_bits; max_rel.bits = stats->max_rel_bits;
    const bool ok = stats->mismatches == 0;
    std::ostream& out = ok ? std::cout : std::cerr;
    out << name << (ok ? " check ok!" : " check failed!")
        << " mismatches=" << stats->mismatches << " max_abs=" << max_abs.value
        << " max_rel=" << max_rel.value << '\n';
    CHECK_CUDA(cudaFree(stats));
    return ok;
}

bool verify_matmul_with_cublas(const char* name, const float* A, const float* B,
                               float* candidate_C, std::size_t M,
                               std::size_t N, std::size_t K) {
    float* reference_C = nullptr;
    CHECK_CUDA(cudaMalloc(&reference_C, M * N * sizeof(float)));
    cublasHandle_t handle{};
    CHECK_CUBLAS(cublasCreate(&handle));
    cublas_matmul_row_major(handle, A, B, reference_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUBLAS(cublasDestroy(handle));
    const bool ok = compare_matmul_on_gpu(candidate_C, reference_C, M * N, name);
    CHECK_CUDA(cudaFree(reference_C));
    return ok;
}
