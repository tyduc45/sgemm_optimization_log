#pragma once

#include <cstddef>

struct VerifyStats {
    unsigned int mismatches;
    unsigned int max_abs_bits;
    unsigned int max_rel_bits;
};

bool compare_matmul_on_gpu(const float* actual, const float* expected,
                           std::size_t count, const char* name,
                           float abs_tol = 1e-2f, float rel_tol = 1e-4f);

bool verify_matmul_with_cublas(const char* name, const float* A, const float* B,
                               float* candidate_C, std::size_t M,
                               std::size_t N, std::size_t K);
