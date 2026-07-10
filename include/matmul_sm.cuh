#pragma once

#include <cstddef>

void sm_matmul_entry(const float* A, const float* B, float* C,
                     std::size_t M, std::size_t N, std::size_t K);
