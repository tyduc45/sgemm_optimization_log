#pragma once

#include "cuda_define.cuh"

class Timer {
public:
    Timer();
    ~Timer();
    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;

    void tic();
    float toc();

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};
