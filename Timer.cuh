#pragma once
#include "cuda_define.cuh"

struct Timer{
    cudaEvent_t start;
    cudaEvent_t stop;

    Timer()
    {
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
    }
    ~Timer()
    {
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;

    void tic()
    {
        CHECK_CUDA(cudaEventRecord(start));
    }
    float toc()
    {
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms , start , stop));
        return ms;
    }
};