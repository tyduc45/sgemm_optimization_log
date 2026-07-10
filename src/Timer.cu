#include "Timer.cuh"

Timer::Timer() {
    CHECK_CUDA(cudaEventCreate(&start_));
    CHECK_CUDA(cudaEventCreate(&stop_));
}

Timer::~Timer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
}

void Timer::tic() { CHECK_CUDA(cudaEventRecord(start_)); }

float Timer::toc() {
    CHECK_CUDA(cudaEventRecord(stop_));
    CHECK_CUDA(cudaEventSynchronize(stop_));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
}
