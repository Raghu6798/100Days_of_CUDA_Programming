#pragma once
#include <iostream>
#include <cuda_runtime.h>

// Macro to wrap CUDA calls and check for errors
#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__; \
        std::cerr << ": " << cudaGetErrorString(err_) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}