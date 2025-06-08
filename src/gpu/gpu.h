#pragma once

#define CUTLASS_CHECK(status)                                                                       \
{                                                                                                   \
    cutlass::Status error = status;                                                                 \
    if (error != cutlass::Status::kSuccess) {                                                       \
        std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: "              \
                  << __FILE__ << ":" << __LINE__ << std::endl;                                      \
    }                                                                                               \
}

#define CUDA_CHECK(status)                                                                          \
{                                                                                                   \
    cudaError_t error = status;                                                                     \
    if (error != cudaSuccess) {                                                                     \
        std::cerr << "Got cuda error: " << cudaGetErrorString(error) << " at: "                     \
                  << __FILE__ << ":" << __LINE__ << std::endl;                                      \
    }                                                                                               \
}