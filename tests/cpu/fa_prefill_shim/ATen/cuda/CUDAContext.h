#pragma once

// The one ATen/cuda entry point the flash-prefill tensor layer uses: the current stream. The shim
// answers with the stream the driver installed, so the fence sees the real call and the real
// handle being carried into the launch.

#include <cstddef>

#include "cuda_runtime.h"

// The stream the driver installed; the production layer must carry it into the launch.
extern cudaStream_t g_shim_stream;

namespace at
{
namespace cuda
{
struct CUDAStream
{
    cudaStream_t s = nullptr;
    cudaStream_t stream() const { return s; }
};

inline CUDAStream getCurrentCUDAStream(int = 0)
{
    return CUDAStream{ ::g_shim_stream };
}
}  // namespace cuda
}  // namespace at