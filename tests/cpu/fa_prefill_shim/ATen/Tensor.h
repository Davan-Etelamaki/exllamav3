#pragma once

// Minimal at::Tensor stand-in, so tests/cpu/inkling_fa_prefill_fence.py can compile and RUN the
// production tensor layer inkling/fa_prefill_torch.cpp on a CPU: the fence drives the REAL
// shape/dtype/stride refusal logic and the real argument marshalling, with the buffers owned by the
// driver. Only the surface that file touches is declared — if the layer starts using another ATen
// entry point, this build fails loudly instead of quietly judging less than the code does.

#include <cstdint>
#include <cstddef>
#include <numeric>
#include <vector>

#include "cuda_runtime.h"

namespace at
{
enum ScalarType { kByte = 0, kChar, kShort, kInt, kLong, kHalf, kFloat, kDouble, kBool };

class Tensor
{
public:
    Tensor() = default;

    Tensor(void* ptr, std::vector<int64_t> sizes, std::vector<int64_t> strides,
           ScalarType dtype, int device = 0)
        : ptr_(ptr), sizes_(std::move(sizes)), strides_(std::move(strides)),
          dtype_(dtype), device_(device) {}

    int64_t dim() const { return int64_t(sizes_.size()); }
    int64_t size(int i) const { return sizes_.at(size_t(i)); }
    int64_t stride(int i) const { return strides_.at(size_t(i)); }
    int64_t numel() const
    {
        int64_t n = 1;
        for (const int64_t s : sizes_) n *= s;
        return n;
    }
    ScalarType dtype() const { return dtype_; }
    ScalarType scalar_type() const { return dtype_; }
    void* data_ptr() const { return ptr_; }
    int get_device() const { return device_; }
    bool is_contiguous() const
    {
        int64_t acc = 1;
        for (int i = int(sizes_.size()) - 1; i >= 0; --i)
        {
            if (sizes_[i] == 1) continue;
            if (strides_[i] != acc) return false;
            acc *= sizes_[i];
        }
        return true;
    }

private:
    void* ptr_ = nullptr;
    std::vector<int64_t> sizes_, strides_;
    ScalarType dtype_ = kFloat;
    int device_ = 0;
};
}  // namespace at