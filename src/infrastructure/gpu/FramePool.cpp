#include "iris/infrastructure/gpu/FramePool.hpp"
#include "iris/infrastructure/gpu/CudaResources.hpp"
#include <cuda_runtime_api.h>
#include <mutex>
#include <vector>
namespace iris::infrastructure::gpu {
struct FramePool::State {
    struct Slot {
        void* ptr{};
        std::size_t pitch{};
        bool used{};
    };
    int device{};
    Extent2D extent{};
    mutable std::mutex mutex;
    std::vector<Slot> slots;
    ~State() {
        cudaSetDevice(device);
        for (auto& s : slots) {
            if (s.ptr) {
                cudaFree(s.ptr);
            }
        }
    }
};
FramePool::FramePool(int device, Extent2D extent, std::size_t bpp, std::size_t capacity)
    : state_(std::make_shared<State>()) {
    state_->device = device;
    state_->extent = extent;
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    state_->slots.resize(capacity);
    for (auto& s : state_->slots) {
        check_cuda(cudaMallocPitch(&s.ptr, &s.pitch, extent.width * bpp, extent.height),
                   "cudaMallocPitch");
    }
}
FramePool::~FramePool() = default;
std::optional<GpuBuffer> FramePool::acquire() {
    std::scoped_lock lock(state_->mutex);
    for (std::size_t i = 0; i < state_->slots.size(); ++i) {
        if (!state_->slots[i].used) {
            auto& s = state_->slots[i];
            s.used = true;
            auto state = state_;
            std::shared_ptr<void> owner(s.ptr, [state, i](void*) {
                std::scoped_lock l(state->mutex);
                state->slots[i].used = false;
            });
            return GpuBuffer{s.ptr, s.pitch, s.pitch * state_->extent.height, state_->device,
                             std::move(owner)};
        }
    }
    return std::nullopt;
}
std::size_t FramePool::available() const {
    std::scoped_lock lock(state_->mutex);
    std::size_t n{};
    for (auto& s : state_->slots) {
        n += !s.used;
    }
    return n;
}
} // namespace iris::infrastructure::gpu
