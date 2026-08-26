#pragma once
#include "iris/pipeline/Frame.hpp"
#include <optional>
namespace iris::infrastructure::gpu {
class FramePool {
  public:
    FramePool(int, Extent2D, std::size_t, std::size_t);
    ~FramePool();
    FramePool(const FramePool&) = delete;
    FramePool& operator=(const FramePool&) = delete;
    std::optional<GpuBuffer> acquire();
    std::size_t available() const;

  private:
    struct State;
    std::shared_ptr<State> state_;
};
} // namespace iris::infrastructure::gpu
