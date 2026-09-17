#pragma once

#include "iris/stages/pose/PoseConfig.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>

namespace iris {

struct TensorRtMultiviewResult {
    std::array<float, 10 * 17 * 3> poses_3d{};
    std::array<unsigned char, 10 * 17> joint_valid{};
    std::array<float, 10 * 3 * 17> joint_scores{};
};

class TensorRtMultiviewEngine {
  public:
    explicit TensorRtMultiviewEngine(const PoseConfig&);
    ~TensorRtMultiviewEngine();
    TensorRtMultiviewEngine(const TensorRtMultiviewEngine&) = delete;
    TensorRtMultiviewEngine& operator=(const TensorRtMultiviewEngine&) = delete;

    void infer(const std::array<const void*, 3>& bgr_device,
               const std::array<std::size_t, 3>& strides,
               const std::array<std::uint32_t, 3>& widths,
               const std::array<std::uint32_t, 3>& heights,
               TensorRtMultiviewResult& result);

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace iris
