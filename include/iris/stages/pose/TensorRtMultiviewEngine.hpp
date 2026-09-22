#pragma once

#include "iris/stages/pose/PoseConfig.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>

namespace iris {

struct TensorRtMultiviewResult {
    // Host intervals are sequential; stream intervals overlap them and must not be added.
    struct Timings {
        double preprocess_host_ms{}, setup_host_ms{}, enqueue_host_ms{};
        double download_host_ms{}, wait_host_ms{};
        double preprocess_stream_ms{}, engine_stream_ms{}, download_stream_ms{};
    } timings;
    // The engine is batched over camera views.  These are indexed [view][candidate][joint].
    std::array<float, 3 * 10 * 17 * 2> keypoints{};
    std::array<float, 3 * 10 * 17> keypoint_scores{};
    std::array<float, 3 * 10> instance_scores{};
    std::array<float, 3 * 10 * 4> boxes{};
    std::array<unsigned char, 3 * 10> candidate_valid{};
    // Intrinsics in the letterboxed 640x640 coordinate system used by keypoints.
    std::array<float, 3 * 9> letterbox_intrinsics{};
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
