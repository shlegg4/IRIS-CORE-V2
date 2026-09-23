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
        double preprocess_host_ms{}, setup_host_ms{}, enqueue_host_ms{}, geometry_setup_host_ms{};
        double download_host_ms{}, wait_host_ms{};
        double preprocess_stream_ms{}, engine_stream_ms{};
        double association_stream_ms{}, gather_stream_ms{}, triangulation_stream_ms{};
        double output_copy_stream_ms{}, download_stream_ms{};
    } timings;
    // Raw detector tensors are retained for camera-local 2-D output. Selected
    // observations remain the input to epipolar triangulation only.
    std::array<float, 3 * 10 * 17 * 2> keypoints{};
    std::array<float, 3 * 10 * 17> keypoint_scores{};
    std::array<unsigned char, 3 * 10> candidate_valid{};
    std::array<float, 10 * 17 * 3> triangulated_xyz{};
    std::array<unsigned char, 10 * 17> triangulated_valid{};
    std::array<unsigned char, 10 * 3> assignments{};
    std::array<float, 10 * 3 * 17 * 2> selected_keypoints{};
    std::array<float, 10 * 3 * 17> selected_scores{};
    std::array<unsigned char, 10 * 3 * 17> selected_valid{};
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
