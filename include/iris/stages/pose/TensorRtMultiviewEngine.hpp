#pragma once

#include "iris/stages/pose/PoseConfig.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace iris {

struct TensorRtMultiviewResult {
    // Host intervals are sequential; stream intervals overlap them and must not be added.
    struct Timings {
        double preprocess_host_ms{}, setup_host_ms{}, enqueue_host_ms{}, geometry_setup_host_ms{};
        double download_host_ms{}, wait_host_ms{};
        double preprocess_stream_ms{}, engine_stream_ms{};
        double association_stream_ms{}, gather_stream_ms{}, triangulation_stream_ms{};
        double association_host_ms{}, triangulation_host_ms{};
        double output_copy_stream_ms{}, download_stream_ms{};
    } timings;
    // Raw detector tensors are retained for camera-local 2-D output. Selected
    // observations remain the input to epipolar triangulation only.
    std::vector<float> keypoints;
    std::vector<float> keypoint_scores;
    std::vector<unsigned char> candidate_valid;
};

class TensorRtMultiviewEngine {
  public:
    explicit TensorRtMultiviewEngine(const PoseConfig&);
    ~TensorRtMultiviewEngine();
    TensorRtMultiviewEngine(const TensorRtMultiviewEngine&) = delete;
    TensorRtMultiviewEngine& operator=(const TensorRtMultiviewEngine&) = delete;

    void infer(const std::vector<const void*>& bgr_device,
               const std::vector<std::size_t>& strides,
               const std::vector<std::uint32_t>& widths,
               const std::vector<std::uint32_t>& heights,
               TensorRtMultiviewResult& result);

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace iris
