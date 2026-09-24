#pragma once

#include "iris/stages/pose/PoseConfig.hpp"

#include <cstddef>
#include <cstdint>
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
        double mapping_stream_ms{};
        double temporal_assignment_stream_ms{};
        double postprocess_gpu_stream_ms{};
        double temporal_stream_ms{}, temporal_host_ms{};
        double output_copy_stream_ms{}, download_stream_ms{};
        bool cuda_graph_active{};
    } timings;
    // Calibrated 2-D packet points and selected track state copied from CUDA.
    std::vector<float> image_keypoints;
    std::vector<unsigned char> image_joint_valid;
    std::vector<float> keypoint_scores;
    std::vector<unsigned char> candidate_valid;
    std::vector<unsigned char> assignments;
    std::uint32_t track_count{};
    std::vector<std::uint64_t> track_ids;
    std::vector<float> tracked_xyz;
    std::vector<unsigned char> tracked_valid;
    std::vector<unsigned char> tracked_predicted;
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
               const std::vector<float>& fundamentals,
               const std::vector<float>& projections,
               float dt_seconds, float maximum_reprojection_error_px,
               TensorRtMultiviewResult& result);

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace iris
