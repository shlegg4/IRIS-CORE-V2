#pragma once

#include <optional>

#include "da3/types.hpp"

namespace da3 {

class TensorRtEngine;

struct InferenceConfig {
    fs::path engine_path;
    std::vector<fs::path> image_paths;
    std::vector<int> camera_ids;
    fs::path out_dir;
    std::optional<fs::path> save_ply;
    bool viewer_align = true;
    std::string model_type{"base"};  // "large" or "base"
};

struct FrameDiagnostics {
    std::uint64_t sequence = 0;
    std::uint64_t batch_timestamp_ns = 0;
    std::vector<std::uint64_t> frame_capture_times_ns;
    std::vector<std::uint64_t> frame_ingest_times_ns;
    std::string source_id;
};

struct PreparedInferenceConfig {
    InputBatch batch;
    std::vector<int> camera_ids;
    fs::path out_dir;
    std::optional<fs::path> save_ply;
    bool viewer_align = true;
    double preprocess_ms = 0.0;
    std::optional<FrameDiagnostics> frame_diagnostics;
};

struct IntrinsicsExport {
    std::array<float, 9> K{};
    int width = 0;
    int height = 0;
};

struct ExtrinsicsExport {
    int camera_id = 0;
    std::array<float, 9> R{};
    std::array<float, 3> t{};
};

struct InferenceResult {
    fs::path cameras_path;
    fs::path intrinsics_dir;
    fs::path extrinsics_path;
    fs::path timings_path;
    fs::path ply_path;
    std::vector<int> camera_ids;
    std::vector<IntrinsicsExport> intrinsics;
    std::vector<ExtrinsicsExport> extrinsics;
    std::size_t point_count = 0;
    float confidence_threshold = 0.0f;
};

InferenceResult RunInference(const InferenceConfig& config);
InferenceResult RunInferenceBase(const InferenceConfig& config);
InferenceResult RunPreparedInference(
    const PreparedInferenceConfig& config,
    TensorRtEngine& engine
);
InferenceResult RunPreparedInferenceBase(
    const PreparedInferenceConfig& config,
    TensorRtEngine& engine
);

}  // namespace da3
