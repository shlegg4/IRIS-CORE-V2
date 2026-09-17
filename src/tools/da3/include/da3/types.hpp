#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

namespace da3 {

namespace fs = std::filesystem;

inline constexpr int kBatchSize = 1;
inline int kNumViews = 4;
inline constexpr int kNumChannels = 3;
inline constexpr int kInputHeight = 504;
inline constexpr int kInputWidth = 280;
inline constexpr int kPatchSize = 14;
inline constexpr int kProcessRes = 504;

inline constexpr int kBaseInputHeight = 504;
inline constexpr int kBaseInputWidth = 504;

inline constexpr std::array<float, 3> kImageNetMean = {0.485f, 0.456f, 0.406f};
inline constexpr std::array<float, 3> kImageNetStd = {0.229f, 0.224f, 0.225f};

inline constexpr const char* kInputTensorName = "images";
inline constexpr std::array<const char*, 5> kOutputTensorNames = {
    "depth",
    "depth_conf",
    "sky",
    "intrinsics",
    "extrinsics",
};
inline constexpr std::array<const char*, 4> kBaseOutputTensorNames = {
    "depth",
    "depth_conf",
    "intrinsics",
    "extrinsics",
};

enum class PoseRefinementMode {
    kNone,
};

inline std::string ToString(PoseRefinementMode mode) {
    return mode == PoseRefinementMode::kNone ? "none" : "unknown";
}

struct PreprocessConfig {
    int process_res = kProcessRes;
    std::string process_res_method = "letterbox_fit";
    int patch_size = kPatchSize;
    int expected_height = kInputHeight;
    int expected_width = kInputWidth;
};

struct ProcessedView {
    fs::path image_path;
    cv::Mat rgb_u8;
    std::vector<float> chw;
    int source_height = 0;
    int source_width = 0;
    int height = 0;
    int width = 0;
    int content_x = 0;
    int content_y = 0;
    int content_width = 0;
    int content_height = 0;
    float scale_x = 1.0f;
    float scale_y = 1.0f;
};

struct InputBatch {
    std::vector<ProcessedView> views;
    std::vector<float> images;
};

struct TensorOutput {
    std::vector<float> values;
    std::vector<int64_t> dims;
};

struct Mv4Outputs {
    TensorOutput depth;
    TensorOutput depth_conf;
    TensorOutput sky;
    TensorOutput intrinsics;
    TensorOutput extrinsics;
};

struct MvBaseOutputs {
    TensorOutput depth;
    TensorOutput depth_conf;
    TensorOutput intrinsics;
    TensorOutput extrinsics;
};

struct ReconstructionConfig {
    float conf_thresh = 1.05f;
    float conf_thresh_percentile = 40.0f;
    float ensure_thresh_percentile = 90.0f;
    float sky_depth_percentile = 98.0f;
    bool viewer_align = true;
    PoseRefinementMode pose_refinement = PoseRefinementMode::kNone;
};

struct PointVertex {
    Eigen::Vector3f position = Eigen::Vector3f::Zero();
    std::array<std::uint8_t, 3> color = {0, 0, 0};
};

struct GroundAlignmentInfo {
    bool plane_found = false;
    std::size_t sampled_point_count = 0;
    std::size_t inlier_count = 0;
    float inlier_ratio = 0.0f;
    Eigen::Vector3f plane_normal = Eigen::Vector3f::UnitY();
    Eigen::Vector3f plane_point = Eigen::Vector3f::Zero();
    Eigen::Vector3f camera_centroid = Eigen::Vector3f::Zero();
    Eigen::Vector3f origin = Eigen::Vector3f::Zero();
};

struct ReconstructionResult {
    std::vector<PointVertex> vertices;
    Eigen::Matrix4f alignment = Eigen::Matrix4f::Identity();
    float applied_conf_threshold = 0.0f;
    GroundAlignmentInfo ground_alignment;
};

}  // namespace da3
