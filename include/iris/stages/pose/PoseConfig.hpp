#pragma once

#include "iris/pipeline/Frame.hpp"

#include <cstddef>
#include <filesystem>
#include <string>
#include <array>
#include <memory>
#include <stdexcept>

namespace iris {
class CalibrationStore;

inline constexpr std::size_t pose_model_height = 256;
inline constexpr std::size_t pose_model_width = 256;

struct PoseConfig {
    // An empty path disables pose inference and preserves the historic pass-through behaviour.
    std::filesystem::path model_path;
    // The supplied PEAR artifact was exported on CPU. CUDA is supported only when the artifact
    // itself is compatible with the selected LibTorch CUDA runtime.
    std::string device{"cpu"};

    // Optional RTMO-S detector + CPU triangulation backend. It is selected
    // instead of the monocular backend and currently requires three cameras.
    std::filesystem::path multiview_engine_path;
    std::filesystem::path multiview_calibration_path;
    // Run the RTMO engine on a single camera and emit only its 2-D keypoints.
    // The fixed-batch engine is fed three copies internally; no rig calibration
    // or triangulation is required.
    bool two_d_only{};
    struct CameraCalibration {
        CameraId camera_id{};
        std::array<float, 9> R_w2c{};
        std::array<float, 3> t_w2c{};
        std::array<float, 9> intrinsics{};
        std::array<float, 5> distortion{};
        // The software rotation applied to the source before inference. Intrinsics and
        // extrinsics are transformed into this rotated image coordinate system.
        Extent2D source_extent{};
        int image_rotation_degrees{};
        bool calibrated{false};
    };
    std::array<CameraCalibration, 3> multiview_calibration{};
    std::shared_ptr<CalibrationStore> calibration_store;
};

inline void apply_capture_rotation(PoseConfig::CameraCalibration& calibration) {
    const auto degrees = calibration.image_rotation_degrees;
    if (degrees == 0) return;
    if (!calibration.source_extent.width || !calibration.source_extent.height)
        throw std::invalid_argument("rotated multiview capture requires a source frame extent");

    const auto width = static_cast<float>(calibration.source_extent.width - 1);
    const auto height = static_cast<float>(calibration.source_extent.height - 1);
    const auto source_k = calibration.intrinsics;
    const auto source_r = calibration.R_w2c;
    const auto source_t = calibration.t_w2c;
    const auto source_distortion = calibration.distortion;
    std::array<float, 9> image_rotation{};

    switch (degrees) {
    case 90: // Clockwise image rotation.
        calibration.intrinsics = {source_k[4], 0.0F, height - source_k[5], 0.0F, source_k[0], source_k[2], 0.0F, 0.0F, 1.0F};
        image_rotation = {0.0F, -1.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F};
        calibration.distortion[2] = source_distortion[3];
        calibration.distortion[3] = -source_distortion[2];
        break;
    case 180:
        calibration.intrinsics = {source_k[0], 0.0F, width - source_k[2], 0.0F, source_k[4], height - source_k[5], 0.0F, 0.0F, 1.0F};
        image_rotation = {-1.0F, 0.0F, 0.0F, 0.0F, -1.0F, 0.0F, 0.0F, 0.0F, 1.0F};
        calibration.distortion[2] = -source_distortion[2];
        calibration.distortion[3] = -source_distortion[3];
        break;
    case 270: // Counter-clockwise image rotation.
        calibration.intrinsics = {source_k[4], 0.0F, source_k[5], 0.0F, source_k[0], width - source_k[2], 0.0F, 0.0F, 1.0F};
        image_rotation = {0.0F, 1.0F, 0.0F, -1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F};
        calibration.distortion[2] = -source_distortion[3];
        calibration.distortion[3] = source_distortion[2];
        break;
    default:
        throw std::invalid_argument("unsupported multiview image rotation");
    }
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            calibration.R_w2c[row * 3 + column] = image_rotation[row * 3] * source_r[column] + image_rotation[row * 3 + 1] * source_r[3 + column] + image_rotation[row * 3 + 2] * source_r[6 + column];
        }
        calibration.t_w2c[row] = image_rotation[row * 3] * source_t[0] + image_rotation[row * 3 + 1] * source_t[1] + image_rotation[row * 3 + 2] * source_t[2];
    }
}

inline const char* pose_backend_name(const PoseConfig& config) noexcept {
    if (!config.multiview_engine_path.empty()) return config.two_d_only ? "2d" : "multiview";
    if (!config.model_path.empty()) return "monocular";
    return "off";
}

} // namespace iris
