#pragma once

#include "iris/pipeline/Frame.hpp"

#include <cstddef>
#include <filesystem>
#include <string>
#include <array>

namespace iris {

inline constexpr std::size_t pose_model_height = 256;
inline constexpr std::size_t pose_model_width = 256;

struct PoseConfig {
    // An empty path disables pose inference and preserves the historic pass-through behaviour.
    std::filesystem::path model_path;
    // The supplied PEAR artifact was exported on CPU. CUDA is supported only when the artifact
    // itself is compatible with the selected LibTorch CUDA runtime.
    std::string device{"cpu"};

    // Optional fixed-shape RTMO-S + epipolar + triangulation backend. It is
    // selected instead of the monocular backend and requires three cameras.
    std::filesystem::path multiview_engine_path;
    std::filesystem::path multiview_calibration_path;
    struct CameraCalibration {
        CameraId camera_id{};
        std::array<float, 9> R_w2c{};
        std::array<float, 3> t_w2c{};
        std::array<float, 9> intrinsics{};
        std::array<float, 5> distortion{};
        bool calibrated{false};
    };
    std::array<CameraCalibration, 3> multiview_calibration{};
};

inline const char* pose_backend_name(const PoseConfig& config) noexcept {
    if (!config.multiview_engine_path.empty()) return "multiview";
    if (!config.model_path.empty()) return "monocular";
    return "off";
}

} // namespace iris
