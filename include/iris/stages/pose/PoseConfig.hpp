#pragma once

#include "iris/pipeline/Frame.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace iris {

inline constexpr std::size_t pose_model_height = 256;
inline constexpr std::size_t pose_model_width = 256;

struct PoseConfig {
    // An empty path disables pose inference and preserves the historic pass-through behaviour.
    std::filesystem::path model_path;
    // The supplied PEAR artifact was exported on CPU. CUDA is supported only when the artifact
    // itself is compatible with the selected LibTorch CUDA runtime.
    std::string device{"cpu"};
};

} // namespace iris
