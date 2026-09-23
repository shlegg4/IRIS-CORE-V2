#pragma once

#include "iris/stages/capture/CaptureConfig.hpp"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace iris {

struct VideoCameraInput {
    CameraId camera_id{};
    std::filesystem::path path;
    FrameRotation rotation{FrameRotation::None};
};

struct VideoDecodeStatus {
    CameraId camera_id{};
    std::string codec;
    std::string backend;
    std::string detail;
};

struct SynchronizedVideoConfig {
    std::vector<VideoCameraInput> cameras;
    int cuda_device{};
    std::size_t frame_pool_capacity{8};
    bool realtime{};
    bool loop{};
};

} // namespace iris
