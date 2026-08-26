#pragma once
#include "iris/pipeline/Frame.hpp"
#include <cstdint>
#include <vector>
namespace iris::capture {
struct CaptureSample {
    CameraId camera{};
    std::uint64_t sequence{};
    std::vector<std::uint8_t> bytes;
    PixelFormat format{PixelFormat::Unknown};
    Extent2D extent{};
    std::chrono::nanoseconds source_timestamp{};
    MonotonicTime host_arrival{};
    bool discontinuity{};
};
} // namespace iris::capture
