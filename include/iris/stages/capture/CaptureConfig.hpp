#pragma once
#include "iris/pipeline/Frame.hpp"
#include "iris/pipeline/OverflowPolicy.hpp"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
namespace iris {
enum class FrameRotation { None, Clockwise90, Rotate180, CounterClockwise90 };

constexpr bool swaps_axes(FrameRotation rotation) noexcept {
    return rotation == FrameRotation::Clockwise90 || rotation == FrameRotation::CounterClockwise90;
}

constexpr int rotation_degrees(FrameRotation rotation) noexcept {
    switch (rotation) {
    case FrameRotation::None:
        return 0;
    case FrameRotation::Clockwise90:
        return 90;
    case FrameRotation::Rotate180:
        return 180;
    case FrameRotation::CounterClockwise90:
        return 270;
    }
    return 0;
}

struct FrameRate {
    std::uint32_t numerator{30}, denominator{1};
    double value() const {
        return denominator ? static_cast<double>(numerator) / denominator : 0.0;
    }
};
struct CaptureConfig {
    std::string device_symbolic_link;
    std::optional<std::uint32_t> device_index{0};
    Extent2D extent{1920, 1080};
    FrameRate frame_rate{};
    PixelFormat format{PixelFormat::Mjpeg};
    int cuda_device{};
    std::size_t sample_queue_capacity{1};
    std::size_t frame_pool_capacity{4};
    OverflowPolicy overflow{OverflowPolicy::DropOldest};
    FrameRotation rotation{FrameRotation::None};
    bool allow_format_fallback{false};
    bool reconnect{true};
};

struct CameraCaptureConfig {
    CameraId camera_id{};
    CaptureConfig capture;
};

enum class IncompleteBatchPolicy { DropBatch, EmitPartial };

struct MultiCameraCaptureConfig {
    std::vector<CameraCaptureConfig> cameras;
    std::chrono::milliseconds sync_tolerance{20};
    std::size_t sync_queue_capacity{4};
    IncompleteBatchPolicy incomplete_batch_policy{IncompleteBatchPolicy::DropBatch};
};
} // namespace iris
