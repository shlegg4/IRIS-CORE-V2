#pragma once
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
namespace iris::infrastructure::gpu {
class CudaEvent;
}
namespace iris {
using CameraId = std::uint32_t;
using MonotonicTime = std::chrono::steady_clock::time_point;
enum class PixelFormat { Unknown, Mjpeg, Yuy2, Bgra8, Bgr8 };
enum class ClockQuality { Uninitialised, WarmingUp, Stable, Degraded };
struct Extent2D {
    std::uint32_t width{};
    std::uint32_t height{};
};
struct FrameTiming {
    std::chrono::nanoseconds source_time{};
    MonotonicTime estimated_capture_time{};
    MonotonicTime host_arrival_time{};
    MonotonicTime decode_submit_time{};
    MonotonicTime ready_time{};
    ClockQuality clock_quality{ClockQuality::Uninitialised};
};
struct GpuBuffer {
    void* data{};
    std::size_t stride_bytes{}, size_bytes{};
    int device_id{};
    std::shared_ptr<void> owner;
};
struct Frame {
    CameraId camera{};
    std::uint64_t sequence{};
    Extent2D extent{};
    PixelFormat format{PixelFormat::Unknown};
    GpuBuffer buffer;
    std::shared_ptr<infrastructure::gpu::CudaEvent> ready;
    FrameTiming timing;
};
} // namespace iris
