#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

namespace iris {

struct SharedMemoryOutputConfig {
    bool enabled{false};
    std::string destination{"Local\\IRIS_V2_Output"};
    std::size_t capacity_bytes{64U * 1024U * 1024U};
    // Publish the original mapping at destination as well as the versioned v2 mapping.
    bool legacy_v1{true};
};

struct MjpegPreviewConfig {
    bool enabled{false};
    std::uint32_t max_fps{10};
    std::uint32_t max_width{960};
    std::uint32_t jpeg_quality{75};
    std::size_t queue_capacity{2};
};

struct HttpPreviewConfig {
    bool enabled{false};
    std::string bind_address{"127.0.0.1"};
    std::uint16_t port{8080};
    std::size_t queue_capacity{16};
};

struct PreviewConfig {
    SharedMemoryOutputConfig shared_memory;
    MjpegPreviewConfig mjpeg;
    HttpPreviewConfig http;
    std::size_t shared_memory_queue_capacity{2};
};

struct DiskOutputConfig {
    std::filesystem::path destination{"recordings/iris-recording.mp4"};
    std::size_t queue_capacity{120};
    std::uint32_t bitrate{8'000'000};
    std::uint32_t frame_rate{30};
    std::size_t surface_count{8};
};

struct OutputConfig {
    SharedMemoryOutputConfig shared_memory;
    PreviewConfig preview;
    DiskOutputConfig disk;
    std::size_t shared_memory_queue_capacity{2};
    std::size_t camera_count{1};
};

} // namespace iris
