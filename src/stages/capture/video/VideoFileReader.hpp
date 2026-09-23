#pragma once

#include "iris/pipeline/Frame.hpp"
#include "iris/stages/capture/SynchronizedVideoConfig.hpp"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace iris::capture::video {

struct DecodedVideoFrame {
    Extent2D extent;
    // Populated for software decoding. NVDEC frames are converted directly into
    // the caller's CUDA BGR buffer and leave this empty.
    std::vector<std::uint8_t> bgr;
    std::chrono::nanoseconds presentation_time{};
    bool converted_on_gpu{};
};

class VideoFileReader {
  public:
    explicit VideoFileReader(const std::filesystem::path&, int cuda_device);
    ~VideoFileReader();
    VideoFileReader(const VideoFileReader&) = delete;
    VideoFileReader& operator=(const VideoFileReader&) = delete;

    [[nodiscard]] Extent2D extent() const noexcept;
    [[nodiscard]] double frame_rate() const noexcept;
    std::optional<DecodedVideoFrame> read_next(void* bgr_device, std::size_t bgr_stride_bytes,
                                              cudaStream_t stream);
    [[nodiscard]] bool hardware_decode_enabled() const noexcept;
    [[nodiscard]] VideoDecodeStatus decode_status(CameraId camera_id) const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace iris::capture::video
