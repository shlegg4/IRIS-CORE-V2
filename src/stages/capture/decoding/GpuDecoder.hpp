#pragma once
#include "iris/infrastructure/gpu/CudaResources.hpp"
#include "iris/infrastructure/gpu/FramePool.hpp"
#include "iris/stages/capture/CaptureConfig.hpp"
#include "stages/capture/CaptureSample.hpp"
#include <nvjpeg.h>
namespace iris::capture {
enum class DecodeFailureReason {
    None,
    PoolExhausted,
    InvalidJpegHeader,
    DimensionMismatch,
    NvjpegDecode,
    UnsupportedFormat,
    RotationPoolExhausted
};

struct DecodeResult {
    std::optional<Frame> frame;
    DecodeFailureReason failure{DecodeFailureReason::None};
    nvjpegStatus_t nvjpeg_status{NVJPEG_STATUS_SUCCESS};
    bool rotation_applied{};
    double rotation_submit_ms{};
    double header_parse_ms{};
    double gpu_submit_ms{};
    double nvjpeg_host_ms{};
    double nvjpeg_device_ms{};
};

class GpuDecoder {
  public:
    GpuDecoder(int device, Extent2D extent, std::size_t pool_capacity, FrameRotation rotation);
    ~GpuDecoder();
    DecodeResult decode(const CaptureSample&, FrameTiming);
    std::size_t pool_available() const;
    std::size_t rotation_pool_available() const;

  private:
    void release_decoder() noexcept;
    int device_;
    Extent2D extent_;
    Extent2D output_extent_;
    FrameRotation rotation_;
    infrastructure::gpu::CudaStream stream_;
    infrastructure::gpu::FramePool pool_;
    std::optional<infrastructure::gpu::FramePool> rotation_pool_;
    nvjpegHandle_t handle_{};
    nvjpegJpegState_t state_{};
    nvjpegJpegDecoder_t decoder_{};
    nvjpegJpegStream_t jpeg_{};
    nvjpegDecodeParams_t params_{};
    nvjpegBufferPinned_t pinned_{};
    nvjpegBufferDevice_t scratch_{};
    cudaEvent_t decode_begin_{}, decode_end_{};
};
void launch_yuy2_to_bgr(const std::uint8_t*, std::size_t, void*, std::size_t, std::uint32_t,
                        std::uint32_t, cudaStream_t);
void launch_bgr_rotation(const void*, std::size_t, void*, std::size_t, std::uint32_t, std::uint32_t,
                         FrameRotation, cudaStream_t);
} // namespace iris::capture
