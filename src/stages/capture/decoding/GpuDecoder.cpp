#include "stages/capture/decoding/GpuDecoder.hpp"
#include <stdexcept>
namespace iris::capture {
GpuDecoder::GpuDecoder(int device, Extent2D extent, std::size_t capacity, FrameRotation rotation)
    : device_(device), extent_(extent),
      output_extent_(swaps_axes(rotation) ? Extent2D{extent.height, extent.width} : extent),
      rotation_(rotation), pool_(device, extent, 3, capacity) {
    infrastructure::gpu::check_cuda(cudaSetDevice(device), "cudaSetDevice");
    if (rotation_ != FrameRotation::None) {
        rotation_pool_.emplace(device, output_extent_, 3, capacity);
    }
    if (nvjpegCreateSimple(&handle_) != NVJPEG_STATUS_SUCCESS ||
        nvjpegJpegStateCreate(handle_, &state_) != NVJPEG_STATUS_SUCCESS) {
        throw std::runtime_error("nvJPEG initialisation failed");
    }
}
GpuDecoder::~GpuDecoder() {
    if (state_) {
        nvjpegJpegStateDestroy(state_);
    }
    if (handle_) {
        nvjpegDestroy(handle_);
    }
}
DecodeResult GpuDecoder::decode(const CaptureSample& s, FrameTiming timing) {
    auto buffer = pool_.acquire();
    if (!buffer) {
        return {{}, DecodeFailureReason::PoolExhausted};
    }
    timing.decode_submit_time = std::chrono::steady_clock::now();
    if (s.format == PixelFormat::Mjpeg) {
        int components{}, widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
        nvjpegChromaSubsampling_t sub;
        const auto header_status = nvjpegGetImageInfo(handle_, s.bytes.data(), s.bytes.size(),
                                                      &components, &sub, widths, heights);
        if (header_status != NVJPEG_STATUS_SUCCESS) {
            return {{}, DecodeFailureReason::InvalidJpegHeader, header_status};
        }
        if (widths[0] != static_cast<int>(extent_.width) ||
            heights[0] != static_cast<int>(extent_.height)) {
            return {{}, DecodeFailureReason::DimensionMismatch};
        }
        nvjpegImage_t image{};
        image.channel[0] = static_cast<unsigned char*>(buffer->data);
        image.pitch[0] = static_cast<unsigned int>(buffer->stride_bytes);
        const auto decode_status = nvjpegDecode(handle_, state_, s.bytes.data(), s.bytes.size(),
                                                NVJPEG_OUTPUT_BGRI, &image, stream_.get());
        if (decode_status != NVJPEG_STATUS_SUCCESS) {
            return {{}, DecodeFailureReason::NvjpegDecode, decode_status};
        }
    } else if (s.format == PixelFormat::Yuy2) {
        void* compressed{};
        infrastructure::gpu::check_cuda(cudaMallocAsync(&compressed, s.bytes.size(), stream_.get()),
                                        "cudaMallocAsync");
        infrastructure::gpu::check_cuda(cudaMemcpyAsync(compressed, s.bytes.data(), s.bytes.size(),
                                                        cudaMemcpyHostToDevice, stream_.get()),
                                        "cudaMemcpyAsync");
        launch_yuy2_to_bgr(static_cast<const std::uint8_t*>(compressed), extent_.width * 2,
                           buffer->data, buffer->stride_bytes, extent_.width, extent_.height,
                           stream_.get());
        cudaFreeAsync(compressed, stream_.get());
    } else if (s.format == PixelFormat::Bgra8) {
        void* raw{};
        infrastructure::gpu::check_cuda(cudaMallocAsync(&raw, s.bytes.size(), stream_.get()),
                                        "cudaMallocAsync");
        infrastructure::gpu::check_cuda(cudaMemcpyAsync(raw, s.bytes.data(), s.bytes.size(),
                                                        cudaMemcpyHostToDevice, stream_.get()),
                                        "cudaMemcpyAsync");
        infrastructure::gpu::check_cuda(cudaMemcpy2DAsync(buffer->data, buffer->stride_bytes, raw,
                                                          extent_.width * 4, extent_.width * 3,
                                                          extent_.height, cudaMemcpyDeviceToDevice,
                                                          stream_.get()),
                                        "cudaMemcpy2DAsync");
        cudaFreeAsync(raw, stream_.get());
    } else {
        return {{}, DecodeFailureReason::UnsupportedFormat};
    }
    GpuBuffer output = std::move(*buffer);
    DecodeResult result;
    if (rotation_ != FrameRotation::None) {
        const auto rotation_begin = std::chrono::steady_clock::now();
        auto rotated = rotation_pool_->acquire();
        if (!rotated) {
            return {{}, DecodeFailureReason::RotationPoolExhausted};
        }
        launch_bgr_rotation(output.data, output.stride_bytes, rotated->data, rotated->stride_bytes,
                            extent_.width, extent_.height, rotation_, stream_.get());
        output = std::move(*rotated);
        result.rotation_applied = true;
        result.rotation_submit_ms = std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - rotation_begin)
                                        .count();
    }
    auto event = std::make_shared<infrastructure::gpu::CudaEvent>();
    event->record(stream_.get());
    timing.ready_time = std::chrono::steady_clock::now();
    result.frame = Frame{s.camera,          s.sequence,       output_extent_, PixelFormat::Bgr8,
                         std::move(output), std::move(event), timing};
    return result;
}
std::size_t GpuDecoder::pool_available() const { return pool_.available(); }
std::size_t GpuDecoder::rotation_pool_available() const {
    return rotation_pool_ ? rotation_pool_->available() : pool_.available();
}
} // namespace iris::capture
