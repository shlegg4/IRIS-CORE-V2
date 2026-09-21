#include "stages/capture/decoding/GpuDecoder.hpp"
#include <cuda_runtime_api.h>
#include <chrono>
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
    try {
        const auto check = [](nvjpegStatus_t status) {
            if (status != NVJPEG_STATUS_SUCCESS)
                throw std::runtime_error("nvJPEG GPU-hybrid initialisation failed: " + std::to_string(status));
        };
        check(nvjpegCreateSimple(&handle_));
        check(nvjpegDecoderCreate(handle_, NVJPEG_BACKEND_GPU_HYBRID, &decoder_));
        check(nvjpegDecoderStateCreate(handle_, decoder_, &state_));
        check(nvjpegJpegStreamCreate(handle_, &jpeg_));
        check(nvjpegDecodeParamsCreate(handle_, &params_));
        check(nvjpegDecodeParamsSetOutputFormat(params_, NVJPEG_OUTPUT_BGRI));
        check(nvjpegBufferPinnedCreate(handle_, nullptr, &pinned_));
        check(nvjpegBufferDeviceCreate(handle_, nullptr, &scratch_));
        check(nvjpegStateAttachPinnedBuffer(state_, pinned_));
        check(nvjpegStateAttachDeviceBuffer(state_, scratch_));
        infrastructure::gpu::check_cuda(cudaEventCreate(&decode_begin_), "decode timing begin");
        infrastructure::gpu::check_cuda(cudaEventCreate(&decode_end_), "decode timing end");
    } catch (...) {
        release_decoder();
        throw;
    }
}
GpuDecoder::~GpuDecoder() {
    cudaSetDevice(device_);
    cudaStreamSynchronize(stream_.get());
    release_decoder();
}
void GpuDecoder::release_decoder() noexcept {
    if (decode_begin_) cudaEventDestroy(decode_begin_);
    if (decode_end_) cudaEventDestroy(decode_end_);
    if (state_) {
        nvjpegJpegStateDestroy(state_);
    }
    if (pinned_) nvjpegBufferPinnedDestroy(pinned_);
    if (scratch_) nvjpegBufferDeviceDestroy(scratch_);
    if (params_) nvjpegDecodeParamsDestroy(params_);
    if (jpeg_) nvjpegJpegStreamDestroy(jpeg_);
    if (decoder_) nvjpegDecoderDestroy(decoder_);
    if (handle_) {
        nvjpegDestroy(handle_);
    }
}
DecodeResult GpuDecoder::decode(const CaptureSample& s, FrameTiming timing) {
    infrastructure::gpu::check_cuda(cudaSetDevice(device_), "decode cudaSetDevice");
    auto buffer = pool_.acquire();
    if (!buffer) {
        return {{}, DecodeFailureReason::PoolExhausted};
    }
    timing.decode_submit_time = std::chrono::steady_clock::now();
    DecodeResult result;
    const auto gpu_begin = std::chrono::steady_clock::now();
    if (s.format == PixelFormat::Mjpeg) {
        const auto header_begin = std::chrono::steady_clock::now();
        int components{}, widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
        nvjpegChromaSubsampling_t sub;
        const auto header_status = nvjpegGetImageInfo(handle_, s.bytes.data(), s.bytes.size(),
                                                      &components, &sub, widths, heights);
        const auto header_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - header_begin)
                                   .count();
        if (header_status != NVJPEG_STATUS_SUCCESS) {
            return {{}, DecodeFailureReason::InvalidJpegHeader, header_status};
        }
        if (widths[0] != static_cast<int>(extent_.width) ||
            heights[0] != static_cast<int>(extent_.height)) {
            return {{}, DecodeFailureReason::DimensionMismatch};
        }
        result.header_parse_ms = header_ms;
        nvjpegImage_t image{};
        image.channel[0] = static_cast<unsigned char*>(buffer->data);
        image.pitch[0] = static_cast<unsigned int>(buffer->stride_bytes);
        infrastructure::gpu::check_cuda(cudaEventRecord(decode_begin_, stream_.get()),
                                        "cudaEventRecord(begin)");
        const auto host_decode_begin = std::chrono::steady_clock::now();
        auto decode_status = nvjpegJpegStreamParse(handle_, s.bytes.data(), s.bytes.size(), 0, 0, jpeg_);
        int unsupported{};
        if (decode_status == NVJPEG_STATUS_SUCCESS)
            decode_status = nvjpegDecoderJpegSupported(decoder_, jpeg_, params_, &unsupported);
        if (decode_status == NVJPEG_STATUS_SUCCESS && unsupported)
            decode_status = NVJPEG_STATUS_JPEG_NOT_SUPPORTED;
        if (decode_status == NVJPEG_STATUS_SUCCESS)
            decode_status = nvjpegDecodeJpegHost(handle_, decoder_, state_, params_, jpeg_);
        if (decode_status == NVJPEG_STATUS_SUCCESS)
            decode_status = nvjpegDecodeJpegTransferToDevice(handle_, decoder_, state_, jpeg_, stream_.get());
        if (decode_status == NVJPEG_STATUS_SUCCESS)
            decode_status = nvjpegDecodeJpegDevice(handle_, decoder_, state_, &image, stream_.get());
        result.nvjpeg_host_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - host_decode_begin)
                                    .count();
        infrastructure::gpu::check_cuda(cudaEventRecord(decode_end_, stream_.get()),
                                        "cudaEventRecord(end)");
        // Keep state, bitstream, and attached scratch alive until all decode work finishes.
        infrastructure::gpu::check_cuda(cudaEventSynchronize(decode_end_),
                                        "cudaEventSynchronize(decode)");
        float device_ms{};
        // Stream interval includes host submission gaps; this is not isolated kernel time.
        infrastructure::gpu::check_cuda(cudaEventElapsedTime(&device_ms, decode_begin_, decode_end_),
                                        "cudaEventElapsedTime(decode)");
        result.nvjpeg_device_ms = device_ms;
        if (decode_status != NVJPEG_STATUS_SUCCESS) {
            return {{}, DecodeFailureReason::NvjpegDecode, decode_status};
        }
        result.gpu_submit_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - gpu_begin)
                                   .count();
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
    result.gpu_submit_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - gpu_begin)
                               .count();
    GpuBuffer output = std::move(*buffer);
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
