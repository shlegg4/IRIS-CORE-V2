#include "stages/output/disk/GpuMp4Writer.hpp"

#include "iris/infrastructure/gpu/CudaResources.hpp"
#include "stages/output/disk/BgrToNv12.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/buffer.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/opt.h>
}

#include <array>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>

namespace iris::output::disk {
namespace {

std::string ffmpeg_error(int result) {
    std::array<char, AV_ERROR_MAX_STRING_SIZE> buffer{};
    av_strerror(result, buffer.data(), buffer.size());
    return buffer.data();
}

void check_ffmpeg(int result, const char* operation) {
    if (result < 0) {
        throw std::runtime_error(std::string(operation) + ": " + ffmpeg_error(result));
    }
}

class Nv12SurfacePool {
  public:
    Nv12SurfacePool(std::size_t count, std::size_t size) {
        if (count == 0) {
            throw std::invalid_argument("NVENC surface count must be greater than zero");
        }
        try {
            for (std::size_t index = 0; index < count; ++index) {
                std::uint8_t* surface = nullptr;
                infrastructure::gpu::check_cuda(
                    cudaMalloc(reinterpret_cast<void**>(&surface), size),
                    "cudaMalloc NV12 surface pool");
                allocated_.push_back(surface);
                free_.push(surface);
            }
        } catch (...) {
            for (auto* surface : allocated_) {
                cudaFree(surface);
            }
            throw;
        }
    }

    ~Nv12SurfacePool() {
        for (auto* surface : allocated_) {
            cudaFree(surface);
        }
    }

    std::uint8_t* acquire() {
        std::unique_lock lock(mutex_);
        available_.wait(lock, [this] { return !free_.empty(); });
        auto* result = free_.front();
        free_.pop();
        return result;
    }

    void release(std::uint8_t* surface) {
        {
            std::scoped_lock lock(mutex_);
            free_.push(surface);
        }
        available_.notify_one();
    }

  private:
    std::vector<std::uint8_t*> allocated_;
    std::queue<std::uint8_t*> free_;
    std::mutex mutex_;
    std::condition_variable available_;
};

struct SurfaceLease {
    std::shared_ptr<Nv12SurfacePool> pool;
};

void release_surface(void* opaque, std::uint8_t* data) {
    std::unique_ptr<SurfaceLease> lease(static_cast<SurfaceLease*>(opaque));
    lease->pool->release(data);
}

std::filesystem::path partial_path_for(const std::filesystem::path& destination) {
    return destination.string() + ".partial";
}

} // namespace

class GpuMp4Writer::Impl {
  public:
    Impl(const DiskOutputConfig& config, const Frame& first_frame)
        : destination_(config.destination), partial_(partial_path_for(destination_)),
          width_(first_frame.extent.width), height_(first_frame.extent.height),
          frame_rate_(config.frame_rate),
          first_timestamp_(first_frame.timing.estimated_capture_time) {
        if (first_frame.format != PixelFormat::Bgr8) {
            throw std::invalid_argument("MP4 writer requires BGR8 GPU frames");
        }
        if (width_ == 0 || height_ == 0 || (width_ & 1U) != 0 || (height_ & 1U) != 0) {
            throw std::invalid_argument("NV12 encoding requires non-zero even frame dimensions");
        }
        if (frame_rate_ == 0 || config.bitrate == 0) {
            throw std::invalid_argument("MP4 frame rate and bitrate must be greater than zero");
        }
        if (destination_.extension() != ".mp4") {
            throw std::invalid_argument("disk recording destination must end in .mp4");
        }
        if (std::filesystem::exists(destination_) || std::filesystem::exists(partial_)) {
            throw std::runtime_error("recording destination or partial file already exists");
        }
        if (const auto parent = destination_.parent_path(); !parent.empty()) {
            std::filesystem::create_directories(parent);
        }
        const auto surface_size = static_cast<std::size_t>(width_) * height_ * 3U / 2U;
        surfaces_ = std::make_shared<Nv12SurfacePool>(config.surface_count, surface_size);
        initialize(config.bitrate);
    }

    ~Impl() { cleanup(); }

    std::uint64_t write(const Frame& frame) {
        if (finalized_) {
            throw std::runtime_error("cannot write to a finalized MP4 recording");
        }
        if (frame.format != PixelFormat::Bgr8 || frame.extent.width != width_ ||
            frame.extent.height != height_) {
            throw std::runtime_error("frame format or dimensions changed during recording");
        }
        if (!frame.buffer.data || frame.buffer.stride_bytes < width_ * 3ULL) {
            throw std::runtime_error("frame has an invalid CUDA BGR buffer");
        }

        if (frame.ready) {
            frame.ready->wait(stream_.get());
        }

        std::uint8_t* nv12 = surfaces_->acquire();
        const auto surface_size = static_cast<std::size_t>(width_) * height_ * 3U / 2U;
        try {
            convert_bgr_to_nv12(static_cast<const std::uint8_t*>(frame.buffer.data),
                                frame.buffer.stride_bytes, nv12, width_, width_, height_,
                                stream_.get());
            infrastructure::gpu::check_cuda(cudaGetLastError(), "BGR to NV12 kernel");
            infrastructure::gpu::check_cuda(cudaStreamSynchronize(stream_.get()),
                                            "NV12 conversion synchronization");

            AVFrame* encoded_frame = av_frame_alloc();
            if (!encoded_frame) {
                throw std::bad_alloc();
            }
            encoded_frame->format = AV_PIX_FMT_CUDA;
            encoded_frame->width = static_cast<int>(width_);
            encoded_frame->height = static_cast<int>(height_);
            encoded_frame->hw_frames_ctx = av_buffer_ref(hw_frames_);
            encoded_frame->data[0] = nv12;
            encoded_frame->data[1] = nv12 + (static_cast<std::size_t>(width_) * height_);
            encoded_frame->linesize[0] = static_cast<int>(width_);
            encoded_frame->linesize[1] = static_cast<int>(width_);
            auto lease = std::make_unique<SurfaceLease>();
            lease->pool = surfaces_;
            encoded_frame->buf[0] =
                av_buffer_create(nv12, surface_size, release_surface, lease.get(), 0);
            if (encoded_frame->buf[0]) {
                lease.release();
                nv12 = nullptr;
            }
            if (!encoded_frame->hw_frames_ctx || !encoded_frame->buf[0]) {
                av_frame_free(&encoded_frame);
                throw std::bad_alloc();
            }

            const auto elapsed = frame.timing.estimated_capture_time - first_timestamp_;
            auto pts = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
            if (pts < 0 || (submitted_frames_ != 0 && pts <= last_pts_)) {
                av_frame_free(&encoded_frame);
                throw std::runtime_error("capture timestamps are not strictly monotonic");
            }
            encoded_frame->pts = pts;
            last_pts_ = pts;

            int result = avcodec_send_frame(codec_, encoded_frame);
            if (result == AVERROR(EAGAIN)) {
                drain(false);
                result = avcodec_send_frame(codec_, encoded_frame);
            }
            av_frame_free(&encoded_frame);
            check_ffmpeg(result, "avcodec_send_frame");
            ++submitted_frames_;
            return drain(false);
        } catch (...) {
            if (nv12) {
                surfaces_->release(nv12);
            }
            throw;
        }
    }

    void finalize() {
        if (finalized_) {
            return;
        }
        check_ffmpeg(avcodec_send_frame(codec_, nullptr), "flush NVENC");
        drain(true);
        check_ffmpeg(av_write_trailer(format_), "write MP4 trailer");
        if (format_->pb) {
            check_ffmpeg(avio_closep(&format_->pb), "close MP4 output");
        }
        std::filesystem::rename(partial_, destination_);
        finalized_ = true;
    }

    std::uint64_t encoded_frames() const noexcept { return encoded_frames_; }
    std::uint64_t bytes_written() const noexcept { return bytes_written_; }

  private:
    void initialize(std::uint32_t bitrate) {
        infrastructure::gpu::check_cuda(cudaFree(nullptr), "initialize CUDA context");
        CUcontext cuda_context{};
        if (cuCtxGetCurrent(&cuda_context) != CUDA_SUCCESS || !cuda_context) {
            throw std::runtime_error("no current CUDA context for NVENC");
        }

        check_ffmpeg(
            avformat_alloc_output_context2(&format_, nullptr, "mp4", partial_.string().c_str()),
            "allocate MP4 context");
        const AVCodec* encoder = avcodec_find_encoder_by_name("h264_nvenc");
        if (!encoder) {
            throw std::runtime_error("pinned FFmpeg runtime does not provide h264_nvenc");
        }
        video_stream_ = avformat_new_stream(format_, nullptr);
        codec_ = avcodec_alloc_context3(encoder);
        if (!video_stream_ || !codec_) {
            throw std::bad_alloc();
        }

        hw_device_ = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_CUDA);
        if (!hw_device_) {
            throw std::runtime_error("could not allocate FFmpeg CUDA device context");
        }
        auto* device = reinterpret_cast<AVHWDeviceContext*>(hw_device_->data);
        auto* cuda_device = reinterpret_cast<AVCUDADeviceContext*>(device->hwctx);
        cuda_device->cuda_ctx = cuda_context;
        check_ffmpeg(av_hwdevice_ctx_init(hw_device_), "initialize FFmpeg CUDA device");

        hw_frames_ = av_hwframe_ctx_alloc(hw_device_);
        if (!hw_frames_) {
            throw std::runtime_error("could not allocate FFmpeg CUDA frame context");
        }
        auto* frames = reinterpret_cast<AVHWFramesContext*>(hw_frames_->data);
        frames->format = AV_PIX_FMT_CUDA;
        frames->sw_format = AV_PIX_FMT_NV12;
        frames->width = static_cast<int>(width_);
        frames->height = static_cast<int>(height_);
        frames->initial_pool_size = 8;
        check_ffmpeg(av_hwframe_ctx_init(hw_frames_), "initialize FFmpeg CUDA frames");

        codec_->width = static_cast<int>(width_);
        codec_->height = static_cast<int>(height_);
        codec_->time_base = AVRational{1, 1'000'000};
        codec_->framerate = AVRational{static_cast<int>(frame_rate_), 1};
        codec_->pix_fmt = AV_PIX_FMT_CUDA;
        codec_->hw_frames_ctx = av_buffer_ref(hw_frames_);
        codec_->bit_rate = bitrate;
        codec_->gop_size = static_cast<int>(frame_rate_);
        codec_->max_b_frames = 0;
        codec_->flags |= AV_CODEC_FLAG_LOW_DELAY;
        if ((format_->oformat->flags & AVFMT_GLOBALHEADER) != 0) {
            codec_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }
        av_opt_set(codec_->priv_data, "preset", "p1", 0);
        av_opt_set(codec_->priv_data, "tune", "ull", 0);
        av_opt_set(codec_->priv_data, "rc", "cbr", 0);
        av_opt_set(codec_->priv_data, "rc-lookahead", "0", 0);
        av_opt_set(codec_->priv_data, "delay", "0", 0);
        av_opt_set(codec_->priv_data, "forced-idr", "1", 0);
        check_ffmpeg(avcodec_open2(codec_, encoder, nullptr), "open NVENC encoder");

        video_stream_->time_base = codec_->time_base;
        check_ffmpeg(avcodec_parameters_from_context(video_stream_->codecpar, codec_),
                     "copy H.264 stream parameters");
        check_ffmpeg(avio_open(&format_->pb, partial_.string().c_str(), AVIO_FLAG_WRITE),
                     "open partial MP4 file");
        check_ffmpeg(avformat_write_header(format_, nullptr), "write MP4 header");
    }

    std::uint64_t drain(bool flushing) {
        AVPacket* packet = av_packet_alloc();
        if (!packet) {
            throw std::bad_alloc();
        }
        std::uint64_t bytes = 0;
        while (true) {
            const int result = avcodec_receive_packet(codec_, packet);
            if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) {
                if (flushing && result == AVERROR(EAGAIN)) {
                    av_packet_free(&packet);
                    throw std::runtime_error("NVENC flush did not reach end of stream");
                }
                break;
            }
            if (result < 0) {
                av_packet_free(&packet);
                check_ffmpeg(result, "receive NVENC packet");
            }
            av_packet_rescale_ts(packet, codec_->time_base, video_stream_->time_base);
            packet->stream_index = video_stream_->index;
            packet->flags &= ~AV_PKT_FLAG_DISCARD;
            if (packet->duration <= 0) {
                packet->duration = av_rescale_q(1, AVRational{1, static_cast<int>(frame_rate_)},
                                                video_stream_->time_base);
            }
            bytes += static_cast<std::uint64_t>(packet->size);
            const int write_result = av_interleaved_write_frame(format_, packet);
            av_packet_unref(packet);
            if (write_result < 0) {
                av_packet_free(&packet);
                check_ffmpeg(write_result, "mux H.264 packet");
            }
            ++encoded_frames_;
            bytes_written_ += static_cast<std::uint64_t>(packet->size);
        }
        av_packet_free(&packet);
        return bytes;
    }

    void cleanup() noexcept {
        if (format_ && format_->pb) {
            avio_closep(&format_->pb);
        }
        if (codec_) {
            avcodec_free_context(&codec_);
        }
        if (hw_frames_) {
            av_buffer_unref(&hw_frames_);
        }
        if (hw_device_) {
            av_buffer_unref(&hw_device_);
        }
        if (format_) {
            avformat_free_context(format_);
            format_ = nullptr;
        }
    }

    std::filesystem::path destination_;
    std::filesystem::path partial_;
    std::uint32_t width_{};
    std::uint32_t height_{};
    std::uint32_t frame_rate_{};
    MonotonicTime first_timestamp_{};
    infrastructure::gpu::CudaStream stream_;
    AVFormatContext* format_{};
    AVStream* video_stream_{};
    AVCodecContext* codec_{};
    AVBufferRef* hw_device_{};
    AVBufferRef* hw_frames_{};
    std::shared_ptr<Nv12SurfacePool> surfaces_;
    std::int64_t last_pts_{-1};
    std::uint64_t submitted_frames_{};
    std::uint64_t encoded_frames_{};
    std::uint64_t bytes_written_{};
    bool finalized_{};
};

GpuMp4Writer::GpuMp4Writer(const DiskOutputConfig& config, const Frame& first_frame)
    : impl_(std::make_unique<Impl>(config, first_frame)) {}

GpuMp4Writer::~GpuMp4Writer() = default;

std::uint64_t GpuMp4Writer::write(const Frame& frame) { return impl_->write(frame); }

void GpuMp4Writer::finalize() { impl_->finalize(); }

std::uint64_t GpuMp4Writer::encoded_frames() const noexcept { return impl_->encoded_frames(); }

std::uint64_t GpuMp4Writer::bytes_written() const noexcept { return impl_->bytes_written(); }

} // namespace iris::output::disk
