#include "stages/capture/video/VideoFileReader.hpp"

#include "iris/infrastructure/gpu/CudaResources.hpp"
#include "stages/capture/video/Nv12ToBgrCuda.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
}

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>

namespace iris::capture::video {
namespace {
std::string ffmpeg_error(int result) {
    char buffer[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(result, buffer, sizeof(buffer));
    return buffer;
}

void check_ffmpeg(int result, const std::string& operation) {
    if (result < 0) {
        throw std::runtime_error(std::string(operation) + ": " + ffmpeg_error(result));
    }
}

std::uint8_t clamp_byte(float value) {
    return static_cast<std::uint8_t>(std::clamp(std::lround(value), 0L, 255L));
}

void yuv_to_bgr(std::uint8_t y, std::uint8_t u, std::uint8_t v, std::uint8_t* target,
                bool full_range) {
    const float yy = full_range ? static_cast<float>(y) : 1.164383F * (static_cast<float>(y) - 16.0F);
    const float uu = static_cast<float>(u) - 128.0F;
    const float vv = static_cast<float>(v) - 128.0F;
    target[0] = clamp_byte(yy + 1.772F * uu);
    target[1] = clamp_byte(yy - 0.344136F * uu - 0.714136F * vv);
    target[2] = clamp_byte(yy + 1.402F * vv);
}
} // namespace

class VideoFileReader::Impl {
  public:
    Impl(const std::filesystem::path& path, int cuda_device)
        : path_(path), cuda_device_(cuda_device) {
        if (cuda_device_ < 0) throw std::invalid_argument("video CUDA device must be zero or greater");
        check_ffmpeg(avformat_open_input(&format_, path.string().c_str(), nullptr, nullptr),
                     "open video file " + path.string());
        try {
            check_ffmpeg(avformat_find_stream_info(format_, nullptr), "read video stream info");
            stream_index_ = av_find_best_stream(format_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
            check_ffmpeg(stream_index_, "find video stream");
            stream_ = format_->streams[stream_index_];
            const auto* decoder = avcodec_find_decoder(stream_->codecpar->codec_id);
            if (!decoder) throw std::runtime_error("no FFmpeg decoder for video file " + path.string());
            codec_name_ = decoder->name ? decoder->name : avcodec_get_name(stream_->codecpar->codec_id);
            set_decode_status("SOFTWARE", "Checking FFmpeg CUDA decode support.");

            if (!open_hardware_decoder(decoder)) open_software_decoder(decoder);
            packet_ = av_packet_alloc();
            frame_ = av_frame_alloc();
            transfer_frame_ = av_frame_alloc();
            if (!packet_ || !frame_ || !transfer_frame_) throw std::bad_alloc();
            if (codec_->width <= 0 || codec_->height <= 0) {
                throw std::runtime_error("video has invalid dimensions: " + path.string());
            }
            extent_ = {static_cast<std::uint32_t>(codec_->width),
                       static_cast<std::uint32_t>(codec_->height)};
            AVRational rate = av_guess_frame_rate(format_, stream_, nullptr);
            if (rate.num <= 0 || rate.den <= 0) rate = stream_->r_frame_rate;
            frame_rate_ = rate.num > 0 && rate.den > 0 ? av_q2d(rate) : 30.0;
            if (frame_rate_ <= 0.0 || !std::isfinite(frame_rate_)) frame_rate_ = 30.0;
            start_pts_ = stream_->start_time == AV_NOPTS_VALUE ? 0 : stream_->start_time;
            frame_interval_ = std::chrono::nanoseconds(
                static_cast<std::int64_t>(1'000'000'000.0 / frame_rate_));
        } catch (...) {
            release();
            throw;
        }
    }

    ~Impl() { release(); }

    static AVPixelFormat select_hardware_format(AVCodecContext* context,
                                                const AVPixelFormat* formats) {
        auto* self = static_cast<Impl*>(context->opaque);
        if (!self) return AV_PIX_FMT_NONE;
        AVPixelFormat software_fallback = AV_PIX_FMT_NONE;
        for (auto format = formats; *format != AV_PIX_FMT_NONE; ++format) {
            if (*format == self->hardware_pixel_format_) {
                self->set_decode_status("NVDEC", "FFmpeg selected the CUDA hardware pixel format.");
                return *format;
            }
            if (*format != AV_PIX_FMT_CUDA && software_fallback == AV_PIX_FMT_NONE) {
                software_fallback = *format;
            }
        }
        if (software_fallback != AV_PIX_FMT_NONE) {
            const auto* name = av_get_pix_fmt_name(software_fallback);
            self->set_decode_status(
                "SOFTWARE", "The decoder did not offer CUDA frames for this stream; selected " +
                                std::string(name ? name : "a software pixel format") + ".");
        } else {
            self->set_decode_status("SOFTWARE", "The decoder offered no usable CUDA or software pixel format.");
        }
        return software_fallback;
    }

    void set_decode_status(std::string backend, std::string detail) {
        std::scoped_lock lock(decode_status_mutex_);
        decode_backend_ = std::move(backend);
        decode_detail_ = std::move(detail);
    }

    VideoDecodeStatus decode_status(CameraId camera_id) const {
        std::scoped_lock lock(decode_status_mutex_);
        return {camera_id, codec_name_, decode_backend_, decode_detail_};
    }

    bool open_hardware_decoder(const AVCodec* decoder) {
        for (int index = 0;; ++index) {
            const auto* config = avcodec_get_hw_config(decoder, index);
            if (!config) break;
            if (config->device_type != AV_HWDEVICE_TYPE_CUDA ||
                !(config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX)) continue;
            hardware_pixel_format_ = config->pix_fmt;
            break;
        }
        if (hardware_pixel_format_ == AV_PIX_FMT_NONE) {
            set_decode_status(
                "SOFTWARE", "FFmpeg decoder '" + codec_name_ +
                                "' exposes no CUDA hardware configuration (codec or FFmpeg build may not support NVDEC).");
            return false;
        }

        // LibTorch or another CUDA runtime user may already have activated the
        // primary context with different scheduling flags. Select it through
        // the CUDA runtime and make FFmpeg borrow the current context instead
        // of asking FFmpeg to reconfigure/retain the primary context itself.
        const auto cuda_result = cudaSetDevice(cuda_device_);
        if (cuda_result != cudaSuccess) {
            hardware_pixel_format_ = AV_PIX_FMT_NONE;
            set_decode_status("SOFTWARE", "CUDA runtime could not select device " +
                                                std::to_string(cuda_device_) + ": " +
                                                cudaGetErrorString(cuda_result) + ".");
            return false;
        }

        const auto device = std::to_string(cuda_device_);
        const auto device_result = av_hwdevice_ctx_create(
            &hardware_device_, AV_HWDEVICE_TYPE_CUDA, device.c_str(), nullptr,
            AV_CUDA_USE_CURRENT_CONTEXT);
        if (device_result < 0) {
            const auto reason = ffmpeg_error(device_result);
            hardware_pixel_format_ = AV_PIX_FMT_NONE;
            set_decode_status("SOFTWARE", "FFmpeg CUDA device initialization failed: " + reason + ".");
            return false;
        }

        codec_ = avcodec_alloc_context3(decoder);
        if (!codec_) throw std::bad_alloc();
        check_ffmpeg(avcodec_parameters_to_context(codec_, stream_->codecpar),
                     "read video codec parameters");
        codec_->hw_device_ctx = av_buffer_ref(hardware_device_);
        if (!codec_->hw_device_ctx) throw std::bad_alloc();
        codec_->opaque = this;
        codec_->get_format = &Impl::select_hardware_format;
        const int result = avcodec_open2(codec_, decoder, nullptr);
        if (result >= 0) {
            set_decode_status("NVDEC", "CUDA decoder opened; waiting for the first decoded frame.");
            return true;
        }

        const auto reason = ffmpeg_error(result);
        avcodec_free_context(&codec_);
        av_buffer_unref(&hardware_device_);
        hardware_pixel_format_ = AV_PIX_FMT_NONE;
        set_decode_status("SOFTWARE", "FFmpeg could not open the CUDA decoder: " + reason + ".");
        return false;
    }

    void open_software_decoder(const AVCodec* decoder) {
        codec_ = avcodec_alloc_context3(decoder);
        if (!codec_) throw std::bad_alloc();
        check_ffmpeg(avcodec_parameters_to_context(codec_, stream_->codecpar),
                     "read video codec parameters");
        check_ffmpeg(avcodec_open2(codec_, decoder, nullptr), "open software video decoder");
    }

    void release() noexcept {
        av_frame_free(&transfer_frame_);
        av_frame_free(&frame_);
        av_packet_free(&packet_);
        avcodec_free_context(&codec_);
        av_buffer_unref(&hardware_device_);
        avformat_close_input(&format_);
    }

    std::optional<DecodedVideoFrame> read_next(void* bgr_device, std::size_t bgr_stride_bytes,
                                               cudaStream_t stream) {
        for (;;) {
            const int receive_result = avcodec_receive_frame(codec_, frame_);
            if (receive_result == 0) {
                auto result = convert_frame(bgr_device, bgr_stride_bytes, stream);
                av_frame_unref(frame_);
                return result;
            }
            if (receive_result == AVERROR_EOF) return std::nullopt;
            if (receive_result != AVERROR(EAGAIN)) check_ffmpeg(receive_result, "decode video frame");

            if (demux_eof_) {
                if (!flush_sent_) {
                    flush_sent_ = true;
                    const int flush_result = avcodec_send_packet(codec_, nullptr);
                    if (flush_result < 0 && flush_result != AVERROR_EOF) {
                        check_ffmpeg(flush_result, "flush video decoder");
                    }
                    continue;
                }
                return std::nullopt;
            }

            bool submitted = false;
            while (!submitted) {
                const int read_result = av_read_frame(format_, packet_);
                if (read_result == AVERROR_EOF) {
                    demux_eof_ = true;
                    break;
                }
                check_ffmpeg(read_result, "read video packet");
                if (packet_->stream_index != stream_index_) {
                    av_packet_unref(packet_);
                    continue;
                }
                const int send_result = avcodec_send_packet(codec_, packet_);
                if (send_result == AVERROR(EAGAIN)) {
                    throw std::runtime_error("FFmpeg decoder requested output before accepting input");
                }
                av_packet_unref(packet_);
                check_ffmpeg(send_result, "submit video packet");
                submitted = true;
            }
        }
    }

    DecodedVideoFrame convert_frame(void* bgr_device, std::size_t bgr_stride_bytes,
                                    cudaStream_t stream) {
        const AVFrame* source = frame_;
        bool converted_on_gpu = false;
        if (frame_->format == AV_PIX_FMT_CUDA && frame_->hw_frames_ctx) {
            hardware_decode_used_ = true;
            const auto* frames = reinterpret_cast<const AVHWFramesContext*>(frame_->hw_frames_ctx->data);
            if (frames->sw_format == AV_PIX_FMT_NV12) {
                set_decode_status("NVDEC", "NVDEC active; NV12 to BGR conversion runs on the GPU.");
                const auto width = static_cast<std::uint32_t>(frame_->width);
                const auto height = static_cast<std::uint32_t>(frame_->height);
                if (width != extent_.width || height != extent_.height) {
                    throw std::runtime_error("video resolution changes are not supported in " + path_.string());
                }
                if (!bgr_device || bgr_stride_bytes < static_cast<std::size_t>(width) * 3) {
                    throw std::invalid_argument("NVDEC requires a valid destination BGR GPU buffer");
                }
                const bool full_range = frame_->color_range == AVCOL_RANGE_JPEG;
                launch_nv12_to_bgr(
                    reinterpret_cast<const std::uint8_t*>(frame_->data[0]),
                    static_cast<std::size_t>(frame_->linesize[0]),
                    reinterpret_cast<const std::uint8_t*>(frame_->data[1]),
                    static_cast<std::size_t>(frame_->linesize[1]), bgr_device,
                    bgr_stride_bytes, width, height, full_range, stream);
                // NVDEC surfaces are owned by FFmpeg and are reused after the
                // AVFrame is unreferenced, so finish the conversion before that.
                iris::infrastructure::gpu::check_cuda(
                    cudaStreamSynchronize(stream), "wait for NVDEC color conversion");
                converted_on_gpu = true;
            } else {
                const auto* name = av_get_pix_fmt_name(frames->sw_format);
                set_decode_status("NVDEC", "NVDEC active; " +
                                                  std::string(name ? name : "unsupported surface format") +
                                                  " surface is converted on the CPU.");
                av_frame_unref(transfer_frame_);
                check_ffmpeg(av_hwframe_transfer_data(transfer_frame_, frame_, 0),
                             "transfer unsupported NVDEC surface format to software frame");
                source = transfer_frame_;
            }
        }

        const auto width = static_cast<std::uint32_t>(frame_->width);
        const auto height = static_cast<std::uint32_t>(frame_->height);
        if (width != extent_.width || height != extent_.height) {
            throw std::runtime_error("video resolution changes are not supported in " + path_.string());
        }
        DecodedVideoFrame result;
        result.extent = extent_;
        result.converted_on_gpu = converted_on_gpu;
        if (!converted_on_gpu) result.bgr = convert_to_bgr(*source);

        const auto pts = frame_->best_effort_timestamp;
        if (pts == AV_NOPTS_VALUE) {
            result.presentation_time = last_timestamp_ + frame_interval_;
        } else {
            const auto relative = pts - start_pts_;
            result.presentation_time = std::chrono::nanoseconds(
                av_rescale_q(relative, stream_->time_base, AVRational{1, 1'000'000'000}));
            if (result.presentation_time <= last_timestamp_ && frame_count_ != 0) {
                result.presentation_time = last_timestamp_ + frame_interval_;
            }
        }
        last_timestamp_ = result.presentation_time;
        ++frame_count_;
        return result;
    }

    std::vector<std::uint8_t> convert_to_bgr(const AVFrame& source) const {
        const auto width = static_cast<std::uint32_t>(source.width);
        const auto height = static_cast<std::uint32_t>(source.height);
        std::vector<std::uint8_t> result(static_cast<std::size_t>(width) * height * 3);
        const auto format = static_cast<AVPixelFormat>(source.format);
        const bool packed_bgr = format == AV_PIX_FMT_BGR24;
        const bool packed_bgra = format == AV_PIX_FMT_BGRA;
        const bool packed_rgba = format == AV_PIX_FMT_RGBA;
        const bool yuv420 = format == AV_PIX_FMT_YUV420P || format == AV_PIX_FMT_YUVJ420P;
        const bool nv12 = format == AV_PIX_FMT_NV12;
        if (!packed_bgr && !packed_bgra && !packed_rgba && !yuv420 && !nv12) {
            throw std::runtime_error("unsupported decoded pixel format " +
                                     std::string(av_get_pix_fmt_name(format) ? av_get_pix_fmt_name(format) : "unknown") +
                                     " in " + path_.string());
        }

        const bool full_range = format == AV_PIX_FMT_YUVJ420P || source.color_range == AVCOL_RANGE_JPEG;
        for (std::uint32_t y = 0; y < height; ++y) {
            for (std::uint32_t x = 0; x < width; ++x) {
                auto* target = result.data() + (static_cast<std::size_t>(y) * width + x) * 3;
                if (packed_bgr) {
                    const auto* pixel = source.data[0] + static_cast<std::ptrdiff_t>(y) * source.linesize[0] + x * 3;
                    std::copy_n(pixel, 3, target);
                } else if (packed_bgra || packed_rgba) {
                    constexpr unsigned int channels = 4;
                    const auto* pixel = source.data[0] + static_cast<std::ptrdiff_t>(y) * source.linesize[0] + x * channels;
                    if (packed_bgra) std::copy_n(pixel, 3, target);
                    else { target[0] = pixel[2]; target[1] = pixel[1]; target[2] = pixel[0]; }
                } else {
                    const auto luma = source.data[0][static_cast<std::ptrdiff_t>(y) * source.linesize[0] + x];
                    std::uint8_t u{}, v{};
                    if (nv12) {
                        const auto* chroma = source.data[1] + static_cast<std::ptrdiff_t>(y / 2) * source.linesize[1] + (x / 2) * 2;
                        u = chroma[0]; v = chroma[1];
                    } else {
                        u = source.data[1][static_cast<std::ptrdiff_t>(y / 2) * source.linesize[1] + x / 2];
                        v = source.data[2][static_cast<std::ptrdiff_t>(y / 2) * source.linesize[2] + x / 2];
                    }
                    yuv_to_bgr(luma, u, v, target, full_range);
                }
            }
        }
        return result;
    }

    std::filesystem::path path_;
    int cuda_device_{};
    AVFormatContext* format_{};
    AVCodecContext* codec_{};
    AVBufferRef* hardware_device_{};
    AVPacket* packet_{};
    AVFrame* frame_{};
    AVFrame* transfer_frame_{};
    AVStream* stream_{};
    int stream_index_{-1};
    AVPixelFormat hardware_pixel_format_{AV_PIX_FMT_NONE};
    Extent2D extent_{};
    double frame_rate_{30.0};
    std::int64_t start_pts_{};
    std::chrono::nanoseconds frame_interval_{std::chrono::milliseconds(33)};
    std::chrono::nanoseconds last_timestamp_{};
    std::uint64_t frame_count_{};
    bool hardware_decode_used_{};
    mutable std::mutex decode_status_mutex_;
    std::string codec_name_;
    std::string decode_backend_;
    std::string decode_detail_;
    bool demux_eof_{};
    bool flush_sent_{};
};

VideoFileReader::VideoFileReader(const std::filesystem::path& path, int cuda_device)
    : impl_(std::make_unique<Impl>(path, cuda_device)) {}
VideoFileReader::~VideoFileReader() = default;
Extent2D VideoFileReader::extent() const noexcept { return impl_->extent_; }
double VideoFileReader::frame_rate() const noexcept { return impl_->frame_rate_; }
std::optional<DecodedVideoFrame> VideoFileReader::read_next(void* bgr_device,
                                                            std::size_t bgr_stride_bytes,
                                                            cudaStream_t stream) {
    return impl_->read_next(bgr_device, bgr_stride_bytes, stream);
}
bool VideoFileReader::hardware_decode_enabled() const noexcept { return impl_->hardware_decode_used_; }
VideoDecodeStatus VideoFileReader::decode_status(CameraId camera_id) const {
    return impl_->decode_status(camera_id);
}

} // namespace iris::capture::video
