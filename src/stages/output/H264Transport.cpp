#include "iris/stages/output/H264Transport.hpp"
#include "iris/infrastructure/gpu/CudaResources.hpp"
#include "iris/stages/output/H264PreviewProtocol.hpp"
#include "iris/pipeline/Channel.hpp"
#include "stages/output/disk/BgrToNv12.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
}
#include <cuda_runtime_api.h>

#include <atomic>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <iostream>

namespace iris::output {
void resize_bgr(const unsigned char*, std::size_t, unsigned char*, std::size_t, unsigned, unsigned, unsigned, unsigned, cudaStream_t);
namespace {
std::vector<std::uint8_t> to_avcc(std::vector<std::uint8_t> input) {
    auto start_code = [&](std::size_t offset) { return offset + 3 < input.size() && input[offset] == 0 && input[offset + 1] == 0 && input[offset + 2] == 1 ? 3U : (offset + 4 <= input.size() && input[offset] == 0 && input[offset + 1] == 0 && input[offset + 2] == 0 && input[offset + 3] == 1 ? 4U : 0U); };
    std::size_t first{}; while (first < input.size() && !start_code(first)) ++first;
    if (first == input.size()) return input;
    std::vector<std::uint8_t> output;
    for (std::size_t cursor = first; cursor < input.size();) {
        const auto prefix = start_code(cursor); if (!prefix) break;
        const auto nal = cursor + prefix; std::size_t next = nal; while (next < input.size() && !start_code(next)) ++next;
        const auto size = static_cast<std::uint32_t>(next - nal); output.push_back(static_cast<std::uint8_t>(size >> 24)); output.push_back(static_cast<std::uint8_t>(size >> 16)); output.push_back(static_cast<std::uint8_t>(size >> 8)); output.push_back(static_cast<std::uint8_t>(size)); output.insert(output.end(), input.begin() + nal, input.begin() + next); cursor = next;
    }
    return output;
}
std::vector<std::uint8_t> to_avcc_description(const std::vector<std::uint8_t>& input) {
    if (input.size() >= 7 && input[0] == 1) return input;
    std::vector<std::uint8_t> sps, pps;
    auto start_code = [&](std::size_t offset) { return offset + 3 < input.size() && input[offset] == 0 && input[offset + 1] == 0 && input[offset + 2] == 1 ? 3U : (offset + 4 <= input.size() && input[offset] == 0 && input[offset + 1] == 0 && input[offset + 2] == 0 && input[offset + 3] == 1 ? 4U : 0U); };
    std::size_t cursor{};
    while (cursor < input.size()) {
        while (cursor < input.size() && !start_code(cursor)) ++cursor;
        if (cursor == input.size()) break;
        const auto nal = cursor + start_code(cursor); std::size_t next = nal; while (next < input.size() && !start_code(next)) ++next;
        if (next > nal && (input[nal] & 0x1fU) == 7U) sps.assign(input.begin() + nal, input.begin() + next);
        if (next > nal && (input[nal] & 0x1fU) == 8U) pps.assign(input.begin() + nal, input.begin() + next);
        cursor = next;
    }
    if (sps.size() < 4 || pps.empty()) return input;
    std::vector<std::uint8_t> output{1, sps[1], sps[2], sps[3], 0xff, 0xe1, static_cast<std::uint8_t>(sps.size() >> 8), static_cast<std::uint8_t>(sps.size()),};
    output.insert(output.end(), sps.begin(), sps.end()); output.push_back(1); output.push_back(static_cast<std::uint8_t>(pps.size() >> 8)); output.push_back(static_cast<std::uint8_t>(pps.size())); output.insert(output.end(), pps.begin(), pps.end()); return output;
}
struct Encoder {
    AVCodecContext* codec{};
    AVBufferRef* device{};
    AVBufferRef* frames{};
    AVFrame* hardware{};
    std::uint32_t width{}, height{}, fps{};
    std::uint64_t sequence{};

    ~Encoder() {
        av_frame_free(&hardware); av_buffer_unref(&frames); av_buffer_unref(&device); avcodec_free_context(&codec);
    }
    void open(std::uint32_t w, std::uint32_t h, std::uint32_t rate, std::uint32_t bitrate) {
        width = w & ~1U; height = h & ~1U; fps = rate;
        const auto* encoder = avcodec_find_encoder_by_name("h264_nvenc");
        if (!encoder) throw std::runtime_error("FFmpeg h264_nvenc encoder is unavailable");
        if (av_hwdevice_ctx_create(&device, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) < 0) throw std::runtime_error("could not initialize FFmpeg CUDA device");
        frames = av_hwframe_ctx_alloc(device); if (!frames) throw std::bad_alloc();
        auto* context = reinterpret_cast<AVHWFramesContext*>(frames->data); context->format = AV_PIX_FMT_CUDA; context->sw_format = AV_PIX_FMT_NV12; context->width = width; context->height = height; context->initial_pool_size = 4;
        if (av_hwframe_ctx_init(frames) < 0) throw std::runtime_error("could not initialize NVENC frame pool");
        codec = avcodec_alloc_context3(encoder); if (!codec) throw std::bad_alloc();
        codec->width = static_cast<int>(width); codec->height = static_cast<int>(height); codec->pix_fmt = AV_PIX_FMT_CUDA; codec->time_base = {1, static_cast<int>(fps)}; codec->framerate = {static_cast<int>(fps), 1}; codec->bit_rate = bitrate; codec->gop_size = static_cast<int>(fps); codec->max_b_frames = 0; codec->flags |= AV_CODEC_FLAG_LOW_DELAY | AV_CODEC_FLAG_GLOBAL_HEADER; codec->hw_frames_ctx = av_buffer_ref(frames);
        av_opt_set(codec->priv_data, "preset", "p1", 0); av_opt_set(codec->priv_data, "tune", "ull", 0); av_opt_set(codec->priv_data, "rc", "cbr", 0); av_opt_set(codec->priv_data, "rc-lookahead", "0", 0); av_opt_set(codec->priv_data, "forced-idr", "1", 0);
        if (avcodec_open2(codec, encoder, nullptr) < 0) throw std::runtime_error("could not open h264_nvenc");
        hardware = av_frame_alloc(); if (!hardware) throw std::bad_alloc();
        hardware->format = AV_PIX_FMT_CUDA; hardware->width = width; hardware->height = height;
    }
    std::vector<std::pair<std::vector<std::uint8_t>, bool>> encode(const Frame& frame, std::uint32_t rate, std::uint32_t bitrate, std::uint32_t max_width) {
        const auto target_width = frame.extent.width > max_width ? (max_width & ~1U) : (frame.extent.width & ~1U);
        const auto target_height = static_cast<std::uint32_t>((static_cast<std::uint64_t>(frame.extent.height) * target_width / frame.extent.width) & ~1ULL);
        if (!codec) open(target_width, target_height, rate, bitrate);
        if (frame.ready) frame.ready->synchronize();
        unsigned char* resized{}; std::size_t resized_pitch{};
        const auto* source = static_cast<const std::uint8_t*>(frame.buffer.data); auto source_pitch = frame.buffer.stride_bytes;
        if (target_width != frame.extent.width) { if (cudaMallocPitch(reinterpret_cast<void**>(&resized), &resized_pitch, target_width * 3U, target_height) != cudaSuccess) throw std::runtime_error("preview resize allocation failed"); resize_bgr(source, source_pitch, resized, resized_pitch, frame.extent.width, frame.extent.height, target_width, target_height, nullptr); source = resized; source_pitch = resized_pitch; }
        av_frame_unref(hardware);
        hardware->format = AV_PIX_FMT_CUDA; hardware->width = width; hardware->height = height;
        if (av_hwframe_get_buffer(frames, hardware, 0) < 0) throw std::runtime_error("NVENC CUDA frame allocation failed");
        disk::convert_bgr_to_nv12(source, source_pitch,
                            reinterpret_cast<std::uint8_t*>(hardware->data[0]), hardware->linesize[0], width, height, nullptr);
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) { if (resized) cudaFree(resized); throw std::runtime_error("CUDA preview conversion failed"); }
        if (resized) cudaFree(resized);
        hardware->pts = static_cast<std::int64_t>(sequence++); if (avcodec_send_frame(codec, hardware) < 0) throw std::runtime_error("NVENC send failed");
        std::vector<std::pair<std::vector<std::uint8_t>, bool>> output; AVPacket* packet = av_packet_alloc(); if (!packet) throw std::bad_alloc();
        while (avcodec_receive_packet(codec, packet) == 0) { output.emplace_back(to_avcc(std::vector<std::uint8_t>(packet->data, packet->data + packet->size)), (packet->flags & AV_PKT_FLAG_KEY) != 0); av_packet_unref(packet); }
        av_packet_free(&packet); return output;
    }
};
}

class H264Transport::Impl {
  public:
    Impl(H264PreviewConfig value, PreviewHttpServer& target, H264Transport::PublishObserver publish_observer)
        : config(std::move(value)), server(target), observer(std::move(publish_observer)),
          queue(config.queue_capacity, OverflowPolicy::DropOldest) { if (!config.queue_capacity || !config.max_fps || !config.max_width || !config.bitrate) throw std::invalid_argument("invalid H.264 preview configuration"); }
    ~Impl() { stop(); }
    void start() { if (running.exchange(true)) return; worker = std::thread([this] { run(); }); }
    void stop() noexcept { if (running.exchange(false)) queue.close(); if (worker.joinable()) worker.join(); }
    void publish(PreviewPacket packet) noexcept { if (!running) return; const auto result = queue.send(std::move(packet)); if (result != SendResult::Sent) ++dropped; }
    PreviewTransportHealth health() const { std::scoped_lock lock(mutex); return {config.enabled, clients, 0, clients, 0, published, dropped, error, "H264/NVENC", 0, 0}; }
  private:
    void run() noexcept {
        while (auto packet = queue.receive()) {
            for (const auto& frame : (*packet)->frames) try {
                if (frame.format != PixelFormat::Bgr8 || !frame.buffer.data) continue;
                auto& encoder = encoders[frame.camera];
                const auto target_width = frame.extent.width > config.max_width ? (config.max_width & ~1U) : (frame.extent.width & ~1U);
                const auto target_height = static_cast<std::uint32_t>((static_cast<std::uint64_t>(frame.extent.height) * target_width / frame.extent.width) & ~1ULL);
                if (encoder && (encoder->width != target_width || encoder->height != target_height)) encoder.reset();
                if (!encoder) encoder = std::make_unique<Encoder>();
                const bool was_open = encoder->codec != nullptr;
                auto encoded = encoder->encode(frame, config.max_fps, config.bitrate, config.max_width);
                if (!was_open && encoder->codec) {
                    H264PreviewStreamConfig stream;
                    stream.camera = frame.camera; stream.width = encoder->width; stream.height = encoder->height;
                    stream.fps = encoder->fps; stream.codec = "avc1.42E01E";
                    if (encoder->codec->extradata && encoder->codec->extradata_size > 0) {
                        std::vector<std::uint8_t> extradata(encoder->codec->extradata, encoder->codec->extradata + encoder->codec->extradata_size);
                        stream.description = to_avcc_description(extradata);
                    }
                    server.set_h264_stream_config(std::move(stream));
                }
                for (auto& [payload, key] : encoded) {
                    H264PreviewAccessUnit unit;
                    unit.flags = key ? h264_flag_keyframe : 0; unit.camera = frame.camera; unit.sequence = frame.sequence;
                    unit.timestamp_us = static_cast<std::uint64_t>(frame.timing.source_time.count() / 1000);
                    unit.payload = std::move(payload); server.publish_h264(std::move(unit)); ++published;
                }
                if (!encoded.empty() && observer)
                    observer(frame.camera, frame.sequence, std::chrono::steady_clock::now());
            } catch (const std::exception& exception) {
                std::scoped_lock lock(mutex); error = exception.what(); ++dropped;
                std::cerr << "IRIS H.264 preview error: " << exception.what() << "\n";
            }
        }
    }
    H264PreviewConfig config; PreviewHttpServer& server; H264Transport::PublishObserver observer; Channel<PreviewPacket> queue; std::atomic_bool running{false}; std::thread worker; mutable std::mutex mutex; std::unordered_map<CameraId, std::unique_ptr<Encoder>> encoders; std::size_t published{}, dropped{}, clients{}; std::string error;
};
H264Transport::H264Transport(H264PreviewConfig config, PreviewHttpServer& server, PublishObserver observer) : impl_(std::make_unique<Impl>(std::move(config), server, std::move(observer))) {}
H264Transport::~H264Transport() = default; void H264Transport::start() { impl_->start(); } void H264Transport::publish(PreviewPacket packet) noexcept { impl_->publish(std::move(packet)); } void H264Transport::stop() noexcept { impl_->stop(); } PreviewTransportHealth H264Transport::health() const { return impl_->health(); }
}
