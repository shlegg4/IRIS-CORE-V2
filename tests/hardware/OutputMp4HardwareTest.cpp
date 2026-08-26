#include "iris/infrastructure/gpu/CudaResources.hpp"
#include "iris/infrastructure/metrics/MetricRegistry.hpp"
#include "iris/stages/OutputStage.hpp"

#include <cuda_runtime_api.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

#include <cassert>
#include <chrono>
#include <filesystem>
#include <memory>
#include <thread>

namespace {

bool wait_for_count(const iris::OutputStage& output, std::size_t expected) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
        if (output.processed_count() >= expected) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}

iris::Frame make_frame(iris::CameraId camera, std::uint64_t sequence, std::uint32_t width,
                       std::uint32_t height, iris::MonotonicTime capture_time) {
    void* pixels = nullptr;
    std::size_t pitch = 0;
    assert(cudaMallocPitch(&pixels, &pitch, width * 3U, height) == cudaSuccess);
    assert(cudaMemset2D(pixels, pitch, static_cast<int>(sequence), width * 3U, height) ==
           cudaSuccess);
    auto owner = std::shared_ptr<void>(pixels, [](void* value) { cudaFree(value); });
    auto ready = std::make_shared<iris::infrastructure::gpu::CudaEvent>();
    ready->record(nullptr);
    iris::Frame frame;
    frame.camera = camera;
    frame.sequence = sequence;
    frame.extent = {width, height};
    frame.format = iris::PixelFormat::Bgr8;
    frame.buffer = {pixels, pitch, pitch * height, 0, std::move(owner)};
    frame.ready = std::move(ready);
    frame.timing.estimated_capture_time = capture_time;
    return frame;
}

std::size_t count_decoded_frames(const std::filesystem::path& path) {
    AVFormatContext* context = nullptr;
    assert(avformat_open_input(&context, path.string().c_str(), nullptr, nullptr) == 0);
    assert(avformat_find_stream_info(context, nullptr) >= 0);
    const int stream = av_find_best_stream(context, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    assert(stream >= 0);
    const AVCodec* decoder = avcodec_find_decoder(
        context->streams[static_cast<std::size_t>(stream)]->codecpar->codec_id);
    assert(decoder);
    AVCodecContext* codec = avcodec_alloc_context3(decoder);
    assert(codec);
    assert(avcodec_parameters_to_context(
               codec, context->streams[static_cast<std::size_t>(stream)]->codecpar) == 0);
    assert(avcodec_open2(codec, decoder, nullptr) == 0);
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    assert(packet);
    assert(frame);
    std::size_t count = 0;
    while (av_read_frame(context, packet) >= 0) {
        if (packet->stream_index == stream) {
            assert(avcodec_send_packet(codec, packet) == 0);
            while (avcodec_receive_frame(codec, frame) == 0) {
                ++count;
                av_frame_unref(frame);
            }
        }
        av_packet_unref(packet);
    }
    assert(avcodec_send_packet(codec, nullptr) == 0);
    while (avcodec_receive_frame(codec, frame) == 0) {
        ++count;
        av_frame_unref(frame);
    }
    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&codec);
    avformat_close_input(&context);
    return count;
}

} // namespace

int main() {
    namespace fs = std::filesystem;
    constexpr std::uint32_t width = 640;
    constexpr std::uint32_t height = 480;
    constexpr std::size_t frame_count = 30;
    const auto output = fs::current_path() / "output-mp4-smoke.mp4";
    const auto partial = output.string() + ".partial";
    fs::remove(output);
    fs::remove(partial);

    iris::infrastructure::metrics::MetricRegistry metrics;
    iris::Channel<iris::Packet> input(8, iris::OverflowPolicy::Block);
    iris::OutputConfig config;
    config.disk.destination = output;
    config.disk.queue_capacity = 30;
    config.disk.frame_rate = 30;
    config.disk.bitrate = 4'000'000;
    iris::OutputStage stage(input, metrics, config);
    stage.start();
    assert(stage.start_recording());

    const auto first_time = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < frame_count; ++index) {
        iris::Packet packet;
        packet.sequence = index;
        packet.frames.push_back(make_frame(0, index, width, height,
                                           first_time + std::chrono::milliseconds(index * 33)));
        assert(input.send(std::move(packet)) == iris::SendResult::Sent);
    }

    assert(wait_for_count(stage, frame_count));
    const auto result = stage.stop_recording();
    assert(result);
    input.close();
    stage.stop();

    assert(fs::exists(output));
    assert(!fs::exists(partial));
    assert(count_decoded_frames(output) == frame_count);
    const auto snapshot = metrics.snapshot();
    assert(snapshot.counters.at("iris_output_disk_accepted_frames_total") == frame_count);
    assert(snapshot.counters.at("iris_output_disk_encoded_frames_total") == frame_count);
    assert(snapshot.counters.at("iris_output_disk_packets_total") == frame_count);
    assert(snapshot.counters.at("iris_output_disk_failures_total") == 0);
    fs::remove(output);

    constexpr std::size_t multi_frame_count = 10;
    const auto multi_base = fs::current_path() / "output-multi-smoke.mp4";
    const auto camera_zero = fs::current_path() / "output-multi-smoke-camera-0.mp4";
    const auto camera_one = fs::current_path() / "output-multi-smoke-camera-1.mp4";
    fs::remove(camera_zero);
    fs::remove(camera_one);
    iris::infrastructure::metrics::MetricRegistry multi_metrics;
    iris::Channel<iris::Packet> multi_input(8, iris::OverflowPolicy::Block);
    iris::OutputConfig multi_config;
    multi_config.camera_count = 2;
    multi_config.disk.destination = multi_base;
    multi_config.disk.queue_capacity = multi_frame_count;
    multi_config.disk.frame_rate = 30;
    multi_config.disk.bitrate = 4'000'000;
    iris::OutputStage multi_stage(multi_input, multi_metrics, multi_config);
    multi_stage.start();
    assert(multi_stage.start_recording());
    for (std::size_t index = 0; index < multi_frame_count; ++index) {
        iris::Packet packet;
        packet.sequence = index;
        const auto capture_time = first_time + std::chrono::milliseconds(index * 33);
        packet.frames.push_back(make_frame(0, index, width, height, capture_time));
        packet.frames.push_back(make_frame(1, index, width, height, capture_time));
        assert(multi_input.send(std::move(packet)) == iris::SendResult::Sent);
    }
    assert(wait_for_count(multi_stage, multi_frame_count));
    assert(multi_stage.stop_recording());
    multi_input.close();
    multi_stage.stop();
    assert(fs::exists(camera_zero));
    assert(fs::exists(camera_one));
    assert(count_decoded_frames(camera_zero) == multi_frame_count);
    assert(count_decoded_frames(camera_one) == multi_frame_count);
    const auto multi_snapshot = multi_metrics.snapshot();
    assert(multi_snapshot.counters.at("iris_output_disk_accepted_frames_total") ==
           multi_frame_count * 2);
    assert(multi_snapshot.counters.at("iris_output_disk_encoded_frames_total") ==
           multi_frame_count * 2);
    assert(multi_snapshot.counters.at("iris_output_disk_failures_total") == 0);
    fs::remove(camera_zero);
    fs::remove(camera_one);
}
