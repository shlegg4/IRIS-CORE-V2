#include "iris/stages/capture/SynchronizedVideoStage.hpp"

#include "iris/infrastructure/gpu/CudaResources.hpp"
#include "iris/infrastructure/gpu/FramePool.hpp"
#include "stages/capture/decoding/GpuDecoder.hpp"
#include "stages/capture/video/VideoFileReader.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace iris {
namespace {
Extent2D output_extent(Extent2D input, FrameRotation rotation) {
    return swaps_axes(rotation) ? Extent2D{input.height, input.width} : input;
}
} // namespace

class SynchronizedVideoStage::Impl {
  public:
    Impl(SynchronizedVideoConfig config, Channel<Packet>& output,
         infrastructure::metrics::MetricRegistry& registry)
        : config_(std::move(config)), output_(output),
          frames_emitted_(registry.counter("iris_video_frames_emitted_total")),
          batches_emitted_(registry.counter("iris_video_batches_emitted_total")),
          nvdec_frames_(registry.counter("iris_video_nvdec_frames_total")),
          software_decode_frames_(registry.counter("iris_video_software_decode_frames_total")),
          batch_decode_ms_(registry.histogram("iris_video_batch_decode_ms", {1, 2, 5, 10, 20, 50, 100, 250, 500, 1000})),
          batch_pool_wait_ms_(registry.histogram("iris_video_batch_pool_wait_ms", {0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100})),
          batch_upload_ms_(registry.histogram("iris_video_batch_upload_ms", {0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 250})),
          batch_submit_wait_ms_(registry.histogram("iris_video_batch_submit_wait_ms", {0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100, 500})),
          batch_work_ms_(registry.histogram("iris_video_batch_work_ms", {1, 2, 5, 10, 20, 50, 100, 250, 500, 1000})),
          batch_interarrival_ms_(registry.histogram("iris_video_batch_interarrival_ms", {1, 2, 5, 10, 20, 33, 50, 100, 250, 500, 1000, 5000})),
          source_fps_(registry.gauge("iris_video_source_frame_rate_fps")),
          batch_fps_(registry.gauge("iris_video_batch_rate_fps")),
          ingestion_active_(registry.gauge("iris_video_ingestion_active")),
          last_batch_decode_ms_(registry.gauge("iris_video_last_batch_decode_ms")),
          last_batch_pool_wait_ms_(registry.gauge("iris_video_last_batch_pool_wait_ms")),
          last_batch_upload_ms_(registry.gauge("iris_video_last_batch_upload_ms")),
          last_batch_submit_wait_ms_(registry.gauge("iris_video_last_batch_submit_wait_ms")),
          last_batch_work_ms_(registry.gauge("iris_video_last_batch_work_ms")) {
        if (config_.cameras.empty()) {
            throw std::invalid_argument("video ingestion requires at least one camera file");
        }
        if (config_.cuda_device < 0 || config_.frame_pool_capacity == 0) {
            throw std::invalid_argument("video CUDA device and frame-pool capacity are invalid");
        }
        std::unordered_set<CameraId> ids;
        for (const auto& camera : config_.cameras) {
            if (!ids.insert(camera.camera_id).second) {
                throw std::invalid_argument("video camera IDs must be unique");
            }
            if (camera.path.empty()) {
                throw std::invalid_argument("video path is empty for camera " +
                                            std::to_string(camera.camera_id));
            }
        }
        readers_.reserve(config_.cameras.size());
        for (const auto& camera : config_.cameras) {
            readers_.push_back(std::make_unique<capture::video::VideoFileReader>(
                camera.path, config_.cuda_device));
            const auto prefix = "iris_video_camera_" + std::to_string(camera.camera_id);
            camera_decode_ms_.push_back(registry.histogram(prefix + "_decode_ms", {1, 2, 5, 10, 20, 50, 100, 250, 500, 1000}));
            camera_source_fps_.push_back(registry.gauge(prefix + "_source_frame_rate_fps"));
            camera_last_decode_ms_.push_back(registry.gauge(prefix + "_last_decode_ms"));
            camera_nvdec_active_.push_back(registry.gauge(prefix + "_nvdec_active"));
            camera_gpu_conversion_active_.push_back(registry.gauge(prefix + "_gpu_conversion_active"));
        }
    }

    void start() {
        if (worker_.joinable()) {
            return;
        }
        finished_ = false;
        healthy_ = true;
        ingestion_active_.set(1.0);
        batch_fps_.set(0.0);
        worker_ = std::jthread([this](std::stop_token stop) { run(stop); });
    }

    void run(std::stop_token stop) noexcept {
        try {
            std::vector<std::unique_ptr<infrastructure::gpu::FramePool>> pools;
            std::vector<std::unique_ptr<infrastructure::gpu::FramePool>> rotation_pools;
            pools.reserve(config_.cameras.size());
            rotation_pools.resize(config_.cameras.size());
            for (std::size_t index = 0; index < config_.cameras.size(); ++index) {
                pools.push_back(std::make_unique<infrastructure::gpu::FramePool>(
                    config_.cuda_device, readers_[index]->extent(), 3,
                    config_.frame_pool_capacity));
                const auto rotation = config_.cameras[index].rotation;
                if (rotation != FrameRotation::None) {
                    rotation_pools[index] = std::make_unique<infrastructure::gpu::FramePool>(
                        config_.cuda_device, output_extent(readers_[index]->extent(), rotation),
                        3, config_.frame_pool_capacity);
                }
            }

            std::vector<double> frame_rates;
            frame_rates.reserve(readers_.size());
            for (const auto& reader : readers_) {
                frame_rates.push_back(reader->frame_rate());
            }
            if (!frame_rates.empty()) source_fps_.set(frame_rates.front());
            for (std::size_t index = 0; index < frame_rates.size(); ++index) {
                camera_source_fps_[index].set(frame_rates[index]);
            }
            const auto playback_start = std::chrono::steady_clock::now();
            MonotonicTime capture_origin = playback_start;
            std::vector<std::string> reported_decode_status(readers_.size());
            std::uint64_t batch_index{};
            std::chrono::nanoseconds previous_time{-1};
            auto previous_emit = std::chrono::steady_clock::time_point{};

            for (;;) {
                if (stop.stop_requested()) {
                    break;
                }
                const auto batch_start = std::chrono::steady_clock::now();
                infrastructure::gpu::check_cuda(cudaSetDevice(config_.cuda_device),
                                                "select video CUDA device");
                std::vector<std::optional<GpuBuffer>> buffers(readers_.size());
                std::vector<std::optional<GpuBuffer>> rotated_buffers(readers_.size());
                double pool_wait_ms{};
                for (std::size_t index = 0; index < readers_.size(); ++index) {
                    const auto pool_wait_start = std::chrono::steady_clock::now();
                    buffers[index] = pools[index]->acquire();
                    while (!buffers[index] && !stop.stop_requested()) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        buffers[index] = pools[index]->acquire();
                    }
                    if (buffers[index] && rotation_pools[index]) {
                        rotated_buffers[index] = rotation_pools[index]->acquire();
                        while (!rotated_buffers[index] && !stop.stop_requested()) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                            rotated_buffers[index] = rotation_pools[index]->acquire();
                        }
                    }
                    pool_wait_ms += elapsed_ms(pool_wait_start);
                    if (stop.stop_requested()) break;
                }
                if (stop.stop_requested()) break;

                std::vector<std::optional<capture::video::DecodedVideoFrame>> decoded;
                decoded.reserve(readers_.size());
                std::size_t ended{};
                const auto decode_start = std::chrono::steady_clock::now();
                for (std::size_t index = 0; index < readers_.size(); ++index) {
                    const auto camera_decode_start = std::chrono::steady_clock::now();
                    decoded.push_back(readers_[index]->read_next(
                        buffers[index]->data, buffers[index]->stride_bytes, nullptr));
                    const auto camera_decode_ms = elapsed_ms(camera_decode_start);
                    if (decoded.back()) {
                        const auto status = readers_[index]->decode_status(
                            config_.cameras[index].camera_id);
                        const auto status_line = status.backend + "|" + status.codec + "|" + status.detail;
                        if (reported_decode_status[index] != status_line) {
                            std::clog << "IRIS video camera " << status.camera_id
                                      << " codec=" << status.codec
                                      << " backend=" << status.backend
                                      << ": " << status.detail << '\n';
                            reported_decode_status[index] = status_line;
                        }
                        camera_decode_ms_[index].observe(camera_decode_ms);
                        camera_last_decode_ms_[index].set(camera_decode_ms);
                        const bool nvdec = readers_[index]->hardware_decode_enabled();
                        camera_nvdec_active_[index].set(nvdec ? 1.0 : 0.0);
                        camera_gpu_conversion_active_[index].set(decoded.back()->converted_on_gpu ? 1.0 : 0.0);
                        if (nvdec) nvdec_frames_.increment();
                        else software_decode_frames_.increment();
                    }
                    ended += !decoded.back().has_value();
                }
                const auto batch_decode_ms = elapsed_ms(decode_start);
                batch_decode_ms_.observe(batch_decode_ms);
                last_batch_decode_ms_.set(batch_decode_ms);
                if (ended == readers_.size()) {
                    if (config_.loop) {
                        if (batch_index == 0) {
                            throw std::runtime_error("cannot loop video feeds that contain no frames");
                        }
                        for (std::size_t index = 0; index < readers_.size(); ++index) {
                            readers_[index] = std::make_unique<capture::video::VideoFileReader>(
                                config_.cameras[index].path, config_.cuda_device);
                        }
                        continue;
                    }
                    finished_ = true;
                    break;
                }
                if (ended != 0) {
                    for (std::size_t index = 0; index < decoded.size(); ++index) {
                        if (!decoded[index]) {
                            throw std::runtime_error(
                                "video files have different frame counts; camera " +
                                std::to_string(config_.cameras[index].camera_id) +
                                " ended at batch " + std::to_string(batch_index));
                        }
                    }
                }

                auto source_time = decoded.front()->presentation_time;
                if (batch_index != 0 && source_time <= previous_time) {
                    const auto fallback = std::chrono::nanoseconds(
                        static_cast<std::int64_t>(1'000'000'000.0 / frame_rates.front()));
                    source_time = previous_time + fallback;
                }
                previous_time = source_time;
                double pacing_wait_ms{};
                if (config_.realtime) {
                    const auto pacing_start = std::chrono::steady_clock::now();
                    sleep_until_deadline(playback_start + source_time);
                    pacing_wait_ms = elapsed_ms(pacing_start);
                }
                const auto capture_time = capture_origin + source_time;
                FrameBatch batch;
                batch.reserve(decoded.size());
                double upload_ms{};
                for (std::size_t index = 0; index < decoded.size(); ++index) {
                    auto& source = *decoded[index];
                    if (source.extent.width != readers_[index]->extent().width ||
                        source.extent.height != readers_[index]->extent().height) {
                        throw std::runtime_error("video frame dimensions changed while decoding");
                    }
                    if (!source.converted_on_gpu) {
                        if (source.bgr.empty()) {
                            throw std::runtime_error("software-decoded video frame has no BGR pixels");
                        }
                        const auto row_bytes = static_cast<std::size_t>(source.extent.width) * 3;
                        const auto upload_start = std::chrono::steady_clock::now();
                        infrastructure::gpu::check_cuda(
                            cudaMemcpy2D(buffers[index]->data, buffers[index]->stride_bytes,
                                         source.bgr.data(), row_bytes, row_bytes,
                                         source.extent.height, cudaMemcpyHostToDevice),
                            "upload software-decoded video frame");
                        upload_ms += elapsed_ms(upload_start);
                    }
                    auto* output_buffer = &*buffers[index];
                    auto output_frame_extent = source.extent;
                    const auto rotation = config_.cameras[index].rotation;
                    if (rotation != FrameRotation::None) {
                        auto& rotated = *rotated_buffers[index];
                        capture::launch_bgr_rotation(
                            buffers[index]->data, buffers[index]->stride_bytes,
                            rotated.data, rotated.stride_bytes, source.extent.width,
                            source.extent.height, rotation, nullptr);
                        infrastructure::gpu::check_cuda(
                            cudaStreamSynchronize(nullptr), "wait for video frame rotation");
                        output_frame_extent = output_extent(source.extent, rotation);
                        output_buffer = &rotated;
                    }
                    auto ready = std::make_shared<infrastructure::gpu::CudaEvent>();
                    ready->record(nullptr);
                    FrameTiming timing;
                    timing.source_time = source_time;
                    timing.estimated_capture_time = capture_time;
                    timing.host_arrival_time = std::chrono::steady_clock::now();
                    timing.ready_time = timing.host_arrival_time;
                    timing.clock_quality = ClockQuality::Stable;
                    Frame frame;
                    frame.camera = config_.cameras[index].camera_id;
                    frame.sequence = batch_index;
                    frame.extent = output_frame_extent;
                    frame.format = PixelFormat::Bgr8;
                    frame.buffer = std::move(*output_buffer);
                    frame.ready = std::move(ready);
                    frame.timing = timing;
                    batch.push_back(std::move(frame));
                }
                if (stop.stop_requested()) {
                    break;
                }
                batch_pool_wait_ms_.observe(pool_wait_ms);
                batch_upload_ms_.observe(upload_ms);
                last_batch_pool_wait_ms_.set(pool_wait_ms);
                last_batch_upload_ms_.set(upload_ms);
                const auto submit_start = std::chrono::steady_clock::now();
                const auto result = output_.send(Packet{batch_index, std::move(batch), std::nullopt});
                const auto submit_wait_ms = elapsed_ms(submit_start);
                batch_submit_wait_ms_.observe(submit_wait_ms);
                last_batch_submit_wait_ms_.set(submit_wait_ms);
                if (result == SendResult::Closed) {
                    break;
                }
                const auto emit_time = std::chrono::steady_clock::now();
                if (previous_emit != std::chrono::steady_clock::time_point{}) {
                    const auto interval_ms = std::chrono::duration<double, std::milli>(emit_time - previous_emit).count();
                    if (interval_ms > 0.0) {
                        batch_interarrival_ms_.observe(interval_ms);
                        batch_fps_.set(1000.0 / interval_ms);
                    }
                }
                previous_emit = emit_time;
                const auto work_ms = elapsed_ms(batch_start) - pacing_wait_ms;
                batch_work_ms_.observe(work_ms);
                last_batch_work_ms_.set(work_ms);
                frames_emitted_.increment(readers_.size());
                batches_emitted_.increment();
                ++batch_index;
            }
        } catch (...) {
            {
                std::scoped_lock lock(error_mutex_);
                error_ = std::current_exception();
            }
            healthy_ = false;
        }
        output_.close();
        batch_fps_.set(0.0);
        ingestion_active_.set(0.0);
        finished_ = true;
    }

    static double elapsed_ms(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }

    static void sleep_until_deadline(std::chrono::steady_clock::time_point deadline) {
#ifdef _WIN32
        while (std::chrono::steady_clock::now() < deadline) {
            const auto remaining = deadline - std::chrono::steady_clock::now();
            const auto rounded = std::chrono::ceil<std::chrono::milliseconds>(remaining).count();
            const auto wait_ms = static_cast<DWORD>(std::clamp<std::int64_t>(
                rounded, 1, static_cast<std::int64_t>(MAXDWORD)));
            ::Sleep(wait_ms);
        }
#else
        std::this_thread::sleep_until(deadline);
#endif
    }

    void stop_producing() {
        if (worker_.joinable()) {
            worker_.request_stop();
        }
    }

    void wait() {
        if (worker_.joinable()) {
            worker_.join();
        }
        std::exception_ptr error;
        {
            std::scoped_lock lock(error_mutex_);
            error = error_;
        }
        if (error) {
            std::rethrow_exception(error);
        }
    }

    void stop() {
        stop_producing();
        try {
            wait();
        } catch (...) {
        }
    }

    std::vector<Extent2D> camera_extents() const {
        std::vector<Extent2D> result;
        result.reserve(readers_.size());
        for (const auto& reader : readers_) {
            const auto index = result.size();
            result.push_back(output_extent(reader->extent(), config_.cameras[index].rotation));
        }
        return result;
    }

    std::vector<VideoDecodeStatus> camera_decode_status() const {
        std::vector<VideoDecodeStatus> result;
        result.reserve(readers_.size());
        for (std::size_t index = 0; index < readers_.size(); ++index) {
            result.push_back(readers_[index]->decode_status(config_.cameras[index].camera_id));
        }
        return result;
    }

    SynchronizedVideoConfig config_;
    Channel<Packet>& output_;
    infrastructure::metrics::Counter frames_emitted_;
    infrastructure::metrics::Counter batches_emitted_;
    infrastructure::metrics::Counter nvdec_frames_;
    infrastructure::metrics::Counter software_decode_frames_;
    infrastructure::metrics::Histogram batch_decode_ms_;
    infrastructure::metrics::Histogram batch_pool_wait_ms_;
    infrastructure::metrics::Histogram batch_upload_ms_;
    infrastructure::metrics::Histogram batch_submit_wait_ms_;
    infrastructure::metrics::Histogram batch_work_ms_;
    infrastructure::metrics::Histogram batch_interarrival_ms_;
    infrastructure::metrics::Gauge source_fps_;
    infrastructure::metrics::Gauge batch_fps_;
    infrastructure::metrics::Gauge ingestion_active_;
    infrastructure::metrics::Gauge last_batch_decode_ms_;
    infrastructure::metrics::Gauge last_batch_pool_wait_ms_;
    infrastructure::metrics::Gauge last_batch_upload_ms_;
    infrastructure::metrics::Gauge last_batch_submit_wait_ms_;
    infrastructure::metrics::Gauge last_batch_work_ms_;
    std::vector<infrastructure::metrics::Histogram> camera_decode_ms_;
    std::vector<infrastructure::metrics::Gauge> camera_source_fps_;
    std::vector<infrastructure::metrics::Gauge> camera_last_decode_ms_;
    std::vector<infrastructure::metrics::Gauge> camera_nvdec_active_;
    std::vector<infrastructure::metrics::Gauge> camera_gpu_conversion_active_;
    std::vector<std::unique_ptr<capture::video::VideoFileReader>> readers_;
    std::jthread worker_;
    std::atomic_bool healthy_{false};
    std::atomic_bool finished_{false};
    std::mutex error_mutex_;
    std::exception_ptr error_;
};

SynchronizedVideoStage::SynchronizedVideoStage(SynchronizedVideoConfig config,
                                               Channel<Packet>& output,
                                               infrastructure::metrics::MetricRegistry& metrics)
    : impl_(std::make_unique<Impl>(std::move(config), output, metrics)) {}
SynchronizedVideoStage::~SynchronizedVideoStage() { stop(); }
void SynchronizedVideoStage::start() { impl_->start(); }
void SynchronizedVideoStage::stop_producing() { impl_->stop_producing(); }
void SynchronizedVideoStage::wait() { impl_->wait(); }
void SynchronizedVideoStage::stop() { impl_->stop(); }
bool SynchronizedVideoStage::healthy() const noexcept { return impl_->healthy_; }
bool SynchronizedVideoStage::finished() const noexcept { return impl_->finished_; }
std::vector<Extent2D> SynchronizedVideoStage::camera_extents() const {
    return impl_->camera_extents();
}

std::vector<VideoDecodeStatus> SynchronizedVideoStage::camera_decode_status() const {
    return impl_->camera_decode_status();
}

} // namespace iris
