#include "iris/stages/capture/CaptureStage.hpp"
#include "stages/capture/CaptureMetrics.hpp"
#include "stages/capture/CaptureSample.hpp"
#include "stages/capture/decoding/GpuDecoder.hpp"
#include "stages/capture/media_foundation/MediaFoundationSource.hpp"
#include "stages/capture/timing/CaptureClock.hpp"
#include <atomic>
#include <exception>
#include <future>
#include <mutex>
#include <objbase.h>
#include <thread>
namespace iris {
using namespace std::chrono;
struct CaptureStage::Impl {
    CameraId camera_id;
    CaptureConfig config;
    Channel<Packet>& output;
    capture::CaptureMetrics metrics;
    infrastructure::metrics::ChannelMetrics sample_channel_metrics;
    std::unique_ptr<Channel<capture::CaptureSample>> samples;
    capture::MediaFoundationSource source;
    std::unique_ptr<capture::GpuDecoder> decoder;
    capture::CaptureClock clock;
    std::jthread reader, decode;
    std::atomic_bool healthy{};
    std::atomic_bool reset_clock_after_reconnect{};
    Extent2D decoder_extent{};
    std::mutex error_mutex;
    std::exception_ptr error;
    Impl(CameraId id, CaptureConfig c, std::string metrics_prefix, std::string channel_prefix,
         Channel<Packet>& o, infrastructure::metrics::MetricRegistry& r)
        : camera_id(id), config(std::move(c)), output(o), metrics(r, metrics_prefix),
          sample_channel_metrics(
              infrastructure::metrics::register_channel_metrics(r, channel_prefix)) {}
    void fail(std::exception_ptr e) {
        std::scoped_lock l(error_mutex);
        if (!error) {
            error = e;
        }
        healthy = false;
        metrics.up.set(0);
        if (samples) {
            samples->close();
        }
    }
    void reader_loop(std::stop_token stop) {
        while (!stop.stop_requested()) {
            try {
                auto sample = source.read();
                if (!sample) {
                    continue;
                }
                metrics.samples_received.increment();
                if (samples->send(std::move(*sample)) == SendResult::Closed) {
                    break;
                }
            } catch (...) {
                if (stop.stop_requested()) {
                    break;
                }
                metrics.source_errors.increment();
                if (!config.reconnect) {
                    fail(std::current_exception());
                    break;
                }
                try {
                    source.close();
                    std::this_thread::sleep_for(milliseconds(100));
                    if (stop.stop_requested()) {
                        break;
                    }
                    const auto negotiated = source.open(config);
                    if (negotiated.extent.width != decoder_extent.width ||
                        negotiated.extent.height != decoder_extent.height) {
                        throw std::runtime_error(
                            "reconnected camera negotiated a different frame extent");
                    }
                    metrics.reconnects.increment();
                    reset_clock_after_reconnect.store(true);
                } catch (...) {
                    fail(std::current_exception());
                    break;
                }
            }
        }
        samples->close();
    }
    void decode_loop(std::stop_token stop) {
        try {
            MonotonicTime previous{};
            while (!stop.stop_requested()) {
                auto sample = samples->receive();
                if (!sample) {
                    break;
                }
                auto received = steady_clock::now();
                metrics.queue_wait_ms.observe(
                    duration<double, std::milli>(received - sample->host_arrival).count());
                metrics.sample_copy_ms.observe(sample->host_copy_ms);
                const bool discontinuity =
                    sample->discontinuity || reset_clock_after_reconnect.exchange(false);
                auto estimate =
                    clock.observe(sample->source_timestamp, sample->host_arrival, discontinuity);
                if (estimate.reset) {
                    metrics.clock_resets.increment();
                    metrics.timestamp_regressions.increment();
                }
                metrics.clock_drift_ppm.set(estimate.drift_ppm);
                metrics.clock_residual_us.set(estimate.residual_us);
                if (previous != MonotonicTime{}) {
                    metrics.interframe_ms.observe(
                        duration<double, std::milli>(estimate.capture_time - previous).count());
                }
                previous = estimate.capture_time;
                FrameTiming timing{
                    sample->source_timestamp, estimate.capture_time, sample->host_arrival, {}, {},
                    estimate.quality};
                auto begin = steady_clock::now();
                auto result = decoder->decode(*sample, timing);
                metrics.decode_submit_ms.observe(
                    duration<double, std::milli>(steady_clock::now() - begin).count());
                if (result.header_parse_ms > 0) {
                    metrics.decode_header_ms.observe(result.header_parse_ms);
                }
                if (result.gpu_submit_ms > 0) {
                    metrics.decode_gpu_submit_ms.observe(result.gpu_submit_ms);
                }
                if (result.nvjpeg_host_ms > 0) {
                    metrics.nvjpeg_host_ms.observe(result.nvjpeg_host_ms);
                }
                if (result.nvjpeg_device_ms > 0) {
                    metrics.nvjpeg_device_ms.observe(result.nvjpeg_device_ms);
                }
                metrics.pool_available.set(static_cast<double>(decoder->pool_available()));
                metrics.rotation_pool_available.set(
                    static_cast<double>(decoder->rotation_pool_available()));
                if (!result.frame) {
                    metrics.decode_failures.increment();
                    switch (result.failure) {
                    case capture::DecodeFailureReason::PoolExhausted:
                        metrics.pool_exhaustions.increment();
                        break;
                    case capture::DecodeFailureReason::RotationPoolExhausted:
                        metrics.rotation_failures.increment();
                        metrics.rotation_pool_exhaustions.increment();
                        break;
                    case capture::DecodeFailureReason::InvalidJpegHeader:
                        metrics.decode_invalid_header.increment();
                        break;
                    case capture::DecodeFailureReason::DimensionMismatch:
                        metrics.decode_dimension_mismatch.increment();
                        break;
                    case capture::DecodeFailureReason::NvjpegDecode:
                        metrics.decode_nvjpeg.increment();
                        break;
                    case capture::DecodeFailureReason::UnsupportedFormat:
                        metrics.decode_unsupported_format.increment();
                        break;
                    case capture::DecodeFailureReason::None:
                        break;
                    }
                    continue;
                }
                if (result.rotation_applied) {
                    metrics.rotation_frames.increment();
                    metrics.rotation_submit_ms.observe(result.rotation_submit_ms);
                }
                result.frame->camera = camera_id;
                auto now = steady_clock::now();
                const auto capture_to_emit_ms =
                    duration<double, std::milli>(now - estimate.capture_time).count();
                metrics.capture_to_emit_ms.observe(capture_to_emit_ms);
                metrics.last_capture_to_emit_ms.set(capture_to_emit_ms);
                metrics.last_frame_age_ms.set(
                    duration<double, std::milli>(now - estimate.capture_time).count());
                Packet packet{result.frame->sequence, FrameBatch{std::move(*result.frame)},
                              std::nullopt};
                if (output.send(std::move(packet)) == SendResult::Closed) {
                    break;
                }
                metrics.frames_emitted.increment();
            }
        } catch (...) {
            metrics.decode_failures.increment();
            metrics.decode_exceptions.increment();
            if (config.rotation != FrameRotation::None) {
                metrics.rotation_failures.increment();
            }
            fail(std::current_exception());
        }
        output.close();
    }
};
CaptureStage::CaptureStage(CaptureConfig c, Channel<Packet>& o,
                           infrastructure::metrics::MetricRegistry& r)
    : CaptureStage(0, std::move(c), "iris_capture", "iris_channel_capture_samples", o, r) {}
CaptureStage::CaptureStage(CameraId id, CaptureConfig c, std::string metrics_prefix,
                           std::string channel_prefix, Channel<Packet>& o,
                           infrastructure::metrics::MetricRegistry& r)
    : impl_(std::make_unique<Impl>(id, std::move(c), std::move(metrics_prefix),
                                   std::move(channel_prefix), o, r)) {}
CaptureStage::~CaptureStage() { stop(); }
void CaptureStage::start() {
    if (impl_->reader.joinable()) {
        return;
    }
    impl_->samples = std::make_unique<Channel<capture::CaptureSample>>(
        impl_->config.sample_queue_capacity, impl_->config.overflow, impl_->sample_channel_metrics);
    std::promise<capture::NegotiatedFormat> opened;
    auto ready = opened.get_future();
    impl_->reader =
        std::jthread([p = impl_.get(), promise = std::move(opened)](std::stop_token s) mutable {
            auto hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
            bool com = SUCCEEDED(hr);
            try {
                promise.set_value(p->source.open(p->config));
                p->reader_loop(s);
            } catch (...) {
                try {
                    promise.set_exception(std::current_exception());
                } catch (...) {
                }
                p->metrics.source_errors.increment();
                p->fail(std::current_exception());
            }
            if (com) {
                CoUninitialize();
            }
        });
    auto negotiated = ready.get();
    impl_->decoder = std::make_unique<capture::GpuDecoder>(
        impl_->config.cuda_device, negotiated.extent, impl_->config.frame_pool_capacity,
        impl_->config.rotation);
    impl_->decoder_extent = negotiated.extent;
    impl_->metrics.rotation_degrees.set(rotation_degrees(impl_->config.rotation));
    impl_->healthy = true;
    impl_->metrics.up.set(1);
    impl_->decode = std::jthread([p = impl_.get()](std::stop_token s) { p->decode_loop(s); });
}
void CaptureStage::stop_producing() {
    if (!impl_) {
        return;
    }
    if (impl_->reader.joinable()) {
        impl_->reader.request_stop();
    }
    impl_->source.close();
    if (impl_->samples) {
        impl_->samples->close();
    }
    if (impl_->decode.joinable()) {
        impl_->decode.request_stop();
    }
}
void CaptureStage::stop() {
    stop_producing();
    try {
        wait();
    } catch (...) {
    }
    impl_->healthy = false;
    impl_->metrics.up.set(0);
}
void CaptureStage::wait() {
    if (impl_->reader.joinable()) {
        impl_->reader.join();
    }
    if (impl_->decode.joinable()) {
        impl_->decode.join();
    }
    std::exception_ptr e;
    {
        std::scoped_lock l(impl_->error_mutex);
        e = impl_->error;
    }
    if (e) {
        std::rethrow_exception(e);
    }
}
bool CaptureStage::healthy() const noexcept { return impl_->healthy; }
} // namespace iris
