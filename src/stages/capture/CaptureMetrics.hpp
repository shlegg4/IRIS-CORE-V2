#pragma once
#include "iris/infrastructure/metrics/MetricRegistry.hpp"
#include <string>
namespace iris::capture {
struct CaptureMetrics {
    infrastructure::metrics::Counter samples_received, frames_emitted, decode_failures,
        decode_invalid_header, decode_dimension_mismatch, decode_nvjpeg, decode_unsupported_format,
        decode_exceptions, source_errors, timestamp_regressions, clock_resets, pool_exhaustions,
        rotation_frames, rotation_failures, rotation_pool_exhaustions;
    infrastructure::metrics::Gauge up, pool_available, clock_drift_ppm, clock_residual_us,
        last_frame_age_ms, rotation_degrees, rotation_pool_available;
    infrastructure::metrics::Histogram interframe_ms, queue_wait_ms, decode_submit_ms,
        capture_to_emit_ms, rotation_submit_ms;
    explicit CaptureMetrics(infrastructure::metrics::MetricRegistry& r,
                            const std::string& prefix = "iris_capture")
        : samples_received(r.counter(prefix + "_samples_received_total")),
          frames_emitted(r.counter(prefix + "_frames_emitted_total")),
          decode_failures(r.counter(prefix + "_decode_failures_total")),
          decode_invalid_header(r.counter(prefix + "_decode_invalid_header_total")),
          decode_dimension_mismatch(r.counter(prefix + "_decode_dimension_mismatch_total")),
          decode_nvjpeg(r.counter(prefix + "_decode_nvjpeg_total")),
          decode_unsupported_format(r.counter(prefix + "_decode_unsupported_format_total")),
          decode_exceptions(r.counter(prefix + "_decode_exceptions_total")),
          source_errors(r.counter(prefix + "_source_errors_total")),
          timestamp_regressions(r.counter(prefix + "_timestamp_regressions_total")),
          clock_resets(r.counter(prefix + "_clock_resets_total")),
          pool_exhaustions(r.counter(prefix + "_pool_exhaustions_total")),
          rotation_frames(r.counter(prefix + "_rotation_frames_total")),
          rotation_failures(r.counter(prefix + "_rotation_failures_total")),
          rotation_pool_exhaustions(r.counter(prefix + "_rotation_pool_exhaustions_total")),
          up(r.gauge(prefix + "_up")), pool_available(r.gauge(prefix + "_pool_available")),
          clock_drift_ppm(r.gauge(prefix + "_clock_drift_ppm")),
          clock_residual_us(r.gauge(prefix + "_clock_residual_us")),
          last_frame_age_ms(r.gauge(prefix + "_last_frame_age_ms")),
          rotation_degrees(r.gauge(prefix + "_rotation_degrees")),
          rotation_pool_available(r.gauge(prefix + "_rotation_pool_available")),
          interframe_ms(r.histogram(prefix + "_interframe_interval_ms",
                                    {5, 10, 16.7, 20, 33.4, 40, 50, 100})),
          queue_wait_ms(r.histogram(prefix + "_queue_wait_ms", {0.1, 0.25, 0.5, 1, 2, 5, 10, 20})),
          decode_submit_ms(
              r.histogram(prefix + "_decode_submit_ms", {0.1, 0.25, 0.5, 1, 2, 5, 10})),
          capture_to_emit_ms(
              r.histogram(prefix + "_capture_to_emit_ms", {1, 2, 5, 10, 16.7, 33.4, 50, 100})),
          rotation_submit_ms(
              r.histogram(prefix + "_rotation_submit_ms", {0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5})) {}
};
} // namespace iris::capture
