#include "stages/capture/timing/CaptureClock.hpp"
#include <algorithm>
#include <cmath>
namespace iris::capture {
CaptureClock::CaptureClock(std::size_t w) : window_(std::max<std::size_t>(w, 8)) {}
void CaptureClock::reset() {
    points_.clear();
    slope_ = 1;
    offset_ = 0;
    last_estimate_ = {};
}
void CaptureClock::fit() {
    if (points_.size() < 2) {
        return;
    }
    double source_mean{};
    double host_mean{};
    for (const auto& point : points_) {
        source_mean += point.source_ns;
        host_mean += point.host_ns;
    }
    const auto count = static_cast<double>(points_.size());
    source_mean /= count;
    host_mean /= count;

    double source_variance{};
    double covariance{};
    for (const auto& point : points_) {
        const double centred_source = point.source_ns - source_mean;
        const double centred_host = point.host_ns - host_mean;
        source_variance += centred_source * centred_source;
        covariance += centred_source * centred_host;
    }
    if (source_variance > 1.0) {
        slope_ = covariance / source_variance;
        offset_ = host_mean - slope_ * source_mean;
    }
}
ClockEstimate CaptureClock::observe(std::chrono::nanoseconds source, MonotonicTime host,
                                    bool disc) {
    bool did_reset = disc || (!points_.empty() &&
                              static_cast<double>(source.count()) <= points_.back().source_ns);
    if (did_reset) {
        reset();
    }
    double h = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(host.time_since_epoch()).count());
    points_.push_back({static_cast<double>(source.count()), h});
    if (points_.size() > window_) {
        points_.pop_front();
    }
    fit();
    double estimate = offset_ + slope_ * static_cast<double>(source.count());
    if (last_estimate_ != MonotonicTime{}) {
        estimate = std::max(
            estimate, static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                              last_estimate_.time_since_epoch())
                                              .count() +
                                          1));
    }
    auto time = MonotonicTime(std::chrono::nanoseconds(static_cast<std::int64_t>(estimate)));
    last_estimate_ = time;
    double residual = std::abs(h - estimate) / 1000.0;
    ClockQuality q = points_.size() < 8
                         ? ClockQuality::WarmingUp
                         : (residual > 5000 ? ClockQuality::Degraded : ClockQuality::Stable);
    return {time, q, (slope_ - 1.0) * 1'000'000.0, residual, did_reset};
}
} // namespace iris::capture
