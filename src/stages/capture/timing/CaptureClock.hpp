#pragma once
#include "iris/pipeline/Frame.hpp"
#include <cstddef>
#include <deque>
namespace iris::capture {
struct ClockEstimate {
    MonotonicTime capture_time{};
    ClockQuality quality{ClockQuality::Uninitialised};
    double drift_ppm{}, residual_us{};
    bool reset{};
};
class CaptureClock {
  public:
    explicit CaptureClock(std::size_t window = 120);
    ClockEstimate observe(std::chrono::nanoseconds, MonotonicTime, bool discontinuity = false);
    void reset();

  private:
    struct Point {
        double source_ns{}, host_ns{};
    };
    void fit();
    std::size_t window_;
    std::deque<Point> points_;
    double slope_{1.0}, offset_{};
    MonotonicTime last_estimate_{};
};
} // namespace iris::capture
