#include "stages/capture/timing/CaptureClock.hpp"
#include <cassert>
#include <chrono>
#include <cmath>
int main() {
    using namespace std::chrono;
    iris::capture::CaptureClock clock;
    auto base = steady_clock::now();
    iris::capture::ClockEstimate e;
    for (int i = 0; i < 120; ++i) {
        auto source = milliseconds(i * 10);
        auto host = base + source + nanoseconds(i * 100);
        e = clock.observe(source, host);
    }
    assert(e.quality == iris::ClockQuality::Stable);
    assert(std::abs(e.drift_ppm - 10.0) < 1.0);
    auto reset = clock.observe(milliseconds(1), base + seconds(2));
    assert(reset.reset);
    assert(reset.quality == iris::ClockQuality::WarmingUp);
}
