#include "iris/infrastructure/metrics/MetricRegistry.hpp"
#include <algorithm>
#include <stdexcept>
namespace iris::infrastructure::metrics {
void Counter::increment(std::uint64_t n) const { state_->fetch_add(n, std::memory_order_relaxed); }
std::uint64_t Counter::value() const { return state_->load(std::memory_order_relaxed); }
void Gauge::set(double v) const { state_->store(v, std::memory_order_relaxed); }
void Gauge::add(double v) const {
    double old = state_->load();
    while (!state_->compare_exchange_weak(old, old + v)) {
    }
}
double Gauge::value() const { return state_->load(std::memory_order_relaxed); }
Histogram::State::State(std::vector<double> b) : bounds(std::move(b)), counts(bounds.size() + 1) {
    std::sort(bounds.begin(), bounds.end());
}
void Histogram::observe(double v) const {
    std::scoped_lock l(state_->mutex);
    auto i =
        std::upper_bound(state_->bounds.begin(), state_->bounds.end(), v) - state_->bounds.begin();
    ++state_->counts[static_cast<std::size_t>(i)];
    ++state_->count;
    state_->sum += v;
}
Counter MetricRegistry::counter(std::string n) {
    std::scoped_lock l(mutex_);
    auto& v = counters_[n];
    if (!v) {
        v = std::make_shared<std::atomic_uint64_t>();
    }
    Counter h;
    h.state_ = v;
    return h;
}
Gauge MetricRegistry::gauge(std::string n) {
    std::scoped_lock l(mutex_);
    auto& v = gauges_[n];
    if (!v) {
        v = std::make_shared<std::atomic<double>>();
    }
    Gauge h;
    h.state_ = v;
    return h;
}
Histogram MetricRegistry::histogram(std::string n, std::vector<double> b) {
    std::scoped_lock l(mutex_);
    auto& v = histograms_[n];
    if (!v) {
        v = std::make_shared<Histogram::State>(std::move(b));
    }
    Histogram h;
    h.state_ = v;
    return h;
}
MetricsSnapshot MetricRegistry::snapshot() const {
    std::scoped_lock l(mutex_);
    MetricsSnapshot s;
    for (auto& [n, v] : counters_) {
        s.counters[n] = v->load();
    }
    for (auto& [n, v] : gauges_) {
        s.gauges[n] = v->load();
    }
    for (auto& [n, v] : histograms_) {
        std::scoped_lock hl(v->mutex);
        s.histograms[n] = {v->bounds, v->counts, v->count, v->sum};
    }
    return s;
}
} // namespace iris::infrastructure::metrics
