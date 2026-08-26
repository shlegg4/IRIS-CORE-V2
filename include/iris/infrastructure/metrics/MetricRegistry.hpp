#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
namespace iris::infrastructure::metrics {
struct HistogramSnapshot {
    std::vector<double> bounds;
    std::vector<std::uint64_t> counts;
    std::uint64_t count{};
    double sum{};
};
struct MetricsSnapshot {
    std::unordered_map<std::string, std::uint64_t> counters;
    std::unordered_map<std::string, double> gauges;
    std::unordered_map<std::string, HistogramSnapshot> histograms;
};
class Counter {
  public:
    void increment(std::uint64_t n = 1) const;
    std::uint64_t value() const;

  private:
    friend class MetricRegistry;
    std::shared_ptr<std::atomic_uint64_t> state_;
};
class Gauge {
  public:
    void set(double) const;
    void add(double) const;
    double value() const;

  private:
    friend class MetricRegistry;
    std::shared_ptr<std::atomic<double>> state_;
};
class Histogram {
  public:
    void observe(double) const;

  private:
    friend class MetricRegistry;
    struct State {
        explicit State(std::vector<double>);
        std::vector<double> bounds;
        std::vector<std::uint64_t> counts;
        std::uint64_t count{};
        double sum{};
        std::mutex mutex;
    };
    std::shared_ptr<State> state_;
};
class MetricRegistry {
  public:
    Counter counter(std::string name);
    Gauge gauge(std::string name);
    Histogram histogram(std::string name, std::vector<double> bounds);
    MetricsSnapshot snapshot() const;

  private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<std::atomic_uint64_t>> counters_;
    std::unordered_map<std::string, std::shared_ptr<std::atomic<double>>> gauges_;
    std::unordered_map<std::string, std::shared_ptr<Histogram::State>> histograms_;
};
} // namespace iris::infrastructure::metrics
