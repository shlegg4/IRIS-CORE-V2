#pragma once
#include "iris/infrastructure/metrics/MetricRegistry.hpp"
#include <chrono>
#include <filesystem>
#include <stop_token>
#include <thread>
namespace iris::infrastructure::metrics {
class MetricsExporter {
  public:
    MetricsExporter(const MetricRegistry&, std::filesystem::path,
                    std::chrono::milliseconds interval = std::chrono::seconds(1));
    ~MetricsExporter();
    void start();
    void stop();

  private:
    void run(std::stop_token);
    const MetricRegistry& registry_;
    std::filesystem::path path_;
    std::chrono::milliseconds interval_;
    std::jthread worker_;
};
} // namespace iris::infrastructure::metrics
