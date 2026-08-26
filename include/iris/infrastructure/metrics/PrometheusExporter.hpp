#pragma once

#include "iris/infrastructure/metrics/MetricRegistry.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace iris::infrastructure::metrics {

std::string format_prometheus(const MetricsSnapshot& snapshot);

class PrometheusExporter {
  public:
    explicit PrometheusExporter(const MetricRegistry&, std::uint16_t port = 9464);
    ~PrometheusExporter();

    PrometheusExporter(const PrometheusExporter&) = delete;
    PrometheusExporter& operator=(const PrometheusExporter&) = delete;

    void start();
    void stop();

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace iris::infrastructure::metrics
