#pragma once
#include "iris/infrastructure/metrics/MetricRegistry.hpp"
#include "iris/pipeline/Channel.hpp"
#include "iris/pipeline/Packet.hpp"
#include "iris/stages/capture/CaptureConfig.hpp"
#include <memory>
#include <string>
namespace iris {
class CaptureStage {
  public:
    CaptureStage(CaptureConfig, Channel<Packet>&, infrastructure::metrics::MetricRegistry&);
    CaptureStage(CameraId, CaptureConfig, std::string metrics_prefix, std::string channel_prefix,
                 Channel<Packet>&, infrastructure::metrics::MetricRegistry&);
    ~CaptureStage();
    CaptureStage(const CaptureStage&) = delete;
    void start();
    // Stop reading and decoding new samples; wait() remains responsible for joining workers.
    void stop_producing();
    void stop();
    void wait();
    bool healthy() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace iris
