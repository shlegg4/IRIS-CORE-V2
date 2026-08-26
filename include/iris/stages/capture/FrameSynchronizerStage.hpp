#pragma once

#include "iris/infrastructure/metrics/MetricRegistry.hpp"
#include "iris/pipeline/Channel.hpp"
#include "iris/pipeline/Packet.hpp"
#include "iris/stages/capture/CaptureConfig.hpp"

#include <memory>
#include <vector>

namespace iris {
class FrameSynchronizerStage final {
  public:
    FrameSynchronizerStage(std::vector<Channel<Packet>*> inputs, Channel<Packet>& output,
                           const MultiCameraCaptureConfig&,
                           infrastructure::metrics::MetricRegistry&);
    ~FrameSynchronizerStage();
    void start();
    void wait();
    void stop();

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace iris
