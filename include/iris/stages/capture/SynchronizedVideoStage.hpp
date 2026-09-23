#pragma once

#include "iris/infrastructure/metrics/MetricRegistry.hpp"
#include "iris/pipeline/Channel.hpp"
#include "iris/pipeline/Packet.hpp"
#include "iris/stages/capture/SynchronizedVideoConfig.hpp"

#include <memory>
#include <vector>

namespace iris {

class SynchronizedVideoStage final {
  public:
    SynchronizedVideoStage(SynchronizedVideoConfig, Channel<Packet>&,
                           infrastructure::metrics::MetricRegistry&);
    ~SynchronizedVideoStage();
    SynchronizedVideoStage(const SynchronizedVideoStage&) = delete;
    SynchronizedVideoStage& operator=(const SynchronizedVideoStage&) = delete;

    void start();
    void stop_producing();
    void wait();
    void stop();
    [[nodiscard]] bool healthy() const noexcept;
    [[nodiscard]] bool finished() const noexcept;
    [[nodiscard]] std::vector<Extent2D> camera_extents() const;
    [[nodiscard]] std::vector<VideoDecodeStatus> camera_decode_status() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace iris
