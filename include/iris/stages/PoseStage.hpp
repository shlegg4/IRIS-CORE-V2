#pragma once

#include "iris/pipeline/Stage.hpp"
#include "iris/infrastructure/metrics/MetricRegistry.hpp"
#include "iris/stages/pose/PoseConfig.hpp"

#include <memory>

namespace iris {

class PoseStage final : public Stage {
  public:
    PoseStage(Channel<Packet>&, Channel<Packet>* = nullptr, PoseConfig = {},
              infrastructure::metrics::MetricRegistry* = nullptr);
    ~PoseStage() override;

    void start() override;
    void stop() override;

  protected:
    void process(Packet& packet) override;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace iris
