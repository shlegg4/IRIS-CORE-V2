#pragma once

#include "iris/pipeline/Stage.hpp"
#include "iris/infrastructure/metrics/MetricRegistry.hpp"
#include "iris/stages/pose/PoseConfig.hpp"

#include <memory>

namespace iris {

// Runs batched RTMO-S detection, selects one person per camera, then
// triangulates its COCO-17 keypoints.  It is deliberately separate from
// PoseStage: PoseStage remains the monocular TorchScript HMR path.
class MultiviewPoseStage final : public Stage {
  public:
    MultiviewPoseStage(Channel<Packet>&, Channel<Packet>* = nullptr, PoseConfig = {},
                       infrastructure::metrics::MetricRegistry* = nullptr);
    ~MultiviewPoseStage() override;
    void start() override;
    void stop() override;

  protected:
    void process(Packet&) override;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace iris
