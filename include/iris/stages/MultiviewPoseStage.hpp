#pragma once

#include "iris/pipeline/Stage.hpp"
#include "iris/infrastructure/metrics/MetricRegistry.hpp"
#include "iris/stages/pose/PoseConfig.hpp"

#include <memory>

namespace iris {

// Runs RTMO-S detection and emits COCO-17 image keypoints. In multiview mode it
// selects one person per camera and triangulates the keypoints; in 2-D mode it
// accepts one camera and skips triangulation. It is deliberately separate from
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
