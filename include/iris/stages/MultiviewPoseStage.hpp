#pragma once

#include "iris/pipeline/Stage.hpp"
#include "iris/stages/pose/PoseConfig.hpp"

#include <memory>

namespace iris {

// Runs the fixed [1,3,3,640,640] RTMO-S multiview TensorRT engine. It is
// deliberately separate from PoseStage: PoseStage remains the monocular
// TorchScript HMR path and can be enabled independently.
class MultiviewPoseStage final : public Stage {
  public:
    MultiviewPoseStage(Channel<Packet>&, Channel<Packet>* = nullptr, PoseConfig = {});
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
