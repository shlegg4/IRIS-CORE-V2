#pragma once

#include "iris/pipeline/Stage.hpp"

namespace iris {

class PoseStage final : public Stage {
  public:
    using Stage::Stage;

  protected:
    void process(Packet& packet) override;
};

} // namespace iris
