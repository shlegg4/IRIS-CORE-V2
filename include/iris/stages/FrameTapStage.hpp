#pragma once
#include "iris/pipeline/Stage.hpp"
#include <functional>
#include <utility>
namespace iris {
class FrameTapStage final : public Stage {
  public:
    FrameTapStage(Channel<Packet>& input, Channel<Packet>* output, std::function<void(const Packet&)> callback)
        : Stage(input, output), callback_(std::move(callback)) {}
  protected:
    void process(Packet& packet) override { callback_(packet); }
  private:
    std::function<void(const Packet&)> callback_;
};
}
