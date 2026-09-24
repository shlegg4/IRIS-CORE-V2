#include "iris/stages/PoseStage.hpp"

#include <stdexcept>
#include <utility>

namespace iris {

class PoseStage::Impl {
  public:
    explicit Impl(PoseConfig config) : config_(std::move(config)) {}

    void start() {
        if (!config_.model_path.empty())
            throw std::runtime_error("PEAR HMR monocular inference is disabled in this build");
    }
    void stop() {}
    void process(Packet&) {}

  private:
    PoseConfig config_;
};

PoseStage::PoseStage(Channel<Packet>& input, Channel<Packet>* output, PoseConfig config,
                     infrastructure::metrics::MetricRegistry*)
    : Stage(input, output), impl_(std::make_unique<Impl>(std::move(config))) {}
PoseStage::~PoseStage() { stop(); }
void PoseStage::start() {
    impl_->start();
    Stage::start();
}
void PoseStage::stop() {
    Stage::stop();
    impl_->stop();
}
void PoseStage::process(Packet& packet) { impl_->process(packet); }

} // namespace iris
