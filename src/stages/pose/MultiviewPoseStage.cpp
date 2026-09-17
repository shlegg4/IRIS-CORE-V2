#include "iris/stages/MultiviewPoseStage.hpp"

#include <filesystem>
#include <stdexcept>
#include <utility>

namespace iris {

class MultiviewPoseStage::Impl {
  public:
    explicit Impl(PoseConfig config) : config_(std::move(config)) {}
    void start() {
        if (config_.multiview_engine_path.empty()) return;
        if (!std::filesystem::is_regular_file(config_.multiview_engine_path))
            throw std::runtime_error("multiview TensorRT engine does not exist: " + config_.multiview_engine_path.string());
        for (const auto& calibration : config_.multiview_calibration)
            if (!calibration.calibrated)
                throw std::runtime_error("multiview TensorRT requires calibration for all three cameras");
#ifndef IRIS_HAS_TENSORRT
        throw std::runtime_error("multiview TensorRT was configured, but IRIS was built without TensorRT support; set IRIS_TENSORRT_ROOT and rebuild");
#else
        // TensorRT engine construction/inference is compiled in the TensorRT-enabled
        // implementation. Keeping this guard makes the base build usable on hosts
        // that only run the existing monocular TorchScript stage.
        started_ = true;
#endif
    }
    void stop() { started_ = false; }
    void process(Packet& packet) {
        if (config_.multiview_engine_path.empty()) return;
        if (!started_) throw std::logic_error("multiview pose stage was not started");
        if (packet.frames.size() != 3)
            throw std::runtime_error("multiview TensorRT engine requires exactly three synchronized frames");
#ifndef IRIS_HAS_TENSORRT
        (void)packet;
        throw std::runtime_error("multiview TensorRT inference is unavailable in this build");
#else
        // TODO: bind images, R_w2c, t_w2c and letterboxed intrinsics, then copy
        // poses_3d/joint_valid/joint_scores into packet.multiview_poses.
#endif
    }
  private:
    PoseConfig config_;
    bool started_{};
};

MultiviewPoseStage::MultiviewPoseStage(Channel<Packet>& input, Channel<Packet>* output, PoseConfig config)
    : Stage(input, output), impl_(std::make_unique<Impl>(std::move(config))) {}
MultiviewPoseStage::~MultiviewPoseStage() { stop(); }
void MultiviewPoseStage::start() { impl_->start(); Stage::start(); }
void MultiviewPoseStage::stop() { Stage::stop(); impl_->stop(); }
void MultiviewPoseStage::process(Packet& packet) { impl_->process(packet); }

} // namespace iris
