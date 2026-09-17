#include "iris/stages/MultiviewPoseStage.hpp"
#include "iris/stages/pose/TensorRtMultiviewEngine.hpp"
#include "iris/infrastructure/gpu/CudaResources.hpp"
#include "iris/calibration/RigCalibration.hpp"

#include <filesystem>
#include <stdexcept>
#include <utility>
#include <ranges>

namespace iris {

class MultiviewPoseStage::Impl {
  public:
    explicit Impl(PoseConfig config) : config_(std::move(config)) {}
    void start() {
        if (config_.multiview_engine_path.empty()) return;
        if (!std::filesystem::is_regular_file(config_.multiview_engine_path))
            throw std::runtime_error("multiview TensorRT engine does not exist: " + config_.multiview_engine_path.string());
        refresh_calibration();
        for (const auto& calibration : config_.multiview_calibration)
            if (!calibration.calibrated)
                throw std::runtime_error("multiview TensorRT requires calibration for all three cameras");
#ifndef IRIS_HAS_TENSORRT
        throw std::runtime_error("multiview TensorRT was configured, but IRIS was built without TensorRT support; set IRIS_TENSORRT_ROOT and rebuild");
#else
        engine_ = std::make_unique<TensorRtMultiviewEngine>(config_);
        started_ = true;
#endif
    }
    void stop() { started_ = false; }
    void process(Packet& packet) {
        if (config_.multiview_engine_path.empty()) return;
        if (!started_) throw std::logic_error("multiview pose stage was not started");
        refresh_calibration();
        if (packet.frames.size() != 3)
            throw std::runtime_error("multiview TensorRT engine requires exactly three synchronized frames");
#ifndef IRIS_HAS_TENSORRT
        (void)packet;
        throw std::runtime_error("multiview TensorRT inference is unavailable in this build");
#else
        std::array<const void*, 3> buffers{};
        std::array<std::size_t, 3> strides{};
        std::array<std::uint32_t, 3> widths{};
        std::array<std::uint32_t, 3> heights{};
        for (std::size_t i = 0; i < 3; ++i) {
            const auto id = config_.multiview_calibration[i].camera_id;
            const auto frame = std::ranges::find(packet.frames, id, &Frame::camera);
            if (frame == packet.frames.end() || frame->format != PixelFormat::Bgr8 || !frame->buffer.data)
                throw std::runtime_error("multiview packet is missing a valid calibrated BGR camera frame");
            if (!frame->extent.width || !frame->extent.height || frame->buffer.stride_bytes < static_cast<std::size_t>(frame->extent.width) * 3)
                throw std::runtime_error("multiview frame has invalid dimensions or stride");
            if (frame->ready) frame->ready->synchronize();
            buffers[i] = frame->buffer.data;
            strides[i] = frame->buffer.stride_bytes;
            widths[i] = frame->extent.width;
            heights[i] = frame->extent.height;
        }
        TensorRtMultiviewResult result;
        engine_->infer(buffers, strides, widths, heights, result);
        std::array<MultiviewPose, 10> poses{};
        for (std::size_t person = 0; person < 10; ++person) {
            std::size_t valid_count = 0;
            for (std::size_t joint = 0; joint < 17; ++joint) {
                const auto base = person * 17 * 3 + joint * 3;
                poses[person].joints_3d[joint] = {result.poses_3d[base], result.poses_3d[base + 1], result.poses_3d[base + 2]};
                poses[person].joint_valid[joint] = result.joint_valid[person * 17 + joint] != 0;
                valid_count += poses[person].joint_valid[joint] ? 1U : 0U;
                for (std::size_t view = 0; view < 3; ++view)
                    poses[person].joint_scores[view][joint] = result.joint_scores[person * 3 * 17 + view * 17 + joint];
            }
            poses[person].active = valid_count >= 5;
        }
        packet.multiview_poses = std::move(poses);
#endif
    }
  private:
    void refresh_calibration() {
        if (!config_.calibration_store) return;
        const auto rig=config_.calibration_store->snapshot();
        if (!rig || rig->revision==calibration_revision_) return;
        if (rig->cameras.size()!=3) throw std::runtime_error("multiview pose requires a three-camera runtime calibration");
        for(std::size_t i=0;i<3;++i){const auto& source=rig->cameras[i];auto& target=config_.multiview_calibration[i];target.camera_id=source.camera_id;target.intrinsics=source.intrinsics;target.distortion=source.distortion;target.R_w2c=source.R_w2c;target.t_w2c=source.t_w2c;target.calibrated=true;}
        calibration_revision_=rig->revision;
#ifdef IRIS_HAS_TENSORRT
        if(started_) engine_=std::make_unique<TensorRtMultiviewEngine>(config_);
#endif
    }
    PoseConfig config_;
    std::unique_ptr<TensorRtMultiviewEngine> engine_;
    bool started_{};
    std::uint64_t calibration_revision_{};
};

MultiviewPoseStage::MultiviewPoseStage(Channel<Packet>& input, Channel<Packet>* output, PoseConfig config)
    : Stage(input, output), impl_(std::make_unique<Impl>(std::move(config))) {}
MultiviewPoseStage::~MultiviewPoseStage() { stop(); }
void MultiviewPoseStage::start() { impl_->start(); Stage::start(); }
void MultiviewPoseStage::stop() { Stage::stop(); impl_->stop(); }
void MultiviewPoseStage::process(Packet& packet) { impl_->process(packet); }

} // namespace iris
