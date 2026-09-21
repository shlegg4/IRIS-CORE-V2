#include "iris/stages/MultiviewPoseStage.hpp"
#include "iris/stages/pose/TensorRtMultiviewEngine.hpp"
#include "iris/infrastructure/gpu/CudaResources.hpp"
#include "iris/calibration/RigCalibration.hpp"

#include <filesystem>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>
#include <ranges>

#ifdef IRIS_HAS_TENSORRT
#include <Eigen/Dense>
#include <cmath>
#endif

namespace iris {

class MultiviewPoseStage::Impl {
  public:
    explicit Impl(PoseConfig config, infrastructure::metrics::MetricRegistry* metrics)
        : config_(std::move(config)), metrics_enabled_(metrics != nullptr), process_ms_(metrics ? metrics->histogram("iris_pose_process_ms", {1, 2, 5, 10, 20, 50, 100, 250, 500, 1000}) : infrastructure::metrics::Histogram{}),
          capture_to_result_ms_(metrics ? metrics->histogram("iris_pose_capture_to_result_ms", {5, 10, 20, 50, 100, 250, 500, 1000, 2000, 5000}) : infrastructure::metrics::Histogram{}),
          last_capture_to_result_ms_(metrics ? metrics->gauge("iris_pose_last_capture_to_result_ms") : infrastructure::metrics::Gauge{}),
          last_process_ms_(metrics ? metrics->gauge("iris_pose_last_process_ms") : infrastructure::metrics::Gauge{}) {}
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
        const auto started = std::chrono::steady_clock::now();
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
        // RTMO candidates are local to each camera.  Select the strongest valid
        // person independently per view; candidate index must not be associated
        // across views before triangulation.
        std::array<int, 3> selected{};
        selected.fill(-1);
        for (std::size_t view = 0; view < 3; ++view)
            for (std::size_t candidate = 0; candidate < 10; ++candidate) {
                const auto index = view * 10 + candidate;
                if (result.candidate_valid[index] && std::isfinite(result.instance_scores[index]) &&
                    (selected[view] < 0 || result.instance_scores[index] > result.instance_scores[view * 10 + selected[view]]))
                    selected[view] = static_cast<int>(candidate);
            }

        std::array<MultiviewPose, 10> poses{};
        auto& pose = poses[0];
        for (std::size_t view = 0; view < pose.view_camera_ids.size(); ++view)
            pose.view_camera_ids[view] = config_.multiview_calibration[view].camera_id;
        for (std::size_t joint = 0; joint < 17; ++joint) {
            Eigen::Matrix<float, 6, 4> equations;
            int rows = 0;
            for (std::size_t view = 0; view < 3; ++view) {
                if (selected[view] < 0) continue;
                const auto candidate = static_cast<std::size_t>(selected[view]);
                const auto score_index = (view * 10 + candidate) * 17 + joint;
                const float score = result.keypoint_scores[score_index];
                pose.joint_scores[view][joint] = score;
                if (!std::isfinite(score) || score <= 0.0F) continue;

                const auto point_index = score_index * 2;
                const float x = result.keypoints[point_index];
                const float y = result.keypoints[point_index + 1];
                if (!std::isfinite(x) || !std::isfinite(y)) continue;
                const auto& source = *std::ranges::find(packet.frames, config_.multiview_calibration[view].camera_id, &Frame::camera);
                const float scale = std::min(640.0F / static_cast<float>(source.extent.width), 640.0F / static_cast<float>(source.extent.height));
                const float resized_width = static_cast<float>(std::max(1, static_cast<int>(source.extent.width * scale + 0.5F)));
                const float resized_height = static_cast<float>(std::max(1, static_cast<int>(source.extent.height * scale + 0.5F)));
                const float model_x = (x - (640.0F - resized_width) * 0.5F) / scale;
                const float model_y = (y - (640.0F - resized_height) * 0.5F) / scale;
                const auto& calibration = config_.multiview_calibration[view];
                const float fx = calibration.intrinsics[0], fy = calibration.intrinsics[4];
                const float cx = calibration.intrinsics[2], cy = calibration.intrinsics[5];
                const float xu = (model_x - cx) / fx, yu = (model_y - cy) / fy;
                const float r2 = xu * xu + yu * yu;
                const float radial = 1.0F + calibration.distortion[0] * r2 + calibration.distortion[1] * r2 * r2 + calibration.distortion[4] * r2 * r2 * r2;
                const float xd = xu * radial + 2.0F * calibration.distortion[2] * xu * yu + calibration.distortion[3] * (r2 + 2.0F * xu * xu);
                const float yd = yu * radial + calibration.distortion[2] * (r2 + 2.0F * yu * yu) + 2.0F * calibration.distortion[3] * xu * yu;
                pose.points_2d_px[view][joint] = {fx * xd + cx, fy * yd + cy};
                pose.point_valid[view][joint] = true;
                Eigen::Matrix<float, 3, 4> extrinsic;
                const auto& camera = config_.multiview_calibration[view];
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) extrinsic(r, c) = camera.R_w2c[r * 3 + c];
                    extrinsic(r, 3) = camera.t_w2c[r];
                }
                Eigen::Matrix3f intrinsic;
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c) intrinsic(r, c) = result.letterbox_intrinsics[view * 9 + r * 3 + c];
                const Eigen::Matrix<float, 3, 4> projection = intrinsic * extrinsic;
                equations.row(rows++) = score * (x * projection.row(2) - projection.row(0));
                equations.row(rows++) = score * (y * projection.row(2) - projection.row(1));
            }
            if (rows < 4) continue;
            const auto decomposition = equations.topRows(rows).jacobiSvd(Eigen::ComputeFullV);
            const Eigen::Vector4f homogeneous = decomposition.matrixV().col(3);
            if (!std::isfinite(homogeneous[3]) || std::abs(homogeneous[3]) < 1e-6F) continue;
            pose.joints_3d[joint] = {homogeneous[0] / homogeneous[3], homogeneous[1] / homogeneous[3], homogeneous[2] / homogeneous[3]};
            pose.joint_valid[joint] = true;
        }
        pose.active = std::ranges::count(pose.joint_valid, true) >= 5;
        packet.multiview_poses = std::move(poses);
        const auto finished = std::chrono::steady_clock::now();
        if (metrics_enabled_) {
            const auto process = std::chrono::duration<double, std::milli>(finished - started).count();
            process_ms_.observe(process);
            last_process_ms_.set(process);
        }
        if (metrics_enabled_ && !packet.frames.empty() && packet.frames.front().timing.estimated_capture_time.time_since_epoch().count() != 0) {
            const auto latency = std::chrono::duration<double, std::milli>(finished - packet.frames.front().timing.estimated_capture_time).count();
            capture_to_result_ms_.observe(latency);
            last_capture_to_result_ms_.set(latency);
        }
#endif
    }
  private:
    void refresh_calibration() {
        if (!config_.calibration_store) return;
        const auto rig=config_.calibration_store->snapshot();
        if (!rig || rig->revision==calibration_revision_) return;
        if (rig->cameras.size()!=3) throw std::runtime_error("multiview pose requires a three-camera runtime calibration");
        for (std::size_t i = 0; i < 3; ++i) {
            const auto& source = rig->cameras[i];
            auto previous = std::ranges::find(
                config_.multiview_calibration, source.camera_id,
                &PoseConfig::CameraCalibration::camera_id);
            const auto source_extent = previous == config_.multiview_calibration.end()
                ? Extent2D{} : previous->source_extent;
            const auto image_rotation_degrees = previous == config_.multiview_calibration.end()
                ? 0 : previous->image_rotation_degrees;
            auto& target = config_.multiview_calibration[i];
            target.camera_id = source.camera_id;
            target.intrinsics = source.intrinsics;
            target.distortion = source.distortion;
            target.R_w2c = source.R_w2c;
            target.t_w2c = source.t_w2c;
            target.source_extent = source_extent;
            target.image_rotation_degrees = image_rotation_degrees;
            target.calibrated = true;
            apply_capture_rotation(target);
        }
        calibration_revision_=rig->revision;
#ifdef IRIS_HAS_TENSORRT
        if(started_) engine_=std::make_unique<TensorRtMultiviewEngine>(config_);
#endif
    }
    PoseConfig config_;
    std::unique_ptr<TensorRtMultiviewEngine> engine_;
    bool started_{};
    std::uint64_t calibration_revision_{};
    bool metrics_enabled_{};
    infrastructure::metrics::Histogram process_ms_, capture_to_result_ms_;
    infrastructure::metrics::Gauge last_capture_to_result_ms_;
    infrastructure::metrics::Gauge last_process_ms_;
};

MultiviewPoseStage::MultiviewPoseStage(Channel<Packet>& input, Channel<Packet>* output, PoseConfig config, infrastructure::metrics::MetricRegistry* metrics)
    : Stage(input, output), impl_(std::make_unique<Impl>(std::move(config), metrics)) {}
MultiviewPoseStage::~MultiviewPoseStage() { stop(); }
void MultiviewPoseStage::start() { impl_->start(); Stage::start(); }
void MultiviewPoseStage::stop() { Stage::stop(); impl_->stop(); }
void MultiviewPoseStage::process(Packet& packet) { impl_->process(packet); }

} // namespace iris
