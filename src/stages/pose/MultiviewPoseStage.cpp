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
#include <cmath>
#endif

namespace iris {
namespace {
}

class MultiviewPoseStage::Impl {
  public:
    explicit Impl(PoseConfig config, infrastructure::metrics::MetricRegistry* metrics)
        : config_(std::move(config)), metrics_enabled_(metrics != nullptr), process_ms_(metrics ? metrics->histogram("iris_pose_process_ms", {1, 2, 5, 10, 20, 50, 100, 250, 500, 1000}) : infrastructure::metrics::Histogram{}),
          capture_to_result_ms_(metrics ? metrics->histogram("iris_pose_capture_to_result_ms", {5, 10, 20, 50, 100, 250, 500, 1000, 2000, 5000}) : infrastructure::metrics::Histogram{}),
          last_capture_to_result_ms_(metrics ? metrics->gauge("iris_pose_last_capture_to_result_ms") : infrastructure::metrics::Gauge{}),
          last_process_ms_(metrics ? metrics->gauge("iris_pose_last_process_ms") : infrastructure::metrics::Gauge{}) {
        if (metrics) {
            const std::array<const char*, 11> names{
                "frame_ready_wait", "preprocess_host", "trt_setup_host", "trt_enqueue_host",
                "download_host", "result_wait_host", "preprocess_stream", "engine_stream",
                "download_stream", "postprocess_cpu", "engine_call_host"};
            for (std::size_t i=0; i<names.size(); ++i) {
                breakdown_[i] = metrics->histogram(std::string("iris_pose_")+names[i]+"_ms", {0.01,0.05,0.1,0.25,0.5,1,2,3,5,10,20});
                latest_[i] = metrics->gauge(std::string("iris_pose_last_")+names[i]+"_ms");
            }
        }
    }
    void start() {
        if (config_.multiview_engine_path.empty()) return;
        if (config_.max_persons == 0 || config_.max_persons > 10)
            throw std::invalid_argument("multiview max_persons must be between 1 and the engine candidate capacity (10)");
        if (!(config_.epipolar_gate_px > 0.0F) || !(config_.minimum_joint_confidence >= 0.0F) ||
            !(config_.maximum_reprojection_error_px > 0.0F))
            throw std::invalid_argument("multiview geometry thresholds must be finite and positive");
        if (!std::filesystem::is_regular_file(config_.multiview_engine_path))
            throw std::runtime_error("multiview TensorRT engine does not exist: " + config_.multiview_engine_path.string());
        refresh_calibration();
        if (config_.two_d_only)
            for (std::size_t i = 1; i < config_.multiview_calibration.size(); ++i)
                config_.multiview_calibration[i] = config_.multiview_calibration[0];
        for (std::size_t i = 0; i < (config_.two_d_only ? 1U : 3U); ++i)
            if (const auto& calibration = config_.multiview_calibration[i]; !calibration.calibrated)
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
        const auto view_count = config_.two_d_only ? 1U : 3U;
        if (packet.frames.size() != view_count)
            throw std::runtime_error(config_.two_d_only
                ? "2-D TensorRT pose requires exactly one frame"
                : "multiview TensorRT engine requires exactly three synchronized frames");
#ifndef IRIS_HAS_TENSORRT
        (void)packet;
        throw std::runtime_error("multiview TensorRT inference is unavailable in this build");
#else
        std::array<const void*, 3> buffers{};
        std::array<std::size_t, 3> strides{};
        std::array<std::uint32_t, 3> widths{};
        std::array<std::uint32_t, 3> heights{};
        double ready_wait_ms = 0;
        for (std::size_t i = 0; i < view_count; ++i) {
            const auto id = config_.multiview_calibration[i].camera_id;
            const auto frame = std::ranges::find(packet.frames, id, &Frame::camera);
            if (frame == packet.frames.end() || frame->format != PixelFormat::Bgr8 || !frame->buffer.data)
                throw std::runtime_error("multiview packet is missing a valid calibrated BGR camera frame");
            if (!frame->extent.width || !frame->extent.height || frame->buffer.stride_bytes < static_cast<std::size_t>(frame->extent.width) * 3)
                throw std::runtime_error("multiview frame has invalid dimensions or stride");
            if (frame->ready) {
                const auto begin = std::chrono::steady_clock::now();
                frame->ready->synchronize();
                ready_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-begin).count();
            }
            buffers[i] = frame->buffer.data;
            strides[i] = frame->buffer.stride_bytes;
            widths[i] = frame->extent.width;
            heights[i] = frame->extent.height;
        }
        // The shipped RTMO engine has a fixed batch of three. In 2-D mode the
        // extra batch entries are harmless copies and their outputs are ignored.
        for (std::size_t i = view_count; i < 3; ++i) {
            buffers[i] = buffers[0];
            strides[i] = strides[0];
            widths[i] = widths[0];
            heights[i] = heights[0];
        }
        TensorRtMultiviewResult result;
        const auto engine_start = std::chrono::steady_clock::now();
        engine_->infer(buffers, strides, widths, heights, result);
        const auto postprocess_start = std::chrono::steady_clock::now();
        std::vector<MultiviewPose> poses;
        poses.reserve(std::min<std::size_t>(config_.max_persons, 10));
        for (std::size_t person_index = 0; person_index < std::min<std::size_t>(config_.max_persons, 10); ++person_index) {
        std::array<int, 3> selected{};
        selected.fill(-1);
        for (std::size_t view = 0; view < view_count; ++view) {
            const auto assignment = result.assignments[person_index * 3 + view];
            selected[view] = assignment == 255 ? -1 : static_cast<int>(assignment);
        }
        auto& pose = poses.emplace_back();
        for (std::size_t view = 0; view < view_count; ++view)
            pose.view_camera_ids[view] = config_.multiview_calibration[view].camera_id;
        for (std::size_t joint = 0; joint < 17; ++joint) {
            for (std::size_t view = 0; view < view_count; ++view) {
                if (selected[view] < 0) continue;
                const auto score_index = (person_index * 3 + view) * 17 + joint;
                const float score = result.selected_scores[score_index];
                pose.joint_scores[view][joint] = score;
                // RTMO emits small positive scores for effectively absent joints.
                // Treating every positive value as an observation lets a single
                // reliable view triangulate against noise from another camera,
                // producing a coherent-looking skeleton in the wrong location.
                if (!std::isfinite(score) || score < config_.minimum_joint_confidence) continue;

                const auto point_index = score_index * 2;
                const float x = result.selected_keypoints[point_index];
                const float y = result.selected_keypoints[point_index + 1];
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
            }
            if (!config_.two_d_only) {
                const auto gpu_index = person_index * 17 + joint;
                if (result.triangulated_valid[gpu_index]) {
                    pose.joints_3d[joint] = {
                        result.triangulated_xyz[gpu_index * 3],
                        result.triangulated_xyz[gpu_index * 3 + 1],
                        result.triangulated_xyz[gpu_index * 3 + 2]};
                    pose.joint_valid[joint] = true;
                }
                continue;
            }
        }
        // Keep 2-D observations publishable even when multiview geometry
        // rejects the reconstruction.  Otherwise one bad epipolar match
        // makes the camera overlays disappear with the entire person.
        const auto valid_2d = std::ranges::count_if(pose.point_valid, [](const auto& view) {
            return std::ranges::count(view, true) >= 5;
        });
        pose.active = config_.two_d_only
            ? valid_2d >= 1
            : valid_2d >= 1;
        if (!pose.active) poses.pop_back();
        }
        packet.multiview_poses = std::move(poses);
        const auto finished = std::chrono::steady_clock::now();
        if (metrics_enabled_) {
            const auto& t=result.timings;
            const std::array<double,11> values{ready_wait_ms, t.preprocess_host_ms, t.setup_host_ms,
                t.enqueue_host_ms, t.download_host_ms, t.wait_host_ms, t.preprocess_stream_ms,
                t.engine_stream_ms, t.download_stream_ms,
                std::chrono::duration<double,std::milli>(finished-postprocess_start).count(),
                std::chrono::duration<double,std::milli>(postprocess_start-engine_start).count()};
            for(std::size_t i=0;i<values.size();++i) {
                breakdown_[i].observe(values[i]); latest_[i].set(values[i]);
            }
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
        if (config_.two_d_only) return;
        if (!config_.calibration_store) return;
        const auto rig=config_.calibration_store->snapshot();
        if (!rig || rig->revision==calibration_revision_) return;
        if (rig->cameras.size()!=3) throw std::runtime_error("multiview pose requires a three-camera runtime calibration");
        for (std::size_t i = 0; i < 3; ++i) {
            const auto& source = rig->cameras[i];
            auto& target = config_.multiview_calibration[i];
            target.camera_id = source.camera_id;
            target.intrinsics = source.intrinsics;
            target.distortion = source.distortion;
            target.R_w2c = source.R_w2c;
            target.t_w2c = source.t_w2c;
            // RigCalibrationTool observes frames after CaptureStage has applied
            // its configured rotation.  The stored calibration is therefore
            // already expressed in the inference image coordinate system.
            // Applying capture rotation here again would rotate K and [R|t]
            // twice and make triangulation disagree with the viewer snapshot.
            target.source_extent = {};
            target.image_rotation_degrees = 0;
            target.calibrated = true;
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
    std::array<infrastructure::metrics::Histogram,11> breakdown_;
    std::array<infrastructure::metrics::Gauge,11> latest_;
};

MultiviewPoseStage::MultiviewPoseStage(Channel<Packet>& input, Channel<Packet>* output, PoseConfig config, infrastructure::metrics::MetricRegistry* metrics)
    : Stage(input, output), impl_(std::make_unique<Impl>(std::move(config), metrics)) {}
MultiviewPoseStage::~MultiviewPoseStage() { stop(); }
void MultiviewPoseStage::start() { impl_->start(); Stage::start(); }
void MultiviewPoseStage::stop() { Stage::stop(); impl_->stop(); }
void MultiviewPoseStage::process(Packet& packet) { impl_->process(packet); }

} // namespace iris
