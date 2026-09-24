#include "iris/runtime/Pipeline.hpp"

#include "iris/pipeline/Channel.hpp"
#include "iris/stages/PoseStage.hpp"
#include "iris/stages/MultiviewPoseStage.hpp"
#include "iris/stages/capture/CaptureStage.hpp"
#include "iris/stages/capture/FrameSynchronizerStage.hpp"
#include "iris/stages/capture/SynchronizedVideoStage.hpp"
#include "iris/stages/FrameTapStage.hpp"
#include "iris/tools/RigCalibrationTool.hpp"
#include "iris/calibration/RigCalibration.hpp"

#include <atomic>
#include <chrono>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <ranges>
#include <unordered_set>
#include <utility>
#include <vector>

namespace iris {
namespace {
MultiCameraCaptureConfig single_camera_config(CaptureConfig config) {
    MultiCameraCaptureConfig result;
    result.cameras.push_back({0, std::move(config)});
    return result;
}
OutputConfig output_config(std::size_t camera_count, std::filesystem::path pose_output_path = {}) {
    OutputConfig result;
    result.camera_count = camera_count;
    result.pose_output_path = std::move(pose_output_path);
    return result;
}
} // namespace

class Pipeline::Impl {
  public:
    Impl(MultiCameraCaptureConfig config, infrastructure::metrics::MetricRegistry& metrics,
         PoseConfig pose_config, std::shared_ptr<RigCalibrationTool> rig_tool,
         std::shared_ptr<SnapshotBatchSource> snapshots)
        : Impl(std::move(config), std::nullopt, metrics, std::move(pose_config),
               std::move(rig_tool), std::move(snapshots)) {}

    Impl(SynchronizedVideoConfig config, infrastructure::metrics::MetricRegistry& metrics,
         PoseConfig pose_config, std::shared_ptr<RigCalibrationTool> rig_tool,
         std::shared_ptr<SnapshotBatchSource> snapshots)
        : Impl(MultiCameraCaptureConfig{}, std::move(config), metrics,
               std::move(pose_config), std::move(rig_tool), std::move(snapshots)) {}

  private:
    Impl(MultiCameraCaptureConfig config, std::optional<SynchronizedVideoConfig> video_config,
         infrastructure::metrics::MetricRegistry& metrics, PoseConfig pose_config,
         std::shared_ptr<RigCalibrationTool> rig_tool,
         std::shared_ptr<SnapshotBatchSource> snapshots)
        : capture_to_pose_(2, video_config ? OverflowPolicy::Block : OverflowPolicy::DropOldest,
                           infrastructure::metrics::register_channel_metrics(
                               metrics, "iris_channel_capture_to_pose")),
          pose_to_output_(2, video_config ? OverflowPolicy::Block : OverflowPolicy::DropOldest,
                           infrastructure::metrics::register_channel_metrics(
                               metrics, "iris_channel_pose_to_output")),
          tap_to_pose_(2, video_config ? OverflowPolicy::Block : OverflowPolicy::DropOldest),
          tap_(capture_to_pose_, &tap_to_pose_, [rig_tool](const Packet& packet){ if(rig_tool) rig_tool->observe(packet); }),
          output_(pose_to_output_, metrics,
                  output_config(video_config ? video_config->cameras.size() : config.cameras.size(),
                                video_config ? video_config->pose_output_path : std::filesystem::path{}),
                  std::move(snapshots)) {
        auto source_cameras = config.cameras;
        if (video_config) {
            video_stage_ = std::make_unique<SynchronizedVideoStage>(
                *video_config, capture_to_pose_, metrics);
            const auto extents = video_stage_->camera_extents();
            source_cameras.reserve(video_config->cameras.size());
            for (std::size_t index = 0; index < video_config->cameras.size(); ++index) {
                CameraCaptureConfig camera;
                camera.camera_id = video_config->cameras[index].camera_id;
                camera.capture.cuda_device = video_config->cuda_device;
                camera.capture.extent = extents[index];
                camera.capture.rotation = video_config->cameras[index].rotation;
                source_cameras.push_back(std::move(camera));
            }
        }
        if (source_cameras.empty()) {
            throw std::invalid_argument("multi-camera pipeline requires at least one camera");
        }
        if (!pose_config.model_path.empty() && !pose_config.multiview_engine_path.empty())
            throw std::invalid_argument("configure either monocular model_path or multiview_engine_path, not both");
        if (!pose_config.multiview_engine_path.empty()) {
            const auto view_count = source_cameras.size();
            if ((pose_config.two_d_only && view_count != 1) ||
                (!pose_config.two_d_only && (view_count < 2 || view_count > 10 ||
                    config.incomplete_batch_policy != IncompleteBatchPolicy::DropBatch)))
                throw std::invalid_argument(pose_config.two_d_only
                    ? "2-D pose requires exactly one camera"
                    : "multiview pose requires 2..10 cameras and drop-partial synchronization");
            if (pose_config.two_d_only) {
                pose_config.multiview_calibration.resize(1);
                auto& calibration = pose_config.multiview_calibration[0];
                calibration.camera_id = source_cameras.front().camera_id;
                calibration.intrinsics = {1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F};
                calibration.distortion = {};
                calibration.source_extent = source_cameras.front().capture.extent;
                calibration.calibrated = true;
            } else if (pose_config.multiview_calibration.empty() && pose_config.calibration_store) {
                pose_config.multiview_calibration.resize(view_count);
                for (std::size_t index = 0; index < view_count; ++index)
                    pose_config.multiview_calibration[index].camera_id = source_cameras[index].camera_id;
            }
            if (!pose_config.two_d_only && !pose_config.multiview_calibration.empty() &&
                pose_config.multiview_calibration.size() != view_count)
                throw std::invalid_argument("multiview calibration count must match configured camera count");
            if (!pose_config.two_d_only && pose_config.calibration_store) {
                const auto rig = pose_config.calibration_store->snapshot();
                if (rig && rig->cameras.size() != view_count)
                    throw std::invalid_argument("runtime calibration camera count must match configured camera count");
                if (rig) for (const auto& calibration : rig->cameras)
                    if (std::ranges::find(source_cameras, calibration.camera_id, &CameraCaptureConfig::camera_id) == source_cameras.end()) throw std::invalid_argument("runtime calibration camera ID is not configured for capture");
            } else if (!pose_config.two_d_only) {
                for (const auto& calibration : pose_config.multiview_calibration)
                    if (calibration.calibrated && std::ranges::find(source_cameras, calibration.camera_id, &CameraCaptureConfig::camera_id) == source_cameras.end()) throw std::invalid_argument("multiview calibration camera ID is not configured for capture");
            }
            const int cuda_device = source_cameras.front().capture.cuda_device;
            for (std::size_t index = 0; index < source_cameras.size(); ++index) {
                const auto& camera = source_cameras[index];
                if (camera.capture.cuda_device != cuda_device)
                    throw std::invalid_argument("multiview pose requires all frames on the same CUDA device");
                if (pose_config.two_d_only) continue;
                auto calibration = std::ranges::find(
                    pose_config.multiview_calibration, camera.camera_id,
                    &PoseConfig::CameraCalibration::camera_id);
                // When using the live rig store, local entries may be
                // placeholders until MultiviewPoseStage refreshes them.
                if (calibration == pose_config.multiview_calibration.end() &&
                    index < pose_config.multiview_calibration.size() &&
                    !pose_config.multiview_calibration[index].calibrated) {
                    calibration = pose_config.multiview_calibration.begin() + index;
                    calibration->camera_id = camera.camera_id;
                }
                if (calibration == pose_config.multiview_calibration.end())
                    throw std::invalid_argument("multiview calibration camera ID is not configured for capture");
                calibration->source_extent = camera.capture.extent;
                calibration->image_rotation_degrees = rotation_degrees(camera.capture.rotation);
                if (calibration->calibrated) apply_capture_rotation(*calibration);
            }
        }
        if (!pose_config.multiview_engine_path.empty())
            pose_ = std::make_unique<MultiviewPoseStage>(tap_to_pose_, &pose_to_output_, std::move(pose_config), &metrics);
        else
            pose_ = std::make_unique<PoseStage>(tap_to_pose_, &pose_to_output_, std::move(pose_config), &metrics);
        if (config.sync_queue_capacity == 0 || config.sync_tolerance.count() < 0) {
            throw std::invalid_argument("invalid multi-camera synchronizer configuration");
        }
        std::unordered_set<CameraId> ids;
        for (const auto& camera : source_cameras) {
            if (!ids.insert(camera.camera_id).second) {
                throw std::invalid_argument("camera IDs must be unique");
            }
        }
        if (video_stage_) {
            return;
        }
        if (source_cameras.size() == 1) {
            const auto& camera = source_cameras.front();
            captures_.push_back(std::make_unique<CaptureStage>(
                camera.camera_id, camera.capture, "iris_capture", "iris_channel_capture_samples",
                capture_to_pose_, metrics));
            return;
        }
        std::vector<Channel<Packet>*> inputs;
        for (const auto& camera : source_cameras) {
            const auto id = std::to_string(camera.camera_id);
            auto channel = std::make_unique<Channel<Packet>>(
                config.sync_queue_capacity, OverflowPolicy::DropOldest,
                infrastructure::metrics::register_channel_metrics(
                    metrics, "iris_channel_capture_camera_" + id + "_to_sync"));
            inputs.push_back(channel.get());
            capture_channels_.push_back(std::move(channel));
        }
        for (std::size_t index = 0; index < source_cameras.size(); ++index) {
            const auto& camera = source_cameras[index];
            const auto id = std::to_string(camera.camera_id);
            captures_.push_back(std::make_unique<CaptureStage>(
                camera.camera_id, camera.capture, "iris_capture_camera_" + id,
                "iris_channel_capture_camera_" + id + "_samples", *capture_channels_[index],
                metrics));
        }
        synchronizer_ = std::make_unique<FrameSynchronizerStage>(std::move(inputs),
                                                                 capture_to_pose_, config, metrics);
    }

  public:
    void start() {
        production_stop_requested_.store(false);
        output_.start();
        pose_->start();
        tap_.start();
        if (synchronizer_) {
            synchronizer_->start();
        }
        if (video_stage_) {
            video_stage_->start();
        }
        for (auto& capture : captures_) {
            capture->start();
        }
    }
    void stop_producing() {
        production_stop_requested_.store(true);
        for (auto& capture : captures_) {
            capture->stop_producing();
        }
        if (video_stage_) {
            video_stage_->stop_producing();
        }
    }
    void wait() {
        const bool natural_video_eof = video_stage_ && video_stage_->finished() &&
                                       video_stage_->healthy() &&
                                       !production_stop_requested_.load();
        while (!production_stop_requested_.load() && healthy() &&
               !(video_stage_ && video_stage_->finished())) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        stop_producing();
        std::exception_ptr failure;
        for (auto& capture : captures_) {
            try {
                capture->wait();
            } catch (...) {
                if (!failure) {
                    failure = std::current_exception();
                }
            }
        }
        if (video_stage_) {
            try {
                video_stage_->wait();
            } catch (...) {
                if (!failure) failure = std::current_exception();
            }
        }
        if (synchronizer_) {
            synchronizer_->wait();
        }
        tap_.stop(); pose_->stop();
        if (natural_video_eof) {
            const auto finalized = output_.stop_recording();
            if (finalized.status == OutputCommandStatus::Failed) {
                if (!failure) failure = std::make_exception_ptr(std::runtime_error(finalized.message));
            }
        }
        output_.stop();
        if (!failure && pose_->failure()) failure = pose_->failure();
        if (!failure) {
            const auto export_failure = output_.failure();
            if (!export_failure.empty())
                failure = std::make_exception_ptr(std::runtime_error(export_failure));
        }
        if (failure) {
            std::rethrow_exception(failure);
        }
    }
    void stop() {
        stop_producing();
        // Video output is lossless and may be blocked on a full channel. Closing
        // downstream channels first releases producers if a stage has failed.
        capture_to_pose_.close();
        tap_to_pose_.close();
        pose_to_output_.close();
        for (auto& capture : captures_) {
            capture->stop();
        }
        if (video_stage_) {
            video_stage_->stop();
        }
        for (auto& channel : capture_channels_) {
            channel->close();
        }
        if (synchronizer_) {
            synchronizer_->stop();
        }
        tap_.stop(); pose_->stop();
        output_.stop();
    }
    bool healthy() const noexcept {
        if (pose_ && !pose_->healthy()) return false;
        if (video_stage_ && !video_stage_->healthy()) return false;
        for (const auto& capture : captures_) {
            if (!capture->healthy()) {
                return false;
            }
        }
        return true;
    }
    std::string output_failure() const { return output_.failure(); }
    std::vector<VideoDecodeStatus> video_decode_status() const {
        return video_stage_ ? video_stage_->camera_decode_status()
                            : std::vector<VideoDecodeStatus>{};
    }

    Channel<Packet> capture_to_pose_;
    Channel<Packet> pose_to_output_;
    Channel<Packet> tap_to_pose_;
    FrameTapStage tap_;
    std::vector<std::unique_ptr<Channel<Packet>>> capture_channels_;
    std::vector<std::unique_ptr<CaptureStage>> captures_;
    std::unique_ptr<SynchronizedVideoStage> video_stage_;
    std::unique_ptr<FrameSynchronizerStage> synchronizer_;
    std::unique_ptr<Stage> pose_;
    OutputStage output_;
    std::atomic_bool production_stop_requested_{false};
};

Pipeline::Pipeline(CaptureConfig config, infrastructure::metrics::MetricRegistry& metrics,
                   PoseConfig pose_config, std::shared_ptr<RigCalibrationTool> rig_tool, std::shared_ptr<SnapshotBatchSource> snapshots)
    : Pipeline(single_camera_config(std::move(config)), metrics, std::move(pose_config), std::move(rig_tool), std::move(snapshots)) {}
Pipeline::Pipeline(MultiCameraCaptureConfig config,
                   infrastructure::metrics::MetricRegistry& metrics, PoseConfig pose_config, std::shared_ptr<RigCalibrationTool> rig_tool, std::shared_ptr<SnapshotBatchSource> snapshots)
    : impl_(std::make_unique<Impl>(std::move(config), metrics, std::move(pose_config), std::move(rig_tool), std::move(snapshots))) {}
Pipeline::Pipeline(SynchronizedVideoConfig config,
                   infrastructure::metrics::MetricRegistry& metrics, PoseConfig pose_config,
                   std::shared_ptr<RigCalibrationTool> rig_tool, std::shared_ptr<SnapshotBatchSource> snapshots)
    : impl_(std::make_unique<Impl>(std::move(config), metrics, std::move(pose_config),
                                  std::move(rig_tool), std::move(snapshots))) {}
Pipeline::~Pipeline() = default;
void Pipeline::start() { impl_->start(); }
void Pipeline::stop_producing() { impl_->stop_producing(); }
void Pipeline::wait() { impl_->wait(); }
void Pipeline::stop() { impl_->stop(); }
OutputCommandResult Pipeline::configure_shared_memory(SharedMemoryOutputConfig config) {
    return impl_->output_.configure_shared_memory(std::move(config));
}
OutputCommandResult Pipeline::configure_preview(PreviewConfig config) { return impl_->output_.configure_preview(std::move(config)); }
OutputCommandResult Pipeline::configure_disk(DiskOutputConfig config) {
    return impl_->output_.configure_disk(std::move(config));
}
OutputCommandResult Pipeline::start_recording() { return impl_->output_.start_recording(); }
OutputCommandResult Pipeline::stop_recording() { return impl_->output_.stop_recording(); }
std::size_t Pipeline::processed_count() const noexcept { return impl_->output_.processed_count(); }
std::vector<VideoDecodeStatus> Pipeline::video_decode_status() const {
    return impl_->video_decode_status();
}
PreviewTransportHealth Pipeline::preview_health() const { return impl_->output_.preview_health(); }
bool Pipeline::healthy() const noexcept { return impl_->healthy(); }
std::string Pipeline::output_failure() const { return impl_->output_failure(); }
} // namespace iris
