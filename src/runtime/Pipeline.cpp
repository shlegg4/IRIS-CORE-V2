#include "iris/runtime/Pipeline.hpp"

#include "iris/pipeline/Channel.hpp"
#include "iris/stages/PoseStage.hpp"
#include "iris/stages/capture/CaptureStage.hpp"
#include "iris/stages/capture/FrameSynchronizerStage.hpp"

#include <atomic>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
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
OutputConfig output_config(std::size_t camera_count) {
    OutputConfig result;
    result.camera_count = camera_count;
    return result;
}
} // namespace

class Pipeline::Impl {
  public:
    Impl(MultiCameraCaptureConfig config, infrastructure::metrics::MetricRegistry& metrics,
         PoseConfig pose_config)
        : capture_to_pose_(2, OverflowPolicy::DropOldest,
                           infrastructure::metrics::register_channel_metrics(
                               metrics, "iris_channel_capture_to_pose")),
          pose_to_output_(2, OverflowPolicy::DropOldest,
                          infrastructure::metrics::register_channel_metrics(
                              metrics, "iris_channel_pose_to_output")),
          pose_(capture_to_pose_, &pose_to_output_, std::move(pose_config)),
          output_(pose_to_output_, metrics, output_config(config.cameras.size())) {
        if (config.cameras.empty()) {
            throw std::invalid_argument("multi-camera pipeline requires at least one camera");
        }
        if (config.sync_queue_capacity == 0 || config.sync_tolerance.count() < 0) {
            throw std::invalid_argument("invalid multi-camera synchronizer configuration");
        }
        std::unordered_set<CameraId> ids;
        for (const auto& camera : config.cameras) {
            if (!ids.insert(camera.camera_id).second) {
                throw std::invalid_argument("camera IDs must be unique");
            }
        }
        if (config.cameras.size() == 1) {
            const auto& camera = config.cameras.front();
            captures_.push_back(std::make_unique<CaptureStage>(
                camera.camera_id, camera.capture, "iris_capture", "iris_channel_capture_samples",
                capture_to_pose_, metrics));
            return;
        }
        std::vector<Channel<Packet>*> inputs;
        for (const auto& camera : config.cameras) {
            const auto id = std::to_string(camera.camera_id);
            auto channel = std::make_unique<Channel<Packet>>(
                config.sync_queue_capacity, OverflowPolicy::DropOldest,
                infrastructure::metrics::register_channel_metrics(
                    metrics, "iris_channel_capture_camera_" + id + "_to_sync"));
            inputs.push_back(channel.get());
            capture_channels_.push_back(std::move(channel));
        }
        for (std::size_t index = 0; index < config.cameras.size(); ++index) {
            const auto& camera = config.cameras[index];
            const auto id = std::to_string(camera.camera_id);
            captures_.push_back(std::make_unique<CaptureStage>(
                camera.camera_id, camera.capture, "iris_capture_camera_" + id,
                "iris_channel_capture_camera_" + id + "_samples", *capture_channels_[index],
                metrics));
        }
        synchronizer_ = std::make_unique<FrameSynchronizerStage>(std::move(inputs),
                                                                 capture_to_pose_, config, metrics);
    }

    void start() {
        production_stop_requested_.store(false);
        output_.start();
        pose_.start();
        if (synchronizer_) {
            synchronizer_->start();
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
    }
    void wait() {
        while (!production_stop_requested_.load() && healthy()) {
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
        if (synchronizer_) {
            synchronizer_->wait();
        }
        pose_.stop();
        output_.stop();
        if (failure) {
            std::rethrow_exception(failure);
        }
    }
    void stop() {
        stop_producing();
        for (auto& capture : captures_) {
            capture->stop();
        }
        for (auto& channel : capture_channels_) {
            channel->close();
        }
        if (synchronizer_) {
            synchronizer_->stop();
        }
        capture_to_pose_.close();
        pose_to_output_.close();
        pose_.stop();
        output_.stop();
    }
    bool healthy() const noexcept {
        for (const auto& capture : captures_) {
            if (!capture->healthy()) {
                return false;
            }
        }
        return true;
    }

    Channel<Packet> capture_to_pose_;
    Channel<Packet> pose_to_output_;
    std::vector<std::unique_ptr<Channel<Packet>>> capture_channels_;
    std::vector<std::unique_ptr<CaptureStage>> captures_;
    std::unique_ptr<FrameSynchronizerStage> synchronizer_;
    PoseStage pose_;
    OutputStage output_;
    std::atomic_bool production_stop_requested_{false};
};

Pipeline::Pipeline(CaptureConfig config, infrastructure::metrics::MetricRegistry& metrics,
                   PoseConfig pose_config)
    : Pipeline(single_camera_config(std::move(config)), metrics, std::move(pose_config)) {}
Pipeline::Pipeline(MultiCameraCaptureConfig config,
                   infrastructure::metrics::MetricRegistry& metrics, PoseConfig pose_config)
    : impl_(std::make_unique<Impl>(std::move(config), metrics, std::move(pose_config))) {}
Pipeline::~Pipeline() = default;
void Pipeline::start() { impl_->start(); }
void Pipeline::stop_producing() { impl_->stop_producing(); }
void Pipeline::wait() { impl_->wait(); }
void Pipeline::stop() { impl_->stop(); }
OutputCommandResult Pipeline::configure_shared_memory(SharedMemoryOutputConfig config) {
    return impl_->output_.configure_shared_memory(std::move(config));
}
OutputCommandResult Pipeline::configure_preview(PreviewConfig config) { return impl_->output_.configure_preview(std::move(config)); }
void Pipeline::set_preview_status_provider(std::function<std::string()> provider) { impl_->output_.set_preview_status_provider(std::move(provider)); }
OutputCommandResult Pipeline::configure_disk(DiskOutputConfig config) {
    return impl_->output_.configure_disk(std::move(config));
}
OutputCommandResult Pipeline::start_recording() { return impl_->output_.start_recording(); }
OutputCommandResult Pipeline::stop_recording() { return impl_->output_.stop_recording(); }
std::size_t Pipeline::processed_count() const noexcept { return impl_->output_.processed_count(); }
PreviewTransportHealth Pipeline::preview_health() const { return impl_->output_.preview_health(); }
bool Pipeline::healthy() const noexcept { return impl_->healthy(); }
} // namespace iris
