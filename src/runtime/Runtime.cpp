#include "iris/runtime/Runtime.hpp"

#include "iris/infrastructure/metrics/MetricsExporter.hpp"
#include "iris/infrastructure/metrics/PrometheusExporter.hpp"
#include "iris/pipeline/Channel.hpp"
#include "iris/runtime/Pipeline.hpp"
#include "iris/tools/RigCalibrationTool.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <fstream>
#include <iostream>
#include <mutex>
#include <ranges>
#include <thread>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

namespace iris {
namespace {
using json = nlohmann::json;

std::filesystem::path resolve_pose_asset(std::filesystem::path path) {
    constexpr std::string_view prefix{"@assets/"};
    const auto value = path.generic_string();
    if (!value.starts_with(prefix)) return path;
#ifdef IRIS_BUILD_ASSET_DIRECTORY
    return std::filesystem::path{IRIS_BUILD_ASSET_DIRECTORY} / value.substr(prefix.size());
#else
    return std::filesystem::current_path() / "assets" / value.substr(prefix.size());
#endif
}

void load_multiview_calibration(PoseConfig& config, const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("could not open multiview calibration file: " + path.string());
    const auto document = json::parse(input);
    if (!document.contains("cameras") || !document["cameras"].is_array() || document["cameras"].size() != 3)
        throw std::runtime_error("multiview calibration must contain exactly three cameras");
    for (std::size_t index = 0; index < 3; ++index) {
        const auto& camera = document["cameras"][index];
        if (!camera.contains("camera_id")) throw std::runtime_error("calibration camera is missing camera_id");
        auto read = [&camera](const char* name, std::size_t count) {
            if (!camera.contains(name) || !camera[name].is_array() || camera[name].size() != count)
                throw std::runtime_error(std::string("calibration field must contain ") + std::to_string(count) + " values: " + name);
            std::vector<float> values;
            for (const auto& value : camera[name]) values.push_back(value.get<float>());
            return values;
        };
        const auto rotation = read("R_w2c", 9);
        const auto translation = read("t_w2c", 3);
        const auto intrinsics = read("intrinsics", 9);
        const auto distortion = camera.contains("distortion") ? read("distortion", 5) : std::vector<float>(5, 0.0F);
        auto& target = config.multiview_calibration[index];
        target.camera_id = camera["camera_id"].get<CameraId>();
        std::copy(rotation.begin(), rotation.end(), target.R_w2c.begin());
        std::copy(translation.begin(), translation.end(), target.t_w2c.begin());
        std::copy(intrinsics.begin(), intrinsics.end(), target.intrinsics.begin());
        std::copy(distortion.begin(), distortion.end(), target.distortion.begin());
        target.calibrated = true;
    }
    std::array<CameraId, 3> ids{};
    for (std::size_t i = 0; i < 3; ++i) ids[i] = config.multiview_calibration[i].camera_id;
    std::sort(ids.begin(), ids.end());
    if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) throw std::runtime_error("calibration camera IDs must be unique");
}
std::string json_quote(const std::string& value) {
    std::string result{"\""};
    for (const char c : value) { if (c == '\\' || c == '\"') result += '\\'; if (c == '\n') result += "\\n"; else result += c; }
    return result + '"';
}

struct CommandRequest {
    RuntimeCommand command;
    std::promise<RuntimeCommandResponse> completion;
};

RuntimeCommandResponse from_output_result(OutputCommandResult result) {
    switch (result.status) {
    case OutputCommandStatus::Applied:
        return {RuntimeCommandStatus::Applied, std::move(result.message), std::nullopt};
    case OutputCommandStatus::Rejected:
        return {RuntimeCommandStatus::Rejected, std::move(result.message), std::nullopt};
    case OutputCommandStatus::Failed:
        return {RuntimeCommandStatus::Failed, std::move(result.message), std::nullopt};
    }
    return {RuntimeCommandStatus::Failed, "unknown output command result", std::nullopt};
}

CaptureConfig apply_capture_patch(CaptureConfig config, const CaptureConfigPatch& patch) {
    if (patch.device_symbolic_link) {
        config.device_symbolic_link = *patch.device_symbolic_link;
        config.device_index.reset();
    }
    if (patch.device_index) {
        config.device_index = *patch.device_index;
        config.device_symbolic_link.clear();
    }
    if (patch.width) {
        config.extent.width = *patch.width;
    }
    if (patch.height) {
        config.extent.height = *patch.height;
    }
    if (patch.frame_rate) {
        config.frame_rate = *patch.frame_rate;
    }
    if (patch.format) {
        config.format = *patch.format;
    }
    if (patch.cuda_device) {
        config.cuda_device = *patch.cuda_device;
    }
    if (patch.sample_queue_capacity) {
        config.sample_queue_capacity = *patch.sample_queue_capacity;
    }
    if (patch.frame_pool_capacity) {
        config.frame_pool_capacity = *patch.frame_pool_capacity;
    }
    if (patch.overflow) {
        config.overflow = *patch.overflow;
    }
    if (patch.rotation) {
        config.rotation = *patch.rotation;
    }
    if (patch.allow_format_fallback) {
        config.allow_format_fallback = *patch.allow_format_fallback;
    }
    if (patch.reconnect) {
        config.reconnect = *patch.reconnect;
    }
    return config;
}

std::optional<std::string> validate_capture_config(const CaptureConfig& config) {
    if (config.extent.width == 0 || config.extent.height == 0) {
        return "capture width and height must be greater than zero";
    }
    if (config.frame_rate.numerator == 0 || config.frame_rate.denominator == 0) {
        return "capture frame rate must be greater than zero";
    }
    if (config.format == PixelFormat::Unknown || config.format == PixelFormat::Bgr8) {
        return "capture format must be mjpeg, yuy2, or bgra8";
    }
    if (config.cuda_device < 0) {
        return "CUDA device index cannot be negative";
    }
    if (config.sample_queue_capacity == 0 || config.frame_pool_capacity == 0) {
        return "capture queue and frame-pool capacities must be greater than zero";
    }
    return std::nullopt;
}

MultiCameraCaptureConfig single_camera_config(CaptureConfig config) {
    MultiCameraCaptureConfig result;
    result.cameras.push_back({0, std::move(config)});
    return result;
}

std::optional<std::string> validate_camera_group(const MultiCameraCaptureConfig& config) {
    if (config.cameras.empty()) {
        return "at least one capture camera is required";
    }
    if (config.sync_queue_capacity == 0 || config.sync_tolerance.count() < 0) {
        return "invalid synchronizer configuration";
    }
    std::vector<CameraId> ids;
    for (const auto& camera : config.cameras) {
        if (std::ranges::find(ids, camera.camera_id) != ids.end()) {
            return "camera IDs must be unique";
        }
        ids.push_back(camera.camera_id);
        if (const auto error = validate_capture_config(camera.capture)) {
            return "camera " + std::to_string(camera.camera_id) + ": " + *error;
        }
    }
    return std::nullopt;
}

} // namespace

const char* to_string(RuntimeState state) noexcept {
    switch (state) {
    case RuntimeState::Stopped:
        return "stopped";
    case RuntimeState::Starting:
        return "starting";
    case RuntimeState::Running:
        return "running";
    case RuntimeState::Stopping:
        return "stopping";
    case RuntimeState::Failed:
        return "failed";
    case RuntimeState::Shutdown:
        return "shutdown";
    }
    return "unknown";
}

class Runtime::Impl {
  public:
    explicit Impl(MultiCameraCaptureConfig config, std::uint16_t metrics_port, PoseConfig pose_config)
        : capture_config_(std::move(config)), pose_config_(std::move(pose_config)), calibration_store_(std::make_shared<CalibrationStore>()), rig_tool_(std::make_shared<RigCalibrationTool>(calibration_store_)), exporter_(metrics_, "iris_metrics.json"),
          prometheus_(metrics_, metrics_port), commands_(32, OverflowPolicy::Block) {}

    ~Impl() { stop(); }

    void start() {
        bool expected = false;
        if (!control_started_.compare_exchange_strong(expected, true)) {
            return;
        }
        try {
            prometheus_.start();
            exporter_.start();
        } catch (...) {
            prometheus_.stop();
            exporter_.stop();
            control_started_.store(false);
            throw;
        }
        control_accepting_.store(true);
        control_thread_ = std::thread(&Impl::control_loop, this);
    }

    RuntimeCommandResponse execute(RuntimeCommand command) {
        if (!control_started_ || !control_accepting_) {
            return {RuntimeCommandStatus::Rejected, "runtime control plane is not running",
                    std::nullopt};
        }
        auto request = std::make_shared<CommandRequest>();
        request->command = std::move(command);
        auto result = request->completion.get_future();
        if (commands_.send(std::move(request)) == SendResult::Closed) {
            return {RuntimeCommandStatus::Rejected, "runtime control plane is closed",
                    std::nullopt};
        }
        return result.get();
    }

    RuntimeSnapshot snapshot() const {
        std::scoped_lock lock(state_mutex_);
        return snapshot_unlocked();
    }

    void stop() {
        if (!control_started_) {
            return;
        }
        if (control_accepting_) {
            execute(ShutdownCommand{});
        }
        control_started_.store(false);
        commands_.close();
        if (control_thread_.joinable()) {
            control_thread_.join();
        }
        stop_pipeline();
        exporter_.stop();
        prometheus_.stop();
    }

  private:
    void control_loop() {
        while (auto request = commands_.receive()) {
            const bool shutdown = std::holds_alternative<ShutdownCommand>((*request)->command);
            RuntimeCommandResponse response;
            try {
                response = std::visit([this](const auto& value) { return handle(value); },
                                      (*request)->command);
            } catch (const std::exception& error) {
                response = {RuntimeCommandStatus::Failed, error.what(), std::nullopt};
            }
            (*request)->completion.set_value(std::move(response));
            if (shutdown) {
                break;
            }
        }
        control_accepting_.store(false);
    }

    RuntimeCommandResponse handle(const StartPipelineCommand&) {
        {
            std::scoped_lock lock(state_mutex_);
            if (state_ == RuntimeState::Running || state_ == RuntimeState::Starting) {
                return {RuntimeCommandStatus::Rejected, "pipeline is already running",
                        std::nullopt};
            }
            if (state_ == RuntimeState::Shutdown) {
                return {RuntimeCommandStatus::Rejected, "runtime is shut down", std::nullopt};
            }
        }
        if (pipeline_thread_.joinable()) {
            pipeline_thread_.join();
        }
        auto pipeline_pose_config=pose_config_; pipeline_pose_config.calibration_store=calibration_store_;
        pipeline_ = std::make_unique<Pipeline>(capture_config_, metrics_, std::move(pipeline_pose_config), rig_tool_);
        auto disk_result = pipeline_->configure_disk(disk_config_);
        if (!disk_result) {
            return from_output_result(std::move(disk_result));
        }
        auto shm_result = pipeline_->configure_shared_memory(shm_config_);
        if (!shm_result) {
            return from_output_result(std::move(shm_result));
        }
        auto requested_preview = preview_config_;
        requested_preview.shared_memory = shm_config_;
        auto preview_result = pipeline_->configure_preview(requested_preview);
        if (!preview_result) {
            return from_output_result(std::move(preview_result));
        }
        pipeline_->set_preview_status_provider([this] {
            const auto current = snapshot();
            std::string result = std::string{"{\"pipeline\":"} + json_quote(to_string(current.state)) +
                   ",\"recording\":" + (current.recording ? "true" : "false") +
                   ",\"processedPackets\":" + std::to_string(current.processed_packets) +
                   ",\"previewDropped\":" + std::to_string(current.preview.dropped_packets) +
                   ",\"previewPublished\":" + std::to_string(current.preview.published_packets) +
                   ",\"lastError\":" + json_quote(current.last_error);
            result += ",\"cameras\":[";
            for (std::size_t i = 0; i < current.cameras.size(); ++i) {
                if (i) result += ',';
                const auto& camera = current.cameras[i];
                result += "{\"camera_id\":" + std::to_string(camera.camera_id) +
                          ",\"width\":" + std::to_string(camera.capture.extent.width) +
                          ",\"height\":" + std::to_string(camera.capture.extent.height) +
                          ",\"fps\":" + std::to_string(camera.capture.frame_rate.value()) +
                          ",\"reconnect\":" + (camera.capture.reconnect ? "true" : "false") + "}";
            }
            result += "]";
            result += std::string{",\"preview\":{\"enabled\":"} +
                      (current.preview.enabled ? "true" : "false") +
                      ",\"port\":" + std::to_string(current.preview.port) +
                      ",\"last_error\":" + json_quote(current.preview.last_error) + "}";
            if (const auto rig = calibration_store_->snapshot()) {
                result += ",\"calibration\":{\"revision\":" + std::to_string(rig->revision) + ",\"cameras\":[";
                for (std::size_t i = 0; i < rig->cameras.size(); ++i) {
                    if (i) result += ',';
                    const auto& c = rig->cameras[i]; result += "{\"camera_id\":" + std::to_string(c.camera_id) + ",\"R_w2c\":[";
                    for (std::size_t j=0;j<c.R_w2c.size();++j) { if(j) result += ','; result += std::to_string(c.R_w2c[j]); }
                    result += "],\"t_w2c\":[";
                    for (std::size_t j=0;j<c.t_w2c.size();++j) { if(j) result += ','; result += std::to_string(c.t_w2c[j]); }
                    result += "]}";
                }
                result += "]}";
            }
            return result + "}";
        });
        {
            std::scoped_lock lock(state_mutex_);
            state_ = RuntimeState::Starting;
            recording_ = false;
            last_error_.clear();
        }
        pipeline_thread_ = std::thread(&Impl::pipeline_loop, this);
        {
            std::unique_lock lock(state_mutex_);
            state_changed_.wait(lock, [this] { return state_ != RuntimeState::Starting; });
            if (state_ == RuntimeState::Running) {
                return {RuntimeCommandStatus::Applied, "pipeline started", snapshot_unlocked()};
            }
            return {RuntimeCommandStatus::Failed,
                    last_error_.empty() ? "pipeline failed to start" : last_error_,
                    snapshot_unlocked()};
        }
    }

    RuntimeCommandResponse handle(const StopPipelineCommand&) {
        const auto result = stop_pipeline();
        return {result ? RuntimeCommandStatus::Applied : RuntimeCommandStatus::Rejected,
                result ? "pipeline stopped" : "pipeline is not running", snapshot()};
    }

    RuntimeCommandResponse handle(const ConfigurePoseCommand& command) {
        const bool was_running = pipeline_running();
        PoseConfig requested;
        if (command.backend == ConfigurePoseCommand::Backend::Monocular)
            requested.model_path = resolve_pose_asset(command.model_path);
        else if (command.backend == ConfigurePoseCommand::Backend::Multiview)
        {
            requested.multiview_engine_path = resolve_pose_asset(command.engine_path);
            // Keep the live calibration store when switching pose backends.  The
            // store is what `rig calibrate` updates, and the multiview stage
            // refreshes its camera matrices from it when it starts.
            requested.calibration_store = calibration_store_;
            requested.multiview_calibration_path = command.calibration_path;
            if (!requested.multiview_calibration_path.empty()) load_multiview_calibration(requested, requested.multiview_calibration_path);
            else if (!calibration_store_->snapshot())
                return {RuntimeCommandStatus::Rejected, "multiview requires a loaded/generated rig calibration", snapshot()};
        }
        PoseConfig previous;
        {
            std::scoped_lock lock(state_mutex_);
            previous = pose_config_;
        }
        if (was_running && !stop_pipeline())
            return {RuntimeCommandStatus::Failed, "could not stop pipeline for pose reconfiguration", snapshot()};
        {
            std::scoped_lock lock(state_mutex_);
            pose_config_ = std::move(requested);
        }
        if (was_running) {
            auto started = handle(StartPipelineCommand{});
            if (started) return started;
            const auto startup_error = started.message;
            { std::scoped_lock lock(state_mutex_); pose_config_ = std::move(previous); }
            auto restored = handle(StartPipelineCommand{});
            return {RuntimeCommandStatus::Failed,
                    restored ? "pose configuration failed; previous pipeline restored: " + startup_error
                             : "pose configuration failed and previous pipeline could not be restored: " + startup_error,
                    snapshot()};
        }
        return {RuntimeCommandStatus::Applied, "pose backend configured", snapshot()};
    }

    RuntimeCommandResponse handle(const StartRigCalibrationCommand& command) {
        if (!pipeline_running()) return {RuntimeCommandStatus::Rejected,"pipeline must be running to calibrate the rig",snapshot()};
        const auto engine=resolve_pose_asset("@assets/da3_base.trt");
        if (!std::filesystem::is_regular_file(engine)) return {RuntimeCommandStatus::Failed,"DA3 engine asset is missing: "+engine.string(),snapshot()};
        if (!rig_tool_->start(engine,command.output_path)) return {RuntimeCommandStatus::Rejected,"rig calibration is already running",snapshot()};
        return {RuntimeCommandStatus::Applied,"DA3 rig calibration started",snapshot()};
    }
    RuntimeCommandResponse handle(const CancelRigCalibrationCommand&) { rig_tool_->cancel(); return {RuntimeCommandStatus::Applied,"rig calibration cancelled",snapshot()}; }
    RuntimeCommandResponse handle(const ClearRigCalibrationCommand&) { calibration_store_->clear(); return {RuntimeCommandStatus::Applied,"rig calibration cleared",snapshot()}; }
    RuntimeCommandResponse handle(const GetRigCalibrationStatusCommand&) {
        const auto s=rig_tool_->status(); const auto calibration=calibration_store_->snapshot();
        return {RuntimeCommandStatus::Applied,"rig "+s.state+(s.message.empty()?"":" - "+s.message)+(calibration?" revision="+std::to_string(calibration->revision):""),snapshot()};
    }

    RuntimeCommandResponse handle(const GetStatusCommand&) {
        return {RuntimeCommandStatus::Applied, "runtime status", snapshot()};
    }

    RuntimeCommandResponse handle(const GetMetricsCommand&) {
        return {RuntimeCommandStatus::Applied, "metrics snapshot", snapshot()};
    }

    RuntimeCommandResponse handle(const StartRecordingCommand& command) {
        if (!pipeline_running()) {
            return {RuntimeCommandStatus::Rejected, "pipeline is not running", std::nullopt};
        }
        if (command.destination.empty()) {
            return {RuntimeCommandStatus::Rejected, "recording destination is empty", std::nullopt};
        }
        auto requested = disk_config_;
        requested.destination = command.destination;
        if (command.bitrate) {
            requested.bitrate = *command.bitrate;
        }
        if (command.frame_rate) {
            requested.frame_rate = *command.frame_rate;
        }
        auto configured = pipeline_->configure_disk(requested);
        if (!configured) {
            return from_output_result(std::move(configured));
        }
        {
            std::scoped_lock lock(state_mutex_);
            disk_config_ = requested;
        }
        auto started = pipeline_->start_recording();
        auto response = from_output_result(started);
        if (response) {
            std::scoped_lock lock(state_mutex_);
            recording_ = true;
        }
        return response;
    }

    RuntimeCommandResponse handle(const StopRecordingCommand&) {
        if (!pipeline_running()) {
            return {RuntimeCommandStatus::Rejected, "pipeline is not running", std::nullopt};
        }
        auto response = from_output_result(pipeline_->stop_recording());
        {
            std::scoped_lock lock(state_mutex_);
            recording_ = false;
            if (!response) {
                last_error_ = response.message;
            }
        }
        return response;
    }

    RuntimeCommandResponse handle(const ConfigureSharedMemoryCommand& command) {
        if (pipeline_running()) {
            auto response = from_output_result(pipeline_->configure_shared_memory(command.config));
            if (!response) {
                return response;
            }
        }
        {
            std::scoped_lock lock(state_mutex_);
            shm_config_ = command.config;
            preview_config_.shared_memory = command.config;
        }
        return {RuntimeCommandStatus::Applied,
                command.config.enabled ? "shared memory enabled" : "shared memory disabled",
                std::nullopt};
    }

    RuntimeCommandResponse handle(const ConfigurePreviewCommand& command) {
        if (command.config.http.enabled &&
            !(command.config.http.bind_address == "127.0.0.1" ||
              command.config.http.bind_address == "::1" ||
              command.config.http.bind_address == "localhost")) {
            return {RuntimeCommandStatus::Rejected, "preview server must bind to loopback", std::nullopt};
        }
        if (pipeline_running()) {
            auto requested = command.config;
            requested.shared_memory = shm_config_;
            auto response = from_output_result(pipeline_->configure_preview(std::move(requested)));
            if (!response) return response;
        }
        preview_config_ = command.config;
        return {RuntimeCommandStatus::Applied,
                pipeline_running() ? "preview configuration applied and transports restarted"
                                   : "preview configuration applied",
                snapshot()};
    }

    RuntimeCommandResponse handle(const ConfigureCaptureCommand& command) {
        if (command.patch.empty()) {
            return {RuntimeCommandStatus::Rejected, "capture configuration patch is empty",
                    std::nullopt};
        }

        auto requested = capture_config_;
        auto target = requested.cameras.end();
        if (command.camera_id) {
            target = std::ranges::find(requested.cameras, *command.camera_id,
                                       &CameraCaptureConfig::camera_id);
        } else if (requested.cameras.size() == 1) {
            target = requested.cameras.begin();
        } else {
            return {RuntimeCommandStatus::Rejected,
                    "--camera is required when multiple cameras are configured", std::nullopt};
        }
        if (target == requested.cameras.end()) {
            return {RuntimeCommandStatus::Rejected, "camera ID is not configured", std::nullopt};
        }
        target->capture = apply_capture_patch(target->capture, command.patch);
        if (const auto error = validate_camera_group(requested)) {
            return {RuntimeCommandStatus::Rejected, *error, std::nullopt};
        }
        return apply_camera_configuration(std::move(requested), "capture configuration updated");
    }

    RuntimeCommandResponse handle(const AddCameraCommand& command) {
        auto requested = capture_config_;
        if (std::ranges::find(requested.cameras, command.camera.camera_id,
                              &CameraCaptureConfig::camera_id) != requested.cameras.end()) {
            return {RuntimeCommandStatus::Rejected, "camera ID is already configured",
                    std::nullopt};
        }
        requested.cameras.push_back(command.camera);
        if (const auto error = validate_camera_group(requested)) {
            return {RuntimeCommandStatus::Rejected, *error, std::nullopt};
        }
        return apply_camera_configuration(std::move(requested), "camera added");
    }

    RuntimeCommandResponse handle(const RemoveCameraCommand& command) {
        if (capture_config_.cameras.size() == 1) {
            return {RuntimeCommandStatus::Rejected, "cannot remove the final camera", std::nullopt};
        }
        auto requested = capture_config_;
        const auto original_size = requested.cameras.size();
        std::erase_if(requested.cameras,
                      [&](const auto& camera) { return camera.camera_id == command.camera_id; });
        if (requested.cameras.size() == original_size) {
            return {RuntimeCommandStatus::Rejected, "camera ID is not configured", std::nullopt};
        }
        return apply_camera_configuration(std::move(requested), "camera removed");
    }

    RuntimeCommandResponse handle(const GetCamerasCommand&) {
        return {RuntimeCommandStatus::Applied, "camera configuration", snapshot()};
    }

    RuntimeCommandResponse handle(const ConfigureSynchronizerCommand& command) {
        if (command.tolerance.count() < 0 || command.queue_capacity == 0) {
            return {RuntimeCommandStatus::Rejected, "invalid synchronizer configuration",
                    std::nullopt};
        }
        auto requested = capture_config_;
        requested.sync_tolerance = command.tolerance;
        requested.sync_queue_capacity = command.queue_capacity;
        requested.incomplete_batch_policy = command.incomplete_batch_policy;
        return apply_camera_configuration(std::move(requested),
                                          "synchronizer configuration updated");
    }

    RuntimeCommandResponse handle(const ShutdownCommand&) {
        stop_pipeline();
        exporter_.stop();
        prometheus_.stop();
        {
            std::scoped_lock lock(state_mutex_);
            state_ = RuntimeState::Shutdown;
            recording_ = false;
        }
        return {RuntimeCommandStatus::Applied, "runtime shut down", snapshot()};
    }

    RuntimeCommandResponse apply_camera_configuration(MultiCameraCaptureConfig requested,
                                                      std::string message) {
        const auto previous = capture_config_;
        const bool restart = pipeline_running();
        if (restart) {
            stop_pipeline();
        }
        capture_config_ = std::move(requested);
        if (!restart) {
            return {RuntimeCommandStatus::Applied, std::move(message), snapshot()};
        }
        auto restarted = handle(StartPipelineCommand{});
        if (restarted) {
            restarted.message = std::move(message) + " and pipeline restarted";
            return restarted;
        }
        capture_config_ = previous;
        auto restored = handle(StartPipelineCommand{});
        if (restored) {
            return {RuntimeCommandStatus::Failed,
                    "requested camera configuration failed; previous pipeline restored: " +
                        restarted.message,
                    restored.snapshot};
        }
        return {
            RuntimeCommandStatus::Failed,
            "requested camera configuration failed and previous pipeline could not be restored: " +
                restarted.message,
            restored.snapshot};
    }

    void pipeline_loop() {
        try {
            pipeline_->start();
            {
                std::scoped_lock lock(state_mutex_);
                state_ = RuntimeState::Running;
            }
            state_changed_.notify_all();
            pipeline_->wait();
            {
                std::scoped_lock lock(state_mutex_);
                if (state_ != RuntimeState::Failed) {
                    state_ = RuntimeState::Stopped;
                }
                recording_ = false;
            }
        } catch (const std::exception& error) {
            pipeline_->stop();
            {
                std::scoped_lock lock(state_mutex_);
                state_ = RuntimeState::Failed;
                recording_ = false;
                last_error_ = error.what();
            }
        }
        state_changed_.notify_all();
    }

    bool stop_pipeline() {
        {
            std::scoped_lock lock(state_mutex_);
            if (state_ != RuntimeState::Running && state_ != RuntimeState::Starting &&
                state_ != RuntimeState::Failed) {
                return false;
            }
            state_ = RuntimeState::Stopping;
        }
        rig_tool_->cancel();
        bool was_recording = false;
        {
            std::scoped_lock lock(state_mutex_);
            was_recording = recording_;
        }
        if (was_recording && pipeline_) {
            auto finalized = pipeline_->stop_recording();
            if (!finalized) {
                std::scoped_lock lock(state_mutex_);
                last_error_ = finalized.message;
            }
        }
        if (pipeline_) {
            pipeline_->stop_producing();
        }
        if (pipeline_thread_.joinable()) {
            pipeline_thread_.join();
        }
        {
            std::scoped_lock lock(state_mutex_);
            state_ = RuntimeState::Stopped;
            recording_ = false;
        }
        return true;
    }

    bool pipeline_running() const {
        std::scoped_lock lock(state_mutex_);
        return state_ == RuntimeState::Running;
    }

    RuntimeSnapshot snapshot_unlocked() const {
        RuntimeSnapshot result;
        result.state = state_;
        result.recording = recording_;
        result.recording_path = disk_config_.destination;
        result.shared_memory_destination = shm_config_.destination;
        result.shared_memory_enabled = shm_config_.enabled;
        result.preview.enabled = preview_config_.http.enabled || preview_config_.mjpeg.enabled ||
                                 preview_config_.shared_memory.enabled;
        result.preview.bind_address = preview_config_.http.bind_address;
        result.preview.port = preview_config_.http.port;
        if (pipeline_) {
            const auto preview = pipeline_->preview_health();
            result.preview.enabled = result.preview.enabled || preview.enabled;
            result.preview.published_packets = preview.published_packets;
            result.preview.dropped_packets = preview.dropped_packets;
            result.preview.connected_clients = preview.connected_clients;
            result.preview.last_error = preview.last_error;
        }
        result.processed_packets = pipeline_ ? pipeline_->processed_count() : 0;
        result.cameras = capture_config_.cameras;
        result.sync_tolerance = capture_config_.sync_tolerance;
        result.sync_queue_capacity = capture_config_.sync_queue_capacity;
        result.incomplete_batch_policy = capture_config_.incomplete_batch_policy;
        result.last_error = last_error_;
        result.pose_backend = pose_backend_name(pose_config_);
        result.pose_model_path = pose_config_.model_path;
        result.pose_engine_path = pose_config_.multiview_engine_path;
        result.metrics = metrics_.snapshot();
        if (const auto active = result.metrics.gauges.find("iris_output_recording_active");
            active != result.metrics.gauges.end()) {
            result.recording = active->second != 0.0;
        }
        return result;
    }

    MultiCameraCaptureConfig capture_config_;
    DiskOutputConfig disk_config_;
    SharedMemoryOutputConfig shm_config_;
    PreviewConfig preview_config_;
    mutable std::mutex state_mutex_;
    std::condition_variable state_changed_;
    RuntimeState state_{RuntimeState::Stopped};
    bool recording_{};
    std::string last_error_;
    PoseConfig pose_config_;
    std::shared_ptr<CalibrationStore> calibration_store_;
    std::shared_ptr<RigCalibrationTool> rig_tool_;
    infrastructure::metrics::MetricRegistry metrics_;
    infrastructure::metrics::MetricsExporter exporter_;
    infrastructure::metrics::PrometheusExporter prometheus_;
    std::unique_ptr<Pipeline> pipeline_;
    std::thread pipeline_thread_;
    Channel<std::shared_ptr<CommandRequest>> commands_;
    std::thread control_thread_;
    std::atomic_bool control_started_{false};
    std::atomic_bool control_accepting_{false};
};

Runtime::Runtime(CaptureConfig config, std::uint16_t metrics_port, PoseConfig pose_config)
    : impl_(std::make_unique<Impl>(single_camera_config(std::move(config)), metrics_port,
                                  std::move(pose_config))) {}
Runtime::Runtime(MultiCameraCaptureConfig config, std::uint16_t metrics_port,
                 PoseConfig pose_config)
    : impl_(std::make_unique<Impl>(std::move(config), metrics_port, std::move(pose_config))) {}

Runtime::~Runtime() = default;

void Runtime::start() { impl_->start(); }

RuntimeCommandResponse Runtime::execute(RuntimeCommand command) {
    return impl_->execute(std::move(command));
}

RuntimeSnapshot Runtime::snapshot() const { return impl_->snapshot(); }

void Runtime::stop() { impl_->stop(); }

int Runtime::run() {
    start();
    auto started = execute(StartPipelineCommand{});
    if (!started) {
        std::cerr << "IRIS runtime failed: " << started.message << '\n';
        stop();
        return 1;
    }
    while (true) {
        const auto current = snapshot();
        if (current.state == RuntimeState::Stopped || current.state == RuntimeState::Failed ||
            current.state == RuntimeState::Shutdown) {
            const bool failed = current.state == RuntimeState::Failed;
            stop();
            return failed ? 1 : 0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

} // namespace iris
