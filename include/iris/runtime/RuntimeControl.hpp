#pragma once

#include "iris/infrastructure/metrics/MetricRegistry.hpp"
#include "iris/stages/capture/CaptureConfig.hpp"
#include "iris/stages/output/OutputConfig.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <variant>

namespace iris {

enum class RuntimeState { Stopped, Starting, Running, Stopping, Failed, Shutdown };

struct StartPipelineCommand {};
struct StopPipelineCommand {};
struct GetStatusCommand {};
struct GetMetricsCommand {
    std::string prefix;
};
struct StartRecordingCommand {
    std::filesystem::path destination;
    std::optional<std::uint32_t> bitrate;
    std::optional<std::uint32_t> frame_rate;
};
struct StopRecordingCommand {};
struct ConfigureSharedMemoryCommand {
    SharedMemoryOutputConfig config;
};
struct CaptureConfigPatch {
    std::optional<std::string> device_symbolic_link;
    std::optional<std::uint32_t> device_index;
    std::optional<std::uint32_t> width;
    std::optional<std::uint32_t> height;
    std::optional<FrameRate> frame_rate;
    std::optional<PixelFormat> format;
    std::optional<int> cuda_device;
    std::optional<std::size_t> sample_queue_capacity;
    std::optional<std::size_t> frame_pool_capacity;
    std::optional<OverflowPolicy> overflow;
    std::optional<FrameRotation> rotation;
    std::optional<bool> allow_format_fallback;
    std::optional<bool> reconnect;

    [[nodiscard]] bool empty() const noexcept {
        return !device_symbolic_link && !device_index && !width && !height && !frame_rate &&
               !format && !cuda_device && !sample_queue_capacity && !frame_pool_capacity &&
               !overflow && !rotation && !allow_format_fallback && !reconnect;
    }
};
struct ConfigureCaptureCommand {
    CaptureConfigPatch patch;
    std::optional<CameraId> camera_id;
};
struct AddCameraCommand {
    CameraCaptureConfig camera;
};
struct RemoveCameraCommand {
    CameraId camera_id{};
};
struct GetCamerasCommand {};
struct ConfigureSynchronizerCommand {
    std::chrono::milliseconds tolerance{3};
    std::size_t queue_capacity{4};
    IncompleteBatchPolicy incomplete_batch_policy{IncompleteBatchPolicy::DropBatch};
};
struct ShutdownCommand {};

using RuntimeCommand =
    std::variant<StartPipelineCommand, StopPipelineCommand, GetStatusCommand, GetMetricsCommand,
                 StartRecordingCommand, StopRecordingCommand, ConfigureSharedMemoryCommand,
                 ConfigureCaptureCommand, AddCameraCommand, RemoveCameraCommand, GetCamerasCommand,
                 ConfigureSynchronizerCommand, ShutdownCommand>;

struct RuntimeSnapshot {
    RuntimeState state{RuntimeState::Stopped};
    bool recording{};
    std::filesystem::path recording_path;
    std::string shared_memory_destination;
    bool shared_memory_enabled{};
    struct PreviewStatus {
        bool enabled{};
        std::string bind_address{"127.0.0.1"};
        std::uint16_t port{};
        std::size_t published_packets{};
        std::size_t dropped_packets{};
        std::size_t connected_clients{};
        std::string last_error;
    } preview;
    std::size_t processed_packets{};
    std::vector<CameraCaptureConfig> cameras;
    std::chrono::milliseconds sync_tolerance{3};
    std::size_t sync_queue_capacity{4};
    IncompleteBatchPolicy incomplete_batch_policy{IncompleteBatchPolicy::DropBatch};
    std::string last_error;
    infrastructure::metrics::MetricsSnapshot metrics;
};

enum class RuntimeCommandStatus { Applied, Rejected, Failed };

struct RuntimeCommandResponse {
    RuntimeCommandStatus status{RuntimeCommandStatus::Applied};
    std::string message;
    std::optional<RuntimeSnapshot> snapshot;

    [[nodiscard]] explicit operator bool() const noexcept {
        return status == RuntimeCommandStatus::Applied;
    }
};

const char* to_string(RuntimeState state) noexcept;

} // namespace iris
