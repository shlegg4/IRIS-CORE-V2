#pragma once

#include "iris/infrastructure/metrics/MetricRegistry.hpp"
#include "iris/stages/capture/CaptureConfig.hpp"
#include "iris/stages/capture/SynchronizedVideoConfig.hpp"
#include "iris/stages/output/OutputConfig.hpp"

#include <filesystem>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace iris {

enum class RuntimeState { Stopped, Starting, Running, Stopping, Failed, Shutdown };

struct StartPipelineCommand {};
struct StopPipelineCommand {};
struct ConfigurePoseCommand {
    enum class Backend { Off, Monocular, TwoDimensional, Multiview } backend{Backend::Off};
    std::filesystem::path model_path;
    std::filesystem::path engine_path;
    std::filesystem::path calibration_path;
};
struct StartRigCalibrationCommand { std::filesystem::path output_path; };
struct CancelRigCalibrationCommand {};
struct ClearRigCalibrationCommand {};
struct GetRigCalibrationStatusCommand {};
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
struct ConfigurePreviewCommand { PreviewConfig config; };
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
    std::chrono::milliseconds tolerance{20};
    std::size_t queue_capacity{4};
    IncompleteBatchPolicy incomplete_batch_policy{IncompleteBatchPolicy::DropBatch};
};
struct ConfigureVideoIngestionCommand { SynchronizedVideoConfig config; };
struct UseLiveCaptureCommand {};
struct ShutdownCommand {};

using RuntimeCommand =
    std::variant<StartPipelineCommand, StopPipelineCommand, ConfigurePoseCommand, StartRigCalibrationCommand,
                 CancelRigCalibrationCommand, ClearRigCalibrationCommand, GetRigCalibrationStatusCommand, GetStatusCommand, GetMetricsCommand,
                 StartRecordingCommand, StopRecordingCommand, ConfigureSharedMemoryCommand,
                 ConfigurePreviewCommand,
                 ConfigureCaptureCommand, AddCameraCommand, RemoveCameraCommand, GetCamerasCommand,
                 ConfigureSynchronizerCommand, ConfigureVideoIngestionCommand,
                 UseLiveCaptureCommand, ShutdownCommand>;

struct RuntimeSnapshot {
    struct CalibrationCamera {
        CameraId camera_id{};
        std::array<float, 9> R_w2c{};
        std::array<float, 3> t_w2c{};
    };
    struct Calibration {
        std::uint64_t revision{};
        std::string source;
        std::vector<CalibrationCamera> cameras;
    };
    struct CalibrationToolStatus {
        std::string state{"idle"};
        std::string message;
        std::uint64_t source_sequence{};
    };

    RuntimeState state{RuntimeState::Stopped};
    std::optional<Calibration> calibration;
    CalibrationToolStatus calibration_tool;
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
        std::size_t event_clients{};
        std::size_t h264_clients{};
        std::size_t mjpeg_clients{};
        std::string last_error;
    } preview;
    std::size_t processed_packets{};
    std::vector<CameraCaptureConfig> cameras;
    std::string input_mode{"live"};
    std::vector<VideoCameraInput> video_inputs;
    std::vector<VideoDecodeStatus> video_decode_status;
    int video_cuda_device{};
    std::size_t video_frame_pool_capacity{8};
    bool video_realtime{};
    bool video_loop{};
    std::chrono::milliseconds sync_tolerance{20};
    std::size_t sync_queue_capacity{4};
    IncompleteBatchPolicy incomplete_batch_policy{IncompleteBatchPolicy::DropBatch};
    std::string last_error;
    std::string pose_backend{"off"};
    std::filesystem::path pose_model_path;
    std::filesystem::path pose_engine_path;
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
