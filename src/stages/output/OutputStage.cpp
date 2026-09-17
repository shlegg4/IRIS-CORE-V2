#include "iris/stages/OutputStage.hpp"
#include "iris/stages/output/PreviewSink.hpp"

#include "stages/output/disk/GpuMp4Writer.hpp"

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#endif

namespace iris {
namespace {

using infrastructure::metrics::MetricRegistry;
using PacketPtr = std::shared_ptr<const Packet>;

std::filesystem::path camera_recording_path(const std::filesystem::path& destination,
                                            CameraId camera) {
    const auto filename = destination.stem().string() + "-camera-" + std::to_string(camera) +
                          destination.extension().string();
    return destination.parent_path() / filename;
}

struct OutputMetrics {
    explicit OutputMetrics(MetricRegistry& registry)
        : packets_received(registry.counter("iris_output_packets_received_total")),
          shm_packets(registry.counter("iris_output_shm_packets_total")),
          shm_failures(registry.counter("iris_output_shm_failures_total")),
          disk_accepted(registry.counter("iris_output_disk_accepted_frames_total")),
          disk_encoded(registry.counter("iris_output_disk_encoded_frames_total")),
          disk_packets(registry.counter("iris_output_disk_packets_total")),
          disk_bytes(registry.counter("iris_output_disk_bytes_total")),
          disk_failures(registry.counter("iris_output_disk_failures_total")),
          recording(registry.gauge("iris_output_recording_active")) {}

    infrastructure::metrics::Counter packets_received;
    infrastructure::metrics::Counter shm_packets;
    infrastructure::metrics::Counter shm_failures;
    infrastructure::metrics::Counter disk_accepted;
    infrastructure::metrics::Counter disk_encoded;
    infrastructure::metrics::Counter disk_packets;
    infrastructure::metrics::Counter disk_bytes;
    infrastructure::metrics::Counter disk_failures;
    infrastructure::metrics::Gauge recording;
};

template <typename T> void append_value(std::vector<std::byte>& bytes, const T& value) {
    const auto* first = reinterpret_cast<const std::byte*>(&value);
    bytes.insert(bytes.end(), first, first + sizeof(T));
}

std::vector<std::byte> serialize_shm_packet(const Packet& packet) {
    constexpr std::uint32_t version = 2;
    std::vector<std::byte> bytes;
    append_value(bytes, version);
    append_value(bytes, packet.sequence);
    const auto frame_count = static_cast<std::uint32_t>(packet.frames.size());
    const auto pose_count = static_cast<std::uint32_t>(packet.poses ? packet.poses->size() : 0);
    append_value(bytes, frame_count);
    append_value(bytes, pose_count);
    for (const auto& frame : packet.frames) {
        append_value(bytes, frame.camera);
        append_value(bytes, frame.sequence);
        append_value(bytes, frame.extent.width);
        append_value(bytes, frame.extent.height);
        append_value(bytes, frame.format);
        append_value(bytes, frame.buffer.stride_bytes);
        append_value(bytes, frame.buffer.size_bytes);
        cudaIpcMemHandle_t handle{};
        if (!frame.buffer.data || cudaIpcGetMemHandle(&handle, frame.buffer.data) != cudaSuccess) {
            throw std::runtime_error("could not export output frame through CUDA IPC");
        }
        const auto* handle_bytes = reinterpret_cast<const std::byte*>(&handle);
        bytes.insert(bytes.end(), handle_bytes, handle_bytes + sizeof(handle));
    }
    if (packet.poses) {
        for (const auto& pose : *packet.poses) {
            append_value(bytes, pose.source_sequence);
        }
    }
    const auto multiview_count = static_cast<std::uint32_t>(packet.multiview_poses ? packet.multiview_poses->size() : 0);
    append_value(bytes, multiview_count);
    if (packet.multiview_poses) for (const auto& pose : *packet.multiview_poses) {
        const std::uint8_t active = pose.active ? 1 : 0; append_value(bytes, active);
        for (const auto& joint : pose.joints_3d) for (float value : joint) append_value(bytes, value);
        for (bool valid : pose.joint_valid) { const std::uint8_t value = valid ? 1 : 0; append_value(bytes, value); }
        for (const auto& view : pose.joint_scores) for (float score : view) append_value(bytes, score);
    }
    return bytes;
}

struct DiskStart {
    std::shared_ptr<std::promise<OutputCommandResult>> completion;
};
struct DiskStop {
    std::shared_ptr<std::promise<OutputCommandResult>> completion;
};
struct DiskConfigure {
    DiskOutputConfig config;
    std::shared_ptr<std::promise<OutputCommandResult>> completion;
};
using DiskWork = std::variant<PacketPtr, DiskStart, DiskStop, DiskConfigure>;

OutputCommandResult
await_command(Channel<DiskWork>& channel, DiskWork command,
              const std::shared_ptr<std::promise<OutputCommandResult>>& completion) {
    auto result = completion->get_future();
    if (channel.send(std::move(command)) == SendResult::Closed) {
        return {OutputCommandStatus::Rejected, "output stage is stopped"};
    }
    return result.get();
}

#ifdef _WIN32
std::wstring widen(const std::string& value) {
    const int size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                                         static_cast<int>(value.size()), nullptr, 0);
    if (size <= 0) {
        throw std::runtime_error("shared-memory destination is not valid UTF-8");
    }
    std::wstring result(static_cast<std::size_t>(size), L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()),
                        result.data(), size);
    return result;
}

struct SharedMappingHeader {
    alignas(8) volatile LONG64 sequence_lock{};
    std::uint64_t packet_sequence{};
    std::uint64_t payload_size{};
};

class SharedMapping {
  public:
    explicit SharedMapping(const SharedMemoryOutputConfig& config) {
        if (config.capacity_bytes <= sizeof(SharedMappingHeader)) {
            throw std::invalid_argument("shared-memory capacity is too small");
        }
        const auto name = widen(config.destination);
        handle_ = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                                     static_cast<DWORD>(config.capacity_bytes >> 32U),
                                     static_cast<DWORD>(config.capacity_bytes), name.c_str());
        if (!handle_) {
            throw std::runtime_error("CreateFileMappingW failed: " +
                                     std::to_string(GetLastError()));
        }
        view_ = MapViewOfFile(handle_, FILE_MAP_ALL_ACCESS, 0, 0, config.capacity_bytes);
        if (!view_) {
            const auto error = GetLastError();
            CloseHandle(handle_);
            handle_ = nullptr;
            throw std::runtime_error("MapViewOfFile failed: " + std::to_string(error));
        }
        capacity_ = config.capacity_bytes;
    }

    ~SharedMapping() {
        if (view_) {
            UnmapViewOfFile(view_);
        }
        if (handle_) {
            CloseHandle(handle_);
        }
    }

    void publish(std::uint64_t sequence, const std::vector<std::byte>& payload) {
        if (payload.size() > capacity_ - sizeof(SharedMappingHeader)) {
            throw std::runtime_error("packet exceeds configured shared-memory capacity");
        }
        auto* header = static_cast<SharedMappingHeader*>(view_);
        InterlockedIncrement64(&header->sequence_lock);
        header->packet_sequence = sequence;
        header->payload_size = payload.size();
        std::memcpy(header + 1, payload.data(), payload.size());
        MemoryBarrier();
        InterlockedIncrement64(&header->sequence_lock);
    }

  private:
    HANDLE handle_{};
    void* view_{};
    std::size_t capacity_{};
};
#endif

} // namespace

class OutputStage::Impl {
  public:
    Impl(Channel<Packet>& input, MetricRegistry& registry, OutputConfig config)
        : input_(input), config_(std::move(config)), metrics_(registry),
          preview_(make_preview_config(config_)),
          shm_queue_(config_.shared_memory_queue_capacity, OverflowPolicy::DropOldest,
                     infrastructure::metrics::register_channel_metrics(
                         registry, "iris_channel_output_to_shm")),
          disk_queue_(config_.disk.queue_capacity, OverflowPolicy::Block,
                      infrastructure::metrics::register_channel_metrics(
                          registry, "iris_channel_output_to_disk")) {
        if (config_.shared_memory_queue_capacity == 0 || config_.disk.queue_capacity == 0) {
            throw std::invalid_argument("output queue capacities must be greater than zero");
        }
    }

    ~Impl() { stop(); }

    void start() {
        if (running_.exchange(true)) {
            return;
        }
        if (config_.shared_memory.enabled) {
            const auto result = configure_shared_memory(config_.shared_memory);
            if (!result) {
                running_.store(false);
                throw std::runtime_error(result.message);
            }
        }
        preview_.start();
        disk_worker_ = std::thread(&Impl::run_disk, this);
        coordinator_ = std::thread(&Impl::run_coordinator, this);
    }

    void stop() {
        if (running_.exchange(false)) {
            input_.close();
        }
        if (coordinator_.joinable()) {
            coordinator_.join();
        }
        preview_.stop();
        shm_queue_.close();
        disk_queue_.close();
        if (disk_worker_.joinable()) {
            disk_worker_.join();
        }
        accepting_disk_.store(false);
        metrics_.recording.set(0.0);
    }

    OutputCommandResult configure_shared_memory(SharedMemoryOutputConfig config) {
        auto preview_result = preview_.configure_shared_memory(config);
        if (!preview_result) {
            return preview_result;
        }
        std::scoped_lock lock(shm_config_mutex_);
        try {
#ifdef _WIN32
            config_.shared_memory = std::move(config);
            return {OutputCommandStatus::Applied, "shared-memory configuration applied"};
#else
            config_.shared_memory = std::move(config);
            return {OutputCommandStatus::Applied, "shared-memory configuration applied"};
#endif
        } catch (const std::exception& error) {
            metrics_.shm_failures.increment();
            return {OutputCommandStatus::Failed, error.what()};
        }
    }
    OutputCommandResult configure_preview(PreviewConfig config) { return preview_.configure(std::move(config)); }
    void set_preview_status_provider(std::function<std::string()> provider) { preview_.set_status_provider(std::move(provider)); }

    OutputCommandResult configure_disk(DiskOutputConfig config) {
        if (config.queue_capacity != config_.disk.queue_capacity) {
            return {OutputCommandStatus::Rejected,
                    "disk queue capacity is fixed when OutputStage is constructed"};
        }
        if (!running_) {
            config_.disk = std::move(config);
            return {OutputCommandStatus::Applied, "disk configuration applied"};
        }
        std::scoped_lock lock(disk_submit_mutex_);
        if (accepting_disk_ || recording_failed_) {
            return {OutputCommandStatus::Rejected,
                    "finish or acknowledge the current recording before reconfiguration"};
        }
        auto completion = std::make_shared<std::promise<OutputCommandResult>>();
        const auto applied = config;
        auto result =
            await_command(disk_queue_, DiskConfigure{std::move(config), completion}, completion);
        if (result) {
            config_.disk = applied;
        }
        return result;
    }

    OutputCommandResult start_recording() {
        if (!running_) {
            return {OutputCommandStatus::Rejected, "start OutputStage before recording"};
        }
        std::scoped_lock lock(disk_submit_mutex_);
        if (accepting_disk_) {
            return {OutputCommandStatus::Rejected, "recording is already active"};
        }
        auto completion = std::make_shared<std::promise<OutputCommandResult>>();
        auto result = await_command(disk_queue_, DiskStart{completion}, completion);
        if (result) {
            recording_failed_.store(false);
            accepting_disk_.store(true);
            metrics_.recording.set(1.0);
        }
        return result;
    }

    OutputCommandResult stop_recording() {
        if (!running_) {
            return {OutputCommandStatus::Rejected, "output stage is stopped"};
        }
        std::scoped_lock lock(disk_submit_mutex_);
        if (!accepting_disk_ && !recording_failed_) {
            return {OutputCommandStatus::Rejected, "recording is not active"};
        }
        accepting_disk_.store(false);
        auto completion = std::make_shared<std::promise<OutputCommandResult>>();
        auto result = await_command(disk_queue_, DiskStop{completion}, completion);
        recording_failed_.store(false);
        metrics_.recording.set(0.0);
        return result;
    }

    std::size_t processed_count() const noexcept { return processed_count_.load(); }
    PreviewTransportHealth preview_health() const { return preview_.shared_memory_health(); }

  private:
    void run_coordinator() {
        while (auto packet = input_.receive()) {
            metrics_.packets_received.increment();
            auto retained = std::make_shared<Packet>(std::move(*packet));
            ++processed_count_;
            {
                preview_.publish(retained);
            }
            std::scoped_lock lock(disk_submit_mutex_);
            if (accepting_disk_) {
                const auto frames = retained->frames.size();
                if (disk_queue_.send(DiskWork{std::move(retained)}) != SendResult::Closed) {
                    metrics_.disk_accepted.increment(frames);
                }
            }
        }
    }

    static PreviewConfig make_preview_config(const OutputConfig& output) {
        auto preview = output.preview;
        preview.shared_memory = output.shared_memory;
        preview.shared_memory_queue_capacity = output.shared_memory_queue_capacity;
        return preview;
    }

    void run_disk() {
        DiskOutputConfig disk_config = config_.disk;
        std::unordered_map<CameraId, std::unique_ptr<output::disk::GpuMp4Writer>> writers;
        std::unordered_map<CameraId, std::uint64_t> accepted_frames;
        std::string failure;
        while (auto work = disk_queue_.receive()) {
            if (auto* packet = std::get_if<PacketPtr>(&*work)) {
                if (!failure.empty()) {
                    continue;
                }
                try {
                    if ((*packet)->frames.empty()) {
                        throw std::runtime_error("cannot record an empty frame batch");
                    }
                    for (const auto& frame : (*packet)->frames) {
                        auto& writer = writers[frame.camera];
                        if (!writer) {
                            auto camera_config = disk_config;
                            if (config_.camera_count > 1) {
                                camera_config.destination =
                                    camera_recording_path(disk_config.destination, frame.camera);
                            }
                            writer =
                                std::make_unique<output::disk::GpuMp4Writer>(camera_config, frame);
                        }
                        const auto muxed_before = writer->encoded_frames();
                        const auto bytes_before = writer->bytes_written();
                        writer->write(frame);
                        ++accepted_frames[frame.camera];
                        metrics_.disk_encoded.increment();
                        metrics_.disk_packets.increment(writer->encoded_frames() - muxed_before);
                        metrics_.disk_bytes.increment(writer->bytes_written() - bytes_before);
                    }
                } catch (const std::exception& error) {
                    failure = error.what();
                    accepting_disk_.store(false);
                    recording_failed_.store(true);
                    metrics_.disk_failures.increment();
                    metrics_.recording.set(0.0);
                }
            } else if (auto* start = std::get_if<DiskStart>(&*work)) {
                writers.clear();
                accepted_frames.clear();
                failure.clear();
                start->completion->set_value({OutputCommandStatus::Applied, "recording armed"});
            } else if (auto* stop = std::get_if<DiskStop>(&*work)) {
                if (!failure.empty()) {
                    writers.clear();
                    stop->completion->set_value({OutputCommandStatus::Failed, failure});
                } else if (writers.empty()) {
                    stop->completion->set_value(
                        {OutputCommandStatus::Failed, "recording contained no frames"});
                } else {
                    try {
                        for (auto& [camera, writer] : writers) {
                            const auto muxed_before = writer->encoded_frames();
                            const auto bytes_before = writer->bytes_written();
                            writer->finalize();
                            metrics_.disk_packets.increment(writer->encoded_frames() -
                                                            muxed_before);
                            metrics_.disk_bytes.increment(writer->bytes_written() - bytes_before);
                            if (writer->encoded_frames() != accepted_frames[camera]) {
                                throw std::runtime_error(
                                    "recording accounting mismatch for camera " +
                                    std::to_string(camera));
                            }
                        }
                        writers.clear();
                        stop->completion->set_value(
                            {OutputCommandStatus::Applied, config_.camera_count > 1
                                                               ? "camera MP4 recordings finalized"
                                                               : "MP4 recording finalized"});
                    } catch (const std::exception& error) {
                        metrics_.disk_failures.increment();
                        writers.clear();
                        stop->completion->set_value({OutputCommandStatus::Failed, error.what()});
                    }
                }
            } else if (auto* configure = std::get_if<DiskConfigure>(&*work)) {
                if (!writers.empty()) {
                    configure->completion->set_value(
                        {OutputCommandStatus::Rejected, "recording is active"});
                } else {
                    disk_config = std::move(configure->config);
                    configure->completion->set_value(
                        {OutputCommandStatus::Applied, "disk configuration applied"});
                }
            }
        }
    }

    Channel<Packet>& input_;
    OutputConfig config_;
    OutputMetrics metrics_;
    PreviewSink preview_;
    Channel<PacketPtr> shm_queue_;
    Channel<DiskWork> disk_queue_;
    std::atomic_bool running_{false};
    std::atomic_bool accepting_disk_{false};
    std::atomic_bool recording_failed_{false};
    std::atomic_size_t processed_count_{0};
    std::thread coordinator_;
    std::thread disk_worker_;
    std::mutex disk_submit_mutex_;
    std::mutex shm_config_mutex_;
#ifdef _WIN32
    std::unique_ptr<SharedMapping> shm_mapping_;
#endif
};

OutputStage::OutputStage(Channel<Packet>& input, MetricRegistry& metrics, OutputConfig config)
    : impl_(std::make_unique<Impl>(input, metrics, std::move(config))) {}

OutputStage::~OutputStage() = default;

void OutputStage::start() { impl_->start(); }

void OutputStage::stop() { impl_->stop(); }

OutputCommandResult OutputStage::configure_shared_memory(SharedMemoryOutputConfig config) {
    return impl_->configure_shared_memory(std::move(config));
}
OutputCommandResult OutputStage::configure_preview(PreviewConfig config) { return impl_->configure_preview(std::move(config)); }
void OutputStage::set_preview_status_provider(std::function<std::string()> provider) { impl_->set_preview_status_provider(std::move(provider)); }

OutputCommandResult OutputStage::configure_disk(DiskOutputConfig config) {
    return impl_->configure_disk(std::move(config));
}

OutputCommandResult OutputStage::start_recording() { return impl_->start_recording(); }

OutputCommandResult OutputStage::stop_recording() { return impl_->stop_recording(); }

std::size_t OutputStage::processed_count() const noexcept { return impl_->processed_count(); }
PreviewTransportHealth OutputStage::preview_health() const { return impl_->preview_health(); }

} // namespace iris
