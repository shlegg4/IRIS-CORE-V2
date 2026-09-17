#include "iris/cli/InteractiveCli.hpp"

#include "iris/runtime/Runtime.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <vector>

namespace iris {
namespace {

std::vector<std::string> tokenize(const std::string& line) {
    std::istringstream input(line);
    std::vector<std::string> tokens;
    std::string token;
    while (input >> std::quoted(token)) {
        tokens.push_back(std::move(token));
    }
    return tokens;
}

std::optional<std::uint32_t> parse_positive(const std::string& value) {
    try {
        std::size_t consumed = 0;
        const auto parsed = std::stoull(value, &consumed);
        if (consumed != value.size() || parsed == 0 || parsed > UINT32_MAX) {
            return std::nullopt;
        }
        return static_cast<std::uint32_t>(parsed);
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<std::uint32_t> parse_non_negative(const std::string& value) {
    try {
        std::size_t consumed = 0;
        const auto parsed = std::stoull(value, &consumed);
        if (consumed != value.size() || parsed > UINT32_MAX) {
            return std::nullopt;
        }
        return static_cast<std::uint32_t>(parsed);
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<int> parse_non_negative_int(const std::string& value) {
    try {
        std::size_t consumed = 0;
        const auto parsed = std::stoll(value, &consumed);
        if (consumed != value.size() || parsed < 0 || parsed > INT_MAX) {
            return std::nullopt;
        }
        return static_cast<int>(parsed);
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<bool> parse_bool(const std::string& value) {
    if (value == "true") {
        return true;
    }
    if (value == "false") {
        return false;
    }
    return std::nullopt;
}

std::optional<FrameRate> parse_frame_rate(const std::string& value) {
    const auto separator = value.find('/');
    const auto numerator = parse_positive(value.substr(0, separator));
    if (!numerator) {
        return std::nullopt;
    }
    if (separator == std::string::npos) {
        return FrameRate{*numerator, 1};
    }
    const auto denominator = parse_positive(value.substr(separator + 1));
    if (!denominator || value.find('/', separator + 1) != std::string::npos) {
        return std::nullopt;
    }
    return FrameRate{*numerator, *denominator};
}

std::optional<PixelFormat> parse_pixel_format(const std::string& value) {
    if (value == "mjpeg") {
        return PixelFormat::Mjpeg;
    }
    if (value == "yuy2") {
        return PixelFormat::Yuy2;
    }
    if (value == "bgra8") {
        return PixelFormat::Bgra8;
    }
    return std::nullopt;
}

std::optional<OverflowPolicy> parse_overflow_policy(const std::string& value) {
    if (value == "block") {
        return OverflowPolicy::Block;
    }
    if (value == "drop-oldest") {
        return OverflowPolicy::DropOldest;
    }
    if (value == "drop-newest") {
        return OverflowPolicy::DropNewest;
    }
    return std::nullopt;
}

std::optional<FrameRotation> parse_rotation(const std::string& value) {
    if (value == "none") {
        return FrameRotation::None;
    }
    if (value == "cw90") {
        return FrameRotation::Clockwise90;
    }
    if (value == "180") {
        return FrameRotation::Rotate180;
    }
    if (value == "ccw90") {
        return FrameRotation::CounterClockwise90;
    }
    return std::nullopt;
}

void print_snapshot(const RuntimeSnapshot& snapshot, std::ostream& output) {
    output << "pipeline:  " << to_string(snapshot.state) << '\n';
    output << "recording: " << (snapshot.recording ? "active" : "inactive") << '\n';
    output << "file:      " << snapshot.recording_path.string() << '\n';
    output << "packets:   " << snapshot.processed_packets << '\n';
    output << "pose:      " << snapshot.pose_backend;
    if (!snapshot.pose_model_path.empty()) output << " " << snapshot.pose_model_path.string();
    if (!snapshot.pose_engine_path.empty()) output << " " << snapshot.pose_engine_path.string();
    output << '\n';
    output << "shm:       " << (snapshot.shared_memory_enabled ? "enabled " : "disabled ")
           << snapshot.shared_memory_destination << '\n';
    output << "preview:   " << (snapshot.preview.enabled ? "enabled" : "disabled")
           << " " << snapshot.preview.bind_address << ':' << snapshot.preview.port
           << " published=" << snapshot.preview.published_packets
           << " dropped=" << snapshot.preview.dropped_packets << '\n';
    output << "cameras:   " << snapshot.cameras.size() << '\n';
    for (const auto& camera : snapshot.cameras) {
        output << "  [" << camera.camera_id << "] ";
        if (!camera.capture.device_symbolic_link.empty()) {
            output << camera.capture.device_symbolic_link;
        } else {
            output << "device-index=" << camera.capture.device_index.value_or(0);
        }
        output << " " << camera.capture.extent.width << "x" << camera.capture.extent.height << "@"
               << camera.capture.frame_rate.value() << '\n';
    }
    output << "sync:      tolerance=" << snapshot.sync_tolerance.count()
           << "ms capacity=" << snapshot.sync_queue_capacity << " policy="
           << (snapshot.incomplete_batch_policy == IncompleteBatchPolicy::DropBatch ? "drop"
                                                                                    : "partial")
           << '\n';
    if (!snapshot.last_error.empty()) {
        output << "error:     " << snapshot.last_error << '\n';
    }
}

void print_metrics(const RuntimeSnapshot& snapshot, const std::string& prefix,
                   std::ostream& output) {
    std::vector<std::pair<std::string, std::string>> values;
    for (const auto& [name, value] : snapshot.metrics.counters) {
        if (prefix.empty() || name.starts_with(prefix)) {
            values.emplace_back(name, std::to_string(value));
        }
    }
    for (const auto& [name, value] : snapshot.metrics.gauges) {
        if (prefix.empty() || name.starts_with(prefix)) {
            std::ostringstream formatted;
            formatted << value;
            values.emplace_back(name, formatted.str());
        }
    }
    for (const auto& [name, value] : snapshot.metrics.histograms) {
        if (prefix.empty() || name.starts_with(prefix)) {
            std::ostringstream formatted;
            formatted << "count=" << value.count << " sum=" << value.sum;
            values.emplace_back(name, formatted.str());
        }
    }
    std::ranges::sort(values, {}, &std::pair<std::string, std::string>::first);
    for (const auto& [name, value] : values) {
        output << name << " = " << value << '\n';
    }
    if (values.empty()) {
        output << "no matching metrics\n";
    }
}

} // namespace

InteractiveCli::InteractiveCli(Runtime& runtime) : runtime_(runtime) {}

int InteractiveCli::run(std::istream& input, std::ostream& output) {
    output << "IRIS interactive control. Type 'help' for commands.\n";
    std::string line;
    while (output << "iris> " << std::flush, std::getline(input, line)) {
        if (line == "help") {
            output << help();
            continue;
        }
        std::string error;
        auto command = parse(line, error);
        if (!command) {
            if (!error.empty()) {
                output << "error: " << error << '\n';
            }
            continue;
        }
        const bool shutdown = std::holds_alternative<ShutdownCommand>(*command);
        const auto metrics_prefix = std::holds_alternative<GetMetricsCommand>(*command)
                                        ? std::get<GetMetricsCommand>(*command).prefix
                                        : std::string{};
        auto response = runtime_.execute(std::move(*command));
        output << (response ? "ok: " : "error: ") << response.message << '\n';
        if (response.snapshot) {
            if (response.message == "metrics snapshot") {
                print_metrics(*response.snapshot, metrics_prefix, output);
            } else {
                print_snapshot(*response.snapshot, output);
            }
        }
        if (shutdown) {
            return response ? 0 : 1;
        }
    }
    runtime_.execute(ShutdownCommand{});
    return 0;
}

std::optional<RuntimeCommand> InteractiveCli::parse(std::string line, std::string& error) {
    error.clear();
    auto tokens = tokenize(line);
    if (tokens.empty()) {
        return std::nullopt;
    }
    if (tokens[0] == "status" && tokens.size() == 1) {
        return GetStatusCommand{};
    }
    if (tokens[0] == "metrics" && tokens.size() <= 2) {
        return GetMetricsCommand{tokens.size() == 2 ? tokens[1] : ""};
    }
    if (tokens[0] == "pipeline" && tokens.size() == 2) {
        if (tokens[1] == "start") {
            return StartPipelineCommand{};
        }
        if (tokens[1] == "stop") {
            return StopPipelineCommand{};
        }
    }
    if (tokens[0] == "pose") {
        if (tokens.size() == 2 && tokens[1] == "status") return GetStatusCommand{};
        if (tokens.size() == 2 && tokens[1] == "off") return ConfigurePoseCommand{};
        if (tokens.size() == 2 && tokens[1] == "monocular")
            return ConfigurePoseCommand{ConfigurePoseCommand::Backend::Monocular, "@assets/pear_ehm_libtorch.pt", {}, {}};
        if (tokens.size() == 3 && tokens[1] == "monocular")
            return ConfigurePoseCommand{ConfigurePoseCommand::Backend::Monocular, tokens[2], {}};
        if (tokens.size() == 3 && tokens[1] == "multiview")
            return ConfigurePoseCommand{ConfigurePoseCommand::Backend::Multiview, {}, "@assets/rtmo_s_full_epipolar_fp16.engine", tokens[2]};
        if (tokens.size() == 4 && tokens[1] == "multiview")
            return ConfigurePoseCommand{ConfigurePoseCommand::Backend::Multiview, {}, tokens[2], tokens[3]};
        error = "pose requires: status | off | monocular [model-path] | multiview [engine-path] <calibration.json>";
        return std::nullopt;
    }
    if (tokens[0] == "record" && tokens.size() >= 2) {
        if (tokens[1] == "stop" && tokens.size() == 2) {
            return StopRecordingCommand{};
        }
        if (tokens[1] == "status" && tokens.size() == 2) {
            return GetStatusCommand{};
        }
        if (tokens[1] == "start" && tokens.size() >= 3 && tokens.size() <= 5) {
            StartRecordingCommand command;
            command.destination = tokens[2];
            if (tokens.size() >= 4) {
                command.bitrate = parse_positive(tokens[3]);
                if (!command.bitrate) {
                    error = "bitrate must be a positive integer";
                    return std::nullopt;
                }
            }
            if (tokens.size() == 5) {
                command.frame_rate = parse_positive(tokens[4]);
                if (!command.frame_rate) {
                    error = "frame rate must be a positive integer";
                    return std::nullopt;
                }
            }
            return command;
        }
    }
    if (tokens[0] == "shm" && tokens.size() >= 2) {
        if (tokens[1] == "disable" && tokens.size() == 2) {
            SharedMemoryOutputConfig config;
            config.enabled = false;
            return ConfigureSharedMemoryCommand{std::move(config)};
        }
        if (tokens[1] == "enable" && tokens.size() == 3) {
            SharedMemoryOutputConfig config;
            config.enabled = true;
            config.destination = tokens[2];
            return ConfigureSharedMemoryCommand{std::move(config)};
        }
    }
    if (tokens[0] == "preview" && tokens.size() == 2 && tokens[1] == "status") {
        return GetStatusCommand{};
    }
    if (tokens[0] == "preview" && tokens.size() >= 2 && tokens.size() <= 3) {
        PreviewConfig config;
        if (tokens[1] == "enable") {
            config.http.enabled = true;
            config.mjpeg.enabled = true;
            if (tokens.size() == 3) {
                const auto port = parse_positive(tokens[2]);
                if (!port || *port > UINT16_MAX) { error = "preview port must be between 1 and 65535"; return std::nullopt; }
                config.http.port = static_cast<std::uint16_t>(*port);
            }
            return ConfigurePreviewCommand{std::move(config)};
        }
        if (tokens[1] == "disable" && tokens.size() == 2) return ConfigurePreviewCommand{std::move(config)};
    }
    if (tokens[0] == "capture" && tokens.size() == 2 && tokens[1] == "list") {
        return GetCamerasCommand{};
    }
    if (tokens[0] == "capture" && tokens.size() == 5 && tokens[1] == "sync") {
        const auto tolerance = parse_non_negative(tokens[2]);
        const auto capacity = parse_positive(tokens[3]);
        if (!tolerance || !capacity || (tokens[4] != "drop" && tokens[4] != "partial")) {
            error = "capture sync requires: <tolerance-ms> <capacity> <drop|partial>";
            return std::nullopt;
        }
        return ConfigureSynchronizerCommand{std::chrono::milliseconds(*tolerance), *capacity,
                                            tokens[4] == "drop"
                                                ? IncompleteBatchPolicy::DropBatch
                                                : IncompleteBatchPolicy::EmitPartial};
    }
    if (tokens[0] == "capture" && (tokens.size() == 4 || tokens.size() == 8) &&
        tokens[1] == "add") {
        const auto camera_id = parse_non_negative(tokens[2]);
        const auto device_index = parse_non_negative(tokens[3]);
        if (!camera_id || !device_index) {
            error = "capture add requires non-negative camera and device indexes";
            return std::nullopt;
        }
        CameraCaptureConfig camera;
        camera.camera_id = *camera_id;
        camera.capture.device_index = *device_index;
        if (tokens.size() == 8) {
            const auto width = parse_positive(tokens[4]);
            const auto height = parse_positive(tokens[5]);
            const auto frame_rate = parse_frame_rate(tokens[6]);
            const auto format = parse_pixel_format(tokens[7]);
            if (!width || !height || !frame_rate || !format) {
                error = "capture add mode must be: <width> <height> <fps> <mjpeg|yuy2|bgra8>";
                return std::nullopt;
            }
            camera.capture.extent = {*width, *height};
            camera.capture.frame_rate = *frame_rate;
            camera.capture.format = *format;
        }
        return AddCameraCommand{std::move(camera)};
    }
    if (tokens[0] == "capture" && tokens.size() == 3 && tokens[1] == "remove") {
        const auto camera_id = parse_non_negative(tokens[2]);
        if (!camera_id) {
            error = "camera ID must be a non-negative integer";
            return std::nullopt;
        }
        return RemoveCameraCommand{*camera_id};
    }
    if (tokens[0] == "capture" && tokens.size() >= 3 && tokens[1] == "configure") {
        if ((tokens.size() - 2) % 2 != 0) {
            error = "capture options must be specified as --name value pairs";
            return std::nullopt;
        }
        CaptureConfigPatch patch;
        std::optional<CameraId> camera_id;
        for (std::size_t index = 2; index < tokens.size(); index += 2) {
            const auto& name = tokens[index];
            const auto& value = tokens[index + 1];
            if (name == "--camera") {
                camera_id = parse_non_negative(value);
                if (!camera_id) {
                    error = "camera ID must be a non-negative integer";
                    return std::nullopt;
                }
            } else if (name == "--device-link") {
                patch.device_symbolic_link = value;
            } else if (name == "--device-index") {
                patch.device_index = parse_non_negative(value);
                if (!patch.device_index) {
                    error = "device index must be a non-negative integer";
                    return std::nullopt;
                }
            } else if (name == "--width") {
                patch.width = parse_positive(value);
                if (!patch.width) {
                    error = "width must be a positive integer";
                    return std::nullopt;
                }
            } else if (name == "--height") {
                patch.height = parse_positive(value);
                if (!patch.height) {
                    error = "height must be a positive integer";
                    return std::nullopt;
                }
            } else if (name == "--fps") {
                patch.frame_rate = parse_frame_rate(value);
                if (!patch.frame_rate) {
                    error = "fps must be a positive integer or numerator/denominator";
                    return std::nullopt;
                }
            } else if (name == "--format") {
                patch.format = parse_pixel_format(value);
                if (!patch.format) {
                    error = "format must be mjpeg, yuy2, or bgra8";
                    return std::nullopt;
                }
            } else if (name == "--cuda-device") {
                patch.cuda_device = parse_non_negative_int(value);
                if (!patch.cuda_device) {
                    error = "CUDA device must be a non-negative integer";
                    return std::nullopt;
                }
            } else if (name == "--sample-queue") {
                patch.sample_queue_capacity = parse_positive(value);
                if (!patch.sample_queue_capacity) {
                    error = "sample queue capacity must be a positive integer";
                    return std::nullopt;
                }
            } else if (name == "--frame-pool") {
                patch.frame_pool_capacity = parse_positive(value);
                if (!patch.frame_pool_capacity) {
                    error = "frame pool capacity must be a positive integer";
                    return std::nullopt;
                }
            } else if (name == "--overflow") {
                patch.overflow = parse_overflow_policy(value);
                if (!patch.overflow) {
                    error = "overflow must be block, drop-oldest, or drop-newest";
                    return std::nullopt;
                }
            } else if (name == "--rotation") {
                patch.rotation = parse_rotation(value);
                if (!patch.rotation) {
                    error = "rotation must be none, cw90, 180, or ccw90";
                    return std::nullopt;
                }
            } else if (name == "--allow-fallback") {
                patch.allow_format_fallback = parse_bool(value);
                if (!patch.allow_format_fallback) {
                    error = "allow-fallback must be true or false";
                    return std::nullopt;
                }
            } else if (name == "--reconnect") {
                patch.reconnect = parse_bool(value);
                if (!patch.reconnect) {
                    error = "reconnect must be true or false";
                    return std::nullopt;
                }
            } else {
                error = "unknown capture option: " + name;
                return std::nullopt;
            }
        }
        return ConfigureCaptureCommand{std::move(patch), camera_id};
    }
    if ((tokens[0] == "quit" || tokens[0] == "exit") && tokens.size() == 1) {
        return ShutdownCommand{};
    }
    error = "unknown command or invalid arguments; type 'help'";
    return std::nullopt;
}

std::string InteractiveCli::help() {
    return "status\n"
           "metrics [prefix]\n"
           "pipeline start\n"
           "pipeline stop\n"
           "pose status\n"
           "pose off\n"
           "pose monocular [model-path]\n"
           "pose multiview <calibration.json>\n"
           "pose multiview <engine-path> <calibration.json>\n"
           "record start <file.mp4> [bitrate] [fps]\n"
           "record stop\n"
           "record status\n"
           "shm enable <name>\n"
           "shm disable\n"
           "preview status\n"
           "preview enable [port]\n"
           "preview disable\n"
           "capture list\n"
           "capture sync <tolerance-ms> <capacity> <drop|partial>\n"
           "capture add <camera-id> <device-index> [width height fps format]\n"
           "capture remove <camera-id>\n"
           "capture configure [--device-link <link> | --device-index <n>] [--width <n>] "
           "[--height <n>] [--fps <n|n/d>] [--format <mjpeg|yuy2|bgra8>] "
           "[--camera <id>] [--cuda-device <n>] [--sample-queue <n>] [--frame-pool <n>] "
           "[--overflow <block|drop-oldest|drop-newest>] "
           "[--rotation <none|cw90|180|ccw90>] [--allow-fallback <true|false>] "
           "[--reconnect <true|false>]\n"
           "quit\n";
}

} // namespace iris
