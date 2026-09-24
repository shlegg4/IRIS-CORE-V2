#include "iris/api/RestApiServer.hpp"
#include "iris/cli/InteractiveCli.hpp"
#include "iris/runtime/Runtime.hpp"

#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <string_view>
#include <optional>
#include <filesystem>
#include <cstdint>
#include <vector>
#include <utility>
#include <fstream>
#include <nlohmann/json.hpp>

namespace {
std::optional<iris::FrameRotation> parse_video_rotation(std::string_view value) {
    if (value == "none") return iris::FrameRotation::None;
    if (value == "cw90") return iris::FrameRotation::Clockwise90;
    if (value == "180") return iris::FrameRotation::Rotate180;
    if (value == "ccw90") return iris::FrameRotation::CounterClockwise90;
    return std::nullopt;
}

std::optional<bool> parse_video_bool(std::string_view value) {
    if (value == "true") return true;
    if (value == "false") return false;
    return std::nullopt;
}

std::optional<iris::SynchronizedVideoConfig> video_config_from_args(int argc, char** argv,
                                                                   int first,
                                                                   std::string& error,
                                                                   std::filesystem::path& engine,
                                                                   std::filesystem::path& calibration,
                                                                   std::filesystem::path& output_dir) {
    if (first >= argc) return std::nullopt;
    iris::SynchronizedVideoConfig config;
    std::vector<std::pair<iris::CameraId, iris::FrameRotation>> rotations;
    for (int index = first; index < argc;) {
        if (std::string_view(argv[index]) == "--loop") {
            if (index + 1 >= argc) {
                error = "--loop requires true or false";
                return std::nullopt;
            }
            const auto loop = parse_video_bool(argv[index + 1]);
            if (!loop) {
                error = "--loop requires true or false";
                return std::nullopt;
            }
            config.loop = *loop;
            index += 2;
            continue;
        }
        if (std::string_view(argv[index]) == "--realtime") {
            if (index + 1 >= argc) { error = "--realtime requires true or false"; return std::nullopt; }
            const auto realtime = parse_video_bool(argv[index + 1]);
            if (!realtime) { error = "--realtime requires true or false"; return std::nullopt; }
            config.realtime = *realtime;
            index += 2;
            continue;
        }
        if (std::string_view(argv[index]) == "--cuda-device") {
            if (index + 1 >= argc) { error = "--cuda-device requires an integer"; return std::nullopt; }
            try {
                std::size_t used{};
                const auto value = std::stoi(argv[index + 1], &used);
                if (used != std::string_view(argv[index + 1]).size() || value < 0)
                    throw std::invalid_argument("invalid CUDA device");
                config.cuda_device = value;
            } catch (...) { error = "--cuda-device requires a non-negative integer"; return std::nullopt; }
            index += 2;
            continue;
        }
        if (std::string_view(argv[index]) == "--metrics-port") {
            if (index + 1 >= argc) { error = "--metrics-port requires an integer"; return std::nullopt; }
            index += 2;
            continue;
        }
        if (std::string_view(argv[index]) == "--engine" ||
            std::string_view(argv[index]) == "--calibration" ||
            std::string_view(argv[index]) == "--output-dir") {
            if (index + 1 >= argc) { error = std::string(argv[index]) + " requires a path"; return std::nullopt; }
            const auto option = std::string_view(argv[index]);
            if (option == "--engine") engine = argv[index + 1];
            else if (option == "--calibration") calibration = argv[index + 1];
            else output_dir = argv[index + 1];
            index += 2;
            continue;
        }
        if (std::string_view(argv[index]) == "--rotation") {
            if (index + 2 >= argc) {
                error = "--rotation requires a camera ID and one of none, cw90, 180, or ccw90";
                return std::nullopt;
            }
            std::size_t used{};
            try {
                const auto parsed = std::stoull(argv[index + 1], &used);
                const auto rotation = parse_video_rotation(argv[index + 2]);
                if (used != std::string_view(argv[index + 1]).size() ||
                    parsed > UINT32_MAX || !rotation) {
                    error = "invalid --rotation camera ID or rotation";
                    return std::nullopt;
                }
                rotations.emplace_back(static_cast<iris::CameraId>(parsed), *rotation);
            } catch (...) {
                error = "invalid --rotation camera ID or rotation";
                return std::nullopt;
            }
            index += 3;
            continue;
        }
        if (index + 1 >= argc) {
            error = "video startup arguments must be <camera-id> <file> pairs, with optional --loop <true|false> and --rotation <camera-id> <none|cw90|180|ccw90>";
            return std::nullopt;
        }
        try {
            std::size_t used{};
            const auto parsed = std::stoull(argv[index], &used);
            if (used != std::string_view(argv[index]).size() || parsed > UINT32_MAX) {
                error = "video camera ID must be a non-negative 32-bit integer";
                return std::nullopt;
            }
            config.cameras.push_back(
                {static_cast<iris::CameraId>(parsed), std::filesystem::path(argv[index + 1])});
            index += 2;
        } catch (...) {
            error = "video camera ID must be a non-negative 32-bit integer";
            return std::nullopt;
        }
    }
    for (const auto& [camera_id, rotation] : rotations) {
        const auto camera = std::find_if(config.cameras.begin(), config.cameras.end(),
                                         [camera_id](const auto& input) {
                                             return input.camera_id == camera_id;
                                         });
        if (camera == config.cameras.end()) {
            error = "--rotation references an unknown camera ID";
            return std::nullopt;
        }
        camera->rotation = rotation;
    }
    if (config.cameras.empty()) {
        error = "at least one video camera/file pair is required";
        return std::nullopt;
    }
    return config;
}
} // namespace

int run_main(int argc, char** argv) {
    std::uint16_t metrics_port = 9464;
    for (int index = 1; index < argc; ++index) {
        if (std::string_view(argv[index]) != "--metrics-port") continue;
        if (index + 1 >= argc) {
            std::cerr << "--metrics-port requires an integer from 1 through 65535\n";
            return 2;
        }
        try {
            std::size_t used{};
            const auto parsed = std::stoul(argv[index + 1], &used);
            if (used != std::string_view(argv[index + 1]).size() || parsed == 0 || parsed > 65535)
                throw std::invalid_argument("invalid metrics port");
            metrics_port = static_cast<std::uint16_t>(parsed);
        } catch (...) {
            std::cerr << "--metrics-port requires an integer from 1 through 65535\n";
            return 2;
        }
        ++index;
    }
    iris::Runtime runtime(iris::CaptureConfig{}, metrics_port);
    const bool api_mode = argc >= 2 && std::string_view(argv[1]) == "--api";
    const bool non_interactive = argc >= 2 && std::string_view(argv[1]) == "--non-interactive";
    const int video_args = api_mode && argc >= 3 && std::string_view(argv[2]) == "--video"
                               ? 3
                               : ((!api_mode && !non_interactive && argc >= 2 &&
                                   std::string_view(argv[1]) == "--video")
                                      ? 2
                                      : (non_interactive && argc >= 3 &&
                                                 std::string_view(argv[2]) == "--video"
                                             ? 3
                                             : 0));
    if (video_args != 0) {
        std::string error;
        std::filesystem::path engine, calibration, output_dir;
        auto config = video_config_from_args(argc, argv, video_args, error, engine, calibration, output_dir);
        if (!config) {
            std::cerr << "invalid video source: "
                      << (error.empty() ? "at least one camera/file pair is required" : error)
                      << '\n';
            return 2;
        }
        const bool batch_run = !output_dir.empty() || !engine.empty() || !calibration.empty();
        if (batch_run) {
            if (output_dir.empty() || engine.empty() || calibration.empty()) {
                std::cerr << "batch video mode requires --output-dir, --engine, and --calibration\n";
                return 2;
            }
            if (api_mode || non_interactive) {
                std::cerr << "batch video mode cannot be combined with --api or --non-interactive\n";
                return 2;
            }
            std::error_code ec;
            std::filesystem::create_directories(output_dir, ec);
            if (ec) { std::cerr << "could not create output directory: " << ec.message() << '\n'; return 1; }
            config->pose_output_path = output_dir / "poses.jsonl";
            runtime.start();
            iris::ConfigurePoseCommand pose;
            pose.backend = iris::ConfigurePoseCommand::Backend::Multiview;
            pose.engine_path = engine;
            pose.calibration_path = calibration;
            auto pose_configured = runtime.execute(pose);
            if (!pose_configured) { std::cerr << "could not configure pose: " << pose_configured.message << '\n'; runtime.stop(); return 1; }
            auto video_configured = runtime.execute(iris::ConfigureVideoIngestionCommand{*config});
            if (!video_configured) { std::cerr << "could not configure video source: " << video_configured.message << '\n'; runtime.stop(); return 1; }
            const auto started_at = std::chrono::steady_clock::now();
            const int result = runtime.run();
            const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at).count();
            const auto final = runtime.snapshot();
            nlohmann::json metrics;
            for (const auto& [name, value] : final.metrics.counters) metrics["counters"][name] = value;
            for (const auto& [name, value] : final.metrics.gauges) metrics["gauges"][name] = value;
            for (const auto& [name, value] : final.metrics.histograms) {
                metrics["histograms"][name] = {{"count", value.count}, {"sum", value.sum},
                                                {"bounds", value.bounds}, {"counts", value.counts}};
            }
            const auto batches = final.processed_packets;
            const auto summary = nlohmann::json{
                {"schema_version", 1}, {"status", result == 0 ? "completed" : "failed"},
                {"error", final.last_error}, {"pose_engine", engine.string()},
                {"calibration", calibration.string()}, {"output_dir", output_dir.string()},
                {"realtime", config->realtime}, {"loop", config->loop},
                {"camera_ids", [&] { nlohmann::json ids = nlohmann::json::array(); for (const auto& c : config->cameras) ids.push_back(c.camera_id); return ids; }()},
                {"input_videos", [&] { nlohmann::json inputs = nlohmann::json::array(); for (const auto& c : config->cameras) inputs.push_back(c.path.string()); return inputs; }()},
                {"processed_batches", batches}, {"processed_camera_frames", batches * config->cameras.size()},
                {"elapsed_seconds", elapsed}, {"batches_per_second", elapsed > 0 ? batches / elapsed : 0.0},
                {"metrics", std::move(metrics)}};
            std::ofstream summary_file(output_dir / "run_summary.json", std::ios::trunc);
            if (!summary_file) { std::cerr << "could not write run_summary.json\n"; return 1; }
            summary_file << summary.dump(2) << '\n';
            std::cout << "processed " << batches << " batches in " << elapsed << " s ("
                      << (elapsed > 0 ? batches / elapsed : 0.0) << " batches/s)\n";
            return result;
        }
        runtime.start();
        const auto configured = runtime.execute(iris::ConfigureVideoIngestionCommand{*config});
        if (!configured) {
            std::cerr << "could not configure video source: " << configured.message << '\n';
            runtime.stop();
            return 1;
        }
    }
    if (api_mode) {
        runtime.start();
        iris::api::RestApiServer api(runtime);
        api.start();
        iris::PreviewConfig preview;
        preview.http.enabled = true;
        preview.http.bind_address = "127.0.0.1";
        preview.http.port = 8080;
        auto preview_configured = runtime.execute(iris::ConfigurePreviewCommand{preview});
        if (!preview_configured) {
            std::cerr << "IRIS preview configuration failed: " << preview_configured.message << '\n';
            api.stop(); runtime.stop(); return 1;
        }
        const auto started = runtime.execute(iris::StartPipelineCommand{});
        if (!started) {
            std::cerr << "IRIS pipeline start failed: " << started.message << '\n';
            api.stop(); runtime.stop(); return 1;
        }
        bool failed = false;
        while (true) {
            const auto status = runtime.snapshot();
            if (status.state == iris::RuntimeState::Failed) {
                std::cerr << "IRIS runtime failed: " << status.last_error << '\n';
                failed = true;
                break;
            }
            if (status.state == iris::RuntimeState::Shutdown) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        api.stop();
        runtime.stop();
        return failed ? 1 : 0;
    }
    if (non_interactive) {
        return runtime.run();
    }

    if (!video_args) runtime.start();
    const auto started = runtime.execute(iris::StartPipelineCommand{});
    std::cout << (started ? "ok: " : "error: ") << started.message << '\n';
    iris::InteractiveCli cli(runtime);
    const int result = cli.run(std::cin, std::cout);
    runtime.stop();
    return result;
}

int main(int argc, char** argv) {
    try {
        return run_main(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "IRIS fatal error: " << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "IRIS fatal error: unknown exception\n";
        return 1;
    }
}
