#include "iris/api/RestApiServer.hpp"
#include "iris/cli/InteractiveCli.hpp"
#include "iris/runtime/Runtime.hpp"

#include <iostream>
#include <chrono>
#include <thread>
#include <string_view>
#include <optional>
#include <filesystem>
#include <cstdint>

namespace {
std::optional<iris::SynchronizedVideoConfig> video_config_from_args(int argc, char** argv,
                                                                   int first,
                                                                   std::string& error) {
    if (first >= argc) return std::nullopt;
    if ((argc - first) % 2 != 0) {
        error = "video startup arguments must be <camera-id> <file> pairs";
        return std::nullopt;
    }
    iris::SynchronizedVideoConfig config;
    for (int index = first; index < argc; index += 2) {
        try {
            std::size_t used{};
            const auto parsed = std::stoull(argv[index], &used);
            if (used != std::string_view(argv[index]).size() || parsed > UINT32_MAX) {
                error = "video camera ID must be a non-negative 32-bit integer";
                return std::nullopt;
            }
            config.cameras.push_back(
                {static_cast<iris::CameraId>(parsed), std::filesystem::path(argv[index + 1])});
        } catch (...) {
            error = "video camera ID must be a non-negative 32-bit integer";
            return std::nullopt;
        }
    }
    return config;
}
} // namespace

int main(int argc, char** argv) {
    iris::Runtime runtime;
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
        auto config = video_config_from_args(argc, argv, video_args, error);
        if (!config) {
            std::cerr << "invalid video source: "
                      << (error.empty() ? "at least one camera/file pair is required" : error)
                      << '\n';
            return 2;
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
        if (!preview_configured) { api.stop(); runtime.stop(); return 1; }
        const auto started = runtime.execute(iris::StartPipelineCommand{});
        if (!started) { runtime.stop(); return 1; }
        while (true) {
            const auto state = runtime.snapshot().state;
            if (state == iris::RuntimeState::Failed || state == iris::RuntimeState::Shutdown) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        api.stop();
        runtime.stop();
        return 0;
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
