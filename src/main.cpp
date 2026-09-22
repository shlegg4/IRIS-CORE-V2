#include "iris/api/RestApiServer.hpp"
#include "iris/cli/InteractiveCli.hpp"
#include "iris/runtime/Runtime.hpp"

#include <iostream>
#include <chrono>
#include <thread>
#include <string_view>

int main(int argc, char** argv) {
    iris::Runtime runtime;
    if (argc == 2 && std::string_view(argv[1]) == "--api") {
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
    if (argc == 2 && std::string_view(argv[1]) == "--non-interactive") {
        return runtime.run();
    }

    runtime.start();
    const auto started = runtime.execute(iris::StartPipelineCommand{});
    std::cout << (started ? "ok: " : "error: ") << started.message << '\n';
    iris::InteractiveCli cli(runtime);
    const int result = cli.run(std::cin, std::cout);
    runtime.stop();
    return result;
}
