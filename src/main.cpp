#include "iris/cli/InteractiveCli.hpp"
#include "iris/runtime/Runtime.hpp"

#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    iris::Runtime runtime;
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
