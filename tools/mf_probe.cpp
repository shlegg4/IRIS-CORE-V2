#include "stages/capture/media_foundation/MediaFoundationSource.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::string value(int& index, int argc, char** argv) {
    if (++index >= argc) {
        throw std::invalid_argument("missing command-line value");
    }
    return argv[index];
}

} // namespace

int main(int argc, char** argv) {
    try {
        iris::CaptureConfig config;
        std::size_t frame_count{5};

        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--device-index") {
                config.device_index =
                    static_cast<std::uint32_t>(std::stoul(value(index, argc, argv)));
            } else if (argument == "--width") {
                config.extent.width =
                    static_cast<std::uint32_t>(std::stoul(value(index, argc, argv)));
            } else if (argument == "--height") {
                config.extent.height =
                    static_cast<std::uint32_t>(std::stoul(value(index, argc, argv)));
            } else if (argument == "--fps") {
                config.frame_rate.numerator =
                    static_cast<std::uint32_t>(std::stoul(value(index, argc, argv)));
            } else if (argument == "--frames") {
                frame_count = static_cast<std::size_t>(std::stoull(value(index, argc, argv)));
            } else {
                throw std::invalid_argument("unknown argument: " + argument);
            }
        }

        iris::capture::MediaFoundationSource source;
        const auto format = source.open(config);
        std::cout << "negotiated=" << format.extent.width << 'x' << format.extent.height << '@'
                  << format.frame_rate.value() << '\n';

        for (std::size_t received = 0; received < frame_count;) {
            auto sample = source.read();
            if (sample) {
                std::cout << "sample=" << sample->sequence << " bytes=" << sample->bytes.size()
                          << '\n';
                ++received;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Media Foundation probe failed: " << error.what() << '\n';
        return 1;
    }
}
