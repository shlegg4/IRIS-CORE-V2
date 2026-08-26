#include "iris/infrastructure/metrics/MetricRegistry.hpp"
#include "iris/pipeline/Channel.hpp"
#include "iris/stages/capture/CaptureStage.hpp"
#include <cstdlib>
#include <iostream>
int main(int argc, char** argv) {
    try {
        std::size_t count =
            argc > 1 ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10)) : 120;
        iris::CaptureConfig config;
        iris::Channel<iris::Packet> frames(2, iris::OverflowPolicy::DropOldest);
        iris::infrastructure::metrics::MetricRegistry metrics;
        iris::CaptureStage stage(config, frames, metrics);
        stage.start();
        bool source_ended = false;
        for (std::size_t i = 0; i < count; ++i) {
            auto packet = frames.receive();
            if (!packet) {
                source_ended = true;
                break;
            }
            if (i % 30 == 0) {
                std::cout << "frame=" << packet->sequence << " gpu_frames=" << packet->frames.size()
                          << '\n';
            }
        }
        if (source_ended) {
            stage.wait();
        } else {
            stage.stop();
        }
        auto snapshot = metrics.snapshot();
        for (auto& [name, value] : snapshot.counters) {
            std::cout << name << '=' << value << '\n';
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "capture probe failed: " << e.what() << '\n';
        return 1;
    }
}
