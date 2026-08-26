#include "iris/infrastructure/gpu/CudaResources.hpp"
#include "iris/infrastructure/metrics/MetricRegistry.hpp"
#include "iris/pipeline/Channel.hpp"
#include "iris/stages/capture/CaptureStage.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Options {
    std::uint32_t device_index{0};
    std::uint32_t width{1920};
    std::uint32_t height{1080};
    std::uint32_t fps{30};
    std::size_t frames{120};
    double duration_seconds{};
    double max_drop_rate{0.05};
    double max_decode_failure_rate{0.0};
    double max_p95_latency_ms{75.0};
    double max_clock_drift_ppm{250.0};
    std::size_t min_drift_samples{120};
    std::uint32_t consumer_delay_ms{};
    std::filesystem::path output{"capture-validation.json"};
};

std::string require_value(int& index, int argc, char** argv) {
    if (++index >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + argv[index - 1]);
    }
    return argv[index];
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--device-index") {
            options.device_index =
                static_cast<std::uint32_t>(std::stoul(require_value(index, argc, argv)));
        } else if (argument == "--width") {
            options.width =
                static_cast<std::uint32_t>(std::stoul(require_value(index, argc, argv)));
        } else if (argument == "--height") {
            options.height =
                static_cast<std::uint32_t>(std::stoul(require_value(index, argc, argv)));
        } else if (argument == "--fps") {
            options.fps = static_cast<std::uint32_t>(std::stoul(require_value(index, argc, argv)));
        } else if (argument == "--frames") {
            options.frames =
                static_cast<std::size_t>(std::stoull(require_value(index, argc, argv)));
        } else if (argument == "--duration") {
            options.duration_seconds = std::stod(require_value(index, argc, argv));
            options.frames = 0;
        } else if (argument == "--max-drop-rate") {
            options.max_drop_rate = std::stod(require_value(index, argc, argv));
        } else if (argument == "--max-decode-failure-rate") {
            options.max_decode_failure_rate = std::stod(require_value(index, argc, argv));
        } else if (argument == "--max-p95-latency-ms") {
            options.max_p95_latency_ms = std::stod(require_value(index, argc, argv));
        } else if (argument == "--max-clock-drift-ppm") {
            options.max_clock_drift_ppm = std::stod(require_value(index, argc, argv));
        } else if (argument == "--min-drift-samples") {
            options.min_drift_samples =
                static_cast<std::size_t>(std::stoull(require_value(index, argc, argv)));
        } else if (argument == "--consumer-delay-ms") {
            options.consumer_delay_ms =
                static_cast<std::uint32_t>(std::stoul(require_value(index, argc, argv)));
        } else if (argument == "--output") {
            options.output = require_value(index, argc, argv);
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    return options;
}

double percentile(std::vector<double> values, double quantile) {
    if (values.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    std::sort(values.begin(), values.end());
    const auto index =
        static_cast<std::size_t>(std::ceil(quantile * static_cast<double>(values.size())) - 1.0);
    return values[(std::min)(index, values.size() - 1)];
}

std::uint64_t counter(const iris::infrastructure::metrics::MetricsSnapshot& snapshot,
                      const std::string& name) {
    const auto found = snapshot.counters.find(name);
    return found == snapshot.counters.end() ? 0 : found->second;
}

double gauge(const iris::infrastructure::metrics::MetricsSnapshot& snapshot,
             const std::string& name) {
    const auto found = snapshot.gauges.find(name);
    return found == snapshot.gauges.end() ? 0.0 : found->second;
}

} // namespace

int main(int argc, char** argv) {
    using namespace std::chrono;

    Options options;
    std::optional<iris::CaptureStage> stage;
    try {
        options = parse_options(argc, argv);
        iris::CaptureConfig config;
        config.device_index = options.device_index;
        config.extent = {options.width, options.height};
        config.frame_rate = {options.fps, 1};

        iris::Channel<iris::Packet> packets{2, iris::OverflowPolicy::DropOldest};
        iris::infrastructure::metrics::MetricRegistry metrics;
        stage.emplace(config, packets, metrics);
        stage->start();

        const auto started = steady_clock::now();
        const auto deadline = started + duration<double>(options.duration_seconds);
        std::vector<double> latencies_ms;
        std::size_t frames_observed{};
        bool source_ended{};

        while ((options.frames == 0 || frames_observed < options.frames) &&
               (options.duration_seconds <= 0.0 || steady_clock::now() < deadline)) {
            auto packet = packets.receive();
            if (!packet) {
                source_ended = true;
                break;
            }
            for (const auto& frame : packet->frames) {
                if (frame.ready) {
                    frame.ready->synchronize();
                }
                const auto latency = duration<double, std::milli>(
                    steady_clock::now() - frame.timing.estimated_capture_time);
                latencies_ms.push_back(latency.count());
                ++frames_observed;
            }
            if (options.consumer_delay_ms > 0) {
                std::this_thread::sleep_for(milliseconds(options.consumer_delay_ms));
            }
        }

        if (source_ended) {
            stage->wait();
        } else {
            stage->stop();
        }

        const auto snapshot = metrics.snapshot();
        const auto received = counter(snapshot, "iris_capture_samples_received_total");
        const auto emitted = counter(snapshot, "iris_capture_frames_emitted_total");
        const auto dropped = counter(snapshot, "iris_channel_capture_samples_dropped_total");
        const auto decode_failures = counter(snapshot, "iris_capture_decode_failures_total");
        const auto source_errors = counter(snapshot, "iris_capture_source_errors_total");
        const double drop_rate = received == 0 ? 1.0 : static_cast<double>(dropped) / received;
        const double decode_failure_rate =
            received == 0 ? 1.0 : static_cast<double>(decode_failures) / received;
        const double p50_latency = percentile(latencies_ms, 0.50);
        const double p95_latency = percentile(latencies_ms, 0.95);
        const double p99_latency = percentile(latencies_ms, 0.99);
        const double drift_ppm = std::abs(gauge(snapshot, "iris_capture_clock_drift_ppm"));
        const bool drift_evaluated = frames_observed >= options.min_drift_samples;

        std::vector<std::string> failures;
        if (frames_observed == 0) {
            failures.emplace_back("no frames observed");
        }
        if (source_errors > 0) {
            failures.emplace_back("source errors=" + std::to_string(source_errors));
        }
        if (decode_failure_rate > options.max_decode_failure_rate) {
            failures.emplace_back("decode failure rate=" + std::to_string(decode_failure_rate) +
                                  " exceeds " + std::to_string(options.max_decode_failure_rate));
        }
        if (drop_rate > options.max_drop_rate) {
            failures.emplace_back("drop rate=" + std::to_string(drop_rate) + " exceeds " +
                                  std::to_string(options.max_drop_rate));
        }
        if (p95_latency > options.max_p95_latency_ms) {
            failures.emplace_back("p95 latency=" + std::to_string(p95_latency) + "ms exceeds " +
                                  std::to_string(options.max_p95_latency_ms) + "ms");
        }
        if (drift_evaluated && drift_ppm > options.max_clock_drift_ppm) {
            failures.emplace_back("clock drift=" + std::to_string(drift_ppm) + "ppm exceeds " +
                                  std::to_string(options.max_clock_drift_ppm) + "ppm");
        }
        const bool passed = failures.empty();

        if (!options.output.parent_path().empty()) {
            std::filesystem::create_directories(options.output.parent_path());
        }
        std::ofstream report(options.output, std::ios::trunc);
        report << "{\n"
               << "    \"passed\": " << (passed ? "true" : "false") << ",\n"
               << "    \"device_index\": " << options.device_index << ",\n"
               << "    \"requested_format\": \"" << options.width << 'x' << options.height << '@'
               << options.fps << " MJPEG\",\n"
               << "    \"frames_observed\": " << frames_observed << ",\n"
               << "    \"samples_received\": " << received << ",\n"
               << "    \"frames_emitted\": " << emitted << ",\n"
               << "    \"samples_dropped\": " << dropped << ",\n"
               << "    \"decode_failures\": " << decode_failures << ",\n"
               << "    \"decode_failure_rate\": " << decode_failure_rate << ",\n"
               << "    \"source_errors\": " << source_errors << ",\n"
               << "    \"drop_rate\": " << drop_rate << ",\n"
               << "    \"latency_ms\": {\"p50\": " << p50_latency << ", \"p95\": " << p95_latency
               << ", \"p99\": " << p99_latency << "},\n"
               << "    \"clock_drift_ppm\": " << drift_ppm << ",\n"
               << "    \"drift_evaluated\": " << (drift_evaluated ? "true" : "false") << ",\n"
               << "    \"min_drift_samples\": " << options.min_drift_samples << ",\n"
               << "    \"failures\": [";
        for (std::size_t index = 0; index < failures.size(); ++index) {
            report << (index == 0 ? "" : ", ") << '"' << failures[index] << '"';
        }
        report << "],\n    \"metrics\": {\n        \"counters\": {";
        bool first = true;
        for (const auto& [name, value] : snapshot.counters) {
            report << (first ? "" : ",") << "\n            \"" << name << "\": " << value;
            first = false;
        }
        report << "\n        },\n        \"gauges\": {";
        first = true;
        for (const auto& [name, value] : snapshot.gauges) {
            report << (first ? "" : ",") << "\n            \"" << name << "\": " << value;
            first = false;
        }
        report << "\n        },\n        \"histograms\": {";
        first = true;
        for (const auto& [name, histogram] : snapshot.histograms) {
            report << (first ? "" : ",") << "\n            \"" << name
                   << "\": {\"count\": " << histogram.count << ", \"sum\": " << histogram.sum
                   << ", \"bounds\": [";
            for (std::size_t index = 0; index < histogram.bounds.size(); ++index) {
                report << (index == 0 ? "" : ", ") << histogram.bounds[index];
            }
            report << "], \"counts\": [";
            for (std::size_t index = 0; index < histogram.counts.size(); ++index) {
                report << (index == 0 ? "" : ", ") << histogram.counts[index];
            }
            report << "]}";
            first = false;
        }
        report << "\n        }\n    }\n}\n";

        std::cout << "capture validation " << (passed ? "PASSED" : "FAILED")
                  << ": frames=" << frames_observed << ", drops=" << dropped
                  << ", p95_latency_ms=" << p95_latency << ", drift_ppm=" << drift_ppm
                  << (drift_evaluated ? " (evaluated)" : " (warm-up only)")
                  << ", report=" << options.output.string() << '\n';
        std::cout << "capture counters:\n";
        for (const auto& [name, value] : snapshot.counters) {
            if (name.starts_with("iris_capture_") || name.starts_with("iris_channel_")) {
                std::cout << "  " << name << '=' << value << '\n';
            }
        }
        std::cout << "capture gauges:\n";
        for (const auto& [name, value] : snapshot.gauges) {
            if (name.starts_with("iris_capture_") || name.starts_with("iris_channel_")) {
                std::cout << "  " << name << '=' << value << '\n';
            }
        }
        for (const auto& failure : failures) {
            std::cout << "  - " << failure << '\n';
        }
        return passed ? 0 : 2;
    } catch (const std::exception& error) {
        if (stage) {
            stage->stop();
        }
        if (!options.output.parent_path().empty()) {
            std::filesystem::create_directories(options.output.parent_path());
        }
        std::ofstream(options.output, std::ios::trunc)
            << "{\n    \"passed\": false,\n    \"error\": \"" << error.what()
            << "\",\n    \"metrics\": {}\n}\n";
        std::cerr << "capture validation ERROR: " << error.what() << '\n';
        return 1;
    }
}
