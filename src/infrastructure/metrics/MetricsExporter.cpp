#include "iris/infrastructure/metrics/MetricsExporter.hpp"
#include <fstream>
#include <sstream>
namespace iris::infrastructure::metrics {
MetricsExporter::MetricsExporter(const MetricRegistry& r, std::filesystem::path p,
                                 std::chrono::milliseconds i)
    : registry_(r), path_(std::move(p)), interval_(i) {}
MetricsExporter::~MetricsExporter() { stop(); }
void MetricsExporter::start() {
    if (!worker_.joinable()) {
        worker_ = std::jthread([this](std::stop_token s) { run(s); });
    }
}
void MetricsExporter::stop() {
    if (worker_.joinable()) {
        worker_.request_stop();
        worker_.join();
    }
}
void MetricsExporter::run(std::stop_token stop) {
    while (!stop.stop_requested()) {
        auto s = registry_.snapshot();
        std::ostringstream o;
        o << "{\n  \"counters\": {";
        bool first = true;
        for (auto& [n, v] : s.counters) {
            if (!first) {
                o << ',';
            }
            o << "\n    \"" << n << "\": " << v;
            first = false;
        }
        o << "\n  },\n  \"gauges\": {";
        first = true;
        for (auto& [n, v] : s.gauges) {
            if (!first) {
                o << ',';
            }
            o << "\n    \"" << n << "\": " << v;
            first = false;
        }
        o << "\n  },\n  \"histograms\": {";
        first = true;
        for (auto& [n, v] : s.histograms) {
            if (!first) {
                o << ',';
            }
            o << "\n    \"" << n << "\": {\"count\": " << v.count << ", \"sum\": " << v.sum << '}';
            first = false;
        }
        o << "\n  }\n}\n";
        std::ofstream(path_, std::ios::trunc) << o.str();
        for (auto waited = std::chrono::milliseconds(0);
             waited < interval_ && !stop.stop_requested();
             waited += std::chrono::milliseconds(50)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
}
} // namespace iris::infrastructure::metrics
