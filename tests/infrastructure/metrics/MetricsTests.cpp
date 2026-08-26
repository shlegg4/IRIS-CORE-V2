#include "iris/infrastructure/metrics/MetricRegistry.hpp"
#include "iris/infrastructure/metrics/PrometheusExporter.hpp"
#include <cassert>
int main() {
    iris::infrastructure::metrics::MetricRegistry r;
    auto c = r.counter("frames");
    auto g = r.gauge("depth");
    auto h = r.histogram("latency", {1, 5, 10});
    c.increment(3);
    g.set(2);
    h.observe(4);
    h.observe(20);
    auto s = r.snapshot();
    assert(s.counters.at("frames") == 3);
    assert(s.gauges.at("depth") == 2);
    assert(s.histograms.at("latency").count == 2);
    const auto prometheus = iris::infrastructure::metrics::format_prometheus(s);
    assert(prometheus.find("# TYPE frames counter\nframes 3\n") != std::string::npos);
    assert(prometheus.find("latency_bucket{le=\"1\"} 0\n") != std::string::npos);
    assert(prometheus.find("latency_bucket{le=\"5\"} 1\n") != std::string::npos);
    assert(prometheus.find("latency_bucket{le=\"+Inf\"} 2\n") != std::string::npos);
    assert(prometheus.find("latency_count 2\n") != std::string::npos);
}
