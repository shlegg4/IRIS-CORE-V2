#pragma once

#include "iris/infrastructure/metrics/MetricRegistry.hpp"

#include <string>
#include <string_view>

namespace iris::infrastructure::metrics {

struct ChannelMetrics {
    Counter sent;
    Counter received;
    Counter dropped;
    Counter dropped_oldest;
    Counter dropped_newest;
    Counter rejected_closed;
    Gauge depth;
    Gauge peak_depth;
};

inline ChannelMetrics register_channel_metrics(MetricRegistry& registry, std::string_view prefix) {
    const std::string name{prefix};
    return {
        registry.counter(name + "_sent_total"),
        registry.counter(name + "_received_total"),
        registry.counter(name + "_dropped_total"),
        registry.counter(name + "_dropped_oldest_total"),
        registry.counter(name + "_dropped_newest_total"),
        registry.counter(name + "_rejected_closed_total"),
        registry.gauge(name + "_depth"),
        registry.gauge(name + "_peak_depth"),
    };
}

} // namespace iris::infrastructure::metrics
