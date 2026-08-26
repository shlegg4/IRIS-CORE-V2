#include "iris/infrastructure/metrics/MetricRegistry.hpp"
#include "iris/pipeline/Channel.hpp"

#include <cassert>

int main() {
    iris::Channel<int> channel{2};
    assert(channel.send(42) == iris::SendResult::Sent);
    assert(channel.receive() == 42);
    channel.close();
    assert(!channel.receive().has_value());

    iris::Channel<int> latest{1, iris::OverflowPolicy::DropOldest};
    assert(latest.send(1) == iris::SendResult::Sent);
    assert(latest.send(2) == iris::SendResult::ReplacedOldest);
    assert(latest.receive() == 2);
    assert(latest.stats().dropped == 1);

    iris::infrastructure::metrics::MetricRegistry registry;
    auto channel_metrics =
        iris::infrastructure::metrics::register_channel_metrics(registry, "test_channel");
    iris::Channel<int> instrumented{1, iris::OverflowPolicy::DropNewest, channel_metrics};
    assert(instrumented.send(7) == iris::SendResult::Sent);
    assert(instrumented.send(8) == iris::SendResult::DroppedNewest);
    assert(instrumented.receive() == 7);
    instrumented.close();
    assert(instrumented.send(9) == iris::SendResult::Closed);

    const auto metrics = registry.snapshot();
    assert(metrics.counters.at("test_channel_sent_total") == 1);
    assert(metrics.counters.at("test_channel_received_total") == 1);
    assert(metrics.counters.at("test_channel_dropped_total") == 1);
    assert(metrics.counters.at("test_channel_dropped_newest_total") == 1);
    assert(metrics.counters.at("test_channel_dropped_oldest_total") == 0);
    assert(metrics.counters.at("test_channel_rejected_closed_total") == 1);
    assert(metrics.gauges.at("test_channel_depth") == 0);
    assert(metrics.gauges.at("test_channel_peak_depth") == 1);
}
