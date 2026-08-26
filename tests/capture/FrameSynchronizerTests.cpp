#include "iris/stages/capture/FrameSynchronizerStage.hpp"

#include <cassert>
#include <chrono>

namespace {
iris::Packet packet(iris::CameraId camera, std::uint64_t sequence, iris::MonotonicTime time) {
    iris::Frame frame;
    frame.camera = camera;
    frame.sequence = sequence;
    frame.timing.estimated_capture_time = time;
    return {sequence, iris::FrameBatch{std::move(frame)}, std::nullopt};
}

iris::MultiCameraCaptureConfig config(iris::IncompleteBatchPolicy policy) {
    iris::MultiCameraCaptureConfig result;
    result.cameras = {{0, {}}, {1, {}}};
    result.sync_tolerance = std::chrono::milliseconds(3);
    result.sync_queue_capacity = 4;
    result.incomplete_batch_policy = policy;
    return result;
}
} // namespace

int main() {
    {
        iris::infrastructure::metrics::MetricRegistry metrics;
        iris::Channel<iris::Packet> first(4, iris::OverflowPolicy::DropOldest);
        iris::Channel<iris::Packet> second(4, iris::OverflowPolicy::DropOldest);
        iris::Channel<iris::Packet> output(4, iris::OverflowPolicy::DropOldest);
        auto settings = config(iris::IncompleteBatchPolicy::DropBatch);
        iris::FrameSynchronizerStage synchronizer({&first, &second}, output, settings, metrics);
        synchronizer.start();
        const auto base = std::chrono::steady_clock::now();
        first.send(packet(0, 10, base));
        second.send(packet(1, 20, base + std::chrono::milliseconds(2)));
        first.close();
        second.close();
        synchronizer.wait();
        auto matched = output.receive();
        assert(matched);
        assert(matched->frames.size() == 2);
        assert(matched->frames[0].camera == 0);
        assert(matched->frames[1].camera == 1);
        assert(!output.receive());
        const auto snapshot = metrics.snapshot();
        assert(snapshot.counters.at("iris_sync_batches_emitted_total") == 1);
        assert(snapshot.histograms.at("iris_sync_match_skew_ms").count == 1);
    }
    {
        iris::infrastructure::metrics::MetricRegistry metrics;
        iris::Channel<iris::Packet> first(4, iris::OverflowPolicy::DropOldest);
        iris::Channel<iris::Packet> second(4, iris::OverflowPolicy::DropOldest);
        iris::Channel<iris::Packet> output(4, iris::OverflowPolicy::DropOldest);
        auto settings = config(iris::IncompleteBatchPolicy::EmitPartial);
        iris::FrameSynchronizerStage synchronizer({&first, &second}, output, settings, metrics);
        synchronizer.start();
        first.send(packet(0, 10, std::chrono::steady_clock::now()));
        first.close();
        second.close();
        synchronizer.wait();
        auto partial = output.receive();
        assert(partial);
        assert(partial->frames.size() == 1);
        assert(partial->frames.front().camera == 0);
        assert(metrics.snapshot().counters.at("iris_sync_partial_batches_total") == 1);
    }
    {
        iris::infrastructure::metrics::MetricRegistry metrics;
        iris::Channel<iris::Packet> first(4, iris::OverflowPolicy::DropOldest);
        iris::Channel<iris::Packet> second(4, iris::OverflowPolicy::DropOldest);
        iris::Channel<iris::Packet> output(8, iris::OverflowPolicy::DropOldest);
        auto settings = config(iris::IncompleteBatchPolicy::DropBatch);
        settings.sync_tolerance = std::chrono::milliseconds(5);
        settings.sync_queue_capacity = 4;
        iris::FrameSynchronizerStage synchronizer({&first, &second}, output, settings, metrics);
        synchronizer.start();
        const auto base = std::chrono::steady_clock::now();
        first.send(packet(0, 1, base + std::chrono::milliseconds(8)));
        first.send(packet(0, 2, base + std::chrono::milliseconds(18)));
        first.send(packet(0, 3, base + std::chrono::milliseconds(28)));
        second.send(packet(1, 1, base));
        second.send(packet(1, 2, base + std::chrono::milliseconds(10)));
        second.send(packet(1, 3, base + std::chrono::milliseconds(20)));
        second.send(packet(1, 4, base + std::chrono::milliseconds(30)));
        first.close();
        second.close();
        synchronizer.wait();

        for (int expected = 1; expected <= 3; ++expected) {
            auto matched = output.receive();
            assert(matched);
            assert(matched->frames.size() == 2);
            assert(matched->frames[0].sequence == static_cast<std::uint64_t>(expected));
            assert(matched->frames[1].sequence == static_cast<std::uint64_t>(expected + 1));
        }
        assert(!output.receive());
        const auto snapshot = metrics.snapshot();
        assert(snapshot.counters.at("iris_sync_batches_emitted_total") == 3);
    }
}
