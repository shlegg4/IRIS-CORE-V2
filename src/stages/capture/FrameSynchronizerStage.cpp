#include "iris/stages/capture/FrameSynchronizerStage.hpp"

#include <algorithm>
#include <chrono>
#include <deque>
#include <optional>
#include <thread>
#include <limits>
#include <functional>

namespace iris {
class FrameSynchronizerStage::Impl {
  public:
    Impl(std::vector<Channel<Packet>*> inputs, Channel<Packet>& output,
         const MultiCameraCaptureConfig& config, infrastructure::metrics::MetricRegistry& metrics)
        : inputs(std::move(inputs)), output(output), tolerance(config.sync_tolerance),
          capacity(config.sync_queue_capacity), policy(config.incomplete_batch_policy),
          buffers(this->inputs.size()), emitted(metrics.counter("iris_sync_batches_emitted_total")),
          dropped(metrics.counter("iris_sync_frames_dropped_total")),
          partial(metrics.counter("iris_sync_partial_batches_total")),
          skew(metrics.histogram("iris_sync_match_skew_ms", {0.1, 0.5, 1, 2, 3, 5, 10})) {}
    void run(std::stop_token stop) {
        while (!stop.stop_requested()) {
            bool changed = false;
            for (std::size_t i = 0; i < inputs.size(); ++i) {
                while (auto packet = inputs[i]->try_receive()) {
                    if (packet->frames.size() == 1) {
                        buffers[i].push_back(std::move(packet->frames[0]));
                        while (buffers[i].size() > capacity) {
                            buffers[i].pop_front();
                            dropped.increment();
                        }
                    }
                    changed = true;
                }
            }
            match();
            bool all_closed = true;
            for (const auto* input : inputs) {
                all_closed = all_closed && input->stats().closed;
            }
            if (policy == IncompleteBatchPolicy::EmitPartial) {
                emit_expired_partial(all_closed);
            }
            if (all_closed) {
                if (policy == IncompleteBatchPolicy::DropBatch) {
                    for (auto& buffer : buffers) {
                        dropped.increment(buffer.size());
                        buffer.clear();
                    }
                }
                break;
            }
            if (!changed) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        output.close();
    }
    void match() {
        for (;;) {
            for (const auto& buffer : buffers) {
                if (buffer.empty()) {
                    return;
                }
            }

            // Search the buffered frames instead of only comparing the fronts.
            // This matters when one camera's frame arrives before the closest
            // frame from another camera. The queue capacity is deliberately
            // bounded, so exhaustive search remains practical and gives us a
            // deterministic closest-skew match.
            std::vector<std::size_t> candidate(buffers.size(), 0);
            std::vector<std::size_t> best_indices;
            auto best_difference = std::chrono::steady_clock::duration::max();
            auto best_latest = MonotonicTime::max();

            std::function<void(std::size_t)> search = [&](std::size_t camera) {
                if (camera == buffers.size()) {
                    auto earliest = buffers[0][candidate[0]].timing.estimated_capture_time;
                    auto latest = earliest;
                    for (std::size_t i = 1; i < buffers.size(); ++i) {
                        const auto time = buffers[i][candidate[i]].timing.estimated_capture_time;
                        earliest = std::min(earliest, time);
                        latest = std::max(latest, time);
                    }
                    const auto difference = latest - earliest;
                    if (difference <= tolerance &&
                        (difference < best_difference ||
                         (difference == best_difference && latest < best_latest))) {
                        best_difference = difference;
                        best_latest = latest;
                        best_indices = candidate;
                    }
                    return;
                }
                for (std::size_t index = 0; index < buffers[camera].size(); ++index) {
                    candidate[camera] = index;
                    search(camera + 1);
                }
            };
            search(0);

            if (best_indices.empty()) {
                // No buffered combination can satisfy the tolerance. The
                // globally oldest frame cannot be part of a valid match and
                // is therefore safe to discard.
                std::size_t earliest_index = 0;
                auto earliest = buffers[0].front().timing.estimated_capture_time;
                for (std::size_t i = 1; i < buffers.size(); ++i) {
                    const auto time = buffers[i].front().timing.estimated_capture_time;
                    if (time < earliest) {
                        earliest = time;
                        earliest_index = i;
                    }
                }
                buffers[earliest_index].pop_front();
                dropped.increment();
                continue;
            }

            FrameBatch batch;
            for (std::size_t i = 0; i < buffers.size(); ++i) {
                batch.push_back(std::move(buffers[i][best_indices[i]]));
            }
            for (std::size_t i = 0; i < buffers.size(); ++i) {
                buffers[i].erase(buffers[i].begin() + static_cast<std::ptrdiff_t>(best_indices[i]));
            }
            skew.observe(std::chrono::duration<double, std::milli>(best_difference).count());
            if (output.send(Packet{sequence++, std::move(batch), std::nullopt}) ==
                SendResult::Closed) {
                return;
            }
            emitted.increment();
        }
    }
    void emit_expired_partial(bool flush) {
        for (;;) {
            std::optional<MonotonicTime> earliest;
            for (const auto& buffer : buffers) {
                if (!buffer.empty() &&
                    (!earliest || buffer.front().timing.estimated_capture_time < *earliest)) {
                    earliest = buffer.front().timing.estimated_capture_time;
                }
            }
            if (!earliest) {
                return;
            }
            if (!flush && std::chrono::steady_clock::now() - *earliest <= tolerance) {
                return;
            }
            FrameBatch batch;
            MonotonicTime latest = *earliest;
            for (auto& buffer : buffers) {
                if (!buffer.empty() &&
                    buffer.front().timing.estimated_capture_time - *earliest <= tolerance) {
                    latest = std::max(latest, buffer.front().timing.estimated_capture_time);
                    batch.push_back(std::move(buffer.front()));
                    buffer.pop_front();
                }
            }
            if (batch.empty()) {
                return;
            }
            skew.observe(std::chrono::duration<double, std::milli>(latest - *earliest).count());
            if (batch.size() != inputs.size()) {
                partial.increment();
            }
            if (output.send(Packet{sequence++, std::move(batch), std::nullopt}) ==
                SendResult::Closed) {
                return;
            }
            emitted.increment();
        }
    }
    std::vector<Channel<Packet>*> inputs;
    Channel<Packet>& output;
    std::chrono::milliseconds tolerance;
    std::size_t capacity;
    IncompleteBatchPolicy policy;
    std::vector<std::deque<Frame>> buffers;
    infrastructure::metrics::Counter emitted, dropped, partial;
    infrastructure::metrics::Histogram skew;
    std::jthread worker;
    std::uint64_t sequence{};
};
FrameSynchronizerStage::FrameSynchronizerStage(std::vector<Channel<Packet>*> i, Channel<Packet>& o,
                                               const MultiCameraCaptureConfig& c,
                                               infrastructure::metrics::MetricRegistry& m)
    : impl_(std::make_unique<Impl>(std::move(i), o, c, m)) {}
FrameSynchronizerStage::~FrameSynchronizerStage() { stop(); }
void FrameSynchronizerStage::start() {
    impl_->worker = std::jthread([this](std::stop_token s) { impl_->run(s); });
}
void FrameSynchronizerStage::wait() {
    if (impl_->worker.joinable()) {
        impl_->worker.join();
    }
}
void FrameSynchronizerStage::stop() {
    if (impl_->worker.joinable()) {
        impl_->worker.request_stop();
    }
    wait();
}
} // namespace iris
