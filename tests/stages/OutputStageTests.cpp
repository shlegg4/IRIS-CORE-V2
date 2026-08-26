#include "iris/infrastructure/metrics/MetricRegistry.hpp"
#include "iris/stages/OutputStage.hpp"
#include "iris/stages/PoseStage.hpp"

#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

namespace {

bool wait_for_count(const iris::OutputStage& output, std::size_t expected) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline) {
        if (output.processed_count() >= expected) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}

} // namespace

int main() {
    namespace fs = std::filesystem;
    iris::infrastructure::metrics::MetricRegistry metrics;
    iris::Channel<iris::Packet> input(8, iris::OverflowPolicy::Block);
    iris::OutputConfig config;
    config.disk.queue_capacity = 8;
    iris::OutputStage output(input, metrics, config);

    const auto test_directory = fs::temp_directory_path() / "iris-v2-output-tests";
    fs::create_directories(test_directory);
    const auto first_recording = test_directory / "first.mp4";
    const auto second_recording = test_directory / "second.mp4";
    fs::remove(first_recording);
    fs::remove(second_recording);

    auto disk = config.disk;
    disk.destination = first_recording;
    assert(output.configure_disk(disk));
    output.start();

    for (std::uint64_t sequence = 1; sequence <= 3; ++sequence) {
        iris::Packet packet;
        packet.sequence = sequence;
        packet.poses = iris::PoseBatch{iris::Pose{sequence}};
        assert(input.send(std::move(packet)) == iris::SendResult::Sent);
    }
    assert(wait_for_count(output, 3));
    assert(output.start_recording());

    auto rejected = disk;
    rejected.destination = second_recording;
    assert(!output.configure_disk(rejected));
    assert(rejected.destination == second_recording);

    auto empty_recording = output.stop_recording();
    assert(!empty_recording);
    assert(!fs::exists(first_recording));

    assert(output.configure_disk(rejected));
    input.close();
    output.stop();

    assert(!fs::exists(second_recording));
    const auto snapshot = metrics.snapshot();
    assert(snapshot.counters.at("iris_output_packets_received_total") == 3);
    assert(snapshot.counters.at("iris_output_disk_packets_total") == 0);
    assert(snapshot.counters.at("iris_output_disk_failures_total") == 0);
    assert(snapshot.gauges.at("iris_output_recording_active") == 0.0);

    fs::remove_all(test_directory);

    iris::Channel<iris::Packet> pose_input(1);
    iris::Channel<iris::Packet> pose_output(1);
    iris::PoseStage pose(pose_input, &pose_output);
    pose.start();
    iris::Packet forwarded;
    forwarded.sequence = 99;
    assert(pose_input.send(std::move(forwarded)) == iris::SendResult::Sent);
    pose_input.close();
    auto received = pose_output.receive();
    assert(received);
    assert(received->sequence == 99);
    assert(!received->poses);
    pose.stop();
}
