#include "iris/stages/PoseStage.hpp"

#include "iris/infrastructure/gpu/CudaResources.hpp"

#include <torch/script.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace iris {
namespace {

void copy_and_resize_rgb(const Frame& frame, float* destination) {
    if (frame.format != PixelFormat::Bgr8 || !frame.buffer.data || frame.extent.width == 0 || frame.extent.height == 0) throw std::invalid_argument("PEAR HMR requires a valid Bgr8 CUDA frame");
    if (frame.ready) frame.ready->synchronize();
    std::vector<std::uint8_t> bgr(frame.extent.width * frame.extent.height * 3);
    infrastructure::gpu::check_cuda(cudaMemcpy2D(bgr.data(), frame.extent.width * 3, frame.buffer.data, frame.buffer.stride_bytes, frame.extent.width * 3, frame.extent.height, cudaMemcpyDeviceToHost), "copy HMR frame to host");
    const auto plane = pose_model_height * pose_model_width;
    for (std::size_t y = 0; y < pose_model_height; ++y) for (std::size_t x = 0; x < pose_model_width; ++x) {
        const auto source_y = std::min<std::size_t>(frame.extent.height - 1, y * frame.extent.height / pose_model_height);
        const auto source_x = std::min<std::size_t>(frame.extent.width - 1, x * frame.extent.width / pose_model_width);
        const auto source = (source_y * frame.extent.width + source_x) * 3, target = y * pose_model_width + x;
        destination[target] = bgr[source + 2] / 255.0F;
        destination[plane + target] = bgr[source + 1] / 255.0F;
        destination[2 * plane + target] = bgr[source] / 255.0F;
    }
}

void require_shape(const torch::Tensor& tensor, std::initializer_list<int64_t> expected, const char* name) {
    if (tensor.sizes() != torch::IntArrayRef(expected.begin(), expected.size())) throw std::runtime_error("unexpected PEAR HMR output shape for " + std::string(name));
}

template <std::size_t Size>
void copy_output(const torch::Tensor& tensor, std::array<float, Size>& destination, std::initializer_list<int64_t> expected, const char* name) {
    require_shape(tensor, expected, name);
    const auto cpu = tensor.to(torch::kCPU, torch::kFloat32).contiguous();
    std::copy_n(cpu.data_ptr<float>(), Size, destination.begin());
}

} // namespace

class PoseStage::Impl {
  public:
    explicit Impl(PoseConfig config) : config_(std::move(config)), device_(config_.device) {}
    void start() {
        if (config_.model_path.empty()) return;
        if (!std::filesystem::is_regular_file(config_.model_path)) throw std::runtime_error("PEAR HMR TorchScript model does not exist: " + config_.model_path.string());
        try { model_ = torch::jit::load(config_.model_path.string(), device_); model_.eval(); started_ = true; }
        catch (const c10::Error& error) { throw std::runtime_error("could not load PEAR HMR TorchScript model: " + std::string(error.what())); }
    }
    void stop() { model_ = torch::jit::script::Module{}; started_ = false; }
    void process(Packet& packet) {
        if (config_.model_path.empty() || packet.frames.empty()) return;
        if (!started_) throw std::logic_error("pose stage was not started");
        const auto batch = packet.frames.size();
        std::vector<float> images(batch * 3 * pose_model_height * pose_model_width);
        for (std::size_t index = 0; index < batch; ++index) copy_and_resize_rgb(packet.frames[index], images.data() + index * 3 * pose_model_height * pose_model_width);
        torch::InferenceMode inference;
        auto input = torch::from_blob(images.data(), {static_cast<int64_t>(batch), 3, pose_model_height, pose_model_width}, torch::TensorOptions().dtype(torch::kFloat32)).clone().to(device_);
        const auto output = model_.forward({input});
        if (!output.isTuple() || output.toTuple()->elements().size() != 15) throw std::runtime_error("PEAR HMR model must return its 15-value parameter tuple");
        const auto& values = output.toTuple()->elements();
        PoseBatch results; results.reserve(batch);
        for (std::size_t index = 0; index < batch; ++index) {
            Pose pose; pose.source_sequence = packet.sequence; pose.source_camera = packet.frames[index].camera; pose.score = 1.0F;
            HmrParameters p;
            copy_output(values[0].toTensor()[index], p.camera_rt, {4, 4}, "camera_rt");
            copy_output(values[1].toTensor()[index], p.global_pose, {1, 3, 3}, "global_pose");
            copy_output(values[2].toTensor()[index], p.body_pose, {21, 3, 3}, "body_pose");
            copy_output(values[3].toTensor()[index], p.left_hand_pose, {15, 3, 3}, "left_hand_pose");
            copy_output(values[4].toTensor()[index], p.right_hand_pose, {15, 3, 3}, "right_hand_pose");
            copy_output(values[5].toTensor()[index], p.hand_scale, {3}, "hand_scale");
            copy_output(values[6].toTensor()[index], p.head_scale, {3}, "head_scale");
            copy_output(values[7].toTensor()[index], p.body_shape, {200}, "body_shape");
            copy_output(values[8].toTensor()[index], p.body_expression, {50}, "body_expression");
            copy_output(values[9].toTensor()[index], p.flame_eye_pose, {6}, "flame_eye_pose");
            copy_output(values[10].toTensor()[index], p.flame_pose, {3}, "flame_pose");
            copy_output(values[11].toTensor()[index], p.flame_jaw_pose, {3}, "flame_jaw_pose");
            copy_output(values[12].toTensor()[index], p.flame_eyelid, {2}, "flame_eyelid");
            copy_output(values[13].toTensor()[index], p.flame_expression, {50}, "flame_expression");
            copy_output(values[14].toTensor()[index], p.flame_shape, {300}, "flame_shape");
            pose.hmr = std::move(p); results.push_back(std::move(pose));
        }
        packet.poses = std::move(results);
    }
  private:
    PoseConfig config_;
    torch::Device device_;
    torch::jit::script::Module model_;
    bool started_{};
};

PoseStage::PoseStage(Channel<Packet>& input, Channel<Packet>* output, PoseConfig config) : Stage(input, output), impl_(std::make_unique<Impl>(std::move(config))) {}
PoseStage::~PoseStage() { stop(); }
void PoseStage::start() { impl_->start(); Stage::start(); }
void PoseStage::stop() { Stage::stop(); impl_->stop(); }
void PoseStage::process(Packet& packet) { impl_->process(packet); }
} // namespace iris
