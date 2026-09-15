#include "iris/stages/PoseStage.hpp"

#include "iris/infrastructure/gpu/CudaResources.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if IRIS_HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace iris {
namespace {

constexpr std::array<const char*, 6> input_names{
    "images", "rotation", "translation", "intrinsic", "distortion", "view_mask"};
constexpr std::array<const char*, 5> output_names{
    "poses_3d_mm", "points_2d", "joint_confidence", "person_scores", "active_mask"};

void require_shape(const std::vector<int64_t>& actual, std::initializer_list<int64_t> expected,
                   std::string_view name) {
    if (actual != std::vector<int64_t>(expected)) {
        throw std::runtime_error("unexpected ONNX shape for " + std::string(name));
    }
}

void copy_and_resize_rgb(const Frame& frame, std::uint8_t* destination) {
    if (frame.format != PixelFormat::Bgr8 || !frame.buffer.data || frame.extent.width == 0 ||
        frame.extent.height == 0) {
        throw std::invalid_argument("pose model requires a valid Bgr8 CUDA frame");
    }
    if (frame.ready) {
        frame.ready->synchronize();
    }
    std::vector<std::uint8_t> bgr(frame.extent.width * frame.extent.height * 3);
    infrastructure::gpu::check_cuda(
        cudaMemcpy2D(bgr.data(), frame.extent.width * 3, frame.buffer.data,
                     frame.buffer.stride_bytes, frame.extent.width * 3, frame.extent.height,
                     cudaMemcpyDeviceToHost),
        "copy pose frame to host");

    for (std::size_t y = 0; y < pose_model_height; ++y) {
        const auto source_y = std::min<std::size_t>(
            frame.extent.height - 1, y * frame.extent.height / pose_model_height);
        for (std::size_t x = 0; x < pose_model_width; ++x) {
            const auto source_x = std::min<std::size_t>(
                frame.extent.width - 1, x * frame.extent.width / pose_model_width);
            const auto source = (source_y * frame.extent.width + source_x) * 3;
            const auto target = y * pose_model_width + x;
            const auto plane = pose_model_height * pose_model_width;
            destination[target] = bgr[source + 2];
            destination[plane + target] = bgr[source + 1];
            destination[2 * plane + target] = bgr[source];
        }
    }
}

CameraCalibration scaled_calibration(CameraCalibration value, const Frame& frame) {
    const auto x_scale = static_cast<float>(pose_model_width) / frame.extent.width;
    const auto y_scale = static_cast<float>(pose_model_height) / frame.extent.height;
    for (std::size_t column = 0; column < 3; ++column) {
        value.intrinsic[column] *= x_scale;
        value.intrinsic[3 + column] *= y_scale;
    }
    return value;
}

} // namespace

class PoseStage::Impl {
  public:
    explicit Impl(PoseConfig config) : config_(std::move(config)) {
        for (const auto& calibration : config_.calibrations) {
            if (!calibrations_.emplace(calibration.camera_id, calibration).second) {
                throw std::invalid_argument("duplicate pose calibration camera ID");
            }
        }
        if (config_.calibrations.size() > pose_model_views) {
            throw std::invalid_argument("pose model supports at most five calibrated cameras");
        }
    }

    void start() {
        if (config_.model_path.empty()) {
            return;
        }
#if IRIS_HAVE_ONNXRUNTIME
        if (!std::filesystem::exists(config_.model_path)) {
            throw std::runtime_error("pose ONNX model does not exist: " + config_.model_path.string());
        }
        session_ = std::make_unique<Ort::Session>(environment(), config_.model_path.c_str(), options_);
        validate_contract();
#else
        throw std::runtime_error("pose model configured but IRIS was built without ONNX Runtime; set IRIS_ONNXRUNTIME_ROOT");
#endif
    }

    void stop() {
#if IRIS_HAVE_ONNXRUNTIME
        session_.reset();
#endif
    }

    void process(Packet& packet) {
        if (config_.model_path.empty()) {
            return;
        }
#if IRIS_HAVE_ONNXRUNTIME
        if (!session_) {
            throw std::logic_error("pose stage was not started");
        }
        if (packet.frames.size() > pose_model_views) {
            throw std::invalid_argument("pose packet contains more than five frames");
        }
        std::vector<std::uint8_t> images(pose_model_views * 3 * pose_model_height * pose_model_width);
        std::array<float, pose_model_views * 9> rotation{};
        std::array<float, pose_model_views * 3> translation{};
        std::array<float, pose_model_views * 9> intrinsic{};
        std::array<float, pose_model_views * 5> distortion{};
        std::array<bool, pose_model_views> view_mask{};
        for (std::size_t view = 0; view < packet.frames.size(); ++view) {
            const auto& frame = packet.frames[view];
            const auto calibration = calibrations_.find(frame.camera);
            if (calibration == calibrations_.end()) {
                throw std::invalid_argument("no pose calibration configured for camera " + std::to_string(frame.camera));
            }
            const auto scaled = scaled_calibration(calibration->second, frame);
            copy_and_resize_rgb(frame, images.data() + view * 3 * pose_model_height * pose_model_width);
            std::copy(scaled.rotation.begin(), scaled.rotation.end(), rotation.begin() + view * 9);
            std::copy(scaled.translation_mm.begin(), scaled.translation_mm.end(), translation.begin() + view * 3);
            std::copy(scaled.intrinsic.begin(), scaled.intrinsic.end(), intrinsic.begin() + view * 9);
            std::copy(scaled.distortion.begin(), scaled.distortion.end(), distortion.begin() + view * 5);
            view_mask[view] = true;
        }
        const std::array<int64_t, 5> image_shape{1, 5, 3, pose_model_height, pose_model_width};
        const std::array<int64_t, 4> matrix_shape{1, 5, 3, 3};
        const std::array<int64_t, 3> vector_shape{1, 5, 3};
        const std::array<int64_t, 3> distortion_shape{1, 5, 5};
        const std::array<int64_t, 2> mask_shape{1, 5};
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<Ort::Value, 6> inputs{
            Ort::Value::CreateTensor<std::uint8_t>(memory, images.data(), images.size(), image_shape.data(), image_shape.size()),
            Ort::Value::CreateTensor<float>(memory, rotation.data(), rotation.size(), matrix_shape.data(), matrix_shape.size()),
            Ort::Value::CreateTensor<float>(memory, translation.data(), translation.size(), vector_shape.data(), vector_shape.size()),
            Ort::Value::CreateTensor<float>(memory, intrinsic.data(), intrinsic.size(), matrix_shape.data(), matrix_shape.size()),
            Ort::Value::CreateTensor<float>(memory, distortion.data(), distortion.size(), distortion_shape.data(), distortion_shape.size()),
            Ort::Value::CreateTensor<bool>(memory, view_mask.data(), view_mask.size(), mask_shape.data(), mask_shape.size())};
        auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(), output_names.data(), output_names.size());
        decode_outputs(packet, outputs);
#else
        (void)packet;
#endif
    }

  private:
#if IRIS_HAVE_ONNXRUNTIME
    static Ort::Env& environment() {
        static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "iris_pose");
        return env;
    }
    void validate_contract() const {
        if (session_->GetInputCount() != input_names.size() || session_->GetOutputCount() != output_names.size()) {
            throw std::runtime_error("pose ONNX graph has an unexpected input/output count");
        }
        Ort::AllocatorWithDefaultOptions allocator;
        std::unordered_set<std::string> graph_inputs;
        std::unordered_set<std::string> graph_outputs;
        for (std::size_t i = 0; i < input_names.size(); ++i) {
            const auto name = session_->GetInputNameAllocated(i, allocator);
            graph_inputs.emplace(name.get());
        }
        for (std::size_t i = 0; i < output_names.size(); ++i) {
            const auto name = session_->GetOutputNameAllocated(i, allocator);
            graph_outputs.emplace(name.get());
        }
        for (const auto* name : input_names) if (!graph_inputs.contains(name)) throw std::runtime_error("required pose ONNX input is missing: " + std::string(name));
        for (const auto* name : output_names) if (!graph_outputs.contains(name)) throw std::runtime_error("required pose ONNX output is missing: " + std::string(name));
    }
    void decode_outputs(Packet& packet, std::vector<Ort::Value>& outputs) const {
        require_shape(outputs[0].GetTensorTypeAndShapeInfo().GetShape(), {1, 32, 19, 3}, output_names[0]);
        require_shape(outputs[1].GetTensorTypeAndShapeInfo().GetShape(), {1, 32, 5, 19, 2}, output_names[1]);
        require_shape(outputs[2].GetTensorTypeAndShapeInfo().GetShape(), {1, 32, 5, 19}, output_names[2]);
        require_shape(outputs[3].GetTensorTypeAndShapeInfo().GetShape(), {1, 32}, output_names[3]);
        require_shape(outputs[4].GetTensorTypeAndShapeInfo().GetShape(), {1, 32}, output_names[4]);
        const auto* poses = outputs[0].GetTensorData<float>();
        const auto* points = outputs[1].GetTensorData<float>();
        const auto* confidences = outputs[2].GetTensorData<float>();
        const auto* scores = outputs[3].GetTensorData<float>();
        const auto* active = outputs[4].GetTensorData<bool>();
        PoseBatch result;
        for (std::size_t person = 0; person < pose_model_proposals; ++person) {
            if (!active[person]) continue;
            Pose pose;
            pose.source_sequence = packet.sequence;
            pose.score = scores[person];
            for (std::size_t joint = 0; joint < panoptic_joint_count; ++joint) {
                const auto index = (person * panoptic_joint_count + joint) * 3;
                pose.joints_3d_mm[joint] = {poses[index], poses[index + 1], poses[index + 2]};
                float sum{};
                for (std::size_t view = 0; view < pose_model_views; ++view) {
                    const auto confidence_index = person * pose_model_views * panoptic_joint_count +
                                                  view * panoptic_joint_count + joint;
                    const auto point_index = confidence_index * 2;
                    pose.points_2d_px[view][joint] = {points[point_index], points[point_index + 1]};
                    pose.per_view_joint_confidence[view][joint] = confidences[confidence_index];
                    sum += confidences[confidence_index];
                }
                pose.joint_confidence[joint] = sum / pose_model_views;
            }
            result.push_back(std::move(pose));
        }
        packet.poses = std::move(result);
    }
    Ort::SessionOptions options_;
    std::unique_ptr<Ort::Session> session_;
#endif
    PoseConfig config_;
    std::unordered_map<CameraId, CameraCalibration> calibrations_;
};

PoseStage::PoseStage(Channel<Packet>& input, Channel<Packet>* output, PoseConfig config)
    : Stage(input, output), impl_(std::make_unique<Impl>(std::move(config))) {}
PoseStage::~PoseStage() { stop(); }
void PoseStage::start() { impl_->start(); Stage::start(); }
void PoseStage::stop() { Stage::stop(); impl_->stop(); }
void PoseStage::process(Packet& packet) { impl_->process(packet); }

} // namespace iris
