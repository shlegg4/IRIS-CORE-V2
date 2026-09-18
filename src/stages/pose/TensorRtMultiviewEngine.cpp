#include "iris/stages/pose/TensorRtMultiviewEngine.hpp"

#include <stdexcept>
#include <algorithm>
#include <utility>

#ifdef IRIS_HAS_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <vector>
extern "C" cudaError_t iris_multiview_preprocess(const void* const*, const std::size_t*, const std::uint32_t*, const std::uint32_t*, const float*, const float*, const float*, float*, cudaStream_t, const std::uint8_t**, std::size_t*, std::uint32_t*, std::uint32_t*, float*, float*, float*);

namespace {
class Logger final : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kERROR) last_error = message ? message : "TensorRT error";
    }
    std::string last_error;
};
template <typename T> struct TrtDeleter { void operator()(T* value) const {
    if (!value) return;
#if NV_TENSORRT_MAJOR >= 10
    delete value;
#else
    value->destroy();
#endif
} };
template <typename T> using TrtPtr = std::unique_ptr<T, TrtDeleter<T>>;
void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
}
#endif

namespace iris {
class TensorRtMultiviewEngine::Impl {
  public:
    explicit Impl(const PoseConfig& config) {
#ifdef IRIS_HAS_TENSORRT
        std::ifstream file(config.multiview_engine_path, std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("could not open TensorRT engine: " + config.multiview_engine_path.string());
        const auto size = static_cast<std::size_t>(file.tellg());
        file.seekg(0);
        std::vector<char> bytes(size);
        file.read(bytes.data(), static_cast<std::streamsize>(size));
        if (!file || size == 0) throw std::runtime_error("could not read TensorRT engine bytes");
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) throw std::runtime_error("could not create TensorRT runtime: " + logger_.last_error);
        engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (!engine_) throw std::runtime_error("could not deserialize TensorRT engine: " + logger_.last_error);
        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("could not create TensorRT execution context");
        validate("images", nvinfer1::Dims4{3,3,640,640}, nvinfer1::TensorIOMode::kINPUT, nvinfer1::DataType::kFLOAT);
        validate("keypoints", nvinfer1::Dims4{3,10,17,2}, nvinfer1::TensorIOMode::kOUTPUT, nvinfer1::DataType::kFLOAT);
        validate("keypoint_scores", nvinfer1::Dims3{3,10,17}, nvinfer1::TensorIOMode::kOUTPUT, nvinfer1::DataType::kFLOAT);
        validate("instance_scores", nvinfer1::Dims2{3,10}, nvinfer1::TensorIOMode::kOUTPUT, nvinfer1::DataType::kFLOAT);
        validate("boxes", nvinfer1::Dims3{3,10,4}, nvinfer1::TensorIOMode::kOUTPUT, nvinfer1::DataType::kFLOAT);
        validate("candidate_valid", nvinfer1::Dims2{3,10}, nvinfer1::TensorIOMode::kOUTPUT, nvinfer1::DataType::kBOOL);
        check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
        check_cuda(cudaMalloc(&images_, sizeof(float) * 3 * 3 * 640 * 640), "cudaMalloc images");
        check_cuda(cudaMalloc(&keypoints_, sizeof(float) * 3 * 10 * 17 * 2), "cudaMalloc keypoints");
        check_cuda(cudaMalloc(&keypoint_scores_, sizeof(float) * 3 * 10 * 17), "cudaMalloc keypoint_scores");
        check_cuda(cudaMalloc(&instance_scores_, sizeof(float) * 3 * 10), "cudaMalloc instance_scores");
        check_cuda(cudaMalloc(&boxes_, sizeof(float) * 3 * 10 * 4), "cudaMalloc boxes");
        check_cuda(cudaMalloc(&candidate_valid_, sizeof(unsigned char) * 3 * 10), "cudaMalloc candidate_valid");
        check_cuda(cudaMalloc(&source_ptrs_, sizeof(void*) * 3), "cudaMalloc source pointers");
        check_cuda(cudaMalloc(&strides_, sizeof(std::size_t) * 3), "cudaMalloc strides");
        check_cuda(cudaMalloc(&widths_, sizeof(std::uint32_t) * 3), "cudaMalloc widths");
        check_cuda(cudaMalloc(&heights_, sizeof(std::uint32_t) * 3), "cudaMalloc heights");
        check_cuda(cudaMalloc(&source_k_, sizeof(float) * 27), "cudaMalloc source intrinsics");
        check_cuda(cudaMalloc(&target_k_, sizeof(float) * 27), "cudaMalloc target intrinsics");
        check_cuda(cudaMalloc(&distortion_, sizeof(float) * 15), "cudaMalloc distortion");
        for (std::size_t i = 0; i < 3; ++i) {
            std::copy(config.multiview_calibration[i].intrinsics.begin(), config.multiview_calibration[i].intrinsics.end(), calibration_k_.begin() + i * 9);
            std::copy(config.multiview_calibration[i].distortion.begin(), config.multiview_calibration[i].distortion.end(), calibration_distortion_.begin() + i * 5);
        }
#else
        (void)config;
        throw std::runtime_error("TensorRT support was not compiled into IRIS");
#endif
    }
    ~Impl() {
#ifdef IRIS_HAS_TENSORRT
        if (stream_) cudaStreamSynchronize(stream_);
        cudaFree(images_);
        cudaFree(keypoints_); cudaFree(keypoint_scores_); cudaFree(instance_scores_); cudaFree(boxes_); cudaFree(candidate_valid_);
        cudaFree(source_ptrs_); cudaFree(strides_); cudaFree(widths_); cudaFree(heights_);
        cudaFree(source_k_); cudaFree(target_k_); cudaFree(distortion_);
        if (stream_) cudaStreamDestroy(stream_);
#endif
    }
    void infer(const std::array<const void*, 3>& sources, const std::array<std::size_t, 3>& strides,
               const std::array<std::uint32_t, 3>& widths, const std::array<std::uint32_t, 3>& heights,
               TensorRtMultiviewResult& result) {
#ifndef IRIS_HAS_TENSORRT
        throw std::runtime_error("TensorRT support was not compiled into IRIS");
#else
        std::array<float, 27> target_k{};
        for (std::size_t view = 0; view < 3; ++view) {
            const float scale = std::min(640.0f / static_cast<float>(widths[view]), 640.0f / static_cast<float>(heights[view]));
            const auto resized_width = std::max(1, static_cast<int>(widths[view] * scale + 0.5f));
            const auto resized_height = std::max(1, static_cast<int>(heights[view] * scale + 0.5f));
            const auto pad_x = static_cast<float>((640 - resized_width) / 2);
            const auto pad_y = static_cast<float>((640 - resized_height) / 2);
            target_k[view * 9] = calibration_k_[view * 9] * scale;
            target_k[view * 9 + 4] = calibration_k_[view * 9 + 4] * scale;
            target_k[view * 9 + 2] = calibration_k_[view * 9 + 2] * scale + pad_x;
            target_k[view * 9 + 5] = calibration_k_[view * 9 + 5] * scale + pad_y;
            target_k[view * 9 + 8] = 1.0f;
        }
        result.letterbox_intrinsics = target_k;
        check_cuda(iris_multiview_preprocess(sources.data(), strides.data(), widths.data(), heights.data(), calibration_k_.data(), target_k.data(), calibration_distortion_.data(), static_cast<float*>(images_), stream_, static_cast<const std::uint8_t**>(source_ptrs_), static_cast<std::size_t*>(strides_), static_cast<std::uint32_t*>(widths_), static_cast<std::uint32_t*>(heights_), static_cast<float*>(source_k_), static_cast<float*>(target_k_), static_cast<float*>(distortion_)), "multiview preprocessing");
        // The pipeline supplies one image for each of its three calibrated views.
        // This is a no-op for a static engine and selects N=3 for a dynamic one.
        if (!context_->setInputShape("images", nvinfer1::Dims4{3, 3, 640, 640}))
            throw std::runtime_error("TensorRT rejected the RTMO batch shape [3,3,640,640]");
        if (!context_->setTensorAddress("images", images_) || !context_->setTensorAddress("keypoints", keypoints_) || !context_->setTensorAddress("keypoint_scores", keypoint_scores_) || !context_->setTensorAddress("instance_scores", instance_scores_) || !context_->setTensorAddress("boxes", boxes_) || !context_->setTensorAddress("candidate_valid", candidate_valid_))
            throw std::runtime_error("TensorRT rejected one or more tensor addresses");
        if (!context_->enqueueV3(stream_)) throw std::runtime_error("TensorRT enqueueV3 failed: " + logger_.last_error);
        check_cuda(cudaMemcpyAsync(result.keypoints.data(), keypoints_, sizeof(float) * result.keypoints.size(), cudaMemcpyDeviceToHost, stream_), "copy keypoints");
        check_cuda(cudaMemcpyAsync(result.keypoint_scores.data(), keypoint_scores_, sizeof(float) * result.keypoint_scores.size(), cudaMemcpyDeviceToHost, stream_), "copy keypoint_scores");
        check_cuda(cudaMemcpyAsync(result.instance_scores.data(), instance_scores_, sizeof(float) * result.instance_scores.size(), cudaMemcpyDeviceToHost, stream_), "copy instance_scores");
        check_cuda(cudaMemcpyAsync(result.boxes.data(), boxes_, sizeof(float) * result.boxes.size(), cudaMemcpyDeviceToHost, stream_), "copy boxes");
        check_cuda(cudaMemcpyAsync(result.candidate_valid.data(), candidate_valid_, sizeof(unsigned char) * result.candidate_valid.size(), cudaMemcpyDeviceToHost, stream_), "copy candidate_valid");
        check_cuda(cudaStreamSynchronize(stream_), "synchronize TensorRT inference");
#endif
    }
#ifdef IRIS_HAS_TENSORRT
    void validate(const char* name, nvinfer1::Dims expected, nvinfer1::TensorIOMode mode, nvinfer1::DataType type) {
        if (engine_->getTensorIOMode(name) != mode) throw std::runtime_error(std::string("unexpected TensorRT I/O mode: ") + name);
        if (engine_->getTensorDataType(name) != type) throw std::runtime_error(std::string("unexpected TensorRT data type: ") + name);
        const auto dims = engine_->getTensorShape(name);
        if (dims.nbDims != expected.nbDims) throw std::runtime_error(std::string("unexpected TensorRT rank: ") + name);
        for (int i = 0; i < dims.nbDims; ++i)
            if (dims.d[i] != -1 && dims.d[i] != expected.d[i])
                throw std::runtime_error(std::string("unexpected TensorRT shape: ") + name);
    }
    Logger logger_;
    TrtPtr<nvinfer1::IRuntime> runtime_;
    TrtPtr<nvinfer1::ICudaEngine> engine_;
    TrtPtr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_{};
    void *images_{}, *keypoints_{}, *keypoint_scores_{}, *instance_scores_{}, *boxes_{}, *candidate_valid_{};
    void *source_ptrs_{}, *strides_{}, *widths_{}, *heights_{}, *source_k_{}, *target_k_{}, *distortion_{};
    std::array<float, 27> calibration_k_{};
    std::array<float, 15> calibration_distortion_{};
#endif
};

TensorRtMultiviewEngine::TensorRtMultiviewEngine(const PoseConfig& config) : impl_(std::make_unique<Impl>(config)) {}
TensorRtMultiviewEngine::~TensorRtMultiviewEngine() = default;
void TensorRtMultiviewEngine::infer(const std::array<const void*, 3>& bgr_device,
                                    const std::array<std::size_t, 3>& strides,
                                    const std::array<std::uint32_t, 3>& widths,
                                    const std::array<std::uint32_t, 3>& heights,
                                    TensorRtMultiviewResult& result) {
    impl_->infer(bgr_device, strides, widths, heights, result);
}
} // namespace iris
