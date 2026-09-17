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
        validate("images", nvinfer1::Dims{5, {1,3,3,640,640}}, nvinfer1::TensorIOMode::kINPUT, nvinfer1::DataType::kFLOAT);
        validate("R_w2c", nvinfer1::Dims4{1,3,3,3}, nvinfer1::TensorIOMode::kINPUT, nvinfer1::DataType::kFLOAT);
        validate("t_w2c", nvinfer1::Dims3{1,3,3}, nvinfer1::TensorIOMode::kINPUT, nvinfer1::DataType::kFLOAT);
        validate("intrinsics", nvinfer1::Dims4{1,3,3,3}, nvinfer1::TensorIOMode::kINPUT, nvinfer1::DataType::kFLOAT);
        validate("poses_3d", nvinfer1::Dims4{1,10,17,3}, nvinfer1::TensorIOMode::kOUTPUT, nvinfer1::DataType::kFLOAT);
        validate("joint_valid", nvinfer1::Dims3{1,10,17}, nvinfer1::TensorIOMode::kOUTPUT, nvinfer1::DataType::kBOOL);
        validate("joint_scores", nvinfer1::Dims4{1,10,3,17}, nvinfer1::TensorIOMode::kOUTPUT, nvinfer1::DataType::kFLOAT);
        check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
        check_cuda(cudaMalloc(&images_, sizeof(float) * 3 * 3 * 640 * 640), "cudaMalloc images");
        check_cuda(cudaMalloc(&r_, sizeof(float) * 27), "cudaMalloc R_w2c");
        check_cuda(cudaMalloc(&t_, sizeof(float) * 9), "cudaMalloc t_w2c");
        check_cuda(cudaMalloc(&k_, sizeof(float) * 27), "cudaMalloc intrinsics");
        check_cuda(cudaMalloc(&poses_, sizeof(float) * 510), "cudaMalloc poses_3d");
        check_cuda(cudaMalloc(&valid_, sizeof(unsigned char) * 170), "cudaMalloc joint_valid");
        check_cuda(cudaMalloc(&scores_, sizeof(float) * 510), "cudaMalloc joint_scores");
        check_cuda(cudaMalloc(&source_ptrs_, sizeof(void*) * 3), "cudaMalloc source pointers");
        check_cuda(cudaMalloc(&strides_, sizeof(std::size_t) * 3), "cudaMalloc strides");
        check_cuda(cudaMalloc(&widths_, sizeof(std::uint32_t) * 3), "cudaMalloc widths");
        check_cuda(cudaMalloc(&heights_, sizeof(std::uint32_t) * 3), "cudaMalloc heights");
        check_cuda(cudaMalloc(&source_k_, sizeof(float) * 27), "cudaMalloc source intrinsics");
        check_cuda(cudaMalloc(&target_k_, sizeof(float) * 27), "cudaMalloc target intrinsics");
        check_cuda(cudaMalloc(&distortion_, sizeof(float) * 15), "cudaMalloc distortion");
        for (std::size_t i = 0; i < 3; ++i) {
            std::copy(config.multiview_calibration[i].R_w2c.begin(), config.multiview_calibration[i].R_w2c.end(), calibration_r_.begin() + i * 9);
            std::copy(config.multiview_calibration[i].t_w2c.begin(), config.multiview_calibration[i].t_w2c.end(), calibration_t_.begin() + i * 3);
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
        cudaFree(images_); cudaFree(r_); cudaFree(t_); cudaFree(k_);
        cudaFree(poses_); cudaFree(valid_); cudaFree(scores_);
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
        check_cuda(cudaMemcpyAsync(r_, calibration_r_.data(), sizeof(float) * 27, cudaMemcpyHostToDevice, stream_), "copy rotations");
        check_cuda(cudaMemcpyAsync(t_, calibration_t_.data(), sizeof(float) * 9, cudaMemcpyHostToDevice, stream_), "copy translations");
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
        check_cuda(cudaMemcpyAsync(k_, target_k.data(), sizeof(float) * 27, cudaMemcpyHostToDevice, stream_), "copy transformed intrinsics");
        check_cuda(iris_multiview_preprocess(sources.data(), strides.data(), widths.data(), heights.data(), calibration_k_.data(), target_k.data(), calibration_distortion_.data(), static_cast<float*>(images_), stream_, static_cast<const std::uint8_t**>(source_ptrs_), static_cast<std::size_t*>(strides_), static_cast<std::uint32_t*>(widths_), static_cast<std::uint32_t*>(heights_), static_cast<float*>(source_k_), static_cast<float*>(target_k_), static_cast<float*>(distortion_)), "multiview preprocessing");
        if (!context_->setTensorAddress("images", images_) || !context_->setTensorAddress("R_w2c", r_) || !context_->setTensorAddress("t_w2c", t_) || !context_->setTensorAddress("intrinsics", k_) || !context_->setTensorAddress("poses_3d", poses_) || !context_->setTensorAddress("joint_valid", valid_) || !context_->setTensorAddress("joint_scores", scores_))
            throw std::runtime_error("TensorRT rejected one or more tensor addresses");
        if (!context_->enqueueV3(stream_)) throw std::runtime_error("TensorRT enqueueV3 failed: " + logger_.last_error);
        check_cuda(cudaMemcpyAsync(result.poses_3d.data(), poses_, sizeof(float) * 510, cudaMemcpyDeviceToHost, stream_), "copy poses_3d");
        check_cuda(cudaMemcpyAsync(result.joint_valid.data(), valid_, sizeof(unsigned char) * 170, cudaMemcpyDeviceToHost, stream_), "copy joint_valid");
        check_cuda(cudaMemcpyAsync(result.joint_scores.data(), scores_, sizeof(float) * 510, cudaMemcpyDeviceToHost, stream_), "copy joint_scores");
        check_cuda(cudaStreamSynchronize(stream_), "synchronize TensorRT inference");
#endif
    }
#ifdef IRIS_HAS_TENSORRT
    void validate(const char* name, nvinfer1::Dims expected, nvinfer1::TensorIOMode mode, nvinfer1::DataType type) {
        if (engine_->getTensorIOMode(name) != mode) throw std::runtime_error(std::string("unexpected TensorRT I/O mode: ") + name);
        if (engine_->getTensorDataType(name) != type) throw std::runtime_error(std::string("unexpected TensorRT data type: ") + name);
        const auto dims = engine_->getTensorShape(name);
        if (dims.nbDims != expected.nbDims) throw std::runtime_error(std::string("unexpected TensorRT rank: ") + name);
        for (int i = 0; i < dims.nbDims; ++i) if (dims.d[i] != expected.d[i]) throw std::runtime_error(std::string("unexpected TensorRT shape: ") + name);
    }
    Logger logger_;
    TrtPtr<nvinfer1::IRuntime> runtime_;
    TrtPtr<nvinfer1::ICudaEngine> engine_;
    TrtPtr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_{};
    void *images_{}, *r_{}, *t_{}, *k_{}, *poses_{}, *valid_{}, *scores_{};
    void *source_ptrs_{}, *strides_{}, *widths_{}, *heights_{}, *source_k_{}, *target_k_{}, *distortion_{};
    std::array<float, 27> calibration_r_{}, calibration_k_{};
    std::array<float, 15> calibration_distortion_{};
    std::array<float, 9> calibration_t_{};
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
