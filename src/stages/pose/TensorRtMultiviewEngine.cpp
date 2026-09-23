#include "iris/stages/pose/TensorRtMultiviewEngine.hpp"
#include "iris/stages/pose/MultiviewCudaPostprocess.hpp"

#include <stdexcept>
#include <algorithm>
#include <utility>
#include <chrono>

#ifdef IRIS_HAS_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
extern "C" cudaError_t iris_multiview_preprocess(const void* const*, const std::size_t*, const std::uint32_t*, const std::uint32_t*, const float*, const float*, const float*, float*, cudaStream_t, const std::uint8_t**, std::size_t*, std::uint32_t*, std::uint32_t*, float*, float*, float*);

namespace {
using TimingClock = std::chrono::steady_clock;
double elapsed_ms(TimingClock::time_point start) {
    return std::chrono::duration<double, std::milli>(TimingClock::now() - start).count();
}
struct TimingEvent {
    cudaEvent_t value{};
    ~TimingEvent() { if (value) cudaEventDestroy(value); }
};
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
        minimum_score_ = config.minimum_joint_confidence;
        maximum_reprojection_error_ = config.maximum_reprojection_error_px;
        config_gate_px_ = config.epipolar_gate_px;
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
        for (auto& event : timing_events_)
            check_cuda(cudaEventCreate(&event.value), "create pose timing event");
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
        check_cuda(cudaMalloc(&projections_, sizeof(float) * 36), "cudaMalloc projections");
        check_cuda(cudaMalloc(&triangulated_xyz_, sizeof(float) * 10 * 17 * 3), "cudaMalloc triangulated xyz");
        check_cuda(cudaMalloc(&triangulated_valid_, sizeof(unsigned char) * 10 * 17), "cudaMalloc triangulated valid");
        check_cuda(cudaMalloc(&fundamentals_, sizeof(float) * 27), "cudaMalloc fundamentals");
        check_cuda(cudaMalloc(&assignments_, sizeof(unsigned char) * 30), "cudaMalloc assignments");
        check_cuda(cudaMalloc(&selected_keypoints_, sizeof(float) * 10 * 3 * 17 * 2), "cudaMalloc selected keypoints");
        check_cuda(cudaMalloc(&selected_scores_, sizeof(float) * 10 * 3 * 17), "cudaMalloc selected scores");
        check_cuda(cudaMalloc(&selected_valid_, sizeof(unsigned char) * 10 * 3 * 17), "cudaMalloc selected valid");
        bind_fixed_io();
        initialize_cuda_graph();
        for (std::size_t i = 0; i < 3; ++i) {
            std::copy(config.multiview_calibration[i].intrinsics.begin(), config.multiview_calibration[i].intrinsics.end(), calibration_k_.begin() + i * 9);
            std::copy(config.multiview_calibration[i].distortion.begin(), config.multiview_calibration[i].distortion.end(), calibration_distortion_.begin() + i * 5);
            calibration_[i] = config.multiview_calibration[i];
        }
#else
        (void)config;
        throw std::runtime_error("TensorRT support was not compiled into IRIS");
#endif
    }
    ~Impl() {
#ifdef IRIS_HAS_TENSORRT
        if (stream_) cudaStreamSynchronize(stream_);
        if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
        if (graph_) cudaGraphDestroy(graph_);
        cudaFree(images_);
        cudaFree(keypoints_); cudaFree(keypoint_scores_); cudaFree(instance_scores_); cudaFree(boxes_); cudaFree(candidate_valid_);
        cudaFree(source_ptrs_); cudaFree(strides_); cudaFree(widths_); cudaFree(heights_);
        cudaFree(source_k_); cudaFree(target_k_); cudaFree(distortion_);
        cudaFree(projections_); cudaFree(triangulated_xyz_); cudaFree(triangulated_valid_);
        cudaFree(fundamentals_); cudaFree(assignments_);
        cudaFree(selected_keypoints_); cudaFree(selected_scores_); cudaFree(selected_valid_);
        if (stream_) cudaStreamDestroy(stream_);
#endif
    }
    void infer(const std::array<const void*, 3>& sources, const std::array<std::size_t, 3>& strides,
               const std::array<std::uint32_t, 3>& widths, const std::array<std::uint32_t, 3>& heights,
               TensorRtMultiviewResult& result) {
#ifndef IRIS_HAS_TENSORRT
        throw std::runtime_error("TensorRT support was not compiled into IRIS");
#else
        const auto preprocess_start = TimingClock::now();
        check_cuda(cudaEventRecord(timing_events_[0].value, stream_), "time preprocessing begin");
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
        check_cuda(cudaEventRecord(timing_events_[1].value, stream_), "time preprocessing end");
        result.timings.preprocess_host_ms = elapsed_ms(preprocess_start);
        const auto setup_start = TimingClock::now();
        // Shape and tensor addresses are fixed during construction so they remain
        // invariant when the TensorRT CUDA graph is replayed.
        result.timings.setup_host_ms = elapsed_ms(setup_start);
        check_cuda(cudaEventRecord(timing_events_[2].value, stream_), "time engine begin");
        const auto enqueue_start = TimingClock::now();
        if (graph_ready_) {
            check_cuda(cudaGraphLaunch(graph_exec_, stream_), "launch TensorRT CUDA graph");
        } else if (!context_->enqueueV3(stream_)) {
            throw std::runtime_error("TensorRT enqueueV3 failed: " + logger_.last_error);
        }
        result.timings.enqueue_host_ms = elapsed_ms(enqueue_start);
        check_cuda(cudaEventRecord(timing_events_[3].value, stream_), "time engine end");
        const auto geometry_setup_start = TimingClock::now();
        std::array<float, 36> projections{};
        for (std::size_t view = 0; view < 3; ++view) {
            const auto& camera = calibration_[view];
            const auto* k = target_k.data() + view * 9;
            for (int row = 0; row < 3; ++row) for (int col = 0; col < 4; ++col) {
                float value = 0.0F;
                for (int inner = 0; inner < 3; ++inner)
                    value += k[row * 3 + inner] * (col < 3 ? camera.R_w2c[inner * 3 + col] : camera.t_w2c[inner]);
                projections[view * 12 + row * 4 + col] = value;
            }
        }
        check_cuda(cudaMemcpyAsync(projections_, projections.data(), sizeof(projections), cudaMemcpyHostToDevice, stream_), "copy projections");
        std::array<float, 27> fundamentals{};
        constexpr int pair_first[3]{0, 0, 1};
        constexpr int pair_second[3]{1, 2, 2};
        for (int pair = 0; pair < 3; ++pair) {
            const int first_view = pair_first[pair], second_view = pair_second[pair];
            const auto& first = calibration_[first_view]; const auto& second = calibration_[second_view];
            Eigen::Matrix3f r0, r1; Eigen::Vector3f t0, t1;
            for (int row = 0; row < 3; ++row) for (int col = 0; col < 3; ++col) {
                r0(row, col) = first.R_w2c[row * 3 + col]; r1(row, col) = second.R_w2c[row * 3 + col];
            }
            for (int i = 0; i < 3; ++i) { t0(i) = first.t_w2c[i]; t1(i) = second.t_w2c[i]; }
            const auto relative_r = r1 * r0.transpose(); const auto relative_t = t1 - relative_r * t0;
            Eigen::Matrix3f skew;
            skew << 0.0F, -relative_t.z(), relative_t.y(), relative_t.z(), 0.0F, -relative_t.x(), -relative_t.y(), relative_t.x(), 0.0F;
            const auto k0 = Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(target_k.data() + first_view * 9);
            const auto k1 = Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(target_k.data() + second_view * 9);
            const auto f = k1.inverse().transpose() * skew * relative_r * k0.inverse();
            for (int row = 0; row < 3; ++row) for (int col = 0; col < 3; ++col)
                fundamentals[pair * 9 + row * 3 + col] = f(row, col);
        }
        check_cuda(cudaMemcpyAsync(fundamentals_, fundamentals.data(), sizeof(fundamentals), cudaMemcpyHostToDevice, stream_), "copy epipolar fundamentals");
        result.timings.geometry_setup_host_ms = elapsed_ms(geometry_setup_start);
        check_cuda(cudaEventRecord(timing_events_[4].value, stream_), "time association begin");
        check_cuda(launch_multiview_epipolar_assignment(static_cast<const float*>(keypoints_), static_cast<const float*>(keypoint_scores_), static_cast<const unsigned char*>(candidate_valid_), static_cast<const float*>(fundamentals_), config_gate_px_, minimum_score_, static_cast<unsigned char*>(assignments_), stream_), "launch epipolar assignment");
        check_cuda(cudaEventRecord(timing_events_[5].value, stream_), "time association end");
        check_cuda(launch_multiview_gather_selected(static_cast<const float*>(keypoints_), static_cast<const float*>(keypoint_scores_), static_cast<const unsigned char*>(candidate_valid_), static_cast<const unsigned char*>(assignments_), static_cast<float*>(selected_keypoints_), static_cast<float*>(selected_scores_), static_cast<unsigned char*>(selected_valid_), stream_), "gather selected observations");
        check_cuda(cudaEventRecord(timing_events_[6].value, stream_), "time gather end");
        check_cuda(launch_multiview_weighted_dlt(static_cast<const float*>(keypoints_), static_cast<const float*>(keypoint_scores_), static_cast<const unsigned char*>(candidate_valid_), static_cast<const unsigned char*>(assignments_), static_cast<const float*>(projections_), minimum_score_, maximum_reprojection_error_, static_cast<float*>(triangulated_xyz_), static_cast<unsigned char*>(triangulated_valid_), stream_), "launch GPU triangulation");
        check_cuda(cudaEventRecord(timing_events_[7].value, stream_), "time triangulation end");
        const auto download_start = TimingClock::now();
        check_cuda(cudaMemcpyAsync(result.selected_keypoints.data(), selected_keypoints_, sizeof(float) * result.selected_keypoints.size(), cudaMemcpyDeviceToHost, stream_), "copy selected keypoints");
        check_cuda(cudaMemcpyAsync(result.selected_scores.data(), selected_scores_, sizeof(float) * result.selected_scores.size(), cudaMemcpyDeviceToHost, stream_), "copy selected scores");
        check_cuda(cudaMemcpyAsync(result.selected_valid.data(), selected_valid_, sizeof(unsigned char) * result.selected_valid.size(), cudaMemcpyDeviceToHost, stream_), "copy selected validity");
        check_cuda(cudaMemcpyAsync(result.keypoints.data(), keypoints_, sizeof(float) * result.keypoints.size(), cudaMemcpyDeviceToHost, stream_), "copy raw keypoints");
        check_cuda(cudaMemcpyAsync(result.keypoint_scores.data(), keypoint_scores_, sizeof(float) * result.keypoint_scores.size(), cudaMemcpyDeviceToHost, stream_), "copy raw keypoint scores");
        check_cuda(cudaMemcpyAsync(result.candidate_valid.data(), candidate_valid_, sizeof(unsigned char) * result.candidate_valid.size(), cudaMemcpyDeviceToHost, stream_), "copy raw candidate validity");
        check_cuda(cudaMemcpyAsync(result.triangulated_xyz.data(), triangulated_xyz_, sizeof(float) * result.triangulated_xyz.size(), cudaMemcpyDeviceToHost, stream_), "copy triangulated xyz");
        check_cuda(cudaMemcpyAsync(result.triangulated_valid.data(), triangulated_valid_, sizeof(unsigned char) * result.triangulated_valid.size(), cudaMemcpyDeviceToHost, stream_), "copy triangulated valid");
        check_cuda(cudaMemcpyAsync(result.assignments.data(), assignments_, sizeof(unsigned char) * result.assignments.size(), cudaMemcpyDeviceToHost, stream_), "copy epipolar assignments");
        check_cuda(cudaEventRecord(timing_events_[8].value, stream_), "time output copies end");
        result.timings.download_host_ms = elapsed_ms(download_start);
        const auto wait_start = TimingClock::now();
        check_cuda(cudaStreamSynchronize(stream_), "synchronize TensorRT inference");
        result.timings.wait_host_ms = elapsed_ms(wait_start);
        const auto interval = [&](int first, int last) {
            float value{};
            check_cuda(cudaEventElapsedTime(&value, timing_events_[first].value, timing_events_[last].value), "read pose timing");
            return static_cast<double>(value);
        };
        result.timings.preprocess_stream_ms = interval(0, 1);
        result.timings.engine_stream_ms = interval(2, 3);
        result.timings.association_stream_ms = interval(4, 5);
        result.timings.gather_stream_ms = interval(5, 6);
        result.timings.triangulation_stream_ms = interval(6, 7);
        result.timings.output_copy_stream_ms = interval(7, 8);
        // Retain the historical aggregate: engine end through geometry uploads,
        // postprocess kernels, and output copies.
        result.timings.download_stream_ms = interval(3, 8);
#endif
    }
#ifdef IRIS_HAS_TENSORRT
    void bind_fixed_io() {
        // The pipeline supplies one image for each of its three calibrated views.
        // This is a no-op for a static engine and selects N=3 for a dynamic one.
        if (!context_->setInputShape("images", nvinfer1::Dims4{3, 3, 640, 640}))
            throw std::runtime_error("TensorRT rejected the RTMO batch shape [3,3,640,640]");
        if (!context_->setTensorAddress("images", images_) ||
            !context_->setTensorAddress("keypoints", keypoints_) ||
            !context_->setTensorAddress("keypoint_scores", keypoint_scores_) ||
            !context_->setTensorAddress("instance_scores", instance_scores_) ||
            !context_->setTensorAddress("boxes", boxes_) ||
            !context_->setTensorAddress("candidate_valid", candidate_valid_))
            throw std::runtime_error("TensorRT rejected one or more tensor addresses");
    }

    void initialize_cuda_graph() {
        // TensorRT graph capture is an optimization. If an engine/plugin rejects
        // capture, retain the ordinary enqueueV3 path.
        if (!context_->enqueueV3(stream_)) {
            throw std::runtime_error("TensorRT warmup enqueue failed: " + logger_.last_error);
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            cudaGetLastError();
            return;
        }

        cudaError_t error = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
        if (error != cudaSuccess)
            return;

        error = context_->enqueueV3(stream_) ? cudaSuccess : cudaErrorUnknown;
        cudaError_t end_error = cudaStreamEndCapture(stream_, &graph_);
        if (error == cudaSuccess)
            error = end_error;
        if (error != cudaSuccess || !graph_) {
            graph_ = nullptr;
            cudaGetLastError();
            return;
        }
        error = cudaGraphInstantiate(&graph_exec_, graph_, 0);
        if (error != cudaSuccess || !graph_exec_) {
            cudaGraphDestroy(graph_);
            graph_ = nullptr;
            graph_exec_ = nullptr;
            cudaGetLastError();
            return;
        }
        graph_ready_ = true;
    }

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
    std::array<TimingEvent, 9> timing_events_;
    cudaGraph_t graph_{};
    cudaGraphExec_t graph_exec_{};
    bool graph_ready_{};
    void *images_{}, *keypoints_{}, *keypoint_scores_{}, *instance_scores_{}, *boxes_{}, *candidate_valid_{};
    void *source_ptrs_{}, *strides_{}, *widths_{}, *heights_{}, *source_k_{}, *target_k_{}, *distortion_{};
    void *projections_{}, *triangulated_xyz_{}, *triangulated_valid_{};
    void *fundamentals_{}, *assignments_{};
    void *selected_keypoints_{}, *selected_scores_{}, *selected_valid_{};
    std::array<float, 27> calibration_k_{};
    std::array<float, 15> calibration_distortion_{};
    std::array<PoseConfig::CameraCalibration, 3> calibration_{};
    float minimum_score_{0.1F};
    float maximum_reprojection_error_{8.0F};
    float config_gate_px_{12.0F};
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
