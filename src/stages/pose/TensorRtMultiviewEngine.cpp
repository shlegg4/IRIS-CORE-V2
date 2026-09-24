#include "iris/stages/pose/TensorRtMultiviewEngine.hpp"
#include "iris/stages/pose/MultiviewCudaPostprocess.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef IRIS_HAS_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#endif

#ifdef IRIS_HAS_TENSORRT
extern "C" cudaError_t iris_multiview_preprocess(
    const void* const*, const std::size_t*, const std::uint32_t*, const std::uint32_t*,
    const float*, const float*, const float*, float*, cudaStream_t, std::size_t,
    const std::uint8_t**, std::size_t*, std::uint32_t*, std::uint32_t*,
    float*, float*, float*);
#endif

namespace iris {
namespace {
constexpr std::size_t max_views = 10;
constexpr std::size_t candidates_per_view = 10;
constexpr std::size_t joints_per_person = 17;
constexpr std::size_t input_channels = 3;
constexpr std::size_t model_extent = 640;
constexpr std::size_t image_values = input_channels * model_extent * model_extent;
constexpr std::uint64_t cross_view_recovery_interval_frames = 10;

#ifdef IRIS_HAS_TENSORRT
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

template <typename T> struct TrtDeleter {
    void operator()(T* value) const {
        if (!value) return;
#if NV_TENSORRT_MAJOR >= 10
        delete value;
#else
        value->destroy();
#endif
    }
};
template <typename T> using TrtPtr = std::unique_ptr<T, TrtDeleter<T>>;

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
#endif
} // namespace

class TensorRtMultiviewEngine::Impl {
  public:
    explicit Impl(const PoseConfig& config) {
#ifdef IRIS_HAS_TENSORRT
        if (config.multiview_calibration.empty() || config.multiview_calibration.size() > max_views)
            throw std::invalid_argument("RTMO inference supports between 1 and 10 views");
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

        validate("images", nvinfer1::Dims4{3, 3, 640, 640}, nvinfer1::TensorIOMode::kINPUT, nvinfer1::DataType::kFLOAT);
        validate("keypoints", nvinfer1::Dims4{3, 10, 17, 2}, nvinfer1::TensorIOMode::kOUTPUT, nvinfer1::DataType::kFLOAT);
        validate("keypoint_scores", nvinfer1::Dims3{3, 10, 17}, nvinfer1::TensorIOMode::kOUTPUT, nvinfer1::DataType::kFLOAT);
        validate("instance_scores", nvinfer1::Dims2{3, 10}, nvinfer1::TensorIOMode::kOUTPUT, nvinfer1::DataType::kFLOAT);
        validate("boxes", nvinfer1::Dims3{3, 10, 4}, nvinfer1::TensorIOMode::kOUTPUT, nvinfer1::DataType::kFLOAT);
        validate("candidate_valid", nvinfer1::Dims2{3, 10}, nvinfer1::TensorIOMode::kOUTPUT, nvinfer1::DataType::kBOOL);

        check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
        for (auto& event : timing_events_)
            check_cuda(cudaEventCreate(&event.value), "create pose timing event");

        check_cuda(cudaMalloc(&images_, sizeof(float) * max_views * image_values), "cudaMalloc images");
        check_cuda(cudaMalloc(&keypoints_, sizeof(float) * max_views * candidates_per_view * joints_per_person * 2), "cudaMalloc keypoints");
        check_cuda(cudaMalloc(&keypoint_scores_, sizeof(float) * max_views * candidates_per_view * joints_per_person), "cudaMalloc keypoint scores");
        check_cuda(cudaMalloc(&instance_scores_, sizeof(float) * max_views * candidates_per_view), "cudaMalloc instance scores");
        check_cuda(cudaMalloc(&boxes_, sizeof(float) * max_views * candidates_per_view * 4), "cudaMalloc boxes");
        check_cuda(cudaMalloc(&candidate_valid_, sizeof(unsigned char) * max_views * candidates_per_view), "cudaMalloc candidate validity");

        check_cuda(cudaMalloc(&source_ptrs_, sizeof(void*) * max_views), "cudaMalloc source pointers");
        check_cuda(cudaMalloc(&strides_, sizeof(std::size_t) * max_views), "cudaMalloc strides");
        check_cuda(cudaMalloc(&widths_, sizeof(std::uint32_t) * max_views), "cudaMalloc widths");
        check_cuda(cudaMalloc(&heights_, sizeof(std::uint32_t) * max_views), "cudaMalloc heights");
        check_cuda(cudaMalloc(&source_k_, sizeof(float) * max_views * 9), "cudaMalloc source intrinsics");
        check_cuda(cudaMalloc(&target_k_, sizeof(float) * max_views * 9), "cudaMalloc target intrinsics");
        check_cuda(cudaMalloc(&distortion_, sizeof(float) * max_views * 5), "cudaMalloc distortion");
        check_cuda(cudaMalloc(&mapping_meta_, sizeof(float) * max_views * 3), "cudaMalloc keypoint mapping metadata");
        check_cuda(cudaMalloc(&image_keypoints_, sizeof(float) * max_views * candidates_per_view * joints_per_person * 2), "cudaMalloc mapped keypoints");
        check_cuda(cudaMalloc(&image_joint_valid_, sizeof(unsigned char) * max_views * candidates_per_view * joints_per_person), "cudaMalloc mapped keypoint validity");

        max_persons_ = config.max_persons;
        tracking_enabled_ = !config.two_d_only;
        epipolar_gate_px_ = config.epipolar_gate_px;
        minimum_joint_confidence_ = config.minimum_joint_confidence;
        std::vector<std::uint32_t> camera_ids;
        camera_ids.reserve(config.multiview_calibration.size());
        for (const auto& camera : config.multiview_calibration)
            camera_ids.push_back(static_cast<std::uint32_t>(camera.camera_id));
        check_cuda(create_multiview_association_workspace(
            camera_ids.size(), max_persons_, camera_ids.data(),
            &association_workspace_, stream_), "create CUDA association workspace");
        if (tracking_enabled_)
            check_cuda(create_multiview_temporal_workspace(camera_ids.size(), max_persons_,
                &temporal_workspace_), "create CUDA temporal tracking workspace");

        bind_io(config.multiview_calibration.size());
        initialize_cuda_graph();

        calibration_k_.resize(config.multiview_calibration.size() * 9);
        calibration_distortion_.resize(config.multiview_calibration.size() * 5);
        for (std::size_t view = 0; view < config.multiview_calibration.size(); ++view) {
            const auto& calibration = config.multiview_calibration[view];
            std::copy(calibration.intrinsics.begin(), calibration.intrinsics.end(), calibration_k_.begin() + view * 9);
            std::copy(calibration.distortion.begin(), calibration.distortion.end(), calibration_distortion_.begin() + view * 5);
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
        destroy_multiview_association_workspace(association_workspace_);
        destroy_multiview_temporal_workspace(temporal_workspace_);
        cudaFree(images_);
        cudaFree(keypoints_); cudaFree(keypoint_scores_); cudaFree(instance_scores_); cudaFree(boxes_); cudaFree(candidate_valid_);
        cudaFree(source_ptrs_); cudaFree(strides_); cudaFree(widths_); cudaFree(heights_);
        cudaFree(source_k_); cudaFree(target_k_); cudaFree(distortion_);
        cudaFree(mapping_meta_); cudaFree(image_keypoints_); cudaFree(image_joint_valid_);
        if (stream_) cudaStreamDestroy(stream_);
#endif
    }

    void infer(const std::vector<const void*>& sources,
               const std::vector<std::size_t>& strides,
               const std::vector<std::uint32_t>& widths,
               const std::vector<std::uint32_t>& heights,
               const std::vector<float>& fundamentals,
               const std::vector<float>& projections,
               float dt_seconds, float maximum_reprojection_error_px,
               TensorRtMultiviewResult& result) {
#ifndef IRIS_HAS_TENSORRT
        (void)sources; (void)strides; (void)widths; (void)heights; (void)fundamentals;
        (void)projections; (void)dt_seconds; (void)maximum_reprojection_error_px; (void)result;
        throw std::runtime_error("TensorRT support was not compiled into IRIS");
#else
        const auto view_count = sources.size();
        if (view_count == 0 || view_count > max_views || strides.size() != view_count ||
            widths.size() != view_count || heights.size() != view_count ||
            calibration_k_.size() != view_count * 9 || calibration_distortion_.size() != view_count * 5 ||
            fundamentals.size() != view_count * view_count * 9 ||
            projections.size() != view_count * 12)
            throw std::invalid_argument("RTMO inference input arrays must contain the same 1..10 views");

        result.timings.setup_host_ms = 0.0;

        std::vector<float> target_k(view_count * 9, 0.0F);
        for (std::size_t view = 0; view < view_count; ++view) {
            const float scale = std::min(640.0F / static_cast<float>(widths[view]),
                                         640.0F / static_cast<float>(heights[view]));
            const auto resized_width = std::max(1, static_cast<int>(widths[view] * scale + 0.5F));
            const auto resized_height = std::max(1, static_cast<int>(heights[view] * scale + 0.5F));
            const auto pad_x = static_cast<float>((640 - resized_width) / 2);
            const auto pad_y = static_cast<float>((640 - resized_height) / 2);
            target_k[view * 9] = calibration_k_[view * 9] * scale;
            target_k[view * 9 + 4] = calibration_k_[view * 9 + 4] * scale;
            target_k[view * 9 + 2] = calibration_k_[view * 9 + 2] * scale + pad_x;
            target_k[view * 9 + 5] = calibration_k_[view * 9 + 5] * scale + pad_y;
            target_k[view * 9 + 8] = 1.0F;
        }

        check_cuda(cudaEventRecord(timing_events_[0].value, stream_), "time preprocessing begin");
        const auto preprocess_start = TimingClock::now();
        check_cuda(iris_multiview_preprocess(
                       sources.data(), strides.data(), widths.data(), heights.data(),
                       calibration_k_.data(), target_k.data(), calibration_distortion_.data(),
                       static_cast<float*>(images_), stream_, view_count,
                       static_cast<const std::uint8_t**>(source_ptrs_), static_cast<std::size_t*>(strides_),
                       static_cast<std::uint32_t*>(widths_), static_cast<std::uint32_t*>(heights_),
                       static_cast<float*>(source_k_), static_cast<float*>(target_k_), static_cast<float*>(distortion_)),
                   "multiview preprocessing");
        check_cuda(cudaEventRecord(timing_events_[1].value, stream_), "time preprocessing end");
        result.timings.preprocess_host_ms = elapsed_ms(preprocess_start);

        check_cuda(cudaEventRecord(timing_events_[2].value, stream_), "time engine begin");
        const auto enqueue_start = TimingClock::now();
        result.timings.cuda_graph_active = graph_exec_ != nullptr;
        if (graph_exec_) {
            check_cuda(cudaGraphLaunch(graph_exec_, stream_), "launch TensorRT CUDA graph");
        } else if (!context_->enqueueV3(stream_)) {
            throw std::runtime_error("TensorRT enqueueV3 failed: " + logger_.last_error);
        }
        result.timings.enqueue_host_ms = elapsed_ms(enqueue_start);
        check_cuda(cudaEventRecord(timing_events_[3].value, stream_), "time engine end");

        check_cuda(cudaEventRecord(timing_events_[10].value, stream_), "time keypoint mapping begin");
        std::vector<float> mapping_meta(view_count * 3);
        for (std::size_t view = 0; view < view_count; ++view) {
            const float scale = std::min(640.0F / static_cast<float>(widths[view]),
                                         640.0F / static_cast<float>(heights[view]));
            const int resized_width = std::max(1, static_cast<int>(widths[view] * scale + 0.5F));
            const int resized_height = std::max(1, static_cast<int>(heights[view] * scale + 0.5F));
            mapping_meta[view * 3] = scale;
            mapping_meta[view * 3 + 1] = static_cast<float>((640 - resized_width) / 2);
            mapping_meta[view * 3 + 2] = static_cast<float>((640 - resized_height) / 2);
        }
        check_cuda(cudaMemcpyAsync(mapping_meta_, mapping_meta.data(), sizeof(float) * mapping_meta.size(),
            cudaMemcpyHostToDevice, stream_), "upload keypoint mapping metadata");
        const auto candidate_count = view_count * candidates_per_view;
        const auto keypoint_count = candidate_count * joints_per_person;
        check_cuda(launch_multiview_unwarp_keypoints(static_cast<const float*>(keypoints_),
            static_cast<const float*>(source_k_), static_cast<const float*>(distortion_),
            static_cast<const float*>(mapping_meta_), view_count,
            static_cast<float*>(image_keypoints_), static_cast<unsigned char*>(image_joint_valid_), stream_),
            "launch CUDA keypoint coordinate mapping");
        check_cuda(cudaEventRecord(timing_events_[11].value, stream_), "time keypoint mapping end");

        result.assignments.resize(max_persons_ * view_count);
        result.track_count = 0;
        check_cuda(cudaEventRecord(timing_events_[6].value, stream_), "time association begin");
        const bool seed_frame = tracking_enabled_ &&
            (frame_index_++ % cross_view_recovery_interval_frames == 0);
        if (seed_frame) {
            check_cuda(launch_multiview_current_association(
                association_workspace_, static_cast<const float*>(keypoints_),
                static_cast<const float*>(keypoint_scores_),
                static_cast<const unsigned char*>(candidate_valid_), fundamentals.data(),
                epipolar_gate_px_, minimum_joint_confidence_, stream_,
                &result.timings.association_host_ms),
                "launch current CUDA association");
        } else {
            check_cuda(clear_multiview_association_assignments(association_workspace_, stream_),
                "clear CUDA association seed assignments");
        }
        check_cuda(cudaEventRecord(timing_events_[7].value, stream_), "time association end");

        check_cuda(cudaEventRecord(timing_events_[8].value, stream_), "time temporal update begin");
        if (tracking_enabled_) {
            check_cuda(launch_multiview_temporal_update(temporal_workspace_,
                static_cast<const float*>(keypoints_), static_cast<const float*>(keypoint_scores_),
                static_cast<const unsigned char*>(candidate_valid_),
                multiview_association_assignments_device(association_workspace_),
                projections.data(), dt_seconds, epipolar_gate_px_, minimum_joint_confidence_,
                maximum_reprojection_error_px, timing_events_[12].value, timing_events_[13].value, stream_),
                "launch CUDA temporal tracking update");
        }
        check_cuda(cudaEventRecord(timing_events_[9].value, stream_), "time temporal update end");

        result.image_keypoints.resize(keypoint_count * 2);
        result.image_joint_valid.resize(keypoint_count);
        result.keypoint_scores.resize(keypoint_count);
        result.candidate_valid.resize(candidate_count);
        check_cuda(cudaEventRecord(timing_events_[4].value, stream_), "time output copies begin");
        const auto copy_start = TimingClock::now();
        check_cuda(cudaMemcpyAsync(result.image_keypoints.data(), image_keypoints_, sizeof(float) * result.image_keypoints.size(), cudaMemcpyDeviceToHost, stream_), "copy mapped keypoints");
        check_cuda(cudaMemcpyAsync(result.image_joint_valid.data(), image_joint_valid_, sizeof(unsigned char) * result.image_joint_valid.size(), cudaMemcpyDeviceToHost, stream_), "copy mapped keypoint validity");
        check_cuda(cudaMemcpyAsync(result.keypoint_scores.data(), keypoint_scores_, sizeof(float) * result.keypoint_scores.size(), cudaMemcpyDeviceToHost, stream_), "copy raw keypoint scores");
        check_cuda(cudaMemcpyAsync(result.candidate_valid.data(), candidate_valid_, sizeof(unsigned char) * result.candidate_valid.size(), cudaMemcpyDeviceToHost, stream_), "copy candidate validity");
        check_cuda(cudaEventRecord(timing_events_[5].value, stream_), "time output copies end");
        result.timings.download_host_ms = elapsed_ms(copy_start);

        const auto wait_start = TimingClock::now();
        check_cuda(cudaStreamSynchronize(stream_), "synchronize TensorRT inference");
        result.timings.wait_host_ms = elapsed_ms(wait_start);
        if (tracking_enabled_) {
            result.track_ids.resize(max_persons_);
            result.tracked_xyz.resize(max_persons_ * joints_per_person * 3);
            result.tracked_valid.resize(max_persons_ * joints_per_person);
            result.tracked_predicted.resize(max_persons_ * joints_per_person);
            const auto temporal_copy_start = TimingClock::now();
            check_cuda(copy_multiview_temporal_result(temporal_workspace_, result.assignments.data(),
                result.track_ids.data(), result.tracked_xyz.data(), result.tracked_valid.data(),
                result.tracked_predicted.data(),
                &result.track_count, stream_), "copy CUDA temporal tracking result");
            result.timings.temporal_host_ms = elapsed_ms(temporal_copy_start);
        } else {
            result.assignments.assign(max_persons_ * view_count, 255);
            result.track_count = 0;
        }
        const auto interval = [&](int first, int last) {
            float value{};
            check_cuda(cudaEventElapsedTime(&value, timing_events_[first].value, timing_events_[last].value), "read pose timing");
            return static_cast<double>(value);
        };
        result.timings.preprocess_stream_ms = interval(0, 1);
        result.timings.engine_stream_ms = interval(2, 3);
        result.timings.association_stream_ms = interval(6, 7);
        result.timings.temporal_stream_ms = interval(8, 9);
        result.timings.temporal_assignment_stream_ms = interval(12, 13);
        result.timings.mapping_stream_ms = interval(10, 11);
        result.timings.postprocess_gpu_stream_ms = interval(10, 9);
        result.timings.output_copy_stream_ms = interval(4, 5);
        result.timings.download_stream_ms = interval(3, 5);
#endif
    }

#ifdef IRIS_HAS_TENSORRT
    void bind_io(std::size_t view_count) {
        if (!context_->setInputShape("images", nvinfer1::Dims4{static_cast<int>(view_count), 3, 640, 640}))
            throw std::runtime_error("TensorRT rejected the requested RTMO batch shape");
        if (!context_->setTensorAddress("images", images_) ||
            !context_->setTensorAddress("keypoints", keypoints_) ||
            !context_->setTensorAddress("keypoint_scores", keypoint_scores_) ||
            !context_->setTensorAddress("instance_scores", instance_scores_) ||
            !context_->setTensorAddress("boxes", boxes_) ||
            !context_->setTensorAddress("candidate_valid", candidate_valid_))
            throw std::runtime_error("TensorRT rejected one or more tensor addresses");
    }

    void initialize_cuda_graph() {
        // Warm up first so TensorRT can perform any lazy initialization before capture.
        if (!context_->enqueueV3(stream_))
            throw std::runtime_error("TensorRT warmup enqueue failed: " + logger_.last_error);
        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            cudaGetLastError();
            return;
        }

        cudaError_t error = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
        if (error != cudaSuccess) {
            cudaGetLastError();
            return;
        }

        error = context_->enqueueV3(stream_) ? cudaSuccess : cudaErrorUnknown;
        const cudaError_t end_error = cudaStreamEndCapture(stream_, &graph_);
        if (error == cudaSuccess) error = end_error;
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
        }
    }
#endif

#ifdef IRIS_HAS_TENSORRT
    void validate(const char* name, nvinfer1::Dims expected,
                  nvinfer1::TensorIOMode mode, nvinfer1::DataType type) {
        if (engine_->getTensorIOMode(name) != mode)
            throw std::runtime_error(std::string("unexpected TensorRT I/O mode: ") + name);
        if (engine_->getTensorDataType(name) != type)
            throw std::runtime_error(std::string("unexpected TensorRT data type: ") + name);
        const auto dims = engine_->getTensorShape(name);
        if (dims.nbDims != expected.nbDims)
            throw std::runtime_error(std::string("unexpected TensorRT rank: ") + name);
        for (int i = 0; i < dims.nbDims; ++i) {
            if (dims.d[i] != -1 && dims.d[i] != expected.d[i])
                throw std::runtime_error(std::string("unexpected TensorRT shape: ") + name);
        }
    }

    Logger logger_;
    TrtPtr<nvinfer1::IRuntime> runtime_;
    TrtPtr<nvinfer1::ICudaEngine> engine_;
    TrtPtr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_{};
        std::array<TimingEvent, 14> timing_events_;
    cudaGraph_t graph_{};
    cudaGraphExec_t graph_exec_{};
    MultiviewAssociationWorkspace* association_workspace_{};
    MultiviewTemporalWorkspace* temporal_workspace_{};
    bool tracking_enabled_{};
    std::uint64_t frame_index_{};
    std::size_t max_persons_{};
    float epipolar_gate_px_{12.0F};
    float minimum_joint_confidence_{0.1F};
    void *images_{}, *keypoints_{}, *keypoint_scores_{}, *instance_scores_{}, *boxes_{}, *candidate_valid_{};
    void *source_ptrs_{}, *strides_{}, *widths_{}, *heights_{}, *source_k_{}, *target_k_{}, *distortion_{};
    void *mapping_meta_{}, *image_keypoints_{}, *image_joint_valid_{};
    std::vector<float> calibration_k_;
    std::vector<float> calibration_distortion_;
#endif
};

TensorRtMultiviewEngine::TensorRtMultiviewEngine(const PoseConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}
TensorRtMultiviewEngine::~TensorRtMultiviewEngine() = default;

void TensorRtMultiviewEngine::infer(const std::vector<const void*>& bgr_device,
    const std::vector<std::size_t>& strides,
    const std::vector<std::uint32_t>& widths,
    const std::vector<std::uint32_t>& heights,
    const std::vector<float>& fundamentals,
    const std::vector<float>& projections,
    float dt_seconds, float maximum_reprojection_error_px,
    TensorRtMultiviewResult& result) {
    impl_->infer(bgr_device, strides, widths, heights, fundamentals, projections,
                 dt_seconds, maximum_reprojection_error_px, result);
}
} // namespace iris
