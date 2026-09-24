#include "da3/tensor_rt.hpp"

#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include "stages/detection/trt_utils.hpp"

namespace da3 {

namespace {

using iris::core::trt_utils::load_engine_file;
using iris::core::trt_utils::TrtDeleter;
using iris::core::trt_utils::TrtLogger;

using ContextPtr =
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter>;
using EnginePtr = std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter>;
using RuntimePtr = std::unique_ptr<nvinfer1::IRuntime, TrtDeleter>;

void CheckCuda(const cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(context) + ": " + cudaGetErrorString(status)
        );
    }
}

nvinfer1::Dims MakeInputDims() {
    nvinfer1::Dims dims{};
    dims.nbDims = 5;
    dims.d[0] = kBatchSize;
    dims.d[1] = kNumViews;
    dims.d[2] = kNumChannels;
    dims.d[3] = kInputHeight;
    dims.d[4] = kInputWidth;
    return dims;
}

nvinfer1::Dims MakeBaseInputDims(const int num_views) {
    nvinfer1::Dims dims{};
    dims.nbDims = 5;
    // Keep batch fixed and vary DA3's view axis: [1, num_views, C, H, W].
    dims.d[0] = kBatchSize;
    dims.d[1] = num_views;
    dims.d[2] = kNumChannels;
    dims.d[3] = kBaseInputHeight;
    dims.d[4] = kBaseInputWidth;
    return dims;
}

std::vector<int64_t> DimsToVector(const nvinfer1::Dims& dims) {
    std::vector<int64_t> values;
    values.reserve(static_cast<std::size_t>(dims.nbDims));
    for (int index = 0; index < dims.nbDims; ++index) {
        values.push_back(static_cast<int64_t>(dims.d[index]));
    }
    return values;
}

std::size_t Volume(const std::vector<int64_t>& dims) {
    if (dims.empty()) {
        throw std::runtime_error("Tensor dims must not be empty.");
    }
    return std::accumulate(
        dims.begin(),
        dims.end(),
        static_cast<std::size_t>(1),
        [](const std::size_t lhs, const int64_t rhs) {
            if (rhs <= 0) {
                throw std::runtime_error("Encountered unresolved or non-positive tensor dimension.");
            }
            return lhs * static_cast<std::size_t>(rhs);
        }
    );
}

void EnsurePluginsRegistered(TrtLogger& logger) {
    if (!initLibNvInferPlugins(&logger, "")) {
        throw std::runtime_error("Failed to initialize TensorRT plugins.");
    }
}

struct Buffer {
    std::string name;
    std::vector<int64_t> dims;
    std::size_t element_count = 0;
    bool is_input = false;
    int binding_index = -1;
    void* device_ptr = nullptr;
    std::vector<float> host_values;
};

TensorOutput ToTensorOutput(const Buffer& buffer) {
    TensorOutput output;
    output.dims = buffer.dims;
    output.values = buffer.host_values;
    return output;
}

}  // namespace

struct TensorRtEngine::Impl {
    TrtLogger logger;
    RuntimePtr runtime;
    EnginePtr engine;
    ContextPtr context;
    cudaStream_t stream = nullptr;
    std::vector<Buffer> buffers;
    std::vector<std::string> tensor_names;
#if NV_TENSORRT_MAJOR < 10
    std::vector<void*> bindings;
#endif

    ~Impl() {
        for (Buffer& buffer : buffers) {
            if (buffer.device_ptr != nullptr) {
                cudaFree(buffer.device_ptr);
                buffer.device_ptr = nullptr;
            }
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
    }
};

TensorRtEngine::TensorRtEngine(const fs::path& engine_path, const bool is_base) : impl_(std::make_unique<Impl>()) {
    if (!fs::exists(engine_path)) {
        throw std::runtime_error("Engine path does not exist: " + engine_path.string());
    }

    EnsurePluginsRegistered(impl_->logger);
    const std::vector<char> engine_bytes = load_engine_file(engine_path.string());

    impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
    if (!impl_->runtime) {
        throw std::runtime_error("Failed to create TensorRT runtime.");
    }

    impl_->engine.reset(
        impl_->runtime->deserializeCudaEngine(engine_bytes.data(), engine_bytes.size())
    );
    if (!impl_->engine) {
        throw std::runtime_error("Failed to deserialize TensorRT engine.");
    }

    impl_->context.reset(impl_->engine->createExecutionContext());
    if (!impl_->context) {
        throw std::runtime_error("Failed to create TensorRT execution context.");
    }

    CheckCuda(cudaStreamCreate(&impl_->stream), "cudaStreamCreate");

#if NV_TENSORRT_MAJOR >= 10
    const nvinfer1::Dims initial_dims = is_base ? MakeBaseInputDims(4) : MakeInputDims();
    if (!impl_->context->setInputShape(kInputTensorName, initial_dims)) {
        throw std::runtime_error("Failed to set TensorRT input shape.");
    }

    const int tensor_count = impl_->engine->getNbIOTensors();
    impl_->buffers.reserve(static_cast<std::size_t>(tensor_count));
    for (int index = 0; index < tensor_count; ++index) {
        const char* name = impl_->engine->getIOTensorName(index);
        Buffer buffer;
        buffer.name = name;
        buffer.is_input = impl_->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        buffer.dims = DimsToVector(impl_->context->getTensorShape(name));
        buffer.element_count = Volume(buffer.dims);
        buffer.host_values.resize(buffer.element_count);
        impl_->tensor_names.push_back(buffer.name);

        if (impl_->engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("Only float32 TensorRT IO tensors are supported in v1.");
        }

        CheckCuda(
            cudaMalloc(&buffer.device_ptr, buffer.element_count * sizeof(float)),
            ("cudaMalloc(" + buffer.name + ")").c_str()
        );
        if (!impl_->context->setTensorAddress(name, buffer.device_ptr)) {
            throw std::runtime_error("Failed to bind TensorRT tensor address for " + buffer.name);
        }
        impl_->buffers.push_back(std::move(buffer));
    }
#else
    const int input_index = impl_->engine->getBindingIndex(kInputTensorName);
    if (input_index < 0) {
        throw std::runtime_error("TensorRT engine is missing the images input binding.");
    }
    const nvinfer1::Dims initial_dims_legacy = is_base ? MakeBaseInputDims(4) : MakeInputDims();
    if (!impl_->context->setBindingDimensions(input_index, initial_dims_legacy)) {
        throw std::runtime_error("Failed to set TensorRT binding dimensions.");
    }
    if (!impl_->context->allInputDimensionsSpecified()) {
        throw std::runtime_error("TensorRT input dimensions are not fully specified.");
    }

    const int binding_count = impl_->engine->getNbBindings();
    impl_->bindings.resize(static_cast<std::size_t>(binding_count), nullptr);
    impl_->buffers.reserve(static_cast<std::size_t>(binding_count));
    for (int index = 0; index < binding_count; ++index) {
        Buffer buffer;
        buffer.binding_index = index;
        buffer.name = impl_->engine->getBindingName(index);
        buffer.is_input = impl_->engine->bindingIsInput(index);
        buffer.dims = DimsToVector(impl_->context->getBindingDimensions(index));
        buffer.element_count = Volume(buffer.dims);
        buffer.host_values.resize(buffer.element_count);
        impl_->tensor_names.push_back(buffer.name);

        if (impl_->engine->getBindingDataType(index) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("Only float32 TensorRT IO tensors are supported in v1.");
        }

        CheckCuda(
            cudaMalloc(&buffer.device_ptr, buffer.element_count * sizeof(float)),
            ("cudaMalloc(" + buffer.name + ")").c_str()
        );
        impl_->bindings[static_cast<std::size_t>(index)] = buffer.device_ptr;
        impl_->buffers.push_back(std::move(buffer));
    }
#endif
}

TensorRtEngine::~TensorRtEngine() = default;
TensorRtEngine::TensorRtEngine(TensorRtEngine&&) noexcept = default;
TensorRtEngine& TensorRtEngine::operator=(TensorRtEngine&&) noexcept = default;

Mv4Outputs TensorRtEngine::Infer(const std::vector<float>& images) {
    auto find_buffer = [&](const std::string& name) -> Buffer& {
        for (Buffer& buffer : impl_->buffers) {
            if (buffer.name == name) {
                return buffer;
            }
        }
        throw std::runtime_error("TensorRT engine is missing tensor: " + name);
    };

    Buffer& input = find_buffer(kInputTensorName);
    if (images.size() != input.element_count) {
        std::ostringstream oss;
        oss << "Expected " << input.element_count << " float values for TensorRT input, got "
            << images.size() << ".";
        throw std::runtime_error(oss.str());
    }

    CheckCuda(
        cudaMemcpyAsync(
            input.device_ptr,
            images.data(),
            images.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            impl_->stream
        ),
        "cudaMemcpyAsync(H2D images)"
    );

#if NV_TENSORRT_MAJOR >= 10
    if (!impl_->context->enqueueV3(impl_->stream)) {
        throw std::runtime_error("TensorRT enqueueV3 failed.");
    }
#else
    if (!impl_->context->enqueueV2(impl_->bindings.data(), impl_->stream, nullptr)) {
        throw std::runtime_error("TensorRT enqueueV2 failed.");
    }
#endif

    for (Buffer& buffer : impl_->buffers) {
        if (buffer.is_input) {
            continue;
        }
        CheckCuda(
            cudaMemcpyAsync(
                buffer.host_values.data(),
                buffer.device_ptr,
                buffer.host_values.size() * sizeof(float),
                cudaMemcpyDeviceToHost,
                impl_->stream
            ),
            ("cudaMemcpyAsync(D2H " + buffer.name + ")").c_str()
        );
    }

    CheckCuda(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize");

    Mv4Outputs outputs;
    outputs.depth = ToTensorOutput(find_buffer("depth"));
    outputs.depth_conf = ToTensorOutput(find_buffer("depth_conf"));
    outputs.sky = ToTensorOutput(find_buffer("sky"));
    outputs.intrinsics = ToTensorOutput(find_buffer("intrinsics"));
    outputs.extrinsics = ToTensorOutput(find_buffer("extrinsics"));
    return outputs;
}

MvBaseOutputs TensorRtEngine::InferBase(const int num_views, const std::vector<float>& images) {
    if (num_views <= 0) {
        throw std::runtime_error("InferBase: num_views must be positive.");
    }

    const nvinfer1::Dims input_dims = MakeBaseInputDims(num_views);

#if NV_TENSORRT_MAJOR >= 10
    if (!impl_->context->setInputShape(kInputTensorName, input_dims)) {
        throw std::runtime_error("InferBase: failed to set dynamic input shape.");
    }
    for (Buffer& buffer : impl_->buffers) {
        const char* name = buffer.name.c_str();
        buffer.dims = DimsToVector(impl_->context->getTensorShape(name));
        const std::size_t new_count = Volume(buffer.dims);
        if (new_count != buffer.element_count) {
            buffer.element_count = new_count;
            buffer.host_values.resize(new_count);
            cudaFree(buffer.device_ptr);
            buffer.device_ptr = nullptr;
            CheckCuda(
                cudaMalloc(&buffer.device_ptr, new_count * sizeof(float)),
                ("cudaMalloc(resize " + buffer.name + ")").c_str()
            );
            if (!impl_->context->setTensorAddress(name, buffer.device_ptr)) {
                throw std::runtime_error("InferBase: failed to rebind tensor address for " + buffer.name);
            }
        }
    }
#else
    const int input_index = impl_->engine->getBindingIndex(kInputTensorName);
    if (input_index < 0) {
        throw std::runtime_error("InferBase: engine missing images input binding.");
    }
    if (!impl_->context->setBindingDimensions(input_index, input_dims)) {
        throw std::runtime_error("InferBase: failed to set binding dimensions.");
    }
    for (Buffer& buffer : impl_->buffers) {
        buffer.dims = DimsToVector(impl_->context->getBindingDimensions(buffer.binding_index));
        const std::size_t new_count = Volume(buffer.dims);
        if (new_count != buffer.element_count) {
            buffer.element_count = new_count;
            buffer.host_values.resize(new_count);
            cudaFree(buffer.device_ptr);
            buffer.device_ptr = nullptr;
            CheckCuda(
                cudaMalloc(&buffer.device_ptr, new_count * sizeof(float)),
                ("cudaMalloc(resize " + buffer.name + ")").c_str()
            );
            impl_->bindings[static_cast<std::size_t>(buffer.binding_index)] = buffer.device_ptr;
        }
    }
#endif

    auto find_buffer = [&](const std::string& name) -> Buffer& {
        for (Buffer& buffer : impl_->buffers) {
            if (buffer.name == name) {
                return buffer;
            }
        }
        throw std::runtime_error("InferBase: engine missing tensor: " + name);
    };

    Buffer& input = find_buffer(kInputTensorName);
    if (images.size() != input.element_count) {
        std::ostringstream oss;
        oss << "InferBase: expected " << input.element_count << " floats, got " << images.size();
        throw std::runtime_error(oss.str());
    }

    CheckCuda(
        cudaMemcpyAsync(
            input.device_ptr,
            images.data(),
            images.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            impl_->stream
        ),
        "cudaMemcpyAsync(H2D base images)"
    );

#if NV_TENSORRT_MAJOR >= 10
    if (!impl_->context->enqueueV3(impl_->stream)) {
        throw std::runtime_error("InferBase: enqueueV3 failed.");
    }
#else
    if (!impl_->context->enqueueV2(impl_->bindings.data(), impl_->stream, nullptr)) {
        throw std::runtime_error("InferBase: enqueueV2 failed.");
    }
#endif

    for (Buffer& buffer : impl_->buffers) {
        if (buffer.is_input) {
            continue;
        }
        CheckCuda(
            cudaMemcpyAsync(
                buffer.host_values.data(),
                buffer.device_ptr,
                buffer.host_values.size() * sizeof(float),
                cudaMemcpyDeviceToHost,
                impl_->stream
            ),
            ("cudaMemcpyAsync(D2H base " + buffer.name + ")").c_str()
        );
    }

    CheckCuda(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize(base)");

    MvBaseOutputs outputs;
    outputs.depth      = ToTensorOutput(find_buffer("depth"));
    outputs.depth_conf = ToTensorOutput(find_buffer("depth_conf"));
    outputs.intrinsics = ToTensorOutput(find_buffer("intrinsics"));
    outputs.extrinsics = ToTensorOutput(find_buffer("extrinsics"));
    return outputs;
}

std::vector<std::string> TensorRtEngine::tensor_names() const {
    return impl_->tensor_names;
}

}  // namespace da3
