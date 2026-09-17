#pragma once

#include <NvInfer.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace iris::core::trt_utils {

struct TrtDeleter {
  template <typename T>
  void operator()(T* ptr) const noexcept {
    if (ptr != nullptr) {
      delete ptr;
    }
  }
};

class TrtLogger : public nvinfer1::ILogger {
 public:
  explicit TrtLogger(Severity severity = Severity::kWARNING)
      : severity_(severity) {}

  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= severity_) {
      std::cerr << "[TRT] " << msg << std::endl;
    }
  }

 private:
  Severity severity_;
};

inline std::vector<char> load_engine_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open engine file: " + path);
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size)) {
    throw std::runtime_error("Failed to read engine file: " + path);
  }

  return buffer;
}

inline size_t get_element_size(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT:  return 4;
    case nvinfer1::DataType::kINT32:  return 4;
    case nvinfer1::DataType::kHALF:   return 2;
    case nvinfer1::DataType::kUINT8:  return 1;
    case nvinfer1::DataType::kINT8:   return 1;
    case nvinfer1::DataType::kINT64:  return 8;
    default:                          return 8;
  }
}

inline size_t compute_volume(const nvinfer1::Dims& dims) {
  size_t volume = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    volume *= static_cast<size_t>(dims.d[i]);
  }
  return volume;
}

inline std::string dims_to_string(const nvinfer1::Dims& dims) {
  std::string result = "[";
  for (int i = 0; i < dims.nbDims; ++i) {
    result += std::to_string(dims.d[i]);
    if (i < dims.nbDims - 1) result += ", ";
  }
  result += "]";
  return result;
}

inline std::string dtype_to_string(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT: return "FLOAT32";
    case nvinfer1::DataType::kINT32: return "INT32";
    case nvinfer1::DataType::kHALF:  return "FLOAT16";
    case nvinfer1::DataType::kUINT8: return "UINT8";
    case nvinfer1::DataType::kINT8:  return "INT8";
    case nvinfer1::DataType::kINT64: return "INT64";
    default: return "UNKNOWN(" + std::to_string(static_cast<int>(type)) + ")";
  }
}

}  // namespace iris::core::trt_utils
