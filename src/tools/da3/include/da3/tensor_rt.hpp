#pragma once

#include <memory>

#include "da3/types.hpp"

namespace da3 {

class TensorRtEngine {
  public:
    explicit TensorRtEngine(const fs::path& engine_path, bool is_base = false);
    ~TensorRtEngine();

    TensorRtEngine(const TensorRtEngine&) = delete;
    TensorRtEngine& operator=(const TensorRtEngine&) = delete;

    TensorRtEngine(TensorRtEngine&&) noexcept;
    TensorRtEngine& operator=(TensorRtEngine&&) noexcept;

    Mv4Outputs Infer(const std::vector<float>& images);
    MvBaseOutputs InferBase(int num_views, const std::vector<float>& images);
    std::vector<std::string> tensor_names() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace da3
