#pragma once

#include "da3/types.hpp"

namespace da3 {

std::vector<fs::path> CollectImagePaths(const fs::path& image_dir);

ProcessedView PrepareInputView(
    const cv::Mat& bgr_image,
    const fs::path& image_label,
    const PreprocessConfig& config = {}
);

InputBatch PrepareInputBatch(
    const std::vector<fs::path>& image_paths,
    const PreprocessConfig& config = {}
);

InputBatch PrepareInputBatch(
    const std::vector<cv::Mat>& bgr_images,
    const std::vector<fs::path>& image_labels,
    const PreprocessConfig& config = {}
);

}  // namespace da3
