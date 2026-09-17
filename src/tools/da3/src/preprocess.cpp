#include "da3/preprocess.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace da3 {

namespace {

struct ProcessedImageGeometry {
    cv::Mat image;
    int content_x = 0;
    int content_y = 0;
    int content_width = 0;
    int content_height = 0;
    float scale_x = 1.0f;
    float scale_y = 1.0f;
};

bool IsSupportedImageExtension(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    static const std::array<const char*, 7> kSupported = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tiff",
        ".tif",
    };
    return std::any_of(kSupported.begin(), kSupported.end(), [&](const char* candidate) {
        return ext == candidate;
    });
}

int NearestMultiple(const int value, const int patch_size) {
    const int down = (value / patch_size) * patch_size;
    const int up = down + patch_size;
    return std::abs(up - value) <= std::abs(value - down) ? up : down;
}

cv::Mat ResizeWithScale(const cv::Mat& image, const int target_longest_side) {
    const int width = image.cols;
    const int height = image.rows;
    const int longest = std::max(width, height);
    if (longest == target_longest_side) {
        return image.clone();
    }

    const double scale = static_cast<double>(target_longest_side) / static_cast<double>(longest);
    const int resized_width = std::max(1, static_cast<int>(std::llround(width * scale)));
    const int resized_height = std::max(1, static_cast<int>(std::llround(height * scale)));
    const int interpolation = scale > 1.0 ? cv::INTER_CUBIC : cv::INTER_AREA;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, interpolation);
    return resized;
}

cv::Mat MakeDivisibleByResize(const cv::Mat& image, const int patch_size) {
    const int resized_width = std::max(1, NearestMultiple(image.cols, patch_size));
    const int resized_height = std::max(1, NearestMultiple(image.rows, patch_size));
    if (resized_width == image.cols && resized_height == image.rows) {
        return image.clone();
    }

    const bool upscale = resized_width > image.cols || resized_height > image.rows;
    const int interpolation = upscale ? cv::INTER_CUBIC : cv::INTER_AREA;
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, interpolation);
    return resized;
}

ProcessedImageGeometry ResizeUpperBound(
    const cv::Mat& image,
    const PreprocessConfig& config
) {
    ProcessedImageGeometry geometry;
    geometry.image = ResizeWithScale(image, config.process_res);
    geometry.image = MakeDivisibleByResize(geometry.image, config.patch_size);
    geometry.content_width = geometry.image.cols;
    geometry.content_height = geometry.image.rows;
    geometry.scale_x =
        static_cast<float>(geometry.content_width) / static_cast<float>(image.cols);
    geometry.scale_y =
        static_cast<float>(geometry.content_height) / static_cast<float>(image.rows);
    return geometry;
}

ProcessedImageGeometry LetterboxFit(
    const cv::Mat& image,
    const PreprocessConfig& config
) {
    if (config.expected_width <= 0 || config.expected_height <= 0) {
        throw std::runtime_error("Expected DA3 input dimensions must be positive.");
    }

    const double scale = std::min(
        static_cast<double>(config.expected_width) / static_cast<double>(image.cols),
        static_cast<double>(config.expected_height) / static_cast<double>(image.rows)
    );
    if (!std::isfinite(scale) || scale <= 0.0) {
        throw std::runtime_error("Invalid letterbox scale for DA3 preprocessing.");
    }

    const int resized_width = std::max(1, static_cast<int>(std::llround(image.cols * scale)));
    const int resized_height = std::max(1, static_cast<int>(std::llround(image.rows * scale)));
    if (resized_width > config.expected_width || resized_height > config.expected_height) {
        throw std::runtime_error("Letterboxed DA3 image exceeded the target tensor size.");
    }

    const int interpolation = scale > 1.0 ? cv::INTER_CUBIC : cv::INTER_AREA;
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, interpolation);

    const int pad_left = (config.expected_width - resized_width) / 2;
    const int pad_right = config.expected_width - resized_width - pad_left;
    const int pad_top = (config.expected_height - resized_height) / 2;
    const int pad_bottom = config.expected_height - resized_height - pad_top;

    ProcessedImageGeometry geometry;
    cv::copyMakeBorder(
        resized,
        geometry.image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114)
    );
    geometry.content_x = pad_left;
    geometry.content_y = pad_top;
    geometry.content_width = resized_width;
    geometry.content_height = resized_height;
    geometry.scale_x = static_cast<float>(resized_width) / static_cast<float>(image.cols);
    geometry.scale_y = static_cast<float>(resized_height) / static_cast<float>(image.rows);
    return geometry;
}

ProcessedView PreprocessRgbImage(
    const cv::Mat& rgb,
    const fs::path& image_label,
    const PreprocessConfig& config
) {
    if (rgb.empty()) {
        throw std::runtime_error("Cannot preprocess an empty DA3 image.");
    }
    if (rgb.type() != CV_8UC3) {
        throw std::runtime_error("DA3 preprocessing expects 8-bit 3-channel RGB input.");
    }

    const int source_height = rgb.rows;
    const int source_width = rgb.cols;

    ProcessedImageGeometry geometry;
    if (config.process_res_method == "upper_bound_resize") {
        geometry = ResizeUpperBound(rgb, config);
        if (geometry.image.rows != config.expected_height ||
            geometry.image.cols != config.expected_width) {
            std::ostringstream oss;
            oss << "Preprocessed image " << image_label.string() << " produced shape "
                << geometry.image.rows << "x" << geometry.image.cols << ", expected "
                << config.expected_height << "x" << config.expected_width
                << ". Export a dedicated ONNX/engine for this geometry.";
            throw std::runtime_error(oss.str());
        }
    } else if (config.process_res_method == "letterbox_fit") {
        geometry = LetterboxFit(rgb, config);
    } else {
        throw std::runtime_error(
            "Unsupported process_res_method for DA3 preprocessing: " + config.process_res_method
        );
    }

    cv::Mat rgb_f32;
    geometry.image.convertTo(rgb_f32, CV_32FC3, 1.0 / 255.0);

    const int height = rgb_f32.rows;
    const int width = rgb_f32.cols;
    std::vector<float> chw(static_cast<std::size_t>(kNumChannels * height * width));

    for (int channel = 0; channel < kNumChannels; ++channel) {
        const float mean = kImageNetMean[static_cast<std::size_t>(channel)];
        const float std = kImageNetStd[static_cast<std::size_t>(channel)];
        for (int y = 0; y < height; ++y) {
            const cv::Vec3f* row = rgb_f32.ptr<cv::Vec3f>(y);
            for (int x = 0; x < width; ++x) {
                const std::size_t offset = static_cast<std::size_t>(
                    channel * height * width + y * width + x
                );
                chw[offset] = (row[x][channel] - mean) / std;
            }
        }
    }

    ProcessedView view;
    view.image_path = image_label;
    view.rgb_u8 = std::move(geometry.image);
    view.chw = std::move(chw);
    view.source_height = source_height;
    view.source_width = source_width;
    view.height = height;
    view.width = width;
    view.content_x = geometry.content_x;
    view.content_y = geometry.content_y;
    view.content_width = geometry.content_width;
    view.content_height = geometry.content_height;
    view.scale_x = geometry.scale_x;
    view.scale_y = geometry.scale_y;
    return view;
}

ProcessedView PreprocessImage(const fs::path& path, const PreprocessConfig& config) {
    cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error("Failed to read image: " + path.string());
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return PreprocessRgbImage(rgb, path, config);
}

}  // namespace

std::vector<fs::path> CollectImagePaths(const fs::path& image_dir) {
    if (!fs::exists(image_dir)) {
        throw std::runtime_error("Image directory does not exist: " + image_dir.string());
    }
    if (!fs::is_directory(image_dir)) {
        throw std::runtime_error("Expected a directory for --image-dir: " + image_dir.string());
    }

    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(image_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (!IsSupportedImageExtension(entry.path())) {
            continue;
        }
        paths.push_back(entry.path());
    }

    std::sort(paths.begin(), paths.end());
    if (paths.empty()) {
        throw std::runtime_error("No supported images found in " + image_dir.string());
    }
    return paths;
}

ProcessedView PrepareInputView(
    const cv::Mat& bgr_image,
    const fs::path& image_label,
    const PreprocessConfig& config
) {
    if (bgr_image.empty()) {
        throw std::runtime_error("Cannot preprocess an empty DA3 BGR image.");
    }
    if (bgr_image.type() != CV_8UC3) {
        throw std::runtime_error("DA3 preprocessing expects BGR8 images.");
    }

    cv::Mat rgb;
    cv::cvtColor(bgr_image, rgb, cv::COLOR_BGR2RGB);
    return PreprocessRgbImage(rgb, image_label, config);
}

InputBatch PrepareInputBatch(const std::vector<fs::path>& image_paths, const PreprocessConfig& config) {
    if (image_paths.empty()) {
        throw std::runtime_error("PrepareInputBatch: no image paths provided.");
    }

    InputBatch batch;
    batch.views.reserve(image_paths.size());
    batch.images.reserve(
        image_paths.size() * static_cast<std::size_t>(kNumChannels * config.expected_height
                                                       * config.expected_width)
    );

    for (const fs::path& image_path : image_paths) {
        ProcessedView view = PreprocessImage(image_path, config);
        batch.images.insert(batch.images.end(), view.chw.begin(), view.chw.end());
        batch.views.push_back(std::move(view));
    }

    return batch;
}

InputBatch PrepareInputBatch(
    const std::vector<cv::Mat>& bgr_images,
    const std::vector<fs::path>& image_labels,
    const PreprocessConfig& config
) {
    if (bgr_images.empty()) {
        throw std::runtime_error("PrepareInputBatch: no BGR images provided.");
    }
    if (image_labels.size() != bgr_images.size()) {
        throw std::runtime_error("DA3 image labels must match the number of input frames.");
    }

    InputBatch batch;
    batch.views.reserve(bgr_images.size());
    batch.images.reserve(
        bgr_images.size() * static_cast<std::size_t>(kNumChannels * config.expected_height
                                                      * config.expected_width)
    );

    for (std::size_t index = 0; index < bgr_images.size(); ++index) {
        ProcessedView view = PrepareInputView(bgr_images[index], image_labels[index], config);
        batch.images.insert(batch.images.end(), view.chw.begin(), view.chw.end());
        batch.views.push_back(std::move(view));
    }

    return batch;
}

}  // namespace da3
