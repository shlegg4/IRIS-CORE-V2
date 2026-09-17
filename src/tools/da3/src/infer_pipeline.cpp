#include "da3/infer_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

#include "da3/ply_writer.hpp"
#include "da3/preprocess.hpp"
#include "da3/reconstruct.hpp"
#include "da3/tensor_rt.hpp"
#include "iris/core/calibration/calibration_io.hpp"
#include "iris/core/calibration/calibration_types.hpp"
#include "iris/core/types.hpp"

namespace da3 {

namespace {

using Clock = std::chrono::steady_clock;

struct StagedOutput {
    fs::path temp_path;
    fs::path final_path;
};

struct DepthPreviewOutput {
    std::vector<fs::path> image_paths;
    float min_depth = 0.0f;
    float max_depth = 1.0f;
    float lower_percentile = 2.0f;
    float upper_percentile = 98.0f;
    bool inverted = true;
};

float Percentile(std::vector<float> values, const float percentile) {
    if (values.empty()) {
        throw std::runtime_error("Cannot compute percentile of empty input.");
    }

    std::sort(values.begin(), values.end());
    const double position =
        (static_cast<double>(percentile) / 100.0) * static_cast<double>(values.size() - 1);
    const std::size_t lower_index = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper_index = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower_index);
    const double lower = static_cast<double>(values[lower_index]);
    const double upper = static_cast<double>(values[upper_index]);
    return static_cast<float>(lower + (upper - lower) * fraction);
}

nlohmann::json MatrixToJson(
    const std::vector<float>& values,
    const std::size_t base,
    const int rows,
    const int cols
) {
    nlohmann::json matrix = nlohmann::json::array();
    for (int row = 0; row < rows; ++row) {
        nlohmann::json row_values = nlohmann::json::array();
        for (int col = 0; col < cols; ++col) {
            row_values.push_back(values[base + static_cast<std::size_t>(row * cols + col)]);
        }
        matrix.push_back(std::move(row_values));
    }
    return matrix;
}

nlohmann::json EigenMatrixToJson(const Eigen::Matrix4f& matrix) {
    nlohmann::json rows = nlohmann::json::array();
    for (int row = 0; row < 4; ++row) {
        nlohmann::json row_json = nlohmann::json::array();
        for (int col = 0; col < 4; ++col) {
            row_json.push_back(matrix(row, col));
        }
        rows.push_back(std::move(row_json));
    }
    return rows;
}

nlohmann::json EigenVectorToJson(const Eigen::Vector3f& vector) {
    return nlohmann::json::array({vector.x(), vector.y(), vector.z()});
}

double MillisecondsSince(const Clock::time_point start, const Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

bool IsContentPixel(const ProcessedView& view, const int x, const int y) {
    return x >= view.content_x &&
           y >= view.content_y &&
           x < view.content_x + view.content_width &&
           y < view.content_y + view.content_height;
}

std::vector<int> ResolveCameraIds(const std::vector<int>& requested_camera_ids) {
    if (requested_camera_ids.empty()) {
        std::vector<int> default_ids;
        default_ids.reserve(static_cast<std::size_t>(kNumViews));
        for (int view = 0; view < kNumViews; ++view) {
            default_ids.push_back(view);
        }
        return default_ids;
    }

    const std::set<int> unique_ids(requested_camera_ids.begin(), requested_camera_ids.end());
    if (unique_ids.size() != requested_camera_ids.size()) {
        throw std::runtime_error("DA3 camera IDs must be unique.");
    }

    return requested_camera_ids;
}

iris::core::IntrinsicsMeta BuildIntrinsicsMeta(
    const std::vector<float>& values,
    const int view_index,
    const ProcessedView& view
) {
    if (view.source_width <= 0 || view.source_height <= 0 || view.width <= 0 || view.height <= 0) {
        throw std::runtime_error("Invalid view dimensions while exporting DA3 intrinsics.");
    }
    if (view.scale_x <= 0.0f || view.scale_y <= 0.0f) {
        throw std::runtime_error("Invalid DA3 preprocessing scale while exporting intrinsics.");
    }

    const std::size_t base = static_cast<std::size_t>(view_index * 9);
    iris::core::IntrinsicsMeta intrinsics;
    intrinsics.K[0] = values[base + 0] / view.scale_x;
    intrinsics.K[1] = values[base + 1] / view.scale_x;
    intrinsics.K[2] =
        (values[base + 2] - static_cast<float>(view.content_x)) / view.scale_x;
    intrinsics.K[3] = values[base + 3] / view.scale_y;
    intrinsics.K[4] = values[base + 4] / view.scale_y;
    intrinsics.K[5] =
        (values[base + 5] - static_cast<float>(view.content_y)) / view.scale_y;
    intrinsics.K[6] = values[base + 6];
    intrinsics.K[7] = values[base + 7];
    intrinsics.K[8] = values[base + 8];
    intrinsics.w = view.source_width;
    intrinsics.h = view.source_height;
    return intrinsics;
}

iris::core::ExtrinsicsMeta BuildExtrinsicsMeta(
    const std::vector<float>& values,
    const int view_index,
    const Eigen::Matrix4f& alignment,
    const int cam_id
) {
    Eigen::Matrix4f w2c = Eigen::Matrix4f::Zero();
    const std::size_t base = static_cast<std::size_t>(view_index * 16);
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            w2c(row, col) = values[base + static_cast<std::size_t>(row * 4 + col)];
        }
    }

    const Eigen::Matrix4f c2w = alignment * w2c.inverse();

    iris::core::ExtrinsicsMeta extrinsics;
    extrinsics.cam_id = cam_id;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            extrinsics.R[row * 3 + col] = c2w(row, col);
        }
    }
    for (int row = 0; row < 3; ++row) {
        extrinsics.t[row] = c2w(row, 3);
    }
    return extrinsics;
}

fs::path ResolvePlyPath(const fs::path& out_dir, const std::optional<fs::path>& save_ply) {
    fs::path ply_path = save_ply.value_or(out_dir / "scene.ply");
    if (ply_path.is_relative()) {
        ply_path = out_dir / ply_path;
    }
    return ply_path;
}

fs::path MakeTempPath(const fs::path& final_path) {
    const fs::path parent = final_path.parent_path();
    const std::string extension = final_path.extension().string();
    const std::string filename = extension.empty()
        ? final_path.filename().string() + ".tmp"
        : final_path.stem().string() + ".tmp" + extension;
    return parent / filename;
}

void WriteJsonFile(const fs::path& output_path, const nlohmann::json& value) {
    if (!output_path.parent_path().empty()) {
        fs::create_directories(output_path.parent_path());
    }

    std::ofstream stream(output_path);
    if (!stream) {
        throw std::runtime_error("Failed to open JSON output for writing: " + output_path.string());
    }
    stream << value.dump(2) << '\n';
    if (!stream) {
        throw std::runtime_error("Failed while writing JSON output: " + output_path.string());
    }
}

void WriteImageFile(const fs::path& output_path, const cv::Mat& image) {
    if (!output_path.parent_path().empty()) {
        fs::create_directories(output_path.parent_path());
    }
    if (!cv::imwrite(output_path.string(), image)) {
        throw std::runtime_error("Failed to write image output: " + output_path.string());
    }
}

void StageJsonOutput(
    const fs::path& final_path,
    const nlohmann::json& value,
    std::vector<StagedOutput>& staged_outputs
) {
    const fs::path temp_path = MakeTempPath(final_path);
    std::error_code ec;
    fs::remove(temp_path, ec);
    WriteJsonFile(temp_path, value);
    staged_outputs.push_back({temp_path, final_path});
}

void StagePlyOutput(
    const fs::path& final_path,
    const std::vector<PointVertex>& vertices,
    std::vector<StagedOutput>& staged_outputs
) {
    const fs::path temp_path = MakeTempPath(final_path);
    std::error_code ec;
    fs::remove(temp_path, ec);
    WriteBinaryPly(temp_path, vertices);
    staged_outputs.push_back({temp_path, final_path});
}

void StageImageOutput(
    const fs::path& final_path,
    const cv::Mat& image,
    std::vector<StagedOutput>& staged_outputs
) {
    const fs::path temp_path = MakeTempPath(final_path);
    std::error_code ec;
    fs::remove(temp_path, ec);
    WriteImageFile(temp_path, image);
    staged_outputs.push_back({temp_path, final_path});
}

void CleanupStagedOutputs(const std::vector<StagedOutput>& staged_outputs) {
    for (const StagedOutput& output : staged_outputs) {
        std::error_code ec;
        fs::remove(output.temp_path, ec);
    }
}

void CommitStagedOutputs(const std::vector<StagedOutput>& staged_outputs) {
    for (const StagedOutput& output : staged_outputs) {
        if (!output.final_path.parent_path().empty()) {
            fs::create_directories(output.final_path.parent_path());
        }

        std::error_code ec;
        fs::remove(output.final_path, ec);
        if (ec) {
            throw std::runtime_error(
                "Failed to replace DA3 output file: " + output.final_path.string()
            );
        }

        fs::rename(output.temp_path, output.final_path);
    }
}

void ValidateConfig(const InferenceConfig& config) {
    if (config.engine_path.empty()) {
        throw std::runtime_error("DA3 engine path must not be empty.");
    }
    if (!fs::exists(config.engine_path)) {
        throw std::runtime_error("DA3 engine path does not exist: " + config.engine_path.string());
    }
    if (config.out_dir.empty()) {
        throw std::runtime_error("DA3 output directory must not be empty.");
    }
    if (config.image_paths.empty()) {
        throw std::runtime_error("DA3 inference expects at least one input image.");
    }
    if (config.model_type == "large" &&
        config.image_paths.size() != static_cast<std::size_t>(kNumViews)) {
        throw std::runtime_error("DA3 large model inference expects exactly 4 input images.");
    }
    if (config.model_type == "base") {
        const std::size_t n = config.image_paths.size();
        if (n < 2 || n > 10) {
            throw std::runtime_error("DA3 base model inference expects between 2 and 10 input images.");
        }
    }
    ResolveCameraIds(config.camera_ids);
    for (const fs::path& image_path : config.image_paths) {
        if (image_path.empty()) {
            throw std::runtime_error("DA3 input image paths must not be empty.");
        }
        if (!fs::exists(image_path)) {
            throw std::runtime_error("DA3 image path does not exist: " + image_path.string());
        }
    }
}

void ValidatePreparedConfig(const PreparedInferenceConfig& config) {
    if (config.out_dir.empty()) {
        throw std::runtime_error("DA3 output directory must not be empty.");
    }
    ResolveCameraIds(config.camera_ids);

    if (config.batch.views.size() != static_cast<std::size_t>(kNumViews)) {
        throw std::runtime_error("DA3 prepared inference expects exactly 4 processed views.");
    }
}

nlohmann::json BuildFrameDiagnosticsJson(const FrameDiagnostics& diagnostics) {
    return {
        {"sequence", diagnostics.sequence},
        {"batch_timestamp_ns", diagnostics.batch_timestamp_ns},
        {"frame_capture_times_ns", diagnostics.frame_capture_times_ns},
        {"frame_ingest_times_ns", diagnostics.frame_ingest_times_ns},
        {"source_id", diagnostics.source_id},
    };
}

DepthPreviewOutput StageDepthPreviewOutputs(
    const fs::path& out_dir,
    const std::vector<int>& camera_ids,
    const InputBatch& batch,
    const Mv4Outputs& outputs,
    std::vector<StagedOutput>& staged_outputs
) {
    const std::size_t expected_pixel_count =
        static_cast<std::size_t>(kNumViews * kInputHeight * kInputWidth);
    if (outputs.depth.values.size() != expected_pixel_count ||
        outputs.sky.values.size() != expected_pixel_count) {
        throw std::runtime_error("Unexpected DA3 tensor size while exporting depth preview PNGs.");
    }

    DepthPreviewOutput preview;
    preview.image_paths.reserve(camera_ids.size());

    std::vector<float> valid_depths;
    valid_depths.reserve(expected_pixel_count);
    for (int view = 0; view < kNumViews; ++view) {
        const ProcessedView& processed_view = batch.views[static_cast<std::size_t>(view)];
        for (int y = 0; y < kInputHeight; ++y) {
            for (int x = 0; x < kInputWidth; ++x) {
                if (!IsContentPixel(processed_view, x, y)) {
                    continue;
                }

                const std::size_t linear = static_cast<std::size_t>(
                    view * kInputHeight * kInputWidth + y * kInputWidth + x
                );
                if (outputs.sky.values[linear] >= 0.5f) {
                    continue;
                }

                const float depth_value = outputs.depth.values[linear];
                if (std::isfinite(depth_value) && depth_value > 0.0f) {
                    valid_depths.push_back(depth_value);
                }
            }
        }
    }

    if (!valid_depths.empty()) {
        preview.min_depth = Percentile(valid_depths, preview.lower_percentile);
        preview.max_depth = Percentile(valid_depths, preview.upper_percentile);
        if (!(preview.max_depth > preview.min_depth)) {
            const auto [min_it, max_it] = std::minmax_element(
                valid_depths.begin(),
                valid_depths.end()
            );
            preview.min_depth = *min_it;
            preview.max_depth = *max_it;
        }
        if (!(preview.max_depth > preview.min_depth)) {
            preview.max_depth = preview.min_depth + 1.0f;
        }
    }

    const float depth_range = std::max(
        preview.max_depth - preview.min_depth,
        std::numeric_limits<float>::epsilon()
    );
    const fs::path depth_dir = out_dir / "depth";

    for (int view = 0; view < kNumViews; ++view) {
        cv::Mat depth_preview(kInputHeight, kInputWidth, CV_8UC1, cv::Scalar(0));
        const ProcessedView& processed_view = batch.views[static_cast<std::size_t>(view)];

        for (int y = 0; y < kInputHeight; ++y) {
            std::uint8_t* row = depth_preview.ptr<std::uint8_t>(y);
            for (int x = 0; x < kInputWidth; ++x) {
                if (!IsContentPixel(processed_view, x, y)) {
                    continue;
                }

                const std::size_t linear = static_cast<std::size_t>(
                    view * kInputHeight * kInputWidth + y * kInputWidth + x
                );
                if (outputs.sky.values[linear] >= 0.5f) {
                    continue;
                }

                const float depth_value = outputs.depth.values[linear];
                if (!std::isfinite(depth_value) || depth_value <= 0.0f) {
                    continue;
                }

                const float normalized = std::clamp(
                    (depth_value - preview.min_depth) / depth_range,
                    0.0f,
                    1.0f
                );
                row[x] = static_cast<std::uint8_t>(std::lround((1.0f - normalized) * 255.0f));
            }
        }

        const fs::path image_path =
            depth_dir / ("cam" + std::to_string(camera_ids[static_cast<std::size_t>(view)]) + ".png");
        StageImageOutput(image_path, depth_preview, staged_outputs);
        preview.image_paths.push_back(image_path);
    }

    return preview;
}

void StageCalibrationOutputs(
    const fs::path& out_dir,
    const std::vector<int>& camera_ids,
    const InputBatch& batch,
    const Mv4Outputs& outputs,
    const ReconstructionResult& reconstruction,
    InferenceResult& result,
    std::vector<StagedOutput>& staged_outputs
) {
    namespace calibration = iris::core::calibration;

    result.intrinsics_dir = out_dir;
    result.extrinsics_path = out_dir / "extrinsics.json";
    result.camera_ids = camera_ids;
    result.intrinsics.clear();
    result.extrinsics.clear();
    result.intrinsics.reserve(camera_ids.size());
    result.extrinsics.reserve(camera_ids.size());

    calibration::MultiCameraExtrinsicsResult extrinsics_result;
    extrinsics_result.camera_ids = camera_ids;
    extrinsics_result.frames_used = 1;
    extrinsics_result.success = true;
    extrinsics_result.mean_reprojection_error = 0.0;
    extrinsics_result.camera_extrinsics.reserve(camera_ids.size());

    for (int view = 0; view < kNumViews; ++view) {
        const std::size_t index = static_cast<std::size_t>(view);
        const iris::core::IntrinsicsMeta intrinsics =
            BuildIntrinsicsMeta(outputs.intrinsics.values, view, batch.views[index]);
        const iris::core::ExtrinsicsMeta extrinsics = BuildExtrinsicsMeta(
            outputs.extrinsics.values,
            view,
            reconstruction.alignment,
            camera_ids[index]
        );

        result.intrinsics.push_back(intrinsics);
        result.extrinsics.push_back(extrinsics);

        const fs::path intrinsics_path =
            out_dir / ("intrinsics_cam" + std::to_string(camera_ids[index]) + ".json");
        StageJsonOutput(
            intrinsics_path,
            calibration::intrinsics_to_json(intrinsics),
            staged_outputs
        );

        calibration::ExtrinsicCalibrationResult extrinsic_result;
        extrinsic_result.success = true;
        extrinsic_result.reprojection_error = 0.0;
        extrinsic_result.extrinsics = extrinsics;
        extrinsics_result.camera_extrinsics.push_back(std::move(extrinsic_result));
    }

    StageJsonOutput(
        result.extrinsics_path,
        calibration::multi_camera_extrinsics_to_json(extrinsics_result),
        staged_outputs
    );
}

nlohmann::json BuildCamerasJson(
    const PreparedInferenceConfig& config,
    const std::vector<int>& camera_ids,
    const Mv4Outputs& outputs,
    const ReconstructionResult& reconstruction,
    const DepthPreviewOutput& depth_preview
) {
    const ReconstructionConfig reconstruction_config = {.viewer_align = config.viewer_align};

    nlohmann::json cameras = {
        {"input_contract",
         {
             {"images", {kBatchSize, kNumViews, kNumChannels, kInputHeight, kInputWidth}},
             {"depth", outputs.depth.dims},
             {"depth_conf", outputs.depth_conf.dims},
             {"sky", outputs.sky.dims},
             {"intrinsics", outputs.intrinsics.dims},
             {"extrinsics", outputs.extrinsics.dims},
         }},
        {"preprocess",
         {
             {"process_res", kProcessRes},
             {"process_res_method", PreprocessConfig{}.process_res_method},
             {"patch_size", kPatchSize},
             {"expected_height", kInputHeight},
             {"expected_width", kInputWidth},
             {"mean", kImageNetMean},
             {"std", kImageNetStd},
         }},
        {"reconstruction",
         {
             {"viewer_align", config.viewer_align},
             {"alignment_mode",
              !config.viewer_align ? "raw" :
              (reconstruction.ground_alignment.plane_found ? "ground_plane" : "viewer_fallback")},
             {"pose_refinement", ToString(reconstruction_config.pose_refinement)},
             {"confidence_threshold", reconstruction.applied_conf_threshold},
             {"ground_plane_found", reconstruction.ground_alignment.plane_found},
             {"point_count", reconstruction.vertices.size()},
         }},
        {"calibration_compatible_outputs",
         {
              {"intrinsics_dir", config.out_dir.string()},
              {"extrinsics", (config.out_dir / "extrinsics.json").string()},
          }},
        {"depth_preview",
         {
             {"format", "png"},
             {"description", "Normalized preview of the raw depth tensor; not a metric depth export."},
             {"output_dir", (config.out_dir / "depth").string()},
             {"inverted", depth_preview.inverted},
             {"lower_percentile", depth_preview.lower_percentile},
             {"upper_percentile", depth_preview.upper_percentile},
             {"near_depth", depth_preview.min_depth},
             {"far_depth", depth_preview.max_depth},
         }},
        {"alignment", EigenMatrixToJson(reconstruction.alignment)},
        {"ground_alignment",
         {
             {"plane_found", reconstruction.ground_alignment.plane_found},
             {"sampled_point_count", reconstruction.ground_alignment.sampled_point_count},
             {"inlier_count", reconstruction.ground_alignment.inlier_count},
             {"inlier_ratio", reconstruction.ground_alignment.inlier_ratio},
             {"plane_normal_world", EigenVectorToJson(reconstruction.ground_alignment.plane_normal)},
             {"plane_point_world", EigenVectorToJson(reconstruction.ground_alignment.plane_point)},
             {"camera_centroid_world",
              EigenVectorToJson(reconstruction.ground_alignment.camera_centroid)},
             {"origin_world", EigenVectorToJson(reconstruction.ground_alignment.origin)},
         }},
        {"views", nlohmann::json::array()},
    };

    if (config.frame_diagnostics.has_value()) {
        cameras["frame_diagnostics"] = BuildFrameDiagnosticsJson(*config.frame_diagnostics);
    }

    for (int view = 0; view < kNumViews; ++view) {
        const std::size_t index = static_cast<std::size_t>(view);
        const std::size_t intrinsics_base = static_cast<std::size_t>(view * 9);
        const std::size_t extrinsics_base = static_cast<std::size_t>(view * 16);
        cameras["views"].push_back(
            {
                {"camera_id", camera_ids[index]},
                {"image", config.batch.views[index].image_path.filename().string()},
                {"image_path", config.batch.views[index].image_path.string()},
                {"depth_preview_png", depth_preview.image_paths[index].string()},
                {"source_size",
                 {
                     config.batch.views[index].source_height,
                     config.batch.views[index].source_width,
                 }},
                {"processed_size",
                 {
                     config.batch.views[index].height,
                     config.batch.views[index].width,
                 }},
                {"content_rect",
                 {
                     {"x", config.batch.views[index].content_x},
                     {"y", config.batch.views[index].content_y},
                     {"width", config.batch.views[index].content_width},
                     {"height", config.batch.views[index].content_height},
                 }},
                {"source_to_processed_scale",
                 {
                     config.batch.views[index].scale_x,
                     config.batch.views[index].scale_y,
                 }},
                {"intrinsics", MatrixToJson(outputs.intrinsics.values, intrinsics_base, 3, 3)},
                {"extrinsics_w2c", MatrixToJson(outputs.extrinsics.values, extrinsics_base, 4, 4)},
            }
        );
    }

    return cameras;
}

}  // namespace

InferenceResult RunPreparedInference(
    const PreparedInferenceConfig& config,
    TensorRtEngine& engine
) {
    ValidatePreparedConfig(config);
    fs::create_directories(config.out_dir);
    const std::vector<int> camera_ids = ResolveCameraIds(config.camera_ids);

    InferenceResult result;
    result.cameras_path = config.out_dir / "cameras.json";
    result.timings_path = config.out_dir / "timings.json";
    result.ply_path = ResolvePlyPath(config.out_dir, config.save_ply);

    const Clock::time_point total_start = Clock::now();

    const Clock::time_point inference_start = Clock::now();
    const Mv4Outputs outputs = engine.Infer(config.batch.images);
    const Clock::time_point inference_end = Clock::now();

    ReconstructionConfig reconstruction_config;
    reconstruction_config.viewer_align = config.viewer_align;

    const Clock::time_point reconstruction_start = Clock::now();
    const ReconstructionResult reconstruction =
        ReconstructPointCloud(outputs, config.batch.views, reconstruction_config);
    const Clock::time_point reconstruction_end = Clock::now();

    const Clock::time_point ply_start = Clock::now();
    std::vector<StagedOutput> staged_outputs;
    try {
        StagePlyOutput(result.ply_path, reconstruction.vertices, staged_outputs);
        const Clock::time_point ply_end = Clock::now();

        const Clock::time_point depth_preview_start = Clock::now();
        const DepthPreviewOutput depth_preview = StageDepthPreviewOutputs(
            config.out_dir,
            camera_ids,
            config.batch,
            outputs,
            staged_outputs
        );
        const Clock::time_point depth_preview_end = Clock::now();

        StageJsonOutput(
            result.cameras_path,
            BuildCamerasJson(config, camera_ids, outputs, reconstruction, depth_preview),
            staged_outputs
        );
        StageCalibrationOutputs(
            config.out_dir,
            camera_ids,
            config.batch,
            outputs,
            reconstruction,
            result,
            staged_outputs
        );

        const Clock::time_point total_end = Clock::now();
        nlohmann::json timings = {
            {"preprocess_ms", config.preprocess_ms},
            {"inference_ms", MillisecondsSince(inference_start, inference_end)},
            {"reconstruction_ms", MillisecondsSince(reconstruction_start, reconstruction_end)},
            {"ply_write_ms", MillisecondsSince(ply_start, ply_end)},
            {"depth_preview_write_ms", MillisecondsSince(depth_preview_start, depth_preview_end)},
            {"total_ms", MillisecondsSince(total_start, total_end)},
        };
        if (config.frame_diagnostics.has_value()) {
            timings["frame_diagnostics"] = BuildFrameDiagnosticsJson(*config.frame_diagnostics);
        }
        StageJsonOutput(result.timings_path, timings, staged_outputs);

        CommitStagedOutputs(staged_outputs);
        CleanupStagedOutputs(staged_outputs);
    } catch (...) {
        CleanupStagedOutputs(staged_outputs);
        throw;
    }

    result.point_count = reconstruction.vertices.size();
    result.confidence_threshold = reconstruction.applied_conf_threshold;
    return result;
}

InferenceResult RunPreparedInferenceBase(
    const PreparedInferenceConfig& config,
    TensorRtEngine& engine
) {
    if (config.out_dir.empty()) {
        throw std::runtime_error("DA3 base output directory must not be empty.");
    }
    const int num_views = static_cast<int>(config.batch.views.size());
    if (num_views == 0) {
        throw std::runtime_error("DA3 base inference: no views in batch.");
    }
    ResolveCameraIds(config.camera_ids);

    fs::create_directories(config.out_dir);
    const std::vector<int> camera_ids = ResolveCameraIds(config.camera_ids);

    InferenceResult result;
    result.cameras_path = config.out_dir / "cameras.json";
    result.timings_path = config.out_dir / "timings.json";
    result.ply_path = ResolvePlyPath(config.out_dir, config.save_ply);

    const Clock::time_point total_start = Clock::now();

    const Clock::time_point inference_start = Clock::now();
    const MvBaseOutputs outputs = engine.InferBase(num_views, config.batch.images);
    const Clock::time_point inference_end = Clock::now();

    ReconstructionConfig reconstruction_config;
    reconstruction_config.viewer_align = config.viewer_align;

    const Clock::time_point reconstruction_start = Clock::now();
    const ReconstructionResult reconstruction =
        ReconstructPointCloudBase(outputs, config.batch.views, reconstruction_config);
    const Clock::time_point reconstruction_end = Clock::now();

    const Clock::time_point ply_start = Clock::now();
    std::vector<StagedOutput> staged_outputs;
    try {
        StagePlyOutput(result.ply_path, reconstruction.vertices, staged_outputs);
        const Clock::time_point ply_end = Clock::now();

        const Clock::time_point depth_preview_start = Clock::now();
        const std::size_t pixels_per_view =
            static_cast<std::size_t>(kBaseInputHeight * kBaseInputWidth);
        const fs::path depth_dir = config.out_dir / "depth";
        std::vector<fs::path> depth_image_paths;
        depth_image_paths.reserve(static_cast<std::size_t>(num_views));

        std::vector<float> valid_depths;
        valid_depths.reserve(outputs.depth.values.size());
        for (int view = 0; view < num_views; ++view) {
            const ProcessedView& pv = config.batch.views[static_cast<std::size_t>(view)];
            for (int y = 0; y < kBaseInputHeight; ++y) {
                for (int x = 0; x < kBaseInputWidth; ++x) {
                    if (!IsContentPixel(pv, x, y)) {
                        continue;
                    }
                    const std::size_t linear =
                        static_cast<std::size_t>(view) * pixels_per_view
                        + static_cast<std::size_t>(y * kBaseInputWidth + x);
                    const float d = outputs.depth.values[linear];
                    if (std::isfinite(d) && d > 0.0f) {
                        valid_depths.push_back(d);
                    }
                }
            }
        }
        float min_depth = 0.0f;
        float max_depth = 1.0f;
        if (!valid_depths.empty()) {
            min_depth = Percentile(valid_depths, 2.0f);
            max_depth = Percentile(valid_depths, 98.0f);
            if (!(max_depth > min_depth)) {
                max_depth = min_depth + 1.0f;
            }
        }
        const float depth_range = std::max(max_depth - min_depth,
                                           std::numeric_limits<float>::epsilon());

        for (int view = 0; view < num_views; ++view) {
            cv::Mat depth_preview(kBaseInputHeight, kBaseInputWidth, CV_8UC1, cv::Scalar(0));
            const ProcessedView& pv = config.batch.views[static_cast<std::size_t>(view)];
            for (int y = 0; y < kBaseInputHeight; ++y) {
                std::uint8_t* row = depth_preview.ptr<std::uint8_t>(y);
                for (int x = 0; x < kBaseInputWidth; ++x) {
                    if (!IsContentPixel(pv, x, y)) {
                        continue;
                    }
                    const std::size_t linear =
                        static_cast<std::size_t>(view) * pixels_per_view
                        + static_cast<std::size_t>(y * kBaseInputWidth + x);
                    const float d = outputs.depth.values[linear];
                    if (!std::isfinite(d) || d <= 0.0f) {
                        continue;
                    }
                    const float normalized = std::clamp(
                        (d - min_depth) / depth_range, 0.0f, 1.0f
                    );
                    row[x] = static_cast<std::uint8_t>(
                        std::lround((1.0f - normalized) * 255.0f)
                    );
                }
            }
            const fs::path img_path =
                depth_dir / ("cam" + std::to_string(camera_ids[static_cast<std::size_t>(view)]) + ".png");
            StageImageOutput(img_path, depth_preview, staged_outputs);
            depth_image_paths.push_back(img_path);
        }
        const Clock::time_point depth_preview_end = Clock::now();

        result.intrinsics_dir = config.out_dir;
        result.extrinsics_path = config.out_dir / "extrinsics.json";
        result.camera_ids = camera_ids;
        result.intrinsics.clear();
        result.extrinsics.clear();
        result.intrinsics.reserve(static_cast<std::size_t>(num_views));
        result.extrinsics.reserve(static_cast<std::size_t>(num_views));

        namespace calibration = iris::core::calibration;
        calibration::MultiCameraExtrinsicsResult extrinsics_result;
        extrinsics_result.camera_ids = camera_ids;
        extrinsics_result.frames_used = 1;
        extrinsics_result.success = true;
        extrinsics_result.mean_reprojection_error = 0.0;
        extrinsics_result.camera_extrinsics.reserve(static_cast<std::size_t>(num_views));

        for (int view = 0; view < num_views; ++view) {
            const std::size_t index = static_cast<std::size_t>(view);
            const iris::core::IntrinsicsMeta intrinsics =
                BuildIntrinsicsMeta(outputs.intrinsics.values, view, config.batch.views[index]);
            const iris::core::ExtrinsicsMeta extrinsics = BuildExtrinsicsMeta(
                outputs.extrinsics.values, view, reconstruction.alignment, camera_ids[index]
            );
            result.intrinsics.push_back(intrinsics);
            result.extrinsics.push_back(extrinsics);

            const fs::path intrinsics_path =
                config.out_dir / ("intrinsics_cam" + std::to_string(camera_ids[index]) + ".json");
            StageJsonOutput(
                intrinsics_path,
                calibration::intrinsics_to_json(intrinsics),
                staged_outputs
            );

            calibration::ExtrinsicCalibrationResult extrinsic_result;
            extrinsic_result.success = true;
            extrinsic_result.reprojection_error = 0.0;
            extrinsic_result.extrinsics = extrinsics;
            extrinsics_result.camera_extrinsics.push_back(std::move(extrinsic_result));
        }

        StageJsonOutput(
            result.extrinsics_path,
            calibration::multi_camera_extrinsics_to_json(extrinsics_result),
            staged_outputs
        );

        nlohmann::json cameras = {
            {"model_type", "base"},
            {"input_contract",
             {
                 {"images", {kBatchSize, num_views, kNumChannels, kBaseInputHeight, kBaseInputWidth}},
                 {"depth", outputs.depth.dims},
                 {"depth_conf", outputs.depth_conf.dims},
                 {"intrinsics", outputs.intrinsics.dims},
                 {"extrinsics", outputs.extrinsics.dims},
             }},
            {"preprocess",
             {
                 {"expected_height", kBaseInputHeight},
                 {"expected_width", kBaseInputWidth},
                 {"mean", kImageNetMean},
                 {"std", kImageNetStd},
             }},
            {"reconstruction",
             {
                 {"viewer_align", config.viewer_align},
                 {"alignment_mode",
                  !config.viewer_align ? "raw" :
                  (reconstruction.ground_alignment.plane_found ? "ground_plane" : "viewer_fallback")},
                 {"confidence_threshold", reconstruction.applied_conf_threshold},
                 {"ground_plane_found", reconstruction.ground_alignment.plane_found},
                 {"point_count", reconstruction.vertices.size()},
             }},
            {"calibration_compatible_outputs",
             {
                 {"intrinsics_dir", config.out_dir.string()},
                 {"extrinsics", (config.out_dir / "extrinsics.json").string()},
             }},
            {"alignment", EigenMatrixToJson(reconstruction.alignment)},
            {"ground_alignment",
             {
                 {"plane_found", reconstruction.ground_alignment.plane_found},
                 {"sampled_point_count", reconstruction.ground_alignment.sampled_point_count},
                 {"inlier_count", reconstruction.ground_alignment.inlier_count},
                 {"inlier_ratio", reconstruction.ground_alignment.inlier_ratio},
                 {"plane_normal_world", EigenVectorToJson(reconstruction.ground_alignment.plane_normal)},
                 {"plane_point_world", EigenVectorToJson(reconstruction.ground_alignment.plane_point)},
                 {"camera_centroid_world",
                  EigenVectorToJson(reconstruction.ground_alignment.camera_centroid)},
                 {"origin_world", EigenVectorToJson(reconstruction.ground_alignment.origin)},
             }},
            {"views", nlohmann::json::array()},
        };
        if (config.frame_diagnostics.has_value()) {
            cameras["frame_diagnostics"] = BuildFrameDiagnosticsJson(*config.frame_diagnostics);
        }
        for (int view = 0; view < num_views; ++view) {
            const std::size_t index = static_cast<std::size_t>(view);
            const std::size_t intrinsics_base = static_cast<std::size_t>(view * 9);
            const std::size_t extrinsics_base = static_cast<std::size_t>(view * 16);
            cameras["views"].push_back({
                {"camera_id", camera_ids[index]},
                {"image", config.batch.views[index].image_path.filename().string()},
                {"image_path", config.batch.views[index].image_path.string()},
                {"depth_preview_png", depth_image_paths[index].string()},
                {"source_size",
                 {
                     config.batch.views[index].source_height,
                     config.batch.views[index].source_width,
                 }},
                {"processed_size",
                 {
                     config.batch.views[index].height,
                     config.batch.views[index].width,
                 }},
                {"content_rect",
                 {
                     {"x", config.batch.views[index].content_x},
                     {"y", config.batch.views[index].content_y},
                     {"width", config.batch.views[index].content_width},
                     {"height", config.batch.views[index].content_height},
                 }},
                {"source_to_processed_scale",
                 {
                     config.batch.views[index].scale_x,
                     config.batch.views[index].scale_y,
                 }},
                {"intrinsics", MatrixToJson(outputs.intrinsics.values, intrinsics_base, 3, 3)},
                {"extrinsics_w2c", MatrixToJson(outputs.extrinsics.values, extrinsics_base, 4, 4)},
            });
        }
        StageJsonOutput(result.cameras_path, cameras, staged_outputs);

        const Clock::time_point total_end = Clock::now();
        nlohmann::json timings = {
            {"preprocess_ms", config.preprocess_ms},
            {"inference_ms", MillisecondsSince(inference_start, inference_end)},
            {"reconstruction_ms", MillisecondsSince(reconstruction_start, reconstruction_end)},
            {"ply_write_ms", MillisecondsSince(ply_start, ply_end)},
            {"depth_preview_write_ms", MillisecondsSince(depth_preview_start, depth_preview_end)},
            {"total_ms", MillisecondsSince(total_start, total_end)},
        };
        if (config.frame_diagnostics.has_value()) {
            timings["frame_diagnostics"] = BuildFrameDiagnosticsJson(*config.frame_diagnostics);
        }
        StageJsonOutput(result.timings_path, timings, staged_outputs);

        CommitStagedOutputs(staged_outputs);
        CleanupStagedOutputs(staged_outputs);
    } catch (...) {
        CleanupStagedOutputs(staged_outputs);
        throw;
    }

    result.point_count = reconstruction.vertices.size();
    result.confidence_threshold = reconstruction.applied_conf_threshold;
    return result;
}

InferenceResult RunInference(const InferenceConfig& config) {
    ValidateConfig(config);
    fs::create_directories(config.out_dir);

    const PreprocessConfig preprocess_config;
    const Clock::time_point preprocess_start = Clock::now();
    InputBatch batch = PrepareInputBatch(config.image_paths, preprocess_config);
    const Clock::time_point preprocess_end = Clock::now();

    TensorRtEngine engine(config.engine_path);

    PreparedInferenceConfig prepared_config;
    prepared_config.batch = std::move(batch);
    prepared_config.camera_ids = ResolveCameraIds(config.camera_ids);
    prepared_config.out_dir = config.out_dir;
    prepared_config.save_ply = config.save_ply;
    prepared_config.viewer_align = config.viewer_align;
    prepared_config.preprocess_ms = MillisecondsSince(preprocess_start, preprocess_end);
    return RunPreparedInference(prepared_config, engine);
}

InferenceResult RunInferenceBase(const InferenceConfig& config) {
    ValidateConfig(config);
    fs::create_directories(config.out_dir);

    PreprocessConfig preprocess_config;
    preprocess_config.expected_height = kBaseInputHeight;
    preprocess_config.expected_width  = kBaseInputWidth;
    const Clock::time_point preprocess_start = Clock::now();
    InputBatch batch = PrepareInputBatch(config.image_paths, preprocess_config);
    const Clock::time_point preprocess_end = Clock::now();

    TensorRtEngine engine(config.engine_path, /*is_base=*/true);

    PreparedInferenceConfig prepared_config;
    prepared_config.batch = std::move(batch);
    prepared_config.camera_ids = ResolveCameraIds(config.camera_ids);
    prepared_config.out_dir = config.out_dir;
    prepared_config.save_ply = config.save_ply;
    prepared_config.viewer_align = config.viewer_align;
    prepared_config.preprocess_ms = MillisecondsSince(preprocess_start, preprocess_end);
    return RunPreparedInferenceBase(prepared_config, engine);
}

}  // namespace da3
