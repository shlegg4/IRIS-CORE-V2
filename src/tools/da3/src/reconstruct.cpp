#include "da3/reconstruct.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include <Eigen/Eigenvalues>

namespace da3 {

namespace {

constexpr std::size_t kGroundPlaneMaxSamplePoints = 20000;
constexpr int kGroundPlaneRansacIterations = 96;
constexpr float kGroundPlaneMinNormalDot = 0.75f;
constexpr float kGroundPlaneMinInlierRatio = 0.08f;

struct PlaneModel {
    Eigen::Vector3f normal = Eigen::Vector3f::UnitY();
    float offset = 0.0f;
};

struct GroundPlaneEstimate {
    bool plane_found = false;
    PlaneModel plane_provisional;
    Eigen::Vector3f point_provisional = Eigen::Vector3f::Zero();
    Eigen::Vector3f camera_centroid_provisional = Eigen::Vector3f::Zero();
    Eigen::Vector3f origin_provisional = Eigen::Vector3f::Zero();
    GroundAlignmentInfo info;
};

void ExpectDims(const TensorOutput& tensor, const std::vector<int64_t>& expected, const char* name) {
    if (tensor.dims != expected) {
        throw std::runtime_error(
            std::string("Unexpected dims for ") + name + "."
        );
    }
}

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

Eigen::Matrix3f LoadIntrinsics(const std::vector<float>& values, const int view_index) {
    Eigen::Matrix3f matrix = Eigen::Matrix3f::Zero();
    const std::size_t base = static_cast<std::size_t>(view_index * 9);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            matrix(row, col) = values[base + static_cast<std::size_t>(row * 3 + col)];
        }
    }
    return matrix;
}

Eigen::Matrix4f LoadExtrinsics(const std::vector<float>& values, const int view_index) {
    Eigen::Matrix4f matrix = Eigen::Matrix4f::Zero();
    const std::size_t base = static_cast<std::size_t>(view_index * 16);
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            matrix(row, col) = values[base + static_cast<std::size_t>(row * 4 + col)];
        }
    }
    return matrix;
}

std::vector<float> CopyDepth(const TensorOutput& depth_output) {
    return depth_output.values;
}

Eigen::Vector3f TransformPoint(const Eigen::Matrix4f& transform, const Eigen::Vector3f& point) {
    return (transform * point.homogeneous()).head<3>();
}

Eigen::Vector3f ComputeCentroid(const std::vector<Eigen::Vector3f>& points) {
    if (points.empty()) {
        return Eigen::Vector3f::Zero();
    }

    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    for (const Eigen::Vector3f& point : points) {
        centroid += point;
    }
    centroid /= static_cast<float>(points.size());
    return centroid;
}

Eigen::Matrix4f BuildViewerFrameAlignment(const Eigen::Matrix4f& first_w2c) {
    Eigen::Matrix4f axis_flip = Eigen::Matrix4f::Identity();
    axis_flip(1, 1) = -1.0f;
    axis_flip(2, 2) = -1.0f;
    return axis_flip * first_w2c;
}

float ComputeAdaptiveConfidenceThreshold(
    const std::vector<float>& confidence,
    const std::vector<std::uint8_t>& sky_mask,
    const std::vector<std::uint8_t>& content_mask,
    const ReconstructionConfig& config
) {
    std::vector<float> conf_pixels;
    conf_pixels.reserve(confidence.size());

    std::size_t non_sky_count = 0;
    for (std::size_t idx = 0; idx < sky_mask.size(); ++idx) {
        if (!content_mask.empty() && !content_mask[idx]) {
            continue;
        }
        if (!sky_mask[idx]) {
            ++non_sky_count;
        }
    }

    if (non_sky_count > 10) {
        for (std::size_t idx = 0; idx < confidence.size(); ++idx) {
            if ((!content_mask.empty() && !content_mask[idx]) || sky_mask[idx]) {
                continue;
            }
            conf_pixels.push_back(confidence[idx]);
        }
    } else {
        for (std::size_t idx = 0; idx < confidence.size(); ++idx) {
            if (!content_mask.empty() && !content_mask[idx]) {
                continue;
            }
            conf_pixels.push_back(confidence[idx]);
        }
    }

    if (conf_pixels.empty()) {
        return config.conf_thresh;
    }

    const float lower = Percentile(conf_pixels, config.conf_thresh_percentile);
    const float upper = Percentile(conf_pixels, config.ensure_thresh_percentile);
    return std::min(std::max(config.conf_thresh, lower), upper);
}

std::vector<std::uint8_t> BuildSkyMask(const TensorOutput& sky_output) {
    std::vector<std::uint8_t> sky_mask(sky_output.values.size(), 0);
    for (std::size_t idx = 0; idx < sky_output.values.size(); ++idx) {
        sky_mask[idx] = sky_output.values[idx] >= 0.5f ? 1U : 0U;
    }
    return sky_mask;
}

bool IsContentPixel(const ProcessedView& view, const int x, const int y) {
    return x >= view.content_x &&
           y >= view.content_y &&
           x < view.content_x + view.content_width &&
           y < view.content_y + view.content_height;
}

std::vector<std::uint8_t> BuildContentMask(const std::vector<ProcessedView>& processed_views) {
    std::vector<std::uint8_t> content_mask(
        static_cast<std::size_t>(kNumViews * kInputHeight * kInputWidth),
        0
    );

    for (int view = 0; view < kNumViews; ++view) {
        const ProcessedView& processed_view = processed_views[static_cast<std::size_t>(view)];
        for (int y = processed_view.content_y;
             y < processed_view.content_y + processed_view.content_height;
             ++y) {
            for (int x = processed_view.content_x;
                 x < processed_view.content_x + processed_view.content_width;
                 ++x) {
                const std::size_t linear = static_cast<std::size_t>(
                    view * kInputHeight * kInputWidth + y * kInputWidth + x
                );
                content_mask[linear] = 1U;
            }
        }
    }

    return content_mask;
}

void ApplySkyDepthFill(
    std::vector<float>& depth,
    const std::vector<std::uint8_t>& sky_mask,
    const std::vector<std::uint8_t>& content_mask,
    const ReconstructionConfig& config
) {
    std::vector<float> valid_depth;
    valid_depth.reserve(depth.size());
    for (std::size_t idx = 0; idx < depth.size(); ++idx) {
        if ((!content_mask.empty() && !content_mask[idx]) || sky_mask[idx]) {
            continue;
        }
        const float value = depth[idx];
        if (std::isfinite(value) && value > 0.0f) {
            valid_depth.push_back(value);
        }
    }
    if (valid_depth.empty()) {
        return;
    }

    const float replacement = Percentile(valid_depth, config.sky_depth_percentile);
    for (std::size_t idx = 0; idx < depth.size(); ++idx) {
        if ((!content_mask.empty() && !content_mask[idx]) || sky_mask[idx]) {
            depth[idx] = replacement;
        }
    }
}

Eigen::Matrix4f ComputeViewerAlignment(
    const Eigen::Matrix4f& first_w2c,
    const std::vector<PointVertex>& vertices
) {
    const Eigen::Matrix4f viewer_frame = BuildViewerFrameAlignment(first_w2c);

    Eigen::Vector3f center = Eigen::Vector3f::Zero();
    if (!vertices.empty()) {
        std::vector<float> xs;
        std::vector<float> ys;
        std::vector<float> zs;
        xs.reserve(vertices.size());
        ys.reserve(vertices.size());
        zs.reserve(vertices.size());
        for (const PointVertex& vertex : vertices) {
            const Eigen::Vector3f aligned = TransformPoint(viewer_frame, vertex.position);
            xs.push_back(aligned.x());
            ys.push_back(aligned.y());
            zs.push_back(aligned.z());
        }
        center.x() = Percentile(std::move(xs), 50.0f);
        center.y() = Percentile(std::move(ys), 50.0f);
        center.z() = Percentile(std::move(zs), 50.0f);
    }

    Eigen::Matrix4f translate = Eigen::Matrix4f::Identity();
    translate.block<3, 1>(0, 3) = -center;
    return translate * viewer_frame;
}

std::vector<Eigen::Vector3f> BuildCameraPositions(const Mv4Outputs& outputs) {
    std::vector<Eigen::Vector3f> camera_positions;
    camera_positions.reserve(kNumViews);
    for (int view = 0; view < kNumViews; ++view) {
        const Eigen::Matrix4f c2w = LoadExtrinsics(outputs.extrinsics.values, view).inverse();
        camera_positions.push_back(c2w.block<3, 1>(0, 3));
    }
    return camera_positions;
}

std::vector<Eigen::Vector3f> SampleTransformedPoints(
    const std::vector<PointVertex>& vertices,
    const Eigen::Matrix4f& transform
) {
    if (vertices.empty()) {
        return {};
    }

    const std::size_t stride = std::max<std::size_t>(
        1,
        (vertices.size() + kGroundPlaneMaxSamplePoints - 1) / kGroundPlaneMaxSamplePoints
    );

    std::vector<Eigen::Vector3f> sample_points;
    sample_points.reserve((vertices.size() + stride - 1) / stride);
    for (std::size_t index = 0; index < vertices.size(); index += stride) {
        sample_points.push_back(TransformPoint(transform, vertices[index].position));
    }
    return sample_points;
}

float ComputeSceneExtent(const std::vector<Eigen::Vector3f>& points) {
    if (points.empty()) {
        return 1.0f;
    }

    Eigen::Vector3f min_point = points.front();
    Eigen::Vector3f max_point = points.front();
    for (const Eigen::Vector3f& point : points) {
        min_point = min_point.cwiseMin(point);
        max_point = max_point.cwiseMax(point);
    }
    return (max_point - min_point).norm();
}

bool BuildPlaneFromPoints(
    const Eigen::Vector3f& a,
    const Eigen::Vector3f& b,
    const Eigen::Vector3f& c,
    PlaneModel& plane
) {
    Eigen::Vector3f normal = (b - a).cross(c - a);
    const float norm = normal.norm();
    if (!std::isfinite(norm) || norm < 1e-5f) {
        return false;
    }

    plane.normal = normal / norm;
    plane.offset = -plane.normal.dot(a);
    return std::isfinite(plane.offset);
}

PlaneModel OrientPlaneTowardCameras(
    PlaneModel plane,
    const std::vector<Eigen::Vector3f>& camera_positions
) {
    if (camera_positions.empty()) {
        return plane;
    }

    float mean_height = 0.0f;
    for (const Eigen::Vector3f& camera : camera_positions) {
        mean_height += plane.normal.dot(camera) + plane.offset;
    }
    mean_height /= static_cast<float>(camera_positions.size());

    if (mean_height < 0.0f) {
        plane.normal = -plane.normal;
        plane.offset = -plane.offset;
    }
    return plane;
}

float SignedDistanceToPlane(const PlaneModel& plane, const Eigen::Vector3f& point) {
    return plane.normal.dot(point) + plane.offset;
}

bool ComputeCameraHeightStats(
    const PlaneModel& plane,
    const std::vector<Eigen::Vector3f>& camera_positions,
    const float min_clearance,
    float& mean_height,
    float& stddev
) {
    if (camera_positions.empty()) {
        return false;
    }

    std::vector<float> heights;
    heights.reserve(camera_positions.size());
    float sum = 0.0f;
    float min_height = std::numeric_limits<float>::infinity();
    for (const Eigen::Vector3f& camera : camera_positions) {
        const float height = SignedDistanceToPlane(plane, camera);
        if (!std::isfinite(height)) {
            return false;
        }
        heights.push_back(height);
        sum += height;
        min_height = std::min(min_height, height);
    }

    mean_height = sum / static_cast<float>(heights.size());
    if (min_height <= min_clearance || mean_height <= min_clearance) {
        return false;
    }

    float variance = 0.0f;
    for (const float height : heights) {
        const float delta = height - mean_height;
        variance += delta * delta;
    }
    variance /= static_cast<float>(heights.size());
    stddev = std::sqrt(std::max(variance, 0.0f));

    return stddev <= std::max(min_clearance * 4.0f, mean_height * 0.35f);
}

std::size_t CountPlaneInliers(
    const std::vector<Eigen::Vector3f>& points,
    const PlaneModel& plane,
    const float distance_threshold
) {
    std::size_t inlier_count = 0;
    for (const Eigen::Vector3f& point : points) {
        if (std::abs(SignedDistanceToPlane(plane, point)) <= distance_threshold) {
            ++inlier_count;
        }
    }
    return inlier_count;
}

bool FitPlaneToPoints(
    const std::vector<Eigen::Vector3f>& points,
    PlaneModel& plane,
    Eigen::Vector3f& point_on_plane
) {
    if (points.size() < 3) {
        return false;
    }

    point_on_plane = ComputeCentroid(points);
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (const Eigen::Vector3f& point : points) {
        const Eigen::Vector3f delta = point - point_on_plane;
        covariance += delta * delta.transpose();
    }

    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    if (solver.info() != Eigen::Success) {
        return false;
    }

    Eigen::Vector3f normal = solver.eigenvectors().col(0);
    const float norm = normal.norm();
    if (!std::isfinite(norm) || norm < 1e-5f) {
        return false;
    }

    plane.normal = normal / norm;
    plane.offset = -plane.normal.dot(point_on_plane);
    return std::isfinite(plane.offset);
}

GroundPlaneEstimate EstimateGroundPlane(
    const Eigen::Matrix4f& provisional_alignment,
    const std::vector<PointVertex>& vertices,
    const std::vector<Eigen::Vector3f>& camera_positions_raw
) {
    GroundPlaneEstimate estimate;
    estimate.info.camera_centroid = ComputeCentroid(camera_positions_raw);

    if (vertices.empty() || camera_positions_raw.empty()) {
        return estimate;
    }

    const std::vector<Eigen::Vector3f> sample_points =
        SampleTransformedPoints(vertices, provisional_alignment);
    estimate.info.sampled_point_count = sample_points.size();
    if (sample_points.size() < 64) {
        return estimate;
    }

    std::vector<Eigen::Vector3f> camera_positions_provisional;
    camera_positions_provisional.reserve(camera_positions_raw.size());
    for (const Eigen::Vector3f& camera_position_raw : camera_positions_raw) {
        camera_positions_provisional.push_back(
            TransformPoint(provisional_alignment, camera_position_raw)
        );
    }
    estimate.camera_centroid_provisional = ComputeCentroid(camera_positions_provisional);

    const float scene_extent = ComputeSceneExtent(sample_points);
    const float distance_threshold = std::clamp(scene_extent * 0.01f, 0.02f, 0.12f);
    const float max_camera_y = std::max_element(
        camera_positions_provisional.begin(),
        camera_positions_provisional.end(),
        [](const Eigen::Vector3f& lhs, const Eigen::Vector3f& rhs) {
            return lhs.y() < rhs.y();
        }
    )->y();

    std::vector<float> sample_y;
    sample_y.reserve(sample_points.size());
    for (const Eigen::Vector3f& point : sample_points) {
        sample_y.push_back(point.y());
    }

    const float candidate_y_limit = std::min(
        max_camera_y + distance_threshold * 4.0f,
        Percentile(sample_y, 65.0f)
    );

    std::vector<std::size_t> candidate_indices;
    candidate_indices.reserve(sample_points.size());
    for (std::size_t index = 0; index < sample_points.size(); ++index) {
        if (sample_points[index].y() <= candidate_y_limit) {
            candidate_indices.push_back(index);
        }
    }
    if (candidate_indices.size() < 3) {
        candidate_indices.resize(sample_points.size());
        for (std::size_t index = 0; index < sample_points.size(); ++index) {
            candidate_indices[index] = index;
        }
    }

    PlaneModel best_plane;
    std::size_t best_inlier_count = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    float best_height_stddev = std::numeric_limits<float>::infinity();
    std::mt19937 rng(0x44413347);
    std::uniform_int_distribution<std::size_t> dist(0, candidate_indices.size() - 1);

    for (int iteration = 0; iteration < kGroundPlaneRansacIterations; ++iteration) {
        const std::size_t ia = candidate_indices[dist(rng)];
        std::size_t ib = candidate_indices[dist(rng)];
        std::size_t ic = candidate_indices[dist(rng)];
        if (ia == ib || ia == ic || ib == ic) {
            --iteration;
            continue;
        }

        PlaneModel plane;
        if (!BuildPlaneFromPoints(sample_points[ia], sample_points[ib], sample_points[ic], plane)) {
            continue;
        }

        if (std::abs(plane.normal.y()) < kGroundPlaneMinNormalDot) {
            continue;
        }

        plane = OrientPlaneTowardCameras(plane, camera_positions_provisional);

        float mean_camera_height = 0.0f;
        float camera_height_stddev = 0.0f;
        if (!ComputeCameraHeightStats(
                plane,
                camera_positions_provisional,
                distance_threshold,
                mean_camera_height,
                camera_height_stddev
            )) {
            continue;
        }

        const std::size_t inlier_count =
            CountPlaneInliers(sample_points, plane, distance_threshold);
        const float inlier_ratio =
            static_cast<float>(inlier_count) / static_cast<float>(sample_points.size());
        if (inlier_ratio < kGroundPlaneMinInlierRatio) {
            continue;
        }

        const float score =
            static_cast<float>(inlier_count) -
            3.0f * (camera_height_stddev / std::max(distance_threshold, 1e-5f));
        if (score > best_score ||
            (std::abs(score - best_score) <= 1e-5f && camera_height_stddev < best_height_stddev)) {
            best_plane = plane;
            best_inlier_count = inlier_count;
            best_score = score;
            best_height_stddev = camera_height_stddev;
        }
    }

    if (best_inlier_count == 0) {
        return estimate;
    }

    std::vector<Eigen::Vector3f> inlier_points;
    inlier_points.reserve(best_inlier_count);
    for (const Eigen::Vector3f& point : sample_points) {
        if (std::abs(SignedDistanceToPlane(best_plane, point)) <= distance_threshold) {
            inlier_points.push_back(point);
        }
    }
    if (inlier_points.size() < 3) {
        return estimate;
    }

    PlaneModel refined_plane = best_plane;
    Eigen::Vector3f point_on_plane = inlier_points.front();
    if (FitPlaneToPoints(inlier_points, refined_plane, point_on_plane)) {
        refined_plane = OrientPlaneTowardCameras(refined_plane, camera_positions_provisional);
        float mean_camera_height = 0.0f;
        float camera_height_stddev = 0.0f;
        if (!ComputeCameraHeightStats(
                refined_plane,
                camera_positions_provisional,
                distance_threshold,
                mean_camera_height,
                camera_height_stddev
            ) || std::abs(refined_plane.normal.y()) < kGroundPlaneMinNormalDot) {
            refined_plane = best_plane;
            point_on_plane = inlier_points.front();
        }
    }

    const std::size_t refined_inlier_count =
        CountPlaneInliers(sample_points, refined_plane, distance_threshold);
    const float refined_inlier_ratio =
        static_cast<float>(refined_inlier_count) / static_cast<float>(sample_points.size());
    if (refined_inlier_ratio < kGroundPlaneMinInlierRatio) {
        return estimate;
    }

    const float centroid_distance =
        SignedDistanceToPlane(refined_plane, estimate.camera_centroid_provisional);
    estimate.origin_provisional =
        estimate.camera_centroid_provisional - centroid_distance * refined_plane.normal;
    estimate.point_provisional = point_on_plane;
    estimate.plane_provisional = refined_plane;
    estimate.plane_found = true;

    const Eigen::Matrix4f provisional_inverse = provisional_alignment.inverse();
    const Eigen::Matrix3f provisional_rotation = provisional_alignment.block<3, 3>(0, 0);
    estimate.info.plane_found = true;
    estimate.info.inlier_count = refined_inlier_count;
    estimate.info.inlier_ratio = refined_inlier_ratio;
    estimate.info.plane_normal =
        (provisional_rotation.transpose() * refined_plane.normal).normalized();
    estimate.info.plane_point = TransformPoint(provisional_inverse, point_on_plane);
    estimate.info.origin = TransformPoint(provisional_inverse, estimate.origin_provisional);
    return estimate;
}

Eigen::Matrix4f ComposeGroundAlignment(
    const Eigen::Matrix4f& provisional_alignment,
    const GroundPlaneEstimate& estimate
) {
    const Eigen::Quaternionf rotation =
        Eigen::Quaternionf::FromTwoVectors(
            estimate.plane_provisional.normal,
            Eigen::Vector3f::UnitY()
        );

    Eigen::Matrix4f rotate = Eigen::Matrix4f::Identity();
    rotate.block<3, 3>(0, 0) = rotation.toRotationMatrix();

    Eigen::Matrix4f translate = Eigen::Matrix4f::Identity();
    translate.block<3, 1>(0, 3) =
        -(rotate.block<3, 3>(0, 0) * estimate.origin_provisional);

    return translate * rotate * provisional_alignment;
}

void ApplyAlignment(
    const Eigen::Matrix4f& alignment,
    std::vector<PointVertex>& vertices
) {
    for (PointVertex& vertex : vertices) {
        vertex.position = TransformPoint(alignment, vertex.position);
    }
}

}  // namespace

ReconstructionResult ReconstructPointCloud(
    const Mv4Outputs& outputs,
    const std::vector<ProcessedView>& processed_views,
    const ReconstructionConfig& config
) {
    if (processed_views.size() != static_cast<std::size_t>(kNumViews)) {
        throw std::runtime_error("Reconstruction expects exactly 4 processed views.");
    }

    ExpectDims(outputs.depth, {1, kNumViews, kInputHeight, kInputWidth}, "depth");
    ExpectDims(outputs.depth_conf, {1, kNumViews, kInputHeight, kInputWidth}, "depth_conf");
    ExpectDims(outputs.sky, {1, kNumViews, kInputHeight, kInputWidth}, "sky");
    ExpectDims(outputs.intrinsics, {1, kNumViews, 3, 3}, "intrinsics");
    ExpectDims(outputs.extrinsics, {1, kNumViews, 4, 4}, "extrinsics");

    std::vector<float> depth = CopyDepth(outputs.depth);
    const std::vector<float>& confidence = outputs.depth_conf.values;
    const std::vector<std::uint8_t> sky_mask = BuildSkyMask(outputs.sky);

    const std::vector<std::uint8_t> content_mask = BuildContentMask(processed_views);

    ApplySkyDepthFill(depth, sky_mask, content_mask, config);
    const float conf_threshold =
        ComputeAdaptiveConfidenceThreshold(confidence, sky_mask, content_mask, config);

    ReconstructionResult result;
    result.applied_conf_threshold = conf_threshold;
    result.vertices.reserve(static_cast<std::size_t>(kNumViews * kInputHeight * kInputWidth));

    for (int view = 0; view < kNumViews; ++view) {
        const Eigen::Matrix3f intrinsics = LoadIntrinsics(outputs.intrinsics.values, view);
        const Eigen::Matrix4f w2c = LoadExtrinsics(outputs.extrinsics.values, view);
        const Eigen::Matrix3f intrinsics_inv = intrinsics.inverse();
        const Eigen::Matrix4f c2w = w2c.inverse();
        const cv::Mat& rgb = processed_views[static_cast<std::size_t>(view)].rgb_u8;

        for (int y = 0; y < kInputHeight; ++y) {
            const cv::Vec3b* row = rgb.ptr<cv::Vec3b>(y);
            for (int x = 0; x < kInputWidth; ++x) {
                if (!IsContentPixel(processed_views[static_cast<std::size_t>(view)], x, y)) {
                    continue;
                }
                const std::size_t linear = static_cast<std::size_t>(
                    view * kInputHeight * kInputWidth + y * kInputWidth + x
                );
                const float depth_value = depth[linear];
                if (!std::isfinite(depth_value) || depth_value <= 0.0f) {
                    continue;
                }
                if (confidence[linear] < conf_threshold) {
                    continue;
                }

                const Eigen::Vector3f pixel(
                    static_cast<float>(x),
                    static_cast<float>(y),
                    1.0f
                );
                const Eigen::Vector3f ray = intrinsics_inv * pixel;
                const Eigen::Vector3f point_camera = ray * depth_value;
                const Eigen::Vector4f point_world =
                    c2w * point_camera.homogeneous();

                PointVertex vertex;
                vertex.position = point_world.head<3>();
                vertex.color = {row[x][0], row[x][1], row[x][2]};
                result.vertices.push_back(vertex);
            }
        }
    }

    const std::vector<Eigen::Vector3f> camera_positions = BuildCameraPositions(outputs);
    result.ground_alignment.camera_centroid = ComputeCentroid(camera_positions);

    if (config.viewer_align) {
        const Eigen::Matrix4f first_w2c = LoadExtrinsics(outputs.extrinsics.values, 0);
        const Eigen::Matrix4f provisional_alignment = BuildViewerFrameAlignment(first_w2c);
        const GroundPlaneEstimate ground_estimate =
            EstimateGroundPlane(provisional_alignment, result.vertices, camera_positions);
        result.ground_alignment = ground_estimate.info;

        if (ground_estimate.plane_found) {
            result.alignment = ComposeGroundAlignment(provisional_alignment, ground_estimate);
        } else {
            result.alignment = ComputeViewerAlignment(first_w2c, result.vertices);
        }

        ApplyAlignment(result.alignment, result.vertices);
    }

    return result;
}

ReconstructionResult ReconstructPointCloudBase(
    const MvBaseOutputs& outputs,
    const std::vector<ProcessedView>& processed_views,
    const ReconstructionConfig& config
) {
    const int num_views = static_cast<int>(processed_views.size());
    if (num_views == 0) {
        throw std::runtime_error("ReconstructPointCloudBase: no processed views provided.");
    }

    const std::size_t pixels_per_view = static_cast<std::size_t>(kBaseInputHeight * kBaseInputWidth);
    const std::size_t expected_depth = static_cast<std::size_t>(num_views) * pixels_per_view;
    if (outputs.depth.values.size() != expected_depth) {
        throw std::runtime_error("ReconstructPointCloudBase: unexpected depth tensor size.");
    }

    std::vector<float> depth = outputs.depth.values;
    const std::vector<float>& confidence = outputs.depth_conf.values;

    const std::vector<std::uint8_t> sky_mask(depth.size(), 0);

    std::vector<std::uint8_t> content_mask(static_cast<std::size_t>(num_views) * pixels_per_view, 0);
    for (int view = 0; view < num_views; ++view) {
        const ProcessedView& pv = processed_views[static_cast<std::size_t>(view)];
        for (int y = pv.content_y; y < pv.content_y + pv.content_height; ++y) {
            for (int x = pv.content_x; x < pv.content_x + pv.content_width; ++x) {
                const std::size_t linear = static_cast<std::size_t>(view) * pixels_per_view
                    + static_cast<std::size_t>(y * kBaseInputWidth + x);
                content_mask[linear] = 1U;
            }
        }
    }

    const float conf_threshold =
        ComputeAdaptiveConfidenceThreshold(confidence, sky_mask, content_mask, config);

    ReconstructionResult result;
    result.applied_conf_threshold = conf_threshold;
    result.vertices.reserve(static_cast<std::size_t>(num_views) * pixels_per_view);

    for (int view = 0; view < num_views; ++view) {
        const Eigen::Matrix3f intrinsics = LoadIntrinsics(outputs.intrinsics.values, view);
        const Eigen::Matrix4f w2c = LoadExtrinsics(outputs.extrinsics.values, view);
        const Eigen::Matrix3f intrinsics_inv = intrinsics.inverse();
        const Eigen::Matrix4f c2w = w2c.inverse();
        const cv::Mat& rgb = processed_views[static_cast<std::size_t>(view)].rgb_u8;

        for (int y = 0; y < kBaseInputHeight; ++y) {
            const cv::Vec3b* row = rgb.ptr<cv::Vec3b>(y);
            for (int x = 0; x < kBaseInputWidth; ++x) {
                if (!IsContentPixel(processed_views[static_cast<std::size_t>(view)], x, y)) {
                    continue;
                }
                const std::size_t linear = static_cast<std::size_t>(view) * pixels_per_view
                    + static_cast<std::size_t>(y * kBaseInputWidth + x);
                const float depth_value = depth[linear];
                if (!std::isfinite(depth_value) || depth_value <= 0.0f) {
                    continue;
                }
                if (confidence[linear] < conf_threshold) {
                    continue;
                }

                const Eigen::Vector3f pixel(
                    static_cast<float>(x),
                    static_cast<float>(y),
                    1.0f
                );
                const Eigen::Vector3f ray = intrinsics_inv * pixel;
                const Eigen::Vector3f point_camera = ray * depth_value;
                const Eigen::Vector4f point_world = c2w * point_camera.homogeneous();

                PointVertex vertex;
                vertex.position = point_world.head<3>();
                vertex.color = {row[x][0], row[x][1], row[x][2]};
                result.vertices.push_back(vertex);
            }
        }
    }

    std::vector<Eigen::Vector3f> camera_positions;
    camera_positions.reserve(static_cast<std::size_t>(num_views));
    for (int view = 0; view < num_views; ++view) {
        const Eigen::Matrix4f c2w = LoadExtrinsics(outputs.extrinsics.values, view).inverse();
        camera_positions.push_back(c2w.block<3, 1>(0, 3));
    }
    result.ground_alignment.camera_centroid = ComputeCentroid(camera_positions);

    if (config.viewer_align) {
        const Eigen::Matrix4f first_w2c = LoadExtrinsics(outputs.extrinsics.values, 0);
        const Eigen::Matrix4f provisional_alignment = BuildViewerFrameAlignment(first_w2c);
        const GroundPlaneEstimate ground_estimate =
            EstimateGroundPlane(provisional_alignment, result.vertices, camera_positions);
        result.ground_alignment = ground_estimate.info;

        if (ground_estimate.plane_found) {
            result.alignment = ComposeGroundAlignment(provisional_alignment, ground_estimate);
        } else {
            result.alignment = ComputeViewerAlignment(first_w2c, result.vertices);
        }

        ApplyAlignment(result.alignment, result.vertices);
    }

    return result;
}

}  // namespace da3
