#include "iris/stages/MultiviewPoseStage.hpp"
#include "iris/stages/pose/TensorRtMultiviewEngine.hpp"
#include "iris/infrastructure/gpu/CudaResources.hpp"
#include "iris/calibration/RigCalibration.hpp"

#include <filesystem>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <ranges>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#ifdef IRIS_HAS_TENSORRT
#include <cmath>
#endif

namespace iris {
namespace {
constexpr std::size_t candidate_capacity = 10;
constexpr std::size_t joint_capacity = coco_joint_count;
constexpr std::size_t maximum_view_capacity = 10;
constexpr float invalid_match_cost = std::numeric_limits<float>::infinity();
constexpr std::size_t minimum_epipolar_joints = 5;
constexpr float epipolar_trim_fraction = 0.2F;

using Projection = Eigen::Matrix<float, 3, 4, Eigen::RowMajor>;

Eigen::Matrix3f letterbox_intrinsics(const PoseConfig::CameraCalibration& calibration,
                                    Extent2D extent) {
    const float scale = std::min(640.0F / static_cast<float>(extent.width),
                                 640.0F / static_cast<float>(extent.height));
    const int resized_width = std::max(1, static_cast<int>(extent.width * scale + 0.5F));
    const int resized_height = std::max(1, static_cast<int>(extent.height * scale + 0.5F));
    const float pad_x = static_cast<float>((640 - resized_width) / 2);
    const float pad_y = static_cast<float>((640 - resized_height) / 2);
    Eigen::Matrix3f result = Eigen::Matrix3f::Identity();
    result(0, 0) = calibration.intrinsics[0] * scale;
    result(1, 1) = calibration.intrinsics[4] * scale;
    result(0, 2) = calibration.intrinsics[2] * scale + pad_x;
    result(1, 2) = calibration.intrinsics[5] * scale + pad_y;
    return result;
}

Projection projection_matrix(const PoseConfig::CameraCalibration& calibration,
                             const Eigen::Matrix3f& intrinsics) {
    Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> rotation_data(calibration.R_w2c.data());
    const Eigen::Matrix3f rotation = rotation_data;
    Eigen::Vector3f translation(calibration.t_w2c[0], calibration.t_w2c[1], calibration.t_w2c[2]);
    Eigen::Matrix<float, 3, 4> extrinsic;
    extrinsic.block<3, 3>(0, 0) = rotation;
    extrinsic.col(3) = translation;
    return intrinsics * extrinsic;
}

Eigen::Matrix3f fundamental_matrix(const PoseConfig::CameraCalibration& first,
                                   const Eigen::Matrix3f& first_k,
                                   const PoseConfig::CameraCalibration& second,
                                   const Eigen::Matrix3f& second_k) {
    Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> first_rotation_data(first.R_w2c.data());
    Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> second_rotation_data(second.R_w2c.data());
    const Eigen::Matrix3f relative_rotation = second_rotation_data * first_rotation_data.transpose();
    const Eigen::Vector3f first_translation(first.t_w2c[0], first.t_w2c[1], first.t_w2c[2]);
    const Eigen::Vector3f second_translation(second.t_w2c[0], second.t_w2c[1], second.t_w2c[2]);
    const Eigen::Vector3f relative_translation = second_translation - relative_rotation * first_translation;
    Eigen::Matrix3f cross;
    cross << 0.0F, -relative_translation.z(), relative_translation.y(),
             relative_translation.z(), 0.0F, -relative_translation.x(),
             -relative_translation.y(), relative_translation.x(), 0.0F;
    return second_k.inverse().transpose() * cross * relative_rotation * first_k.inverse();
}

float symmetric_epipolar_distance(const Eigen::Matrix3f& fundamental,
                                  float x1, float y1, float x2, float y2) {
    const Eigen::Vector3f first(x1, y1, 1.0F), second(x2, y2, 1.0F);
    const Eigen::Vector3f line_second = fundamental * first;
    const Eigen::Vector3f line_first = fundamental.transpose() * second;
    const float residual = std::abs(second.dot(line_second));
    return 0.5F * (residual / std::max(1e-6F, line_second.head<2>().norm()) +
                   residual / std::max(1e-6F, line_first.head<2>().norm()));
}

float detection_pair_cost(const TensorRtMultiviewResult& result,
                          std::size_t first_view, int first_candidate,
                          std::size_t second_view, int second_candidate,
                          const Eigen::Matrix3f& fundamental, float minimum_score) {
    if (first_candidate < 0 || second_candidate < 0) return invalid_match_cost;
    std::array<std::pair<float, float>, joint_capacity> residuals{};
    std::size_t count = 0;
    const auto first_base = (first_view * candidate_capacity + static_cast<std::size_t>(first_candidate)) * joint_capacity;
    const auto second_base = (second_view * candidate_capacity + static_cast<std::size_t>(second_candidate)) * joint_capacity;
    for (std::size_t joint = 0; joint < joint_capacity; ++joint) {
        const auto first_index = first_base + joint;
        const auto second_index = second_base + joint;
        const float first_score = result.keypoint_scores[first_index];
        const float second_score = result.keypoint_scores[second_index];
        const float x1 = result.keypoints[first_index * 2];
        const float y1 = result.keypoints[first_index * 2 + 1];
        const float x2 = result.keypoints[second_index * 2];
        const float y2 = result.keypoints[second_index * 2 + 1];
        if (!std::isfinite(first_score) || !std::isfinite(second_score) ||
            first_score < minimum_score || second_score < minimum_score ||
            !std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(x2) || !std::isfinite(y2))
            continue;
        const float residual = symmetric_epipolar_distance(fundamental, x1, y1, x2, y2);
        if (!std::isfinite(residual)) continue;
        const float confidence = std::sqrt(std::clamp(first_score, 0.0F, 1.0F) *
                                           std::clamp(second_score, 0.0F, 1.0F));
        if (!(confidence > 0.0F)) continue;
        residuals[count++] = {residual, confidence};
    }
    if (count < minimum_epipolar_joints) return invalid_match_cost;

    std::sort(residuals.begin(), residuals.begin() + static_cast<std::ptrdiff_t>(count),
              [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    float total_weight = 0.0F;
    for (std::size_t i = 0; i < count; ++i) total_weight += residuals[i].second;
    const float retained_weight = total_weight * (1.0F - epipolar_trim_fraction);
    float accumulated_weight = 0.0F;
    float weighted_residual = 0.0F;
    for (std::size_t i = 0; i < count && accumulated_weight < retained_weight; ++i) {
        const float weight = std::min(residuals[i].second, retained_weight - accumulated_weight);
        weighted_residual += residuals[i].first * weight;
        accumulated_weight += weight;
    }
    return accumulated_weight > 0.0F ? weighted_residual / accumulated_weight : invalid_match_cost;
}

struct DetectionEdge {
    std::size_t first_view{};
    std::size_t first_candidate{};
    std::size_t second_view{};
    std::size_t second_candidate{};
    float cost{};
};

std::vector<std::vector<int>> associate_detections(
    const TensorRtMultiviewResult& result, std::size_t view_count,
    std::size_t max_persons,
    const std::vector<PoseConfig::CameraCalibration>& cameras,
    const std::vector<Eigen::Matrix3f>& fundamentals,
    float gate, float minimum_score) {
    const auto node_count = view_count * candidate_capacity;
    std::vector<float> pair_costs(node_count * node_count, invalid_match_cost);
    const auto pair_cost = [&](std::size_t view_a, std::size_t candidate_a,
                               std::size_t view_b, std::size_t candidate_b) {
        return pair_costs[(view_a * candidate_capacity + candidate_a) * node_count +
                          view_b * candidate_capacity + candidate_b];
    };
    std::vector<DetectionEdge> edges;
    for (std::size_t first_view = 0; first_view < view_count; ++first_view) {
        for (std::size_t second_view = first_view + 1; second_view < view_count; ++second_view) {
            const auto& fundamental = fundamentals[first_view * view_count + second_view];
            for (std::size_t first = 0; first < candidate_capacity; ++first) {
                if (!result.candidate_valid[first_view * candidate_capacity + first]) continue;
                for (std::size_t second = 0; second < candidate_capacity; ++second) {
                    if (!result.candidate_valid[second_view * candidate_capacity + second]) continue;
                    const float cost = detection_pair_cost(result, first_view,
                        static_cast<int>(first), second_view, static_cast<int>(second),
                        fundamental, minimum_score);
                    auto& forward = pair_costs[(first_view * candidate_capacity + first) * node_count +
                                               second_view * candidate_capacity + second];
                    auto& reverse = pair_costs[(second_view * candidate_capacity + second) * node_count +
                                               first_view * candidate_capacity + first];
                    forward = reverse = cost;
                    if (std::isfinite(cost) && cost < gate)
                        edges.push_back({first_view, first, second_view, second, cost});
                }
            }
        }
    }

    const auto canonical_edge_key = [&](const DetectionEdge& edge) {
        const auto first_id = cameras[edge.first_view].camera_id;
        const auto second_id = cameras[edge.second_view].camera_id;
        if (first_id < second_id)
            return std::tuple{first_id, second_id, edge.first_candidate, edge.second_candidate};
        return std::tuple{second_id, first_id, edge.second_candidate, edge.first_candidate};
    };
    std::ranges::sort(edges, [&](const DetectionEdge& lhs, const DetectionEdge& rhs) {
        if (lhs.cost != rhs.cost) return lhs.cost < rhs.cost;
        return canonical_edge_key(lhs) < canonical_edge_key(rhs);
    });

    std::vector<std::vector<int>> tracks;
    std::vector<bool> active;
    std::vector<int> node_track(node_count, -1);
    const auto nodes_compatible = [&](const std::vector<int>& track,
                                      std::size_t view, std::size_t candidate) {
        if (track[view] >= 0) return false;
        for (std::size_t other_view = 0; other_view < view_count; ++other_view) {
            if (track[other_view] < 0) continue;
            const auto cost = pair_cost(other_view, static_cast<std::size_t>(track[other_view]),
                                        view, candidate);
            if (!std::isfinite(cost) || cost >= gate) return false;
        }
        return true;
    };
    const auto tracks_compatible = [&](const std::vector<int>& first,
                                       const std::vector<int>& second) {
        for (std::size_t first_view = 0; first_view < view_count; ++first_view) {
            if (first[first_view] < 0) continue;
            if (second[first_view] >= 0) return false;
            for (std::size_t second_view = 0; second_view < view_count; ++second_view) {
                if (second[second_view] < 0) continue;
                const auto cost = pair_cost(first_view, static_cast<std::size_t>(first[first_view]),
                                            second_view, static_cast<std::size_t>(second[second_view]));
                if (!std::isfinite(cost) || cost >= gate) return false;
            }
        }
        return true;
    };

    for (const auto& edge : edges) {
        const auto first_node = edge.first_view * candidate_capacity + edge.first_candidate;
        const auto second_node = edge.second_view * candidate_capacity + edge.second_candidate;
        const int first_track = node_track[first_node];
        const int second_track = node_track[second_node];
        if (first_track < 0 && second_track < 0) {
            std::vector<int> track(view_count, -1);
            track[edge.first_view] = static_cast<int>(edge.first_candidate);
            track[edge.second_view] = static_cast<int>(edge.second_candidate);
            const auto index = static_cast<int>(tracks.size());
            tracks.push_back(std::move(track));
            active.push_back(true);
            node_track[first_node] = node_track[second_node] = index;
            continue;
        }
        if (first_track >= 0 && second_track < 0) {
            auto& track = tracks[static_cast<std::size_t>(first_track)];
            if (nodes_compatible(track, edge.second_view, edge.second_candidate)) {
                track[edge.second_view] = static_cast<int>(edge.second_candidate);
                node_track[second_node] = first_track;
            }
            continue;
        }
        if (first_track < 0 && second_track >= 0) {
            auto& track = tracks[static_cast<std::size_t>(second_track)];
            if (nodes_compatible(track, edge.first_view, edge.first_candidate)) {
                track[edge.first_view] = static_cast<int>(edge.first_candidate);
                node_track[first_node] = second_track;
            }
            continue;
        }
        if (first_track == second_track) continue;
        auto& first = tracks[static_cast<std::size_t>(first_track)];
        auto& second = tracks[static_cast<std::size_t>(second_track)];
        if (!active[static_cast<std::size_t>(first_track)] ||
            !active[static_cast<std::size_t>(second_track)] ||
            !tracks_compatible(first, second)) continue;
        for (std::size_t view = 0; view < view_count; ++view) {
            if (second[view] < 0) continue;
            first[view] = second[view];
            node_track[view * candidate_capacity + static_cast<std::size_t>(second[view])] = first_track;
            second[view] = -1;
        }
        active[static_cast<std::size_t>(second_track)] = false;
    }

    struct RankedTrack {
        std::vector<int> assignments;
        std::size_t view_count{};
        float mean_cost{};
    };
    std::vector<RankedTrack> ranked;
    for (std::size_t index = 0; index < tracks.size(); ++index) {
        if (!active[index]) continue;
        std::size_t assigned_views = 0, pairs = 0;
        float total_cost = 0.0F;
        for (std::size_t first_view = 0; first_view < view_count; ++first_view) {
            if (tracks[index][first_view] < 0) continue;
            ++assigned_views;
            for (std::size_t second_view = first_view + 1; second_view < view_count; ++second_view) {
                if (tracks[index][second_view] < 0) continue;
                total_cost += pair_cost(first_view, static_cast<std::size_t>(tracks[index][first_view]),
                                        second_view, static_cast<std::size_t>(tracks[index][second_view]));
                ++pairs;
            }
        }
        if (assigned_views >= 2)
            ranked.push_back({tracks[index], assigned_views, pairs ? total_cost / static_cast<float>(pairs) : gate});
    }
    std::vector<std::size_t> canonical_views(view_count);
    std::iota(canonical_views.begin(), canonical_views.end(), 0);
    std::ranges::sort(canonical_views, [&](std::size_t lhs, std::size_t rhs) {
        return cameras[lhs].camera_id < cameras[rhs].camera_id;
    });
    std::ranges::sort(ranked, [&](const RankedTrack& lhs, const RankedTrack& rhs) {
        if (lhs.view_count != rhs.view_count) return lhs.view_count > rhs.view_count;
        if (lhs.mean_cost != rhs.mean_cost) return lhs.mean_cost < rhs.mean_cost;
        for (const auto view : canonical_views) {
            const auto lhs_value = lhs.assignments[view] < 0 ? candidate_capacity :
                static_cast<std::size_t>(lhs.assignments[view]);
            const auto rhs_value = rhs.assignments[view] < 0 ? candidate_capacity :
                static_cast<std::size_t>(rhs.assignments[view]);
            if (lhs_value != rhs_value) return lhs_value < rhs_value;
        }
        return false;
    });
    std::vector<std::vector<int>> output;
    output.reserve(std::min(max_persons, ranked.size()));
    for (std::size_t i = 0; i < ranked.size() && i < max_persons; ++i)
        output.push_back(std::move(ranked[i].assignments));
    return output;
}

std::optional<Eigen::Vector3f> triangulate_joint(
    const TensorRtMultiviewResult& result, const std::vector<int>& track, std::size_t joint,
    const std::vector<Projection>& projections, float minimum_score, float max_reprojection_error) {
    Eigen::Matrix4f normal = Eigen::Matrix4f::Zero();
    std::size_t observations = 0;
    for (std::size_t view = 0; view < track.size(); ++view) {
        if (track[view] < 0) continue;
        const auto index = (view * candidate_capacity + static_cast<std::size_t>(track[view])) * joint_capacity + joint;
        const float score = result.keypoint_scores[index];
        const float x = result.keypoints[index * 2], y = result.keypoints[index * 2 + 1];
        if (!std::isfinite(score) || score < minimum_score || !std::isfinite(x) || !std::isfinite(y)) continue;
        const auto& p = projections[view];
        const Eigen::Vector4f row_x(x * p(2, 0) - p(0, 0), x * p(2, 1) - p(0, 1),
                                    x * p(2, 2) - p(0, 2), x * p(2, 3) - p(0, 3));
        const Eigen::Vector4f row_y(y * p(2, 0) - p(1, 0), y * p(2, 1) - p(1, 1),
                                    y * p(2, 2) - p(1, 2), y * p(2, 3) - p(1, 3));
        normal.noalias() += score * score * (row_x * row_x.transpose() + row_y * row_y.transpose());
        ++observations;
    }
    if (observations < 2) return std::nullopt;
    const Eigen::Matrix3f lhs = normal.block<3, 3>(0, 0);
    const Eigen::Vector3f rhs = -normal.block<3, 1>(0, 3);
    Eigen::FullPivLU<Eigen::Matrix3f> solver(lhs);
    if (solver.rank() < 3) return std::nullopt;
    const Eigen::Vector3f point = solver.solve(rhs);
    if (!point.allFinite()) return std::nullopt;
    for (std::size_t view = 0; view < track.size(); ++view) {
        if (track[view] < 0) continue;
        const auto index = (view * candidate_capacity + static_cast<std::size_t>(track[view])) * joint_capacity + joint;
        const float score = result.keypoint_scores[index];
        if (!std::isfinite(score) || score < minimum_score) continue;
        const auto& p = projections[view];
        const float denominator = p(2, 0) * point.x() + p(2, 1) * point.y() + p(2, 2) * point.z() + p(2, 3);
        if (!(denominator > 1e-5F)) return std::nullopt;
        const float projected_x = (p(0, 0) * point.x() + p(0, 1) * point.y() + p(0, 2) * point.z() + p(0, 3)) / denominator;
        const float projected_y = (p(1, 0) * point.x() + p(1, 1) * point.y() + p(1, 2) * point.z() + p(1, 3)) / denominator;
        const float dx = projected_x - result.keypoints[index * 2];
        const float dy = projected_y - result.keypoints[index * 2 + 1];
        if (!std::isfinite(projected_x) || !std::isfinite(projected_y) ||
            std::hypot(dx, dy) > max_reprojection_error) return std::nullopt;
    }
    return point;
}
}

class MultiviewPoseStage::Impl {
  public:
    explicit Impl(PoseConfig config, infrastructure::metrics::MetricRegistry* metrics)
        : config_(std::move(config)), metrics_enabled_(metrics != nullptr), process_ms_(metrics ? metrics->histogram("iris_pose_process_ms", {1, 2, 5, 10, 20, 50, 100, 250, 500, 1000}) : infrastructure::metrics::Histogram{}),
          capture_to_result_ms_(metrics ? metrics->histogram("iris_pose_capture_to_result_ms", {5, 10, 20, 50, 100, 250, 500, 1000, 2000, 5000}) : infrastructure::metrics::Histogram{}),
          last_capture_to_result_ms_(metrics ? metrics->gauge("iris_pose_last_capture_to_result_ms") : infrastructure::metrics::Gauge{}),
          last_process_ms_(metrics ? metrics->gauge("iris_pose_last_process_ms") : infrastructure::metrics::Gauge{}) {
        if (metrics) {
            const std::array<const char*, 18> names{
                "frame_ready_wait", "preprocess_host", "trt_setup_host", "trt_enqueue_host",
                "download_host", "result_wait_host", "preprocess_stream", "engine_stream",
                "download_stream", "postprocess_cpu", "engine_call_host", "geometry_setup_host",
                "association_stream", "gather_stream", "triangulation_stream", "output_copy_stream",
                "association_host", "triangulation_host"};
            for (std::size_t i=0; i<names.size(); ++i) {
                breakdown_[i] = metrics->histogram(std::string("iris_pose_")+names[i]+"_ms", {0.01,0.05,0.1,0.25,0.5,1,2,3,5,10,20});
                latest_[i] = metrics->gauge(std::string("iris_pose_last_")+names[i]+"_ms");
            }
        }
    }
    void start() {
        if (config_.multiview_engine_path.empty()) return;
        if (config_.max_persons == 0 || config_.max_persons > candidate_capacity)
            throw std::invalid_argument("multiview max_persons must be between 1 and the engine candidate capacity (10)");
        if (config_.multiview_calibration.size() > maximum_view_capacity ||
            (config_.two_d_only ? config_.multiview_calibration.size() != 1
                                : config_.multiview_calibration.size() < 2))
            throw std::invalid_argument(config_.two_d_only
                ? "2-D pose requires exactly one configured camera"
                : "multiview pose requires between two and ten configured cameras");
        if (!(config_.epipolar_gate_px > 0.0F) || !(config_.minimum_joint_confidence >= 0.0F) ||
            !(config_.maximum_reprojection_error_px > 0.0F))
            throw std::invalid_argument("multiview geometry thresholds must be finite and positive");
        if (!std::filesystem::is_regular_file(config_.multiview_engine_path))
            throw std::runtime_error("multiview TensorRT engine does not exist: " + config_.multiview_engine_path.string());
        refresh_calibration();
        for (std::size_t i = 0; i < config_.multiview_calibration.size(); ++i)
            if (const auto& calibration = config_.multiview_calibration[i]; !calibration.calibrated)
                throw std::runtime_error("RTMO multiview requires calibration for every configured camera");
#ifndef IRIS_HAS_TENSORRT
        throw std::runtime_error("multiview TensorRT was configured, but IRIS was built without TensorRT support; set IRIS_TENSORRT_ROOT and rebuild");
#else
        engine_ = std::make_unique<TensorRtMultiviewEngine>(config_);
        started_ = true;
#endif
    }
    void stop() { started_ = false; }
    void process(Packet& packet) {
        const auto started = std::chrono::steady_clock::now();
        if (config_.multiview_engine_path.empty()) return;
        if (!started_) throw std::logic_error("multiview pose stage was not started");
        refresh_calibration();
        const auto view_count = config_.multiview_calibration.size();
        if (packet.frames.size() != view_count)
            throw std::runtime_error(config_.two_d_only
                ? "2-D TensorRT pose requires exactly one frame"
                : "multiview TensorRT engine requires one synchronized frame per calibrated camera");
#ifndef IRIS_HAS_TENSORRT
        (void)packet;
        throw std::runtime_error("multiview TensorRT inference is unavailable in this build");
#else
        std::vector<const void*> buffers(view_count);
        std::vector<std::size_t> strides(view_count);
        std::vector<std::uint32_t> widths(view_count);
        std::vector<std::uint32_t> heights(view_count);
        double ready_wait_ms = 0;
        for (std::size_t i = 0; i < view_count; ++i) {
            const auto id = config_.multiview_calibration[i].camera_id;
            const auto frame = std::ranges::find(packet.frames, id, &Frame::camera);
            if (frame == packet.frames.end() || frame->format != PixelFormat::Bgr8 || !frame->buffer.data)
                throw std::runtime_error("multiview packet is missing a valid calibrated BGR camera frame");
            if (!frame->extent.width || !frame->extent.height || frame->buffer.stride_bytes < static_cast<std::size_t>(frame->extent.width) * 3)
                throw std::runtime_error("multiview frame has invalid dimensions or stride");
            if (frame->ready) {
                const auto begin = std::chrono::steady_clock::now();
                frame->ready->synchronize();
                ready_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-begin).count();
            }
            buffers[i] = frame->buffer.data;
            strides[i] = frame->buffer.stride_bytes;
            widths[i] = frame->extent.width;
            heights[i] = frame->extent.height;
        }
        TensorRtMultiviewResult result;
        const auto engine_start = std::chrono::steady_clock::now();
        engine_->infer(buffers, strides, widths, heights, result);
        const auto postprocess_start = std::chrono::steady_clock::now();

        const auto geometry_start = std::chrono::steady_clock::now();
        std::vector<Eigen::Matrix3f> intrinsics(view_count);
        std::vector<Projection> projections(view_count);
        for (std::size_t view = 0; view < view_count; ++view) {
            intrinsics[view] = letterbox_intrinsics(config_.multiview_calibration[view],
                                                    {widths[view], heights[view]});
            projections[view] = projection_matrix(config_.multiview_calibration[view], intrinsics[view]);
        }
        std::vector<Eigen::Matrix3f> fundamentals(view_count * view_count,
                                                   Eigen::Matrix3f::Zero());
        for (std::size_t first = 0; first < view_count; ++first)
            for (std::size_t second = first + 1; second < view_count; ++second)
                fundamentals[first * view_count + second] = fundamental_matrix(
                    config_.multiview_calibration[first], intrinsics[first],
                    config_.multiview_calibration[second], intrinsics[second]);
        result.timings.geometry_setup_host_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - geometry_start).count();

        const auto association_start = std::chrono::steady_clock::now();
        auto tracks = associate_detections(result, view_count, config_.max_persons,
                                           config_.multiview_calibration,
                                           fundamentals, config_.epipolar_gate_px,
                                           config_.minimum_joint_confidence);
        result.timings.association_host_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - association_start).count();

        std::vector<ViewPose2d> view_poses_2d;
        for (std::size_t view = 0; view < view_count; ++view) {
            const auto& source = *std::ranges::find(packet.frames,
                config_.multiview_calibration[view].camera_id, &Frame::camera);
            const auto& calibration = config_.multiview_calibration[view];
            const float scale = std::min(640.0F / static_cast<float>(source.extent.width),
                                         640.0F / static_cast<float>(source.extent.height));
            const float resized_width = static_cast<float>(std::max(1, static_cast<int>(source.extent.width * scale + 0.5F)));
            const float resized_height = static_cast<float>(std::max(1, static_cast<int>(source.extent.height * scale + 0.5F)));
            for (std::size_t candidate = 0; candidate < candidate_capacity; ++candidate) {
                if (!result.candidate_valid[view * candidate_capacity + candidate]) continue;
                auto& observation = view_poses_2d.emplace_back();
                observation.camera_id = calibration.camera_id;
                observation.person_id = candidate;
                for (std::size_t joint = 0; joint < joint_capacity; ++joint) {
                    const auto score_index = (view * candidate_capacity + candidate) * joint_capacity + joint;
                    const auto point_index = score_index * 2;
                    const float score = result.keypoint_scores[score_index];
                    const float x = result.keypoints[point_index];
                    const float y = result.keypoints[point_index + 1];
                    observation.scores[joint] = score;
                    if (!std::isfinite(score) || !std::isfinite(x) || !std::isfinite(y)) continue;
                    const float model_x = (x - (640.0F - resized_width) * 0.5F) / scale;
                    const float model_y = (y - (640.0F - resized_height) * 0.5F) / scale;
                    const float fx = calibration.intrinsics[0], fy = calibration.intrinsics[4];
                    const float cx = calibration.intrinsics[2], cy = calibration.intrinsics[5];
                    const float xu = (model_x - cx) / fx, yu = (model_y - cy) / fy;
                    const float r2 = xu * xu + yu * yu;
                    const float radial = 1.0F + calibration.distortion[0] * r2 + calibration.distortion[1] * r2 * r2 + calibration.distortion[4] * r2 * r2 * r2;
                    const float xd = xu * radial + 2.0F * calibration.distortion[2] * xu * yu + calibration.distortion[3] * (r2 + 2.0F * xu * xu);
                    const float yd = yu * radial + calibration.distortion[2] * (r2 + 2.0F * yu * yu) + 2.0F * calibration.distortion[3] * xu * yu;
                    observation.points_px[joint] = {fx * xd + cx, fy * yd + cy};
                    observation.valid[joint] = true;
                }
            }
        }
        packet.view_poses_2d = std::move(view_poses_2d);

        const auto triangulation_start = std::chrono::steady_clock::now();
        std::vector<MultiviewPose> poses;
        poses.reserve(tracks.size());
        for (const auto& track : tracks) {
            auto& pose = poses.emplace_back();
            pose.view_camera_ids.reserve(view_count);
            pose.selected_detection_indices = track;
            pose.joint_scores.resize(view_count);
            pose.points_2d_px.resize(view_count);
            pose.point_valid.resize(view_count);
            for (std::size_t view = 0; view < view_count; ++view) {
                pose.view_camera_ids.push_back(config_.multiview_calibration[view].camera_id);
                pose.joint_scores[view].fill(0.0F);
                pose.point_valid[view].fill(false);
                if (track[view] < 0) continue;
                const auto candidate = static_cast<std::size_t>(track[view]);
                for (std::size_t joint = 0; joint < joint_capacity; ++joint) {
                    const auto index = (view * candidate_capacity + candidate) * joint_capacity + joint;
                    const float score = result.keypoint_scores[index];
                    const float x = result.keypoints[index * 2];
                    const float y = result.keypoints[index * 2 + 1];
                    pose.joint_scores[view][joint] = score;
                    if (!std::isfinite(score) || !std::isfinite(x) || !std::isfinite(y)) continue;
                    const auto& calibration = config_.multiview_calibration[view];
                    const auto& source = *std::ranges::find(packet.frames, calibration.camera_id, &Frame::camera);
                    const float scale = std::min(640.0F / static_cast<float>(source.extent.width),
                                                 640.0F / static_cast<float>(source.extent.height));
                    const float resized_width = static_cast<float>(std::max(1, static_cast<int>(source.extent.width * scale + 0.5F)));
                    const float resized_height = static_cast<float>(std::max(1, static_cast<int>(source.extent.height * scale + 0.5F)));
                    const float model_x = (x - (640.0F - resized_width) * 0.5F) / scale;
                    const float model_y = (y - (640.0F - resized_height) * 0.5F) / scale;
                    const float fx = calibration.intrinsics[0], fy = calibration.intrinsics[4];
                    const float cx = calibration.intrinsics[2], cy = calibration.intrinsics[5];
                    const float xu = (model_x - cx) / fx, yu = (model_y - cy) / fy;
                    const float r2 = xu * xu + yu * yu;
                    const float radial = 1.0F + calibration.distortion[0] * r2 + calibration.distortion[1] * r2 * r2 + calibration.distortion[4] * r2 * r2 * r2;
                    const float xd = xu * radial + 2.0F * calibration.distortion[2] * xu * yu + calibration.distortion[3] * (r2 + 2.0F * xu * xu);
                    const float yd = yu * radial + calibration.distortion[2] * (r2 + 2.0F * yu * yu) + 2.0F * calibration.distortion[3] * xu * yu;
                    pose.points_2d_px[view][joint] = {fx * xd + cx, fy * yd + cy};
                    pose.point_valid[view][joint] = true;
                }
            }
            for (std::size_t joint = 0; joint < joint_capacity; ++joint) {
                if (config_.two_d_only) break;
                const auto point = triangulate_joint(result, track, joint, projections,
                    config_.minimum_joint_confidence, config_.maximum_reprojection_error_px);
                if (!point) continue;
                pose.joints_3d[joint] = {point->x(), point->y(), point->z()};
                pose.joint_valid[joint] = true;
            }
            const auto valid_2d = std::ranges::count_if(pose.point_valid, [](const auto& view) {
                return std::ranges::count(view, true) >= 5;
            });
            pose.active = valid_2d >= 1;
        }
        result.timings.triangulation_host_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - triangulation_start).count();
        packet.multiview_poses = std::move(poses);
        const auto finished = std::chrono::steady_clock::now();
        if (metrics_enabled_) {
            const auto& t=result.timings;
            const std::array<double,18> values{ready_wait_ms, t.preprocess_host_ms, t.setup_host_ms,
                t.enqueue_host_ms, t.download_host_ms, t.wait_host_ms, t.preprocess_stream_ms,
                t.engine_stream_ms, t.download_stream_ms,
                std::chrono::duration<double,std::milli>(finished-postprocess_start).count(),
                std::chrono::duration<double,std::milli>(postprocess_start-engine_start).count(),
                t.geometry_setup_host_ms, t.association_stream_ms, t.gather_stream_ms,
                t.triangulation_stream_ms, t.output_copy_stream_ms,
                t.association_host_ms, t.triangulation_host_ms};
            for(std::size_t i=0;i<values.size();++i) {
                breakdown_[i].observe(values[i]); latest_[i].set(values[i]);
            }
            const auto process = std::chrono::duration<double, std::milli>(finished - started).count();
            process_ms_.observe(process);
            last_process_ms_.set(process);
        }
        if (metrics_enabled_ && !packet.frames.empty() && packet.frames.front().timing.estimated_capture_time.time_since_epoch().count() != 0) {
            const auto latency = std::chrono::duration<double, std::milli>(finished - packet.frames.front().timing.estimated_capture_time).count();
            capture_to_result_ms_.observe(latency);
            last_capture_to_result_ms_.set(latency);
        }
#endif
    }
  private:
    void refresh_calibration() {
        if (config_.two_d_only) return;
        if (!config_.calibration_store) return;
        const auto rig=config_.calibration_store->snapshot();
        if (!rig || rig->revision==calibration_revision_) return;
        if (rig->cameras.size() != config_.multiview_calibration.size() ||
            rig->cameras.size() < 2 || rig->cameras.size() > maximum_view_capacity)
            throw std::runtime_error("runtime calibration count must match the configured 2..10 multiview cameras");
        for (auto& target : config_.multiview_calibration) {
            const auto source = std::ranges::find(rig->cameras, target.camera_id,
                                                  &RigCameraCalibration::camera_id);
            if (source == rig->cameras.end())
                throw std::runtime_error("runtime calibration is missing a configured camera ID");
            target.intrinsics = source->intrinsics;
            target.distortion = source->distortion;
            target.R_w2c = source->R_w2c;
            target.t_w2c = source->t_w2c;
            // RigCalibrationTool observes frames after CaptureStage has applied
            // its configured rotation.  The stored calibration is therefore
            // already expressed in the inference image coordinate system.
            // Applying capture rotation here again would rotate K and [R|t]
            // twice and make triangulation disagree with the viewer snapshot.
            target.source_extent = {};
            target.image_rotation_degrees = 0;
            target.calibrated = true;
        }
        calibration_revision_=rig->revision;
#ifdef IRIS_HAS_TENSORRT
        if(started_) engine_=std::make_unique<TensorRtMultiviewEngine>(config_);
#endif
    }
    PoseConfig config_;
    std::unique_ptr<TensorRtMultiviewEngine> engine_;
    bool started_{};
    std::uint64_t calibration_revision_{};
    bool metrics_enabled_{};
    infrastructure::metrics::Histogram process_ms_, capture_to_result_ms_;
    infrastructure::metrics::Gauge last_capture_to_result_ms_;
    infrastructure::metrics::Gauge last_process_ms_;
    std::array<infrastructure::metrics::Histogram,18> breakdown_;
    std::array<infrastructure::metrics::Gauge,18> latest_;
};

MultiviewPoseStage::MultiviewPoseStage(Channel<Packet>& input, Channel<Packet>* output, PoseConfig config, infrastructure::metrics::MetricRegistry* metrics)
    : Stage(input, output), impl_(std::make_unique<Impl>(std::move(config), metrics)) {}
MultiviewPoseStage::~MultiviewPoseStage() { stop(); }
void MultiviewPoseStage::start() { impl_->start(); Stage::start(); }
void MultiviewPoseStage::stop() { Stage::stop(); impl_->stop(); }
void MultiviewPoseStage::process(Packet& packet) { impl_->process(packet); }

} // namespace iris
