#pragma once
#include "iris/pipeline/Frame.hpp"
#include <array>
#include <optional>
#include <vector>
#include <vector>
namespace iris {
using FrameBatch = std::vector<Frame>;

inline constexpr std::size_t panoptic_joint_count = 19;
inline constexpr std::size_t coco_joint_count = 17;

struct MultiviewPose {
    bool active{};
    std::vector<CameraId> view_camera_ids;
    // Candidate index selected by the cross-view association stage, or -1 if unmatched.
    std::vector<std::int32_t> selected_detection_indices;
    std::array<std::array<float, 3>, coco_joint_count> joints_3d{};
    std::array<bool, coco_joint_count> joint_valid{};
    std::vector<std::array<float, coco_joint_count>> joint_scores;
    std::vector<std::array<std::array<float, 2>, coco_joint_count>> points_2d_px;
    std::vector<std::array<bool, coco_joint_count>> point_valid;
};

struct ViewPose2d {
    CameraId camera_id{};
    std::size_t person_id{};
    std::array<std::array<float, 2>, coco_joint_count> points_px{};
    std::array<float, coco_joint_count> scores{};
    std::array<bool, coco_joint_count> valid{};
};

// The PEAR EHM TorchScript model regresses SMPL-X and FLAME parameters, rather than joint XYZ
// positions. Rotation matrices are row-major. The model does not return a detection confidence.
struct HmrParameters {
    std::array<float, 16> camera_rt{};
    std::array<float, 9> global_pose{};
    std::array<float, 21 * 9> body_pose{};
    std::array<float, 15 * 9> left_hand_pose{};
    std::array<float, 15 * 9> right_hand_pose{};
    std::array<float, 3> hand_scale{};
    std::array<float, 3> head_scale{};
    std::array<float, 200> body_shape{};
    std::array<float, 50> body_expression{};
    std::array<float, 6> flame_eye_pose{};
    std::array<float, 3> flame_pose{};
    std::array<float, 3> flame_jaw_pose{};
    std::array<float, 2> flame_eyelid{};
    std::array<float, 50> flame_expression{};
    std::array<float, 300> flame_shape{};
};

struct Pose {
    std::uint64_t source_sequence{};
    CameraId source_camera{};
    std::array<std::array<float, 3>, panoptic_joint_count> joints_3d_mm{};
    std::array<std::array<std::array<float, 2>, panoptic_joint_count>, 5> points_2d_px{};
    std::array<std::array<float, panoptic_joint_count>, 5> per_view_joint_confidence{};
    std::array<float, panoptic_joint_count> joint_confidence{};
    float score{};
    std::optional<HmrParameters> hmr;
};
using PoseBatch = std::vector<Pose>;
struct Packet {
    std::uint64_t sequence{};
    FrameBatch frames;
    std::optional<PoseBatch> poses;
    // Results from the fixed three-view COCO-17 TensorRT engine. Coordinates
    // use the calibration world frame and units (e.g. centimetres for Panoptic).
    std::optional<std::vector<MultiviewPose>> multiview_poses;
    // Raw per-camera RTMO detections, before epipolar assignment or triangulation.
    std::optional<std::vector<ViewPose2d>> view_poses_2d;
};
} // namespace iris
