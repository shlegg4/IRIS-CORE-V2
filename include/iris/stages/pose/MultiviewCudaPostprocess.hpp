#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace iris {

struct MultiviewAssociationWorkspace;
struct MultiviewTemporalWorkspace;

cudaError_t create_multiview_temporal_workspace(
    std::size_t view_count, std::size_t max_tracks,
    MultiviewTemporalWorkspace** workspace);
void destroy_multiview_temporal_workspace(MultiviewTemporalWorkspace* workspace);
// Matches detections to persistent 3D tracks, seeds unmatched tracks from the
// cross-view fallback assignments, triangulates updates, and predicts through
// short occlusions. Projections are a small host array; other inputs are device pointers.
cudaError_t launch_multiview_temporal_update(
    MultiviewTemporalWorkspace* workspace,
    const float* keypoints, const float* scores,
    const unsigned char* candidate_valid,
    const unsigned char* seed_assignments,
    const float* projections_host, float dt_seconds,
    float gate_px, float minimum_score, float maximum_reprojection_error_px,
    cudaEvent_t assignment_begin, cudaEvent_t assignment_end, cudaStream_t stream);
cudaError_t launch_multiview_unwarp_keypoints(
    const float* model_keypoints, const float* source_intrinsics,
    const float* distortion, const float* mapping_meta_device,
    std::size_t view_count, float* image_keypoints,
    unsigned char* joint_valid, cudaStream_t stream);
cudaError_t copy_multiview_temporal_result(
    MultiviewTemporalWorkspace* workspace,
    unsigned char* host_assignments, std::uint64_t* host_track_ids,
    float* host_xyz, unsigned char* host_valid, unsigned char* host_predicted,
    std::uint32_t* host_track_count,
    cudaStream_t stream);

cudaError_t create_multiview_association_workspace(
    std::size_t view_count, std::size_t max_persons,
    const std::uint32_t* camera_ids,
    MultiviewAssociationWorkspace** workspace, cudaStream_t stream);
void destroy_multiview_association_workspace(MultiviewAssociationWorkspace* workspace);

// Runs the current confidence-weighted, trimmed epipolar edge matching on the
// GPU. fundamentals is a row-major [view][view][3][3] host array with the
// upper triangle populated. Assignments are [max_persons][view_count]; 255
// represents a missing detection.
cudaError_t launch_multiview_current_association(
    MultiviewAssociationWorkspace* workspace,
    const float* keypoints, const float* scores,
    const unsigned char* candidate_valid,
    const float* fundamentals,
    float gate_px, float minimum_score,
    cudaStream_t stream, double* host_ms);
cudaError_t clear_multiview_association_assignments(
    MultiviewAssociationWorkspace* workspace, cudaStream_t stream);
const unsigned char* multiview_association_assignments_device(
    const MultiviewAssociationWorkspace* workspace);

// Triangulates the fixed RTMO layout [view][candidate][joint]. All pointers are
// device pointers and the operation is enqueued on stream. Invalid or
// non-finite observations produce valid=0 and xyz=0.
cudaError_t launch_multiview_weighted_dlt(const float* keypoints,
                                          const float* scores,
                                          const unsigned char* candidate_valid,
                                          const unsigned char* assignments,
                                          const float* projections,
                                          float minimum_score,
                                          float maximum_reprojection_error,
                                          float* xyz,
                                          unsigned char* valid,
                                          cudaStream_t stream);

// fundamentals contains row-major pair matrices in order (0,1), (0,2), (1,2).
// The matcher writes up to ten rows of per-view candidate indices; 255 marks
// a missing view. Unmatched view 1/2 candidates may form pair-only rows.
cudaError_t launch_multiview_epipolar_assignment(const float* keypoints,
                                                 const float* scores,
                                                 const unsigned char* candidate_valid,
                                                 const float* fundamentals,
                                                 float gate_px,
                                                 float minimum_score,
                                                 unsigned char* assignments,
                                                 cudaStream_t stream);

cudaError_t launch_multiview_gather_selected(const float* keypoints,
                                             const float* scores,
                                             const unsigned char* candidate_valid,
                                             const unsigned char* assignments,
                                             float* selected_keypoints,
                                             float* selected_scores,
                                             unsigned char* selected_valid,
                                             cudaStream_t stream);

} // namespace iris
