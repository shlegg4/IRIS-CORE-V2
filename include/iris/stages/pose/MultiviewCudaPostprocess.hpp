#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace iris {

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
