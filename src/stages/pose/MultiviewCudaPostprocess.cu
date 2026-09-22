#include "iris/stages/pose/MultiviewCudaPostprocess.hpp"

#include <cmath>

namespace iris {
namespace {

__global__ void triangulate_kernel(const float* points, const float* scores,
                                   const unsigned char* candidates, const unsigned char* assignments,
                                   const float* projection,
                                   float minimum_score, float maximum_reprojection_error,
                                   float* xyz, unsigned char* valid) {
    const int work = blockIdx.x * blockDim.x + threadIdx.x;
    if (work >= 10 * 17) return;
    const int person = work / 17, joint = work % 17;
    float normal[4][4]{};
    int rows = 0;
    for (int view = 0; view < 3; ++view) {
        const int candidate = assignments[person * 3 + view];
        if (candidate >= 10) continue;
        const int score_index = (view * 10 + candidate) * 17 + joint;
        const float score = scores[score_index];
        if (!candidates[view * 10 + candidate] || !isfinite(score) || score < minimum_score) continue;
        const float x = points[score_index * 2], y = points[score_index * 2 + 1];
        if (!isfinite(x) || !isfinite(y)) continue;
        const float* p = projection + view * 12;
        const float r0[4] = {x * p[8] - p[0], x * p[9] - p[1], x * p[10] - p[2], x * p[11] - p[3]};
        const float r1[4] = {y * p[8] - p[4], y * p[9] - p[5], y * p[10] - p[6], y * p[11] - p[7]};
        for (int row = 0; row < 4; ++row) for (int column = 0; column < 4; ++column)
            normal[row][column] += score * score * (r0[row] * r0[column] + r1[row] * r1[column]);
        rows += 2;
    }
    const int output = work * 3;
    valid[work] = 0;
    xyz[output] = xyz[output + 1] = xyz[output + 2] = 0.0F;
    if (rows < 4) return;
    float matrix[3][4]{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) matrix[row][column] = normal[row][column];
        matrix[row][3] = -normal[row][3];
    }
    for (int pivot = 0; pivot < 3; ++pivot) {
        int best = pivot;
        for (int row = pivot + 1; row < 3; ++row) if (fabsf(matrix[row][pivot]) > fabsf(matrix[best][pivot])) best = row;
        if (fabsf(matrix[best][pivot]) < 1e-7F) return;
        if (best != pivot) for (int column = pivot; column < 4; ++column) {
            const float value = matrix[pivot][column]; matrix[pivot][column] = matrix[best][column]; matrix[best][column] = value;
        }
        for (int row = pivot + 1; row < 3; ++row) {
            const float factor = matrix[row][pivot] / matrix[pivot][pivot];
            for (int column = pivot; column < 4; ++column) matrix[row][column] -= factor * matrix[pivot][column];
        }
    }
    const float z = matrix[2][3] / matrix[2][2];
    const float y = (matrix[1][3] - matrix[1][2] * z) / matrix[1][1];
    const float x = (matrix[0][3] - matrix[0][1] * y - matrix[0][2] * z) / matrix[0][0];
    if (!isfinite(x) || !isfinite(y) || !isfinite(z)) return;
    for (int view = 0; view < 3; ++view) {
        const int candidate = assignments[person * 3 + view];
        if (candidate >= 10) continue;
        const int score_index = (view * 10 + candidate) * 17 + joint;
        if (!candidates[view * 10 + candidate] || !isfinite(scores[score_index]) || scores[score_index] < minimum_score) continue;
        const float* p = projection + view * 12;
        const float denominator = p[8] * x + p[9] * y + p[10] * z + p[11];
        if (!(denominator > 1e-5F)) return;
        const float projected_x = (p[0] * x + p[1] * y + p[2] * z + p[3]) / denominator;
        const float projected_y = (p[4] * x + p[5] * y + p[6] * z + p[7]) / denominator;
        const float dx = projected_x - points[score_index * 2];
        const float dy = projected_y - points[score_index * 2 + 1];
        if (!isfinite(projected_x) || !isfinite(projected_y) || hypotf(dx, dy) > maximum_reprojection_error) return;
    }
    xyz[output] = x; xyz[output + 1] = y; xyz[output + 2] = z; valid[work] = 1;
}

__device__ float epipolar_distance(const float* f, float x1, float y1, float x2, float y2) {
    const float l2x = f[0] * x1 + f[1] * y1 + f[2];
    const float l2y = f[3] * x1 + f[4] * y1 + f[5];
    const float l2z = f[6] * x1 + f[7] * y1 + f[8];
    const float l1x = f[0] * x2 + f[3] * y2 + f[6];
    const float l1y = f[1] * x2 + f[4] * y2 + f[7];
    const float residual = x2 * l2x + y2 * l2y + l2z;
    return 0.5F * (fabsf(residual) / fmaxf(1e-6F, hypotf(l2x, l2y)) + fabsf(residual) / fmaxf(1e-6F, hypotf(l1x, l1y)));
}

__global__ void assignment_kernel(const float* points, const float* scores,
                                  const unsigned char* candidates, const float* fundamentals,
                                  float gate, unsigned char* assignments) {
    // One bounded exact assignment solve per non-reference view.  The 10x10
    // cost matrix is small enough for a 2^10 dynamic program in one CUDA
    // thread, and guarantees that two people cannot consume the same
    // candidate in a view.
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int anchors[10];
    for (int person = 0; person < 10; ++person) anchors[person] = -1;
    bool used_anchor[10]{};
    for (int person = 0; person < 10; ++person) {
        int best = -1;
        for (int candidate = 0; candidate < 10; ++candidate)
            if (!used_anchor[candidate] && candidates[candidate] && isfinite(scores[candidate * 17]) &&
                (best < 0 || scores[candidate * 17] > scores[best * 17])) best = candidate;
        anchors[person] = best;
        if (best >= 0) used_anchor[best] = true;
        assignments[person * 3] = best < 0 ? 255 : static_cast<unsigned char>(best);
        assignments[person * 3 + 1] = assignments[person * 3 + 2] = 255;
    }
    for (int view = 1; view < 3; ++view) {
        constexpr int state_count = 1 << 10;
        float dp[state_count], next[state_count];
        unsigned char choice[10][state_count];
        for (int mask = 0; mask < state_count; ++mask) dp[mask] = mask == 0 ? 0.0F : 1e20F;
        for (int person = 0; person < 10; ++person) {
            for (int mask = 0; mask < state_count; ++mask) { next[mask] = dp[mask]; choice[person][mask] = 255; }
            const int anchor = anchors[person];
            for (int mask = 0; mask < state_count; ++mask) {
                if (dp[mask] >= 1e19F || anchor < 0) continue;
                const float x1 = points[anchor * 17 * 2], y1 = points[anchor * 17 * 2 + 1];
                if (!isfinite(x1) || !isfinite(y1)) continue;
                for (int candidate = 0; candidate < 10; ++candidate) {
                    if (mask & (1 << candidate)) continue;
                    const int index = view * 10 + candidate;
                    if (!candidates[index] || !isfinite(scores[index * 17])) continue;
                    const float x2 = points[(index * 17) * 2], y2 = points[(index * 17) * 2 + 1];
                    if (!isfinite(x2) || !isfinite(y2)) continue;
                    const float cost = epipolar_distance(fundamentals + (view - 1) * 9, x1, y1, x2, y2);
                    if (cost > gate) continue;
                    const int new_mask = mask | (1 << candidate);
                    if (dp[mask] + cost < next[new_mask]) {
                        // Reward every gated match so the solver prefers the
                        // largest consistent matching before minimizing error.
                        next[new_mask] = dp[mask] + cost - gate;
                        choice[person][new_mask] = static_cast<unsigned char>(candidate);
                    }
                }
            }
            for (int mask = 0; mask < state_count; ++mask) dp[mask] = next[mask];
        }
        int mask = 0;
        for (int candidate_mask = 1; candidate_mask < state_count; ++candidate_mask)
            if (dp[candidate_mask] < dp[mask]) mask = candidate_mask;
        for (int person = 9; person >= 0; --person) {
            const unsigned char candidate = choice[person][mask];
            if (candidate != 255) {
                assignments[person * 3 + view] = candidate;
                mask &= ~(1 << candidate);
            }
        }
    }
}

__global__ void gather_selected_kernel(const float* points, const float* scores,
                                       const unsigned char* candidates, const unsigned char* assignments,
                                       float* selected_points, float* selected_scores,
                                       unsigned char* selected_valid) {
    const int work = blockIdx.x * blockDim.x + threadIdx.x;
    if (work >= 10 * 3 * 17) return;
    const int person = work / (3 * 17), view = (work / 17) % 3, joint = work % 17;
    const int candidate = assignments[person * 3 + view];
    const int output_point = work * 2;
    selected_valid[work] = 0;
    selected_scores[work] = 0.0F;
    selected_points[output_point] = selected_points[output_point + 1] = 0.0F;
    if (candidate >= 10) return;
    const int input = (view * 10 + candidate) * 17 + joint;
    if (!candidates[view * 10 + candidate] || !isfinite(scores[input])) return;
    selected_scores[work] = scores[input];
    selected_points[output_point] = points[input * 2];
    selected_points[output_point + 1] = points[input * 2 + 1];
    selected_valid[work] = 1;
}

} // namespace

cudaError_t launch_multiview_weighted_dlt(const float* keypoints, const float* scores,
                                          const unsigned char* candidate_valid,
                                          const unsigned char* assignments,
                                          const float* projections, float minimum_score,
                                          float maximum_reprojection_error,
                                          float* xyz, unsigned char* valid, cudaStream_t stream) {
    triangulate_kernel<<<1, 256, 0, stream>>>(keypoints, scores, candidate_valid, assignments, projections,
                                               minimum_score, maximum_reprojection_error, xyz, valid);
    return cudaGetLastError();
}

cudaError_t launch_multiview_epipolar_assignment(const float* keypoints, const float* scores,
                                                 const unsigned char* candidate_valid,
                                                 const float* fundamentals, float gate_px,
                                                 unsigned char* assignments, cudaStream_t stream) {
    assignment_kernel<<<1, 10, 0, stream>>>(keypoints, scores, candidate_valid, fundamentals, gate_px, assignments);
    return cudaGetLastError();
}

cudaError_t launch_multiview_gather_selected(const float* keypoints, const float* scores,
                                             const unsigned char* candidate_valid,
                                             const unsigned char* assignments,
                                             float* selected_keypoints, float* selected_scores,
                                             unsigned char* selected_valid, cudaStream_t stream) {
    gather_selected_kernel<<<1, 512, 0, stream>>>(keypoints, scores, candidate_valid, assignments,
                                                   selected_keypoints, selected_scores, selected_valid);
    return cudaGetLastError();
}

} // namespace iris
