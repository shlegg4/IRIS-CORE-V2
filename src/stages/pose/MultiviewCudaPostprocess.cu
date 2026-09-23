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

__device__ void solve_assignment_parallel(const float* costs, const int* sources,
                                          int count, float gate, int row_offset,
                                          int output_view, unsigned char* assignments,
                                          float* dp_a, float* dp_b,
                                          unsigned char* choices) {
    constexpr int state_count = 1 << 10;
    const int lane = threadIdx.x;
    for (int mask = lane; mask < state_count; mask += blockDim.x)
        dp_a[mask] = mask == 0 ? 0.0F : 1e20F;
    __syncthreads();

    float* current = dp_a;
    float* next = dp_b;
    for (int person = 0; person < count; ++person) {
        const int source = sources[person];
        for (int mask = lane; mask < state_count; mask += blockDim.x) {
            float best = current[mask]; // Leave this source detection unmatched.
            unsigned char selected = 255;
            if (source >= 0) {
                for (int candidate = 0; candidate < 10; ++candidate) {
                    const int bit = 1 << candidate;
                    if (!(mask & bit)) continue;
                    const float edge = costs[person * 10 + candidate];
                    if (edge >= 1e19F) continue;
                    const float candidate_cost = current[mask ^ bit] + edge - gate;
                    if (candidate_cost < best) {
                        best = candidate_cost;
                        selected = static_cast<unsigned char>(candidate);
                    }
                }
            }
            next[mask] = best;
            choices[person * state_count + mask] = selected;
        }
        __syncthreads();
        float* swap = current;
        current = next;
        next = swap;
    }

    if (lane == 0) {
        int mask = 0;
        for (int candidate_mask = 1; candidate_mask < state_count; ++candidate_mask)
            if (current[candidate_mask] < current[mask]) mask = candidate_mask;
        for (int person = count - 1; person >= 0; --person) {
            const unsigned char candidate = choices[person * state_count + mask];
            if (candidate < 10) {
                assignments[(row_offset + person) * 3 + output_view] = candidate;
                mask ^= 1 << candidate;
            }
        }
    }
    __syncthreads();
}

__global__ void assignment_kernel(const float* points, const float* scores,
                                  const unsigned char* candidates, const float* fundamentals,
                                  float gate, float minimum_score, unsigned char* assignments) {
    // The 3 pairwise 10x10 cost matrices are evaluated in parallel.  The
    // bounded DP then assigns one thread to each candidate mask per person.
    // Camera 0 detections seed tracks; leftover camera 1 detections are matched
    // to camera 2 and emitted as rows without a camera 0 observation.
    if (blockIdx.x != 0) return;
    constexpr int state_count = 1 << 10;
    constexpr float invalid_cost = 1e20F;
    __shared__ float pair_costs[3 * 10 * 10];
    __shared__ float match_costs[10 * 10];
    __shared__ float dp_a[state_count];
    __shared__ float dp_b[state_count];
    __shared__ unsigned char choices[10 * state_count];
    __shared__ int anchors[10];
    __shared__ int source_candidates[10];
    __shared__ int anchor_count;
    __shared__ int fallback_count;

    const int lane = threadIdx.x;
    for (int work = lane; work < 3 * 10 * 10; work += blockDim.x) {
        const int pair = work / 100;
        const int pair_item = work % 100;
        const int first_candidate = pair_item / 10;
        const int second_candidate = pair_item % 10;
        const int first_view = pair == 0 ? 0 : (pair == 1 ? 0 : 1);
        const int second_view = pair == 0 ? 1 : 2;
        const int first_index = first_view * 10 + first_candidate;
        const int second_index = second_view * 10 + second_candidate;
        const int first_score_index = first_index * 17;
        const int second_score_index = second_index * 17;
        float cost = invalid_cost;
        if (candidates[first_index] && candidates[second_index] &&
            isfinite(scores[first_score_index]) && scores[first_score_index] >= minimum_score &&
            isfinite(scores[second_score_index]) && scores[second_score_index] >= minimum_score) {
            const float x1 = points[first_score_index * 2];
            const float y1 = points[first_score_index * 2 + 1];
            const float x2 = points[second_score_index * 2];
            const float y2 = points[second_score_index * 2 + 1];
            if (isfinite(x1) && isfinite(y1) && isfinite(x2) && isfinite(y2)) {
                const float measured = epipolar_distance(fundamentals + pair * 9, x1, y1, x2, y2);
                if (measured <= gate) cost = measured;
            }
        }
        pair_costs[work] = cost;
    }
    if (lane == 0) {
        for (int index = 0; index < 30; ++index) assignments[index] = 255;
        for (int person = 0; person < 10; ++person) anchors[person] = -1;
        bool used[10]{};
        anchor_count = 0;
        for (int person = 0; person < 10; ++person) {
            int best = -1;
            for (int candidate = 0; candidate < 10; ++candidate) {
                const int score_index = candidate * 17;
                if (!used[candidate] && candidates[candidate] && isfinite(scores[score_index]) &&
                    scores[score_index] >= minimum_score &&
                    (best < 0 || scores[score_index] > scores[best * 17])) best = candidate;
            }
            if (best < 0) break;
            anchors[person] = best;
            assignments[person * 3] = static_cast<unsigned char>(best);
            source_candidates[person] = best;
            used[best] = true;
            ++anchor_count;
        }
    }
    __syncthreads();

    for (int work = lane; work < anchor_count * 10; work += blockDim.x) {
        const int person = work / 10;
        const int candidate = work % 10;
        match_costs[work] = pair_costs[anchors[person] * 10 + candidate];
    }
    __syncthreads();
    solve_assignment_parallel(match_costs, source_candidates, anchor_count, gate, 0, 1,
                              assignments, dp_a, dp_b, choices);

    // A view 2 match must agree with the anchor and, when present, the chosen
    // view 1 detection. This avoids building geometrically inconsistent triples.
    for (int work = lane; work < anchor_count * 10; work += blockDim.x) {
        const int person = work / 10;
        const int candidate = work % 10;
        const int anchor = anchors[person];
        const int view1_candidate = assignments[person * 3 + 1];
        const float cost02 = pair_costs[100 + anchor * 10 + candidate];
        float cost = cost02;
        if (view1_candidate < 10) {
            const float cost12 = pair_costs[200 + view1_candidate * 10 + candidate];
            cost = cost02 < invalid_cost && cost12 < invalid_cost
                ? 0.5F * (cost02 + cost12) : invalid_cost;
        }
        match_costs[work] = cost;
    }
    __syncthreads();
    solve_assignment_parallel(match_costs, source_candidates, anchor_count, gate, 0, 2,
                              assignments, dp_a, dp_b, choices);

    if (lane == 0) {
        bool used_view1[10]{};
        for (int person = 0; person < anchor_count; ++person) {
            const int candidate = assignments[person * 3 + 1];
            if (candidate < 10) used_view1[candidate] = true;
        }
        fallback_count = 0;
        const int capacity = 10 - anchor_count;
        bool selected[10]{};
        while (fallback_count < capacity) {
            int best = -1;
            for (int candidate = 0; candidate < 10; ++candidate) {
                const int score_index = (10 + candidate) * 17;
                if (!used_view1[candidate] && !selected[candidate] && candidates[10 + candidate] &&
                    isfinite(scores[score_index]) && scores[score_index] >= minimum_score &&
                    (best < 0 || scores[score_index] > scores[(10 + best) * 17])) best = candidate;
            }
            if (best < 0) break;
            source_candidates[fallback_count] = best;
            const int row = anchor_count + fallback_count;
            assignments[row * 3 + 1] = static_cast<unsigned char>(best);
            selected[best] = true;
            ++fallback_count;
        }
    }
    __syncthreads();

    bool used_view2[10]{};
    for (int person = 0; person < anchor_count; ++person) {
        const int candidate = assignments[person * 3 + 2];
        if (candidate < 10) used_view2[candidate] = true;
    }
    for (int work = lane; work < fallback_count * 10; work += blockDim.x) {
        const int person = work / 10;
        const int candidate = work % 10;
        const int view2_candidate = candidate;
        match_costs[work] = used_view2[view2_candidate]
            ? invalid_cost
            : pair_costs[200 + source_candidates[person] * 10 + view2_candidate];
    }
    __syncthreads();
    solve_assignment_parallel(match_costs, source_candidates, fallback_count, gate,
                              anchor_count, 2, assignments, dp_a, dp_b, choices);

    for (int person = anchor_count; person < anchor_count + fallback_count; ++person) {
        const int candidate = assignments[person * 3 + 2];
        if (candidate < 10) used_view2[candidate] = true;
    }
    if (lane == 0) {
        for (int row = anchor_count + fallback_count; row < 10; ++row) {
            int best = -1;
            for (int candidate = 0; candidate < 10; ++candidate) {
                const int score_index = (20 + candidate) * 17;
                if (!used_view2[candidate] && candidates[20 + candidate] &&
                    isfinite(scores[score_index]) && scores[score_index] >= minimum_score &&
                    (best < 0 || scores[score_index] > scores[(20 + best) * 17])) best = candidate;
            }
            if (best < 0) break;
            assignments[row * 3 + 2] = static_cast<unsigned char>(best);
            used_view2[best] = true;
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
                                                 const float* fundamentals, float gate_px, float minimum_score,
                                                 unsigned char* assignments, cudaStream_t stream) {
    assignment_kernel<<<1, 256, 0, stream>>>(keypoints, scores, candidate_valid, fundamentals, gate_px, minimum_score, assignments);
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
