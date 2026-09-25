#include "iris/stages/pose/MultiviewCudaPostprocess.hpp"

#include <cub/cub.cuh>
#include <math_constants.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace iris {
namespace {
constexpr int association_candidate_capacity = 10;
constexpr int association_joint_capacity = 17;
constexpr int association_max_views = 10;
constexpr int association_max_tracks = 50;
constexpr int association_max_nodes = association_max_views * association_candidate_capacity;

__device__ float weighted_trimmed_epipolar_cost(
    const float* keypoints, const float* scores, const unsigned char* candidates,
    const float* fundamentals, int view_count, int first_view, int first_candidate,
    int second_view, int second_candidate, float minimum_score) {
    if (!candidates[first_view * association_candidate_capacity + first_candidate] ||
        !candidates[second_view * association_candidate_capacity + second_candidate])
        return CUDART_INF_F;

    float f[9];
    const int low_view = min(first_view, second_view);
    const int high_view = max(first_view, second_view);
    const float* stored = fundamentals + (low_view * view_count + high_view) * 9;
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            f[row * 3 + column] = first_view < second_view
                ? stored[row * 3 + column] : stored[column * 3 + row];

    float residuals[association_joint_capacity];
    float weights[association_joint_capacity];
    int count = 0;
    for (int joint = 0; joint < association_joint_capacity; ++joint) {
        const int first_index = (first_view * association_candidate_capacity + first_candidate) * association_joint_capacity + joint;
        const int second_index = (second_view * association_candidate_capacity + second_candidate) * association_joint_capacity + joint;
        const float first_score = scores[first_index];
        const float second_score = scores[second_index];
        const float x1 = keypoints[first_index * 2];
        const float y1 = keypoints[first_index * 2 + 1];
        const float x2 = keypoints[second_index * 2];
        const float y2 = keypoints[second_index * 2 + 1];
        if (!isfinite(first_score) || !isfinite(second_score) ||
            first_score < minimum_score || second_score < minimum_score ||
            !isfinite(x1) || !isfinite(y1) || !isfinite(x2) || !isfinite(y2)) continue;

        const float line2_x = f[0] * x1 + f[1] * y1 + f[2];
        const float line2_y = f[3] * x1 + f[4] * y1 + f[5];
        const float line2_z = f[6] * x1 + f[7] * y1 + f[8];
        const float line1_x = f[0] * x2 + f[3] * y2 + f[6];
        const float line1_y = f[1] * x2 + f[4] * y2 + f[7];
        const float residual = fabsf(x2 * line2_x + y2 * line2_y + line2_z);
        const float distance = 0.5F * residual *
            (1.0F / fmaxf(1e-6F, hypotf(line2_x, line2_y)) +
             1.0F / fmaxf(1e-6F, hypotf(line1_x, line1_y)));
        if (!isfinite(distance)) continue;
        const float confidence = sqrtf(fminf(1.0F, fmaxf(0.0F, first_score))) *
                                 sqrtf(fminf(1.0F, fmaxf(0.0F, second_score)));
        if (!(confidence > 0.0F)) continue;
        residuals[count] = distance;
        weights[count] = confidence;
        ++count;
    }
    if (count < 5) return CUDART_INF_F;

    // The COCO-17 array is small; insertion sort avoids general-purpose sort
    // machinery while preserving the CPU matcher's residual ordering.
    for (int index = 1; index < count; ++index) {
        const float residual = residuals[index];
        const float weight = weights[index];
        int position = index;
        while (position > 0 && residuals[position - 1] > residual) {
            residuals[position] = residuals[position - 1];
            weights[position] = weights[position - 1];
            --position;
        }
        residuals[position] = residual;
        weights[position] = weight;
    }
    float total_weight = 0.0F;
    for (int index = 0; index < count; ++index) total_weight += weights[index];
    const float retained_weight = total_weight * 0.8F;
    float accumulated_weight = 0.0F;
    float weighted_residual = 0.0F;
    for (int index = 0; index < count && accumulated_weight < retained_weight; ++index) {
        const float weight = fminf(weights[index], retained_weight - accumulated_weight);
        weighted_residual += residuals[index] * weight;
        accumulated_weight += weight;
    }
    return accumulated_weight > 0.0F ? weighted_residual / accumulated_weight : CUDART_INF_F;
}

__global__ void association_pair_cost_kernel(
    const float* keypoints, const float* scores, const unsigned char* candidates,
    const float* fundamentals, int view_count, int pair_count,
    const unsigned char* pair_views, float gate, float minimum_score,
    float* pair_costs, unsigned long long* edge_keys) {
    const int edge_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int edge_count = pair_count * association_candidate_capacity * association_candidate_capacity;
    if (edge_index >= edge_count) return;
    const int pair_index = edge_index / (association_candidate_capacity * association_candidate_capacity);
    const int local_index = edge_index % (association_candidate_capacity * association_candidate_capacity);
    const int first_candidate = local_index / association_candidate_capacity;
    const int second_candidate = local_index % association_candidate_capacity;
    const int first_view = pair_views[pair_index * 2];
    const int second_view = pair_views[pair_index * 2 + 1];
    const float cost = weighted_trimmed_epipolar_cost(
        keypoints, scores, candidates, fundamentals, view_count,
        first_view, first_candidate, second_view, second_candidate, minimum_score);
    const int node_count = view_count * association_candidate_capacity;
    const int first_node = first_view * association_candidate_capacity + first_candidate;
    const int second_node = second_view * association_candidate_capacity + second_candidate;
    pair_costs[first_node * node_count + second_node] = cost;
    pair_costs[second_node * node_count + first_node] = cost;
    const unsigned long long bits = static_cast<unsigned long long>(__float_as_uint(cost));
    edge_keys[edge_index] = (bits << 32) | static_cast<unsigned int>(edge_index);
    (void)gate;
}

__device__ bool association_tracks_compatible(const int* first, const int* second,
                                               int view_count, int node_count,
                                               const float* pair_costs, float gate) {
    for (int first_view = 0; first_view < view_count; ++first_view) {
        const int first_candidate = first[first_view];
        if (first_candidate < 0) continue;
        if (second[first_view] >= 0) return false;
        const int first_node = first_view * association_candidate_capacity + first_candidate;
        for (int second_view = 0; second_view < view_count; ++second_view) {
            const int second_candidate = second[second_view];
            if (second_candidate < 0) continue;
            const int second_node = second_view * association_candidate_capacity + second_candidate;
            const float cost = pair_costs[first_node * node_count + second_node];
            if (!isfinite(cost) || cost >= gate) return false;
        }
    }
    return true;
}

__device__ bool association_node_compatible(const int* track, int view,
                                             int candidate, int view_count,
                                             int node_count, const float* pair_costs,
                                             float gate) {
    if (track[view] >= 0) return false;
    const int candidate_node = view * association_candidate_capacity + candidate;
    for (int other_view = 0; other_view < view_count; ++other_view) {
        if (track[other_view] < 0) continue;
        const int other_node = other_view * association_candidate_capacity + track[other_view];
        const float cost = pair_costs[other_node * node_count + candidate_node];
        if (!isfinite(cost) || cost >= gate) return false;
    }
    return true;
}

__global__ void association_merge_kernel(
    const unsigned long long* sorted_edge_keys, int edge_count,
    const unsigned char* pair_views, int pair_count, int view_count,
    int max_persons, float gate, const float* pair_costs,
    const int* canonical_view_order,
    unsigned char* assignments, unsigned int* output_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int node_count = view_count * association_candidate_capacity;
    int tracks[association_max_tracks][association_max_views];
    unsigned char active[association_max_tracks]{};
    int node_track[association_max_nodes];
    for (int track = 0; track < association_max_tracks; ++track)
        for (int view = 0; view < association_max_views; ++view) tracks[track][view] = -1;
    for (int node = 0; node < association_max_nodes; ++node) node_track[node] = -1;
    for (int index = 0; index < max_persons * view_count; ++index) assignments[index] = 255;
    unsigned int track_count = 0;

    for (int edge_index = 0; edge_index < edge_count; ++edge_index) {
        const unsigned long long key = sorted_edge_keys[edge_index];
        const float cost = __uint_as_float(static_cast<unsigned int>(key >> 32));
        if (!isfinite(cost) || cost >= gate) break;
        const unsigned int original_index = static_cast<unsigned int>(key);
        const int pair_index = original_index /
            (association_candidate_capacity * association_candidate_capacity);
        if (pair_index >= pair_count) continue;
        const int local_index = original_index %
            (association_candidate_capacity * association_candidate_capacity);
        const int first_candidate = local_index / association_candidate_capacity;
        const int second_candidate = local_index % association_candidate_capacity;
        const int first_view = pair_views[pair_index * 2];
        const int second_view = pair_views[pair_index * 2 + 1];
        const int first_node = first_view * association_candidate_capacity + first_candidate;
        const int second_node = second_view * association_candidate_capacity + second_candidate;
        const int first_track = node_track[first_node];
        const int second_track = node_track[second_node];

        if (first_track < 0 && second_track < 0) {
            if (track_count >= association_max_tracks) continue;
            int* track = tracks[track_count];
            track[first_view] = first_candidate;
            track[second_view] = second_candidate;
            active[track_count] = 1;
            node_track[first_node] = node_track[second_node] = static_cast<int>(track_count);
            ++track_count;
        } else if (first_track >= 0 && second_track < 0) {
            int* track = tracks[first_track];
            if (association_node_compatible(track, second_view, second_candidate,
                                            view_count, node_count, pair_costs, gate)) {
                track[second_view] = second_candidate;
                node_track[second_node] = first_track;
            }
        } else if (first_track < 0 && second_track >= 0) {
            int* track = tracks[second_track];
            if (association_node_compatible(track, first_view, first_candidate,
                                            view_count, node_count, pair_costs, gate)) {
                track[first_view] = first_candidate;
                node_track[first_node] = second_track;
            }
        } else if (first_track != second_track && active[first_track] && active[second_track]) {
            int* first = tracks[first_track];
            int* second = tracks[second_track];
            if (!association_tracks_compatible(first, second, view_count,
                                               node_count, pair_costs, gate)) continue;
            for (int view = 0; view < view_count; ++view) {
                if (second[view] < 0) continue;
                first[view] = second[view];
                const int node = view * association_candidate_capacity + second[view];
                node_track[node] = first_track;
                second[view] = -1;
            }
            active[second_track] = 0;
        }
    }

    int ranked_tracks[association_max_tracks];
    int ranked_view_counts[association_max_tracks];
    float ranked_mean_costs[association_max_tracks];
    int ranked_count = 0;
    for (int track_index = 0; track_index < static_cast<int>(track_count); ++track_index) {
        if (!active[track_index]) continue;
        int assigned_views = 0;
        int pairs = 0;
        float total_cost = 0.0F;
        const int* track = tracks[track_index];
        for (int first_view = 0; first_view < view_count; ++first_view) {
            if (track[first_view] < 0) continue;
            ++assigned_views;
            const int first_node = first_view * association_candidate_capacity + track[first_view];
            for (int second_view = first_view + 1; second_view < view_count; ++second_view) {
                if (track[second_view] < 0) continue;
                const int second_node = second_view * association_candidate_capacity + track[second_view];
                total_cost += pair_costs[first_node * node_count + second_node];
                ++pairs;
            }
        }
        if (assigned_views < 2) continue;
        const float mean_cost = pairs ? total_cost / static_cast<float>(pairs) : gate;
        int insert_at = ranked_count;
        while (insert_at > 0) {
            const int previous = ranked_tracks[insert_at - 1];
            bool before = assigned_views > ranked_view_counts[insert_at - 1];
            if (assigned_views == ranked_view_counts[insert_at - 1]) {
                before = mean_cost < ranked_mean_costs[insert_at - 1];
                if (mean_cost == ranked_mean_costs[insert_at - 1]) {
                    const int* previous_track = tracks[previous];
                    for (int order = 0; order < view_count; ++order) {
                        const int view = canonical_view_order[order];
                        const int current_value = track[view] < 0 ? association_candidate_capacity : track[view];
                        const int previous_value = previous_track[view] < 0
                            ? association_candidate_capacity : previous_track[view];
                        if (current_value == previous_value) continue;
                        before = current_value < previous_value;
                        break;
                    }
                }
            }
            if (!before) break;
            ranked_tracks[insert_at] = ranked_tracks[insert_at - 1];
            ranked_view_counts[insert_at] = ranked_view_counts[insert_at - 1];
            ranked_mean_costs[insert_at] = ranked_mean_costs[insert_at - 1];
            --insert_at;
        }
        ranked_tracks[insert_at] = track_index;
        ranked_view_counts[insert_at] = assigned_views;
        ranked_mean_costs[insert_at] = mean_cost;
        ++ranked_count;
    }

    const int selected_count = min(max_persons, ranked_count);
    for (int output = 0; output < selected_count; ++output) {
        const int* track = tracks[ranked_tracks[output]];
        for (int view = 0; view < view_count; ++view)
            assignments[output * view_count + view] = track[view] < 0
                ? 255 : static_cast<unsigned char>(track[view]);
    }
    *output_count = static_cast<unsigned int>(selected_count);
}

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
                // On exact cost ties, reserve lower candidate indices for
                // earlier rows by assigning the highest available index to
                // the row being processed now. This keeps deterministic
                // one-to-one results for symmetric detections.
                for (int candidate = 9; candidate >= 0; --candidate) {
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

constexpr int temporal_max_views = 10;
constexpr int temporal_max_tracks = 10;
constexpr int temporal_candidates = 10;
constexpr int temporal_joints = 17;
constexpr int temporal_dormant_after_missed = 2;
constexpr int temporal_dormant_lifetime_frames = 60;
constexpr float temporal_reidentification_gate_px = 32.0F;

__global__ void temporal_cost_kernel(
    const float* points, const float* scores, const unsigned char* candidates,
    const float* projection, const float* xyz, const float* velocity,
    const unsigned char* joint_valid, const unsigned char* active,
    float dt, float minimum_score, float gate, int view_count, int track_count,
    float* costs) {
    const int work = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = view_count * track_count * temporal_candidates;
    if (work >= total) return;
    const int candidate = work % temporal_candidates;
    const int track = (work / temporal_candidates) % track_count;
    const int view = work / (temporal_candidates * track_count);
    float cost = CUDART_INF_F;
    if (active[track] && candidates[view * temporal_candidates + candidate]) {
        const float* p = projection + view * 12;
        float sum = 0.0F;
        float weight_sum = 0.0F;
        int count = 0;
        for (int joint = 0; joint < temporal_joints; ++joint) {
            const int state = track * temporal_joints + joint;
            const int detection = (view * temporal_candidates + candidate) * temporal_joints + joint;
            const float confidence = scores[detection];
            if (!joint_valid[state] || !isfinite(confidence) || confidence < minimum_score) continue;
            const float px = points[detection * 2], py = points[detection * 2 + 1];
            if (!isfinite(px) || !isfinite(py)) continue;
            const float x = xyz[state * 3] + velocity[state * 3] * dt;
            const float y = xyz[state * 3 + 1] + velocity[state * 3 + 1] * dt;
            const float z = xyz[state * 3 + 2] + velocity[state * 3 + 2] * dt;
            const float w = p[8] * x + p[9] * y + p[10] * z + p[11];
            if (!(w > 1e-5F)) continue;
            const float projected_x = (p[0] * x + p[1] * y + p[2] * z + p[3]) / w;
            const float projected_y = (p[4] * x + p[5] * y + p[6] * z + p[7]) / w;
            if (projected_x < 0.0F || projected_x >= 640.0F ||
                projected_y < 0.0F || projected_y >= 640.0F) continue;
            const float dx = projected_x - px;
            const float dy = projected_y - py;
            const float residual = hypotf(dx, dy);
            if (!isfinite(residual)) continue;
            // Huber-like robustification prevents a few bad joints dominating.
            const float weight = fminf(1.0F, confidence);
            sum += fminf(residual, gate * 2.0F) * weight;
            weight_sum += weight;
            ++count;
        }
        if (count >= 3 && weight_sum > 0.0F) cost = sum / weight_sum;
    }
    costs[work] = cost;
}

__global__ void temporal_assignment_kernel(
    const float* costs, const unsigned char* active,
    unsigned char* assignments, int views, int tracks, float gate) {
    constexpr int columns = temporal_candidates * 2;
    if (blockIdx.x >= static_cast<unsigned>(views)) return;
    const int view = static_cast<int>(blockIdx.x);
    const int lane = threadIdx.x;
    __shared__ float u[temporal_max_tracks + 1];
    __shared__ float v[columns + 1];
    __shared__ float minv[columns + 1];
    __shared__ int p[columns + 1];
    __shared__ int way[columns + 1];
    __shared__ unsigned char used_columns[columns + 1];
    __shared__ int current_row, column_zero, column_one, done;
    __shared__ float delta;

    for (int i = lane; i <= tracks; i += blockDim.x) u[i] = 0.0F;
    for (int j = lane; j <= columns; j += blockDim.x) {
        v[j] = 0.0F;
        p[j] = 0;
        way[j] = 0;
    }
    __syncthreads();

    // Rectangular Hungarian assignment with one dummy column per track.
    // CUDA lanes scan and relax columns in parallel; row augmentations remain
    // ordered to preserve the exact minimum-cost one-to-one solution.
    for (int row = 1; row <= tracks; ++row) {
        if (lane == 0) {
            p[0] = row;
            column_zero = 0;
            for (int j = 0; j <= columns; ++j) {
                minv[j] = CUDART_INF_F;
                used_columns[j] = 0;
            }
        }
        __syncthreads();
        do {
            if (lane == 0) {
                used_columns[column_zero] = 1;
                current_row = p[column_zero];
            }
            __syncthreads();
            for (int column = lane + 1; column <= columns; column += blockDim.x) {
                if (used_columns[column]) continue;
                float edge = gate;
                if (column <= temporal_candidates) {
                    const int candidate = column - 1;
                    edge = CUDART_INF_F;
                    if (active[current_row - 1])
                        edge = costs[(view * tracks + current_row - 1) * temporal_candidates + candidate];
                    if (!isfinite(edge)) edge = 1e6F;
                }
                const float current = edge - u[current_row] - v[column];
                if (current < minv[column]) {
                    minv[column] = current;
                    way[column] = column_zero;
                }
            }
            __syncthreads();
            if (lane == 0) {
                delta = CUDART_INF_F;
                column_one = 0;
                for (int column = 1; column <= columns; ++column) {
                    if (!used_columns[column] && minv[column] < delta) {
                        delta = minv[column];
                        column_one = column;
                    }
                }
            }
            __syncthreads();
            for (int column = lane; column <= columns; column += blockDim.x) {
                if (used_columns[column]) {
                    u[p[column]] += delta;
                    v[column] -= delta;
                } else {
                    minv[column] -= delta;
                }
            }
            __syncthreads();
            if (lane == 0) {
                column_zero = column_one;
                done = p[column_zero] == 0;
            }
            __syncthreads();
        } while (!done);

        if (lane == 0) {
            do {
                const int previous_column = way[column_zero];
                p[column_zero] = p[previous_column];
                column_zero = previous_column;
            } while (column_zero != 0);
        }
        __syncthreads();
    }

    for (int track = lane; track < tracks; track += blockDim.x) {
        int selected = -1;
        for (int column = 1; column <= temporal_candidates; ++column)
            if (p[column] == track + 1) { selected = column - 1; break; }
        const float value = selected >= 0
            ? costs[(view * tracks + track) * temporal_candidates + selected]
            : CUDART_INF_F;
        assignments[track * views + view] = active[track] && selected >= 0 && value < gate
            ? static_cast<unsigned char>(selected) : 255;
    }
}

__device__ float temporal_reidentification_cost(
    int seed, int track, const float* points, const float* scores,
    const unsigned char* candidates, const unsigned char* seeds,
    const bool* used, const float* projections, const float* anchor_xyz,
    const float* anchor_velocity, const float* anchor_age,
    const unsigned char* anchor_valid, float minimum_score, int views) {
    float total = 0.0F;
    int shared_joints = 0;
    for (int joint = 0; joint < temporal_joints; ++joint) {
        const int state = track * temporal_joints + joint;
        if (!anchor_valid[state]) continue;
        float joint_sum = 0.0F;
        float joint_weight = 0.0F;
        for (int view = 0; view < views; ++view) {
            const int candidate = seeds[seed * views + view];
            if (candidate >= temporal_candidates || used[view * temporal_candidates + candidate] ||
                !candidates[view * temporal_candidates + candidate]) continue;
            const int detection = (view * temporal_candidates + candidate) * temporal_joints + joint;
            const float confidence = scores[detection];
            if (!isfinite(confidence) || confidence < minimum_score) continue;
            const float* p = projections + view * 12;
            const float age = fminf(anchor_age[state], 1.0F);
            const float x = anchor_xyz[state * 3] + anchor_velocity[state * 3] * age;
            const float y = anchor_xyz[state * 3 + 1] + anchor_velocity[state * 3 + 1] * age;
            const float z = anchor_xyz[state * 3 + 2] + anchor_velocity[state * 3 + 2] * age;
            const float w = p[8] * x + p[9] * y + p[10] * z + p[11];
            if (!(w > 1e-5F)) continue;
            const float projected_x = (p[0] * x + p[1] * y + p[2] * z + p[3]) / w;
            const float projected_y = (p[4] * x + p[5] * y + p[6] * z + p[7]) / w;
            const float px = points[detection * 2], py = points[detection * 2 + 1];
            if (!isfinite(projected_x) || !isfinite(projected_y) || !isfinite(px) || !isfinite(py)) continue;
            const float residual = hypotf(projected_x - px, projected_y - py);
            if (!isfinite(residual)) continue;
            const float weight = fminf(1.0F, confidence);
            joint_sum += fminf(residual, 64.0F) * weight;
            joint_weight += weight;
        }
        if (joint_weight > 0.0F) {
            total += joint_sum / joint_weight;
            ++shared_joints;
        }
    }
    return shared_joints >= 5 ? total / static_cast<float>(shared_joints) : CUDART_INF_F;
}

__global__ void temporal_lifecycle_kernel(
    const float* points, const float* scores, const unsigned char* candidates,
    const unsigned char* seeds, const float* projections,
    unsigned char* assignments, unsigned char* active, unsigned char* dormant,
    unsigned char* missed, unsigned char* dormant_age,
    unsigned char* joint_valid, unsigned char* joint_missed, float* velocity,
    float* anchor_xyz, float* anchor_velocity, float* anchor_age,
    unsigned char* anchor_valid, std::uint64_t* ids, std::uint64_t* next_id,
    float minimum_score, int views, int tracks) {
    if (blockIdx.x || threadIdx.x) return;
    bool used[temporal_max_views * temporal_candidates]{};
    for (int view = 0; view < views; ++view)
        for (int track = 0; track < tracks; ++track) {
            const int candidate = assignments[track * views + view];
            if (candidate < temporal_candidates) used[view * temporal_candidates + candidate] = true;
        }

    // Move tracks with two consecutive unsupported frames into dormancy.
    for (int track = 0; track < tracks; ++track) {
        if (active[track]) {
            int observed_views = 0;
            for (int view = 0; view < views; ++view)
                observed_views += assignments[track * views + view] != 255;
            if (observed_views >= 2) {
                missed[track] = 0;
            } else if (++missed[track] >= temporal_dormant_after_missed) {
                active[track] = 0;
                dormant[track] = 1;
                missed[track] = 0;
                dormant_age[track] = 0;
                for (int joint = 0; joint < temporal_joints; ++joint) {
                    const int state = track * temporal_joints + joint;
                    joint_valid[state] = 0;
                    joint_missed[state] = 0;
                    for (int axis = 0; axis < 3; ++axis) velocity[state * 3 + axis] = 0.0F;
                }
            }
        } else if (dormant[track]) {
            if (++dormant_age[track] > temporal_dormant_lifetime_frames) {
                dormant[track] = 0;
                ids[track] = 0;
                dormant_age[track] = 0;
                for (int joint = 0; joint < temporal_joints; ++joint) {
                    const int state = track * temporal_joints + joint;
                    anchor_valid[state] = 0;
                    anchor_age[state] = 0.0F;
                    for (int axis = 0; axis < 3; ++axis) {
                        anchor_xyz[state * 3 + axis] = 0.0F;
                        anchor_velocity[state * 3 + axis] = 0.0F;
                    }
                }
            }
        }
    }

    int available_views[temporal_max_tracks]{};
    bool has_recovery_seed = false;
    float reid_costs[temporal_max_tracks][temporal_max_tracks];
    for (int seed = 0; seed < tracks; ++seed) {
        for (int view = 0; view < views; ++view) {
            const int candidate = seeds[seed * views + view];
            if (candidate < temporal_candidates && !used[view * temporal_candidates + candidate] &&
                candidates[view * temporal_candidates + candidate]) ++available_views[seed];
        }
        has_recovery_seed |= available_views[seed] >= 2;
        for (int track = 0; track < tracks; ++track) {
            reid_costs[seed][track] = CUDART_INF_F;
            if (available_views[seed] >= 2 && dormant[track])
                reid_costs[seed][track] = temporal_reidentification_cost(seed, track,
                    points, scores, candidates, seeds, used, projections,
                    anchor_xyz, anchor_velocity, anchor_age, anchor_valid,
                    minimum_score, views);
        }
    }

    // The seed buffer is cleared on ordinary frames. Avoid running the
    // dormant-track assignment when there is nothing to recover.
    if (!has_recovery_seed) {
        for (int track = 0; track < tracks; ++track) {
            if (!active[track] || ids[track] != 0) continue;
            bool assigned = false;
            for (int view = 0; view < views; ++view)
                assigned |= assignments[track * views + view] != 255;
            if (assigned) ids[track] = (*next_id)++;
        }
        return;
    }

    // One-to-one Hungarian matching of recovery seeds to dormant identities.
    constexpr int columns = temporal_max_tracks * 2;
    float u[temporal_max_tracks + 1]{}, v[columns + 1]{}, minv[columns + 1]{};
    int p[columns + 1]{}, way[columns + 1]{};
    unsigned char used_columns[columns + 1]{};
    for (int row = 1; row <= tracks; ++row) {
        p[0] = row;
        int column_zero = 0;
        for (int column = 0; column <= columns; ++column) {
            minv[column] = CUDART_INF_F;
            used_columns[column] = 0;
        }
        do {
            used_columns[column_zero] = 1;
            const int current_row = p[column_zero];
            float delta = CUDART_INF_F;
            int column_one = 0;
            for (int column = 1; column <= columns; ++column) {
                if (used_columns[column]) continue;
                float edge = temporal_reidentification_gate_px;
                if (column <= tracks) {
                    edge = reid_costs[current_row - 1][column - 1];
                    if (!isfinite(edge)) edge = 1e6F;
                }
                const float current = edge - u[current_row] - v[column];
                if (current < minv[column]) {
                    minv[column] = current;
                    way[column] = column_zero;
                }
                if (minv[column] < delta) {
                    delta = minv[column];
                    column_one = column;
                }
            }
            for (int column = 0; column <= columns; ++column) {
                if (used_columns[column]) {
                    u[p[column]] += delta;
                    v[column] -= delta;
                } else {
                    minv[column] -= delta;
                }
            }
            column_zero = column_one;
        } while (p[column_zero] != 0);
        do {
            const int column_one = way[column_zero];
            p[column_zero] = p[column_one];
            column_zero = column_one;
        } while (column_zero != 0);
    }

    bool seed_matched[temporal_max_tracks]{};
    for (int column = 1; column <= tracks; ++column) {
        const int row = p[column] - 1;
        if (row < 0 || row >= tracks || !dormant[column - 1] ||
            !(reid_costs[row][column - 1] < temporal_reidentification_gate_px)) continue;
        const int track = column - 1;
        seed_matched[row] = true;
        active[track] = 1;
        dormant[track] = 0;
        missed[track] = 0;
        dormant_age[track] = 0;
        for (int joint = 0; joint < temporal_joints; ++joint) {
            const int state = track * temporal_joints + joint;
            joint_valid[state] = 0;
            joint_missed[state] = 0;
            for (int axis = 0; axis < 3; ++axis)
                velocity[state * 3 + axis] = anchor_velocity[state * 3 + axis];
        }
        for (int view = 0; view < views; ++view) {
            const int candidate = seeds[row * views + view];
            if (candidate < temporal_candidates && !used[view * temporal_candidates + candidate] &&
                candidates[view * temporal_candidates + candidate]) {
                assignments[track * views + view] = static_cast<unsigned char>(candidate);
                used[view * temporal_candidates + candidate] = true;
            }
        }
    }

    // Unmatched recovery seeds create fresh identities in free or oldest dormant slots.
    for (int seed = 0; seed < tracks; ++seed) {
        if (seed_matched[seed] || available_views[seed] < 2) continue;
        int slot = -1;
        for (int track = 0; track < tracks; ++track)
            if (!active[track] && !dormant[track]) { slot = track; break; }
        if (slot < 0) {
            int oldest = -1;
            for (int track = 0; track < tracks; ++track)
                if (dormant[track] && (oldest < 0 || dormant_age[track] > dormant_age[oldest])) oldest = track;
            slot = oldest;
        }
        if (slot < 0) continue;
        active[slot] = 1;
        dormant[slot] = 0;
        missed[slot] = 0;
        dormant_age[slot] = 0;
        ids[slot] = (*next_id)++;
        for (int joint = 0; joint < temporal_joints; ++joint) {
            const int state = slot * temporal_joints + joint;
            joint_valid[state] = 0;
            joint_missed[state] = 0;
            anchor_valid[state] = 0;
            anchor_age[state] = 0.0F;
            for (int axis = 0; axis < 3; ++axis) {
                velocity[state * 3 + axis] = 0.0F;
                anchor_xyz[state * 3 + axis] = 0.0F;
                anchor_velocity[state * 3 + axis] = 0.0F;
            }
        }
        for (int view = 0; view < views; ++view) {
            const int candidate = seeds[seed * views + view];
            if (candidate < temporal_candidates && !used[view * temporal_candidates + candidate] &&
                candidates[view * temporal_candidates + candidate]) {
                assignments[slot * views + view] = static_cast<unsigned char>(candidate);
                used[view * temporal_candidates + candidate] = true;
            }
        }
    }
    for (int track = 0; track < tracks; ++track) {
        if (!active[track] || ids[track] != 0) continue;
        bool assigned = false;
        for (int view = 0; view < views; ++view) assigned |= assignments[track * views + view] != 255;
        if (assigned) ids[track] = (*next_id)++;
    }
}

__global__ void temporal_update_kernel(
    const float* points, const float* scores, const unsigned char* candidates,
    const unsigned char* assignments, const float* projections,
    float minimum_score, float max_error, float dt,
    int views, int tracks, float* xyz, float* velocity,
    unsigned char* joint_valid, unsigned char* joint_missed,
    float* anchor_xyz, float* anchor_velocity, float* anchor_age,
    unsigned char* anchor_valid,
    const unsigned char* active) {
    const int work = blockIdx.x * blockDim.x + threadIdx.x;
    if (work >= tracks * temporal_joints) return;
    const int track = work / temporal_joints, joint = work % temporal_joints;
    float normal[4][4]{};
    int observations = 0;
    for (int view = 0; view < views; ++view) {
        const int candidate = assignments[track * views + view];
        if (candidate >= temporal_candidates) continue;
        const int index = (view * temporal_candidates + candidate) * temporal_joints + joint;
        const float score = scores[index], x = points[index * 2], y = points[index * 2 + 1];
        if (!candidates[view * temporal_candidates + candidate] || !isfinite(score) || score < minimum_score || !isfinite(x) || !isfinite(y)) continue;
        const float* p = projections + view * 12;
        const float a[4] = {x*p[8]-p[0], x*p[9]-p[1], x*p[10]-p[2], x*p[11]-p[3]};
        const float b[4] = {y*p[8]-p[4], y*p[9]-p[5], y*p[10]-p[6], y*p[11]-p[7]};
        const float weight = score * score;
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) normal[r][c] += weight * (a[r]*a[c] + b[r]*b[c]);
        ++observations;
    }
    const int state = track * temporal_joints + joint;
    bool solved = observations >= 2;
    float matrix[3][4]{};
    if (solved) {
        for (int r = 0; r < 3; ++r) { for (int c = 0; c < 3; ++c) matrix[r][c] = normal[r][c]; matrix[r][3] = -normal[r][3]; }
        for (int pivot = 0; pivot < 3 && solved; ++pivot) {
            int best = pivot;
            for (int r = pivot + 1; r < 3; ++r) if (fabsf(matrix[r][pivot]) > fabsf(matrix[best][pivot])) best = r;
            if (fabsf(matrix[best][pivot]) < 1e-7F) { solved = false; break; }
            if (best != pivot) for (int c = pivot; c < 4; ++c) { float t = matrix[pivot][c]; matrix[pivot][c] = matrix[best][c]; matrix[best][c] = t; }
            for (int r = pivot + 1; r < 3; ++r) { const float f = matrix[r][pivot] / matrix[pivot][pivot]; for (int c = pivot; c < 4; ++c) matrix[r][c] -= f * matrix[pivot][c]; }
        }
    }
    float measured[3]{};
    if (solved) {
        measured[2] = matrix[2][3] / matrix[2][2];
        measured[1] = (matrix[1][3] - matrix[1][2]*measured[2]) / matrix[1][1];
        measured[0] = (matrix[0][3] - matrix[0][1]*measured[1] - matrix[0][2]*measured[2]) / matrix[0][0];
        solved = isfinite(measured[0]) && isfinite(measured[1]) && isfinite(measured[2]);
    }
    if (solved) {
        for (int view = 0; view < views; ++view) {
            const int candidate = assignments[track * views + view];
            if (candidate >= temporal_candidates) continue;
            const int index = (view * temporal_candidates + candidate) * temporal_joints + joint;
            if (!isfinite(scores[index]) || scores[index] < minimum_score) continue;
            const float* p = projections + view * 12;
            const float w = p[8]*measured[0]+p[9]*measured[1]+p[10]*measured[2]+p[11];
            if (!(w > 1e-5F)) { solved = false; break; }
            const float px = (p[0]*measured[0]+p[1]*measured[1]+p[2]*measured[2]+p[3])/w;
            const float py = (p[4]*measured[0]+p[5]*measured[1]+p[6]*measured[2]+p[7])/w;
            if (hypotf(px-points[index*2], py-points[index*2+1]) > max_error) { solved = false; break; }
        }
    }
    if (solved) {
        if (joint_valid[state] && dt > 1e-4F) {
            for (int axis = 0; axis < 3; ++axis) {
                const float old = xyz[state*3+axis];
                const float measured_velocity = (measured[axis] - old) / dt;
                velocity[state*3+axis] = 0.5F * velocity[state*3+axis] + 0.5F * measured_velocity;
            }
        }
        for (int axis = 0; axis < 3; ++axis) xyz[state*3+axis] = measured[axis];
        joint_valid[state] = 1;
        joint_missed[state] = 0;
        anchor_valid[state] = 1;
        anchor_age[state] = 0.0F;
        for (int axis = 0; axis < 3; ++axis) {
            anchor_xyz[state * 3 + axis] = measured[axis];
            anchor_velocity[state * 3 + axis] = velocity[state * 3 + axis];
        }
    } else if (joint_valid[state]) {
        if (joint_missed[state] == 0) {
            for (int axis = 0; axis < 3; ++axis) xyz[state*3+axis] += velocity[state*3+axis] * dt;
            joint_missed[state] = 1;
        } else {
            joint_valid[state] = 0;
            joint_missed[state] = 0;
            for (int axis = 0; axis < 3; ++axis) velocity[state*3+axis] = 0.0F;
        }
    }
    if (!solved && anchor_valid[state]) anchor_age[state] += dt;
    if (!active[track]) {
        joint_valid[state] = 0;
        joint_missed[state] = 0;
    }
}

__global__ void unwarp_keypoints_kernel(const float* input, const float* intrinsics,
    const float* distortion, const float* mapping, int views,
    float* output, unsigned char* valid) {
    const int work = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = views * temporal_candidates * temporal_joints;
    if (work >= total) return;
    const int view = work / (temporal_candidates * temporal_joints);
    const int point = work * 2;
    const float x = input[point], y = input[point + 1];
    valid[work] = 0;
    output[point] = output[point + 1] = 0.0F;
    if (!isfinite(x) || !isfinite(y)) return;
    const float* meta = mapping + view * 3;
    const float scale = meta[0], pad_x = meta[1], pad_y = meta[2];
    if (!(scale > 0.0F)) return;
    const float* k = intrinsics + view * 9;
    const float* d = distortion + view * 5;
    const float model_x = (x - pad_x) / scale;
    const float model_y = (y - pad_y) / scale;
    const float xu = (model_x - k[2]) / k[0];
    const float yu = (model_y - k[5]) / k[4];
    const float r2 = xu*xu + yu*yu;
    const float radial = 1.0F + d[0]*r2 + d[1]*r2*r2 + d[4]*r2*r2*r2;
    const float xd = xu*radial + 2.0F*d[2]*xu*yu + d[3]*(r2 + 2.0F*xu*xu);
    const float yd = yu*radial + d[2]*(r2 + 2.0F*yu*yu) + 2.0F*d[3]*xu*yu;
    const float px = k[0]*xd + k[2], py = k[4]*yd + k[5];
    if (!isfinite(px) || !isfinite(py)) return;
    output[point] = px; output[point+1] = py; valid[work] = 1;
}

} // namespace

struct MultiviewAssociationWorkspace {
    std::size_t view_count{};
    std::size_t max_persons{};
    std::size_t pair_count{};
    std::size_t edge_count{};
    std::size_t sort_temp_bytes{};
    unsigned char* pair_views{};
    int* canonical_view_order{};
    float* fundamentals{};
    float* pair_costs{};
    unsigned long long* edge_keys_in{};
    unsigned long long* edge_keys_out{};
    void* sort_temp{};
    unsigned char* assignments{};
    unsigned int* output_count{};
    float* host_fundamentals{};
};

struct MultiviewTemporalWorkspace {
    std::size_t views{}, tracks{};
    float *projection{}, *costs{}, *xyz{}, *velocity{}, *anchor_xyz{}, *anchor_velocity{}, *anchor_age{};
    unsigned char *joint_valid{}, *joint_missed{}, *anchor_valid{}, *active{}, *dormant{}, *missed{}, *dormant_age{}, *assignments{};
    std::uint64_t *ids{}, *next_id{};
    unsigned char *host_assignments{}, *host_valid{}, *host_predicted{}, *host_active{};
    std::uint64_t *host_ids{};
    float *host_xyz{};
};

cudaError_t create_multiview_temporal_workspace(std::size_t views, std::size_t tracks,
                                                 MultiviewTemporalWorkspace** output) {
    if (!output || views == 0 || views > temporal_max_views || tracks == 0 || tracks > temporal_max_tracks)
        return cudaErrorInvalidValue;
    *output = nullptr;
    auto* w = new MultiviewTemporalWorkspace{};
    w->views = views; w->tracks = tracks;
    auto fail = [&](cudaError_t e) { destroy_multiview_temporal_workspace(w); return e; };
    cudaError_t e = cudaMalloc(&w->projection, sizeof(float) * views * 12); if (e != cudaSuccess) return fail(e);
    e = cudaMalloc(&w->costs, sizeof(float) * views * tracks * temporal_candidates); if (e != cudaSuccess) return fail(e);
    e = cudaMalloc(&w->xyz, sizeof(float) * tracks * temporal_joints * 3); if (e != cudaSuccess) return fail(e);
    e = cudaMalloc(&w->velocity, sizeof(float) * tracks * temporal_joints * 3); if (e != cudaSuccess) return fail(e);
    e = cudaMalloc(&w->anchor_xyz, sizeof(float) * tracks * temporal_joints * 3); if (e != cudaSuccess) return fail(e);
    e = cudaMalloc(&w->anchor_velocity, sizeof(float) * tracks * temporal_joints * 3); if (e != cudaSuccess) return fail(e);
    e = cudaMalloc(&w->anchor_age, sizeof(float) * tracks * temporal_joints); if (e != cudaSuccess) return fail(e);
    e = cudaMalloc(&w->joint_valid, sizeof(unsigned char) * tracks * temporal_joints); if (e != cudaSuccess) return fail(e);
    e = cudaMalloc(&w->joint_missed, sizeof(unsigned char) * tracks * temporal_joints); if (e != cudaSuccess) return fail(e);
    e = cudaMalloc(&w->anchor_valid, sizeof(unsigned char) * tracks * temporal_joints); if (e != cudaSuccess) return fail(e);
    e = cudaMalloc(&w->active, sizeof(unsigned char) * tracks); if (e != cudaSuccess) return fail(e);
    e = cudaMalloc(&w->dormant, sizeof(unsigned char) * tracks); if (e != cudaSuccess) return fail(e);
    e = cudaMalloc(&w->missed, sizeof(unsigned char) * tracks); if (e != cudaSuccess) return fail(e);
    e = cudaMalloc(&w->dormant_age, sizeof(unsigned char) * tracks); if (e != cudaSuccess) return fail(e);
    e = cudaMalloc(&w->assignments, sizeof(unsigned char) * tracks * views); if (e != cudaSuccess) return fail(e);
    e = cudaMalloc(&w->ids, sizeof(std::uint64_t) * tracks); if (e != cudaSuccess) return fail(e);
    e = cudaMalloc(&w->next_id, sizeof(std::uint64_t)); if (e != cudaSuccess) return fail(e);
    e = cudaMemset(w->xyz, 0, sizeof(float) * tracks * temporal_joints * 3); if (e != cudaSuccess) return fail(e);
    e = cudaMemset(w->velocity, 0, sizeof(float) * tracks * temporal_joints * 3); if (e != cudaSuccess) return fail(e);
    e = cudaMemset(w->anchor_xyz, 0, sizeof(float) * tracks * temporal_joints * 3); if (e != cudaSuccess) return fail(e);
    e = cudaMemset(w->anchor_velocity, 0, sizeof(float) * tracks * temporal_joints * 3); if (e != cudaSuccess) return fail(e);
    e = cudaMemset(w->anchor_age, 0, sizeof(float) * tracks * temporal_joints); if (e != cudaSuccess) return fail(e);
    e = cudaMemset(w->joint_valid, 0, sizeof(unsigned char) * tracks * temporal_joints); if (e != cudaSuccess) return fail(e);
    e = cudaMemset(w->joint_missed, 0, sizeof(unsigned char) * tracks * temporal_joints); if (e != cudaSuccess) return fail(e);
    e = cudaMemset(w->anchor_valid, 0, sizeof(unsigned char) * tracks * temporal_joints); if (e != cudaSuccess) return fail(e);
    e = cudaMemset(w->active, 0, sizeof(unsigned char) * tracks); if (e != cudaSuccess) return fail(e);
    e = cudaMemset(w->dormant, 0, sizeof(unsigned char) * tracks); if (e != cudaSuccess) return fail(e);
    e = cudaMemset(w->missed, 0, sizeof(unsigned char) * tracks); if (e != cudaSuccess) return fail(e);
    e = cudaMemset(w->dormant_age, 0, sizeof(unsigned char) * tracks); if (e != cudaSuccess) return fail(e);
    e = cudaMemset(w->ids, 0, sizeof(std::uint64_t) * tracks); if (e != cudaSuccess) return fail(e);
    { const std::uint64_t first_id = 1; e = cudaMemcpy(w->next_id, &first_id, sizeof(first_id), cudaMemcpyHostToDevice); }
    if (e != cudaSuccess) return fail(e);
    e = cudaHostAlloc(&w->host_assignments, sizeof(unsigned char) * tracks * views, cudaHostAllocPortable); if (e != cudaSuccess) return fail(e);
    e = cudaHostAlloc(&w->host_ids, sizeof(std::uint64_t) * tracks, cudaHostAllocPortable); if (e != cudaSuccess) return fail(e);
    e = cudaHostAlloc(&w->host_xyz, sizeof(float) * tracks * temporal_joints * 3, cudaHostAllocPortable); if (e != cudaSuccess) return fail(e);
    e = cudaHostAlloc(&w->host_valid, sizeof(unsigned char) * tracks * temporal_joints, cudaHostAllocPortable); if (e != cudaSuccess) return fail(e);
    e = cudaHostAlloc(&w->host_predicted, sizeof(unsigned char) * tracks * temporal_joints, cudaHostAllocPortable); if (e != cudaSuccess) return fail(e);
    e = cudaHostAlloc(&w->host_active, sizeof(unsigned char) * tracks, cudaHostAllocPortable); if (e != cudaSuccess) return fail(e);
    *output = w;
    return cudaSuccess;
}

void destroy_multiview_temporal_workspace(MultiviewTemporalWorkspace* w) {
    if (!w) return;
    if (w->host_active) cudaFreeHost(w->host_active);
    if (w->host_predicted) cudaFreeHost(w->host_predicted);
    if (w->host_valid) cudaFreeHost(w->host_valid);
    if (w->host_xyz) cudaFreeHost(w->host_xyz);
    if (w->host_ids) cudaFreeHost(w->host_ids);
    if (w->host_assignments) cudaFreeHost(w->host_assignments);
    if (w->next_id) cudaFree(w->next_id);
    if (w->ids) cudaFree(w->ids);
    if (w->assignments) cudaFree(w->assignments);
    if (w->missed) cudaFree(w->missed);
    if (w->dormant_age) cudaFree(w->dormant_age);
    if (w->dormant) cudaFree(w->dormant);
    if (w->active) cudaFree(w->active);
    if (w->anchor_valid) cudaFree(w->anchor_valid);
    if (w->joint_missed) cudaFree(w->joint_missed);
    if (w->joint_valid) cudaFree(w->joint_valid);
    if (w->anchor_age) cudaFree(w->anchor_age);
    if (w->anchor_velocity) cudaFree(w->anchor_velocity);
    if (w->anchor_xyz) cudaFree(w->anchor_xyz);
    if (w->velocity) cudaFree(w->velocity);
    if (w->xyz) cudaFree(w->xyz);
    if (w->costs) cudaFree(w->costs);
    if (w->projection) cudaFree(w->projection);
    delete w;
}

cudaError_t launch_multiview_temporal_update(MultiviewTemporalWorkspace* w,
    const float* keypoints, const float* scores, const unsigned char* candidate_valid,
    const unsigned char* seeds, const float* projections_host, float dt,
    float gate, float minimum_score, float maximum_reprojection_error,
    cudaEvent_t assignment_begin, cudaEvent_t assignment_end, cudaStream_t stream) {
    if (!w || !keypoints || !scores || !candidate_valid || !seeds || !projections_host)
        return cudaErrorInvalidValue;
    dt = fminf(0.1F, fmaxf(0.0F, dt));
    cudaError_t e = cudaMemcpyAsync(w->projection, projections_host,
        sizeof(float) * w->views * 12, cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) return e;
    const int work = static_cast<int>(w->views * w->tracks * temporal_candidates);
    temporal_cost_kernel<<<(work + 127) / 128, 128, 0, stream>>>(keypoints, scores, candidate_valid,
        w->projection, w->xyz, w->velocity, w->joint_valid, w->active, dt, minimum_score,
        gate, static_cast<int>(w->views), static_cast<int>(w->tracks), w->costs);
    e = cudaGetLastError(); if (e != cudaSuccess) return e;
    e = cudaEventRecord(assignment_begin, stream); if (e != cudaSuccess) return e;
    temporal_assignment_kernel<<<static_cast<unsigned>(w->views), 32, 0, stream>>>(
        w->costs, w->active, w->assignments, static_cast<int>(w->views),
        static_cast<int>(w->tracks), gate);
    e = cudaGetLastError(); if (e != cudaSuccess) return e;
    temporal_lifecycle_kernel<<<1, 1, 0, stream>>>(keypoints, scores, candidate_valid, seeds,
        w->projection, w->assignments, w->active, w->dormant, w->missed, w->dormant_age,
        w->joint_valid, w->joint_missed, w->velocity, w->anchor_xyz, w->anchor_velocity,
        w->anchor_age, w->anchor_valid, w->ids, w->next_id, minimum_score,
        static_cast<int>(w->views), static_cast<int>(w->tracks));
    e = cudaGetLastError(); if (e != cudaSuccess) return e;
    e = cudaEventRecord(assignment_end, stream); if (e != cudaSuccess) return e;
    const int joint_work = static_cast<int>(w->tracks * temporal_joints);
    temporal_update_kernel<<<(joint_work + 127) / 128, 128, 0, stream>>>(keypoints, scores,
        candidate_valid, w->assignments, w->projection, minimum_score, maximum_reprojection_error, dt,
        static_cast<int>(w->views), static_cast<int>(w->tracks), w->xyz, w->velocity,
        w->joint_valid, w->joint_missed, w->anchor_xyz, w->anchor_velocity,
        w->anchor_age, w->anchor_valid, w->active);
    return cudaGetLastError();
}

cudaError_t launch_multiview_unwarp_keypoints(const float* model_keypoints,
    const float* source_intrinsics, const float* distortion, const float* mapping,
    std::size_t views, float* image_keypoints, unsigned char* joint_valid,
    cudaStream_t stream) {
    if (!model_keypoints || !source_intrinsics || !distortion || !mapping ||
        !image_keypoints || !joint_valid || views == 0 || views > temporal_max_views)
        return cudaErrorInvalidValue;
    const int work = static_cast<int>(views * temporal_candidates * temporal_joints);
    unwarp_keypoints_kernel<<<(work + 255) / 256, 256, 0, stream>>>(model_keypoints,
        source_intrinsics, distortion, mapping, static_cast<int>(views), image_keypoints,
        joint_valid);
    return cudaGetLastError();
}

cudaError_t copy_multiview_temporal_result(MultiviewTemporalWorkspace* w,
    unsigned char* assignments, std::uint64_t* ids, float* xyz, unsigned char* valid,
    unsigned char* predicted,
    std::uint32_t* count, cudaStream_t stream) {
    if (!w || !assignments || !ids || !xyz || !valid || !predicted || !count) return cudaErrorInvalidValue;
    cudaError_t e = cudaMemcpyAsync(w->host_assignments, w->assignments,
        sizeof(unsigned char) * w->tracks * w->views, cudaMemcpyDeviceToHost, stream); if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(w->host_ids, w->ids, sizeof(std::uint64_t) * w->tracks, cudaMemcpyDeviceToHost, stream); if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(w->host_xyz, w->xyz, sizeof(float) * w->tracks * temporal_joints * 3, cudaMemcpyDeviceToHost, stream); if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(w->host_valid, w->joint_valid, sizeof(unsigned char) * w->tracks * temporal_joints, cudaMemcpyDeviceToHost, stream); if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(w->host_predicted, w->joint_missed, sizeof(unsigned char) * w->tracks * temporal_joints, cudaMemcpyDeviceToHost, stream); if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(w->host_active, w->active, sizeof(unsigned char) * w->tracks, cudaMemcpyDeviceToHost, stream); if (e != cudaSuccess) return e;
    e = cudaStreamSynchronize(stream); if (e != cudaSuccess) return e;
    std::memcpy(assignments, w->host_assignments, sizeof(unsigned char) * w->tracks * w->views);
    std::memcpy(ids, w->host_ids, sizeof(std::uint64_t) * w->tracks);
    for (std::size_t i = 0; i < w->tracks; ++i) if (!w->host_active[i]) ids[i] = 0;
    std::memcpy(xyz, w->host_xyz, sizeof(float) * w->tracks * temporal_joints * 3);
    std::memcpy(valid, w->host_valid, sizeof(unsigned char) * w->tracks * temporal_joints);
    std::memcpy(predicted, w->host_predicted, sizeof(unsigned char) * w->tracks * temporal_joints);
    *count = 0;
    for (std::size_t i = 0; i < w->tracks; ++i) *count += w->host_active[i] != 0;
    return cudaSuccess;
}

cudaError_t create_multiview_association_workspace(
    std::size_t view_count, std::size_t max_persons,
    const std::uint32_t* camera_ids,
    MultiviewAssociationWorkspace** output, cudaStream_t stream) {
    if (!output || !camera_ids || view_count == 0 || view_count > association_max_views ||
        max_persons == 0 || max_persons > association_candidate_capacity)
        return cudaErrorInvalidValue;
    *output = nullptr;
    auto* workspace = new MultiviewAssociationWorkspace{};
    workspace->view_count = view_count;
    workspace->max_persons = max_persons;

    std::vector<int> view_order(view_count);
    for (std::size_t view = 0; view < view_count; ++view) view_order[view] = static_cast<int>(view);
    std::sort(view_order.begin(), view_order.end(), [&](int lhs, int rhs) {
        return camera_ids[lhs] < camera_ids[rhs];
    });
    std::vector<unsigned char> pair_views;
    for (std::size_t first = 0; first < view_count; ++first)
        for (std::size_t second = first + 1; second < view_count; ++second) {
            pair_views.push_back(static_cast<unsigned char>(view_order[first]));
            pair_views.push_back(static_cast<unsigned char>(view_order[second]));
        }
    workspace->pair_count = pair_views.size() / 2;
    workspace->edge_count = workspace->pair_count * association_candidate_capacity * association_candidate_capacity;

    auto fail = [&](cudaError_t error) {
        destroy_multiview_association_workspace(workspace);
        return error;
    };
    cudaError_t error = cudaMalloc(&workspace->fundamentals,
        sizeof(float) * view_count * view_count * 9);
    if (error != cudaSuccess) return fail(error);
    error = cudaMalloc(&workspace->pair_costs,
        sizeof(float) * view_count * association_candidate_capacity *
        view_count * association_candidate_capacity);
    if (error != cudaSuccess) return fail(error);
    error = cudaMalloc(&workspace->canonical_view_order, sizeof(int) * view_count);
    if (error != cudaSuccess) return fail(error);
    error = cudaMemcpy(workspace->canonical_view_order, view_order.data(),
                       sizeof(int) * view_count, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return fail(error);
    error = cudaMalloc(&workspace->assignments,
                       sizeof(unsigned char) * max_persons * view_count);
    if (error != cudaSuccess) return fail(error);
    error = cudaMalloc(&workspace->output_count, sizeof(unsigned int));
    if (error != cudaSuccess) return fail(error);
    error = cudaHostAlloc(&workspace->host_fundamentals,
        sizeof(float) * view_count * view_count * 9, cudaHostAllocPortable);
    if (error != cudaSuccess) return fail(error);
    if (workspace->edge_count > 0) {
        error = cudaMalloc(&workspace->pair_views, sizeof(unsigned char) * pair_views.size());
        if (error != cudaSuccess) return fail(error);
        error = cudaMemcpy(workspace->pair_views, pair_views.data(), pair_views.size(), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) return fail(error);
        error = cudaMalloc(&workspace->edge_keys_in, sizeof(unsigned long long) * workspace->edge_count);
        if (error != cudaSuccess) return fail(error);
        error = cudaMalloc(&workspace->edge_keys_out, sizeof(unsigned long long) * workspace->edge_count);
        if (error != cudaSuccess) return fail(error);
        error = cub::DeviceRadixSort::SortKeys(nullptr, workspace->sort_temp_bytes,
            workspace->edge_keys_in, workspace->edge_keys_out,
            static_cast<int>(workspace->edge_count), 0, 64, stream);
        if (error != cudaSuccess) return fail(error);
        error = cudaMalloc(&workspace->sort_temp, workspace->sort_temp_bytes);
        if (error != cudaSuccess) return fail(error);
    }
    *output = workspace;
    return cudaSuccess;
}

void destroy_multiview_association_workspace(MultiviewAssociationWorkspace* workspace) {
    if (!workspace) return;
    if (workspace->sort_temp) cudaFree(workspace->sort_temp);
    if (workspace->host_fundamentals) cudaFreeHost(workspace->host_fundamentals);
    if (workspace->edge_keys_out) cudaFree(workspace->edge_keys_out);
    if (workspace->edge_keys_in) cudaFree(workspace->edge_keys_in);
    if (workspace->pair_views) cudaFree(workspace->pair_views);
    if (workspace->output_count) cudaFree(workspace->output_count);
    if (workspace->assignments) cudaFree(workspace->assignments);
    if (workspace->canonical_view_order) cudaFree(workspace->canonical_view_order);
    if (workspace->pair_costs) cudaFree(workspace->pair_costs);
    if (workspace->fundamentals) cudaFree(workspace->fundamentals);
    delete workspace;
}

cudaError_t launch_multiview_current_association(
    MultiviewAssociationWorkspace* workspace,
    const float* keypoints, const float* scores,
    const unsigned char* candidate_valid,
    const float* fundamentals,
    float gate_px, float minimum_score,
    cudaStream_t stream, double* host_ms) {
    if (!workspace || !keypoints || !scores || !candidate_valid || !fundamentals)
        return cudaErrorInvalidValue;
    const auto start = std::chrono::steady_clock::now();
    if (workspace->edge_count == 0) {
        if (host_ms) *host_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        return cudaSuccess;
    }

    const auto fundamental_bytes = sizeof(float) * workspace->view_count * workspace->view_count * 9;
    std::memcpy(workspace->host_fundamentals, fundamentals, fundamental_bytes);
    cudaError_t error = cudaMemcpyAsync(workspace->fundamentals, workspace->host_fundamentals,
        fundamental_bytes,
        cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess) return error;
    const int edge_count = static_cast<int>(workspace->edge_count);
    association_pair_cost_kernel<<<(edge_count + 127) / 128, 128, 0, stream>>>(
        keypoints, scores, candidate_valid, workspace->fundamentals,
        static_cast<int>(workspace->view_count), static_cast<int>(workspace->pair_count),
        workspace->pair_views, gate_px, minimum_score, workspace->pair_costs,
        workspace->edge_keys_in);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    error = cub::DeviceRadixSort::SortKeys(workspace->sort_temp,
        workspace->sort_temp_bytes, workspace->edge_keys_in, workspace->edge_keys_out,
        edge_count, 0, 64, stream);
    if (error != cudaSuccess) return error;
    association_merge_kernel<<<1, 1, 0, stream>>>(
        workspace->edge_keys_out, edge_count, workspace->pair_views,
        static_cast<int>(workspace->pair_count), static_cast<int>(workspace->view_count),
        static_cast<int>(workspace->max_persons), gate_px, workspace->pair_costs,
        workspace->canonical_view_order, workspace->assignments, workspace->output_count);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    if (host_ms) *host_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return cudaSuccess;
}

cudaError_t clear_multiview_association_assignments(
    MultiviewAssociationWorkspace* workspace, cudaStream_t stream) {
    if (!workspace || !workspace->assignments) return cudaErrorInvalidValue;
    return cudaMemsetAsync(workspace->assignments, 255,
        workspace->max_persons * workspace->view_count, stream);
}

const unsigned char* multiview_association_assignments_device(
    const MultiviewAssociationWorkspace* workspace) {
    return workspace ? workspace->assignments : nullptr;
}

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
