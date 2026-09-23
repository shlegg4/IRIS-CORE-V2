#include "iris/stages/pose/MultiviewCudaPostprocess.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>

static bool cuda_ok(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return true;
    std::cerr << operation << ": " << cudaGetErrorString(error) << "\n";
    return false;
}

int main() {
    int devices = 0;
    if (!cuda_ok(cudaGetDeviceCount(&devices), "cudaGetDeviceCount") || devices == 0) return 0;

    std::array<float, 3 * 10 * 17 * 2> points{};
    std::array<float, 3 * 10 * 17> scores{};
    std::array<unsigned char, 3 * 10> candidates{};
    for (int view = 0; view < 3; ++view) {
        candidates[view * 10] = 1;
        candidates[view * 10 + 1] = 1;
        for (int joint = 0; joint < 17; ++joint) {
            const auto index = (view * 10) * 17 + joint;
            points[index * 2] = view == 0 ? 0.0F : (view == 1 ? -0.2F : 0.2F);
            points[index * 2 + 1] = 0.0F;
            scores[index] = 1.0F;
            const auto second = (view * 10 + 1) * 17 + joint;
            points[second * 2] = points[index * 2];
            points[second * 2 + 1] = points[index * 2 + 1];
            scores[second] = 1.0F;
        }
    }
    std::array<float, 27> fundamentals{};
    std::array<float, 36> projections{};
    for (int view = 0; view < 3; ++view) {
        projections[view * 12 + 0] = 1.0F;
        projections[view * 12 + 5] = 1.0F;
        projections[view * 12 + 10] = 1.0F;
        projections[view * 12 + 3] = view == 1 ? -1.0F : (view == 2 ? 1.0F : 0.0F);
    }
    std::array<unsigned char, 30> assignments{};
    std::array<float, 10 * 17 * 3> xyz{};
    std::array<unsigned char, 10 * 17> valid{};
    float *d_points{}, *d_scores{}, *d_fundamentals{}, *d_projections{}, *d_xyz{};
    unsigned char *d_candidates{}, *d_assignments{}, *d_valid{};
    assert(cudaMalloc(reinterpret_cast<void**>(&d_points), sizeof(points)) == cudaSuccess);
    assert(cudaMalloc(reinterpret_cast<void**>(&d_scores), sizeof(scores)) == cudaSuccess);
    assert(cudaMalloc(reinterpret_cast<void**>(&d_fundamentals), sizeof(fundamentals)) == cudaSuccess);
    assert(cudaMalloc(reinterpret_cast<void**>(&d_projections), sizeof(projections)) == cudaSuccess);
    assert(cudaMalloc(reinterpret_cast<void**>(&d_xyz), sizeof(xyz)) == cudaSuccess);
    assert(cudaMalloc(reinterpret_cast<void**>(&d_candidates), sizeof(candidates)) == cudaSuccess);
    assert(cudaMalloc(reinterpret_cast<void**>(&d_assignments), sizeof(assignments)) == cudaSuccess);
    assert(cudaMalloc(reinterpret_cast<void**>(&d_valid), sizeof(valid)) == cudaSuccess);
    cudaMemcpy(d_points, points.data(), sizeof(points), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scores, scores.data(), sizeof(scores), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fundamentals, fundamentals.data(), sizeof(fundamentals), cudaMemcpyHostToDevice);
    cudaMemcpy(d_projections, projections.data(), sizeof(projections), cudaMemcpyHostToDevice);
    cudaMemcpy(d_candidates, candidates.data(), sizeof(candidates), cudaMemcpyHostToDevice);
    cudaStream_t stream{}; assert(cudaStreamCreate(&stream) == cudaSuccess);
    if (!cuda_ok(iris::launch_multiview_epipolar_assignment(d_points, d_scores, d_candidates, d_fundamentals, 2.0F, 0.1F, d_assignments, stream), "epipolar assignment launch")) return 1;
    if (!cuda_ok(iris::launch_multiview_weighted_dlt(d_points, d_scores, d_candidates, d_assignments, d_projections, 0.1F, 2.0F, d_xyz, d_valid, stream), "triangulation launch")) return 1;
    assert(cudaMemcpyAsync(assignments.data(), d_assignments, sizeof(assignments), cudaMemcpyDeviceToHost, stream) == cudaSuccess);
    assert(cudaMemcpyAsync(xyz.data(), d_xyz, sizeof(xyz), cudaMemcpyDeviceToHost, stream) == cudaSuccess);
    assert(cudaMemcpyAsync(valid.data(), d_valid, sizeof(valid), cudaMemcpyDeviceToHost, stream) == cudaSuccess);
    if (!cuda_ok(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) return 1;
    std::cerr << "assign=" << static_cast<int>(assignments[0]) << "," << static_cast<int>(assignments[1]) << "," << static_cast<int>(assignments[2]) << " gpu valid=" << static_cast<int>(valid[0]) << " xyz=" << xyz[0] << "," << xyz[1] << "," << xyz[2] << "\n";
    const bool passed = assignments[0] == 0 && assignments[1] == 0 && assignments[2] == 0 &&
        assignments[3] == 1 && assignments[4] == 1 && assignments[5] == 1 && valid[0] != 0 && valid[17] != 0 &&
        std::abs(xyz[0]) < 1e-3F && std::abs(xyz[1]) < 1e-3F && std::abs(xyz[2] - 5.0F) < 1e-2F;
    points[10 * 17 * 2] = 100.0F;
    assert(cudaMemcpy(d_points, points.data(), sizeof(points), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(iris::launch_multiview_weighted_dlt(d_points, d_scores, d_candidates, d_assignments, d_projections,
        0.1F, 2.0F, d_xyz, d_valid, stream) == cudaSuccess);
    assert(cudaMemcpyAsync(valid.data(), d_valid, sizeof(valid), cudaMemcpyDeviceToHost, stream) == cudaSuccess);
    assert(cudaStreamSynchronize(stream) == cudaSuccess);
    const bool rejects_reprojection_outlier = valid[0] == 0;
    cudaStreamDestroy(stream);
    cudaFree(d_points); cudaFree(d_scores); cudaFree(d_fundamentals); cudaFree(d_projections);
    cudaFree(d_xyz); cudaFree(d_candidates); cudaFree(d_assignments); cudaFree(d_valid);
    return passed && rejects_reprojection_outlier ? 0 : 1;
}
