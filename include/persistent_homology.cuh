#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <vector>
#include <utility>
#include <cstdio>
#include <cmath>
#include <cstdint>

// ============================================================
// Persistence diagram point (birth, death)
// ============================================================
struct PDPoint {
    float birth;
    float death;
};

// ============================================================
// Edge for filtration (undirected: i < j)
// ============================================================
struct Edge {
    float dist;
    int i;
    int j;
};

// ============================================================
// API: Distance matrix
// ============================================================
// Compute tiled pairwise Euclidean distances.
// points: flat row-major [N x D], device pointer
// dist_matrix: output [N x N], device pointer (symmetric, diagonal = 0)
cudaError_t compute_distance_matrix(const float* d_points, int N, int D, float* d_dist_matrix);

// ============================================================
// API: H0 persistence
// ============================================================
// Input: edges sorted by distance (thrust device vector)
// Output: persistence diagram (birth, death pairs for finite H0 features)
// Also returns component labels (parent array) for use in H1.
cudaError_t compute_h0_persistence(
    int num_points,
    thrust::device_vector<Edge>& d_edges,
    thrust::device_vector<PDPoint>& d_h0_diagram,
    thrust::device_vector<int>& d_parents
);

// ============================================================
// API: H1 persistence
// ============================================================
// Input: edges sorted by distance, parents from H0
// Output: persistence diagram for H1 features
cudaError_t compute_h1_persistence(
    int num_points,
    thrust::device_vector<Edge>& d_edges,
    const thrust::device_vector<int>& d_parents,
    thrust::device_vector<PDPoint>& d_h1_diagram
);

// ============================================================
// API: Wasserstein-1 distance
// ============================================================
// Approximate Wasserstein-1 distance between two persistence diagrams.
float compute_wasserstein_distance(
    const thrust::host_vector<PDPoint>& diag1,
    const thrust::host_vector<PDPoint>& diag2
);
