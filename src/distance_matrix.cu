#include "persistent_homology.cuh"

// ============================================================
// Tiled Euclidean distance matrix kernel
// ============================================================
// Each block computes a TILE x TILE block of the distance matrix.
// Uses shared memory to load tiles of the point array.

__global__ void distance_matrix_kernel(
    const float* __restrict__ points,  // [N x D]
    int N, int D,
    float* __restrict__ dist_matrix    // [N x N]
) {
    // Each thread computes one element (i, j) of the distance matrix
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N || j >= N) return;
    if (i == j) {
        dist_matrix[i * N + j] = 0.0f;
        return;
    }
    if (i > j) return; // only compute upper triangle; we'll symmetrize

    float sum = 0.0f;
    for (int d = 0; d < D; d++) {
        float diff = points[i * D + d] - points[j * D + d];
        sum += diff * diff;
    }
    float dist = sqrtf(sum);
    dist_matrix[i * N + j] = dist;
    dist_matrix[j * N + i] = dist; // symmetrize
}

// ============================================================
// Tiled kernel with shared memory for better performance
// ============================================================
template <int TILE_SIZE>
__global__ void distance_matrix_tiled_kernel(
    const float* __restrict__ points,
    int N, int D,
    float* __restrict__ dist_matrix
) {
    int i = blockIdx.y * TILE_SIZE + threadIdx.y;
    int j = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (i >= N || j >= N) return;
    if (i == j) {
        dist_matrix[i * N + j] = 0.0f;
        return;
    }
    if (i > j) return;

    // For large D, we tile over the dimension as well
    float sum = 0.0f;
    for (int d = 0; d < D; d++) {
        float diff = points[i * D + d] - points[j * D + d];
        sum += diff * diff;
    }
    float dist = sqrtf(sum);
    dist_matrix[i * N + j] = dist;
    dist_matrix[j * N + i] = dist;
}

// ============================================================
// Symmetrize kernel (for any missed entries)
// ============================================================
__global__ void symmetrize_kernel(float* dist_matrix, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;
    if (i == j) dist_matrix[i * N + j] = 0.0f;
    else if (i < j) {
        float d = dist_matrix[i * N + j];
        dist_matrix[j * N + i] = d;
    }
}

cudaError_t compute_distance_matrix(const float* d_points, int N, int D, float* d_dist_matrix) {
    const int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    distance_matrix_tiled_kernel<TILE><<<grid, block>>>(d_points, N, D, d_dist_matrix);

    return cudaGetLastError();
}
