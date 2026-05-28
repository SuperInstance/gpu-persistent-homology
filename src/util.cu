#include "persistent_homology.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

// Utility implementations
float* gpu_upload_points(const float* h_points, int N, int dim) {
    float* d_points;
    cudaMalloc(&d_points, N * dim * sizeof(float));
    cudaMemcpy(d_points, h_points, N * dim * sizeof(float), cudaMemcpyHostToDevice);
    return d_points;
}

void gpu_free(void* d_ptr) {
    cudaFree(d_ptr);
}
