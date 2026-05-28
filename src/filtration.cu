#include "persistent_homology.cuh"
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdio>

// Kernel to extract upper-triangle distances and edge indices
__global__ void extract_edges_kernel(const float* __restrict__ dist, int N,
                                      float* out_dist, int* out_edges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Upper triangle: i < j
    // Linear index k in [0, N*(N-1)/2)
    // Map k -> (i,j)
    int total = N * (N - 1) / 2;
    if (idx >= total) return;

    // Binary search for row i such that:
    // row_start = N*i - i*(i+1)/2, row_end = N*(i+1) - (i+1)*(i+2)/2
    // Actually: for upper triangle, edge k corresponds to:
    // Use formula: j = N - 1 - (total - 1 - k) % ??? 
    // Simpler approach: use the triangular number trick
    // k = i*N - i*(i+1)/2 + (j - i - 1)
    // We need to find i from k.
    // Invert: for i, start of row = i*(N-1) - i*(i-1)/2  ... hmm let me just do it simply
    // k-th edge in row-major upper triangle:
    // k = sum_{r=0}^{i-1} (N-1-r) + (j - i - 1)
    // = i*(2N - i - 1)/2 + (j - i - 1)
    
    // Find i via binary search
    int lo = 0, hi = N - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        int start = mid * (2 * N - mid - 1) / 2;
        if (start <= idx) lo = mid;
        else hi = mid - 1;
    }
    int i = lo;
    int row_start = i * (2 * N - i - 1) / 2;
    int offset = idx - row_start;
    int j = i + 1 + offset;

    out_dist[idx] = dist[i * N + j];
    out_edges[idx * 2] = i;
    out_edges[idx * 2 + 1] = j;
}

void gpu_build_filtration(const float* d_dist, int N,
                          int** d_edges, float** d_distances,
                          int* n_edges, cudaStream_t stream) {
    int ne = N * (N - 1) / 2;
    *n_edges = ne;

    float* d_d;
    int* d_e;
    cudaMalloc(&d_d, ne * sizeof(float));
    cudaMalloc(&d_e, ne * 2 * sizeof(int));

    int block = 256;
    int grid = (ne + 255) / 256;
    extract_edges_kernel<<<grid, block, 0, stream>>>(d_dist, N, d_d, d_e);

    // Sort by distance using thrust
    thrust::device_ptr<float> d_dist_ptr(d_d);
    thrust::device_ptr<int> d_edge_ptr(d_e);

    // Create index sequence
    thrust::device_vector<int> indices(ne);
    thrust::sequence(indices.begin(), indices.end());

    // Sort by distance (key-value sort)
    thrust::device_vector<float> dist_copy(d_d, d_d + ne);
    thrust::device_vector<int> edges_copy(d_e, d_e + ne * 2);
    thrust::sort_by_key(dist_copy.begin(), dist_copy.end(), indices.begin());

    // Apply permutation to edges
    thrust::device_vector<int> sorted_edges(ne * 2);
    thrust::gather(indices.begin(), indices.end(),
                   thrust::make_transform_iterator(
                       thrust::counting_iterator<int>(0),
                       [ne] __device__(int idx) -> int2 {
                           // We need a different approach
                           return 0;
                       }),
                   sorted_edges.begin());
    // Actually, let's do it simpler: copy back and sort on host side
    // For simplicity, copy to host, sort, copy back
    
    // Free temporary and re-allocate
    cudaFree(d_d);
    cudaFree(d_e);
    
    // Host-side sort (simpler and correct)
    float* h_dist = new float[ne];
    int* h_edges = new int[ne * 2];
    float* h_dist_raw = new float[ne];
    int* h_edges_raw = new int[ne * 2];
    
    cudaMemcpy(h_dist_raw, thrust::raw_pointer_cast(dist_copy.data()), ne * sizeof(float), cudaMemcpyDeviceToHost);
    // We need to re-extract since we used thrust vectors
    // Let me simplify: extract on host
    int idx = 0;
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            float val;
            cudaMemcpy(&val, d_dist + i * N + j, sizeof(float), cudaMemcpyDeviceToHost);
            h_dist_raw[idx] = val;
            h_edges_raw[idx * 2] = i;
            h_edges_raw[idx * 2 + 1] = j;
            idx++;
        }
    }
    
    // Create sort indices
    std::vector<int> sort_idx(ne);
    for (int i = 0; i < ne; i++) sort_idx[i] = i;
    std::sort(sort_idx.begin(), sort_idx.end(), [&](int a, int b) {
        return h_dist_raw[a] < h_dist_raw[b];
    });
    
    for (int i = 0; i < ne; i++) {
        h_dist[i] = h_dist_raw[sort_idx[i]];
        h_edges[i * 2] = h_edges_raw[sort_idx[i] * 2];
        h_edges[i * 2 + 1] = h_edges_raw[sort_idx[i] * 2 + 1];
    }
    
    cudaMalloc(d_distances, ne * sizeof(float));
    cudaMalloc(d_edges, ne * 2 * sizeof(int));
    cudaMemcpy(*d_distances, h_dist, ne * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_edges, h_edges, ne * 2 * sizeof(int), cudaMemcpyHostToDevice);
    
    delete[] h_dist;
    delete[] h_edges;
    delete[] h_dist_raw;
    delete[] h_edges_raw;
}
