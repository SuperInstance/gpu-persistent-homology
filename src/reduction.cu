#include "persistent_homology.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <thrust/device_vector.h>

// Standard persistence algorithm: column reduction (brute-force, correctness first)
// Reduce boundary matrix to Smith normal form (over Z/2Z)
// Each column is reduced from left to right

// GPU kernel: reduce one column at a time (sequential dependency)
// For a correct implementation, we must process columns left-to-right
// and each column reduction may depend on previous columns

__global__ void reduce_columns_kernel(float* matrix, int n) {
    // Single thread does the full reduction (standard algorithm)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // lowest_one[col] = row index of lowest 1 in column col (or -1)
    int* lowest_one = new int[n];
    for (int i = 0; i < n; i++) lowest_one[i] = -1;
    
    for (int col = 0; col < n; col++) {
        // Find lowest 1 in this column
        int low = -1;
        for (int row = n - 1; row >= 0; row--) {
            if (matrix[row * n + col] != 0.0f) {
                low = row;
                break;
            }
        }
        
        // While low != -1 and there's another column with same low
        while (low != -1) {
            // Check if any previous column has this low
            bool found = false;
            int match_col = -1;
            for (int c = 0; c < col; c++) {
                if (lowest_one[c] == low) {
                    match_col = c;
                    found = true;
                    break;
                }
            }
            
            if (!found) break;
            
            // Add column match_col to column col (XOR in Z/2Z)
            for (int row = 0; row <= low; row++) {
                float v = matrix[row * n + match_col];
                if (v != 0.0f) {
                    matrix[row * n + col] = (matrix[row * n + col] != 0.0f) ? 0.0f : 1.0f;
                }
            }
            
            // Recompute low
            low = -1;
            for (int row = n - 1; row >= 0; row--) {
                if (matrix[row * n + col] != 0.0f) {
                    low = row;
                    break;
                }
            }
        }
        
        lowest_one[col] = low;
    }
    
    delete[] lowest_one;
}

// Extract persistence pairs from reduced matrix
// A pair (i,j) means simplex i was born and died when simplex j was added
// low[j] = i means simplex j killed simplex i
// If low[j] = -1, simplex j creates a new feature (essential)

void extract_persistence_pairs(const float* matrix, int n,
                                const std::vector<struct SimplexWrapper>& simplices,
                                int target_dim,
                                std::vector<std::pair<float,float>>& pairs) {
    // Find low[j] for each column j
    for (int col = 0; col < n; col++) {
        int low = -1;
        for (int row = n - 1; row >= 0; row--) {
            if (matrix[row * n + col] != 0.0f) {
                low = row;
                break;
            }
        }
        
        if (low != -1) {
            // simplex low died when simplex col was added
            // This is a persistence pair for dim = simplices[low].dim
            int birth_dim = simplices[low].dim;
            int death_dim = simplices[col].dim;
            if (birth_dim == target_dim && death_dim == target_dim + 1) {
                float birth = simplices[low].filtration;
                float death = simplices[col].filtration;
                if (death > birth) {
                    pairs.push_back({birth, death});
                }
            }
        }
    }
    
    // Essential features: columns with low = -1 and dim == target_dim
    for (int col = 0; col < n; col++) {
        int low = -1;
        for (int row = n - 1; row >= 0; row--) {
            if (matrix[row * n + col] != 0.0f) {
                low = row;
                break;
            }
        }
        if (low == -1 && simplices[col].dim == target_dim) {
            pairs.push_back({simplices[col].filtration, INFINITY});
        }
    }
}

struct SimplexWrapper {
    int verts[3];
    int dim;
    float filtration;
};

// Main H1 persistence function
std::vector<std::pair<float,float>> gpu_h1_persistence(
    const float* d_points, int N, int dim, cudaStream_t stream) {
    
    // For small N, build on host and reduce on GPU
    // For very large N, triangles are O(N^3) which won't fit, so limit to reasonable sizes
    
    // Step 1: Get distance matrix on host
    float* d_dist;
    cudaMalloc(&d_dist, N * N * sizeof(float));
    gpu_compute_distance_matrix(d_points, d_dist, N, dim, stream);
    
    float* h_dist = new float[N * N];
    cudaMemcpy(h_dist, d_dist, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_dist);
    
    // Step 2: Build simplices and boundary matrix on host
    // Reuse the build function from boundary_matrix.cu
    // We'll just do it inline here
    
    std::vector<SimplexWrapper> simplices;
    
    // Vertices
    for (int i = 0; i < N; i++) {
        SimplexWrapper s = {{i, -1, -1}, 0, 0.0f};
        simplices.push_back(s);
    }
    
    // Edges
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            SimplexWrapper s = {{i, j, -1}, 1, h_dist[i * N + j]};
            simplices.push_back(s);
        }
    }
    
    // Triangles
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            for (int k = j + 1; k < N; k++) {
                float max_d = fmaxf(fmaxf(h_dist[i*N+j], h_dist[i*N+k]), h_dist[j*N+k]);
                SimplexWrapper s = {{i, j, k}, 2, max_d};
                simplices.push_back(s);
            }
        }
    }
    
    std::sort(simplices.begin(), simplices.end(), [](const SimplexWrapper& a, const SimplexWrapper& b) {
        if (a.filtration != b.filtration) return a.filtration < b.filtration;
        return a.dim < b.dim;
    });
    
    int ns = simplices.size();
    
    // Build boundary matrix
    float* h_matrix = new float[ns * ns]();
    
    for (int col = 0; col < ns; col++) {
        auto& s = simplices[col];
        if (s.dim == 0) continue;
        if (s.dim == 1) {
            for (int row = 0; row < ns; row++) {
                if (simplices[row].dim == 0) {
                    if (simplices[row].verts[0] == s.verts[0] ||
                        simplices[row].verts[0] == s.verts[1]) {
                        h_matrix[row * ns + col] = 1.0f;
                    }
                }
            }
        }
        if (s.dim == 2) {
            int v0 = s.verts[0], v1 = s.verts[1], v2 = s.verts[2];
            int edges[3][2] = {{v0,v1}, {v0,v2}, {v1,v2}};
            for (int e = 0; e < 3; e++) {
                for (int row = 0; row < ns; row++) {
                    if (simplices[row].dim == 1 &&
                        simplices[row].verts[0] == edges[e][0] &&
                        simplices[row].verts[1] == edges[e][1]) {
                        h_matrix[row * ns + col] = 1.0f;
                    }
                }
            }
        }
    }
    
    // Upload matrix to GPU and reduce
    float* d_matrix;
    cudaMalloc(&d_matrix, ns * ns * sizeof(float));
    cudaMemcpy(d_matrix, h_matrix, ns * ns * sizeof(float), cudaMemcpyHostToDevice);
    
    reduce_columns_kernel<<<1, 1, 0, stream>>>(d_matrix, ns);
    cudaStreamSynchronize(stream);
    
    // Copy back
    cudaMemcpy(h_matrix, d_matrix, ns * ns * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Extract H1 persistence pairs
    std::vector<std::pair<float,float>> result;
    extract_persistence_pairs(h_matrix, ns, simplices, 1, result);
    
    // Cleanup
    delete[] h_dist;
    delete[] h_matrix;
    cudaFree(d_matrix);
    
    return result;
}
