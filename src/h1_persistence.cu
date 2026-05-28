#include "persistent_homology.cuh"

// ============================================================
// H1 Persistence: detect loops
// ============================================================
// When processing edges in filtration order, if both endpoints
// are already in the same component, that edge creates a cycle (H1 feature).

// Global-memory version for larger point clouds
__global__ void h1_init_parents_kernel(int* parents, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) parents[i] = i;
}

__global__ void h1_process_edges_global_kernel(
    const Edge* __restrict__ edges,
    int num_edges,
    int num_points,
    int* __restrict__ parents,
    PDPoint* __restrict__ diagram,
    int* __restrict__ num_features
) {
    if (blockIdx.x > 0) return;
    if (threadIdx.x != 0) return;

    // Initialize parents
    for (int i = 0; i < num_points; i++) {
        parents[i] = i;
    }
    *num_features = 0;

    for (int e = 0; e < num_edges; e++) {
        int a = edges[e].i;
        int b = edges[e].j;
        float dist = edges[e].dist;

        // Find roots with path compression
        int ra = a;
        while (parents[ra] != ra) ra = parents[ra];
        int c = a;
        while (parents[c] != ra) { int n = parents[c]; parents[c] = ra; c = n; }

        int rb = b;
        while (parents[rb] != rb) rb = parents[rb];
        c = b;
        while (parents[c] != rb) { int n = parents[c]; parents[c] = rb; c = n; }

        if (ra != rb) {
            // Merge: smaller root becomes child
            int child = (ra > rb) ? ra : rb;
            int root  = (ra > rb) ? rb : ra;
            parents[child] = root;
        } else {
            // Same component -> creates a cycle (H1 feature)
            int idx = (*num_features);
            *num_features = idx + 1;
            diagram[idx].birth = dist;
            diagram[idx].death = 1e30f; // infinite persistence (no 2-simplices)
        }
    }
}

cudaError_t compute_h1_persistence(
    int num_points,
    thrust::device_vector<Edge>& d_edges,
    const thrust::device_vector<int>& /* d_parents */,
    thrust::device_vector<PDPoint>& d_h1_diagram
) {
    int num_edges = d_edges.size();

    // Sort edges by distance
    thrust::sort(d_edges.begin(), d_edges.end(),
        [] __device__ (const Edge& a, const Edge& b) {
            return a.dist < b.dist;
        });

    d_h1_diagram.resize(num_edges); // max possible cycles = num_edges - (N-1)
    thrust::device_vector<int> d_num_features(1, 0);

    // Use global memory for parents (works for any size)
    thrust::device_vector<int> d_parents_h1(num_points);

    h1_process_edges_global_kernel<<<1, 1>>>(
        thrust::raw_pointer_cast(d_edges.data()),
        num_edges,
        num_points,
        thrust::raw_pointer_cast(d_parents_h1.data()),
        thrust::raw_pointer_cast(d_h1_diagram.data()),
        thrust::raw_pointer_cast(d_num_features.data())
    );

    cudaDeviceSynchronize();

    int h_num = d_num_features[0];
    d_h1_diagram.resize(h_num);

    return cudaGetLastError();
}
