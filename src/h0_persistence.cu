#include "persistent_homology.cuh"

// ============================================================
// H0 Persistence: compute connected component persistence
// ============================================================
// Standard union-find based approach:
// 1. Sort edges by distance (filtration)
// 2. Process edges in order; if endpoints in different components, merge
//    and record death time (edge distance) for the younger component
// 3. The last surviving component is the essential (infinite) feature

// Kernel to initialize parents
__global__ void init_parents_kernel(int* parents, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) parents[i] = i;
}

// Kernel to process edges for H0 persistence
// Each thread processes one edge, but we use atomics to serialize merges
// Actually, for correctness we process sequentially on a single thread block
// since the union-find order matters.

__global__ void h0_process_edges_kernel(
    const Edge* __restrict__ edges,
    int num_edges,
    int num_points,
    int* __restrict__ parents,
    int* __restrict__ birth_component,  // which component died [num_points]
    PDPoint* __restrict__ diagram,       // output persistence diagram
    int* __restrict__ num_features       // atomic counter
) {
    // We use a single block to process edges sequentially for correctness
    // (union-find order determines persistence pairs)
    if (blockIdx.x > 0) return;

    // Initialize
    for (int i = threadIdx.x; i < num_points; i += blockDim.x) {
        parents[i] = i;
        birth_component[i] = 0;
    }
    if (threadIdx.x == 0) *num_features = 0;
    __syncthreads();

    // Process edges sequentially (thread 0 does the work)
    if (threadIdx.x == 0) {
        for (int e = 0; e < num_edges; e++) {
            int a = edges[e].i;
            int b = edges[e].j;
            float dist = edges[e].dist;

            // Find roots
            int ra = a, rb = b;
            while (parents[ra] != ra) ra = parents[ra];
            while (parents[rb] != rb) rb = parents[rb];
            // Path compression
            int c;
            c = a; while (parents[c] != ra) { int n = parents[c]; parents[c] = ra; c = n; }
            c = b; while (parents[c] != rb) { int n = parents[c]; parents[c] = rb; c = n; }

            if (ra != rb) {
                // Merge: smaller root becomes child
                int child = (ra > rb) ? ra : rb;
                int root  = (ra > rb) ? rb : ra;
                parents[child] = root;

                // Record death: component 'child' dies at distance 'dist'
                int idx = (*num_features)++;
                diagram[idx].birth = 0.0f;  // H0 features are born at 0
                diagram[idx].death = dist;
            }
        }
    }
}

cudaError_t compute_h0_persistence(
    int num_points,
    thrust::device_vector<Edge>& d_edges,
    thrust::device_vector<PDPoint>& d_h0_diagram,
    thrust::device_vector<int>& d_parents
) {
    int num_edges = d_edges.size();

    // Sort edges by distance
    thrust::sort(d_edges.begin(), d_edges.end(),
        [] __device__ (const Edge& a, const Edge& b) {
            return a.dist < b.dist;
        });

    // Allocate working memory
    d_h0_diagram.resize(num_edges); // at most N-1 finite features
    d_parents.resize(num_points);
    thrust::device_vector<int> d_birth_component(num_points);
    thrust::device_vector<int> d_num_features(1, 0);

    // Initialize parents
    int bs = 256;
    int gs = (num_points + bs - 1) / bs;
    init_parents_kernel<<<gs, bs>>>(thrust::raw_pointer_cast(d_parents.data()), num_points);

    // Process edges
    h0_process_edges_kernel<<<1, 256>>>(
        thrust::raw_pointer_cast(d_edges.data()),
        num_edges,
        num_points,
        thrust::raw_pointer_cast(d_parents.data()),
        thrust::raw_pointer_cast(d_birth_component.data()),
        thrust::raw_pointer_cast(d_h0_diagram.data()),
        thrust::raw_pointer_cast(d_num_features.data())
    );

    cudaDeviceSynchronize();

    // Resize diagram to actual number of features
    int h_num_features = d_num_features[0];
    d_h0_diagram.resize(h_num_features);

    // Rebuild parents for H1 use (re-run union-find to get final state)
    thrust::device_vector<Edge> d_edges_copy = d_edges;
    thrust::sort(d_edges_copy.begin(), d_edges_copy.end(),
        [] __device__ (const Edge& a, const Edge& b) {
            return a.dist < b.dist;
        });

    init_parents_kernel<<<gs, bs>>>(thrust::raw_pointer_cast(d_parents.data()), num_points);

    // Process again to get final parent state
    thrust::device_vector<PDPoint> d_h0_tmp(num_points);
    thrust::device_vector<int> d_bc_tmp(num_points);
    thrust::device_vector<int> d_nf_tmp(1, 0);

    h0_process_edges_kernel<<<1, 256>>>(
        thrust::raw_pointer_cast(d_edges_copy.data()),
        num_edges,
        num_points,
        thrust::raw_pointer_cast(d_parents.data()),
        thrust::raw_pointer_cast(d_bc_tmp.data()),
        thrust::raw_pointer_cast(d_h0_tmp.data()),
        thrust::raw_pointer_cast(d_nf_tmp.data())
    );

    cudaDeviceSynchronize();

    return cudaGetLastError();
}
