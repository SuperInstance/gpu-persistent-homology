#include "persistent_homology.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

// Build boundary matrix from a Vietoris-Rips complex up to dimension 2
// (vertices, edges, triangles)
// Simplices are sorted by filtration value (distance for edges, max edge for triangles)

// Extract unique simplices and build boundary matrix on host, upload to device
// This is the setup step; reduction is separate.

// Structure: simplex represented by its vertices (sorted)
// For VR complex: edges (2-simplices) have filtration = edge distance
//                 triangles (3-simplices) have filtration = max of 3 edge distances

struct Simplex {
    int verts[3];
    int dim; // 0=vertex, 1=edge, 2=triangle
    float filtration;
};

// Host function to build boundary matrix for H1
// Returns pairs of (edge_index, triangle_index) = non-zero entries
// Boundary of triangle σ = [i,j,k] is edges [i,j], [i,k], [j,k] with signs

// For H1 persistence, we need:
// simplices sorted by filtration: all vertices (filtration=0), then edges (by distance), then triangles (by max edge)
// Boundary matrix B where B[i][j] = 1 if simplex i is on boundary of simplex j
// Then reduce B to compute persistence

// This function builds the boundary matrix on host for H1 (dim-1 simplices)
extern "C"
void build_h1_boundary_matrix_host(const float* h_dist, int N,
                                    float** h_matrix, int* n_simplices,
                                    std::vector<Simplex>* simplices_out) {
    std::vector<Simplex> simplices;
    
    // Vertices (dim 0)
    for (int i = 0; i < N; i++) {
        Simplex s;
        s.verts[0] = i;
        s.verts[1] = -1;
        s.verts[2] = -1;
        s.dim = 0;
        s.filtration = 0.0f;
        simplices.push_back(s);
    }
    
    // Edges (dim 1)
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            Simplex s;
            s.verts[0] = i;
            s.verts[1] = j;
            s.verts[2] = -1;
            s.dim = 1;
            s.filtration = h_dist[i * N + j];
            simplices.push_back(s);
        }
    }
    
    // Triangles (dim 2)
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            for (int k = j + 1; k < N; k++) {
                float d01 = h_dist[i * N + j];
                float d02 = h_dist[i * N + k];
                float d12 = h_dist[j * N + k];
                float max_d = fmaxf(fmaxf(d01, d02), d12);
                Simplex s;
                s.verts[0] = i;
                s.verts[1] = j;
                s.verts[2] = k;
                s.dim = 2;
                s.filtration = max_d;
                simplices.push_back(s);
            }
        }
    }
    
    // Sort by (filtration, dim) — filtration primary, dim secondary
    std::sort(simplices.begin(), simplices.end(), [](const Simplex& a, const Simplex& b) {
        if (a.filtration != b.filtration) return a.filtration < b.filtration;
        return a.dim < b.dim;
    });
    
    // Build index map: simplex -> column index
    // For edges: map (i,j) -> column
    // For triangles: map (i,j,k) -> column
    // We need to find which column corresponds to which edge/triangle
    
    *n_simplices = (int)simplices.size();
    
    // Build boundary matrix (dense, float: 0 or 1)
    // Only need columns for dim>=1 simplices (edges + triangles)
    // Rows for dim-1 simplices (vertices + edges)
    // Actually for full persistence: matrix is n_simplices x n_simplices
    // B[row][col] = 1 if simplex[row] is a face of simplex[col]
    
    *h_matrix = new float[(*n_simplices) * (*n_simplices)]();
    
    // Build lookup: vertex/edge -> index
    // vertex i -> index in sorted simplices
    // edge (i,j) -> index
    
    for (int col = 0; col < *n_simplices; col++) {
        const auto& s = simplices[col];
        if (s.dim == 0) continue; // vertices have no boundary
        if (s.dim == 1) {
            // Boundary of edge [i,j] = vertices [i] and [j]
            // Find row indices of vertices s.verts[0] and s.verts[1]
            for (int row = 0; row < *n_simplices; row++) {
                if (simplices[row].dim == 0) {
                    if (simplices[row].verts[0] == s.verts[0] ||
                        simplices[row].verts[0] == s.verts[1]) {
                        (*h_matrix)[row * (*n_simplices) + col] = 1.0f;
                    }
                }
            }
        }
        if (s.dim == 2) {
            // Boundary of triangle [i,j,k] = edges [i,j], [i,k], [j,k]
            int v0 = s.verts[0], v1 = s.verts[1], v2 = s.verts[2];
            // edges: (v0,v1), (v0,v2), (v1,v2)
            int edges[3][2] = {{v0,v1}, {v0,v2}, {v1,v2}};
            for (int e = 0; e < 3; e++) {
                for (int row = 0; row < *n_simplices; row++) {
                    if (simplices[row].dim == 1) {
                        int a = simplices[row].verts[0], b = simplices[row].verts[1];
                        if ((a == edges[e][0] && b == edges[e][1])) {
                            (*h_matrix)[row * (*n_simplices) + col] = 1.0f;
                        }
                    }
                }
            }
        }
    }
    
    if (simplices_out) *simplices_out = simplices;
}
