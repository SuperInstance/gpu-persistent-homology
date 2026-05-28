#include "persistent_homology.cuh"

// ============================================================
// GPU Union-Find with path compression and atomic operations
// ============================================================

// Find root with path compression (executed on GPU in single-thread or small kernel context)
// We use a device-side function callable from kernels.

__device__ __forceinline__ int uf_find(int* parents, int x) {
    // Iterative find with path compression
    int root = x;
    while (parents[root] != root) {
        root = parents[root];
    }
    // Path compression
    int current = x;
    while (parents[current] != root) {
        int next = parents[current];
        parents[current] = root;
        current = next;
    }
    return root;
}

__device__ __forceinline__ int uf_find_volatile(volatile int* parents, int x) {
    int root = x;
    while (parents[root] != root) {
        root = parents[root];
    }
    int current = x;
    while (parents[current] != root) {
        int next = parents[current];
        parents[current] = root;
        current = next;
    }
    return root;
}

// Union by setting parent (no rank for simplicity; we union a -> b)
__device__ __forceinline__ void uf_union(int* parents, int a, int b) {
    // a's root becomes child of b's root
    parents[a] = b;
}
