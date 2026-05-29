# gpu-persistent-homology

**CUDA persistent homology — H⁰ and H¹ persistence diagrams, union-find on GPU, Wasserstein distance computation.**

Computes persistent homology of Vietoris-Rips complexes on the GPU. H⁰ (connected components) via parallel union-find, H¹ (loops) via boundary matrix reduction, and Wasserstein/bottleneck distances between persistence diagrams. All kernels optimized with shared memory and warp-level primitives.

## What This Gives You

- **H⁰ persistence** — GPU union-find for connected component tracking across filtration
- **H⁰ persistence** — Loop detection via boundary matrix operations
- **Wasserstein distance** — p-Wasserstein and bottleneck distances between diagrams
- **Distance matrix** — Tiled pairwise Euclidean distances
- **Test suite** — Correctness verification against CPU reference

## Quick Start

```cuda
#include "persistent_homology.cuh"

// Compute distance matrix
compute_distance_matrix_gpu(d_points, d_dist, N, D, stream);

// H⁰ persistence
compute_h0_persistence(d_dist, N, &births, &deaths, &n_pairs, stream);

// H⁰ persistence  
compute_h1_persistence(d_dist, N, &births, &deaths, &n_pairs, stream);

// Wasserstein distance between two diagrams
compute_wasserstein(d_pairs1, n1, d_pairs2, n2, p, &distance, stream);
```

## Build

```bash
nvcc -O3 -o test_correctness tests/test_correctness.cu src/*.cu
./test_correctness
```

## How It Fits

Part of the SuperInstance ecosystem:

- **[persistent-sheaf](https://github.com/SuperInstance/persistent-sheaf)** — Rust persistent sheaf cohomology
- **gpu-persistent-homology** — CUDA persistent homology (this repo)

## License

MIT
