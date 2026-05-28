# gpu-persistent-homology

CUDA-accelerated persistent homology computation for the RTX 4050 (compute capability 8.9).

## Features

- **Tiled distance matrix** — pairwise Euclidean distances with shared memory tiling
- **H0 persistence** — connected component analysis via GPU union-find with path compression
- **H1 persistence** — cycle detection in filtration order
- **Wasserstein-1 distance** — greedy matching approximation between persistence diagrams
- Scales to N=10,000+ points in arbitrary dimensions

## Build Requirements

- CUDA Toolkit 12.6+
- RTX 4050 (sm_89) or compatible GPU

## Build & Test

```bash
export PATH="/usr/local/cuda-12.6/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH"
make
```

## Architecture

| File | Purpose |
|------|---------|
| `include/persistent_homology.cuh` | Public API header |
| `src/distance_matrix.cu` | Tiled pairwise Euclidean distances |
| `src/union_find.cu` | GPU union-find with atomic path compression |
| `src/h0_persistence.cu` | H0 persistence diagram computation |
| `src/h1_persistence.cu` | H1 persistence (cycle detection) |
| `src/wasserstein.cu` | Wasserstein-1 distance approximation |

## License

MIT
