#include "persistent_homology.cuh"
#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>

// ============================================================
// Wasserstein-1 distance (greedy matching approximation)
// ============================================================
// Uses a greedy algorithm: for each point in diag1, match to
// closest unmatched point in diag2. Unmatched points contribute
// their distance to the diagonal.

static float point_to_diag_dist(const PDPoint& p) {
    return fabsf(p.death - p.birth) * 0.5f;
}

float compute_wasserstein_distance(
    const thrust::host_vector<PDPoint>& diag1,
    const thrust::host_vector<PDPoint>& diag2
) {
    int n1 = diag1.size();
    int n2 = diag2.size();

    if (n1 == 0 && n2 == 0) return 0.0f;

    // Greedy matching
    std::vector<bool> matched2(n2, false);
    float total = 0.0f;

    // For each point in diag1, find closest unmatched in diag2
    for (int i = 0; i < n1; i++) {
        float best_dist = std::numeric_limits<float>::max();
        int best_j = -1;

        for (int j = 0; j < n2; j++) {
            if (matched2[j]) continue;
            float dx = diag1[i].birth - diag2[j].birth;
            float dy = diag1[i].death - diag2[j].death;
            float d = sqrtf(dx * dx + dy * dy);
            if (d < best_dist) {
                best_dist = d;
                best_j = j;
            }
        }

        // Compare matching cost vs sending to diagonal
        float diag_cost = point_to_diag_dist(diag1[i]);

        if (best_j >= 0 && best_dist < diag_cost + point_to_diag_dist(diag2[best_j])) {
            total += best_dist;
            matched2[best_j] = true;
        } else {
            total += diag_cost;
            // Don't match best_j, leave it for potential later matching
        }
    }

    // Add diagonal cost for unmatched points in diag2
    for (int j = 0; j < n2; j++) {
        if (!matched2[j]) {
            total += point_to_diag_dist(diag2[j]);
        }
    }

    return total;
}
