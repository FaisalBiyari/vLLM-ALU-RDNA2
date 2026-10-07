// SPDX-License-Identifier: Apache-2.0
// Host-only scheduling math, shared with CPU regression tests. No GPU runtime dependency.
#pragma once
#include <cstdint>

inline int alu_decode_splits(int64_t rows, int kv_heads, int64_t context_bound,
                             int cu_count, int requested=0) {
    if (requested > 0) return requested;
    if (rows == 0) return 1;
    const int64_t groups = rows * kv_heads;
    const int64_t target = 4LL * (cu_count > 0 ? cu_count : 72);
    int64_t n = (target + groups - 1) / groups;
    int64_t maxn = (context_bound + 255) / 256;
    if (maxn < 1) maxn = 1;
    if (n > maxn) n = maxn;
    if (n < 1) n = 1;
    if (n > 512) n = 512;
    return static_cast<int>(n);
}
inline int64_t alu_decode_scratch_elements(int64_t rows, int heads, int dim, int splits) {
    return splits == 1 ? 0 : rows * heads * splits * (dim + 2LL);
}
