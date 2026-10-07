// SPDX-License-Identifier: Apache-2.0
// The same request mapping and mask bounds run on the GPU and in host-side tests.
#pragma once
#include "fa_decode_params.hpp"
#ifdef __HIPCC__
#define ALU_INDEX_HD __host__ __device__ __forceinline__
#else
#define ALU_INDEX_HD inline
#endif

struct AluQueryIndex { int batch, token, length; bool active; };
ALU_INDEX_HD AluQueryIndex alu_query_index(const DecParams& p, int row) {
    if (!p.cu_q) return {row / p.nq, row % p.nq, p.nq, true};
    if (row >= p.cu_q[p.batch]) return {0, 0, 0, false};
    // Upper bound skips empty requests (repeated offsets). No host readback or per-request launch.
    int lo=0, hi=p.batch;
    while (lo < hi) {
        const int mid=lo+(hi-lo)/2;
        if (p.cu_q[mid+1] <= row) lo=mid+1; else hi=mid;
    }
    return {lo, row-p.cu_q[lo], p.cu_q[lo+1]-p.cu_q[lo], true};
}
struct AluQueryBounds { int position, begin, end; };
ALU_INDEX_HD AluQueryBounds alu_query_bounds(const DecParams& p, AluQueryIndex q) {
    if (!q.active) return {0, 0, 0};
    const int cached=p.ctx_len[q.batch];
    const int full=cached+(p.append ? q.length : 0);
    const int pos=full-q.length+q.token;
    int begin=p.leftpad ? p.leftpad[q.batch] : 0;
    int end=full;
    if (p.window >= 0 && pos-p.window+1 > begin) begin=pos-p.window+1;
    if (p.attention_chunk > 0 && pos >= 0) {
        const int cbegin=(pos/p.attention_chunk)*p.attention_chunk;
        if (cbegin > begin) begin=cbegin;
        const int cend=cbegin+p.attention_chunk;
        if (cend < end) end=cend;
    }
    if (p.causal && pos+1 < end) end=pos+1;
    if (p.window_right >= 0 && pos+p.window_right+1 < end) end=pos+p.window_right+1;
    if (end < begin) end=begin;
    return {pos, begin, end};
}
#undef ALU_INDEX_HD
