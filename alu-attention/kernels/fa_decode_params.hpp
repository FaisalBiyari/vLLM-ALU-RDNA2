// SPDX-License-Identifier: Apache-2.0
// Shared host/device decode ABI. Keep a single definition to prevent layout drift.
#pragma once

struct DecParams {
    const void* q;
    const void* Kc; const void* Vc;
    const int* block_table;
    void* o;
    float* scratch;              // [total_q*NH*nsplit*(HD+2)]; null for nsplit=1
    const int* ctx_len;
    const float* cos; const float* sin;
    int batch, NH, gqa, HD, page, max_blocks, nsplit, window;
    int page_shift, page_mask;   // set by the launcher
    int rotary;                  // 0 none, 1 NeoX, 2 interleaved
    float scale;
    float sc_q, sc_k, sc_v;
    const int* leftpad;          // cache_leftpad[batch]; nullptr => 0
    float* lse_out;              // return_softmax_lse: [batch*nq, NH]; nullptr means do not write
    float softcap;               // <=0 disables it
    const float* alibi;          // [NH] or nullptr
    int nq;                      // rectangular query count per request; ignored when cu_q is set
    int append;                  // set by the launcher based on new_k
    int rotary_dim;              // 0 means the full head_dim
    const int* rotary_pos;       // rotary_seqlens; nullptr means derive position from ctx_len
    int kv_row, v_off;            // row stride and V offset in elements; defaults are HD and 0
    // Packed continuous-batch ABI. Metadata stays on the device during graph replay.
    const int* cu_q;             // [batch+1] query offsets; null means rectangular [B,nq,H,D]
    int total_q;                // allocated query rows, including graph padding
    int causal;                 // bottom-right causal mask when nonzero
    int window_right;           // right window radius; -1 is unbounded
    int attention_chunk;        // 0 disables chunk masking
    long long k_block_stride, k_token_stride, k_head_stride;
    long long v_block_stride, v_token_stride, v_head_stride;
    int block_table_stride;     // row stride in int32 elements (may exceed max_blocks)
};
