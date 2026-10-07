// Host/HIP ABI declarations. Struct layouts must match their kernel definitions.
// Stream handles cross this C-linkage boundary as opaque pointers.
#pragma once
#include <cstdint>

// --- forward (fa_forward.hip) ---
struct Strides { long long b, h, s; };
struct FwdParams {
    const void* Q; const void* K; const void* V; void* O; float* LSE;
    Strides sQ, sK, sV, sO;
    int SQ, SK, HD, NH, gqa;
    float scale, softcap;
    int win_left, win_right;   // flash-attn semantics: <0 means no bound on that side
    const float* alibi_slopes;
    int causal;
    float dropout_p; unsigned long long philox_seed;
    const int* cu_q; const int* cu_k;
    int batch;
    float sc_q, sc_k, sc_v;             // bf16 power-of-two scales; 0 means the kernel substitutes 1
    int attn_chunk;                     // FA-3 attention_chunk; 0 disables it
    const int* seqused_q; const int* seqused_k;   // FA-3 varlen; nullptr means full length
};

// --- backward (backward.hip) ---
struct Str { long long b, h, s; };
struct BwdParams {
    const void *Q,*K,*V,*O,*dO;  void *dQ,*dK,*dV;
    const float *lse; float *dsum;
    Str qs,ks,vs,os,dos,dqs,dks,dvs;
    int B,Hq,Hk,SQ,SK,D,gqa;
    float scale, softcap;
    const float *alibi;
    int causal, win_left, win_right;
    int dropout; unsigned long long drop_seed; float dropout_p, keep_scale;
    int shQ,shK,shV,shdO;
};

// --- decode with KV cache (fa_decode.hip) ---
// Paged-cache layout: canonical separate [num_blocks,page,nheads_k,HD] tensors or an
// interleaved [K | V] row described by kv_row/v_off.
#include "../../kernels/fa_decode_params.hpp"

extern "C" void fa_forward_launch(FwdParams p, int dtype, void* stream);
extern "C" void alu_attn_backward_launch(BwdParams pr, int MODE, void* stream);
extern "C" void fa_decode_launch(DecParams p, int dtype, void* stream,
                                 const void* new_k, const void* new_v);
