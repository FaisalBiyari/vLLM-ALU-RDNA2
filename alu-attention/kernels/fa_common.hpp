// fa_common.hpp — shared types, bf16 conversion, Philox dropout, and fp64 CPU oracle for gates.
// gfx1030 hardware path only: v_dot2_f32_f16 via __builtin_amdgcn_fdot2, exp2f, __shfl_xor(width=16).
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <type_traits>

#define HIP_CHECK(x) do{hipError_t e=(x);if(e!=hipSuccess){printf("HIP %s L%d\n",hipGetErrorString(e),__LINE__);exit(1);}}while(0)

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
typedef _Float16 half8_t __attribute__((ext_vector_type(8)));

// ---- cross-lane reduction across 16 lanes via DPP -----------------------
// __shfl_xor compiles to ds_bpermute_b32 — an LDS-pipeline operation with high latency
// (RGA showed 64 such instructions per KV tile). DPP does the same entirely on VALU.
// Masks: quad_perm{1,0,3,2}=0xB1 (xor1), quad_perm{2,3,0,1}=0x4E (xor2),
//        row_half_mirror=0x141 (within 8 lanes i->7-i, equivalent to xor7),
//        row_mirror=0x140 (within 16 lanes i->15-i, equivalent to xor15).
// {1,2,7,15} are independent over GF(2) on 4 bits => the sequence gives a complete 16-lane reduction.
template<int CTRL>
__device__ __forceinline__ float dpp_mov_f32(float x){
    union { float f; int i; } a, b;
    a.f = x;
    b.i = __builtin_amdgcn_update_dpp(0, a.i, CTRL, 0xF, 0xF, false);
    return b.f;
}
__device__ __forceinline__ float red16_max(float x){
    x = fmaxf(x, dpp_mov_f32<0xB1 >(x));
    x = fmaxf(x, dpp_mov_f32<0x4E >(x));
    x = fmaxf(x, dpp_mov_f32<0x141>(x));
    x = fmaxf(x, dpp_mov_f32<0x140>(x));
    return x;
}
__device__ __forceinline__ float red16_sum(float x){
    x += dpp_mov_f32<0xB1 >(x);
    x += dpp_mov_f32<0x4E >(x);
    x += dpp_mov_f32<0x141>(x);
    x += dpp_mov_f32<0x140>(x);
    return x;
}

// ---- dtype tags ----------------------------------------------------------
struct DT_F16 { typedef _Float16 T; static constexpr bool is_bf16=false, is_fp32=false; };
struct DT_BF16{ typedef __hip_bfloat16 T; static constexpr bool is_bf16=true,  is_fp32=false; };
struct DT_F32 { typedef float T;         static constexpr bool is_bf16=false, is_fp32=true;  };

// ---- bf16 <-> float (lossless): bf16 value = (bits<<16) interpreted as float ------
__device__ __forceinline__ float bf16_to_f(__hip_bfloat16 v){
    uint32_t b = ((uint32_t)*reinterpret_cast<const uint16_t*>(&v)) << 16;
    float f; __builtin_memcpy(&f,&b,4); return f;
}
__device__ __forceinline__ float f16_to_f(_Float16 v){ return (float)v; }

// pow2 tile scaling: bring max|value| to ~2^13 (inside fp16 normal range, ~2^27 headroom downward).
// Returns power-of-two S (exact float); values are multiplied by S before bf16->fp16 conversion.
__device__ __forceinline__ float pow2_scale_for(float maxabs){
    if(maxabs<=0.0f) return 1.0f;
    int e; frexpf(maxabs,&e);            // maxabs in [2^(e-1),2^e)
    int shift = 13 - e;                  // target e ~ 13
    if(shift> 60) shift= 60; if(shift<-60) shift=-60;
    return ldexpf(1.0f,shift);
}

// ---- Philox 4x32-10 (deterministic dropout) --------------------------
__device__ __forceinline__ uint2 philox(uint64_t seed, uint64_t offset){
    uint32_t k0=(uint32_t)seed, k1=(uint32_t)(seed>>32);
    uint32_t c0=(uint32_t)offset, c1=(uint32_t)(offset>>32), c2=0, c3=0;
    #pragma unroll
    for(int i=0;i<10;i++){
        uint32_t hi0=__umulhi(0xD2511F53u,c0), lo0=0xD2511F53u*c0;
        uint32_t hi1=__umulhi(0xCD9E8D57u,c2), lo1=0xCD9E8D57u*c2;
        uint32_t n0=hi1^c1^k0, n1=lo1, n2=hi0^c3^k1, n3=lo0;
        c0=n0;c1=n1;c2=n2;c3=n3; k0+=0x9E3779B9u; k1+=0xBB67AE85u;
    }
    return make_uint2(c0,c1);
}
// keep bit for linear index idx at drop probability p; u in [0,1).
__device__ __forceinline__ float dropout_mul(uint64_t seed,uint64_t idx,float p,float inv_keep){
    uint2 r=philox(seed,idx);
    float u=(r.x>>8)*(1.0f/16777216.0f);
    return (u>=p)?inv_keep:0.0f;
}

// ================= fp64 CPU oracle (shared by all gates) =================
// Reference specification: s=scale*sum(Q*K); softcap>0 -> softcap*tanh(s/softcap);
// alibi -> s+=slope*(j-i); causal(j>i+delta)/window mask -> -inf; softmax; O=sum(p*V); LSE=m+log(l).
struct RefCfg {
    int SQ, SK, HD, NH, gqa;
    float scale, softcap;
    int win_left, win_right;               // flash-attn semantics: <0 => no boundary
    bool causal, alibi_on;
    const float* alibi_slopes;             // [NH] or nullptr
    int attn_chunk;                        // 0 => off (FA-3 attention_chunk)
};
// Inputs are float values already rounded to dtype on the host. O_out[SQ*NH*HD], LSE_out[NH*SQ] optional.
static inline void ref_attention(const float* Q,const float* K,const float* V,
                                 float* O_out,float* LSE_out,const RefCfg& c){
    const int delta=c.SK-c.SQ;
    std::vector<double> pr(c.SK);
    for(int h=0;h<c.NH;h++){ int kvh=h/c.gqa;
        double slope = c.alibi_on ? c.alibi_slopes[h] : 0.0;
        for(int i=0;i<c.SQ;i++){
            double mx=-1e300; int cnt=0;
            for(int j=0;j<c.SK;j++){
                bool masked=false;
                if(c.causal && j> i+delta) masked=true;
                if(c.win_left >=0 && j < i+delta-c.win_left ) masked=true;
                if(c.win_right>=0 && j > i+delta+c.win_right) masked=true;
                if(c.attn_chunk>0){ int pp=i+delta; if(pp<0) pp=0;
                    int cs=(pp/c.attn_chunk)*c.attn_chunk;
                    if(j<cs || j>=cs+c.attn_chunk) masked=true; }
                if(masked){ pr[j]=-1e300; continue; }
                double s=0; for(int d=0;d<c.HD;d++)
                    s+=(double)Q[((size_t)i*c.NH+h)*c.HD+d]*(double)K[((size_t)kvh*c.SK+j)*c.HD+d];
                s*=c.scale;
                if(c.softcap>0) s=c.softcap*tanh(s/c.softcap);
                if(c.alibi_on) s+=slope*(double)(j-i);
                pr[j]=s; if(s>mx)mx=s; cnt++;
            }
            double sum=0; for(int j=0;j<c.SK;j++){ if(pr[j]<=-1e299){pr[j]=0;continue;} pr[j]=exp(pr[j]-mx); sum+=pr[j]; }
            for(int d=0;d<c.HD;d++){ double acc=0;
                for(int j=0;j<c.SK;j++) acc+=pr[j]*(double)V[((size_t)kvh*c.SK+j)*c.HD+d];
                O_out[((size_t)i*c.NH+h)*c.HD+d]=(float)(cnt? acc/sum : 0.0);
            }
            if(LSE_out) LSE_out[(size_t)h*c.SQ+i]=(float)(cnt? (mx+log(sum)) : -INFINITY);
        }
    }
}

static inline double rel_rms(const float* got,const float* ref,size_t n){
    double e=0,r=0; for(size_t i=0;i<n;i++){ double d=(double)got[i]-ref[i]; e+=d*d; r+=(double)ref[i]*ref[i]; }
    return r>0? sqrt(e/r): sqrt(e);
}
static inline uint32_t xrng(uint32_t& s){ s=s*1664525u+1013904223u; return s; }
static inline float frand(uint32_t& s){ return ((xrng(s)>>16)&4095)/4096.0f-0.5f; }
