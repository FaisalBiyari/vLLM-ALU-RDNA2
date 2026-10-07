#include "../package/csrc/alu_compiler_policy.h" // ALU_COMPILER_HARDENING_POLICY
// backward.hip — FlashAttention-2 BACKWARD for gfx1030 (RX 6800 XT), dtype-generic (fp16/bf16/fp32).
// FA-2 scheme WITHOUT ATOMICS => determinism by construction (bit-for-bit reproducible; deterministic=True/False
// produce ONE result — a strict strengthening of the flash-attn contract):
//   1) preprocess:  D_i = sum_d O_id * dO_id          (fp32, shape [B,Hq,SQ])
//   2) dK/dV kernel: parallel over K tiles; each key tile sums over ALL queries and all
//                   query heads in the GQA group. dK,dV stay in registers. No atomics.
//   3) dQ kernel:    parallel over Q tiles; dQ stays in registers. No atomics.
// SxS is NEVER materialized — only a BQxBK tile in LDS; P is recomputed from saved LSE.
// Minimal VRAM: beyond dQ/dK/dV, only fp32 workspace D_i=[B,Hq,SQ] (tiny).
//
// dtype: fp16 and bf16 run at fp16 speed through v_dot2_f32_f16. bf16 uses a power-of-two scale
// per tensor (exponent shift, pow2_scale_for) => lossless bf16->fp16 => fdot2 => divide by 2^shift
// (exact). fp32 uses a separate fmaf path (~1e-6 accuracy; not a performance target, fdot2 is fp16 only).
//
// MATCHED TO FORWARD (fa_forward.hip): LSE in natural units [NH][SQ] per batch; dropout idx=((h*SQ+qq)*SK+kk)
// WITHOUT batch + fa_common::dropout_mul; mask uses the same window (allowed [qq+delta-window+1..qq+delta]),
// causal bottom-right delta=SK-SQ; softcap/alibi exactly match forward. P recomputation uses natural
// units => matches forward LSE bit-for-bit by formula.
//
// Build and run: bash tools/build_kernel_gates.sh
#include "fa_common.hpp"

struct Str { long long b, h, s; };      // strides in elements, dim contiguous

enum FaBwdDT { BW_F16 = 0, BW_BF16 = 1, BW_F32 = 2 };

template<int MODE> struct ElemT        { using type = _Float16; };
template<>         struct ElemT<BW_F32>{ using type = float; };

// exact 2^e for reasonable e
__host__ __device__ __forceinline__ float exp2i(int e) {
    union { uint32_t u; float f; } v; int be = e + 127;
    if (be <= 0)   return 0.0f;
    if (be >= 255) return __builtin_huge_valf();
    v.u = (uint32_t)be << 23; return v.f;
}
__device__ __forceinline__ float bf16bits_to_f(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; __builtin_memcpy(&f,&u,4); return f; }
__device__ __forceinline__ uint16_t f_to_bf16_rn(float f){ union{uint32_t u;float f;}c; c.f=f; uint32_t lsb=(c.u>>16)&1u; c.u+=0x7fffu+lsb; return (uint16_t)(c.u>>16); }

// read an element as an fp16 tile value (with bf16 scale scl); fp32 mode reads as float.
template<int MODE> __device__ __forceinline__ typename ElemT<MODE>::type
load_tile(const void* p, long long idx, float scl){
    if (MODE == BW_F16)  return ((const _Float16*)p)[idx];
    if (MODE == BW_BF16) return (_Float16)(bf16bits_to_f(((const uint16_t*)p)[idx]) * scl);
    return (typename ElemT<MODE>::type)(((const float*)p)[idx]);
}
// exact float read (for D_i, avoiding bf16 precision loss through fp16)
template<int MODE> __device__ __forceinline__ float load_f32(const void* p, long long idx){
    if (MODE == BW_F16)  return (float)((const _Float16*)p)[idx];
    if (MODE == BW_BF16) return bf16bits_to_f(((const uint16_t*)p)[idx]);
    return ((const float*)p)[idx];
}
template<int MODE> __device__ __forceinline__ void store_out(void* p, long long idx, float v){
    if (MODE == BW_F16)  { ((_Float16*)p)[idx] = (_Float16)v; return; }
    if (MODE == BW_BF16) { ((uint16_t*)p)[idx] = f_to_bf16_rn(v); return; }
    ((float*)p)[idx] = v;
}

struct BwdParams {
    const void *Q,*K,*V,*O,*dO;  void *dQ,*dK,*dV;
    const float *lse; float *dsum;
    Str qs,ks,vs,os,dos,dqs,dks,dvs;
    int B,Hq,Hk,SQ,SK,D,gqa;
    float scale, softcap;
    const float *alibi;          // [Hq] or nullptr
    // Window using flash-attn semantics: allowed keys [qq+delta-win_left .. qq+delta+win_right],
    // negative means no boundary. causal is equivalent to win_right==0 (kept as a separate field
    // for fast tile-level branches).
    int causal, win_left, win_right;
    int dropout; unsigned long long drop_seed; float dropout_p, keep_scale;
    int shQ,shK,shV,shdO;        // bf16 pow2 shifts (0 for fp16/fp32)
    int probe;                   // 1 = probe: skip second GEMM (result is invalid; measurement only)
};

// forward mask: masked if key>=SK || (causal&&key>qq+delta) || (window: outside [lo..qq+delta])
__device__ __forceinline__ bool fwd_ok(const BwdParams& pr, int qq, int kk){
    if (kk >= pr.SK || qq >= pr.SQ) return false;
    const int delta = pr.SK - pr.SQ, diag = qq + delta;
    if (pr.causal && kk > diag) return false;
    if (pr.win_left  >= 0 && kk < diag - pr.win_left ) return false;
    if (pr.win_right >= 0 && kk > diag + pr.win_right) return false;
    return true;
}

// ============================ preprocess: D_i ============================
template<int D, int MODE>
__global__ __launch_bounds__(256)
void fa_bwd_preprocess(BwdParams pr){
    const int b=blockIdx.z, h=blockIdx.y, qq=blockIdx.x*256+threadIdx.x;
    if (qq >= pr.SQ) return;
    const long long ob = (long long)b*pr.os.b + (long long)h*pr.os.h + (long long)qq*pr.os.s;
    const long long db = (long long)b*pr.dos.b + (long long)h*pr.dos.h + (long long)qq*pr.dos.s;
    float acc=0.f;
    for (int d=0; d<D; d++) acc += load_f32<MODE>(pr.O, ob+d) * load_f32<MODE>(pr.dO, db+d);
    pr.dsum[((long long)b*pr.Hq + h)*pr.SQ + qq] = acc;
}

// ---- compute P_tilde (=z*P) and dS in LDS for a BQxBK tile ----
// OUT is the output layout. The second GEMM reads LDS in pairs, and HOW data is laid out here
// determines whether it reads one dword or two u16 + v_perm (the latter cost backward 2.7 TFLOP/s):
//   OUT_PLAIN — original layout: Pt[i*BK+j], dSt[i*BK+j] (fp32 path, accuracy not performance).
//   OUT_PAIR_J — dQ kernel: dS2[i*(BK/2)+jh] = half2{dS[i][2jh],dS[i][2jh+1]}. P is not needed.
//   OUT_PAIR_I — dK/dV kernel: P2[j*(BQ/2)+ih] = half2{P[2ih][j],P[2ih+1][j]} (same for dS).
// Packing is free here: each thread owns consecutive i (QPT values) and j (JPT values).
enum { OUT_PLAIN=0, OUT_PAIR_J=1, OUT_PAIR_I=2 };

// LP is LDS row padding (in half elements). A row of length D maps all 16 lanes to one bank
// (D/2 is divisible by 32 for D=64/128/256) => 16-way conflict on K/V reads. LP breaks it.
template<int D,int MODE,int BQ,int BK,int OUT,int LP,int UN,typename E>
__device__ __forceinline__ void fill_pd(
    const E* Qf,const E* Kf,const E* Vf,const E* dOf,const float* lse,const float* di,
    E* Pt,E* dSt,const BwdParams& pr,int h,int qbase,int kbase,
    float invQK,float invdOV,float alibi_slope)
{
    constexpr int LD=D+LP;
    const int QPT=BQ/16, JPT=BK/16;
    const int t=threadIdx.x, ty=t>>4, tx=t&15;
    const float sc=pr.softcap;
    // register fdot2 tiling for S=Q*K^T and dP=dO*V^T: operands load once per dh and are reused
    // QPT/JPT times (row-major tiles provide contiguous half2 values over d).
    float Sa[QPT][JPT], Pa[QPT][JPT];
    #pragma unroll
    for (int a=0;a<QPT;a++)
        #pragma unroll
        for (int bb=0;bb<JPT;bb++){ Sa[a][bb]=0.f; Pa[a][bb]=0.f; }
    if constexpr (MODE==BW_F32){
        #pragma unroll 2
        for (int d=0; d<D; d++){
            float qv[QPT], ov[QPT], kv[JPT], vv[JPT];
            #pragma unroll
            for (int a=0;a<QPT;a++){ qv[a]=(float)Qf[(ty*QPT+a)*LD+d]; ov[a]=(float)dOf[(ty*QPT+a)*LD+d]; }
            #pragma unroll
            for (int bb=0;bb<JPT;bb++){ kv[bb]=(float)Kf[(tx*JPT+bb)*LD+d]; vv[bb]=(float)Vf[(tx*JPT+bb)*LD+d]; }
            #pragma unroll
            for (int a=0;a<QPT;a++)
                #pragma unroll
                for (int bb=0;bb<JPT;bb++){ Sa[a][bb]=fmaf(qv[a],kv[bb],Sa[a][bb]); Pa[a][bb]=fmaf(ov[a],vv[bb],Pa[a][bb]); }
        }
    } else {
        // Manually unroll UN steps: `#pragma unroll UN` does not accept a template parameter.
        // UN controls how many LDS reads are in flight before the first fdot2 — this is the ILP.
        #pragma unroll 1
        for (int dh0=0; dh0<D/2; dh0+=UN){
            #pragma unroll
            for (int uu=0; uu<UN; uu++){
                const int dh=dh0+uu;
                half2_t qv[QPT], ov[QPT], kv[JPT], vv[JPT];
                #pragma unroll
                for (int a=0;a<QPT;a++){ qv[a]=((const half2_t*)(Qf+(ty*QPT+a)*LD))[dh]; ov[a]=((const half2_t*)(dOf+(ty*QPT+a)*LD))[dh]; }
                #pragma unroll
                for (int bb=0;bb<JPT;bb++){ kv[bb]=((const half2_t*)(Kf+(tx*JPT+bb)*LD))[dh]; vv[bb]=((const half2_t*)(Vf+(tx*JPT+bb)*LD))[dh]; }
                #pragma unroll
                for (int a=0;a<QPT;a++)
                    #pragma unroll
                    for (int bb=0;bb<JPT;bb++){ Sa[a][bb]=__builtin_amdgcn_fdot2(qv[a],kv[bb],Sa[a][bb],false); Pa[a][bb]=__builtin_amdgcn_fdot2(ov[a],vv[bb],Pa[a][bb],false); }
            }
        }
    }
    E Pv[QPT][JPT], Dv[QPT][JPT];
    #pragma unroll
    for (int a=0; a<QPT; a++){
        const int i=ty*QPT+a, qq=qbase+i; const float LSEi=lse[i], Di=di[i];
        #pragma unroll
        for (int bb=0; bb<JPT; bb++){
            const int j=tx*JPT+bb, kk=kbase+j;
            float s=Sa[a][bb]*invQK, dp=Pa[a][bb]*invdOV;
            float g0=pr.scale*s, deriv=1.f, logit;
            if (sc>0.f){ float tnh=tanhf(g0/sc); logit=sc*tnh; deriv=1.f-tnh*tnh; } else logit=g0;
            logit += alibi_slope*(float)(kk-qq);
            float p = fwd_ok(pr,qq,kk) ? __expf(logit-LSEi) : 0.f;
            float z = 1.f;
            if (pr.dropout){ unsigned long long idx=(((unsigned long long)h*pr.SQ+qq)*pr.SK+kk);
                z = dropout_mul(pr.drop_seed, idx, pr.dropout_p, pr.keep_scale); }
            Pv[a][bb] = (E)(z*p);
            Dv[a][bb] = (E)(p*(z*dp - Di)*deriv);
        }
    }
    if constexpr (OUT==OUT_PLAIN){
        #pragma unroll
        for (int a=0;a<QPT;a++){ const int i=ty*QPT+a;
            #pragma unroll
            for (int bb=0;bb<JPT;bb++){ const int j=tx*JPT+bb;
                Pt[i*BK+j]=Pv[a][bb]; dSt[i*BK+j]=Dv[a][bb]; } }
    } else if constexpr (OUT==OUT_PAIR_J){
        // dS2[i][jh]; do not write P — the dQ kernel does not need it
        half2_t* dS2=(half2_t*)dSt;
        #pragma unroll
        for (int a=0;a<QPT;a++){ const int i=ty*QPT+a;
            #pragma unroll
            for (int bb=0;bb<JPT;bb+=2){ const int jh=(tx*JPT+bb)>>1;
                dS2[i*(BK/2)+jh]=half2_t{Dv[a][bb],Dv[a][bb+1]}; } }
    } else {
        // P2/dS2[j][ih] — k-major, paired over i: second GEMM reads a ready dword
        half2_t* P2=(half2_t*)Pt; half2_t* D2=(half2_t*)dSt;
        #pragma unroll
        for (int bb=0;bb<JPT;bb++){ const int j=tx*JPT+bb;
            #pragma unroll
            for (int a=0;a<QPT;a+=2){ const int ih=(ty*QPT+a)>>1;
                P2[j*(BQ/2)+ih]=half2_t{Pv[a][bb],Pv[a+1][bb]};
                D2[j*(BQ/2)+ih]=half2_t{Dv[a][bb],Dv[a+1][bb]}; } }
    }
}

// ============================ dK/dV kernel ============================
template<int D,int MODE,int BQ,int BK,int OUT,int LP,int UN,int DINT>
__global__ __launch_bounds__(256)
void fa_bwd_dkdv(BwdParams pr){
    using E=typename ElemT<MODE>::type;
    constexpr int LD=D+LP;
    const int KPT=BK/16, EPT=D/16;
    const int t=threadIdx.x, ty=t>>4, tx=t&15;
    const int kt=blockIdx.x, kvh=blockIdx.y, b=blockIdx.z, kbase=kt*BK;
    const float invQ=exp2i(-pr.shQ), invdO=exp2i(-pr.shdO), invdOV=exp2i(-(pr.shdO+pr.shV));
    const float invQK=exp2i(-(pr.shQ+pr.shK));

    __shared__ __align__(16) E Kf[BK*LD]; __shared__ __align__(16) E Vf[BK*LD];
    __shared__ __align__(16) E Qf[BQ*LD]; __shared__ __align__(16) E dOf[BQ*LD];
    __shared__ __align__(16) E Pt[BQ*BK]; __shared__ __align__(16) E dSt[BQ*BK];
    __shared__ float lse[BQ]; __shared__ float di[BQ];

    float accK[BK/16][D/16], accV[BK/16][D/16];   // D/16, not 8: EPT grows when D>128
    #pragma unroll
    for (int r=0;r<KPT;r++)
        #pragma unroll
        for (int e=0;e<EPT;e++){ accK[r][e]=0.f; accV[r][e]=0.f; }

    for (int hg=0; hg<pr.gqa; hg++){
        const int h=kvh*pr.gqa+hg; const float aslope = pr.alibi? pr.alibi[h] : 0.f;
        const long long kb0=(long long)b*pr.ks.b+(long long)kvh*pr.ks.h;
        const long long vb0=(long long)b*pr.vs.b+(long long)kvh*pr.vs.h;
        for (int e=t; e<BK*D; e+=256){ int j=e/D, d=e%D, sk=kbase+j;
            Kf[j*LD+d]=(sk<pr.SK)?load_tile<MODE>(pr.K,kb0+(long long)sk*pr.ks.s+d,exp2i(pr.shK)):(E)0;
            Vf[j*LD+d]=(sk<pr.SK)?load_tile<MODE>(pr.V,vb0+(long long)sk*pr.vs.s+d,exp2i(pr.shV)):(E)0; }

        // For key kk, allowed queries are [kk-delta-win_right .. kk-delta+win_left].
        // Take the union across all keys in the tile: start from the first key and end at the last.
        const int delta=pr.SK-pr.SQ; int qstart=0, qend=pr.SQ;
        if (pr.causal)          qstart=max(0, kbase-delta);
        if (pr.win_right>=0)    qstart=max(qstart, max(0, kbase-delta-pr.win_right));
        if (pr.win_left>=0)     qend  =min(pr.SQ, kbase+BK-1-delta+pr.win_left+1);
        qstart=(qstart/BQ)*BQ;

        for (int qb=qstart; qb<qend; qb+=BQ){
            const long long qb0=(long long)b*pr.qs.b+(long long)h*pr.qs.h;
            const long long ob0=(long long)b*pr.dos.b+(long long)h*pr.dos.h;
            __syncthreads();
            for (int e=t; e<BQ*D; e+=256){ int i=e/D, d=e%D, sq=qb+i;
                Qf[i*LD+d] =(sq<pr.SQ)?load_tile<MODE>(pr.Q, qb0+(long long)sq*pr.qs.s+d, exp2i(pr.shQ)):(E)0;
                dOf[i*LD+d]=(sq<pr.SQ)?load_tile<MODE>(pr.dO,ob0+(long long)sq*pr.dos.s+d,exp2i(pr.shdO)):(E)0; }
            for (int i=t; i<BQ; i+=256){ int sq=qb+i; long long li=(((long long)b*pr.Hq+h)*pr.SQ+(sq<pr.SQ?sq:0));
                lse[i]=(sq<pr.SQ)?pr.lse[li]:1e30f; di[i]=(sq<pr.SQ)?pr.dsum[li]:0.f; }
            __syncthreads();
            fill_pd<D,MODE,BQ,BK,OUT,LP,UN,E>(
                Qf,Kf,Vf,dOf,lse,di,Pt,dSt,pr,h,qb,kbase,invQK,invdOV,aslope);
            __syncthreads();
            // dV[k][d]+=sum_i P_tilde[i][k]*dO[i][d]; dK[k][d]+=sum_i dS[i][k]*Q[i][d] (contract over i)
            if constexpr (MODE==BW_F32 || OUT==OUT_PLAIN){
                #pragma unroll
                for (int r=0;r<KPT;r++){ const int k=ty*KPT+r;
                    #pragma unroll
                    for (int e=0;e<EPT;e++){ const int d=tx*EPT+e; float av=0.f, ak=0.f;
                        for (int i=0;i<BQ;i++){ av+=(float)Pt[i*BK+k]*(float)dOf[i*LD+d]; ak+=(float)dSt[i*BK+k]*(float)Qf[i*LD+d]; }
                        accV[r][e]+=av; accK[r][e]+=ak; } }
            } else if (!pr.probe) {
                const half2_t* P2=(const half2_t*)Pt; const half2_t* D2=(const half2_t*)dSt;
                #pragma unroll 4
                for (int ih=0; ih<BQ/2; ih++){
                    half2_t pv[KPT], dv[KPT], ov[D/16], qv[D/16];
                    #pragma unroll
                    for (int r=0;r<KPT;r++){ const int k=ty*KPT+r;
                        pv[r]=P2[k*(BQ/2)+ih];      // ready dword instead of two u16 + v_perm
                        dv[r]=D2[k*(BQ/2)+ih]; }
                    #pragma unroll
                    for (int e=0;e<EPT;e++){ const int d=DINT? (e*16+tx) : (tx*EPT+e);
                        ov[e]=half2_t{dOf[(2*ih)*LD+d], dOf[(2*ih+1)*LD+d]};
                        qv[e]=half2_t{Qf[(2*ih)*LD+d],  Qf[(2*ih+1)*LD+d]}; }
                    #pragma unroll
                    for (int r=0;r<KPT;r++)
                        #pragma unroll
                        for (int e=0;e<EPT;e++){ accV[r][e]=__builtin_amdgcn_fdot2(pv[r],ov[e],accV[r][e],false);
                                                 accK[r][e]=__builtin_amdgcn_fdot2(dv[r],qv[e],accK[r][e],false); }
                }
            }
        }
    }
    const long long dk0=(long long)b*pr.dks.b+(long long)kvh*pr.dks.h;
    const long long dv0=(long long)b*pr.dvs.b+(long long)kvh*pr.dvs.h;
    #pragma unroll
    for (int r=0;r<KPT;r++){ const int k=ty*KPT+r, sk=kbase+k; if(sk>=pr.SK) continue;
        #pragma unroll
        for (int e=0;e<EPT;e++){ const int d=DINT? (e*16+tx) : (tx*EPT+e);
            store_out<MODE>(pr.dV, dv0+(long long)sk*pr.dvs.s+d, accV[r][e]*invdO);
            store_out<MODE>(pr.dK, dk0+(long long)sk*pr.dks.s+d, accK[r][e]*invQ*pr.scale); }
    }
}

// ============================ dQ kernel ============================
template<int D,int MODE,int BQ,int BK,int OUT,int LP,int UN,int DINT>
__global__ __launch_bounds__(256)
void fa_bwd_dq(BwdParams pr){
    using E=typename ElemT<MODE>::type;
    constexpr int LD=D+LP;
    const int QPT=BQ/16, EPT=D/16;
    const int t=threadIdx.x, ty=t>>4, tx=t&15;
    const int qt=blockIdx.x, h=blockIdx.y, b=blockIdx.z, kvh=h/pr.gqa, qbase=qt*BQ;
    const float invK=exp2i(-pr.shK), invdOV=exp2i(-(pr.shdO+pr.shV)), invQK=exp2i(-(pr.shQ+pr.shK));
    const float aslope = pr.alibi? pr.alibi[h] : 0.f;

    // __align__(16): buffers are reused as half2 (packed pairs), requiring dword alignment
    __shared__ __align__(16) E Qf[BQ*LD]; __shared__ __align__(16) E dOf[BQ*LD];
    __shared__ __align__(16) E Kf[BK*LD]; __shared__ __align__(16) E Vf[BK*LD];
    __shared__ __align__(16) E Pt[BQ*BK]; __shared__ __align__(16) E dSt[BQ*BK];
    __shared__ float lse[BQ]; __shared__ float di[BQ];

    float accQ[BQ/16][D/16];                      // D/16, not 8: EPT grows when D>128
    #pragma unroll
    for (int a=0;a<QPT;a++)
        #pragma unroll
        for (int e=0;e<EPT;e++) accQ[a][e]=0.f;

    const long long qb0=(long long)b*pr.qs.b+(long long)h*pr.qs.h;
    const long long ob0=(long long)b*pr.dos.b+(long long)h*pr.dos.h;
    for (int e=t; e<BQ*D; e+=256){ int i=e/D, d=e%D, sq=qbase+i;
        Qf[i*LD+d] =(sq<pr.SQ)?load_tile<MODE>(pr.Q, qb0+(long long)sq*pr.qs.s+d, exp2i(pr.shQ)):(E)0;
        dOf[i*LD+d]=(sq<pr.SQ)?load_tile<MODE>(pr.dO,ob0+(long long)sq*pr.dos.s+d,exp2i(pr.shdO)):(E)0; }
    for (int i=t; i<BQ; i+=256){ int sq=qbase+i; long long li=(((long long)b*pr.Hq+h)*pr.SQ+(sq<pr.SQ?sq:0));
        lse[i]=(sq<pr.SQ)?pr.lse[li]:1e30f; di[i]=(sq<pr.SQ)?pr.dsum[li]:0.f; }

    // For query qq, allowed keys are [qq+delta-win_left .. qq+delta+win_right].
    const int delta=pr.SK-pr.SQ; int kstart=0, kend=pr.SK;
    if (pr.causal)       kend  =min(pr.SK, qbase+BQ-1+delta+1);
    if (pr.win_right>=0) kend  =min(kend,  min(pr.SK, qbase+BQ-1+delta+pr.win_right+1));
    if (pr.win_left>=0)  kstart=max(0, qbase+delta-pr.win_left);
    kstart=(kstart/BK)*BK;

    const long long kb0=(long long)b*pr.ks.b+(long long)kvh*pr.ks.h;
    const long long vb0=(long long)b*pr.vs.b+(long long)kvh*pr.vs.h;
    for (int kb=kstart; kb<kend; kb+=BK){
        __syncthreads();
        for (int e=t; e<BK*D; e+=256){ int j=e/D, d=e%D, sk=kb+j;
            Kf[j*LD+d]=(sk<pr.SK)?load_tile<MODE>(pr.K,kb0+(long long)sk*pr.ks.s+d,exp2i(pr.shK)):(E)0;
            Vf[j*LD+d]=(sk<pr.SK)?load_tile<MODE>(pr.V,vb0+(long long)sk*pr.vs.s+d,exp2i(pr.shV)):(E)0; }
        __syncthreads();
        if constexpr (MODE==BW_F32 || OUT==OUT_PLAIN){
            fill_pd<D,MODE,BQ,BK,OUT_PLAIN,LP,UN,E>(Qf,Kf,Vf,dOf,lse,di,Pt,dSt,pr,h,qbase,kb,invQK,invdOV,aslope);
            __syncthreads();
            #pragma unroll
            for (int a=0;a<QPT;a++){ const int q=ty*QPT+a;
                #pragma unroll
                for (int e=0;e<EPT;e++){ const int d=tx*EPT+e; float aq=0.f;
                    for(int j=0;j<BK;j++) aq+=(float)dSt[q*BK+j]*(float)Kf[j*LD+d]; accQ[a][e]+=aq; } }
        } else {
            fill_pd<D,MODE,BQ,BK,OUT_PAIR_J,LP,UN,E>(Qf,Kf,Vf,dOf,lse,di,Pt,dSt,pr,h,qbase,kb,invQK,invdOV,aslope);
            // K paired over j: KT[jh][d] = half2{K[2jh][d], K[2jh+1][d]}. Store it in Vf memory —
            // after fill_pd it is dead, and the next iteration reloads it from global memory anyway.
            // The transpose costs (BK/2)*D writes per tile, while GEMM previously repeated the same packing 16 times
            // (once per ty), making this exactly 16x cheaper.
            half2_t* KT=(half2_t*)Vf;
            __syncthreads();
            for (int e=t; e<(BK/2)*D; e+=256){ int jh=e/D, d=e%D;
                KT[jh*D+d]=half2_t{Kf[(2*jh)*LD+d], Kf[(2*jh+1)*LD+d]}; }
            const half2_t* dS2=(const half2_t*)dSt;
            __syncthreads();
            // dQ[q][d]+=sum_j dS[q][j]*K[j][d] — both operands are already ready dwords
            if (pr.probe) continue;
            #pragma unroll 4
            for (int jh=0; jh<BK/2; jh++){
                half2_t sv[QPT], kv[D/16];
                #pragma unroll
                for (int a=0;a<QPT;a++) sv[a]=dS2[(ty*QPT+a)*(BK/2)+jh];
                #pragma unroll
                for (int e=0;e<EPT;e++) kv[e]=KT[jh*D+(DINT? (e*16+tx) : (tx*EPT+e))];
                #pragma unroll
                for (int a=0;a<QPT;a++)
                    #pragma unroll
                    for (int e=0;e<EPT;e++) accQ[a][e]=__builtin_amdgcn_fdot2(sv[a],kv[e],accQ[a][e],false);
            }
        }
    }
    const long long dq0=(long long)b*pr.dqs.b+(long long)h*pr.dqs.h;
    #pragma unroll
    for (int a=0;a<QPT;a++){ const int q=ty*QPT+a, sq=qbase+q; if(sq>=pr.SQ) continue;
        #pragma unroll
        for (int e=0;e<EPT;e++){ const int d=DINT? (e*16+tx) : (tx*EPT+e);
            store_out<MODE>(pr.dQ, dq0+(long long)sq*pr.dqs.s+d, accQ[a][e]*invK*pr.scale); }
    }
}

// ============================ launcher ============================
// stage: 0 = all three (production path), 1 = preprocess only, 2 = dK/dV only, 3 = dQ only.
// Used to measure stages independently: total time alone does not reveal which stage dominates.
static int g_bwd_stage = 0;
// Ablation INSIDE the kernel: pr.probe=1 skips the second GEMM (only fill_pd remains).
// The result is invalid — this is a probe for measuring contribution, not an operating mode.
static int g_bwd_probe = 0;

// Layout/tile variant is selected at runtime — the test bench sweeps the entire cube and measures each one.
// 0 = production configuration (BQ64 BK32, paired, padding 0). Change only based on measurements.
static int g_bwd_variant = 0;

template<int D,int MODE,int BQ,int BK,int OUT,int LP,int UN,int DINT>
static void launch_mode(BwdParams pr, hipStream_t st){
    dim3 pblk(256), pgrid((pr.SQ+255)/256, pr.Hq, pr.B);
    dim3 gK((pr.SK+BK-1)/BK, pr.Hk, pr.B), gQ((pr.SQ+BQ-1)/BQ, pr.Hq, pr.B), blk(256);
    const int s = g_bwd_stage;
    if (s==0 || s==1) hipLaunchKernelGGL((fa_bwd_preprocess<D,MODE>), pgrid, pblk, 0, st, pr);
    if (s==0 || s==2) hipLaunchKernelGGL((fa_bwd_dkdv<D,MODE,BQ,BK,OUT,LP,UN,DINT>), gK, blk, 0, st, pr);
    if (s==0 || s==3) hipLaunchKernelGGL((fa_bwd_dq<D,MODE,BQ,BK,OUT,LP,UN,DINT>),   gQ, blk, 0, st, pr);
}

// LDS for one variant (bytes): K,V,Q,dO over (D+LP) + P,dS + lse,di
static constexpr size_t bwd_lds(int D,int BQ,int BK,int LP,int el){
    return (size_t)el*( (size_t)2*BK*(D+LP) + (size_t)2*BQ*(D+LP) + (size_t)2*BQ*BK ) + 8*(size_t)BQ;
}

// A variant is viable if it fits in LDS and the paired layout is compatible with the tile.
template<int D,int MODE,int BQ,int BK,int OUT,int LP>
static constexpr bool cfg_ok(){
    constexpr int el = (MODE==BW_F32)?4:2;
    return bwd_lds(D,BQ,BK,LP,el) <= 65536
        && (OUT!=OUT_PAIR_I || (BQ/16)%2==0)   // pair over i within a thread
        && (OUT!=OUT_PAIR_J || (BK/16)%2==0);  // pair over j within a thread
}

// probe=true -> only report viability; probe=false -> launch.
template<int D,int MODE,int BQ,int BK,int LP,bool PAIR,int UN,int DINT>
static bool run_cfg(BwdParams& pr, hipStream_t st, bool probe){
    constexpr int OUT = PAIR ? OUT_PAIR_I : OUT_PLAIN;
    if constexpr (cfg_ok<D,MODE,BQ,BK,OUT,LP>() && (!PAIR || (BK/16)%2==0)){
        if (!probe) launch_mode<D,MODE,BQ,BK,OUT,LP,UN,DINT>(pr,st);
        return true;
    } else { (void)pr; (void)st; return false; }
}

// Test-bench variant list. Remove a row to revert a variant; production is always #0.
#define BWD_VARIANT_LIST(X) \
    X(0, 64,32, 6, true , 2, 0, "LP6 UN2 (v boyu)") \
    X(1, 64,32, 6, true , 2, 1, "LP6 UN2 + d bez konflikta") \
    X(2, 64,32, 6, true , 4, 1, "LP6 UN4 + d bez konflikta") \
    X(3, 64,32, 6, true , 1, 1, "LP6 UN1 + d bez konflikta") \
    X(4, 64,32, 2, true , 2, 1, "LP2 UN2 + d bez konflikta") \
    X(5, 64,32, 0, true , 2, 1, "LP0 UN2 + d bez konflikta") \
    X(6, 32,32, 6, true , 2, 1, "BQ32 LP6 + d") \
    X(7, 32,64, 6, true , 2, 1, "BQ32 BK64 LP6 + d") \
    X(8, 64,32, 0, true , 4, 0, "LP0 UN4 (do paddinga)") \
    X(9, 64,32, 0, false, 4, 0, "LP0 plain UN4 (ishodnoe)")

template<int D,int MODE>
static bool launch_variant(BwdParams& pr, hipStream_t st, int v, bool probe){
    switch(v){
        #define X(id,bq,bk,lp,pair,un,di,nm) case id: return run_cfg<D,MODE,bq,bk,lp,pair,un,di>(pr,st,probe);
        BWD_VARIANT_LIST(X)
        #undef X
        default: return false;
    }
}
static const char* bwd_variant_name(int v){
    switch(v){
        #define X(id,bq,bk,lp,pair,un,di,nm) case id: return nm;
        BWD_VARIANT_LIST(X)
        #undef X
        default: return "?";
    }
}
static constexpr int BWD_NVARIANT = 10;

// Fallback configs with progressively smaller tiles — for large D the production (BQ=64,BK=32,LP=6) does not fit
// in 64 KB LDS. Previously NO KERNEL LAUNCHED in this case (launch_variant returned false
// for the production variant too), while dQ/dK/dV came from empty_like, so uninitialized
// garbage escaped as gradients. Silently. Now select the first config that fits,
// and fail loudly if none fit.
// LDS is computed as: (Kf+Vf)=2*BK*(D+LP) + (Qf+dOf)=2*BQ*(D+LP) + (Pt+dSt)=2*BQ*BK elements.
// PAIR requires (BK/16)%2==0, so it is disabled when BK=16.
template<int D,int MODE>
static void launch_fallback(BwdParams& pr, hipStream_t st){
    // BQ and BK cannot be below 16: tiling maps onto 16 lanes (QPT=BQ/16, JPT=BK/16),
    // and smaller sizes create zero-length arrays and fail compilation.
    if (run_cfg<D,MODE,64,32,6,true ,2,0>(pr,st,false)) return;
    if (run_cfg<D,MODE,32,32,6,true ,2,0>(pr,st,false)) return;
    if (run_cfg<D,MODE,32,16,6,false,2,0>(pr,st,false)) return;
    if (run_cfg<D,MODE,16,16,6,false,2,0>(pr,st,false)) return;
    if (run_cfg<D,MODE,16,16,0,false,1,0>(pr,st,false)) return;
    printf("alu_attn backward: head_dim=%d does not fit in LDS under any layout\n", D);
    abort();
}
template<int D>
static void launch_D(BwdParams pr, int MODE, hipStream_t st){
    const int v = g_bwd_variant;
    if (MODE==BW_F16)       { if(!launch_variant<D,BW_F16 >(pr,st,v,false)) launch_fallback<D,BW_F16 >(pr,st); }
    else if (MODE==BW_BF16) { if(!launch_variant<D,BW_BF16>(pr,st,v,false)) launch_fallback<D,BW_BF16>(pr,st); }
    else {
        // fp32 uses four times as many bytes per element, so at D=256 even the
        // minimum 16x16 tile does not fit. This is not a parity gap: flash-attn has no fp32 path at all; ours is a
        // gradcheck bonus. The wrapper catches this case in advance and reports it clearly.
        if (!run_cfg<D,BW_F32,32,16,0,false,4,0>(pr,st,false))
            if (!run_cfg<D,BW_F32,16,16,0,false,2,0>(pr,st,false)) {
                printf("alu_attn backward fp32: head_dim=%d does not fit in LDS (ceiling 224)\n", D);
                abort(); }
    }
}
// whether a variant is viable for the production gate shape (D=128, fp16/bf16)
static bool bwd_variant_buildable(int v,int MODE){
    BwdParams dummy{};
    if (MODE==BW_F16)  return launch_variant<128,BW_F16 >(dummy,0,v,true);
    if (MODE==BW_BF16) return launch_variant<128,BW_BF16>(dummy,0,v,true);
    return v==0;
}
extern "C" void alu_attn_backward_launch(BwdParams pr, int MODE, hipStream_t st){
    switch (pr.D){
        case 16: launch_D<16>(pr,MODE,st); break;   case 32: launch_D<32>(pr,MODE,st); break;
        case 48: launch_D<48>(pr,MODE,st); break;   case 64: launch_D<64>(pr,MODE,st); break;
        case 80: launch_D<80>(pr,MODE,st); break;   case 96: launch_D<96>(pr,MODE,st); break;
        case 112:launch_D<112>(pr,MODE,st);break;   case 128:launch_D<128>(pr,MODE,st);break;
        case 160:launch_D<160>(pr,MODE,st);break;   case 192:launch_D<192>(pr,MODE,st);break;
        case 224:launch_D<224>(pr,MODE,st);break;   case 256:launch_D<256>(pr,MODE,st);break;
        // This used to have an empty default: at head_dim=256 backward launched NOTHING,
        // and uninitialized dQ/dK/dV escaped. This must not fail silently.
        default:
            printf("alu_attn backward: head_dim=%d not supported\n", pr.D); abort();
    }
}

// ==========================================================================
//                          STANDALONE GATE (fp64 oracle)
// ==========================================================================
#ifdef FA_STANDALONE
#include <cstring>
static uint32_t GRS=12345u;
static float rndf(){ GRS=GRS*1664525u+1013904223u; return ((int)((GRS>>9)&0xFFFF)-32768)/32768.0f; }
// host replica of fa_common::dropout_mul (same Philox)
static float hdrop(unsigned long long seed, unsigned long long idx, float p, float invk){
    uint32_t k0=(uint32_t)seed,k1=(uint32_t)(seed>>32),c0=(uint32_t)idx,c1=(uint32_t)(idx>>32),c2=0,c3=0;
    for(int i=0;i<10;i++){ uint32_t hi0=(uint32_t)(((uint64_t)0xD2511F53u*c0)>>32), lo0=0xD2511F53u*c0;
        uint32_t hi1=(uint32_t)(((uint64_t)0xCD9E8D57u*c2)>>32), lo1=0xCD9E8D57u*c2;
        uint32_t n0=hi1^c1^k0,n1=lo1,n2=hi0^c3^k1,n3=lo0; c0=n0;c1=n1;c2=n2;c3=n3; k0+=0x9E3779B9u;k1+=0xBB67AE85u; }
    float u=(c0>>8)*(1.0f/16777216.0f); return (u>=p)?invk:0.0f;
}
static void pack(std::vector<uint8_t>& o,const std::vector<float>& v,int m){
    if(m==BW_F32){ o.resize(v.size()*4); memcpy(o.data(),v.data(),o.size()); return; }
    o.resize(v.size()*2);
    for(size_t i=0;i<v.size();i++){ if(m==BW_F16){ _Float16 h=(_Float16)v[i]; memcpy(&o[i*2],&h,2);}
        else { union{uint32_t u;float f;}c;c.f=v[i];uint32_t lsb=(c.u>>16)&1;c.u+=0x7fff+lsb;uint16_t bf=(uint16_t)(c.u>>16);memcpy(&o[i*2],&bf,2);} } }
static float unpk(const uint8_t* p,int i,int m){
    if(m==BW_F32){ float f; memcpy(&f,p+i*4,4); return f; }
    if(m==BW_F16){ _Float16 h; memcpy(&h,p+i*2,2); return (float)h; }
    uint16_t bf; memcpy(&bf,p+i*2,2); uint32_t u=(uint32_t)bf<<16; float f; memcpy(&f,&u,4); return f; }
static int host_shift(const std::vector<float>& v,int m){ if(m!=BW_BF16) return 0;
    float mx=0; for(float x:v) mx=fmaxf(mx,fabsf(x)); if(mx<=0) return 0; int e; frexpf(mx,&e); return 13-e; }

struct Case{ int B,Hq,Hk,SQ,SK,D; bool causal; int wl,wr; float softcap; bool alibi; float pdrop; const char* name; };

static int run_case(const Case& c,int mode,double tol,bool print){
    // fp32 uses four times as many bytes per element: at D>224 even the minimum
    // 16x16 tile (16896 elements * 4 = 67.6 KB versus a 64 KB limit) does not fit. flash-attn has no fp32
    // path at all, so this is not a parity gap — simply skip the cell explicitly.
    if (mode==BW_F32 && c.D>224){
        if(print) printf("  [fp32/%-22s] skip: does not fit in LDS (fp32 ceiling is D=224)\n", c.name);
        return 0;
    }
    const int B=c.B,Hq=c.Hq,Hk=c.Hk,SQ=c.SQ,SK=c.SK,D=c.D,gqa=Hq/Hk;
    const float scale=1.0f/sqrtf((float)D);
    std::vector<float> Q(B*Hq*SQ*D),K(B*Hk*SK*D),V(B*Hk*SK*D),dO(B*Hq*SQ*D);
    for(auto&x:Q)x=rndf(); for(auto&x:K)x=rndf(); for(auto&x:V)x=rndf(); for(auto&x:dO)x=rndf();
    std::vector<float> al(Hq); for(int h=0;h<Hq;h++) al[h]= c.alibi? -0.1f*(h+1):0.f;
    unsigned long long seed=777; float keep=1.0f-c.pdrop, invk=1.0f/keep;
    auto ok=[&](int qq,int kk){ if(kk>=SK||qq>=SQ)return false; int delta=SK-SQ,diag=qq+delta;
        if(c.causal&&kk>diag)return false;
        if(c.wl>=0&&kk<diag-c.wl)return false; if(c.wr>=0&&kk>diag+c.wr)return false; return true; };
    // forward oracle (fp64): O, LSE(fp32) with dropout matching the kernel (idx without batch => h,qq,kk)
    std::vector<float> O(B*Hq*SQ*D,0.f), LSE(B*Hq*SQ);
    for(int b=0;b<B;b++)for(int h=0;h<Hq;h++){int kvh=h/gqa;
        for(int qq=0;qq<SQ;qq++){ std::vector<double> lg(SK,-1e30); double mx=-1e30;
            for(int kk=0;kk<SK;kk++){ if(!ok(qq,kk))continue; double s=0;
                for(int d=0;d<D;d++) s+=(double)Q[(((long long)b*Hq+h)*SQ+qq)*D+d]*(double)K[(((long long)b*Hk+kvh)*SK+kk)*D+d];
                double g0=scale*s,logit; if(c.softcap>0){double t=tanh(g0/c.softcap);logit=c.softcap*t;}else logit=g0;
                logit+=al[h]*(double)(kk-qq); lg[kk]=logit; if(logit>mx)mx=logit; }
            double sum=0; for(int kk=0;kk<SK;kk++){ if(lg[kk]<=-1e29){lg[kk]=0;continue;} double e=exp(lg[kk]-mx); lg[kk]=e; sum+=e; }
            LSE[((long long)b*Hq+h)*SQ+qq]=(float)(mx+log(sum>0?sum:1));
            for(int d=0;d<D;d++){ double acc=0; for(int kk=0;kk<SK;kk++){ double p=lg[kk]/(sum>0?sum:1);
                double z=(c.pdrop>0)? (double)hdrop(seed,(((unsigned long long)h*SQ+qq)*SK+kk),c.pdrop,invk):1.0;
                acc+=z*p*(double)V[(((long long)b*Hk+kvh)*SK+kk)*D+d]; }
                O[(((long long)b*Hq+h)*SQ+qq)*D+d]=(float)acc; } } }
    // backward oracle (fp64)
    std::vector<double> gQ(Q.size(),0),gK(K.size(),0),gV(V.size(),0);
    for(int b=0;b<B;b++)for(int h=0;h<Hq;h++){int kvh=h/gqa;
        for(int qq=0;qq<SQ;qq++){ double LSEi=LSE[((long long)b*Hq+h)*SQ+qq], Di=0;
            for(int d=0;d<D;d++) Di+=(double)O[(((long long)b*Hq+h)*SQ+qq)*D+d]*(double)dO[(((long long)b*Hq+h)*SQ+qq)*D+d];
            for(int kk=0;kk<SK;kk++){ if(!ok(qq,kk))continue; double s=0;
                for(int d=0;d<D;d++) s+=(double)Q[(((long long)b*Hq+h)*SQ+qq)*D+d]*(double)K[(((long long)b*Hk+kvh)*SK+kk)*D+d];
                double g0=scale*s,deriv=1,logit; if(c.softcap>0){double t=tanh(g0/c.softcap);logit=c.softcap*t;deriv=1-t*t;}else logit=g0;
                logit+=al[h]*(double)(kk-qq); double p=exp(logit-LSEi);
                double z=(c.pdrop>0)? (double)hdrop(seed,(((unsigned long long)h*SQ+qq)*SK+kk),c.pdrop,invk):1.0;
                double dp=0; for(int d=0;d<D;d++) dp+=(double)dO[(((long long)b*Hq+h)*SQ+qq)*D+d]*(double)V[(((long long)b*Hk+kvh)*SK+kk)*D+d];
                double dS=p*(z*dp-Di)*deriv;
                for(int d=0;d<D;d++){ gV[(((long long)b*Hk+kvh)*SK+kk)*D+d]+=z*p*(double)dO[(((long long)b*Hq+h)*SQ+qq)*D+d];
                    gK[(((long long)b*Hk+kvh)*SK+kk)*D+d]+=scale*dS*(double)Q[(((long long)b*Hq+h)*SQ+qq)*D+d];
                    gQ[(((long long)b*Hq+h)*SQ+qq)*D+d]+=scale*dS*(double)K[(((long long)b*Hk+kvh)*SK+kk)*D+d]; } } } }
    // GPU
    std::vector<uint8_t> qb,kb,vb,ob,dob; pack(qb,Q,mode);pack(kb,K,mode);pack(vb,V,mode);pack(ob,O,mode);pack(dob,dO,mode);
    void *dQp,*dKp,*dVp,*dOp,*ddOp,*ddQ,*ddK,*ddV; float *dl,*ds,*dal=nullptr;
    HIP_CHECK(hipMalloc(&dQp,qb.size()));HIP_CHECK(hipMalloc(&dKp,kb.size()));HIP_CHECK(hipMalloc(&dVp,vb.size()));
    HIP_CHECK(hipMalloc(&dOp,ob.size()));HIP_CHECK(hipMalloc(&ddOp,dob.size()));
    HIP_CHECK(hipMalloc(&ddQ,qb.size()));HIP_CHECK(hipMalloc(&ddK,kb.size()));HIP_CHECK(hipMalloc(&ddV,vb.size()));
    HIP_CHECK(hipMalloc(&dl,LSE.size()*4));HIP_CHECK(hipMalloc(&ds,LSE.size()*4));
    if(c.alibi){HIP_CHECK(hipMalloc(&dal,Hq*4));HIP_CHECK(hipMemcpy(dal,al.data(),Hq*4,hipMemcpyHostToDevice));}
    HIP_CHECK(hipMemcpy(dQp,qb.data(),qb.size(),hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(dKp,kb.data(),kb.size(),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dVp,vb.data(),vb.size(),hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(dOp,ob.data(),ob.size(),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(ddOp,dob.data(),dob.size(),hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(dl,LSE.data(),LSE.size()*4,hipMemcpyHostToDevice));
    auto sD=[&](int H,int S){ return Str{(long long)H*S*D,(long long)S*D,(long long)D}; };
    BwdParams pr{}; pr.Q=dQp;pr.K=dKp;pr.V=dVp;pr.O=dOp;pr.dO=ddOp;pr.dQ=ddQ;pr.dK=ddK;pr.dV=ddV;pr.lse=dl;pr.dsum=ds;
    pr.qs=sD(Hq,SQ);pr.ks=sD(Hk,SK);pr.vs=sD(Hk,SK);pr.os=sD(Hq,SQ);pr.dos=sD(Hq,SQ);pr.dqs=sD(Hq,SQ);pr.dks=sD(Hk,SK);pr.dvs=sD(Hk,SK);
    pr.B=B;pr.Hq=Hq;pr.Hk=Hk;pr.SQ=SQ;pr.SK=SK;pr.D=D;pr.gqa=gqa;pr.scale=scale;pr.softcap=c.softcap;pr.alibi=dal;
    pr.causal=c.causal;pr.win_left=c.wl;pr.win_right=c.wr;pr.dropout=(c.pdrop>0);pr.drop_seed=seed;pr.dropout_p=c.pdrop;pr.keep_scale=invk;
    pr.shQ=host_shift(Q,mode);pr.shK=host_shift(K,mode);pr.shV=host_shift(V,mode);pr.shdO=host_shift(dO,mode);
    alu_attn_backward_launch(pr,mode,0); HIP_CHECK(hipDeviceSynchronize());
    std::vector<uint8_t> hq(qb.size()),hk(kb.size()),hv(vb.size());
    HIP_CHECK(hipMemcpy(hq.data(),ddQ,qb.size(),hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hk.data(),ddK,kb.size(),hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hv.data(),ddV,vb.size(),hipMemcpyDeviceToHost));
    auto rr=[&](const std::vector<uint8_t>& g,const std::vector<double>& r)->double{ double se=0,sr=0;
        for(size_t i=0;i<r.size();i++){ double gg=unpk(g.data(),(int)i,mode),d=gg-r[i]; se+=d*d; sr+=r[i]*r[i]; } return sqrt(se/(sr>0?sr:1)); };
    double eQ=rr(hq,gQ),eK=rr(hk,gK),eV=rr(hv,gV);
    // NaN passes NO comparison, so eQ<tol on NaN returned false... but under -ffast-math
    // the compiler may assume NaN is impossible and fold the comparison to true. The gate then printed
    // "dQ -nan ... PASS" for a clearly broken kernel. Check finiteness explicitly through a range test.
    auto finite=[](double x){ return x>-1e300 && x<1e300; };
    bool pass = finite(eQ)&&finite(eK)&&finite(eV) && eQ<tol&&eK<tol&&eV<tol;
    // determinism
    alu_attn_backward_launch(pr,mode,0); HIP_CHECK(hipDeviceSynchronize());
    std::vector<uint8_t> hq2(qb.size()); HIP_CHECK(hipMemcpy(hq2.data(),ddQ,qb.size(),hipMemcpyDeviceToHost));
    bool det=(memcmp(hq.data(),hq2.data(),qb.size())==0);
    if(print) printf("  [%-13s %-4s] dQ %.2e dK %.2e dV %.2e  %s%s\n", c.name,
        mode==BW_F16?"f16":mode==BW_BF16?"bf16":"f32", eQ,eK,eV, pass?"PASS":"FAIL", det?"":" [NONDETERMINISTIC]");
    hipFree(dQp);hipFree(dKp);hipFree(dVp);hipFree(dOp);hipFree(ddOp);hipFree(ddQ);hipFree(ddK);hipFree(ddV);hipFree(dl);hipFree(ds);if(dal)hipFree(dal);
    return (pass&&det)?0:1;
}

// print=true prints a benchmark row; always returns total time and breaks it down by stage.
static float bench(int mode, bool print=true, float* out_kv=nullptr, float* out_q=nullptr){
    const int B=2,H=16,S=2048,D=128; const float scale=1.0f/sqrtf((float)D);
    std::vector<float> Q(B*H*S*D); for(auto&x:Q)x=rndf(); auto K=Q,V=Q,dO=Q,O=Q; std::vector<float> LSE(B*H*S,0.f);
    std::vector<uint8_t> qb,kb,vb,ob,dob; pack(qb,Q,mode);pack(kb,K,mode);pack(vb,V,mode);pack(ob,O,mode);pack(dob,dO,mode);
    void *dQp,*dKp,*dVp,*dOp,*ddOp,*ddQ,*ddK,*ddV; float*dl,*ds;
    HIP_CHECK(hipMalloc(&dQp,qb.size()));HIP_CHECK(hipMalloc(&dKp,kb.size()));HIP_CHECK(hipMalloc(&dVp,vb.size()));HIP_CHECK(hipMalloc(&dOp,ob.size()));HIP_CHECK(hipMalloc(&ddOp,dob.size()));
    HIP_CHECK(hipMalloc(&ddQ,qb.size()));HIP_CHECK(hipMalloc(&ddK,kb.size()));HIP_CHECK(hipMalloc(&ddV,vb.size()));HIP_CHECK(hipMalloc(&dl,LSE.size()*4));HIP_CHECK(hipMalloc(&ds,LSE.size()*4));
    HIP_CHECK(hipMemcpy(dQp,qb.data(),qb.size(),hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(dKp,kb.data(),kb.size(),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dVp,vb.data(),vb.size(),hipMemcpyHostToDevice));HIP_CHECK(hipMemcpy(dOp,ob.data(),ob.size(),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(ddOp,dob.data(),dob.size(),hipMemcpyHostToDevice));HIP_CHECK(hipMemset(dl,0,LSE.size()*4));
    auto sD=[&](int S2){ return Str{(long long)H*S2*D,(long long)S2*D,(long long)D}; };
    BwdParams pr{}; pr.Q=dQp;pr.K=dKp;pr.V=dVp;pr.O=dOp;pr.dO=ddOp;pr.dQ=ddQ;pr.dK=ddK;pr.dV=ddV;pr.lse=dl;pr.dsum=ds;
    pr.qs=sD(S);pr.ks=sD(S);pr.vs=sD(S);pr.os=sD(S);pr.dos=sD(S);pr.dqs=sD(S);pr.dks=sD(S);pr.dvs=sD(S);
    pr.B=B;pr.Hq=H;pr.Hk=H;pr.SQ=S;pr.SK=S;pr.D=D;pr.gqa=1;pr.scale=scale;pr.softcap=0;pr.alibi=nullptr;pr.causal=1;pr.win_left=-1;pr.win_right=-1;
    pr.dropout=0;pr.shQ=host_shift(Q,mode);pr.shK=host_shift(K,mode);pr.shV=host_shift(V,mode);pr.shdO=host_shift(dO,mode);
    auto timeit=[&](int stage,int probe=0){
        g_bwd_stage=stage; pr.probe=probe;
        for(int i=0;i<3;i++) alu_attn_backward_launch(pr,mode,0); HIP_CHECK(hipDeviceSynchronize());
        hipEvent_t t0,t1; HIP_CHECK(hipEventCreate(&t0));HIP_CHECK(hipEventCreate(&t1));
        const int IT=20; HIP_CHECK(hipEventRecord(t0));
        for(int i=0;i<IT;i++) alu_attn_backward_launch(pr,mode,0);
        HIP_CHECK(hipEventRecord(t1));HIP_CHECK(hipEventSynchronize(t1));
        float ms=0;HIP_CHECK(hipEventElapsedTime(&ms,t0,t1));
        hipEventDestroy(t0);hipEventDestroy(t1); g_bwd_stage=0; pr.probe=0; return ms/IT; };
    float ms=timeit(0), mp=timeit(1), mkv=timeit(2), mq=timeit(3);
    if(out_kv)*out_kv=mkv; if(out_q)*out_q=mq;
    double fwd=2.0*2.0*B*H*(double)S*S*D*0.5; double flops=2.5*fwd;
    if(print){
    float pkv=timeit(2,1), pq=timeit(3,1);   // probe: same kernels without the second GEMM
    printf("  bench %-4s B%dH%dS%dD%d causal: %.3f ms -> %.2f TFLOP/s (effective useful)\n",
        mode==BW_F16?"f16":mode==BW_BF16?"bf16":"f32",B,H,S,D,ms,flops/(ms*1e-3)/1e12);
    printf("        by stage: preprocess %.3f ms (%.1f%%) | dK/dV %.3f ms (%.1f%%) | dQ %.3f ms (%.1f%%)\n",
        mp,100*mp/ms, mkv,100*mkv/ms, mq,100*mq/ms);
    printf("        inside: dK/dV = fill_pd %.3f + GEMM2 %.3f | dQ = fill_pd %.3f + GEMM2 %.3f\n",
        pkv, mkv-pkv, pq, mq-pq); }
    hipFree(dQp);hipFree(dKp);hipFree(dVp);hipFree(dOp);hipFree(ddOp);hipFree(ddQ);hipFree(ddK);hipFree(ddV);hipFree(dl);hipFree(ds);
    return ms;
}

// Test bench: sweep ALL variants, validate each against the oracle, and measure each one. Do not choose
// by reasoning — let the table choose.
static void sweep(int mode){
    printf("\n=== BACKWARD TEST BENCH (%s), B2 H16 S2048 D128 causal ===\n", mode==BW_F16?"fp16":"bf16");
    printf("  %-28s %9s %9s %8s %9s %9s  %s\n","variant","total","dK/dV","dQ","TFLOP/s","vs base","correctness");
    { char bar[104]; memset(bar,'-',103); bar[103]=0; printf("  %s\n",bar); }
    float base=-1;
    for (int v=0; v<BWD_NVARIANT; v++){
        if (!bwd_variant_buildable(v,mode)){
            printf("  %-28s skip: does not fit in LDS or tile is incompatible with packing\n", bwd_variant_name(v));
            continue;
        }
        g_bwd_variant=v;
        // correctness on the same cases as the gate
        Case cs[]={{1,4,4,128,128,128,true, -1,-1,0.f,false,0.f,"sweep-causal"},
                   {1,4,4,128,128,128,false, -1,-1,0.f,false,0.f,"sweep-full"},
                   {2,8,2,128,128,128,true, -1,-1,0.f,false,0.f,"sweep-gqa"}};
        int bad=0; for(auto&c:cs) bad+=run_case(c,mode,mode==BW_F16?2.5e-3:6.0e-3,false);
        float mkv=0, mq=0;
        float ms=bench(mode,false,&mkv,&mq);
        if (base<0) base=ms;
        double fwd=2.0*2.0*2*16*(double)2048*2048*128*0.5, flops=2.5*fwd;
        printf("  %-28s %7.3f ms %7.3f %7.3f ms %8.2f  %+6.1f%%  %s\n",
               bwd_variant_name(v), ms, mkv, mq, flops/(ms*1e-3)/1e12,
               base>0?100.0*(base-ms)/base:0.0, bad?"FAIL":"OK");
    }
    g_bwd_variant=0;
}

int main(int argc,char** argv){
    bool do_bench=(argc>1&&strcmp(argv[1],"--bench")==0);
    bool do_sweep=(argc>1&&strcmp(argv[1],"--sweep")==0);
    Case cs[]={
        {1,4,4, 64, 64,64,false, -1,-1,  0.f,false,0.f,"base"},
        {1,4,4, 64, 64,64,true, -1,-1,  0.f,false,0.f,"causal"},
        {2,8,2,128,128,64,true, -1,-1,  0.f,false,0.f,"causal+gqa"},
        {1,4,4,100,100,64,true, -1,-1,  0.f,false,0.f,"causal-tail"},
        {1,4,4, 96,128,64,true, -1,-1,  0.f,false,0.f,"cross SQ<SK"},
        {1,4,4,128, 96,64,true, -1,-1,  0.f,false,0.f,"cross SQ>SK"},
        {1,4,4,128,128,64,false,47,0,  0.f,false,0.f,"window (47,0)"},
        {1,4,4,128,128,64,false,24,24, 0.f,false,0.f,"window (24,24)"},
        {1,4,4,128,128,64,false,-1, 0,  0.f,false,0.f,"window (-1,0)=causal"},
        {1,4,4, 64, 64,192,false,-1,-1, 0.f,false,0.f,"D192"},
        {1,4,4, 64, 64,256,true, -1,-1, 0.f,false,0.f,"D256"},
        {1,4,4, 64, 64,160,true, -1,-1, 0.f,false,0.f,"D160"},
        {1,4,4, 64, 64,224,false,-1,-1, 0.f,false,0.f,"D224"},
        {1,4,4,128,128,64,false, -1,-1, 30.f,false,0.f,"softcap"},
        {1,4,4,128,128,64,true, -1,-1,  0.f,true, 0.f,"alibi"},
        {1,4,4,128,128,64,false, -1,-1,  0.f,false,0.2f,"dropout"},
        {1,4,4, 64, 64,16,false, -1,-1,  0.f,false,0.f,"D16"},
        {1,4,4, 64, 64,128,false, -1,-1, 0.f,false,0.f,"D128"},
        {1,4,4, 80, 80,96,true, -1,-1,  0.f,false,0.f,"D96"},
    };
    int fails=0;
    if(!do_sweep){   // in test-bench mode correctness is checked inside it for every variant
        printf("=== BACKWARD fp64 oracle (rel_rms dQ/dK/dV) ===\n");
        for(auto&c:cs){ fails+=run_case(c,BW_F16,2.5e-3,true); fails+=run_case(c,BW_BF16,6.0e-3,true); fails+=run_case(c,BW_F32,1.0e-4,true); }
        printf(fails?"\n*** FAILURES: %d ***\n":"\nALL GREEN\n",fails);
    }
    if(do_bench){ printf("\n=== BENCHMARK (one short run) ===\n"); bench(BW_F16); bench(BW_BF16); }
    if(do_sweep){ sweep(BW_F16); sweep(BW_BF16); }
    return fails;
}
#endif
