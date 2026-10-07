#include "../package/csrc/alu_compiler_policy.h" // ALU_COMPILER_HARDENING_POLICY
// fa_forward.hip — exact FlashAttention-2 forward for gfx1030 (RX 6800 XT), inference path.
// MAIN LEVER vs the champion (flash_prefill3, memory-bound at 94% HBM): GQA REUSE — one workgroup
// serves G query heads for ONE KV head; K/V load once and are reused => global K/V traffic divided by G.
// This moves the kernel from memory-bound to compute-bound (structural ceiling ~30 TFLOP/s at full clocks).
//
// Design (optimization hooks built in FROM THE START):
//  * block = (q tile BR, group of G heads, batch). 256 threads = 16 ty x 16 tx, t=ty*16+tx.
//  * both matrix products (S=QKᵀ, O=PV) use register fdot2 tiling; softmax reduction over keys =
//    intrawave __shfl_xor(width=16) WITHOUT LDS; exp2f; NEVER materializes SxS.
//  * K/V half8 (128-bit) coalesced loads; Q tiles remain resident in LDS (loaded once).
//  * lazy rescale: skip acco corr multiplication when max does not move (lossless).
//  * dtype-generic <fp16,bf16>: bf16 uses the SAME v_dot2 — per-tile pow2 scaling brings bf16
//    into fp16 range WITHOUT mantissa loss (7-bit bf16 subset of 10-bit fp16); power-of-two unfolding is exact.
//  * features: causal (bottom-right when SQ!=SK), sliding window, softcap, alibi, GQA, batch, arbitrary strides
//    (dim contiguous), varlen (cu_seqlens), O and LSE(fp32) outputs, dropout (Philox, p=0 is free).
// fp32 path uses a separate kernel (fa_forward_fp32) below, correctness 1e-5 (fdot2 is fp16 only).
// Build and run: bash tools/build_kernel_gates.sh
#include "fa_common.hpp"

// ---- tensor strides (dim contiguous, stride_d=1) ----
struct Strides { long long b,h,s; };   // batch, head, seq (in elements)

struct FwdParams {
    const void* Q; const void* K; const void* V; void* O; float* LSE;   // LSE optional
    Strides sQ,sK,sV,sO;
    int SQ, SK, HD, NH, gqa;
    float scale, softcap;               // softcap<=0 => off
    // Window using flash-attn semantics: allowed keys [qq+delta-win_left .. qq+delta+win_right],
    // a negative value means no boundary on that side. window_size=(-1,-1) => both <0;
    // causal = win_right==0. This used to have one window field (left window only, with the right
    // boundary on the diagonal), so bidirectional attention ranges could not be represented at all.
    int   win_left, win_right;
    const float* alibi_slopes;          // [NH] or nullptr
    int   causal;                       // bottom-right when SQ!=SK
    float dropout_p; unsigned long long philox_seed;
    const int* cu_q; const int* cu_k;   // varlen (nullptr => dense)
    int batch;
    // bf16-range pow2 scales (lossless): fp16 value = bf16 value * sc_*. Default 1.
    // bf16 mantissa (7 bits) is a subset of fp16 (10), so conversion is exact; sc shifts exponent into fp16 range.
    float sc_q, sc_k, sc_v;
    // --- flash-attn compatibility fields; all disabled by zero/nullptr ---
    // Chunked attention (FA-3 attention_chunk, Llama-4 scheme): query at position i sees
    // only keys in its chunk [i/chunk*chunk, (i/chunk+1)*chunk-1], layered on top of causal/window.
    // 0 => disabled. Read ONLY under FEAT, so the dense path does not gain even one instruction.
    int attn_chunk;
    // seqused_q/seqused_k (varlen): actually used sequence length, potentially shorter
    // than cu_seqlens indicates. nullptr => use the full length from cu_seqlens.
    const int* seqused_q; const int* seqused_k;
};

// ========================= main kernel (fp16/bf16) =========================
template<class DT,int HD,int BR,int BC,int G,bool CAUSAL,bool FEAT>
__global__ __launch_bounds__(256)
void fa_forward_k(FwdParams p)
{
    typedef typename DT::T T;
    constexpr bool SCALED = DT::is_bf16;   // bf16: inline conversion + per-tensor pow2 scale (lossless, fp16 speed)
    constexpr int QPT=BR/16, KPT=BC/16, DPT=HD/16, HH=HD/2, BCH=BC/2;
    constexpr float LOG2E=1.442695041f;

    const int t=threadIdx.x, tx=t&15, ty=t>>4;
    const int z=blockIdx.z;
    const int hg=blockIdx.y;
    const int h0=hg*G, kvh=h0/p.gqa;

    int SQ,SK; long long qrow0,krow0,orow0;
    if(p.cu_q){ int qs=p.cu_q[z], ks=p.cu_k[z]; SQ=p.cu_q[z+1]-qs; SK=p.cu_k[z+1]-ks;
        // seqused_*: sequence in the buffer is longer than the portion actually used (FA-3).
        // One scalar read per workgroup, outside all loops.
        if(p.seqused_q){ int u=p.seqused_q[z]; if(u<SQ) SQ=u; }
        if(p.seqused_k){ int u=p.seqused_k[z]; if(u<SK) SK=u; }
        qrow0=(long long)qs*p.sQ.s; krow0=(long long)ks*p.sK.s; orow0=(long long)qs*p.sO.s; }
    else { SQ=p.SQ; SK=p.SK; qrow0=(long long)z*p.sQ.b; krow0=(long long)z*p.sK.b; orow0=(long long)z*p.sO.b; }
    const int qb=blockIdx.x*BR;
    if(qb>=SQ) return;
    const int delta=SK-SQ;
    const float scale2=p.scale*LOG2E;
    // bf16 pow2 scales: bf16->fp16 conversion multiplies by sc_*; unfold divides back (exact, pow2)
    const float scq = SCALED? p.sc_q : 1.0f, sck = SCALED? p.sc_k : 1.0f, scv = SCALED? p.sc_v : 1.0f;
    const float uqk = SCALED? 1.0f/(scq*sck) : 1.0f;   // unfold Q·K
    const float uv  = SCALED? 1.0f/scv       : 1.0f;   // unfold in P storage

    // ---- LDS layout (manual pool, <=64KB) ----
    constexpr int QS_B = G*HH*BR*4;
    constexpr int KS_B = HH*BC*4;
    constexpr int VS_B = BCH*HD*4;
    constexpr int PS_B = (BR<=HD)?0:(BC*BR*2);      // Ps aliases Ks when BR<=HD
    __shared__ char pool[QS_B+KS_B+VS_B+PS_B];
    half2_t* Qs=(half2_t*)pool;
    half2_t* Ks=(half2_t*)(pool+QS_B);
    half2_t* Vs=(half2_t*)(pool+QS_B+KS_B);
    _Float16* Ps=(BR<=HD)?(_Float16*)Ks:(_Float16*)(pool+QS_B+KS_B+VS_B);
    half2_t*  Pp=(half2_t*)Ps;   // same buffer, viewed as "key pairs in one dword"
    #define QSA(g,dh,q) Qs[((g)*HH+(dh))*BR+(q)]
    // Swizzle kc by dh. K writes to LDS use lines where dh changes at fixed kc
    // (required for coalesced global reads), and address (dh)*BC+kc placed all
    // 16 lines in ONE bank — a 16-way conflict in the load stage, which accounts for 18% of time.
    // XOR spreads them across 16 banks; GEMM1 reads (fixed dh, varying kc) remain
    // conflict-free. Row padding does not fit here — only 256 bytes remain before the 64 KB limit.
    #define KSA(dh,kc)  Ks[(dh)*BC+((kc)^((dh)&31))]
    // Same technique for V writes: lines vary dg with stride 8 dwords => 4 banks for 16 lines.
    // XOR on high d bits spreads writes over 16 banks, while GEMM2 reads (lines vary tx
    // within one e) remain conflict-free — XOR is constant there.
    #define VSA(kh,d)   Vs[(kh)*HD+((d)^(((d)>>4)&7))]
    #define PSA(kc,q)   Ps[(kc)*BR+(q)]
    #define PPA(kh,q)   Pp[(kh)*BR+(q)]

    // inline load of 8 elements (128-bit) with bf16->fp16*sc conversion
    auto load8=[&](const T* pt,bool valid,float sc,_Float16 o[8]){
        if constexpr(SCALED){
            uint4 raw=valid?*(const uint4*)pt:uint4{0,0,0,0}; const uint16_t* w=(const uint16_t*)&raw;
            #pragma unroll
            for(int m=0;m<8;m++){ uint32_t b=((uint32_t)w[m])<<16; float f; __builtin_memcpy(&f,&b,4); o[m]=(_Float16)(f*sc); }
        } else {
            half8_t a=valid?*(const half8_t*)pt:half8_t{0,0,0,0,0,0,0,0};
            #pragma unroll
            for(int m=0;m<8;m++) o[m]=a[m];
        }
    };

    // ---- load Q tiles (resident) ----
    #pragma unroll
    for(int g=0; g<G; g++){
        const T* Qb=(const T*)p.Q + qrow0 + (long long)(h0+g)*p.sQ.h;
        for(int e=t; e<BR*(HD/8); e+=256){
            int dg=e%(HD/8), q=e/(HD/8); int sq=qb+q; _Float16 o[8];
            load8(Qb+(long long)sq*p.sQ.s+dg*8, sq<SQ, scq, o);
            #pragma unroll
            for(int m=0;m<4;m++) QSA(g,dg*4+m,q)=half2_t{o[2*m],o[2*m+1]};
        }
    }

    float acco[G][QPT][DPT]; float rm[G][QPT], rl[G][QPT];
    #pragma unroll
    for(int g=0;g<G;g++)
        #pragma unroll
        for(int i=0;i<QPT;i++){ rm[g][i]=-1e30f; rl[g][i]=0.0f;
            #pragma unroll
            for(int e=0;e<DPT;e++) acco[g][i][e]=0.0f; }
    __syncthreads();

    int kmax = CAUSAL? (((qb+BR-1)+delta < SK-1)?(qb+BR-1)+delta:SK-1) : SK-1;
    // Early start/end from window: skip tiles entirely outside the window. Lower bound uses
    // the FIRST row of the tile, upper bound the last, so they trim the tile for all rows.
    int kb0=0;
    if(FEAT && p.win_left>=0){ int lo=qb+delta-p.win_left; if(lo<0) lo=0; kb0=(lo/BC)*BC; }
    if(FEAT && p.win_right>=0){ int hi=qb+BR-1+delta+p.win_right; if(hi<kmax) kmax=hi; }
    // Same for chunked attention: the first row starts its chunk earliest,
    // and the last row ends its chunk latest, so these two bounds trim tiles for the entire block.
    if(FEAT && p.attn_chunk>0){
        int p0=qb+delta;         if(p0<0) p0=0;      // C integer division of negatives truncates toward zero
        int p1=qb+BR-1+delta;    if(p1<0) p1=0;
        int lo=(p0/p.attn_chunk)*p.attn_chunk;
        if(lo>kb0) kb0=(lo/BC)*BC;
        int hi=(p1/p.attn_chunk+1)*p.attn_chunk-1;
        if(hi<kmax) kmax=hi;
    }

    for(int kb=kb0; kb<=kmax; kb+=BC){
        const bool diag = CAUSAL && ((kb+BC-1)>=qb+delta);
        __syncthreads();
        // K (transposed into Ks[dh][kc]) — unrolled load, multiple 128-bit reads in flight (MLP)
        constexpr int NK=(BC*(HD/8)+255)/256;
        #pragma unroll
        for(int it=0; it<NK; it++){ int e=t+it*256; if(e<BC*(HD/8)){
            int dg=e%(HD/8), kc=e/(HD/8); int sk=kb+kc; _Float16 o[8];
            load8((const T*)p.K+krow0+(long long)kvh*p.sK.h+(long long)sk*p.sK.s+dg*8, sk<SK, sck, o);
            #pragma unroll
            for(int m=0;m<4;m++) KSA(dg*4+m,kc)=half2_t{o[2*m],o[2*m+1]}; } }
        // V (Vs[kh][d], 2 rows packed into half2)
        constexpr int NV=(BCH*(HD/8)+255)/256;
        #pragma unroll
        for(int it=0; it<NV; it++){ int e=t+it*256; if(e<BCH*(HD/8)){
            int dg=e%(HD/8), kh=e/(HD/8); int s0=kb+2*kh,s1=s0+1; int d0=dg*8; _Float16 a[8],b[8];
            load8((const T*)p.V+krow0+(long long)kvh*p.sV.h+(long long)s0*p.sV.s+d0, s0<SK, scv, a);
            load8((const T*)p.V+krow0+(long long)kvh*p.sV.h+(long long)s1*p.sV.s+d0, s1<SK, scv, b);
            #pragma unroll
            for(int m=0;m<8;m++) VSA(kh,d0+m)=half2_t{a[m],b[m]}; } }
        __syncthreads();

        // GEMM1 (K live, all G heads)
        float accs[G][QPT][KPT];
        #pragma unroll
        for(int g=0;g<G;g++)
            #pragma unroll
            for(int i=0;i<QPT;i++)
                #pragma unroll
                for(int j=0;j<KPT;j++) accs[g][i][j]=0.0f;
        #pragma unroll 4
        for(int dh=0; dh<HH; dh++){
            half2_t kv[KPT];
            // Thread keys: (j/2)*32 + 2tx + (j%2) instead of tx*KPT+j. Line stride is 2 dwords
            // instead of KPT=4 => 16 distinct banks instead of 8, while key pairs remain adjacent,
            // so the half2 P packing below still works.
            #pragma unroll
            for(int j=0;j<KPT;j++) kv[j]=KSA(dh,(j>>1)*32+2*tx+(j&1));
            #pragma unroll
            for(int g=0;g<G;g++){
                half2_t qv[QPT];
                #pragma unroll
                for(int i=0;i<QPT;i++) qv[i]=QSA(g,dh,ty*QPT+i);
                #pragma unroll
                for(int i=0;i<QPT;i++)
                    #pragma unroll
                    for(int j=0;j<KPT;j++) accs[g][i][j]=__builtin_amdgcn_fdot2(qv[i],kv[j],accs[g][i][j],false);
            }
        }
        __syncthreads();  // K dead -> Ps alias

        #pragma unroll
        for(int g=0; g<G; g++){
            const int h=h0+g;
            const float aslope = (FEAT && p.alibi_slopes)? p.alibi_slopes[h] : 0.0f;
            #pragma unroll
            for(int i=0;i<QPT;i++){
                const int qq=qb+ty*QPT+i;
                float sj[KPT], lmax=-1e30f;
                #pragma unroll
                for(int j=0;j<KPT;j++){
                    const int key=kb+(j>>1)*32+2*tx+(j&1);   // same layout as kv[]
                    float a=accs[g][i][j]; if constexpr(SCALED) a*=uqk;   // true Q*K
                    float sv;
                    if constexpr(FEAT){
                        if(p.softcap>0.0f){ float x=a*p.scale; sv=p.softcap*tanhf(x/p.softcap)*LOG2E; }
                        else sv=a*scale2;
                        if(p.alibi_slopes) sv+=aslope*(float)(key-qq)*LOG2E;
                        bool masked = CAUSAL ? false : (key>=SK);
                        if(CAUSAL && diag && key>qq+delta) masked=true;
                        if(p.win_left>=0  && key < qq+delta-p.win_left ) masked=true;
                        if(p.win_right>=0 && key > qq+delta+p.win_right) masked=true;
                        if(p.attn_chunk>0){ int pp=qq+delta; if(pp<0) pp=0;
                            int cs=(pp/p.attn_chunk)*p.attn_chunk;
                            if(key<cs||key>=cs+p.attn_chunk) masked=true; }
                        if(masked) sv=-1e30f;
                    } else {
                        sv=a*scale2;
                        // CAUSAL: mask only in the diagonal tile (key>=SK tail is covered by causality).
                        if constexpr(CAUSAL){ if(diag && key>qq+delta) sv=-1e30f; }
                        else { if(key>=SK) sv=-1e30f; }
                    }
                    sj[j]=sv; lmax=fmaxf(lmax,sv);
                }
                lmax=red16_max(lmax);                 // DPP instead of ds_bpermute
                float mnew=fmaxf(rm[g][i],lmax);
                float corr = (mnew>rm[g][i])? __builtin_amdgcn_exp2f(rm[g][i]-mnew) : 1.0f;  // lazy-rescale
                float lsum=0.0f;
                #pragma unroll
                for(int j=0;j<KPT;j++){ float pj=__builtin_amdgcn_exp2f(sj[j]-mnew); sj[j]=pj; lsum+=pj; }
                lsum=red16_sum(lsum);                 // DPP instead of ds_bpermute
                rl[g][i]=rl[g][i]*corr+lsum; rm[g][i]=mnew;
                if(corr!=1.0f){
                    #pragma unroll
                    for(int e=0;e<DPT;e++) acco[g][i][e]*=corr;
                }
                if(FEAT && p.dropout_p>0.0f){ float ik=1.0f/(1.0f-p.dropout_p);
                    #pragma unroll
                    for(int j=0;j<KPT;j++){ int key=kb+(j>>1)*32+2*tx+(j&1);
                        unsigned long long idx=(((unsigned long long)h*SQ+qq)*SK+key);
                        sj[j]*=dropout_mul(p.philox_seed,idx,p.dropout_p,ik); } }
                // Thread keys are consecutive (tx*KPT .. +KPT-1), so pack pairs directly into half2.
                // GEMM2 then reads a ready dword instead of two half values + v_perm (RGA: 32 v_perm per tile).
                #pragma unroll
                for(int j=0;j<KPT;j+=2){
                    float a = SCALED? sj[j]*uv   : sj[j];
                    float b = SCALED? sj[j+1]*uv : sj[j+1];
                    PPA((j>>1)*16+tx, ty*QPT+i)=half2_t{(_Float16)a,(_Float16)b};
                }
            }
            __syncthreads();
            #pragma unroll 4
            for(int kh=0; kh<BCH; kh++){
                half2_t pv[QPT], vv[DPT];
                #pragma unroll
                for(int i=0;i<QPT;i++) pv[i]=PPA(kh,ty*QPT+i);   // one b32, no unpacking
                // LDS measurements: with d=tx*DPT+e, line stride DPT=8 dwords, and 16 lines
                // hit only 4 banks — a 4-way conflict in the hottest loop (GEMM2
                // is 36% of time). Stride 16 spreads lines over 16 different banks. Measured: +9%.
                // Bonus: the epilogue writes O contiguously across lines instead of stride 8.
                #pragma unroll
                for(int e=0;e<DPT;e++) vv[e]=VSA(kh,e*16+tx);
                #pragma unroll
                for(int i=0;i<QPT;i++)
                    #pragma unroll
                    for(int e=0;e<DPT;e++) acco[g][i][e]=__builtin_amdgcn_fdot2(pv[i],vv[e],acco[g][i][e],false);
            }
            if(g+1<G) __syncthreads();   // barrier only BETWEEN heads; after the last, the next tile's top sync handles it
        }
    }

    // ---- epilogue: O = acco/l, LSE ----
    #pragma unroll
    for(int g=0; g<G; g++){
        const int h=h0+g;
        T* Ob=(T*)p.O+orow0+(long long)h*p.sO.h;
        #pragma unroll
        for(int i=0;i<QPT;i++){ int qq=qb+ty*QPT+i; if(qq>=SQ) continue;
            // A row with ALL keys masked (causal with SQ>SK, narrow window, chunk,
            // seqused_k): rm remains -1e30, and exp2(-1e30 - (-1e30)) gives 1 rather than 0 —
            // rl becomes nonzero and acco accumulates sum(V). Detect via the unchanged maximum
            // and return zeros with LSE=-inf like flash-attn. Check is in the epilogue, outside the hot loop.
            const bool empty = rm[g][i] <= -1e29f;
            float inv = (rl[g][i]>0.0f && !empty)? 1.0f/rl[g][i] : 0.0f;
            #pragma unroll
            for(int e=0;e<DPT;e++){ int d=e*16+tx; Ob[(long long)qq*p.sO.s+d]=(T)(acco[g][i][e]*inv); }
            if(p.LSE && tx==0){ float lse = (rl[g][i]>0.0f && !empty)? (rm[g][i]*0.693147181f + logf(rl[g][i])) : -INFINITY;
                p.LSE[(size_t)h*SQ+qq]=lse; }
        }
    }
    #undef QSA
    #undef KSA
    #undef VSA
    #undef PSA
}


// ================= gfx1030 D256 phase-separated FP16 fast path =================
//
// Qwen3.8 TP4 optimization discovered in Optimization 16.
//
// D256 production historically uses BR=64 / BC=32 and keeps:
//
//   Q = 32 KB
//   K = 16 KB
//   V = 16 KB
//
// resident simultaneously, consuming the complete 64 KB LDS budget.
//
// For dense causal FP16 attention, K is dead before GEMM2 begins. This
// specialization aliases K and V into one 16 KB scratch region and gives P
// separate storage:
//
//   Q        = 40 KB   (BR=80)
//   K or V   = 16 KB
//   P        =  5 KB
//   total    = 61 KB
//
// BR=80 increases reuse of each K/V tile across query rows by 25% relative
// to BR=64. The existing generic kernel remains the fallback for every
// unsupported configuration.
//
// This kernel intentionally implements only the narrowly gated dense causal
// FP16 path. Do not broaden dispatch without separate correctness/performance
// qualification.
//
__global__ __launch_bounds__(256)
void fa_forward_phase80_d256_f16(FwdParams p)
{
    constexpr int HD  = 256;
    constexpr int BR  = 80;
    constexpr int BC  = 32;
    constexpr int QPT = BR / 16;
    constexpr int KPT = BC / 16;
    constexpr int DPT = HD / 16;
    constexpr int HH  = HD / 2;
    constexpr int BCH = BC / 2;

    constexpr float LOG2E = 1.442695041f;

    static_assert(QPT == 5);
    static_assert(KPT == 2);
    static_assert(DPT == 16);

    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;

    const int z   = blockIdx.z;
    const int h   = blockIdx.y;
    const int kvh = h / p.gqa;

    int SQ, SK;
    long long qrow0, krow0, orow0;

    if (p.cu_q) {
        const int qs = p.cu_q[z];
        const int ks = p.cu_k[z];

        SQ = p.cu_q[z + 1] - qs;
        SK = p.cu_k[z + 1] - ks;

        qrow0 = (long long)qs * p.sQ.s;
        krow0 = (long long)ks * p.sK.s;
        orow0 = (long long)qs * p.sO.s;
    } else {
        SQ = p.SQ;
        SK = p.SK;

        qrow0 = (long long)z * p.sQ.b;
        krow0 = (long long)z * p.sK.b;
        orow0 = (long long)z * p.sO.b;
    }

    const int qb = blockIdx.x * BR;

    if (qb >= SQ)
        return;

    const int delta = SK - SQ;
    const float scale2 = p.scale * LOG2E;

    constexpr int QS_B = HH * BR * 4;
    constexpr int XS_B = HH * BC * 4;
    constexpr int PS_B = BC * BR * 2;

    static_assert(QS_B == 40960);
    static_assert(XS_B == 16384);
    static_assert(PS_B == 5120);
    static_assert(QS_B + XS_B + PS_B == 62464);
    static_assert(QS_B + XS_B + PS_B <= 64 * 1024);

    __shared__ char pool[QS_B + XS_B + PS_B];

    half2_t* Qs = (half2_t*)pool;

    // K and V deliberately occupy the same LDS scratch region.
    half2_t* Xs = (half2_t*)(pool + QS_B);
    half2_t* Ks = Xs;
    half2_t* Vs = Xs;

    // P must survive while V occupies Xs, so it has dedicated LDS.
    half2_t* Pp = (half2_t*)(pool + QS_B + XS_B);

    #define PH80_QSA(dh,q) \
        Qs[(dh) * BR + (q)]

    #define PH80_KSA(dh,kc) \
        Ks[(dh) * BC + ((kc) ^ ((dh) & 31))]

    #define PH80_VSA(kh,d) \
        Vs[(kh) * HD + ((d) ^ (((d) >> 4) & 7))]

    #define PH80_PPA(kh,q) \
        Pp[(kh) * BR + (q)]

    const _Float16* Qb =
        (const _Float16*)p.Q +
        qrow0 +
        (long long)h * p.sQ.h;

    const _Float16* Kb =
        (const _Float16*)p.K +
        krow0 +
        (long long)kvh * p.sK.h;

    const _Float16* Vb =
        (const _Float16*)p.V +
        krow0 +
        (long long)kvh * p.sV.h;

    // Q -> LDS once and remain resident.
    for (int e = t; e < BR * (HD / 8); e += 256) {
        const int dg = e % (HD / 8);
        const int q  = e / (HD / 8);
        const int sq = qb + q;

        half8_t a =
            sq < SQ
            ? *(const half8_t*)(
                Qb +
                (long long)sq * p.sQ.s +
                dg * 8
              )
            : half8_t{0,0,0,0,0,0,0,0};

        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            PH80_QSA(dg * 4 + m, q) =
                half2_t{a[2*m], a[2*m + 1]};
        }
    }

    float acco[QPT][DPT];
    float rm[QPT];
    float rl[QPT];

    #pragma unroll
    for (int i = 0; i < QPT; ++i) {
        rm[i] = -1e30f;
        rl[i] = 0.0f;

        #pragma unroll
        for (int e = 0; e < DPT; ++e)
            acco[i][e] = 0.0f;
    }

    __syncthreads();

    int kmax = qb + BR - 1 + delta;

    if (kmax >= SK)
        kmax = SK - 1;

    for (int kb = 0; kb <= kmax; kb += BC) {
        const bool diag =
            (kb + BC - 1) >= (qb + delta);

        // Previous GEMM2 must finish before Xs changes from V back to K.
        __syncthreads();

        // ---------------- phase A: K -> Xs ----------------
        constexpr int NK =
            (BC * (HD / 8) + 255) / 256;

        #pragma unroll
        for (int it = 0; it < NK; ++it) {
            const int e = t + it * 256;

            if (e < BC * (HD / 8)) {
                const int dg = e % (HD / 8);
                const int kc = e / (HD / 8);
                const int sk = kb + kc;

                half8_t a =
                    sk < SK
                    ? *(const half8_t*)(
                        Kb +
                        (long long)sk * p.sK.s +
                        dg * 8
                      )
                    : half8_t{0,0,0,0,0,0,0,0};

                #pragma unroll
                for (int m = 0; m < 4; ++m) {
                    PH80_KSA(dg * 4 + m, kc) =
                        half2_t{a[2*m], a[2*m + 1]};
                }
            }
        }

        __syncthreads();

        // ---------------- GEMM1: Q K^T ----------------
        float accs[QPT][KPT];

        #pragma unroll
        for (int i = 0; i < QPT; ++i)
            #pragma unroll
            for (int j = 0; j < KPT; ++j)
                accs[i][j] = 0.0f;

        #pragma unroll 4
        for (int dh = 0; dh < HH; ++dh) {
            half2_t kv[KPT];
            half2_t qv[QPT];

            #pragma unroll
            for (int j = 0; j < KPT; ++j) {
                kv[j] =
                    PH80_KSA(
                        dh,
                        (j >> 1) * 32 +
                        2 * tx +
                        (j & 1)
                    );
            }

            #pragma unroll
            for (int i = 0; i < QPT; ++i) {
                qv[i] =
                    PH80_QSA(
                        dh,
                        ty * QPT + i
                    );
            }

            #pragma unroll
            for (int i = 0; i < QPT; ++i)
                #pragma unroll
                for (int j = 0; j < KPT; ++j) {
                    accs[i][j] =
                        __builtin_amdgcn_fdot2(
                            qv[i],
                            kv[j],
                            accs[i][j],
                            false
                        );
                }
        }

        // ---------------- online softmax + dedicated P ----------------
        #pragma unroll
        for (int i = 0; i < QPT; ++i) {
            const int qq = qb + ty * QPT + i;

            float sj[KPT];
            float lmax = -1e30f;

            #pragma unroll
            for (int j = 0; j < KPT; ++j) {
                const int key =
                    kb +
                    (j >> 1) * 32 +
                    2 * tx +
                    (j & 1);

                float sv = accs[i][j] * scale2;

                if (diag && key > qq + delta)
                    sv = -1e30f;

                sj[j] = sv;
                lmax = fmaxf(lmax, sv);
            }

            lmax = red16_max(lmax);

            const float mnew =
                fmaxf(rm[i], lmax);

            const float corr =
                (mnew > rm[i])
                ? __builtin_amdgcn_exp2f(rm[i] - mnew)
                : 1.0f;

            float lsum = 0.0f;

            #pragma unroll
            for (int j = 0; j < KPT; ++j) {
                const float pj =
                    __builtin_amdgcn_exp2f(sj[j] - mnew);

                sj[j] = pj;
                lsum += pj;
            }

            lsum = red16_sum(lsum);

            rl[i] = rl[i] * corr + lsum;
            rm[i] = mnew;

            if (corr != 1.0f) {
                #pragma unroll
                for (int e = 0; e < DPT; ++e)
                    acco[i][e] *= corr;
            }

            #pragma unroll
            for (int j = 0; j < KPT; j += 2) {
                PH80_PPA(
                    (j >> 1) * 16 + tx,
                    ty * QPT + i
                ) =
                    half2_t{
                        (_Float16)sj[j],
                        (_Float16)sj[j + 1]
                    };
            }
        }

        // All K reads and P writes complete before Xs becomes V.
        __syncthreads();

        // ---------------- phase B: V -> same Xs ----------------
        constexpr int NV =
            (BCH * (HD / 8) + 255) / 256;

        #pragma unroll
        for (int it = 0; it < NV; ++it) {
            const int e = t + it * 256;

            if (e < BCH * (HD / 8)) {
                const int dg = e % (HD / 8);
                const int kh = e / (HD / 8);

                const int s0 = kb + 2 * kh;
                const int s1 = s0 + 1;
                const int d0 = dg * 8;

                half8_t a =
                    s0 < SK
                    ? *(const half8_t*)(
                        Vb +
                        (long long)s0 * p.sV.s +
                        d0
                      )
                    : half8_t{0,0,0,0,0,0,0,0};

                half8_t b =
                    s1 < SK
                    ? *(const half8_t*)(
                        Vb +
                        (long long)s1 * p.sV.s +
                        d0
                      )
                    : half8_t{0,0,0,0,0,0,0,0};

                #pragma unroll
                for (int m = 0; m < 8; ++m) {
                    PH80_VSA(kh, d0 + m) =
                        half2_t{a[m], b[m]};
                }
            }
        }

        __syncthreads();

        // ---------------- GEMM2: P V ----------------
        #pragma unroll 4
        for (int kh = 0; kh < BCH; ++kh) {
            half2_t pv[QPT];
            half2_t vv[DPT];

            #pragma unroll
            for (int i = 0; i < QPT; ++i) {
                pv[i] =
                    PH80_PPA(
                        kh,
                        ty * QPT + i
                    );
            }

            #pragma unroll
            for (int e = 0; e < DPT; ++e) {
                vv[e] =
                    PH80_VSA(
                        kh,
                        e * 16 + tx
                    );
            }

            #pragma unroll
            for (int i = 0; i < QPT; ++i)
                #pragma unroll
                for (int e = 0; e < DPT; ++e) {
                    acco[i][e] =
                        __builtin_amdgcn_fdot2(
                            pv[i],
                            vv[e],
                            acco[i][e],
                            false
                        );
                }
        }
    }

    _Float16* Ob =
        (_Float16*)p.O +
        orow0 +
        (long long)h * p.sO.h;

    #pragma unroll
    for (int i = 0; i < QPT; ++i) {
        const int qq = qb + ty * QPT + i;

        if (qq >= SQ)
            continue;

        // Match the generic path for rows with no legal keys.
        const bool empty = rm[i] <= -1e29f;

        const float inv =
            (rl[i] > 0.0f && !empty)
            ? 1.0f / rl[i]
            : 0.0f;

        #pragma unroll
        for (int e = 0; e < DPT; ++e) {
            const int d = e * 16 + tx;

            Ob[
                (long long)qq * p.sO.s + d
            ] =
                (_Float16)(
                    acco[i][e] * inv
                );
        }
    }

    #undef PH80_QSA
    #undef PH80_KSA
    #undef PH80_VSA
    #undef PH80_PPA
}

// ======================= correct fp32 path (no fdot2) =======================
// Same tiling, MAC via fmaf over float LDS. Only for 1e-5 accuracy; not a performance target.
template<int HD,int BR,int BC,bool CAUSAL>
__global__ __launch_bounds__(256)
void fa_forward_fp32_k(FwdParams p)
{
    constexpr int QPT=BR/16, KPT=BC/16, DPT=HD/16;
    const int t=threadIdx.x, tx=t&15, ty=t>>4;
    const int z=blockIdx.z, h=blockIdx.y, kvh=h/p.gqa;
    int SQ,SK; long long qrow0,krow0,orow0;
    if(p.cu_q){ int qs=p.cu_q[z], ks=p.cu_k[z]; SQ=p.cu_q[z+1]-qs; SK=p.cu_k[z+1]-ks;
        if(p.seqused_q){ int u=p.seqused_q[z]; if(u<SQ) SQ=u; }
        if(p.seqused_k){ int u=p.seqused_k[z]; if(u<SK) SK=u; }
        qrow0=(long long)qs*p.sQ.s; krow0=(long long)ks*p.sK.s; orow0=(long long)qs*p.sO.s; }
    else { SQ=p.SQ; SK=p.SK; qrow0=(long long)z*p.sQ.b; krow0=(long long)z*p.sK.b; orow0=(long long)z*p.sO.b; }
    const int qb=blockIdx.x*BR; if(qb>=SQ) return;
    const int delta=SK-SQ; const float scale2=p.scale*1.442695041f;
    __shared__ float Qs[BR*HD], Ks[BC*HD], Vs[BC*HD]; __shared__ float Ps[BC*BR];
    const float* Qg=(const float*)p.Q+qrow0+(long long)h*p.sQ.h;
    for(int e=t;e<BR*HD;e+=256){ int q=e/HD,d=e%HD; int sq=qb+q; Qs[q*HD+d]=(sq<SQ)?Qg[(long long)sq*p.sQ.s+d]:0.0f; }
    float acco[QPT][DPT], rm[QPT], rl[QPT];
    #pragma unroll
    for(int i=0;i<QPT;i++){ rm[i]=-1e30f; rl[i]=0.0f; for(int e=0;e<DPT;e++) acco[i][e]=0.0f; }
    __syncthreads();
    int kmax = CAUSAL? (( (qb+BR-1)+delta < SK-1)? (qb+BR-1)+delta : SK-1) : SK-1;
    for(int kb=0;kb<=kmax;kb+=BC){
        const bool diag=CAUSAL&&((kb+BC-1)>=qb+delta);
        __syncthreads();
        const float* Kg=(const float*)p.K+krow0+(long long)kvh*p.sK.h;
        const float* Vg=(const float*)p.V+krow0+(long long)kvh*p.sV.h;
        for(int e=t;e<BC*HD;e+=256){ int kc=e/HD,d=e%HD; int sk=kb+kc;
            Ks[kc*HD+d]=(sk<SK)?Kg[(long long)sk*p.sK.s+d]:0.0f; Vs[kc*HD+d]=(sk<SK)?Vg[(long long)sk*p.sV.s+d]:0.0f; }
        __syncthreads();
        float accs[QPT][KPT];
        #pragma unroll
        for(int i=0;i<QPT;i++) for(int j=0;j<KPT;j++) accs[i][j]=0.0f;
        for(int d=0;d<HD;d++){ float qv[QPT],kv[KPT];
            #pragma unroll
            for(int i=0;i<QPT;i++) qv[i]=Qs[(ty*QPT+i)*HD+d];
            #pragma unroll
            for(int j=0;j<KPT;j++) kv[j]=Ks[(tx*KPT+j)*HD+d];
            #pragma unroll
            for(int i=0;i<QPT;i++) for(int j=0;j<KPT;j++) accs[i][j]=fmaf(qv[i],kv[j],accs[i][j]); }
        __syncthreads();
        #pragma unroll
        for(int i=0;i<QPT;i++){ int qq=qb+ty*QPT+i; float sj[KPT],lmax=-1e30f;
            float aslope=p.alibi_slopes?p.alibi_slopes[h]:0.0f;
            #pragma unroll
            for(int j=0;j<KPT;j++){ int key=kb+tx*KPT+j; float raw=accs[i][j]; float sv;
                if(p.softcap>0.0f){ float x=raw*p.scale; sv=p.softcap*tanhf(x/p.softcap)*1.442695041f; } else sv=raw*scale2;
                if(p.alibi_slopes) sv+=aslope*(float)(key-qq)*1.442695041f;
                bool masked=(key>=SK); if(CAUSAL&&diag&&key>qq+delta) masked=true;
                if(p.win_left>=0  && key < qq+delta-p.win_left ) masked=true;
                if(p.win_right>=0 && key > qq+delta+p.win_right) masked=true;
                if(p.attn_chunk>0){ int pp=qq+delta; if(pp<0) pp=0;
                    int cs=(pp/p.attn_chunk)*p.attn_chunk; if(key<cs||key>=cs+p.attn_chunk) masked=true; }
                if(masked) sv=-1e30f; sj[j]=sv; lmax=fmaxf(lmax,sv); }
            lmax=fmaxf(lmax,__shfl_xor(lmax,1,16)); lmax=fmaxf(lmax,__shfl_xor(lmax,2,16));
            lmax=fmaxf(lmax,__shfl_xor(lmax,4,16)); lmax=fmaxf(lmax,__shfl_xor(lmax,8,16));
            float mnew=fmaxf(rm[i],lmax); float corr=(mnew>rm[i])?__builtin_amdgcn_exp2f(rm[i]-mnew):1.0f; float lsum=0;
            #pragma unroll
            for(int j=0;j<KPT;j++){ float pj=__builtin_amdgcn_exp2f(sj[j]-mnew); sj[j]=pj; lsum+=pj; }
            lsum+=__shfl_xor(lsum,1,16); lsum+=__shfl_xor(lsum,2,16); lsum+=__shfl_xor(lsum,4,16); lsum+=__shfl_xor(lsum,8,16);
            rl[i]=rl[i]*corr+lsum; rm[i]=mnew;
            if(corr!=1.0f){ for(int e=0;e<DPT;e++) acco[i][e]*=corr; }
            #pragma unroll
            for(int j=0;j<KPT;j++) Ps[(tx*KPT+j)*BR+ty*QPT+i]=sj[j]; }
        __syncthreads();
        for(int kc=0;kc<BC;kc++){ float pv[QPT];
            #pragma unroll
            for(int i=0;i<QPT;i++) pv[i]=Ps[kc*BR+ty*QPT+i];
            #pragma unroll
            for(int e=0;e<DPT;e++){ float v=Vs[kc*HD+tx*DPT+e];
                #pragma unroll
                for(int i=0;i<QPT;i++) acco[i][e]=fmaf(pv[i],v,acco[i][e]); } }
        __syncthreads();
    }
    float* Ob=(float*)p.O+orow0+(long long)h*p.sO.h;
    #pragma unroll
    for(int i=0;i<QPT;i++){ int qq=qb+ty*QPT+i; if(qq>=SQ) continue;
        const bool empty = rm[i] <= -1e29f;             // all keys in the row are masked
        float inv=(rl[i]>0&&!empty)?1.0f/rl[i]:0.0f;
        #pragma unroll
        for(int e=0;e<DPT;e++) Ob[(long long)qq*p.sO.s+tx*DPT+e]=acco[i][e]*inv;
        if(p.LSE&&tx==0) p.LSE[(size_t)h*SQ+qq]= (rl[i]>0&&!empty)?(rm[i]*0.693147181f+logf(rl[i])):-INFINITY; }
}

// fp32 D512 (correctness-only, 1e-5): same D-split design as fa_forward_d512_k,
// with the original fp32 kernel's simpler thread mapping (score per thread,
// shfl_xor row reductions, output dims tx*(HD/16)+e). LDS: Qs 32KB +
// K/P/V-region 17.25KB = 49.25KB <= 64KB.
template<bool CAUSAL>
__global__ __launch_bounds__(256)
void fa_forward_fp32_d512_k(FwdParams p)
{
    constexpr int HD=512, BR=16, BC=16, DCH=256, HC=HD/DCH, DPT=HD/16;
    const int t=threadIdx.x, tx=t&15, ty=t>>4;
    const int z=blockIdx.z, h=blockIdx.y, kvh=h/p.gqa;
    int SQ,SK; long long qrow0,krow0,orow0;
    if(p.cu_q){ int qs=p.cu_q[z], ks=p.cu_k[z]; SQ=p.cu_q[z+1]-qs; SK=p.cu_k[z+1]-ks;
        if(p.seqused_q){ int u=p.seqused_q[z]; if(u<SQ) SQ=u; }
        if(p.seqused_k){ int u=p.seqused_k[z]; if(u<SK) SK=u; }
        qrow0=(long long)qs*p.sQ.s; krow0=(long long)ks*p.sK.s; orow0=(long long)qs*p.sO.s; }
    else { SQ=p.SQ; SK=p.SK; qrow0=(long long)z*p.sQ.b; krow0=(long long)z*p.sK.b; orow0=(long long)z*p.sO.b; }
    const int qb=blockIdx.x*BR; if(qb>=SQ) return;
    const int delta=SK-SQ; const float scale2=p.scale*1.442695041f;
    __shared__ float Qs[BR*HD];
    __shared__ char kpool[(BC*DCH+BC*BR)*4];   // K chunk (16KB) | P (1KB) + V chunk (16KB)
    float* Kc=(float*)kpool;                   // Kc[kc*DCH+d] (16KB)
    float* Ps=(float*)kpool;                   // Ps[kc*BR+ty] (1KB), offset 0, K dead
    float* Vc=(float*)kpool+BC*BR;             // Vc[kh*DCH+d] (16KB), offset BC*BR floats
    const float* Qg=(const float*)p.Q+qrow0+(long long)h*p.sQ.h;
    for(int e=t;e<BR*HD;e+=256){ int q=e/HD,d=e%HD; int sq=qb+q; Qs[q*HD+d]=(sq<SQ)?Qg[(long long)sq*p.sQ.s+d]:0.0f; }
    float acco[DPT], rm=-1e30f, rl=0.0f;
    for(int e=0;e<DPT;e++) acco[e]=0.0f;
    __syncthreads();
    int kmax = CAUSAL? (( (qb+BR-1)+delta < SK-1)? (qb+BR-1)+delta : SK-1) : SK-1;
    for(int kb=0;kb<=kmax;kb+=BC){
        const bool diag=CAUSAL&&((kb+BC-1)>=qb+delta);
        __syncthreads();
        // K phase: two 256-dim chunks; score = full 512-dim dot, accumulated in the register
        const float* Kg=(const float*)p.K+krow0+(long long)kvh*p.sK.h;
        float accs=0.0f;
        for(int c=0;c<HC;c++){
            const int d0=c*DCH;
            for(int e=t;e<BC*DCH;e+=256){ int kc=e/DCH,d=e%DCH; int sk=kb+kc;
                Kc[kc*DCH+d]=(sk<SK)?Kg[(long long)sk*p.sK.s+d0+d]:0.0f; }
            __syncthreads();
            for(int d=0;d<DCH;d++){ float qv=Qs[ty*HD+d0+d], kv=Kc[tx*DCH+d]; accs=fmaf(qv,kv,accs); }
            __syncthreads();
        }
        // softmax over the FULL score (one key per thread; 16 keys across the lane's 16 lanes)
        {
            int qq=qb+ty, key=kb+tx; float raw=accs; float sv;
            if(p.softcap>0.0f){ float x=raw*p.scale; sv=p.softcap*tanhf(x/p.softcap)*1.442695041f; } else sv=raw*scale2;
            float aslope=p.alibi_slopes?p.alibi_slopes[h]:0.0f;
            if(p.alibi_slopes) sv+=aslope*(float)(key-qq)*1.442695041f;
            bool masked=(key>=SK); if(CAUSAL&&diag&&key>qq+delta) masked=true;
            if(p.win_left>=0  && key < qq+delta-p.win_left ) masked=true;
            if(p.win_right>=0 && key > qq+delta+p.win_right) masked=true;
            if(p.attn_chunk>0){ int pp=qq+delta; if(pp<0) pp=0; int cs=(pp/p.attn_chunk)*p.attn_chunk;
                if(key<cs||key>=cs+p.attn_chunk) masked=true; }
            if(masked) sv=-1e30f;
            float lmax=sv;
            lmax=fmaxf(lmax,__shfl_xor(lmax,1,16)); lmax=fmaxf(lmax,__shfl_xor(lmax,2,16));
            lmax=fmaxf(lmax,__shfl_xor(lmax,4,16)); lmax=fmaxf(lmax,__shfl_xor(lmax,8,16));
            float mnew=fmaxf(rm,lmax); float corr=(mnew>rm)?__builtin_amdgcn_exp2f(rm-mnew):1.0f;
            float pj=__builtin_amdgcn_exp2f(sv-mnew); float lsum=pj;
            lsum+=__shfl_xor(lsum,1,16); lsum+=__shfl_xor(lsum,2,16);
            lsum+=__shfl_xor(lsum,4,16); lsum+=__shfl_xor(lsum,8,16);
            rl=rl*corr+lsum; rm=mnew;
            if(corr!=1.0f){ for(int e=0;e<DPT;e++) acco[e]*=corr; }
            Ps[tx*BR+ty]=pj;
        }
        __syncthreads();
        // V phase: two 256-dim chunks (K dead); output dims tx*DPT+e, chunk c owns tx in [c*8, c*8+7]
        const float* Vg=(const float*)p.V+krow0+(long long)kvh*p.sV.h;
        for(int c=0;c<HC;c++){
            const int d0=c*DCH;
            for(int e=t;e<BC*DCH;e+=256){ int kh=e/DCH,d=e%DCH; int s0=kb+kh;
                Vc[kh*DCH+d]=(s0<SK)?Vg[(long long)s0*p.sV.s+d0+d]:0.0f; }
            __syncthreads();
            if((tx/8)==c){
                for(int kc=0;kc<BC;kc++){ float pv=Ps[kc*BR+ty];
                    for(int e=0;e<DPT;e++) acco[e]=fmaf(pv,Vc[kc*DCH+(tx-c*8)*DPT+e],acco[e]); }
            }
            __syncthreads();
        }
    }
    float* Ob=(float*)p.O+orow0+(long long)h*p.sO.h;
    for(int e=0;e<DPT;e++){ int qq=qb+ty; if(qq>=SQ) continue;
        const bool empty = rm <= -1e29f;
        float inv=(rl>0&&!empty)?1.0f/rl:0.0f;
        Ob[(long long)qq*p.sO.s+tx*DPT+e]=acco[e]*inv; }
    if(p.LSE&&tx==0){ int qq=qb+ty; if(qq<SQ){
        const bool empty = rm <= -1e29f;
        p.LSE[(size_t)h*SQ+qq]= (rl>0.0f&&!empty)?(rm*0.693147181f+logf(rl)):-INFINITY; } }
}



// ===================================================================
// D512 forward — D-split (Gemma4 full-attention head_dim 512).
// A 512-dim K+V does NOT fit in 64KB LDS with a good tile (BC>=32 needs
// 32KB K + 32KB V + 32KB Q = 96KB), and the score Q.K^T must sum ALL 512
// dims BEFORE softmax (a per-chunk softmax is mathematically wrong).
// Design: Q resident (full 512); K streamed in two 256-dim chunks with the
// score accumulated across chunks in registers; ONE full softmax; then V
// streamed in two 256-dim chunks (dead-K LDS region reused) with the output
// accumulated. Reuses the proven D256 BR=32,BC=32 thread mapping (QPT=KPT=2).
// LDS: Qs(256 half2 * 32 rows * 4B)=32KB + K-region(16KB)=48KB <= 64KB.
//   After the score, K is dead; the K-region then holds P (512 half2=2KB) at
//   offset 0 and one 256-dim V chunk (4096 half2=8KB) at offset 512 (no overlap).
// This is a SEPARATE kernel; the D16-D256 fa_forward_k is untouched.
// ===================================================================
template<class DT,int HD,int BR,int BC,int DCH,bool CAUSAL>
__global__ __launch_bounds__(256)
void fa_forward_d512_k(FwdParams p)
{
    typedef typename DT::T T;
    constexpr bool SCALED = DT::is_bf16;
    constexpr int QPT=BR/16, KPT=BC/16, DH=HD/2, BCH=BC/2, DPT=DCH/16, HC=HD/DCH;
    constexpr float LOG2E=1.442695041f;
    const int t=threadIdx.x, tx=t&15, ty=t>>4;
    const int z=blockIdx.z, hg=blockIdx.y;
    const int h0=hg, kvh=h0/p.gqa;                 // G=1: one query head per workgroup
    int SQ,SK; long long qrow0,krow0,orow0;
    if(p.cu_q){ int qs=p.cu_q[z], ks=p.cu_k[z]; SQ=p.cu_q[z+1]-qs; SK=p.cu_k[z+1]-ks;
        if(p.seqused_q){ int u=p.seqused_q[z]; if(u<SQ) SQ=u; }
        if(p.seqused_k){ int u=p.seqused_k[z]; if(u<SK) SK=u; }
        qrow0=(long long)qs*p.sQ.s; krow0=(long long)ks*p.sK.s; orow0=(long long)qs*p.sO.s; }
    else { SQ=p.SQ; SK=p.SK; qrow0=(long long)z*p.sQ.b; krow0=(long long)z*p.sK.b; orow0=(long long)z*p.sO.b; }
    const int qb=blockIdx.x*BR; if(qb>=SQ) return;
    const int delta=SK-SQ;
    const float scale2=p.scale*LOG2E;
    const float scq = SCALED? p.sc_q : 1.0f, sck = SCALED? p.sc_k : 1.0f, scv = SCALED? p.sc_v : 1.0f;
    const float uqk = SCALED? 1.0f/(scq*sck) : 1.0f;
    const float uv  = SCALED? 1.0f/scv       : 1.0f;

    // LDS: Qs (full 512) + K-region (one 256-dim K-chunk; after K dies: P + one V-chunk).
    constexpr int QS_B = DH*BR*4;                          // 256*32 half2 = 32KB (Q, full 512)
    constexpr int KR_B = (DCH/2*BC + BR*BC/2)*4;           // max(K-chunk 4096, P 512 + V-chunk 4096) half2 = 18KB
    __shared__ char pool[QS_B+KR_B];                       // 50KB <= 64KB gfx1030 ceiling
    half2_t* Qs=(half2_t*)pool;
    half2_t* Ks=(half2_t*)(pool+QS_B);     // K chunk (kh-major? no: dk-major) during K phase
    half2_t* Ps=Ks;                        // P: 512 half2 (2KB), offset 0, K dead
    half2_t* Vs=Ks+512;                    // V chunk: 4096 half2 (8KB), offset 512
    #define DQSA(dh,q)   Qs[(dh)*BR+(q)]
    #define DKSA(dh,kc)  Ks[(dh)*BC+((kc)^((dh)&31))]
    #define DVSA(kh,d)   Vs[(kh)*DCH+((d)^(((d)>>4)&7))]
    #define DPSA(kc,q)   Ps[(kc)*BR+(q)]

    auto load8=[&](const T* pt,bool valid,float sc,_Float16 o[8]){
        if constexpr(SCALED){
            uint4 raw=valid?*(const uint4*)pt:uint4{0,0,0,0}; const uint16_t* w=(const uint16_t*)&raw;
            #pragma unroll
            for(int m=0;m<8;m++){ uint32_t b=((uint32_t)w[m])<<16; float f; __builtin_memcpy(&f,&b,4); o[m]=(_Float16)(f*sc); }
        } else {
            half8_t a=valid?*(const half8_t*)pt:half8_t{0,0,0,0,0,0,0,0};
            #pragma unroll
            for(int m=0;m<8;m++) o[m]=a[m];
        }
    };

    // ---- load Q (resident, full 512) ----
    const T* Qb=(const T*)p.Q + qrow0 + (long long)h0*p.sQ.h;
    for(int e=t; e<BR*(HD/8); e+=256){
        int dg=e%(HD/8), q=e/(HD/8); int sq=qb+q; _Float16 o[8];
        load8(Qb+(long long)sq*p.sQ.s+dg*8, sq<SQ, scq, o);
        #pragma unroll
        for(int m=0;m<4;m++) DQSA(dg*4+m,q)=half2_t{o[2*m],o[2*m+1]};
    }
    float acco[QPT][HD/16]; float rm[QPT], rl[QPT];
    #pragma unroll
    for(int i=0;i<QPT;i++){ rm[i]=-1e30f; rl[i]=0.0f;
        #pragma unroll
        for(int e=0;e<HD/16;e++) acco[i][e]=0.0f; }
    __syncthreads();

    int kmax = CAUSAL? (((qb+BR-1)+delta < SK-1)?(qb+BR-1)+delta:SK-1) : SK-1;
    int kb0=0;
    if(p.win_left>=0){ int lo=qb+delta-p.win_left; if(lo<0) lo=0; kb0=(lo/BC)*BC; }
    if(p.win_right>=0){ int hi=qb+BR-1+delta+p.win_right; if(hi<kmax) kmax=hi; }
    const float aslope = p.alibi_slopes? p.alibi_slopes[h0] : 0.0f;

    for(int kb=kb0; kb<=kmax; kb+=BC){
        const bool diag = CAUSAL && ((kb+BC-1)>=qb+delta);
        __syncthreads();
        // ---- K phase: two 256-dim chunks; accumulate the FULL score (all 512 dims) ----
        float sc[QPT][KPT];
        #pragma unroll
        for(int i=0;i<QPT;i++)
            #pragma unroll
            for(int j=0;j<KPT;j++) sc[i][j]=0.0f;
        #pragma unroll
        for(int c=0;c<HC;c++){
            const int d0=c*DCH;
            constexpr int NK=(BC*(DCH/8)+255)/256;         // 4 half8 rows per 256-dim K-chunk
            #pragma unroll
            for(int it=0; it<NK; it++){ int e=t+it*256; if(e<BC*(DCH/8)){
                int dg=e%(DCH/8), kc=e/(DCH/8); int sk=kb+kc; _Float16 o[8];
                load8((const T*)p.K+krow0+(long long)kvh*p.sK.h+(long long)sk*p.sK.s+(d0+dg*8), sk<SK, sck, o);
                #pragma unroll
                for(int m=0;m<4;m++) DKSA(dg*4+m,kc)=half2_t{o[2*m],o[2*m+1]}; } }
            __syncthreads();
            float accs[QPT][KPT];
            #pragma unroll
            for(int i=0;i<QPT;i++)
                #pragma unroll
                for(int j=0;j<KPT;j++) accs[i][j]=0.0f;
            #pragma unroll 4
            for(int dh=0; dh<DCH/2; dh++){
                half2_t kv[KPT];
                #pragma unroll
                for(int j=0;j<KPT;j++) kv[j]=DKSA(dh,(j>>1)*32+2*tx+(j&1));
                #pragma unroll
                for(int i=0;i<QPT;i++){ half2_t qv=DQSA(c*(DCH/2)+dh, ty*QPT+i);
                    #pragma unroll
                    for(int j=0;j<KPT;j++) accs[i][j]=__builtin_amdgcn_fdot2(qv,kv[j],accs[i][j],false); }
            }
            #pragma unroll
            for(int i=0;i<QPT;i++)
                #pragma unroll
                for(int j=0;j<KPT;j++) sc[i][j]+=accs[i][j];   // accumulate across D-chunks
            __syncthreads();
        }
        // ---- softmax over the FULL score (512-dim dot); K dead -> Ps/Vs region ----
        __syncthreads();
        #pragma unroll
        for(int i=0;i<QPT;i++){
            const int qq=qb+ty*QPT+i;
            float sj[KPT], lmax=-1e30f;
            #pragma unroll
            for(int j=0;j<KPT;j++){
                const int key=kb+(j>>1)*32+2*tx+(j&1);
                float a=sc[i][j]; if constexpr(SCALED) a*=uqk;
                float sv = (p.softcap>0.0f)? p.softcap*tanhf(a*p.scale/p.softcap)*LOG2E : a*scale2;
                if(p.alibi_slopes) sv+=aslope*(float)(key-qq)*LOG2E;
                bool masked=false;
                if constexpr(CAUSAL){ if(diag && key>qq+delta) masked=true; }
                if(p.win_left>=0  && key < qq+delta-p.win_left ) masked=true;
                if(p.win_right>=0 && key > qq+delta+p.win_right) masked=true;
                if(p.attn_chunk>0){ int pp=qq+delta; if(pp<0) pp=0; int cs=(pp/p.attn_chunk)*p.attn_chunk;
                    if(key<cs||key>=cs+p.attn_chunk) masked=true; }
                if(key>=SK) masked=true;
                if(masked) sv=-1e30f;
                sj[j]=sv; lmax=fmaxf(lmax,sv);
            }
            lmax=red16_max(lmax);
            float mnew=fmaxf(rm[i],lmax);
            float corr = (mnew>rm[i])? __builtin_amdgcn_exp2f(rm[i]-mnew) : 1.0f;
            float lsum=0.0f;
            #pragma unroll
            for(int j=0;j<KPT;j++){ float pj=__builtin_amdgcn_exp2f(sj[j]-mnew); sj[j]=pj; lsum+=pj; }
            lsum=red16_sum(lsum);
            rl[i]=rl[i]*corr+lsum; rm[i]=mnew;
            if(corr!=1.0f){
                #pragma unroll
                for(int e=0;e<HD/16;e++) acco[i][e]*=corr;
            }
            #pragma unroll
            for(int j=0;j<KPT;j+=2){
                float a = SCALED? sj[j]*uv   : sj[j];
                float b = SCALED? sj[j+1]*uv : sj[j+1];
                DPSA((j>>1)*16+tx, ty*QPT+i)=half2_t{(_Float16)a,(_Float16)b};
            }
        }
        __syncthreads();
        // ---- V phase: two 256-dim chunks; accumulate O over the full 512 ----
        #pragma unroll
        for(int c=0;c<HC;c++){
            const int d0=c*DCH;
            #pragma unroll
            for(int it=0; it<2; it++){ int e=t+it*256; if(e<BCH*(DCH/8)){
                int dg=e%(DCH/8), kh=e/(DCH/8); int s0=kb+2*kh,s1=s0+1; int dg0=dg*8; _Float16 a[8],b[8];
                load8((const T*)p.V+krow0+(long long)kvh*p.sV.h+(long long)s0*p.sV.s+(d0+dg0), s0<SK, scv, a);
                load8((const T*)p.V+krow0+(long long)kvh*p.sV.h+(long long)s1*p.sV.s+(d0+dg0), s1<SK, scv, b);
                #pragma unroll
                for(int m=0;m<8;m++) DVSA(kh,dg0+m)=half2_t{a[m],b[m]}; } }
            __syncthreads();
            #pragma unroll 4
            for(int kh=0; kh<BCH; kh++){
                half2_t pv[QPT];
                #pragma unroll
                for(int i=0;i<QPT;i++) pv[i]=DPSA(kh, ty*QPT+i);   // one b32
                #pragma unroll
                for(int e=0;e<DPT;e++){
                    half2_t vv=DVSA(kh,e*16+tx);
                    #pragma unroll
                    for(int i=0;i<QPT;i++) acco[i][c*DPT+e]=__builtin_amdgcn_fdot2(pv[i],vv,acco[i][c*DPT+e],false);
                }
            }
            __syncthreads();
        }
    }
    // ---- epilogue ----
    T* Ob=(T*)p.O+orow0+(long long)h0*p.sO.h;
    #pragma unroll
    for(int i=0;i<QPT;i++){ int qq=qb+ty*QPT+i; if(qq>=SQ) continue;
        const bool empty = rm[i] <= -1e29f;
        float inv = (rl[i]>0.0f && !empty)? 1.0f/rl[i] : 0.0f;
        #pragma unroll
        for(int e=0;e<HD/16;e++){ int d=e*16+tx; Ob[(long long)qq*p.sO.s+d]=(T)(acco[i][e]*inv); }
        if(p.LSE && tx==0){ float lse = (rl[i]>0.0f && !empty)? (rm[i]*0.693147181f + logf(rl[i])) : -INFINITY;
            p.LSE[(size_t)h0*SQ+qq]=lse; }
    }
    #undef DQSA
    #undef DKSA
    #undef DVSA
    #undef DPSA
}

// host launch for the D512 D-split kernel (BR=32, BC=32, DCH=256, G=1).
template<class DT>
static void launch_d512(FwdParams& p, hipStream_t st){
    dim3 grid((p.SQ+32-1)/32, p.NH, p.batch);
    if(p.causal) fa_forward_d512_k<DT,512,32,32,256,true ><<<grid,256,0,st>>>(p);
    else         fa_forward_d512_k<DT,512,32,32,256,false><<<grid,256,0,st>>>(p);
}

// ======================= launch dispatcher (host) =======================
enum FaDtype { FA_F16=0, FA_BF16=1, FA_F32=2 };

// gfx1030 config (fp16 AND bf16 share the LDS layout; bf16 converts inline on load):
//   HD<=128 -> BR=128, BC=64 (champion structure: maximum K/V reuse, memory ceiling).
//   HD=256  -> BR=64,  BC=32.
// CAUSAL and FEAT (softcap/alibi/window/dropout) are compile-time branches so the dense path has no overhead.
template<class DT,int HD,int BR,int BC>
static void launch_cfg(FwdParams& p, hipStream_t st){
    dim3 grid((p.SQ+BR-1)/BR, p.NH, p.batch);
    bool feat = (p.softcap>0.0f)||(p.alibi_slopes!=nullptr)||(p.dropout_p>0.0f)
                ||(p.attn_chunk>0)||(p.win_left>=0)
                // right boundary 0 is exactly causal, handled by the compile-time CAUSAL branch,
                // so it is not counted as a "feature" and the dense path does not pay extra
                ||(p.win_right>=0 && !(p.win_right==0 && p.causal));
    if(p.causal){ if(feat) fa_forward_k<DT,HD,BR,BC,1,true ,true ><<<grid,256,0,st>>>(p);
                  else      fa_forward_k<DT,HD,BR,BC,1,true ,false><<<grid,256,0,st>>>(p); }
    else        { if(feat) fa_forward_k<DT,HD,BR,BC,1,false,true ><<<grid,256,0,st>>>(p);
                  else      fa_forward_k<DT,HD,BR,BC,1,false,false><<<grid,256,0,st>>>(p); }
}

// public entry point for the torch wrapper (second agent)
extern "C" void fa_forward_launch(FwdParams p, int dtype, hipStream_t st){
    if(p.sc_q==0.0f) p.sc_q=1.0f; if(p.sc_k==0.0f) p.sc_k=1.0f; if(p.sc_v==0.0f) p.sc_v=1.0f;

    // Optimization 16: narrowly qualified gfx1030 Qwen3.8 TP1/TP2/TP4/TP8 D256 path.
    //
    // Preserve every existing feature and dtype by falling through to the
    // original dispatcher unless the call exactly matches the configuration
    // qualified by the phaseKV80 development gates.
    const bool phase80_d256_f16 =
        dtype == FA_F16 &&
        p.HD == 256 &&
        (
            (p.NH == 24 && p.gqa == 6) ||  // TP1
            (p.NH == 12 && p.gqa == 6) ||  // TP2
            (p.NH == 6  && p.gqa == 6) ||  // TP4
            (p.NH == 3  && p.gqa == 3)     // TP8
        ) &&
        p.batch == 1 &&
        p.causal &&
        p.softcap <= 0.0f &&
        p.alibi_slopes == nullptr &&
        p.dropout_p <= 0.0f &&
        p.attn_chunk <= 0 &&
        p.win_left < 0 &&
        (p.win_right < 0 || p.win_right == 0) &&
        p.seqused_q == nullptr &&
        p.seqused_k == nullptr &&
        p.LSE == nullptr &&
        ((p.cu_q == nullptr) == (p.cu_k == nullptr));

    if (phase80_d256_f16) {
        dim3 grid(
            (p.SQ + 79) / 80,
            p.NH,
            p.batch
        );

        fa_forward_phase80_d256_f16<<<grid,256,0,st>>>(p);
        return;
    }
    // Tile is selected by LDS size: at BR=128,BC=64 the pool uses HD*512 bytes (+16KB for Ps if
    // BR>HD) — fits in 64 KB through HD=128, but not beyond. For HD>=112 use BR=64,BC=32:
    // pool HD*256, Ps aliases Ks (BR<=HD), fitting through HD=256.
    #define BYHD(DTt) do{ switch(p.HD){ \
        case 16: launch_cfg<DTt, 16,128,64>(p,st);break; case 32: launch_cfg<DTt, 32,128,64>(p,st);break; \
        case 48: launch_cfg<DTt, 48,128,64>(p,st);break; case 64: launch_cfg<DTt, 64,128,64>(p,st);break; \
        case 80: launch_cfg<DTt, 80,128,64>(p,st);break; case 96: launch_cfg<DTt, 96,128,64>(p,st);break; \
        case 128:launch_cfg<DTt,128,128,64>(p,st);break; \
        case 112:launch_cfg<DTt,112, 64,32>(p,st);break; case 160:launch_cfg<DTt,160, 64,32>(p,st);break; \
        case 192:launch_cfg<DTt,192, 64,32>(p,st);break; case 224:launch_cfg<DTt,224, 64,32>(p,st);break; \
        case 256:launch_cfg<DTt,256, 64,32>(p,st);break; \
        /* D512 (Gemma4 full-attention): dedicated D-split kernel (fa_forward_d512_k):
           Q resident (full 512), K/V streamed in 256-dim chunks, score accumulated across
           chunks, one full softmax, output accumulated. BR=32, BC=32, LDS 50KB. */ \
        case 512:launch_d512<DTt>(p,st);break; \
        default: printf("fa_forward: head_dim=%d not instantiated\n",p.HD); abort(); } }while(0)
    if(dtype==FA_F16){ BYHD(DT_F16); }
    else if(dtype==FA_BF16){ BYHD(DT_BF16); }
    else { // fp32 (1e-5 correctness; small tiles to fit float LDS)
        #define DISP32(HD) do{ if(p.causal) fa_forward_fp32_k<HD,16,16,true ><<<dim3((p.SQ+15)/16,p.NH,p.batch),256,0,st>>>(p); \
                               else          fa_forward_fp32_k<HD,16,16,false><<<dim3((p.SQ+15)/16,p.NH,p.batch),256,0,st>>>(p); }while(0)
        switch(p.HD){ case 16:DISP32(16);break; case 32:DISP32(32);break; case 48:DISP32(48);break;
            case 64:DISP32(64);break; case 80:DISP32(80);break; case 96:DISP32(96);break;
            case 112:DISP32(112);break; case 128:DISP32(128);break; case 160:DISP32(160);break;
            case 192:DISP32(192);break; case 224:DISP32(224);break; case 256:DISP32(256);break;
            case 512: do{ if(p.causal) fa_forward_fp32_d512_k<true ><<<dim3((p.SQ+15)/16,p.NH,p.batch),256,0,st>>>(p); \
                              else      fa_forward_fp32_d512_k<false><<<dim3((p.SQ+15)/16,p.NH,p.batch),256,0,st>>>(p); }while(0);break;
            default: printf("fa_forward fp32: head_dim=%d not instantiated\n",p.HD); abort(); }
        #undef DISP32
    }
    #undef BYHD
}

#ifdef FA_STANDALONE
#include "fa_forward_gate.inc"
#endif
