#include "../package/csrc/alu_compiler_policy.h" // ALU_COMPILER_HARDENING_POLICY
// fa_decode.hip — FlashAttention decode with KV cache for gfx1030, inference path.
// Features: split-KV (parallelism over context -> hundreds of workgroups to fill the GPU), PAGED KV
// (block_table + page_size), on-the-fly rotary (neox / interleaved) for q, appending new_k/new_v to cache
// (with rotary on k), GQA, sliding window, head_dim {64,128,256}, dtype {fp16,bf16,fp32}.
// Decode is KV-bandwidth-bound (one query) => do math in fp32 after load conversion: accurate and simple,
// bf16 converts losslessly inline (7-bit mantissa fits in fp32). No quantization.
// Structural references: flash_split.hip (split-KV) + flash_decode.hip (online softmax by tiles).
// Build and run: bash tools/build_kernel_gates.sh
#include "fa_common.hpp"

enum FaDtype { FA_F16=0, FA_BF16=1, FA_F32=2 };

#include "fa_decode_params.hpp"
#include "fa_decode_index.hpp"
#include "fa_decode_schedule.hpp"

template<class DT> __device__ __forceinline__ float ld1(const typename DT::T* p,float sc){
    if constexpr(DT::is_bf16) return bf16_to_f(*p)*sc;
    else if constexpr(DT::is_fp32) return *p;
    else return (float)(*p)*sc;
}
// PSTRIDE = HD+2: [0]=m,[1]=l,[2..HD+1]=acc
// grid (allocated query rows, KV heads, nsplit), block 256.
// Redesign (old version fd_partial_v0 below is retained for test-bench comparison).
// What was wrong, based on measurements and code inspection:
//  1) page index was read from global memory for EVERY position by EVERY thread: during V this meant
//     128 threads x 256 positions = 32768 block_table reads per tile instead of 256.
//  2) only 128 of 256 threads accumulated V, and each loaded 2 bytes instead of 16.
//  3) reduction used two LDS trees, 18 __syncthreads per tile.
// Here: block_table is staged into LDS once, ALL 256 threads accumulate V with 16-byte loads
// (each owns 8 dimensions and a subgroup of positions; partial sums are reduced once
// at the very end), with DPP intrawave reduction + one barrier for 16 partials.
// G — number of query heads for one KV head served by a workgroup (KV reuse).
// Without it, each KV head is read gqa times by different workgroups: useful bytes stay the same,
// but issued bytes multiply by gqa, shifting the bottleneck to cache rather than bandwidth. With G=gqa, K/V read once,
// while G dot products/accumulations reuse the same loaded row.
template<class DT,int HD,int G>
__global__ __launch_bounds__(256)
void fd_partial(DecParams p)
{
    typedef typename DT::T T;
    // Grid x is a packed query row, not a padded (request,max_query_len) rectangle.
    const int bx=blockIdx.x, kvh=blockIdx.y, sp=blockIdx.z, t=threadIdx.x;
    const AluQueryIndex qi=alu_query_index(p,bx);
    const int b=qi.batch, tk=qi.token;
    const int h0=kvh*G;
    const AluQueryBounds bounds=alu_query_bounds(p,qi);
    const int qpos=bounds.position;
    const int span=bounds.end-bounds.begin;
    const int chunk=(span+p.nsplit-1)/p.nsplit;
    const int start=bounds.begin+sp*chunk, end=min(start+chunk,bounds.end);
    constexpr bool F32 = DT::is_fp32;
    typedef typename std::conditional<F32,float,_Float16>::type QT;
    constexpr int NDG = HD/8;          // dimension groups (8 dimensions per thread)
    constexpr int NRG = 256/NDG;       // position groups processed in parallel (floor!)
    const int dl = t % NDG, rg = t / NDG, d0 = dl*8;
    // When head_dim does not divide 256/NDG evenly (for example 112 -> 18 groups x 14 threads = 252),
    // trailing threads belong to no group and must remain idle, otherwise they write out of bounds
    // in mrg and also double-count other positions in the V accumulation.
    const bool vact = (rg < NRG);

    __shared__ QT qs[G][HD];
    __shared__ float ptile[G][256];
    __shared__ int   btile[256];       // block indices for tile positions — read once
    __shared__ float wred[G][16];      // partial maxima from half-waves, one row per head
    __shared__ float wsum[G][16];      // partial sums — SEPARATE: otherwise writing the sum
                                       // overwrites a maximum that other threads are still reading
    __shared__ float mrg[NRG][HD];     // reduction of partial V sums (reused across heads)
    // Load the query row; request-local positions are derived independently below.
    #pragma unroll
    for(int g=0;g<G;g++){
        const T* qg=(const T*)p.q+((size_t)bx*p.NH+h0+g)*HD;
        for(int d=t; d<HD; d+=256) qs[g][d]=qi.active ? (QT)ld1<DT>(qg+d,p.sc_q) : (QT)0;
    }
    __syncthreads();
    // rotary: cos/sin are a TABLE [seqlen_ro, rotary_dim/2] (as in flash-attn), selecting a row by
    // token position. rotary_dim may be smaller than head_dim, leaving trailing dimensions unrotated.
    const int rdim = p.rotary_dim>0 ? p.rotary_dim : HD;
    if(qi.active && p.rotary && t<rdim/2){
        const int rt=(p.causal || p.window>=0 || p.window_right>=0) ? tk : 0;
        const int rp=(p.rotary_pos? p.rotary_pos[b] : p.ctx_len[b])+rt;
        int d=t; const size_t ro=(size_t)rp*(rdim/2)+d;
        float c=p.cos[ro], s=p.sin[ro];
        #pragma unroll
        for(int g=0;g<G;g++){
            if(p.rotary==1){ float x=qs[g][d], y=qs[g][d+rdim/2]; qs[g][d]=(QT)(x*c-y*s); qs[g][d+rdim/2]=(QT)(y*c+x*s); }
            else { float x=qs[g][2*d], y=qs[g][2*d+1]; qs[g][2*d]=(QT)(x*c-y*s); qs[g][2*d+1]=(QT)(y*c+x*s); }
        }
    }
    __syncthreads();

    float acc[G][8];
    #pragma unroll
    for(int g=0;g<G;g++)
        #pragma unroll
        for(int i=0;i<8;i++) acc[g][i]=0.0f;

    float m[G], l[G];
    #pragma unroll
    for(int g=0;g<G;g++){ m[g]=-1e30f; l[g]=0.0f; }

    for(int base=start; base<end; base+=256){
        // block index for each tile position — ONE global-memory read per position
        { int pos=base+t;
          btile[t] = (pos<end)? p.block_table[(size_t)b*p.block_table_stride + (pos>>p.page_shift)] : 0; }
        __syncthreads();
        const int pos=base+t;
        const bool valid = pos<end;
        float s[G];
        #pragma unroll
        for(int g=0;g<G;g++) s[g]=-1e30f;
        if(valid){
            int blk=btile[t], slot=pos&p.page_mask;
            const size_t off=(size_t)blk*p.k_block_stride+(size_t)slot*p.k_token_stride
                             +(size_t)kvh*p.k_head_stride;
            const T* kp=(const T*)p.Kc + off;
            if constexpr(F32){
                #pragma unroll
                for(int g=0;g<G;g++){ float dd=0;
                    #pragma unroll 4
                    for(int i=0;i<HD;i++) dd+=(float)qs[g][i]*ld1<DT>(kp+i,1.0f);
                    s[g]=dd*p.scale; }
            } else if constexpr(DT::is_bf16){
                // convert K row into registers ONCE, then compute G dot products from it
                half2_t kk[HD/2]; float sck=p.sc_k;
                #pragma unroll
                for(int i=0;i<HD/8;i++){ uint4 r=*(const uint4*)(kp+i*8); const uint16_t* w=(const uint16_t*)&r;
                    #pragma unroll
                    for(int mm=0;mm<4;mm++){ uint32_t b0=((uint32_t)w[2*mm])<<16,b1=((uint32_t)w[2*mm+1])<<16;
                        float f0,f1; __builtin_memcpy(&f0,&b0,4); __builtin_memcpy(&f1,&b1,4);
                        kk[i*4+mm]=half2_t{(_Float16)(f0*sck),(_Float16)(f1*sck)}; } }
                #pragma unroll
                for(int g=0;g<G;g++){ const half2_t* q2=(const half2_t*)qs[g]; float dd=0;
                    #pragma unroll
                    for(int i=0;i<HD/2;i++) dd=__builtin_amdgcn_fdot2(q2[i],kk[i],dd,false);
                    s[g]=dd*p.scale/(p.sc_q*p.sc_k); }
            } else {
                half2_t kk[HD/2];
                const half2_t* k2=(const half2_t*)kp;
                #pragma unroll
                for(int i=0;i<HD/2;i++) kk[i]=k2[i];      // K row in registers, followed by G dot products
                #pragma unroll
                for(int g=0;g<G;g++){ const half2_t* q2=(const half2_t*)qs[g]; float dd=0;
                    #pragma unroll
                    for(int i=0;i<HD/2;i++) dd=__builtin_amdgcn_fdot2(q2[i],kk[i],dd,false);
                    s[g]=dd*p.scale; }
            }
        }
        // softcap and alibi match forward: softcap compresses score to s=c*tanh(s/c), alibi
        // adds an absolute distance-based slope for this request-local query position.
        if(valid && (p.softcap>0.0f || p.alibi)){
            #pragma unroll
            for(int g=0;g<G;g++){
                if(p.softcap>0.0f) s[g]=p.softcap*tanhf(s[g]/p.softcap);
                if(p.alibi)        s[g]-=p.alibi[h0+g]*fabsf((float)(pos-qpos));
            }
        }
        // ---- tile max and sum: DPP within 16 lanes + one barrier for 16 partials ----
        #pragma unroll
        for(int g=0;g<G;g++){ float lm=red16_max(s[g]); if((t&15)==0) wred[g][t>>4]=lm; }
        __syncthreads();
        float nm[G], so[G];
        #pragma unroll
        for(int g=0;g<G;g++){
            float tmax=-1e30f;
            #pragma unroll
            for(int r=0;r<16;r++) tmax=fmaxf(tmax,wred[g][r]);
            nm[g]=fmaxf(m[g],tmax); so[g]=__expf(m[g]-nm[g]);
            float pv=valid?__expf(s[g]-nm[g]):0.0f;
            ptile[g][t]=pv;
            float ls=red16_sum(pv);
            if((t&15)==0) wsum[g][t>>4]=ls;
        }
        __syncthreads();
        #pragma unroll
        for(int g=0;g<G;g++){
            float tsum=0.0f;
            #pragma unroll
            for(int r=0;r<16;r++) tsum+=wsum[g][r];
            l[g]=l[g]*so[g]+tsum; m[g]=nm[g];
            #pragma unroll
            for(int i=0;i<8;i++) acc[g][i]*=so[g];
        }

        // ---- tile contribution to O: all 256 threads, 16-byte reads, one V row for G heads ----
        const int tn=min(end-base,256);
        if(vact) for(int pp=rg; pp<tn; pp+=NRG){
            int gp=base+pp, blk=btile[pp], slot=gp&p.page_mask;
            const size_t off=(size_t)blk*p.v_block_stride+(size_t)slot*p.v_token_stride
                             +(size_t)kvh*p.v_head_stride;
            const T* vp=(const T*)p.Vc + off + p.v_off + d0;
            float vv[8];
            if constexpr(F32){
                #pragma unroll
                for(int i=0;i<8;i++) vv[i]=(float)vp[i];
            } else if constexpr(DT::is_bf16){
                uint4 r=*(const uint4*)vp; const uint16_t* wd=(const uint16_t*)&r;
                #pragma unroll
                for(int i=0;i<8;i++){ uint32_t u=((uint32_t)wd[i])<<16; float f;
                    __builtin_memcpy(&f,&u,4); vv[i]=f*p.sc_v; }
            } else {
                half8_t a=*(const half8_t*)vp;
                #pragma unroll
                for(int i=0;i<8;i++) vv[i]=(float)a[i];
            }
            #pragma unroll
            for(int g=0;g<G;g++){ float w=ptile[g][pp];
                #pragma unroll
                for(int i=0;i<8;i++) acc[g][i]+=w*vv[i]; }
        }
        __syncthreads();
    }
    // ---- reduce NRG partial sums, one head at a time (shared buffer) ----
    #pragma unroll
    for(int g=0;g<G;g++){
        __syncthreads();
        #pragma unroll
        for(int i=0;i<8;i++) if(vact) mrg[rg][d0+i]=acc[g][i];
        __syncthreads();
        const size_t row=(size_t)bx*p.NH+h0+g;
        float* partial=p.nsplit>1 ? p.scratch+(row*p.nsplit+sp)*(HD+2) : nullptr;
        if(t==0){
            if(partial){ partial[0]=(start<end)?m[g]:-1e30f; partial[1]=l[g]; }
            else if(p.lse_out) p.lse_out[row]=l[g]>0 ? m[g]+__logf(l[g]) : -INFINITY;
        }
        for(int d=t; d<HD; d+=256){
            float a=0.0f;
            #pragma unroll
            for(int r=0;r<NRG;r++) a+=mrg[r][d];
            a/=p.sc_v;
            if(partial) partial[2+d]=a;
            else {
                const float value=l[g]>0 ? a/l[g] : 0.0f;
                ((T*)p.o)[row*HD+d]=(T)value;
            }
        }
    }
}

// grid (batch, NH), block HD. Merge nsplit slices.
template<int HD>
__global__ void fd_combine(void* o,const float* scratch,int nsplit,int NH,int dtype,float* lse_out){
    const int b=blockIdx.x, h=blockIdx.y, t=threadIdx.x;
    const float* base=scratch + ((size_t)b*NH+h)*nsplit*(HD+2);
    float gm=-1e30f;
    for(int s=0;s<nsplit;s++) gm=fmaxf(gm, base[(size_t)s*(HD+2)]);
    float acc=0.0f, l=0.0f;
    for(int s=0;s<nsplit;s++){ float w=__expf(base[(size_t)s*(HD+2)]-gm);
        acc += w*base[(size_t)s*(HD+2)+2+t]; l += w*base[(size_t)s*(HD+2)+1]; }
    float r = l>0? acc/l : 0.0f;
    // return_softmax_lse: online softmax here uses natural base (__expf),
    // so full row logsumexp = gm + ln(l). One thread writes it, only when requested.
    if(lse_out && t==0) lse_out[(size_t)b*NH+h] = l>0? (gm+__logf(l)) : -INFINITY;
    size_t idx=((size_t)b*NH+h)*HD+t;
    if(dtype==FA_F32) ((float*)o)[idx]=r;
    else if(dtype==FA_BF16) ((__hip_bfloat16*)o)[idx]=(__hip_bfloat16)r;
    else ((_Float16*)o)[idx]=(_Float16)r;
}

// Append packed new K/V after each request cache end, with optional rotary on K.
template<class DT,int HD>
__global__ void fd_append(DecParams p,const void* new_k,const void* new_v){
    typedef typename DT::T T;
    const int bx=blockIdx.x, kvh=blockIdx.y, t=threadIdx.x;
    const AluQueryIndex qi=alu_query_index(p,bx);
    if(!qi.active) return;  // Uniform across the workgroup, before any barrier.
    const int b=qi.batch, tk=qi.token;
    const int pos=p.ctx_len[b]+tk;   // new tokens are placed immediately after current cache end
    int blk=p.block_table[(size_t)b*p.block_table_stride + (pos>>p.page_shift)], slot=pos&p.page_mask;
    const int NKV=p.NH/p.gqa;
    const T* nk=(const T*)new_k+((size_t)bx*NKV+kvh)*HD;
    const T* nv=(const T*)new_v+((size_t)bx*NKV+kvh)*HD;
    const size_t ko=(size_t)blk*p.k_block_stride+(size_t)slot*p.k_token_stride+(size_t)kvh*p.k_head_stride;
    const size_t vo=(size_t)blk*p.v_block_stride+(size_t)slot*p.v_token_stride+(size_t)kvh*p.v_head_stride;
    T* kc=(T*)p.Kc + ko;
    T* vc=(T*)p.Vc + vo + p.v_off;
    // rotary on k (in float, converted back to T)
    __shared__ float ks[HD];
    if(t<HD) ks[t]=ld1<DT>(nk+t,1.0f);
    __syncthreads();
    const int rdim = p.rotary_dim>0 ? p.rotary_dim : HD;
    if(p.rotary && t<rdim/2){
        const int rp=(p.rotary_pos? p.rotary_pos[b] : p.ctx_len[b])+tk;
        int d=t; const size_t ro=(size_t)rp*(rdim/2)+d;
        float c=p.cos[ro], s=p.sin[ro];
        if(p.rotary==1){ float x=ks[d],y=ks[d+rdim/2]; ks[d]=x*c-y*s; ks[d+rdim/2]=y*c+x*s; }
        else { float x=ks[2*d],y=ks[2*d+1]; ks[2*d]=x*c-y*s; ks[2*d+1]=y*c+x*s; } }
    __syncthreads();
    if(t<HD){ if constexpr(DT::is_bf16){ kc[t]=(__hip_bfloat16)ks[t]; vc[t]=(__hip_bfloat16)ld1<DT>(nv+t,1.0f); }
              else if constexpr(DT::is_fp32){ kc[t]=ks[t]; vc[t]=(*(nv+t)); }
              else { kc[t]=(_Float16)ks[t]; vc[t]=(_Float16)ld1<DT>(nv+t,1.0f); } }
}

// Workgroups = batch * (NH/gqa) * nsplit. With GQA reuse there are gqa times fewer groups,
// so split the context enough to keep the active GPU occupied. The target is four workgroups per CU
// (288 on a 72-CU device, 240 on a 60-CU device) without splitting further than the context allows;
// each extra slice adds another scratch row and fd_combine work.
static int decode_target_workgroups(){
    int dev=0; hipDeviceProp_t prop{};
    if(hipGetDevice(&dev)==hipSuccess && hipGetDeviceProperties(&prop,dev)==hipSuccess && prop.multiProcessorCount>0)
        return prop.multiProcessorCount*4;
    return 288;
}
static int calc_nsplit(int ctx,int gqa=1,int NH=32,int batch=1){
    const int NKV = NH/(gqa>0?gqa:1);
    int groups = batch*(NKV>0?NKV:1);
    const int target = decode_target_workgroups();
    int n = (target + groups - 1)/groups;
    int maxn=(ctx+255)/256; if(maxn<1) maxn=1;
    if(n>maxn) n=maxn;
    if(n<1) n=1; if(n>512) n=512;
    return n;
}

// ---- host launch ----
template<class DT,int HD>
static void launch_dec(DecParams& p,int dtype,hipStream_t st){
    const int NKV=p.NH/p.gqa;
    // One workgroup per KV head serves G=gqa query heads => K and V are read once instead of gqa times.
    // This creates gqa times fewer workgroups, so increase context splits by the same factor,
    // otherwise the GPU is underfilled and idle time consumes the gain.
    #define DEC_G(GG) do{ dim3 gp(p.total_q,NKV,p.nsplit); fd_partial<DT,HD,GG><<<gp,256,0,st>>>(p); }while(0)
    switch(p.gqa){
        case 8: DEC_G(8); break;
        case 6: DEC_G(6); break;
        case 4: DEC_G(4); break;
        case 3: DEC_G(3); break;
        case 2: DEC_G(2); break;
        case 1: DEC_G(1); break;
        default:
            printf("fa_decode: GQA ratio=%d is not instantiated; supported ratios are 1,2,3,4,6,8\n", p.gqa);
            abort();
    }
    #undef DEC_G
    if(p.nsplit>1) fd_combine<HD><<<dim3(p.total_q,p.NH),HD,0,st>>>(p.o,p.scratch,p.nsplit,p.NH,dtype,p.lse_out);
}
template<class DT> static void append_hd(DecParams& p,const void* nk,const void* nv,hipStream_t st){
    int NKV=p.NH/p.gqa; dim3 g(p.total_q,NKV);
    #define APP(HD) case HD: fd_append<DT,HD><<<g,((HD)<64?64:(HD)),0,st>>>(p,nk,nv); break;
    switch(p.HD){ APP(16) APP(32) APP(48) APP(64) APP(80) APP(96)
                  APP(112) APP(128) APP(160) APP(192) APP(224) APP(256) APP(512)
                  default: printf("fa_decode append: head_dim=%d not instantiated\n",p.HD); abort(); }
    #undef APP
}
extern "C" void fa_decode_launch(DecParams p,int dtype,hipStream_t st,const void* new_k,const void* new_v){
    if(p.sc_q==0.0f)p.sc_q=1.0f; if(p.sc_k==0.0f)p.sc_k=1.0f; if(p.sc_v==0.0f)p.sc_v=1.0f;
    if(p.nq<1) p.nq=1;
    if(p.total_q==0) p.total_q=p.batch*p.nq; // Standalone legacy harness.
    if(p.block_table_stride==0) p.block_table_stride=p.max_blocks;
    p.append = new_k? 1 : 0;
    if(p.k_block_stride==0){
        const long long row=p.kv_row>0 ? p.kv_row : p.HD;
        p.k_head_stride=p.v_head_stride=row;
        p.k_token_stride=p.v_token_stride=(p.NH/p.gqa)*row;
        p.k_block_stride=p.v_block_stride=p.page*p.k_token_stride;
    }
    if(p.kv_row<=0) p.kv_row=p.HD;
    if(p.v_off<0) p.v_off=0;
    p.page_shift=__builtin_ctz(p.page); p.page_mask=p.page-1;   // page is a power of two
    if(new_k){ if(dtype==FA_F16) append_hd<DT_F16>(p,new_k,new_v,st);
               else if(dtype==FA_BF16) append_hd<DT_BF16>(p,new_k,new_v,st);
               else append_hd<DT_F32>(p,new_k,new_v,st);
        // No device-wide synchronize here: append and attention are enqueued on the same current
        // PyTorch stream, so stream ordering provides the required dependency and remains graph-safe.
    }
    #define DECHD(DTt) do{ switch(p.HD){ \
        case 16: launch_dec<DTt, 16>(p,dtype,st);break; case 32: launch_dec<DTt, 32>(p,dtype,st);break; \
        case 48: launch_dec<DTt, 48>(p,dtype,st);break; case 64: launch_dec<DTt, 64>(p,dtype,st);break; \
        case 80: launch_dec<DTt, 80>(p,dtype,st);break; case 96: launch_dec<DTt, 96>(p,dtype,st);break; \
        case 112:launch_dec<DTt,112>(p,dtype,st);break; case 128:launch_dec<DTt,128>(p,dtype,st);break; \
        case 160:launch_dec<DTt,160>(p,dtype,st);break; case 192:launch_dec<DTt,192>(p,dtype,st);break; \
        case 224:launch_dec<DTt,224>(p,dtype,st);break; case 256:launch_dec<DTt,256>(p,dtype,st);break; \
        /* D512 (Gemma4 full-attention): register-resident decode (no LDS K/V tiling), so only
           the dispatch changes. K row = HD/2 half2 = 256 half2 (256 VGPRs) — may spill, but
           decode is KV-bandwidth-bound so that is acceptable; correctness is oracle-gated. */ \
        case 512:launch_dec<DTt,512>(p,dtype,st);break; \
        default: printf("fa_decode: head_dim=%d not instantiated\n",p.HD); abort(); } }while(0)
    if(dtype==FA_F16) DECHD(DT_F16); else if(dtype==FA_BF16) DECHD(DT_BF16); else DECHD(DT_F32);
    #undef DECHD
}

#ifdef FA_STANDALONE
#include "fa_decode_gate.inc"
#endif
