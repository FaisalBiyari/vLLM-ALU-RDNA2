#include "../package/csrc/alu_compiler_policy.h" // ALU_COMPILER_HARDENING_POLICY
// fa2_gfx1030.hip — FlashAttention-2 forward based on an external proposal (BR=64/BC=64, 8 waves,
// XOR-swizzled K, padded V^T (stride 72), register prefetch of the next KV tile, causal tile skipping).
// Goal: determine whether this layout beats the current fa_forward.hip kernel (16 TFLOP/s).
//
// Differences from the original proposal text (bugs fixed; everything else is 1:1):
//  1) removed a dead commit_tile fragment containing a placeholder line (did not compile) — noted by the author;
//  2) commit_v source indexing used ra[blk*2+...] / ra[4+blk*2+...] — overlapping
//     ranges that read the wrong dwords. Correct: row c   -> ra[blk*4 + (hi>>1)],
//                                     row c+1 -> ra[8 + blk*4 + (hi>>1)];
//  3) commit_v wrote d twice (first via (tid&7), then via kch) — retained the second, correct write;
//  4) replaced __builtin_bit_cast with a union cast (the current build uses C++20);
//  5) exp2f -> __builtin_amdgcn_exp2f (guarantees v_exp_f32 without relying on -ffast-math).
// Build: bash tools/build_hip_tool.sh kernels/fa2_gfx1030.hip
//
// MEASUREMENTS 2026-07-20, RX 6800 XT (gfx1030, 72 CU, 2065 MHz), HD=128, causal, fp16, NH=32, B=1.
// Correctness vs fp64 oracle: max|err| 2.2e-4, rel_rms 3.4e-4 (n=64/100/128/256) — pass.
//   variant                                  n=2048        n=8192      VGPR spills
//   current fa_forward.hip kernel (baseline) 14.6 TF/s     16.5 TF/s   —
//   proposal as written (prefetch)            5.1 TF/s      5.4 TF/s   297 (scratch 1328 B/lane)
//   without register prefetch                14.7 TF/s     15.4 TF/s   18
//   + unroll QK=4/PV=4 (zero spills)         14.3 TF/s     15.5 TF/s   0
//   ablation: without K swizzle              11.0 TF/s     11.9 TF/s   0
//   ablation: without V^T padding            11.1 TF/s     12.1 TF/s   0
//   ablation: without either                  9.0 TF/s      9.8 TF/s   0
// Conclusion: swizzle and padding work (+58% within this layout), but register prefetch on RDNA2
// does not pay off — its 32 VGPR explode the register file and cut speed by 3x. With corrected prefetch
// the design only reaches the current kernel, not beyond it: the ~16 TF/s ceiling here is structural
// (LDS transpose + softmax), not a bank-conflict problem.
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <cmath>
#include <vector>

typedef _Float16 half_t;
typedef half_t   half2_t __attribute__((ext_vector_type(2)));
typedef uint32_t u32;

#define HIP_CHECK(x) do{hipError_t e=(x);if(e!=hipSuccess){printf("HIP %s L%d\n",hipGetErrorString(e),__LINE__);exit(1);}}while(0)

#define D_HEAD 128
#define BR 64
#define BC 64
#define NW 8            // waves per workgroup
#define WG (NW*32)
#ifdef NO_PAD
#define VT_STRIDE 64
#else
#define VT_STRIDE 72
#endif
// VT_STRIDE = 64 + 8 padding (9 chunks of 8 half); NO_PAD/NO_SWIZZLE — ablation branches
#ifndef QK_UNROLL
#define QK_UNROLL 16
#endif
#ifndef PV_UNROLL
#define PV_UNROLL 8
#endif

__device__ __forceinline__ half2_t h2(u32 w){ union{u32 u; half2_t h;} c; c.u=w; return c.h; }
typedef __fp16 fp16x2_t __attribute__((ext_vector_type(2)));   // cvt_pkrtz return type
__device__ __forceinline__ u32 u32_of(fp16x2_t h){ union{fp16x2_t h; u32 u;} c; c.h=h; return c.u; }

__device__ __forceinline__ float wave_reduce_max(float x){
  #pragma unroll
  for(int m=16;m>=1;m>>=1) x=fmaxf(x,__shfl_xor(x,m,32));
  return x;
}
__device__ __forceinline__ float wave_reduce_sum(float x){
  #pragma unroll
  for(int m=16;m>=1;m>>=1) x+=__shfl_xor(x,m,32);
  return x;
}

// K swizzle: physical 16-byte chunk within row c
__device__ __forceinline__ u32 kswz(u32 c,u32 k16){
#ifdef NO_SWIZZLE
  return k16;
#else
  return k16 ^ ((c>>1)&7);
#endif
}

__global__ __launch_bounds__(WG,1)
void fa2_fwd(const half_t* __restrict__ Q, const half_t* __restrict__ K,
             const half_t* __restrict__ V, half_t* __restrict__ O,
             int n, int gqa, float scale)
{
  __shared__ half_t sQ [BR*D_HEAD];          // 16 KB
  __shared__ half_t sK [BC*D_HEAD];          // 16 KB, swizzled
  __shared__ half_t sVT[D_HEAD*VT_STRIDE];   // 18 KB, transposed
  __shared__ half_t sP [NW*8*BC];            // 8 KB, private per wave

  const int tid=threadIdx.x;
  const int lane=tid&31, wv=tid>>5;
  const int qblk=blockIdx.x, h=blockIdx.y, b=blockIdx.z;
  const int H=gridDim.y;
  const int q0=qblk*BR;

  const half_t* q_base=Q+(((size_t)b*H+h)*n)*D_HEAD;
  const half_t* k_base=K+(((size_t)b*(H/gqa)+h/gqa)*n)*D_HEAD;
  const half_t* v_base=V+(((size_t)b*(H/gqa)+h/gqa)*n)*D_HEAD;
  half_t*       o_base=O+(((size_t)b*H+h)*n)*D_HEAD;

  // ---- Q -> LDS (once): 256 threads x 4 int4
  {
    const int4* src=(const int4*)q_base;
    int4* dst=(int4*)sQ;
    #pragma unroll
    for(int i=0;i<4;i++){
      int idx=tid+i*WG;            // 16-byte chunks of Q block
      int row=idx>>4, r_g=q0+row;
      dst[idx]=(r_g<n)? src[(size_t)r_g*16+(idx&15)] : int4{0,0,0,0};
    }
  }

  float o_acc[8][4];
  float m_r[8], l_r[8];
  #pragma unroll
  for(int r=0;r<8;r++){ m_r[r]=-1e30f; l_r[r]=0.f;   // -inf is unsafe under -ffast-math; -1e30 is equivalent here
    #pragma unroll
    for(int t=0;t<4;t++) o_acc[r][t]=0.f; }

  const float cscale=scale*1.44269504f;
  const int n_tiles_full=(n+BC-1)/BC;
  const int n_tiles=min(n_tiles_full,(q0+BR+BC-1)/BC);   // causal tile skipping

  const int krow=tid>>3, kch=tid&7;
  int4 rgK[4], rgV[4];

  auto load_tile=[&](int tile){
    const int4* ks=(const int4*)k_base;
    const int4* vs=(const int4*)v_base;
    int c0=tile*BC;
    #pragma unroll
    for(int i=0;i<2;i++){
      int rg=min(c0+krow+32*i,n-1);
      rgK[i*2+0]=ks[(size_t)rg*16+kch];
      rgK[i*2+1]=ks[(size_t)rg*16+kch+8];
    }
    int vr=min(c0+2*(tid>>3),n-1), vr1=min(vr+1,n-1);
    rgV[0]=vs[(size_t)vr *16+kch]; rgV[1]=vs[(size_t)vr *16+kch+8];
    rgV[2]=vs[(size_t)vr1*16+kch]; rgV[3]=vs[(size_t)vr1*16+kch+8];
  };

  auto commit_k=[&](){
    int4* kd=(int4*)sK;
    #pragma unroll
    for(int i=0;i<2;i++){
      int r=krow+32*i;
      kd[r*16+kswz(r,kch)  ]=rgK[i*2+0];
      kd[r*16+kswz(r,kch+8)]=rgK[i*2+1];
    }
  };
  // V -> LDS transposed, row pairs (c,c+1) in one dword; write rotation avoids bank conflicts
  auto commit_v=[&](){
    u32* vt=(u32*)sVT;
    const u32* ra=(const u32*)rgV;      // rgV[0..1]=row c (8 dwords), rgV[2..3]=row c+1
    int cp=tid>>3;                      // column-pair index (0..31)
    #pragma unroll
    for(int hh=0;hh<16;hh++){
      int e=(hh+(tid&7))&15;            // local half index 0..15
      int blk=e>>3, hi=e&7;             // blk: 0 -> d=kch*8.., 1 -> d=64+kch*8..
      u32 wa=ra[    blk*4+(hi>>1)];     // row c
      u32 wb=ra[8 + blk*4+(hi>>1)];     // row c+1
      u32 va=(hi&1)?(wa>>16):(wa&0xFFFFu);
      u32 vb=(hi&1)?(wb>>16):(wb&0xFFFFu);
      int d=kch*8+64*blk+hi;
      vt[d*(VT_STRIDE/2)+cp]=va|(vb<<16);
    }
  };

#ifdef NO_PREFETCH
  // no-register-prefetch variant: load tile immediately before compute (LDS barriers instead of 32 VGPR)
  #define TILE_HEAD(t) do{ __syncthreads(); load_tile(t); commit_k(); commit_v(); __syncthreads(); }while(0)
#else
  load_tile(0);
  __syncthreads();
  commit_k(); commit_v();
  __syncthreads();
#endif

  const int my_r0=wv*8;
  half_t* myP=sP+wv*8*BC;

  for(int tile=0; tile<n_tiles; tile++){
#ifdef NO_PREFETCH
    TILE_HEAD(tile);
#else
    if(tile+1<n_tiles) load_tile(tile+1);      // prefetch into registers
#endif

    // ================= QK^T =================
    float s[8][2];
    #pragma unroll
    for(int r=0;r<8;r++) s[r][0]=s[r][1]=0.f;
    const int c0=2*lane, c1=2*lane+1;
    #pragma unroll QK_UNROLL
    for(int k16=0;k16<16;k16++){
      int4 kb0=((int4*)sK)[c0*16+kswz(c0,k16)];
      int4 kb1=((int4*)sK)[c1*16+kswz(c1,k16)];
      const u32* kb0w=(const u32*)&kb0; const u32* kb1w=(const u32*)&kb1;
      #pragma unroll
      for(int r=0;r<8;r++){
        int4 qv=((int4*)sQ)[(my_r0+r)*16+k16];      // broadcast
        const u32* qw=(const u32*)&qv;
        #pragma unroll
        for(int p=0;p<4;p++){
          s[r][0]=__builtin_amdgcn_fdot2(h2(qw[p]),h2(kb0w[p]),s[r][0],false);
          s[r][1]=__builtin_amdgcn_fdot2(h2(qw[p]),h2(kb1w[p]),s[r][1],false);
        }
      }
    }

    // ============ mask + online softmax ============
    const int gc0=tile*BC+c0, gc1=tile*BC+c1;
    #pragma unroll
    for(int r=0;r<8;r++){
      int gr=q0+my_r0+r;
      float s0=s[r][0]*cscale, s1=s[r][1]*cscale;
      if(gc0>gr||gc0>=n) s0=-1e30f;
      if(gc1>gr||gc1>=n) s1=-1e30f;
      float mt=wave_reduce_max(fmaxf(s0,s1));
      float mn=fmaxf(m_r[r],mt);
      float alpha=__builtin_amdgcn_exp2f(m_r[r]-mn);
      float p0=__builtin_amdgcn_exp2f(s0-mn), p1=__builtin_amdgcn_exp2f(s1-mn);
      l_r[r]=l_r[r]*alpha+wave_reduce_sum(p0+p1);
      m_r[r]=mn;
      #pragma unroll
      for(int t=0;t<4;t++) o_acc[r][t]*=alpha;
      ((u32*)myP)[r*32+lane]=u32_of(__builtin_amdgcn_cvt_pkrtz(p0,p1));
    }

    // ================= P·V =================
    #pragma unroll PV_UNROLL
    for(int kc=0;kc<8;kc++){
      int4 vb[4];
      #pragma unroll
      for(int t=0;t<4;t++)
        vb[t]=((int4*)sVT)[(lane+32*t)*(VT_STRIDE/8)+kc];
      #pragma unroll
      for(int r=0;r<8;r++){
        int4 pb=((int4*)myP)[r*8+kc];     // broadcast
        const u32* pw=(const u32*)&pb;
        #pragma unroll
        for(int t=0;t<4;t++){
          const u32* vw=(const u32*)&vb[t];
          #pragma unroll
          for(int p=0;p<4;p++)
            o_acc[r][t]=__builtin_amdgcn_fdot2(h2(pw[p]),h2(vw[p]),o_acc[r][t],false);
        }
      }
    }

    // ============ commit prefetch ============
#ifndef NO_PREFETCH
    if(tile+1<n_tiles){
      __syncthreads();
      commit_k(); commit_v();
      __syncthreads();
    }
#endif
  }

  // ---- final: O/l -> fp16
  #pragma unroll
  for(int r=0;r<8;r++){
    int gr=q0+my_r0+r;
    if(gr>=n) continue;
    float rl=(l_r[r]>0.f)?1.f/l_r[r]:0.f;
    #pragma unroll
    for(int t=0;t<4;t++)
      o_base[(size_t)gr*D_HEAD+lane+32*t]=(half_t)(o_acc[r][t]*rl);
  }
}

// ============================ gate: correctness + benchmark ============================
static uint32_t xrng(uint32_t& s){ s=s*1664525u+1013904223u; return s; }
static float frand(uint32_t& s){ return ((xrng(s)>>16)&4095)/4096.0f-0.5f; }

// fp64 oracle: O = softmax(scale*QK^T + causal) V, layout [B=1][H][n][D]
static void ref_attn(const std::vector<float>& Q,const std::vector<float>& K,const std::vector<float>& V,
                     std::vector<float>& O,int n,int H,int gqa,float scale){
  std::vector<double> pr(n);
  for(int h=0;h<H;h++){ int kvh=h/gqa;
    for(int i=0;i<n;i++){
      double mx=-1e300;
      for(int j=0;j<n;j++){
        if(j>i){ pr[j]=-1e300; continue; }
        double s=0;
        for(int d=0;d<D_HEAD;d++)
          s+=(double)Q[((size_t)h*n+i)*D_HEAD+d]*(double)K[((size_t)kvh*n+j)*D_HEAD+d];
        s*=scale; pr[j]=s; if(s>mx)mx=s;
      }
      double sum=0;
      for(int j=0;j<n;j++){ if(pr[j]<=-1e299){pr[j]=0;continue;} pr[j]=exp(pr[j]-mx); sum+=pr[j]; }
      for(int d=0;d<D_HEAD;d++){ double acc=0;
        for(int j=0;j<n;j++) acc+=pr[j]*(double)V[((size_t)kvh*n+j)*D_HEAD+d];
        O[((size_t)h*n+i)*D_HEAD+d]=(float)(acc/sum);
      }
    }
  }
}

static void* to_dev_h(const std::vector<float>& v){
  std::vector<half_t> h(v.size()); for(size_t i=0;i<v.size();i++) h[i]=(half_t)v[i];
  void* d; HIP_CHECK(hipMalloc(&d,v.size()*2));
  HIP_CHECK(hipMemcpy(d,h.data(),v.size()*2,hipMemcpyHostToDevice)); return d;
}

static int check(int n,int H,int gqa){
  float scale=1.0f/sqrtf((float)D_HEAD);
  uint32_t s=99u+n*7+H;
  int NKV=H/gqa;
  std::vector<float> Q((size_t)H*n*D_HEAD), K((size_t)NKV*n*D_HEAD), V(K.size()), O(Q.size(),0.f), Oref(Q.size());
  for(auto& x:Q)x=frand(s); for(auto& x:K)x=frand(s); for(auto& x:V)x=frand(s);
  for(auto& x:Q)x=(float)(half_t)x; for(auto& x:K)x=(float)(half_t)x; for(auto& x:V)x=(float)(half_t)x;
  ref_attn(Q,K,V,Oref,n,H,gqa,scale);
  void *dQ=to_dev_h(Q),*dK=to_dev_h(K),*dV=to_dev_h(V),*dO=to_dev_h(O);
  dim3 grid((n+BR-1)/BR,H,1);
  hipLaunchKernelGGL(fa2_fwd,grid,dim3(WG),0,0,(const half_t*)dQ,(const half_t*)dK,(const half_t*)dV,(half_t*)dO,n,gqa,scale);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<half_t> oh(O.size()); HIP_CHECK(hipMemcpy(oh.data(),dO,O.size()*2,hipMemcpyDeviceToHost));
  double maxe=0,se=0,sr=0;
  for(size_t i=0;i<O.size();i++){ double g=(double)(float)oh[i], r=Oref[i], d=fabs(g-r);
    if(d>maxe)maxe=d; se+=d*d; sr+=r*r; }
  double rms=sqrt(se/(sr>0?sr:1));
  bool ok=(maxe<2e-2)&&(rms<5e-3);
  printf("  n=%-5d H=%-2d gqa=%d : max|err|=%.2e  rel_rms=%.2e  %s\n",n,H,gqa,maxe,rms,ok?"OK":"FAIL");
  hipFree(dQ);hipFree(dK);hipFree(dV);hipFree(dO);
  return ok?0:1;
}

static void bench(int n,int H,int gqa){
  float scale=1.0f/sqrtf((float)D_HEAD);
  int NKV=H/gqa;
  size_t nQ=(size_t)H*n*D_HEAD, nK=(size_t)NKV*n*D_HEAD;
  std::vector<half_t> hq(nQ,(half_t)0.02f), hk(nK,(half_t)0.02f);
  void *dQ,*dK,*dV,*dO;
  HIP_CHECK(hipMalloc(&dQ,nQ*2)); HIP_CHECK(hipMalloc(&dO,nQ*2));
  HIP_CHECK(hipMalloc(&dK,nK*2)); HIP_CHECK(hipMalloc(&dV,nK*2));
  HIP_CHECK(hipMemcpy(dQ,hq.data(),nQ*2,hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dK,hk.data(),nK*2,hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dV,hk.data(),nK*2,hipMemcpyHostToDevice));
  dim3 grid((n+BR-1)/BR,H,1);
  auto run=[&](){ hipLaunchKernelGGL(fa2_fwd,grid,dim3(WG),0,0,(const half_t*)dQ,(const half_t*)dK,(const half_t*)dV,(half_t*)dO,n,gqa,scale); };
  run(); HIP_CHECK(hipDeviceSynchronize());
  hipError_t e=hipGetLastError(); if(e!=hipSuccess){ printf("  ERR %s\n",hipGetErrorString(e)); return; }
  for(int i=0;i<3;i++) run(); HIP_CHECK(hipDeviceSynchronize());   // warm-up
  hipEvent_t t0,t1; hipEventCreate(&t0); hipEventCreate(&t1);
  int it = n>=8192?10:20;
  hipEventRecord(t0); for(int i=0;i<it;i++) run(); hipEventRecord(t1); hipEventSynchronize(t1);
  float ms=0; hipEventElapsedTime(&ms,t0,t1); ms/=it;
  double fl=2.0*(double)H*(double)n*(double)n*D_HEAD;   // causal: 4*H*n*n*D/2
  printf("  n=%-5d H=%-2d gqa=%d : %.3f ms  ~%.1f TFLOP/s\n",n,H,gqa,ms,fl/(ms*1e-3)/1e12);
  hipFree(dQ);hipFree(dK);hipFree(dV);hipFree(dO);
}

int main(){
  hipDeviceProp_t pr; HIP_CHECK(hipGetDeviceProperties(&pr,0));
  printf("GPU: %s  CU=%d  clock=%d MHz\n",pr.gcnArchName,pr.multiProcessorCount,pr.clockRate/1000);
  printf("== correctness (fp64 oracle, causal, HD=128) ==\n");
  int bad=0;
  bad+=check(64,2,1); bad+=check(128,2,1); bad+=check(100,2,1); bad+=check(256,4,2);
  printf("== benchmark (causal, HD=128, fp16) ==\n");
  bench(2048,32,1); bench(2048,32,4); bench(8192,32,1); bench(8192,32,4);
  printf(bad? "GATE: FAIL (%d)\n":"GATE: PASS\n",bad);
  return bad;
}
