#include "../package/csrc/alu_compiler_policy.h" // ALU_COMPILER_HARDENING_POLICY
// bench_cfg.hip — [TEMPORARY] bottleneck isolation: MODE0 full, MODE1 without softmax (transpose+GEMM2 active),
// MODE2 GEMM1 only (QK^T from LDS, no transpose/GEMM2). Shows where performance drops 42->20.7->16.4.
#include "fa_forward.hip"
#include <vector>
#include <cstdio>
#include <cmath>

static void* alloc_fill(size_t n){
    std::vector<_Float16> h(n,(_Float16)0.02f); void* d;
    HIP_CHECK(hipMalloc(&d,n*2)); HIP_CHECK(hipMemcpy(d,h.data(),n*2,hipMemcpyHostToDevice)); return d;
}

template<int HD,int BR,int BC,int MODE>
__global__ __launch_bounds__(256)
void fa_diag_k(FwdParams p){
    constexpr int QPT=BR/16, KPT=BC/16, DPT=HD/16, HH=HD/2, BCH=BC/2;
    constexpr float LOG2E=1.442695041f;
    const int t=threadIdx.x, tx=t&15, ty=t>>4;
    const int z=blockIdx.z, h0=blockIdx.y, kvh=h0/p.gqa;
    int SQ=p.SQ,SK=p.SK;
    long long qrow0=(long long)z*p.sQ.b, krow0=(long long)z*p.sK.b, orow0=(long long)z*p.sO.b;
    const int qb=blockIdx.x*BR; if(qb>=SQ) return;
    const int delta=SK-SQ; const float scale2=p.scale*LOG2E;
    constexpr int QS_B=HH*BR*4, KS_B=HH*BC*4, VS_B=BCH*HD*4;
    __shared__ char pool[QS_B+KS_B+VS_B];
    half2_t* Qs=(half2_t*)pool; half2_t* Ks=(half2_t*)(pool+QS_B); half2_t* Vs=(half2_t*)(pool+QS_B+KS_B);
    _Float16* Ps=(_Float16*)Ks;
    #define QSA(dh,q) Qs[(dh)*BR+(q)]
    #define KSA(dh,kc) Ks[(dh)*BC+(kc)]
    #define VSA(kh,d) Vs[(kh)*HD+(d)]
    #define PSA(kc,q) Ps[(kc)*BR+(q)]
    const _Float16* Qb=(const _Float16*)p.Q+qrow0+(long long)h0*p.sQ.h;
    for(int e=t;e<BR*(HD/8);e+=256){ int dg=e%(HD/8),q=e/(HD/8); int sq=qb+q;
        half8_t a= sq<SQ ? *(const half8_t*)(Qb+(long long)sq*p.sQ.s+dg*8) : half8_t{0,0,0,0,0,0,0,0};
        #pragma unroll
        for(int m=0;m<4;m++) QSA(dg*4+m,q)=half2_t{a[2*m],a[2*m+1]}; }
    float acco[QPT][DPT], rm[QPT], rl[QPT];
    #pragma unroll
    for(int i=0;i<QPT;i++){ rm[i]=-1e30f; rl[i]=0; for(int e=0;e<DPT;e++) acco[i][e]=0; }
    __syncthreads();
    int kmax=((qb+BR-1)+delta<SK-1)?(qb+BR-1)+delta:SK-1;
    for(int kb=0;kb<=kmax;kb+=BC){
        const bool diag=((kb+BC-1)>=qb+delta);
        __syncthreads();
        constexpr int NK=(BC*(HD/8)+255)/256;
        #pragma unroll
        for(int it=0;it<NK;it++){ int e=t+it*256; if(e<BC*(HD/8)){ int dg=e%(HD/8),kc=e/(HD/8); int sk=kb+kc;
            half8_t a= sk<SK ? *(const half8_t*)((const _Float16*)p.K+krow0+(long long)kvh*p.sK.h+(long long)sk*p.sK.s+dg*8):half8_t{0,0,0,0,0,0,0,0};
            #pragma unroll
            for(int m=0;m<4;m++) KSA(dg*4+m,kc)=half2_t{a[2*m],a[2*m+1]}; } }
        constexpr int NV=(BCH*(HD/8)+255)/256;
        #pragma unroll
        for(int it=0;it<NV;it++){ int e=t+it*256; if(e<BCH*(HD/8)){ int dg=e%(HD/8),kh=e/(HD/8); int s0=kb+2*kh,s1=s0+1,d0=dg*8;
            half8_t a= s0<SK ? *(const half8_t*)((const _Float16*)p.V+krow0+(long long)kvh*p.sV.h+(long long)s0*p.sV.s+d0):half8_t{0,0,0,0,0,0,0,0};
            half8_t b= s1<SK ? *(const half8_t*)((const _Float16*)p.V+krow0+(long long)kvh*p.sV.h+(long long)s1*p.sV.s+d0):half8_t{0,0,0,0,0,0,0,0};
            #pragma unroll
            for(int m=0;m<8;m++) VSA(kh,d0+m)=half2_t{a[m],b[m]}; } }
        __syncthreads();
        float accs[QPT][KPT];
        #pragma unroll
        for(int i=0;i<QPT;i++)for(int j=0;j<KPT;j++)accs[i][j]=0;
#ifdef PIPE1
        { half2_t kc[KPT],qc[QPT],kn[KPT],qn[QPT];
          #pragma unroll
          for(int j=0;j<KPT;j++)kc[j]=KSA(0,tx*KPT+j);
          #pragma unroll
          for(int i=0;i<QPT;i++)qc[i]=QSA(0,ty*QPT+i);
          #pragma unroll 2
          for(int dh=0;dh<HH;dh++){
            if(dh+1<HH){
              #pragma unroll
              for(int j=0;j<KPT;j++)kn[j]=KSA(dh+1,tx*KPT+j);
              #pragma unroll
              for(int i=0;i<QPT;i++)qn[i]=QSA(dh+1,ty*QPT+i);
            }
            #pragma unroll
            for(int i=0;i<QPT;i++)
              #pragma unroll
              for(int j=0;j<KPT;j++)accs[i][j]=__builtin_amdgcn_fdot2(qc[i],kc[j],accs[i][j],false);
            #pragma unroll
            for(int j=0;j<KPT;j++)kc[j]=kn[j];
            #pragma unroll
            for(int i=0;i<QPT;i++)qc[i]=qn[i];
          }
        }
#else
        #pragma unroll 4
        for(int dh=0;dh<HH;dh++){ half2_t kv[KPT];
            #pragma unroll
            for(int j=0;j<KPT;j++)kv[j]=KSA(dh,tx*KPT+j);
            half2_t qv[QPT];
            #pragma unroll
            for(int i=0;i<QPT;i++)qv[i]=QSA(dh,ty*QPT+i);
            #pragma unroll
            for(int i=0;i<QPT;i++)
                #pragma unroll
                for(int j=0;j<KPT;j++)accs[i][j]=__builtin_amdgcn_fdot2(qv[i],kv[j],accs[i][j],false); }
#endif
        if constexpr(MODE==2){ // GEMM1-only: consume accs, skip transpose/GEMM2
            #pragma unroll
            for(int i=0;i<QPT;i++)
                #pragma unroll
                for(int e=0;e<DPT;e++) acco[i][e]+=accs[i][e&(KPT-1)];
            continue;
        }
        __syncthreads();
        #pragma unroll
        for(int i=0;i<QPT;i++){ int qq=qb+ty*QPT+i;
            if constexpr(MODE==0){
                float sj[KPT],lmax=-1e30f;
                #pragma unroll
                for(int j=0;j<KPT;j++){ int key=kb+tx*KPT+j; float sv=accs[i][j]*scale2; if(diag&&key>qq+delta)sv=-1e30f; sj[j]=sv; lmax=fmaxf(lmax,sv); }
                lmax=fmaxf(lmax,__shfl_xor(lmax,1,16));lmax=fmaxf(lmax,__shfl_xor(lmax,2,16));lmax=fmaxf(lmax,__shfl_xor(lmax,4,16));lmax=fmaxf(lmax,__shfl_xor(lmax,8,16));
                float mnew=fmaxf(rm[i],lmax); float corr=(mnew>rm[i])?__builtin_amdgcn_exp2f(rm[i]-mnew):1.0f; float lsum=0;
                #pragma unroll
                for(int j=0;j<KPT;j++){ float pj=__builtin_amdgcn_exp2f(sj[j]-mnew); sj[j]=pj; lsum+=pj; }
                lsum+=__shfl_xor(lsum,1,16);lsum+=__shfl_xor(lsum,2,16);lsum+=__shfl_xor(lsum,4,16);lsum+=__shfl_xor(lsum,8,16);
                rl[i]=rl[i]*corr+lsum; rm[i]=mnew; if(corr!=1.0f){ for(int e=0;e<DPT;e++)acco[i][e]*=corr; }
                #pragma unroll
                for(int j=0;j<KPT;j++) PSA(tx*KPT+j,ty*QPT+i)=(_Float16)sj[j];
            } else {
                #pragma unroll
                for(int j=0;j<KPT;j++){ int key=kb+tx*KPT+j; float sv=accs[i][j]*scale2; if(diag&&key>qq+delta)sv=0.0f; PSA(tx*KPT+j,ty*QPT+i)=(_Float16)sv; }
            }
        }
        __syncthreads();
        #pragma unroll 4
        for(int kh=0;kh<BCH;kh++){ half2_t pv[QPT],vv[DPT];
            #pragma unroll
            for(int i=0;i<QPT;i++)pv[i]=half2_t{PSA(2*kh,ty*QPT+i),PSA(2*kh+1,ty*QPT+i)};
            #pragma unroll
            for(int e=0;e<DPT;e++)vv[e]=VSA(kh,tx*DPT+e);
            #pragma unroll
            for(int i=0;i<QPT;i++)
                #pragma unroll
                for(int e=0;e<DPT;e++)acco[i][e]=__builtin_amdgcn_fdot2(pv[i],vv[e],acco[i][e],false); }
    }
    _Float16* Ob=(_Float16*)p.O+orow0+(long long)h0*p.sO.h;
    #pragma unroll
    for(int i=0;i<QPT;i++){ int qq=qb+ty*QPT+i; if(qq>=SQ)continue; float inv=(MODE==0&&rl[i]>0.0f)?1.0f/rl[i]:1.0f;
        #pragma unroll
        for(int e=0;e<DPT;e++){ int d=tx*DPT+e; Ob[(long long)qq*p.sO.s+d]=(_Float16)(acco[i][e]*inv); } }
    #undef QSA
    #undef KSA
    #undef VSA
    #undef PSA
}

static void setup(FwdParams& p,int S,int NH,int gqa,int HD,void*&dQ,void*&dK,void*&dV,void*&dO){
    int NKV=NH/gqa; size_t nQ=(size_t)S*NH*HD, nK=(size_t)NKV*S*HD;
    dQ=alloc_fill(nQ);dK=alloc_fill(nK);dV=alloc_fill(nK);dO=alloc_fill(nQ);
    p=FwdParams{}; p.Q=dQ;p.K=dK;p.V=dV;p.O=dO;p.LSE=nullptr;
    p.sQ={0,(long long)HD,(long long)NH*HD}; p.sK={0,(long long)S*HD,(long long)HD}; p.sV=p.sK; p.sO=p.sQ;
    p.SQ=S;p.SK=S;p.HD=HD;p.NH=NH;p.gqa=gqa;p.scale=1.f/sqrtf((float)HD);
    p.softcap=0;p.window=-1;p.alibi_slopes=nullptr;p.causal=1;p.dropout_p=0;p.philox_seed=0;p.cu_q=nullptr;p.cu_k=nullptr;p.batch=1;p.sc_q=p.sc_k=p.sc_v=1.f;
}
static double tflops(int S,int NH,int HD,float ms,double mmul){ double pairs=(double)S*S/2; return mmul*NH*pairs*HD*2/(ms*1e-3)/1e12; }

template<int MODE>
static void bench_diag(int S){
    const int NH=32,gqa=4,HD=128,BR=128,BC=64; void*dQ,*dK,*dV,*dO; FwdParams p; setup(p,S,NH,gqa,HD,dQ,dK,dV,dO);
    dim3 grid((S+BR-1)/BR,NH,1);
    fa_diag_k<HD,BR,BC,MODE><<<grid,256>>>(p); HIP_CHECK(hipDeviceSynchronize());
    hipError_t e=hipGetLastError(); if(e!=hipSuccess){ printf("  MODE%d ERR %s\n",MODE,hipGetErrorString(e)); return; }
    hipEvent_t t0,t1; hipEventCreate(&t0);hipEventCreate(&t1); int it=S>=8192?10:20;
    hipEventRecord(t0); for(int i=0;i<it;i++) fa_diag_k<HD,BR,BC,MODE><<<grid,256>>>(p); hipEventRecord(t1); hipEventSynchronize(t1);
    float ms=0; hipEventElapsedTime(&ms,t0,t1); ms/=it;
    double mmul = (MODE==2)?1.0:2.0;
    const char* nm = MODE==0?"full          ":MODE==1?"no-softmax    ":"GEMM1-only    ";
    printf("  MODE%d %s: %.3f ms  ~%.1f TFLOP/s\n",MODE,nm,ms,tflops(S,NH,HD,ms,mmul));
    hipFree(dQ);hipFree(dK);hipFree(dV);hipFree(dO);
}
int main(){
    int Ss[2]={2048,8192};
    for(int si=0;si<2;si++){ int S=Ss[si];
        printf("S=%d HD128 GQA4 causal fp16 (MODE2 flops=1 matmul):\n",S);
        bench_diag<0>(S); bench_diag<1>(S); bench_diag<2>(S);
    }
    return 0;
}
