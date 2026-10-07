
// Standalone qualification helper.
//
// ROCm declares hipFree() [[nodiscard]].  Keep -Werror enabled and
// explicitly validate deallocation errors rather than suppressing the
// compiler diagnostic.
static inline void alu_forward_gate_free(void* ptr)
{
    if (ptr == nullptr)
        return;

    const hipError_t err = hipFree(ptr);

    if (err != hipSuccess) {
        printf(
            "fa_forward gate: hipFree failed: %s\n",
            hipGetErrorString(err)
        );
        abort();
    }
}

// fa_forward_gate.inc — standalone gate (fp64 oracle) + benchmark. Included with -DFA_STANDALONE.
// Test layout: Q,O = [SQ,NH,HD] (BSHD, batch=1); K,V = [NKV,SK,HD] (BHSD) — covers mixed strides.

static uint16_t f2bf16(float f){ uint32_t x; __builtin_memcpy(&x,&f,4);
    uint32_t r=x+0x7fffu+((x>>16)&1u); return (uint16_t)(r>>16); }
static float bf162f(uint16_t b){ uint32_t x=((uint32_t)b)<<16; float f; __builtin_memcpy(&f,&x,4); return f; }

// round input to dtype and back to float (for a fair oracle)
static void round_dtype(std::vector<float>& v,int dt){
    for(auto& x:v){ if(dt==FA_F16) x=(float)(_Float16)x; else if(dt==FA_BF16) x=bf162f(f2bf16(x)); }
}
// pack float array into a device buffer of the requested dtype
static void* to_dev(const std::vector<float>& v,int dt){
    void* d; size_t n=v.size();
    if(dt==FA_F32){ HIP_CHECK(hipMalloc(&d,n*4)); HIP_CHECK(hipMemcpy(d,v.data(),n*4,hipMemcpyHostToDevice)); return d; }
    if(dt==FA_F16){ std::vector<_Float16> h(n); for(size_t i=0;i<n;i++) h[i]=(_Float16)v[i];
        HIP_CHECK(hipMalloc(&d,n*2)); HIP_CHECK(hipMemcpy(d,h.data(),n*2,hipMemcpyHostToDevice)); return d; }
    std::vector<uint16_t> h(n); for(size_t i=0;i<n;i++) h[i]=f2bf16(v[i]);
    HIP_CHECK(hipMalloc(&d,n*2)); HIP_CHECK(hipMemcpy(d,h.data(),n*2,hipMemcpyHostToDevice)); return d;
}
static void from_dev(std::vector<float>& out,void* d,int dt){
    size_t n=out.size();
    if(dt==FA_F32){ HIP_CHECK(hipMemcpy(out.data(),d,n*4,hipMemcpyDeviceToHost)); return; }
    if(dt==FA_F16){ std::vector<_Float16> h(n); HIP_CHECK(hipMemcpy(h.data(),d,n*2,hipMemcpyDeviceToHost));
        for(size_t i=0;i<n;i++) out[i]=(float)h[i]; return; }
    std::vector<uint16_t> h(n); HIP_CHECK(hipMemcpy(h.data(),d,n*2,hipMemcpyDeviceToHost));
    for(size_t i=0;i<n;i++) out[i]=bf162f(h[i]);
}

// wl/wr — flash-attn window semantics: allow [qq+delta-wl .. qq+delta+wr], <0 = no boundary
struct TestSpec{ const char* name; int SQ,SK,HD,NH,gqa; bool causal,alibi; int wl,wr; float softcap; int chunk; };

static int run_test(const TestSpec& ts,int dt,double tol){
    int NKV=ts.NH/ts.gqa; float scale=1.0f/sqrtf((float)ts.HD);
    uint32_t s=1234567u+ts.HD*7+ts.NH+ts.SQ+dt*13;
    std::vector<float> Q((size_t)ts.SQ*ts.NH*ts.HD), K((size_t)NKV*ts.SK*ts.HD), V(K.size()), O(Q.size(),0.0f);
    std::vector<float> LSE((size_t)ts.NH*ts.SQ,0.0f);
    for(auto& x:Q)x=frand(s); for(auto& x:K)x=frand(s); for(auto& x:V)x=frand(s);
    round_dtype(Q,dt); round_dtype(K,dt); round_dtype(V,dt);
    std::vector<float> slopes(ts.NH);
    for(int h=0;h<ts.NH;h++) slopes[h]=-powf(2.0f,-(8.0f*(h+1)/ts.NH));  // ALiBi-like slopes

    // oracle
    std::vector<float> Oref(Q.size()), LSEref((size_t)ts.NH*ts.SQ);
    RefCfg rc{ts.SQ,ts.SK,ts.HD,ts.NH,ts.gqa,scale,ts.softcap,ts.wl,ts.wr,ts.causal,ts.alibi,slopes.data(),ts.chunk};
    ref_attention(Q.data(),K.data(),V.data(),Oref.data(),LSEref.data(),rc);

    void *dQ=to_dev(Q,dt),*dK=to_dev(K,dt),*dV=to_dev(V,dt),*dO=to_dev(O,dt); float* dLSE;
    HIP_CHECK(hipMalloc(&dLSE,LSE.size()*4));
    FwdParams p{}; p.Q=dQ;p.K=dK;p.V=dV;p.O=dO;p.LSE=dLSE;
    p.sQ={(long long)ts.SQ*ts.NH*ts.HD,(long long)ts.HD,(long long)ts.NH*ts.HD};
    p.sK={(long long)NKV*ts.SK*ts.HD,(long long)ts.SK*ts.HD,(long long)ts.HD};
    p.sV=p.sK; p.sO=p.sQ;
    p.SQ=ts.SQ;p.SK=ts.SK;p.HD=ts.HD;p.NH=ts.NH;p.gqa=ts.gqa;p.scale=scale;
    p.softcap=ts.softcap;p.win_left=ts.wl;p.win_right=ts.wr;p.alibi_slopes=ts.alibi?nullptr:nullptr;
    p.causal=ts.causal;p.dropout_p=0.0f;p.philox_seed=0;p.cu_q=nullptr;p.cu_k=nullptr;p.batch=1;
    p.attn_chunk=ts.chunk;
    float* dSlopes=nullptr;
    if(ts.alibi){ HIP_CHECK(hipMalloc(&dSlopes,ts.NH*4)); HIP_CHECK(hipMemcpy(dSlopes,slopes.data(),ts.NH*4,hipMemcpyHostToDevice)); p.alibi_slopes=dSlopes; }
    fa_forward_launch(p,dt,0);
    HIP_CHECK(hipDeviceSynchronize());
    from_dev(O,dO,dt);
    HIP_CHECK(hipMemcpy(LSE.data(),dLSE,LSE.size()*4,hipMemcpyDeviceToHost));
    double rel=rel_rms(O.data(),Oref.data(),O.size());
    // LSE. For a fully masked row the reference is -inf, and the kernel must match it.
    // Check empty rows separately so subtracting two negative infinities cannot produce NaN.
    // The current compiler policy preserves NaN/infinity semantics; retain the explicit threshold check.
    double le=0,lr=0; int bad_inf=0;
    for(size_t i=0;i<LSE.size();i++){
        bool ref_empty = !(LSEref[i] > -1e30f), got_empty = !(LSE[i] > -1e30f);
        if(ref_empty || got_empty){ if(ref_empty!=got_empty) bad_inf++; continue; }
        double d=LSE[i]-LSEref[i]; le+=d*d; lr+=LSEref[i]*LSEref[i]; }
    double lserel=lr>0?sqrt(le/lr):sqrt(le);
    const char* dn=dt==FA_F16?"fp16":dt==FA_BF16?"bf16":"fp32";
    bool ok=rel<tol && bad_inf==0;
    if(bad_inf) printf("      empty rows mismatched: %d\n",bad_inf);
    printf("  [%s/%-4s] %-26s rel_rms=%.2e  LSE_rel=%.2e  %s\n",dn,ts.name,"",rel,lserel,ok?"OK":"FAIL");
    alu_forward_gate_free(dQ);alu_forward_gate_free(dK);alu_forward_gate_free(dV);alu_forward_gate_free(dO);alu_forward_gate_free(dLSE); if(dSlopes)alu_forward_gate_free(dSlopes);
    return ok?0:1;
}

// ---- varlen ----
// used>0 => validate seqused_q/seqused_k: buffers contain sequences of length seqs[],
// but only the first used_q[i] queries and used_k[i] keys are actually used.
static int run_varlen(int dt,double tol,bool seqused=false,int HD=128,int NH=8,int gqa=4){
    int NKV=NH/gqa; float scale=1.0f/sqrtf((float)HD);
    int seqs[3]={100,64,200}; int B=3; int cq[4]={0,0,0,0}; for(int i=0;i<B;i++)cq[i+1]=cq[i]+seqs[i];
    int usedq[3]={70,64,150}, usedk[3]={90,30,180};
    int TOT=cq[B]; uint32_t s=999u+dt;
    std::vector<float> Q((size_t)TOT*NH*HD), K((size_t)NKV*TOT*HD), V(K.size()), O(Q.size(),0);
    for(auto&x:Q)x=frand(s); for(auto&x:K)x=frand(s); for(auto&x:V)x=frand(s);
    round_dtype(Q,dt);round_dtype(K,dt);round_dtype(V,dt);
    // oracle for each sequence independently
    std::vector<float> Oref(Q.size());
    for(int b=0;b<B;b++){ int off=cq[b];
        int LQ = seqused? usedq[b] : seqs[b];
        int LK = seqused? usedk[b] : seqs[b];
        // local strided views: Q[(off+i)*NH+h]... but the oracle expects compact storage — gather a slice
        std::vector<float> q((size_t)LQ*NH*HD),k((size_t)NKV*LK*HD),v(k.size()),o(q.size());
        for(int i=0;i<LQ;i++)for(int h=0;h<NH;h++)for(int d=0;d<HD;d++) q[(i*NH+h)*HD+d]=Q[((size_t)(off+i)*NH+h)*HD+d];
        for(int kh=0;kh<NKV;kh++)for(int j=0;j<LK;j++)for(int d=0;d<HD;d++){ k[(kh*LK+j)*HD+d]=K[((size_t)kh*TOT+off+j)*HD+d]; v[(kh*LK+j)*HD+d]=V[((size_t)kh*TOT+off+j)*HD+d]; }
        RefCfg rc{LQ,LK,HD,NH,gqa,scale,0,-1,-1,true,false,nullptr};
        ref_attention(q.data(),k.data(),v.data(),o.data(),nullptr,rc);
        // rows beyond seqused_q are not written by the kernel — they remain zero in Oref, as in O
        for(int i=0;i<LQ;i++)for(int h=0;h<NH;h++)for(int d=0;d<HD;d++) Oref[((size_t)(off+i)*NH+h)*HD+d]=o[(i*NH+h)*HD+d];
    }
    void *dQ=to_dev(Q,dt),*dK=to_dev(K,dt),*dV=to_dev(V,dt),*dO=to_dev(O,dt);
    int *dcq; HIP_CHECK(hipMalloc(&dcq,(B+1)*4)); HIP_CHECK(hipMemcpy(dcq,cq,(B+1)*4,hipMemcpyHostToDevice));
    FwdParams p{}; p.Q=dQ;p.K=dK;p.V=dV;p.O=dO;p.LSE=nullptr;
    p.sQ={0,(long long)HD,(long long)NH*HD}; p.sK={0,(long long)TOT*HD,(long long)HD}; p.sV=p.sK; p.sO=p.sQ;
    p.SQ=0;p.SK=0;p.HD=HD;p.NH=NH;p.gqa=gqa;p.scale=scale;p.softcap=0;p.win_left=-1;p.win_right=-1;p.alibi_slopes=nullptr;
    p.causal=1;p.dropout_p=0;p.philox_seed=0;p.cu_q=dcq;p.cu_k=dcq;p.batch=B;
    int *duq=nullptr,*duk=nullptr;
    if(seqused){ HIP_CHECK(hipMalloc(&duq,B*4)); HIP_CHECK(hipMemcpy(duq,usedq,B*4,hipMemcpyHostToDevice));
                 HIP_CHECK(hipMalloc(&duk,B*4)); HIP_CHECK(hipMemcpy(duk,usedk,B*4,hipMemcpyHostToDevice));
                 p.seqused_q=duq; p.seqused_k=duk; }
    // grid.x must cover the maximum sequence length
    int maxL=0; for(int i=0;i<B;i++) maxL=std::max(maxL,seqs[i]);
    // launch directly (dispatcher SQ = grid.x); use temporary SQ=maxL for the grid
    p.SQ=maxL;
    fa_forward_launch(p,dt,0);
    HIP_CHECK(hipDeviceSynchronize());
    from_dev(O,dO,dt);
    double rel=rel_rms(O.data(),Oref.data(),O.size());
    if(seqused && rel>=tol){     // per-sequence breakdown only when the gate fails
        for(int b=0;b<B;b++){ int off=cq[b];
            double r1=rel_rms(O.data()+(size_t)off*NH*HD, Oref.data()+(size_t)off*NH*HD,(size_t)usedq[b]*NH*HD);
            double r2=rel_rms(O.data()+(size_t)(off+usedq[b])*NH*HD, Oref.data()+(size_t)(off+usedq[b])*NH*HD,
                              (size_t)(seqs[b]-usedq[b])*NH*HD);
            printf("      seq%d used_q=%d used_k=%d: rows within used %.2e, tail %.2e\n",b,usedq[b],usedk[b],r1,r2); }
    }
    const char* dn=dt==FA_F16?"fp16":dt==FA_BF16?"bf16":"fp32";
    printf("  [%s/varlen%s] 3 seq (100,64,200) HD%d GQA%d  rel_rms=%.2e  %s\n",
           dn,seqused?"+seqused":"        ",HD,gqa,rel,rel<tol?"OK":"FAIL");
    alu_forward_gate_free(dQ);alu_forward_gate_free(dK);alu_forward_gate_free(dV);alu_forward_gate_free(dO);alu_forward_gate_free(dcq);
    if(duq){alu_forward_gate_free(duq);alu_forward_gate_free(duk);}
    return rel<tol?0:1;
}

static void bench(int dt,int S,int window){
    int NH=32,gqa=4,NKV=NH/gqa,HD=128; float scale=1.0f/sqrtf((float)HD);
    std::vector<float> Q((size_t)S*NH*HD,0.02f),K((size_t)NKV*S*HD,0.02f),V(K.size(),0.02f),O(Q.size());
    void*dQ=to_dev(Q,dt),*dK=to_dev(K,dt),*dV=to_dev(V,dt),*dO=to_dev(O,dt);
    FwdParams p{}; p.Q=dQ;p.K=dK;p.V=dV;p.O=dO;p.LSE=nullptr;
    p.sQ={0,(long long)HD,(long long)NH*HD}; p.sK={0,(long long)S*HD,(long long)HD}; p.sV=p.sK; p.sO=p.sQ;
    p.SQ=S;p.SK=S;p.HD=HD;p.NH=NH;p.gqa=gqa;p.scale=scale;p.softcap=0;
    p.win_left=(window<0?-1:window-1);p.win_right=(window<0?-1:0);p.alibi_slopes=nullptr;
    p.causal=1;p.dropout_p=0;p.philox_seed=0;p.cu_q=nullptr;p.cu_k=nullptr;p.batch=1;
    fa_forward_launch(p,dt,0); HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t t0,t1; HIP_CHECK(hipEventCreate(&t0)); HIP_CHECK(hipEventCreate(&t1));
    int iters = S>=8192?10:20;
    HIP_CHECK(hipEventRecord(t0));
    for(int i=0;i<iters;i++) fa_forward_launch(p,dt,0);
    HIP_CHECK(hipEventRecord(t1)); HIP_CHECK(hipEventSynchronize(t1));
    float ms=0; HIP_CHECK(hipEventElapsedTime(&ms,t0,t1)); ms/=iters;
    // FLOPs: full causal triangle, or window band
    double pairs = window<0? (double)S*S/2 : (double)S*window - (double)window*window/2;
    double tf=2.0*NH*pairs*HD*2/(ms*1e-3)/1e12;
    double kvGB=(double)NKV*S*HD*2*2/1e9;  // one K+V pass (lower-bound traffic estimate)
    const char* dn=dt==FA_F16?"fp16":"bf16";
    if(window<0) printf("  [%s] S=%5d HD128 GQA4 causal      : %.3f ms  ~%.1f TFLOP/s\n",dn,S,ms,tf);
    else         printf("  [%s] S=%5d HD128 GQA4 window=%-5d: %.3f ms  ~%.1f TFLOP/s (sparse path)\n",dn,S,window,ms,tf);
    alu_forward_gate_free(dQ);alu_forward_gate_free(dK);alu_forward_gate_free(dV);alu_forward_gate_free(dO);
}

int main(){
    int fails=0;
    printf("=== fa_forward: correctness gate (fp64 oracle) ===\n");
    // tolerances: fp16 2.6e-4, bf16 4e-3, fp32 1e-5 (half dtype; bf16 is looser)
    TestSpec base[]={
        {"d128 causal GQA4",   256,256,128,32,4, true,false,-1,-1,0.0f},
        {"d128 causal GQA1",   256,256,128, 8,1, true,false,-1,-1,0.0f},
        {"d128 full(noncausal)",192,192,128,16,4,false,false,-1,-1,0.0f},
        {"d128 SQ<SK bottomR", 128,320,128,16,4, true,false,-1,-1,0.0f},
        // With causal SQ>SK, the first SQ-SK rows see NO keys. flash-attn returns
        // zeros and LSE=-inf there. This was previously untested, and the kernel returned the sum of V.
        {"d128 SQ>SK empty rows",256,64,128,16,4,true,false,-1,-1,0.0f},
        // attention ranges: left-only (as before), bidirectional, right-only, exact diagonal
        {"d128 window (63,0)",   256,256,128,16,4, true,false,63, 0,0.0f},
        {"d128 window (32,32)",  256,256,128,16,4,false,false,32,32,0.0f},
        {"d128 window (0,0) diagonal",256,256,128,16,4,false,false, 0, 0,0.0f},
        {"d128 window (-1,0)=causal",256,256,128,16,4,false,false,-1,0,0.0f},
        {"d128 window (0,-1)",   256,256,128,16,4,false,false, 0,-1,0.0f},
        {"d128 window (16,48)",  192,192,128,16,4,false,false,16,48,0.0f},
        {"d128 window (32,32) SQ<SK",128,320,128,16,4,false,false,32,32,0.0f},
        {"d128 softcap=30",    200,200,128,16,4, true,false,-1,-1,30.0f},
        {"d128 alibi",         200,200,128,16,4, true,true ,-1,-1,0.0f},
        {"d64",                200,200, 64,16,4, true,false,-1,-1,0.0f},
        {"d80",                200,200, 80,16,4, true,false,-1,-1,0.0f},
        {"d96",                160,160, 96,16,4, true,false,-1,-1,0.0f},
        {"d256",               160,160,256, 8,4, true,false,-1,-1,0.0f},
        {"d32",                160,160, 32,16,4, true,false,-1,-1,0.0f},
        // new head_dim values: instantiated with BR=64,BC=32 (BR=128 does not fit in LDS)
        {"d112",               160,160,112,16,4, true,false,-1,-1,0.0f},
        {"d160",               160,160,160, 8,4, true,false,-1,-1,0.0f},
        {"d192",               160,160,192, 8,4, true,false,-1,-1,0.0f},
        {"d224",               160,160,224, 8,4, true,false,-1,-1,0.0f},
        {"d192 full",          128,192,192, 8,2,false,false,-1,-1,0.0f},
        // attention_chunk (FA-3): a query sees only keys in its own chunk.
        // Test both a tile-aligned chunk (64) and a non-aligned one (100) to catch alignment bugs.
        {"chunk=64 causal",    256,256,128,16,4, true,false,-1,-1,0.0f, 64},
        {"chunk=100 causal",   256,256,128,16,4, true,false,-1,-1,0.0f,100},
        {"chunk=128 full",     256,256,128,16,4,false,false,-1,-1,0.0f,128},
        {"chunk=64 SQ<SK",     128,320,128,16,4, true,false,-1,-1,0.0f, 64},
        // D512 (Gemma4 full-attention group): HD=512, GQA=2 (global), NH=32 -> 8 KV heads.
        // Dedicated D-split kernel (fa_forward_d512_k): BR=32, BC=32, LDS 50KB.
        // Score = full 512-dim dot accumulated across two 256-dim K chunks; one full softmax.
        // Matrix mirrors the d128 feature set so D512 is held to the same D256-era standard.
        {"d512 causal GQA2",   256,256,512,32,2, true,false,-1,-1,0.0f},
        {"d512 full GQA2",     192,192,512,32,2,false,false,-1,-1,0.0f},
        {"d512 causal long",   512,512,512,32,2, true,false,-1,-1,0.0f},
        {"d512 SQ<SK bottomR", 128,320,512,32,2, true,false,-1,-1,0.0f},
        {"d512 SQ>SK empty rows",256,64,512,16,2,true,false,-1,-1,0.0f},
        {"d512 window (63,0)",  256,256,512,16,2, true,false,63, 0,0.0f},
        {"d512 window (32,32)", 256,256,512,16,2,false,false,32,32,0.0f},
        {"d512 softcap=30",    200,200,512,16,2, true,false,-1,-1,30.0f},
        {"d512 alibi",         200,200,512,16,2, true,true ,-1,-1,0.0f},
    };
    for(int dt=0;dt<3;dt++){ double tol = dt==FA_F16?3e-4:dt==FA_BF16?5e-3:1e-5;
        for(auto& ts:base) fails+=run_test(ts,dt,tol);
        double vtol = dt==FA_F16?3e-4:dt==FA_BF16?5e-3:1e-5;
        fails+=run_varlen(dt, vtol, false);
        fails+=run_varlen(dt, vtol, true);
        // D512 varlen: the vLLM serving path (cu_seqlens, GQA2) must pass too.
        fails+=run_varlen(dt, vtol, false, 512, 32, 2);
        fails+=run_varlen(dt, vtol, true , 512, 32, 2);
    }
    printf("\n=== benchmark (one short run per shape) ===\n");
    bench(FA_F16,2048,-1); bench(FA_BF16,2048,-1);
    bench(FA_F16,8192,-1); bench(FA_BF16,8192,-1);
    bench(FA_F16,8192,512);   // sparse (sliding-window) — traffic and work scale with window, not S
    printf(fails?"\n*** FAILURES: %d ***\n":"\nALL GATES PASS\n",fails);
    return fails;
}
