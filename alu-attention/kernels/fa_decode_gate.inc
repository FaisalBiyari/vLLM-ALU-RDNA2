// fa_decode_gate.inc — standalone decode gate (fp64 oracle): paged, GQA, rotary, window, append,
// dtype, cache_leftpad, LSE, softcap, alibi, multiple new tokens per call, all head_dim values.
//
// Semantics match flash_attn_with_kvcache: ctx_len[b] is cache length BEFORE append
// (its cache_seqlens); new tokens are placed at ctx_len..ctx_len+nq-1, and rotary is read
// from table [seqlen_ro, rotary_dim/2] using the token position.
static uint16_t f2bf16d(float f){ uint32_t x; __builtin_memcpy(&x,&f,4); uint32_t r=x+0x7fffu+((x>>16)&1u); return (uint16_t)(r>>16); }
static float bf162fd(uint16_t b){ uint32_t x=((uint32_t)b)<<16; float f; __builtin_memcpy(&f,&x,4); return f; }
static void rnd_dt(std::vector<float>& v,int dt){ for(auto&x:v){ if(dt==FA_F16)x=(float)(_Float16)x; else if(dt==FA_BF16)x=bf162fd(f2bf16d(x)); } }
static void* dev_dt(const std::vector<float>& v,int dt){ void* d; size_t n=v.size();
    if(dt==FA_F32){ HIP_CHECK(hipMalloc(&d,n*4)); HIP_CHECK(hipMemcpy(d,v.data(),n*4,hipMemcpyHostToDevice)); return d; }
    if(dt==FA_F16){ std::vector<_Float16> h(n); for(size_t i=0;i<n;i++)h[i]=(_Float16)v[i]; HIP_CHECK(hipMalloc(&d,n*2)); HIP_CHECK(hipMemcpy(d,h.data(),n*2,hipMemcpyHostToDevice)); return d; }
    std::vector<uint16_t> h(n); for(size_t i=0;i<n;i++)h[i]=f2bf16d(v[i]); HIP_CHECK(hipMalloc(&d,n*2)); HIP_CHECK(hipMemcpy(d,h.data(),n*2,hipMemcpyHostToDevice)); return d; }
static void host_dt(std::vector<float>& o,void* d,int dt){ size_t n=o.size();
    if(dt==FA_F32){ HIP_CHECK(hipMemcpy(o.data(),d,n*4,hipMemcpyDeviceToHost)); return; }
    if(dt==FA_F16){ std::vector<_Float16> h(n); HIP_CHECK(hipMemcpy(h.data(),d,n*2,hipMemcpyDeviceToHost)); for(size_t i=0;i<n;i++)o[i]=(float)h[i]; return; }
    std::vector<uint16_t> h(n); HIP_CHECK(hipMemcpy(h.data(),d,n*2,hipMemcpyDeviceToHost)); for(size_t i=0;i<n;i++)o[i]=bf162fd(h[i]); }

// apply rotary to one row at position pos using table cs/sn[seqlen_ro][rdim/2]
// Test-harness-only: discard hipFree's nodiscard result (ROCm>=9 marks it [[nodiscard]]).
// Semantics identical to the previous bare hipFree(p) — the return value was never checked here.
static inline void hf(void* p){ (void)(hipFree(p)); }
static void rot_at(std::vector<float>& v,int off,int HD,int rdim,
                   const std::vector<float>& cs,const std::vector<float>& sn,int mode,int pos){
    const int half=rdim/2; const float* c=cs.data()+(size_t)pos*half; const float* s=sn.data()+(size_t)pos*half;
    if(mode==1){ for(int d=0;d<half;d++){ float x=v[off+d],y=v[off+d+half]; v[off+d]=x*c[d]-y*s[d]; v[off+d+half]=y*c[d]+x*s[d]; } }
    else if(mode==2){ for(int d=0;d<half;d++){ float x=v[off+2*d],y=v[off+2*d+1]; v[off+2*d]=x*c[d]-y*s[d]; v[off+2*d+1]=y*c[d]+x*s[d]; } }
}

struct DecCase{
    const char* name;
    int HD,NH,gqa,ctx;      // ctx — cache length BEFORE append (cache_seqlens)
    int window;             // <0 off (number of keys backward from query position)
    int rotary;             // 0 none, 1 neox, 2 interleaved
    bool append;            // whether to append new_k/new_v
    int leftpad;            // 0 => none
    int nq;                 // new tokens per call
    float softcap;          // <=0 off
    bool alibi;
};

static int dec_test(const DecCase& c,int dt,double tol){
    const int HD=c.HD,NH=c.NH,gqa=c.gqa,ctx=c.ctx,nq=c.nq;
    const int NKV=NH/gqa, page=16, batch=1;
    const int total = ctx + (c.append? nq : 0);           // positions actually present in cache
    const int max_blocks=(total+page-1)/page;
    const float scale=1.0f/sqrtf((float)HD);
    uint32_t sd=42u+HD+ctx+dt*7+c.rotary*3+(c.append?1:0)+nq*11+c.leftpad;
    int nsplit=calc_nsplit(total,gqa,NH,batch*nq);

    // rotary table for all positions
    const int seqro=total+8;
    std::vector<float> cs((size_t)seqro*(HD/2)), sn(cs.size());
    for(int p0=0;p0<seqro;p0++) for(int d=0;d<HD/2;d++){
        float ang=p0*powf(10000.0f,-2.0f*d/HD); cs[(size_t)p0*(HD/2)+d]=cosf(ang); sn[(size_t)p0*(HD/2)+d]=sinf(ang); }

    std::vector<float> q((size_t)nq*NH*HD); for(auto&x:q)x=frand(sd); rnd_dt(q,dt);
    std::vector<float> Kc((size_t)max_blocks*NKV*page*HD,0.0f), Vc(Kc.size(),0.0f);
    std::vector<float> newK((size_t)nq*NKV*HD), newV((size_t)nq*NKV*HD);
    for(int kh=0;kh<NKV;kh++) for(int pos=0;pos<ctx;pos++) for(int d=0;d<HD;d++){
        Kc[((size_t)pos*NKV+kh)*HD+d]=frand(sd);
        Vc[((size_t)pos*NKV+kh)*HD+d]=frand(sd); }
    if(c.append){ for(auto&x:newK)x=frand(sd); for(auto&x:newV)x=frand(sd); rnd_dt(newK,dt); rnd_dt(newV,dt); }
    rnd_dt(Kc,dt); rnd_dt(Vc,dt);
    std::vector<float> slopes(NH); for(int h=0;h<NH;h++) slopes[h]=-powf(2.0f,-(8.0f*(h+1)/NH));

    std::vector<int> bt(max_blocks); for(int i=0;i<max_blocks;i++) bt[i]=i;

    // ---------------- oracle ----------------
    std::vector<float> Oref((size_t)nq*NH*HD), LSEref((size_t)nq*NH);
    for(int tk=0;tk<nq;tk++){
        const int qpos=ctx+(c.append ? 0 : -nq)+tk;
        const int cend=std::max(0,qpos+1);        // keys 0..cend-1
        std::vector<float> qr((size_t)NH*HD);
        for(size_t i=0;i<qr.size();i++) qr[i]=q[(size_t)tk*NH*HD+i];
        if(c.rotary) for(int h=0;h<NH;h++) rot_at(qr,h*HD,HD,HD,cs,sn,c.rotary,qpos);
        int lo=0;
        if(c.window>=0) lo=std::max(lo,qpos-c.window+1);
        if(c.leftpad>0) lo=std::max(lo,c.leftpad);
        for(int h=0;h<NH;h++){ int kvh=h/gqa; std::vector<double> pr(cend>0?cend:1,-1e300); double mx=-1e300;
            for(int pos=0;pos<cend;pos++){ if(pos<lo) continue;
                std::vector<float> kk(HD);
                if(c.append && pos>=ctx){ int j=pos-ctx;
                    for(int d=0;d<HD;d++) kk[d]=newK[((size_t)j*NKV+kvh)*HD+d];
                    if(c.rotary) rot_at(kk,0,HD,HD,cs,sn,c.rotary,pos); }
                else for(int d=0;d<HD;d++) kk[d]=Kc[((size_t)pos*NKV+kvh)*HD+d];
                double s=0; for(int d=0;d<HD;d++) s+=(double)qr[h*HD+d]*kk[d]; s*=scale;
                if(c.softcap>0) s=c.softcap*tanh(s/c.softcap);
                if(c.alibi)     s+=slopes[h]*(double)(pos-qpos);
                pr[pos]=s; mx=fmax(mx,s); }
            double sum=0; for(int pos=0;pos<cend;pos++){ if(pr[pos]<=-1e299){pr[pos]=0;continue;} pr[pos]=exp(pr[pos]-mx); sum+=pr[pos]; }
            for(int d=0;d<HD;d++){ double acc=0;
                for(int pos=0;pos<cend;pos++){ if(pr[pos]==0) continue; float vv;
                    if(c.append && pos>=ctx) vv=newV[((size_t)(pos-ctx)*NKV+kvh)*HD+d];
                    else vv=Vc[((size_t)pos*NKV+kvh)*HD+d]; acc+=pr[pos]*vv; }
                Oref[((size_t)tk*NH+h)*HD+d]=(float)(sum>0? acc/sum : 0.0); }
            LSEref[(size_t)tk*NH+h]=(float)(sum>0? (mx+log(sum)) : -1e30);
        }
    }

    // ---------------- device ----------------
    std::vector<float> O((size_t)nq*NH*HD,0);
    void *dq=dev_dt(q,dt),*dKc=dev_dt(Kc,dt),*dVc=dev_dt(Vc,dt),*dO=dev_dt(O,dt);
    void *dnk=nullptr,*dnv=nullptr; if(c.append){ dnk=dev_dt(newK,dt); dnv=dev_dt(newV,dt); }
    int *dbt; HIP_CHECK(hipMalloc(&dbt,bt.size()*4)); HIP_CHECK(hipMemcpy(dbt,bt.data(),bt.size()*4,hipMemcpyHostToDevice));
    int *dctx; HIP_CHECK(hipMalloc(&dctx,4)); HIP_CHECK(hipMemcpy(dctx,&ctx,4,hipMemcpyHostToDevice));
    float *dcs,*dsn; HIP_CHECK(hipMalloc(&dcs,cs.size()*4)); HIP_CHECK(hipMalloc(&dsn,sn.size()*4));
    HIP_CHECK(hipMemcpy(dcs,cs.data(),cs.size()*4,hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dsn,sn.data(),sn.size()*4,hipMemcpyHostToDevice));
    float* dsc; HIP_CHECK(hipMalloc(&dsc,(size_t)batch*nq*NH*nsplit*(HD+2)*4));
    float* dlse; HIP_CHECK(hipMalloc(&dlse,(size_t)batch*nq*NH*4));
    int* dlp=nullptr; if(c.leftpad>0){ HIP_CHECK(hipMalloc(&dlp,4)); HIP_CHECK(hipMemcpy(dlp,&c.leftpad,4,hipMemcpyHostToDevice)); }
    float* dal=nullptr; if(c.alibi){ HIP_CHECK(hipMalloc(&dal,NH*4)); HIP_CHECK(hipMemcpy(dal,slopes.data(),NH*4,hipMemcpyHostToDevice)); }

    DecParams p{}; p.q=dq;p.Kc=dKc;p.Vc=dVc;p.block_table=dbt;p.o=dO;p.scratch=dsc;p.ctx_len=dctx;
    p.cos=dcs;p.sin=dsn;p.batch=batch;p.NH=NH;p.gqa=gqa;p.HD=HD;p.page=page;p.max_blocks=max_blocks;
    p.nsplit=nsplit;p.causal=1;p.window_right=-1;p.window=c.window;p.rotary=c.rotary;p.scale=scale;p.sc_q=1;p.sc_k=1;p.sc_v=1;
    p.leftpad=dlp; p.lse_out=dlse; p.softcap=c.softcap; p.alibi=dal; p.nq=nq; p.rotary_dim=0;
    fa_decode_launch(p,dt,0,dnk,dnv);
    HIP_CHECK(hipDeviceSynchronize());
    host_dt(O,dO,dt);
    std::vector<float> LSE((size_t)nq*NH); HIP_CHECK(hipMemcpy(LSE.data(),dlse,LSE.size()*4,hipMemcpyDeviceToHost));
    double rel=rel_rms(O.data(),Oref.data(),O.size());
    double lserel=rel_rms(LSE.data(),LSEref.data(),LSE.size());
    auto fin=[](double x){ return x>-1e300 && x<1e300; };
    bool ok = fin(rel)&&fin(lserel)&&rel<tol&&lserel<2e-3;
    const char* dn=dt==FA_F16?"fp16":dt==FA_BF16?"bf16":"fp32";
    printf("  [%s/%-26s] rel_rms=%.2e  LSE_rel=%.2e  %s\n",dn,c.name,rel,lserel,ok?"OK":"FAIL");
    hf(dq);hf(dKc);hf(dVc);hf(dO);hf(dbt);hf(dctx);hf(dcs);hf(dsn);
    hf(dsc);hf(dlse); if(dlp)hf(dlp); if(dal)hf(dal);
    if(dnk){hf(dnk);hf(dnv);}
    return ok?0:1;
}

// Gate bench: D512 (Gemma4 full-attention: NH=32, GQA=2) and D256 (sliding / general baseline)
// decode across the doc's contexts. Direct comparison vs the TRITON D512 reference (separate bench).
static void gate_bench(int dt,int HD,int NH,int gqa,const int* ctxs,int nctx){
    int NKV=NH/gqa,page=16,batch=1; float scale=1.0f/sqrtf((float)HD);
    for(int c=0;c<nctx;c++){ int ctx=ctxs[c]; int max_blocks=(ctx+page-1)/page, nsplit=calc_nsplit(ctx,gqa,NH,batch);
        std::vector<float> q((size_t)NH*HD,0.02f),Kc((size_t)max_blocks*NKV*page*HD,0.02f),Vc(Kc.size(),0.02f),O((size_t)NH*HD);
        std::vector<int> bt(max_blocks); for(int i=0;i<max_blocks;i++)bt[i]=i;
        void*dq=dev_dt(q,dt),*dKc=dev_dt(Kc,dt),*dVc=dev_dt(Vc,dt),*dO=dev_dt(O,dt);
        int*dbt;HIP_CHECK(hipMalloc(&dbt,bt.size()*4));HIP_CHECK(hipMemcpy(dbt,bt.data(),bt.size()*4,hipMemcpyHostToDevice));
        int*dctx;HIP_CHECK(hipMalloc(&dctx,4));HIP_CHECK(hipMemcpy(dctx,&ctx,4,hipMemcpyHostToDevice));
        float*dsc;HIP_CHECK(hipMalloc(&dsc,(size_t)NH*nsplit*(HD+2)*4));
        DecParams p{};p.q=dq;p.Kc=dKc;p.Vc=dVc;p.block_table=dbt;p.o=dO;p.scratch=dsc;p.ctx_len=dctx;
        p.batch=batch;p.NH=NH;p.gqa=gqa;p.HD=HD;p.page=page;p.max_blocks=max_blocks;p.nsplit=nsplit;
        p.window=-1;p.causal=1;p.window_right=-1;p.rotary=0;p.scale=scale;p.sc_q=1;p.sc_k=1;p.sc_v=1;p.nq=1;
        fa_decode_launch(p,dt,0,nullptr,nullptr); HIP_CHECK(hipDeviceSynchronize());
        hipEvent_t t0,t1;HIP_CHECK(hipEventCreate(&t0)); HIP_CHECK(hipEventCreate(&t1));
        int iters= ctx>=32768?20:50;
        HIP_CHECK(hipEventRecord(t0)); for(int i=0;i<iters;i++) fa_decode_launch(p,dt,0,nullptr,nullptr);
        HIP_CHECK(hipEventRecord(t1)); HIP_CHECK(hipEventSynchronize(t1));
        float ms=0;HIP_CHECK(hipEventElapsedTime(&ms,t0,t1)); ms/=iters;
        double gb=(double)NKV*ctx*HD*2*2/(ms*1e-3)/1e9;
        printf("  [%s] GATE ALU decode HD=%3d GQA%d ctx=%6d: %.4f ms  KV stream %.0f GB/s\n",dt==FA_F16?"fp16":"bf16",HD,gqa,ctx,ms,gb);
        hf(dq);hf(dKc);hf(dVc);hf(dO);hf(dbt);hf(dctx);hf(dsc);
    }
}
static void dec_bench(int dt){
    int HD=128,NH=32,gqa=4,NKV=NH/gqa,page=16,batch=1; float scale=1.0f/sqrtf((float)HD);
    for(int ctx:{2048,32768}){ int max_blocks=(ctx+page-1)/page, nsplit=calc_nsplit(ctx,gqa,NH,batch);
        std::vector<float> q((size_t)NH*HD,0.02f),Kc((size_t)max_blocks*NKV*page*HD,0.02f),Vc(Kc.size(),0.02f),O((size_t)NH*HD);
        std::vector<int> bt(max_blocks); for(int i=0;i<max_blocks;i++)bt[i]=i;
        void*dq=dev_dt(q,dt),*dKc=dev_dt(Kc,dt),*dVc=dev_dt(Vc,dt),*dO=dev_dt(O,dt);
        int*dbt;HIP_CHECK(hipMalloc(&dbt,bt.size()*4));HIP_CHECK(hipMemcpy(dbt,bt.data(),bt.size()*4,hipMemcpyHostToDevice));
        int*dctx;HIP_CHECK(hipMalloc(&dctx,4));HIP_CHECK(hipMemcpy(dctx,&ctx,4,hipMemcpyHostToDevice));
        float*dsc;HIP_CHECK(hipMalloc(&dsc,(size_t)NH*nsplit*(HD+2)*4));
        DecParams p{};p.q=dq;p.Kc=dKc;p.Vc=dVc;p.block_table=dbt;p.o=dO;p.scratch=dsc;p.ctx_len=dctx;
        p.batch=batch;p.NH=NH;p.gqa=gqa;p.HD=HD;p.page=page;p.max_blocks=max_blocks;p.nsplit=nsplit;
        p.window=-1;p.causal=1;p.window_right=-1;p.rotary=0;p.scale=scale;p.sc_q=1;p.sc_k=1;p.sc_v=1;p.nq=1;
        fa_decode_launch(p,dt,0,nullptr,nullptr); HIP_CHECK(hipDeviceSynchronize());
        hipEvent_t t0,t1;HIP_CHECK(hipEventCreate(&t0));HIP_CHECK(hipEventCreate(&t1));
        HIP_CHECK(hipEventRecord(t0)); for(int i=0;i<50;i++) fa_decode_launch(p,dt,0,nullptr,nullptr);
        HIP_CHECK(hipEventRecord(t1));HIP_CHECK(hipEventSynchronize(t1));
        float ms=0;HIP_CHECK(hipEventElapsedTime(&ms,t0,t1)); ms/=50;
        double gb=(double)NKV*ctx*HD*2*2/(ms*1e-3)/1e9;
        printf("  [%s] decode ctx=%6d: %.4f ms  KV stream %.0f GB/s (%.0f%% of 512)\n",dt==FA_F16?"fp16":"bf16",ctx,ms,gb,gb/512*100);
        hf(dq);hf(dKc);hf(dVc);hf(dO);hf(dbt);hf(dctx);hf(dsc);
    }
}

int main(){
    int fails=0;
    printf("=== fa_decode: correctness gate (fp64 oracle) ===\n");
    // {name, HD,NH,gqa,ctx, window, rotary, append, leftpad, nq, softcap, alibi}
    DecCase cases[]={
        {"paged GQA4 ctx333",      128,32,4,333, -1,0,false,0,1, 0.f,false},
        {"paged GQA1 ctx200",      128, 8,1,200, -1,0,false,0,1, 0.f,false},
        {"rotary-neox ctx256",     128,32,4,256, -1,1,false,0,1, 0.f,false},
        {"rotary-interl ctx256",   128,32,4,256, -1,2,false,0,1, 0.f,false},
        {"window=64 ctx500",       128,32,4,500, 64,0,false,0,1, 0.f,false},
        {"append+rotary ctx257",   128,32,4,257, -1,1,true ,0,1, 0.f,false},
        {"leftpad=100 ctx400",     128,32,4,400, -1,0,false,100,1,0.f,false},
        {"leftpad=37 ctx333",      128, 8,1,333, -1,0,false, 37,1,0.f,false},
        {"leftpad+window ctx500",  128,32,4,500, 64,0,false, 50,1,0.f,false},
        // softcap and alibi in decode (Gemma-2 decodes with softcap)
        {"softcap=30 ctx333",      128,32,4,333, -1,0,false,0,1,30.f,false},
        {"softcap=50 append",      128,32,4,257, -1,0,true ,0,1,50.f,false},
        {"alibi ctx333",           128,32,4,333, -1,0,false,0,1, 0.f,true },
        {"alibi+window ctx500",    128,32,4,500,128,0,false,0,1, 0.f,true },
        // multiple new tokens per call (speculative / chunked prefill)
        {"nq=4 without append",      128,32,4,300, -1,0,false,0,4, 0.f,false},
        {"nq=4 with append",       128,32,4,300, -1,0,true ,0,4, 0.f,false},
        {"nq=8 append+rotary",   128,32,4,257, -1,1,true ,0,8, 0.f,false},
        {"nq=5 GQA1 window",       128, 8,1,400, 96,0,true ,0,5, 0.f,false},
        {"nq=3 softcap+alibi",     128,32,4,300, -1,0,true ,0,3,30.f,true },
        // all head_dim values
        {"HD64 ctx300",             64,32,4,300, -1,0,false,0,1, 0.f,false},
        {"HD256 ctx300",           256, 8,4,300, -1,0,false,0,1, 0.f,false},
        {"HD16 ctx200",             16,32,4,200, -1,0,false,0,1, 0.f,false},
        {"HD32 ctx200",             32,32,4,200, -1,0,false,0,1, 0.f,false},
        {"HD48 ctx200",             48,32,4,200, -1,0,false,0,1, 0.f,false},
        {"HD80 ctx200",             80,32,4,200, -1,0,false,0,1, 0.f,false},
        {"HD96 ctx200",             96,32,4,200, -1,0,false,0,1, 0.f,false},
        {"HD112 ctx200",           112,32,4,200, -1,0,false,0,1, 0.f,false},
        {"HD160 ctx200",           160, 8,4,200, -1,0,false,0,1, 0.f,false},
        {"HD192 ctx200",           192, 8,4,200, -1,0,false,0,1, 0.f,false},
        {"HD224 ctx200",           224, 8,4,200, -1,0,false,0,1, 0.f,false},
        // D512 (Gemma4 full-attention): HD=512, GQA=2 (global), NH=32 -> 8 KV heads.
        // Register-resident decode; K row = 256 half2 registers (may spill).
        {"HD512 ctx300 GQA2",       512,32,2,300, -1,0,false,0,1, 0.f,false},
        {"HD512 ctx4096 GQA2",      512,32,2,4096,-1,0,false,0,1, 0.f,false},
        {"HD512 append+rotary",     512,32,2,257, -1,1,true ,0,1, 0.f,false},
        {"HD512 window=2048",       512,32,2,3000,2048,0,false,0,1, 0.f,false},
    };
    for(int dt=0;dt<3;dt++){ double tol=dt==FA_F16?3e-3:dt==FA_BF16?8e-3:1e-5;
        for(const auto& c:cases) fails+=dec_test(c,dt,tol);
    }
    printf("\n=== decode benchmark (split-KV, paged fp16/bf16) ===\n");
    dec_bench(FA_F16); dec_bench(FA_BF16);
    printf("\n=== GATE decode: ALU D512 (Gemma4 full) vs D256 (sliding) baseline, fp16, pinned clocks ===\n");
    { const int ctxs3[]={512,4096,32768};
      gate_bench(FA_F16,512,32,2,ctxs3,3);   // Gemma4 full-attention group
      gate_bench(FA_F16,256,32,2,ctxs3,3);   // Gemma4 sliding group (proven ALU territory)
    }
    printf(fails?"\n*** FAILURES: %d ***\n":"\nALL GATES PASS\n",fails);
    return fails;
}
