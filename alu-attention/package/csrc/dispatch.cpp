#include "alu_compiler_policy.h" // ALU_COMPILER_HARDENING_POLICY
// dispatch.cpp — host-side torch wrapper around the HIP kernels: fa_forward_launch for forward and
// alu_attn_backward_launch for backward. Exposes torch.ops.alu_attn.{fwd,bwd}. ROCm intentionally
// uses the CUDA dispatch key for HIP tensors in torch; this is standard behavior, not a workaround.
#include <torch/extension.h>
#include <cmath>
#include <climits>
#include <c10/core/DeviceGuard.h>
#include "../../kernels/fa_decode_schedule.hpp"
#include "fa_iface.h"

// HIP runtime calls are isolated from this host-only torch binding.
extern "C" void* alu_current_stream(long long dev);
extern "C" int alu_device_cu_count(long long dev);
extern "C" int alu_last_launch_error();


static int dt_of(const torch::Tensor& t){
    if (t.scalar_type()==torch::kHalf)     return 0;
    if (t.scalar_type()==torch::kBFloat16) return 1;
    return 2; // float32
}
static int bf16_shift(const torch::Tensor& t, int dt){
    if (dt!=1) return 0;
    float m = t.abs().max().item<float>();
    if (m<=0.f) return 0;
    int e; std::frexp(m,&e);   // m in [2^(e-1),2^e)
    return 13 - e;
}
static void* cur_stream(const torch::Tensor& t){
    return alu_current_stream((long long)t.get_device());
}

// ============================ FORWARD ============================
// q:[B,Hq,SQ,D], k/v:[B,Hk,SK,D], contiguous in D. Returns (out[B,Hq,SQ,D], lse[B,Hq,SQ] fp32).
static std::vector<torch::Tensor> fwd(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    double scale, bool causal, int64_t win_left, int64_t win_right, double softcap,
    c10::optional<torch::Tensor> alibi, double dropout_p, int64_t seed,
    int64_t attn_chunk, c10::optional<torch::Tensor> out_opt)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "GPU tensors are required");
    c10::DeviceGuard device_guard(q.device());
    TORCH_CHECK(q.dim()==4 && k.dim()==4 && v.dim()==4, "expected [B,H,S,D]");
    const int B=q.size(0), Hq=q.size(1), SQ=q.size(2), D=q.size(3);
    const int Hk=k.size(1), SK=k.size(2);
    TORCH_CHECK(Hq%Hk==0, "Hq must be divisible by Hk (GQA)");
    const int dt = dt_of(q), gqa = Hq/Hk;
    // out can be supplied by the wrapper in (B,S,H,D) layout so callers receive a CONTIGUOUS tensor,
    // matching real flash-attn, rather than a transposed view on which .view() fails. The kernel
    // accepts arbitrary output strides through separate stride fields.
    auto out = out_opt.has_value() ? *out_opt : torch::empty_like(q);
    if (out_opt.has_value()) {
        TORCH_CHECK(out.sizes()==q.sizes(), "out must have q shape [B,H,SQ,D]");
        TORCH_CHECK(out.scalar_type()==q.scalar_type(), "out must have the same dtype as q");
        TORCH_CHECK(out.stride(3)==1, "out: the last dimension must be contiguous");
    }
    auto lse = torch::empty({B,Hq,SQ}, q.options().dtype(torch::kFloat32));
    const float* al = alibi.has_value() ? alibi->data_ptr<float>() : nullptr;
    void* st = cur_stream(q);
    const size_t esz = q.element_size();
    auto S=[&](const torch::Tensor& x){ return Strides{ x.stride(0), x.stride(1), x.stride(2) }; };
    // Forward writes LSE[h*SQ+qq] without a batch offset, so launch one batch at a time; O remains correct.
    for (int b=0; b<B; b++){
        FwdParams p{};
        p.Q=(const char*)q.data_ptr()+ (size_t)b*q.stride(0)*esz;
        p.K=(const char*)k.data_ptr()+ (size_t)b*k.stride(0)*esz;
        p.V=(const char*)v.data_ptr()+ (size_t)b*v.stride(0)*esz;
        p.O=(char*)out.data_ptr()   + (size_t)b*out.stride(0)*esz;
        p.LSE=lse.data_ptr<float>() + (size_t)b*Hq*SQ;
        p.sQ=S(q); p.sK=S(k); p.sV=S(v); p.sO=S(out);
        p.SQ=SQ; p.SK=SK; p.HD=D; p.NH=Hq; p.gqa=gqa;
        p.scale=(float)scale; p.softcap=(float)softcap;
        p.win_left=(int)win_left; p.win_right=(int)win_right;
        p.alibi_slopes=al; p.causal=causal?1:0;
        p.dropout_p=(float)dropout_p; p.philox_seed=(unsigned long long)seed;
        p.cu_q=nullptr; p.cu_k=nullptr; p.batch=1;
        p.attn_chunk=(int)attn_chunk; p.seqused_q=nullptr; p.seqused_k=nullptr;
        fa_forward_launch(p, dt, st);
    }
    return {out, lse};
}

// ============================ BACKWARD ============================
static std::vector<torch::Tensor> bwd(
    torch::Tensor dout, torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor out, torch::Tensor lse,
    double scale, bool causal, int64_t win_left, int64_t win_right, double softcap,
    c10::optional<torch::Tensor> alibi, double dropout_p, int64_t seed)
{
    c10::DeviceGuard device_guard(q.device());
    const int B=q.size(0), Hq=q.size(1), SQ=q.size(2), D=q.size(3);
    const int Hk=k.size(1), SK=k.size(2);
    const int dt = dt_of(q), gqa = Hq/Hk;
    auto dq = torch::empty_like(q), dk = torch::empty_like(k), dv = torch::empty_like(v);
    auto dsum = torch::empty({B,Hq,SQ}, q.options().dtype(torch::kFloat32));
    const float* al = alibi.has_value() ? alibi->data_ptr<float>() : nullptr;
    void* st = cur_stream(q);
    auto S=[&](const torch::Tensor& x){ return Str{ x.stride(0), x.stride(1), x.stride(2) }; };
    BwdParams pr{};
    pr.Q=q.data_ptr(); pr.K=k.data_ptr(); pr.V=v.data_ptr(); pr.O=out.data_ptr(); pr.dO=dout.data_ptr();
    pr.dQ=dq.data_ptr(); pr.dK=dk.data_ptr(); pr.dV=dv.data_ptr();
    pr.lse=lse.data_ptr<float>(); pr.dsum=dsum.data_ptr<float>();
    pr.qs=S(q); pr.ks=S(k); pr.vs=S(v); pr.os=S(out); pr.dos=S(dout);
    pr.dqs=S(dq); pr.dks=S(dk); pr.dvs=S(dv);
    pr.B=B; pr.Hq=Hq; pr.Hk=Hk; pr.SQ=SQ; pr.SK=SK; pr.D=D; pr.gqa=gqa;
    pr.scale=(float)scale; pr.softcap=(float)softcap; pr.alibi=al;
    pr.causal=causal?1:0; pr.win_left=(int)win_left; pr.win_right=(int)win_right;
    pr.dropout=(dropout_p>0)?1:0; pr.drop_seed=(unsigned long long)seed;
    pr.dropout_p=(float)dropout_p; pr.keep_scale=(dropout_p>0)?(float)(1.0/(1.0-dropout_p)):1.f;
    pr.shQ=bf16_shift(q,dt); pr.shK=bf16_shift(k,dt); pr.shV=bf16_shift(v,dt); pr.shdO=bf16_shift(dout,dt);
    alu_attn_backward_launch(pr, dt, st);
    return {dq, dk, dv};
}

// ============================ VARLEN FORWARD ============================
// The kernel handles cu_seqlens natively through FwdParams.cu_q/cu_k. The older wrapper did not use
// this path and launched a Python loop once per sequence. q,k,v are
// (total_tokens, nheads, headdim). LSE is not written because only dense backward needs it.
static torch::Tensor fwd_varlen(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor cu_q, torch::Tensor cu_k, int64_t max_sq, int64_t max_sk,
    double scale, bool causal, int64_t win_left, int64_t win_right, double softcap,
    c10::optional<torch::Tensor> alibi, double dropout_p, int64_t seed,
    int64_t attn_chunk,
    c10::optional<torch::Tensor> seqused_q, c10::optional<torch::Tensor> seqused_k)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "GPU tensors are required");
    c10::DeviceGuard device_guard(q.device());
    TORCH_CHECK(q.dim()==3 && k.dim()==3 && v.dim()==3, "varlen expects [total, H, D]");
    const int Hq=q.size(1), D=q.size(2), Hk=k.size(1);
    TORCH_CHECK(Hq%Hk==0, "Hq must be divisible by Hk (GQA)");
    const int dt = dt_of(q);
    const int batch = (int)cu_q.numel() - 1;

    // With seqused_q, the kernel does not write rows beyond the effective length. empty_like would
    // expose uninitialized data (observed as NaN), so zero the tensor only for this case. The normal
    // varlen path avoids the extra memset.
    auto out = seqused_q.has_value() ? torch::zeros_like(q) : torch::empty_like(q);
    const float* al = alibi.has_value() ? alibi->data_ptr<float>() : nullptr;

    FwdParams p{};
    p.Q=q.data_ptr(); p.K=k.data_ptr(); p.V=v.data_ptr(); p.O=out.data_ptr(); p.LSE=nullptr;
    // (total,H,D): token stride = H*D, head stride = D.
    p.sQ=Strides{0, (long long)D, (long long)Hq*D};
    p.sK=Strides{0, (long long)D, (long long)Hk*D};
    p.sV=p.sK;
    p.sO=p.sQ;
    p.SQ=(int)max_sq; p.SK=(int)max_sk; p.HD=D; p.NH=Hq; p.gqa=Hq/Hk;
    p.scale=(float)scale; p.softcap=(float)softcap;
        p.win_left=(int)win_left; p.win_right=(int)win_right;
    p.alibi_slopes=al; p.causal=causal?1:0;
    p.dropout_p=(float)dropout_p; p.philox_seed=(unsigned long long)seed;
    p.cu_q=cu_q.data_ptr<int>(); p.cu_k=cu_k.data_ptr<int>();
    p.batch=batch;
    p.attn_chunk=(int)attn_chunk;
    p.seqused_q = seqused_q.has_value() ? seqused_q->data_ptr<int>() : nullptr;
    p.seqused_k = seqused_k.has_value() ? seqused_k->data_ptr<int>() : nullptr;
    fa_forward_launch(p, dt, cur_stream(q));
    return out;
}

// ============================ DECODE (KV cache) ============================
// Rectangular q: [B,NQ,Hq,D]. Packed q: [T,Hq,D] with GPU query_start_loc[B+1].
// Cache views always expose logical [blocks,page,Hkv,D] axes; actual strides may be
// token-major, head-major, or interleaved. No cache repacking or metadata readback.
static void decode_tensor(const torch::Tensor& t, const torch::Tensor& q,
                          c10::ScalarType dtype, const char* name, bool contiguous=true) {
    TORCH_CHECK(t.is_cuda() && t.device()==q.device(), name, " must be on the query GPU");
    TORCH_CHECK(t.scalar_type()==dtype, name, " has an unsupported dtype");
    TORCH_CHECK(!contiguous || t.is_contiguous(), name, " must be contiguous");
}
static bool decode_dim(int d) {
    switch(d) {
        case 16: case 32: case 48: case 64: case 80: case 96:
        case 112: case 128: case 160: case 192: case 224: case 256: case 512: return true; // D512_ENABLEMENT_MARKER (Gemma4 full-attention group)
        default: return false;
    }
}
static std::vector<torch::Tensor> dec(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor cache_seqlens,
    c10::optional<torch::Tensor> new_k, c10::optional<torch::Tensor> new_v,
    c10::optional<torch::Tensor> rotary_cos, c10::optional<torch::Tensor> rotary_sin,
    bool rotary_interleaved, double scale, int64_t window, int64_t num_splits,
    c10::optional<torch::Tensor> cache_leftpad, bool return_lse,
    double softcap, c10::optional<torch::Tensor> alibi,
    int64_t rotary_dim, c10::optional<torch::Tensor> rotary_pos,
    int64_t kv_row, int64_t v_off,
    c10::optional<torch::Tensor> query_start_loc, bool causal, int64_t window_right,
    int64_t attention_chunk, c10::optional<torch::Tensor> output,
    c10::optional<torch::Tensor> workspace, c10::optional<torch::Tensor> lse_buffer)
{
    TORCH_CHECK(q.is_cuda(), "ALU Attention decode requires a HIP/GPU query tensor");
    c10::DeviceGuard device_guard(q.device());
    TORCH_CHECK(q.scalar_type()==torch::kHalf || q.scalar_type()==torch::kBFloat16 ||
                q.scalar_type()==torch::kFloat32, "decode supports fp16, bf16, and fp32 only");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    const bool packed=query_start_loc.has_value();
    TORCH_CHECK(q.dim()==(packed?3:4), "expected packed [T,Hq,D] with query_start_loc, or [B,NQ,Hq,D]");
    TORCH_CHECK(k_cache.dim()==4 && v_cache.dim()==4, "cache expects logical [blocks,page,Hkv,D] axes");
    decode_tensor(k_cache,q,q.scalar_type(),"k_cache",false);
    decode_tensor(v_cache,q,q.scalar_type(),"v_cache",false);
    TORCH_CHECK(k_cache.stride(3)==1 && v_cache.stride(3)==1, "cache head dimension must have stride 1");
    for(int i=0;i<3;i++) {
        TORCH_CHECK(k_cache.size(i)==v_cache.size(i), "K/V cache leading dimensions must match");
        TORCH_CHECK(k_cache.size(i)>0 && k_cache.size(i)<=INT_MAX/4, "cache dimension out of range");
        TORCH_CHECK(k_cache.stride(i)>0 && v_cache.stride(i)>0, "broadcast cache views are not supported");
    }
    const int Hq=q.size(q.dim()-2), D=q.size(q.dim()-1);
    const int page=k_cache.size(1), Hk=k_cache.size(2);
    TORCH_CHECK(Hq>0 && Hq<=65535 && decode_dim(D), "unsupported query heads or head dimension");
    TORCH_CHECK(Hq%Hk==0, "Hq must be divisible by Hkv");
    const int gqa=Hq/Hk;
    TORCH_CHECK(gqa==1 || gqa==2 || gqa==3 || gqa==4 || gqa==6 || gqa==8,
                "supported decode GQA ratios are 1,2,3,4,6,8");
    TORCH_CHECK((page & (page-1))==0, "page size must be a positive power of two");
    TORCH_CHECK(kv_row>=0 && kv_row<=INT_MAX/4 && v_off>=0 && v_off<=INT_MAX/4,
                "invalid interleaved row stride/offset");
    TORCH_CHECK(k_cache.size(3)>=D && v_cache.size(3)>=D+v_off, "cache content dimension is too small");
    if(kv_row>0) {
        TORCH_CHECK(kv_row>=D+v_off && (Hk==1 || k_cache.stride(2)==kv_row),
                    "kv_row disagrees with the cache view; use logical [blocks,page,Hkv,content] axes");
    }
    // The 16-bit kernels use 16-byte vector loads. Reject misaligned views rather
    // than silently copying a live cache (which would break append semantics).
    if(q.scalar_type()!=torch::kFloat32) {
        TORCH_CHECK(reinterpret_cast<uintptr_t>(k_cache.data_ptr())%16==0 &&
                    (reinterpret_cast<uintptr_t>(v_cache.data_ptr())+v_off*q.element_size())%16==0,
                    "16-bit cache row bases must be 16-byte aligned");
        for(int i=0;i<3;i++) if(k_cache.size(i)>1)
            TORCH_CHECK((k_cache.stride(i)*q.element_size())%16==0 &&
                        (v_cache.stride(i)*q.element_size())%16==0,
                        "16-bit cache strides must preserve 16-byte row alignment");
    }
    TORCH_CHECK(std::isfinite(scale) && std::isfinite(softcap) && softcap>=0,
                "scale must be finite and softcap nonnegative");
    TORCH_CHECK(window>=-1 && window<=INT_MAX/4 && window_right>=-1 && window_right<=INT_MAX/4,
                "invalid attention window");
    TORCH_CHECK(attention_chunk>=0 && attention_chunk<=INT_MAX/4, "invalid attention_chunk");
    TORCH_CHECK(num_splits>=0 && num_splits<=512, "num_splits must be between 0 and 512");
    decode_tensor(block_table,q,torch::kInt32,"block_table",false);
    TORCH_CHECK(block_table.dim()==2 && block_table.stride(1)==1 &&
                block_table.stride(0)>=block_table.size(1) && block_table.stride(0)<=INT_MAX,
                "block_table must be a nonoverlapping int32 matrix with contiguous rows");
    const int64_t B=packed ? query_start_loc->numel()-1 : q.size(0);
    const int64_t NQ=packed ? 1 : q.size(1);
    const int64_t T=packed ? q.size(0) : B*NQ;
    TORCH_CHECK(B>0 && B<=INT_MAX/4 && NQ>=0 && T<=INT_MAX/4, "invalid decode batch size");
    TORCH_CHECK(block_table.size(0)==B && block_table.size(1)>0, "one nonempty block-table row is required per request");
    const int64_t ctx_bound=block_table.size(1)*(int64_t)page;
    TORCH_CHECK(ctx_bound<=INT_MAX/4, "cache address space exceeds the decode index range");
    decode_tensor(cache_seqlens,q,torch::kInt32,"cache_seqlens");
    TORCH_CHECK(cache_seqlens.dim()==1 && cache_seqlens.numel()==B, "cache_seqlens must have shape [B]");
    if(packed) {
        decode_tensor(*query_start_loc,q,torch::kInt32,"query_start_loc");
        TORCH_CHECK(query_start_loc->dim()==1, "query_start_loc must have shape [B+1]");
    }
    auto int_meta=[&](const c10::optional<torch::Tensor>& value, const char* name){
        if(value.has_value()) {
            decode_tensor(*value,q,torch::kInt32,name);
            TORCH_CHECK(value->dim()==1 && value->numel()==B, name, " must have shape [B]");
        }
    };
    int_meta(cache_leftpad,"cache_leftpad"); int_meta(rotary_pos,"rotary_pos");
    TORCH_CHECK(new_k.has_value()==new_v.has_value(), "new_k and new_v must be supplied together");
    if(new_k.has_value()) {
        decode_tensor(*new_k,q,q.scalar_type(),"new_k");
        decode_tensor(*new_v,q,q.scalar_type(),"new_v");
        auto expected=q.sizes().vec(); expected[expected.size()-2]=Hk;
        TORCH_CHECK(new_k->sizes()==c10::IntArrayRef(expected) && new_v->sizes()==new_k->sizes(),
                    "new K/V must match the query token layout with Hkv heads");
        TORCH_CHECK(!new_k->is_alias_of(k_cache) && !new_k->is_alias_of(v_cache) &&
                    !new_v->is_alias_of(k_cache) && !new_v->is_alias_of(v_cache),
                    "cache append inputs must not alias the destination cache");
    }
    TORCH_CHECK(rotary_cos.has_value()==rotary_sin.has_value(), "rotary cos/sin must be supplied together");
    if(rotary_cos.has_value()) {
        decode_tensor(*rotary_cos,q,torch::kFloat32,"rotary_cos");
        decode_tensor(*rotary_sin,q,torch::kFloat32,"rotary_sin");
        TORCH_CHECK(rotary_cos->dim()==2 && rotary_cos->sizes()==rotary_sin->sizes(), "invalid rotary tables");
        if(rotary_dim==0) rotary_dim=D;
        TORCH_CHECK(rotary_dim>0 && rotary_dim<=D && rotary_dim%2==0 &&
                    rotary_cos->size(1)==rotary_dim/2, "rotary table width does not match rotary_dim");
    }
    if(alibi.has_value()) {
        decode_tensor(*alibi,q,torch::kFloat32,"alibi");
        TORCH_CHECK(alibi->dim()==1 && alibi->numel()==Hq, "decode ALiBi supports shape [Hq] only");
    }
    const int nsplit=alu_decode_splits(T,Hk,ctx_bound,
                                      num_splits>0 ? 0 : alu_device_cu_count(q.get_device()),
                                      (int)num_splits);
    const int64_t scratch_count=alu_decode_scratch_elements(T,Hq,D,nsplit);
    auto out=output.has_value() ? *output : torch::empty(q.sizes(),q.options());
    decode_tensor(out,q,q.scalar_type(),"output");
    TORCH_CHECK(out.sizes()==q.sizes(), "output must have the query shape");
    auto scratch=workspace.has_value() ? *workspace : torch::empty({scratch_count},q.options().dtype(torch::kFloat32));
    decode_tensor(scratch,q,torch::kFloat32,"workspace");
    TORCH_CHECK(scratch.numel()>=scratch_count, "workspace too small; rebuild it for this token capacity and split count");
    auto lse_shape=q.sizes().vec(); lse_shape.pop_back();
    TORCH_CHECK(!lse_buffer.has_value() || return_lse, "lse_buffer requires return_lse=True");
    auto lse=lse_buffer.has_value() ? *lse_buffer : torch::empty(return_lse ? lse_shape : std::vector<int64_t>{0},q.options().dtype(torch::kFloat32));
    decode_tensor(lse,q,torch::kFloat32,"lse_buffer");
    TORCH_CHECK(!return_lse || lse.sizes()==c10::IntArrayRef(lse_shape), "incorrect LSE buffer shape");
    // Caller-owned scratch is explicitly opt-in. No global buffers or cross-stream pool exists.
    const std::vector<torch::Tensor> writable={out,scratch,lse};
    std::vector<torch::Tensor> readable={q,k_cache,v_cache,block_table,cache_seqlens};
    for(const auto& x : {query_start_loc,new_k,new_v,rotary_cos,rotary_sin,cache_leftpad,alibi,rotary_pos})
        if(x.has_value()) readable.push_back(*x);
    for(size_t i=0;i<writable.size();i++) {
        if(writable[i].numel()==0) continue;
        for(const auto& x:readable) TORCH_CHECK(!writable[i].is_alias_of(x), "output/workspace/LSE must not alias inputs");
        for(size_t j=0;j<i;j++) if(writable[j].numel())
            TORCH_CHECK(!writable[i].is_alias_of(writable[j]), "output/workspace/LSE buffers must not alias each other");
    }
    if(T==0) return {out,lse};
    DecParams p{};
    p.q=q.data_ptr(); p.Kc=k_cache.data_ptr(); p.Vc=v_cache.data_ptr();
    p.block_table=block_table.data_ptr<int>(); p.block_table_stride=block_table.stride(0);
    p.o=out.data_ptr(); p.scratch=nsplit>1 ? scratch.data_ptr<float>() : nullptr;
    p.ctx_len=cache_seqlens.data_ptr<int>();
    p.cos=rotary_cos.has_value() ? rotary_cos->data_ptr<float>() : nullptr;
    p.sin=rotary_sin.has_value() ? rotary_sin->data_ptr<float>() : nullptr;
    p.batch=B; p.nq=NQ; p.total_q=T; p.NH=Hq; p.gqa=gqa; p.HD=D; p.page=page;
    p.max_blocks=block_table.size(1); p.nsplit=nsplit; p.window=window;
    p.causal=causal; p.window_right=window_right; p.attention_chunk=attention_chunk;
    p.cu_q=packed ? query_start_loc->data_ptr<int>() : nullptr;
    p.rotary=rotary_cos.has_value() ? (rotary_interleaved?2:1) : 0;
    p.scale=(float)scale; p.sc_q=1.f; p.sc_k=1.f; p.sc_v=1.f;
    p.leftpad=cache_leftpad.has_value() ? cache_leftpad->data_ptr<int>() : nullptr;
    p.lse_out=return_lse ? lse.data_ptr<float>() : nullptr;
    p.softcap=(float)softcap; p.alibi=alibi.has_value() ? alibi->data_ptr<float>() : nullptr;
    p.rotary_dim=rotary_dim; p.rotary_pos=rotary_pos.has_value() ? rotary_pos->data_ptr<int>() : nullptr;
    p.kv_row=kv_row; p.v_off=v_off;
    p.k_block_stride=k_cache.stride(0); p.k_token_stride=k_cache.stride(1); p.k_head_stride=k_cache.stride(2);
    p.v_block_stride=v_cache.stride(0); p.v_token_stride=v_cache.stride(1); p.v_head_stride=v_cache.stride(2);
    const void* nk=new_k.has_value() ? new_k->data_ptr() : nullptr;
    const void* nv=new_v.has_value() ? new_v->data_ptr() : nullptr;
    fa_decode_launch(p,dt_of(q),cur_stream(q),nk,nv);
    const int error=alu_last_launch_error();
    TORCH_CHECK(error==0,"ALU Attention HIP launch failed (error code ",error,")");
    return {out,lse};
}

// Export native entry points through pybind after PyTorch has initialized.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.attr("COMPILER_POLICY_ID") = ALU_COMPILER_POLICY_ID;
    m.attr("CXX_STANDARD") = 20;
    m.doc() = "alu_attn FlashAttention HIP kernels (fwd/bwd) for gfx1030";
    m.def("fwd", &fwd, "FlashAttention forward",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"), py::arg("causal"),
          py::arg("win_left"), py::arg("win_right"), py::arg("softcap"), py::arg("alibi"), py::arg("dropout_p"),
          py::arg("seed"), py::arg("attn_chunk")=0,
          py::arg("out")=c10::optional<torch::Tensor>());
    m.def("bwd", &bwd, "FlashAttention backward",
          py::arg("dout"), py::arg("q"), py::arg("k"), py::arg("v"), py::arg("out"),
          py::arg("lse"), py::arg("scale"), py::arg("causal"),
          py::arg("win_left"), py::arg("win_right"),
          py::arg("softcap"), py::arg("alibi"), py::arg("dropout_p"), py::arg("seed"));
    m.def("fwd_varlen", &fwd_varlen, "FlashAttention forward, varlen through cu_seqlens",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("cu_q"), py::arg("cu_k"),
          py::arg("max_sq"), py::arg("max_sk"), py::arg("scale"), py::arg("causal"),
          py::arg("win_left"), py::arg("win_right"), py::arg("softcap"), py::arg("alibi"), py::arg("dropout_p"),
          py::arg("seed"), py::arg("attn_chunk")=0,
          py::arg("seqused_q")=c10::optional<torch::Tensor>(),
          py::arg("seqused_k")=c10::optional<torch::Tensor>());
    m.def("dec", &dec, "FlashAttention decode with paged KV cache",
          py::arg("q"), py::arg("k_cache"), py::arg("v_cache"), py::arg("block_table"),
          py::arg("cache_seqlens"), py::arg("new_k"), py::arg("new_v"),
          py::arg("rotary_cos"), py::arg("rotary_sin"), py::arg("rotary_interleaved"),
          py::arg("scale"), py::arg("window"), py::arg("num_splits"),
          py::arg("cache_leftpad")=c10::optional<torch::Tensor>(),
          py::arg("return_lse")=false, py::arg("softcap")=0.0,
          py::arg("alibi")=c10::optional<torch::Tensor>(),
          py::arg("rotary_dim")=0,
          py::arg("rotary_pos")=c10::optional<torch::Tensor>(),
          py::arg("kv_row")=0,
          py::arg("v_off")=0,
          py::arg("query_start_loc")=c10::optional<torch::Tensor>(),
          py::arg("causal")=true, py::arg("window_right")=-1,
          py::arg("attention_chunk")=0,
          py::arg("output")=c10::optional<torch::Tensor>(),
          py::arg("workspace")=c10::optional<torch::Tensor>(),
          py::arg("lse_buffer")=c10::optional<torch::Tensor>());
    m.attr("DECODE_ABI_VERSION")=3;
}
