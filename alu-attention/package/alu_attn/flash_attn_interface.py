"""Drop-in flash-attn API over the alu_attn._C HIP kernels (gfx1030 / RX 6800 XT).

The public API provides flash-attn-compatible entry points for the supported configurations.
Unsupported combinations raise explicit errors; see docs/CONCURRENCY.md for current limitations.
The rectangular input layout is (batch, seqlen, nheads, headdim). Internally tensors are viewed as [B,H,S,D],
matching the kernels, exact FlashAttention is computed, and outputs are returned in public layout.

Argument ordering in this module follows FA-2 because this distribution can be installed as
`flash_attn`. FA-3-only arguments (qv, attention_chunk, num_splits, pack_gqa, sm_margin, and
q/k/v_descale) are keyword-only here so FA-2 positional arguments never shift. Callers using the
FA-3 convention through `import flash_attn_interface` receive `alu_attn.fa3_interface`, which
exposes FA-3 ordering.

There is NO silent fallback. Unsupported configurations raise clear exceptions. An earlier version
silently substituted a Python reference implementation that materialized the full S×S matrix
(8.6 GB at S=8192 with 32 heads), making the package appear functional while being hundreds of
times slower and quadratic in memory.
"""
import itertools
import math
import random
import torch
import torch.nn.functional as F

try:
    from alu_attn import _C  # Provides _C.fwd / _C.bwd through pybind bindings, without TORCH_LIBRARY.
except Exception as _e:  # pragma: no cover
    # NO silent fallback. An older path set _HAS_C=False and silently ran a Python reference that
    # materialized the complete S×S matrix: 8.6 GB of VRAM at S=8192 with 32 heads and 34 GB at
    # S=16384, potentially spilling to system memory. That made the package appear to work while
    # running hundreds of times slower with quadratic memory use.
    raise ImportError(
        "alu_attn._C failed to load. The native extension may be missing or incompatible with this torch/ROCm environment.\n"
        f"Reason: {type(_e).__name__}: {_e}\n"
        "Rebuild it for the current interpreter:\n"
        "  bash tools/build_rocm_linux.sh"
    ) from _e

# head_dim values instantiated by the kernels (see BYHD in fa_forward.hip and the switch in
# backward.hip). Other values are zero-padded to the next supported size, matching upstream
# flash-attn behavior for arbitrary head dimensions.
_VALID_D = (16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 512)   # 512 = D-split kernel (fa_forward_d512_k)
_MAX_D = _VALID_D[-1]


def _pad_dim(d):
    """Return the nearest supported head_dim >= d."""
    for x in _VALID_D:
        if x >= d:
            return x
    return None


def _pad_to(t, d):
    """Zero-pad the last dimension to d. For Q/K this preserves dot products; for V it adds
    zero output columns that are sliced away afterward."""
    if t.shape[-1] == d:
        return t
    return F.pad(t, (0, d - t.shape[-1]))


def _map_window(causal, window_size):
    """Map flash-attn window semantics to (causal_flag, win_left, win_right).

    For window_size=(left, right), key j is visible to query i when
        i+delta-left <= j <= i+delta+right.
    A negative value on either side means no bound. causal=True forces the right bound to the
    diagonal, matching flash-attn. `causal_flag` is enabled whenever win_right==0 so the kernel can
    use its compile-time causal branch to skip tiles above the diagonal.
    """
    l, r = window_size
    wl = int(l) if l is not None and l >= 0 else -1
    wr = int(r) if r is not None and r >= 0 else -1
    if causal:
        wr = 0
    return (1 if wr == 0 else 0), wl, wr


# ------------------------- FA-3 arguments that do not alter the math here -------------------------
def _reject_fp8(q_descale, k_descale, v_descale):
    """Reject FP8 descales explicitly. gfx1030 has no tensor cores, so FP8 offers no acceleration
    here and would only reduce precision. Silently accepting and ignoring these values would return
    numerically incorrect results relative to the requested operation."""
    if q_descale is not None or k_descale is not None or v_descale is not None:
        raise NotImplementedError(
            "alu_attn: q_descale/k_descale/v_descale (FP8) are not supported. "
            "gfx1030 has no tensor cores: FP8 provides no speedup here and loses precision, "
            "so the kernel would no longer be exact FlashAttention. Use fp16 or bf16.")


def _accept_hints(num_splits=None, pack_gqa=None, sm_margin=None):
    """Accept FA-3 scheduler hints while preserving deterministic output semantics.

    `num_splits`, `pack_gqa`, and `sm_margin` tune scheduling in FA-3 rather than changing the
    mathematical result. ALU Attention uses its own scheduler: prefill already has enough workgroups
    for 72 CUs, decode passes num_splits to the kernel, GQA packing is always active in decode, and
    sm_margin is irrelevant on this single-GPU path. Values are still type-checked so invalid input
    is not silently ignored."""
    if num_splits is not None and not isinstance(num_splits, int):
        raise TypeError(f"alu_attn: num_splits must be int, got {type(num_splits)}")
    if pack_gqa is not None and not isinstance(pack_gqa, bool):
        raise TypeError(f"alu_attn: pack_gqa must be bool or None, got {type(pack_gqa)}")
    if sm_margin is not None and not isinstance(sm_margin, int):
        raise TypeError(f"alu_attn: sm_margin must be int, got {type(sm_margin)}")


def _check_chunk(attention_chunk):
    if attention_chunk is None:
        return 0
    if not isinstance(attention_chunk, int) or attention_chunk < 0:
        raise ValueError(f"alu_attn: attention_chunk must be a non-negative int, "
                         f"got {attention_chunk!r}")
    return attention_chunk


def _check_supported(q, k, window_size, softcap, dim_ok=True):
    """Reject unsupported configurations explicitly instead of silently substituting a slow reference."""
    if not q.is_cuda:
        raise ValueError("alu_attn: tensors must be on the GPU (q.is_cuda == False)")
    D = q.shape[-1]
    if dim_ok and D > _MAX_D:
        raise ValueError(f"alu_attn: head_dim={D} exceeds the maximum {_MAX_D} "
                         "(flash-attn uses the same limit)")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"alu_attn: dtype {q.dtype} is not supported; expected fp16/bf16/fp32")
    # Public API layout is (batch, seqlen, nheads, headdim), so the head count is axis 2. An older
    # check used shape[1], comparing sequence lengths instead of heads and failing to validate GQA.
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError(
            f"alu_attn: nheads_q={q.shape[2]} is not divisible by nheads_k={k.shape[2]} (GQA)")


# ------------------------- qv and headdim_v != headdim -------------------------
def _fuse_qv(q, k, v, qv):
    """Implement FA-3 qv semantics by concatenation rather than a dedicated kernel.

    FA-3 defines qv scoring as `qk += einsum('bshd,bthd->bhst', qv, v)`, so
        S = Q·Kᵀ + QV·Vᵀ = [Q | QV] · [K | V]ᵀ,   O = softmax(S)·V.
    We therefore form q'=[q|qv], k'=[k|v], and v'=[v|zeros], then slice the first dv output
    columns. The same construction handles headdim_v != headdim when qv is absent by using a zero
    qv term. This compatibility path allocates temporary tensors and requires d+dv to fit a kernel
    head dimension, so it is not intended as the normal hot path."""
    d, dv = q.shape[-1], v.shape[-1]
    hds = d + dv
    if hds > _MAX_D:
        raise NotImplementedError(
            f"alu_attn: qv/headdim_v requires score head_dim = headdim+headdim_v = {d}+{dv} = {hds}, "
            f"but the kernel maximum is {_MAX_D} (same as flash-attn). The sum must fit {_MAX_D}.")
    if qv is None:
        qv = torch.zeros(q.shape[:-1] + (dv,), dtype=q.dtype, device=q.device)
    elif qv.shape[-1] != dv:
        raise ValueError(f"alu_attn: qv.shape[-1]={qv.shape[-1]} must match "
                         f"v.shape[-1]={dv}")
    qc = torch.cat([q, qv], dim=-1)
    kc = torch.cat([k, v], dim=-1)
    vc = torch.cat([v, torch.zeros_like(v[..., :d])], dim=-1) if d else v
    return qc, kc, vc, dv


def _needs_qv_path(q, v, qv):
    return qv is not None or v.shape[-1] != q.shape[-1]


# ------------------------- input layout -------------------------
# Public API layout is (B,S,H,D), while the kernel consumes [B,H,S,D]. An older wrapper called
# .transpose(1,2).contiguous() on q, k, and v for every invocation, creating three full copies even
# though the kernel supports arbitrary strides as long as stride_d==1. Measurements in
# lab/sweep_strides.py and lab/sweep_rule_check.py showed bit-for-bit identical output. Copies cost
# 9–30% of public-call time; strided K/V reads cost 0–10%, with the penalty driven by reread
# multiplicity rather than total K/V size. The policy below is deliberately conservative and only
# enables strided K/V in regions where a win was measured reliably.
_BYTES_128MB = 128 * 2 ** 20


def _kv_can_be_strided(SQ, D, Hq, Hkv, causal, kv_bytes):
    # Kernel q tile; see launch_cfg / launch_d512 in fa_forward.hip. D512 uses the
    # D-split kernel (BR=32) so the conservative reread bound keeps strided K/V off by default.
    reread = math.ceil(SQ / BR) * (Hq / Hkv) / (2 if causal else 1)
    if reread <= 32:
        return True     # Measured win across all sampled cells, sometimes 20–40%.
    if reread <= 64 and kv_bytes >= _BYTES_128MB:
        return True     # At multiplicity 64, large K/V copy cost exceeds the stride penalty.
    return False        # Remain conservative outside the measured winning regions.


# ------------------------- autograd over the kernels -------------------------
class _ALUAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, causal, wl, wr, softcap, alibi, dropout_p, deterministic,
                attn_chunk, out_buf):
        # q,k,v: [B,H,S,D]; only the D dimension must be contiguous. This FA-2-style backward uses
        # no atomics and is always deterministic, so the deterministic flag is accepted as a no-op.
        seed = random.getrandbits(63) if dropout_p > 0 else 0
        out, lse = _C.fwd(q, k, v, float(scale), bool(causal), int(wl), int(wr),
                          float(softcap), alibi, float(dropout_p), int(seed), int(attn_chunk),
                          out_buf)
        ctx.save_for_backward(q, k, v, out, lse, alibi if alibi is not None else torch.empty(0))
        ctx.scale = scale; ctx.causal = causal; ctx.wl = wl; ctx.wr = wr; ctx.softcap = softcap
        ctx.dropout_p = dropout_p; ctx.seed = seed; ctx.has_alibi = alibi is not None
        ctx.attn_chunk = attn_chunk
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        # FA-3 also does not support backward with attention_chunk. Raise explicitly rather than
        # silently computing a different operation.
        if ctx.attn_chunk:
            raise NotImplementedError(
                "alu_attn: backward with attention_chunk is not supported, matching FA-3. "
                "For chunked-attention inference, run forward under torch.no_grad().")
        q, k, v, out, lse, alibi = ctx.saved_tensors
        al = alibi if ctx.has_alibi else None
        dq, dk, dv = _C.bwd(dout.contiguous(), q, k, v, out, lse,
                            float(ctx.scale), bool(ctx.causal), int(ctx.wl), int(ctx.wr),
                            float(ctx.softcap), al, float(ctx.dropout_p), int(ctx.seed))
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


# ------------------------- attn_probs debug path -------------------------
def _attn_probs(q, k, lse, scale, causal, wl, wr, softcap, alibi, attn_chunk):
    """Return S_dmask for return_attn_probs as a full [B,H,SQ,SK] probability matrix.

    This is a debug path, as in flash-attn. It materializes S×S, which the production kernel never
    does; at B=1,H=32,S=8192 that is 8.6 GB. The path is enabled only by explicit
    return_attn_probs=True and checks available VRAM first. Probabilities are reconstructed from the
    kernel LSE as p=exp(s-lse), so they are consistent with the actual kernel result. Dropout is not
    applied to the returned probabilities because they represent the pre-dropout distribution.
    q,k: [B,H,S,D]. Returns [B,Hq,SQ,SK]."""
    B, Hq, SQ, D = q.shape
    Hk, SK = k.shape[1], k.shape[2]
    need = B * Hq * SQ * SK * 4
    free = torch.cuda.mem_get_info(q.device)[0]
    if need > free * 0.8:
        raise torch.cuda.OutOfMemoryError(
            f"alu_attn: return_attn_probs requires an S×S matrix of {need/2**30:.1f} GB "
            f"({free/2**30:.1f} GB free). This is a debug path; the kernel itself never materializes S×S.")
    kk = k if Hk == Hq else k.repeat_interleave(Hq // Hk, dim=1)
    s = torch.matmul(q.float(), kk.float().transpose(-1, -2))
    if softcap and softcap > 0:
        s = softcap * torch.tanh(s * scale / softcap)
    else:
        s = s * scale
    delta = SK - SQ
    qi = torch.arange(SQ, device=q.device).view(-1, 1)
    ki = torch.arange(SK, device=q.device).view(1, -1)
    if alibi is not None:
        s = s + alibi.view(1, Hq, 1, 1).float() * (ki - qi - delta).float()
    mask = torch.zeros(SQ, SK, device=q.device, dtype=torch.bool)
    if causal:
        mask |= ki > qi + delta
    if wl >= 0:
        mask |= ki < qi + delta - wl
    if wr >= 0:
        mask |= ki > qi + delta + wr
    if attn_chunk:
        cs = ((qi + delta).clamp_min(0) // attn_chunk) * attn_chunk
        mask |= (ki < cs) | (ki >= cs + attn_chunk)
    p = torch.exp(s - lse.unsqueeze(-1))
    return p.masked_fill(mask, 0.0)


# ------------------------- shared forward path -------------------------
def _fwd_bshd(q, k, v, dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes,
              deterministic, return_attn_probs, qv, attention_chunk,
              q_descale, k_descale, v_descale, num_splits, pack_gqa, sm_margin):
    """Shared implementation for flash_attn_func and packed wrappers. Input layout is (B,S,H,D)."""
    _reject_fp8(q_descale, k_descale, v_descale)
    _accept_hints(num_splits, pack_gqa, sm_margin)
    chunk = _check_chunk(attention_chunk)

    dv_out = None
    if _needs_qv_path(q, v, qv):
        if softmax_scale is None:   # FA-3: 1/sqrt(headdim + headdim_qv)
            softmax_scale = (q.shape[-1] + v.shape[-1]) ** (-0.5)
        q, k, v, dv_out = _fuse_qv(q, k, v, qv)
    elif softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    _check_supported(q, k, window_size, softcap)
    cflag, wl, wr = _map_window(causal, window_size)

    # Zero-pad unsupported head_dim values to the next instantiated size.
    # Zero columns preserve dot products; extra output columns are zero and sliced away.
    d_real = q.shape[-1]
    d_pad = _pad_dim(d_real)
    if d_pad != d_real:
        q = _pad_to(q, d_pad); k = _pad_to(k, d_pad); v = _pad_to(v, d_pad)
        if dv_out is None:
            dv_out = d_real          # Slice away padding from the output.

    B, SQ, Hq, D = q.shape
    Hkv = k.shape[2]
    need_grad = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)

    qh = q.transpose(1, 2); kh = k.transpose(1, 2); vh = v.transpose(1, 2)
    if need_grad:
        # Keep the training path contiguous. Backward stride gains measured around one percent and
        # changed sign across shapes, so preserving the established path is safer.
        qh = qh.contiguous(); kh = kh.contiguous(); vh = vh.contiguous()
    else:
        # Q is read once into LDS, so copying it has no benefit. K/V follow the measured policy in
        # _kv_can_be_strided.
        kv_bytes = 2 * B * k.shape[1] * Hkv * D * q.element_size()
        if not _kv_can_be_strided(SQ, D, Hq, Hkv, bool(cflag), kv_bytes):
            kh = kh.contiguous(); vh = vh.contiguous()

    # Allocate output directly in (B,S,H,D) layout and pass the kernel a transposed view. The caller
    # therefore receives a contiguous tensor, matching real flash-attn. The older implementation
    # returned a transposed view of [B,H,S,D], which breaks `.view(B, S, H*D)` in HuggingFace blocks.
    out_buf = torch.empty((B, SQ, Hq, D), dtype=q.dtype, device=q.device).transpose(1, 2)

    al = alibi_slopes.float().contiguous() if alibi_slopes is not None else None
    out, lse = _ALUAttnFn.apply(qh, kh, vh, softmax_scale, cflag, wl, wr, softcap, al,
                                  dropout_p, deterministic, chunk, out_buf)
    res = out.transpose(1, 2)
    if dv_out is not None:
        # Slicing the final axis creates a non-contiguous view, so materialize it here to preserve
        # the contiguous-output contract. The qv compatibility path is not the hot path.
        res = res[..., :dv_out].contiguous()
    if return_attn_probs:
        # FA-2 returns (out, softmax_lse, S_dmask); preserve that contract.
        probs = _attn_probs(qh, kh, lse, softmax_scale, bool(cflag), wl, wr, softcap, al, chunk)
        return res, lse, probs
    return res


# ------------------------- public API -------------------------
def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                    window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                    deterministic=False, return_attn_probs=False, *,
                    qv=None, attention_chunk=0,
                    q_descale=None, k_descale=None, v_descale=None,
                    num_splits=1, pack_gqa=None, sm_margin=0):
    """q,k,v: (batch, seqlen, nheads, headdim). Returns output in the same layout.
    With return_attn_probs=True, returns the FA-2 triple (out, softmax_lse, S_dmask)."""
    return _fwd_bshd(q, k, v, dropout_p, softmax_scale, causal, window_size, softcap,
                     alibi_slopes, deterministic, return_attn_probs, qv, attention_chunk,
                     q_descale, k_descale, v_descale, num_splits, pack_gqa, sm_margin)


def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False,
                              window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                              deterministic=False, return_attn_probs=False, *,
                              qv=None, attention_chunk=0,
                              q_descale=None, k_descale=None, v_descale=None,
                              num_splits=1, pack_gqa=None, sm_margin=0):
    """qkv: (batch, seqlen, 3, nheads, headdim)."""
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    return _fwd_bshd(q, k, v, dropout_p, softmax_scale, causal, window_size, softcap,
                     alibi_slopes, deterministic, return_attn_probs, qv, attention_chunk,
                     q_descale, k_descale, v_descale, num_splits, pack_gqa, sm_margin)


def flash_attn_kvpacked_func(q, kv, dropout_p=0.0, softmax_scale=None, causal=False,
                             window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                             deterministic=False, return_attn_probs=False, *,
                             qv=None, attention_chunk=0,
                             q_descale=None, k_descale=None, v_descale=None,
                             num_splits=1, pack_gqa=None, sm_margin=0):
    """q: (batch, sq, nheads, d); kv: (batch, sk, 2, nheads_k, d)."""
    k, v = kv[:, :, 0], kv[:, :, 1]
    return _fwd_bshd(q, k, v, dropout_p, softmax_scale, causal, window_size, softcap,
                     alibi_slopes, deterministic, return_attn_probs, qv, attention_chunk,
                     q_descale, k_descale, v_descale, num_splits, pack_gqa, sm_margin)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                           dropout_p=0.0, softmax_scale=None, causal=False,
                           window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                           deterministic=False, return_attn_probs=False, *,
                           seqused_q=None, seqused_k=None,
                           qv=None, attention_chunk=0,
                           q_descale=None, k_descale=None, v_descale=None,
                           num_splits=1, pack_gqa=None, sm_margin=0,
                           block_table=None):
    """Varlen packed API: q,k,v are (total_tokens, nheads, headdim).

    Without gradients, the native kernel receives cu_seqlens directly and handles all sequences in
    one launch. With gradients, sequences are processed individually because the backward kernel is
    dense. seqused_q/seqused_k can shorten the effective sequence lengths relative to cu_seqlens
    and are passed directly to the native path without extra copies.
    """
    _reject_fp8(q_descale, k_descale, v_descale)
    _accept_hints(num_splits, pack_gqa, sm_margin)
    chunk = _check_chunk(attention_chunk)

    if block_table is not None:
        # Paged KV in varlen prefill stores k/v as (num_blocks, page, nheads_k, d), while
        # block_table identifies pages belonging to each sequence. Flatten the selected pages into
        # dense sequences and use the normal varlen path; the prefill kernel reads K/V sequentially.
        if k.dim() != 4 or v.dim() != 4:
            raise ValueError(
                "alu_attn: with block_table, keys and values must use paged layout "
                f"(num_blocks, page_block_size, nheads_k, headdim); got "
                f"k.shape={tuple(k.shape)}. block_table is unnecessary for dense K/V.")
        bt = block_table.to(torch.int32)
        nseq = bt.shape[0]
        page = k.shape[1]
        if bt.dim() != 2 or bt.shape[0] != cu_seqlens_k.numel() - 1:
            raise ValueError(
                f"alu_attn: expected block_table shape (batch, max_blocks), got "
                f"{tuple(bt.shape)} with batch={cu_seqlens_k.numel() - 1}")
        lens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).tolist() if seqused_k is None \
            else seqused_k.tolist()
        kd, vd = [], []
        for i in range(nseq):
            L = int(lens_k[i])
            nb = (L + page - 1) // page
            idx = bt[i, :nb].long()
            kd.append(k[idx].reshape(-1, k.shape[2], k.shape[3])[:L])
            vd.append(v[idx].reshape(-1, v.shape[2], v.shape[3])[:L])
        k = torch.cat(kd, 0).contiguous()
        v = torch.cat(vd, 0).contiguous()
        cu_seqlens_k = torch.tensor([0] + list(itertools.accumulate(int(x) for x in lens_k)),
                                    device=k.device, dtype=torch.int32)
        seqused_k = None
        block_table = None

    dv_out = None
    if _needs_qv_path(q, v, qv):
        if softmax_scale is None:
            softmax_scale = (q.shape[-1] + v.shape[-1]) ** (-0.5)
        q, k, v, dv_out = _fuse_qv(q, k, v, qv)
    elif softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    need_grad = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)
    if not need_grad:
        if q.dim() != 3:
            raise ValueError(f"alu_attn: varlen expects (total, nheads, headdim), got {tuple(q.shape)}")
        if q.shape[-1] > _MAX_D:
            raise ValueError(f"alu_attn: head_dim={q.shape[-1]} exceeds maximum {_MAX_D}")
        d_real = q.shape[-1]
        d_pad = _pad_dim(d_real)
        if d_pad != d_real:
            q = _pad_to(q, d_pad); k = _pad_to(k, d_pad); v = _pad_to(v, d_pad)
            if dv_out is None:
                dv_out = d_real
        cflag, wl, wr = _map_window(causal, window_size)
        al = alibi_slopes.float().contiguous() if alibi_slopes is not None else None
        seed = random.getrandbits(63) if dropout_p > 0 else 0
        su_q = seqused_q.to(torch.int32).contiguous() if seqused_q is not None else None
        su_k = seqused_k.to(torch.int32).contiguous() if seqused_k is not None else None
        out = _C.fwd_varlen(q.contiguous(), k.contiguous(), v.contiguous(),
                            cu_seqlens_q.to(torch.int32).contiguous(),
                            cu_seqlens_k.to(torch.int32).contiguous(),
                            int(max_seqlen_q), int(max_seqlen_k),
                            float(softmax_scale), bool(cflag), int(wl), int(wr), float(softcap),
                            al, float(dropout_p), int(seed), int(chunk), su_q, su_k)
        if dv_out is not None:
            out = out[..., :dv_out]
        if return_attn_probs:
            # Ragged sequences do not share one S×S shape, so return a list with one
            # (nheads, Lq, Lk) matrix per sequence. This is a debug path that materializes data the
            # production kernel never stores.
            lse_l, probs_l = [], []
            for i in range(cu_seqlens_q.numel() - 1):
                a, b_ = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
                c_, d_ = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
                if seqused_q is not None:
                    b_ = a + int(seqused_q[i])
                if seqused_k is not None:
                    d_ = c_ + int(seqused_k[i])
                _o, _l, _p = flash_attn_func(
                    q[a:b_].unsqueeze(0), k[c_:d_].unsqueeze(0), v[c_:d_].unsqueeze(0),
                    dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes,
                    deterministic, True, attention_chunk=chunk)
                lse_l.append(_l.squeeze(0)); probs_l.append(_p.squeeze(0))
            return out, lse_l, probs_l
        return out
    outs = []
    nseq = cu_seqlens_q.numel() - 1
    for i in range(nseq):
        qs, qe = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
        ks, ke = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
        if seqused_q is not None:
            qe = qs + int(seqused_q[i])
        if seqused_k is not None:
            ke = ks + int(seqused_k[i])
        oi = flash_attn_func(q[qs:qe].unsqueeze(0), k[ks:ke].unsqueeze(0), v[ks:ke].unsqueeze(0),
                             dropout_p, softmax_scale, causal, window_size,
                             softcap, alibi_slopes, deterministic, False,
                             attention_chunk=chunk)
        outs.append(oi.squeeze(0))
    res = torch.cat(outs, dim=0)
    if dv_out is not None:
        res = res[..., :dv_out]
    return res


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p=0.0,
                                     softmax_scale=None, causal=False, window_size=(-1, -1),
                                     softcap=0.0, alibi_slopes=None, deterministic=False,
                                     return_attn_probs=False, **kw):
    """qkv: (total_tokens, 3, nheads, headdim)."""
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    return flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                                  dropout_p, softmax_scale, causal, window_size, softcap,
                                  alibi_slopes, deterministic, return_attn_probs, **kw)


def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                                    max_seqlen_k, dropout_p=0.0, softmax_scale=None,
                                    causal=False, window_size=(-1, -1), softcap=0.0,
                                    alibi_slopes=None, deterministic=False,
                                    return_attn_probs=False, **kw):
    """q: (total_q, nheads, d); kv: (total_k, 2, nheads_k, d)."""
    k, v = kv[:, 0], kv[:, 1]
    return flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                                  max_seqlen_k, dropout_p, softmax_scale, causal,
                                  window_size, softcap, alibi_slopes, deterministic,
                                  return_attn_probs, **kw)


def flash_attn_combine(out_partial, lse_partial, out=None, out_dtype=None):
    """Combine split-KV partial outputs. out_partial is (nsplit, B, S, H, D) and lse_partial
    is (nsplit, B, H, S) in natural-log units. Compatible with FA-3."""
    lse = lse_partial.float()
    m = lse.amax(dim=0, keepdim=True)
    w = torch.exp(lse - m)                      # (nsplit,B,H,S)
    wsum = w.sum(dim=0)                         # (B,H,S)
    w = w.permute(0, 1, 3, 2).unsqueeze(-1)     # (nsplit,B,S,H,1)
    acc = (out_partial.float() * w).sum(dim=0)  # (B,S,H,D)
    res = acc / wsum.permute(0, 2, 1).unsqueeze(-1).clamp_min(1e-30)
    res = res.to(out_dtype or out_partial.dtype)
    if out is not None:
        out.copy_(res); return out
    return res


@torch.no_grad()
def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, qv=None,
                            rotary_cos=None, rotary_sin=None,
                            cache_seqlens=None, cache_batch_idx=None, cache_leftpad=None,
                            block_table=None, softmax_scale=None, causal=False,
                            window_size=(-1, -1), softcap=0.0, rotary_interleaved=True,
                            alibi_slopes=None, num_splits=0, return_softmax_lse=False, *,
                            page_table=None, attention_chunk=0,
                            q_descale=None, k_descale=None, v_descale=None,
                            pack_gqa=None, sm_margin=0, scheduler_metadata=None,
                            cu_seqlens_q=None, cu_seqlens_k_new=None, max_seqlen_q=None,
                            rotary_seqlens=None):
    """Inference decode through fa_decode (split-KV, paged cache, and on-the-fly rotary).

    q has shape (batch, seqlen_q, nheads, headdim), or (total_tokens, nheads, headdim)
    with GPU cu_seqlens_q. Packed requests may have different query lengths; no per-request
    Python loop or CPU metadata readback is used. seqlen_q > 1 covers speculative decoding and
    chunked prefill. With appended K/V, new tokens occupy cache_seqlens through
    cache_seqlens+seqlen_q-1. Without append, cache_seqlens includes the prewritten queries.
    causal=True selects bottom-right causal visibility; causal=False does not. Paged cache layout follows flash-attn: k_cache/v_cache are
    (num_blocks, page_block_size, nheads_k, headdim) with block_table (batch, max_blocks). Dense
    (batch, seqlen, nheads_k, headdim) cache is also accepted and treated as one block per batch
    element. cache_seqlens is the cache length before appending optional k/v. rotary_cos/sin are
    tables of shape (seqlen_ro, rotary_dim/2), with partial rotary supported. page_table aliases
    block_table; cache_batch_idx selects dense-cache rows; cache_leftpad marks the first valid
    context position; return_softmax_lse returns (out, softmax_lse) with LSE shaped
    (batch, nheads, seqlen_q), or (nheads, total_tokens) for packed queries.

    An older implementation ignored block_table and ran prefill on a cache slice, which silently
    produced incorrect paged-cache results. This path now calls the real decode kernel.
    """
    _reject_fp8(q_descale, k_descale, v_descale)
    _accept_hints(num_splits, pack_gqa, sm_margin)
    packed = cu_seqlens_q is not None
    if q.dim() != (3 if packed else 4):
        raise ValueError("alu_attn: decode expects [T,H,D] with cu_seqlens_q or [B,NQ,H,D]")
    if softmax_scale is None:
        softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** -0.5
    _ = scheduler_metadata, max_seqlen_q  # Scheduling hints do not change attention semantics.
    if cu_seqlens_k_new is not None:
        if not packed or cu_seqlens_k_new is not cu_seqlens_q:
            raise NotImplementedError(
                "alu_attn: ragged append currently requires new K/V to use the same "
                "offset tensor as q; independent append/query lengths are not supported")
    if page_table is not None:
        if block_table is not None:
            raise ValueError("alu_attn: block_table and page_table are aliases; provide only one")
        block_table = page_table

    B = cu_seqlens_q.numel() - 1 if packed else q.shape[0]
    NQ = 1 if packed else q.shape[1]
    dev = q.device
    d_real = q.shape[-1]

    # Decode qv uses the same concatenation identity as prefill, but must concatenate the entire
    # cache. It is therefore a correctness-oriented compatibility path that roughly doubles cache
    # traffic per step, not a performance path.
    if qv is not None:
        if k is not None or v is not None:
            raise NotImplementedError("alu_attn: qv cache append would write a temporary cache; use prewritten K/V")
        if qv.shape[-1] != v_cache.shape[-1]:
            raise ValueError("alu_attn: qv.shape[-1] must match the value-cache head dimension")
        if d_real + qv.shape[-1] > _MAX_D:
            raise NotImplementedError(
                f"alu_attn: decode qv requires headdim+headdim_v <= {_MAX_D}")
        q = torch.cat([q, qv], dim=-1)
        k_cache = torch.cat([k_cache, v_cache], dim=-1)
        v_cache = torch.cat([v_cache, torch.zeros_like(v_cache[..., :d_real])], dim=-1)
        if k is not None:
            k = torch.cat([k, v], dim=-1)
            v = torch.cat([v, torch.zeros_like(v[..., :d_real])], dim=-1)

    if cache_seqlens is None:
        seqlens = torch.full((B,), k_cache.shape[1], dtype=torch.int32, device=dev)
    elif isinstance(cache_seqlens, int):
        seqlens = torch.full((B,), cache_seqlens, dtype=torch.int32, device=dev)
    else:
        seqlens = cache_seqlens.to(device=dev, dtype=torch.int32).contiguous()

    if block_table is None:
        # Dense cache: one block per batch element, with the page equal to the full cache length.
        # cache_batch_idx selects the cache row explicitly instead of using b directly.
        page = k_cache.shape[1]
        if page & (page - 1):
            raise NotImplementedError(
                f"alu_attn: dense cache length {page} must be a power of two "
                "(or provide block_table)")
        if cache_batch_idx is None:
            bt = torch.arange(B, dtype=torch.int32, device=dev).view(B, 1)
        else:
            bt = cache_batch_idx.to(device=dev, dtype=torch.int32).contiguous().view(B, 1)
        kc, vc = k_cache, v_cache
    else:
        if cache_batch_idx is not None:
            raise ValueError(
                "alu_attn: cache_batch_idx applies only to dense cache; for paged cache, "
                "block_table selects cache rows")
        bt = block_table.to(device=dev, dtype=torch.int32).contiguous()
        kc, vc = k_cache, v_cache

    _, wl, wr = _map_window(causal, window_size)
    win = (wl + 1) if wl >= 0 else -1
    lp = cache_leftpad.to(device=dev, dtype=torch.int32).contiguous() if cache_leftpad is not None else None
    chunk = _check_chunk(attention_chunk)

    al = alibi_slopes.float().contiguous() if alibi_slopes is not None else None
    cos = rotary_cos.float().contiguous() if rotary_cos is not None else None
    sin = rotary_sin.float().contiguous() if rotary_sin is not None else None
    # Derive rotary_dim from the flash-attn table shape (seqlen_ro, rotary_dim/2).
    rdim = int(cos.shape[-1] * 2) if cos is not None else 0
    rpos = (rotary_seqlens.to(device=dev, dtype=torch.int32).contiguous()
            if rotary_seqlens is not None else None)
    nk = k.contiguous() if k is not None else None
    nv = v.contiguous() if v is not None else None
    if (nk is None) != (nv is None):
        raise ValueError("alu_attn: k and v must be appended to the cache together")

    # Zero-pad head_dim to the next instantiated kernel size, as in prefill.
    dv_cut = None
    dq = q.shape[-1]
    d_pad = _pad_dim(dq)
    if d_pad is None:
        raise ValueError(f"alu_attn: head_dim={dq} exceeds maximum {_MAX_D}")
    if d_pad != dq:
        if nk is not None:
            raise NotImplementedError("alu_attn: append requires a native head dimension; padding would copy the cache")
        q = _pad_to(q, d_pad); kc = _pad_to(kc, d_pad); vc = _pad_to(vc, d_pad)
        if nk is not None:
            nk = _pad_to(nk, d_pad); nv = _pad_to(nv, d_pad)
        dv_cut = dq
    if qv is not None:
        dv_cut = v_cache.shape[-1] - d_real if dv_cut is None else dv_cut

    if getattr(_C, "DECODE_ABI_VERSION", 0) < 3:
        raise RuntimeError("ALU Attention decode ABI 3 is required; rebuild the native extension")
    out, lse = _C.dec(q.contiguous(), kc, vc, bt, seqlens, nk, nv,
                      cos, sin, bool(rotary_interleaved), float(softmax_scale), int(win),
                      int(num_splits or 0), lp, bool(return_softmax_lse),
                      float(softcap or 0.0), al, int(rdim), rpos,
                      query_start_loc=cu_seqlens_q, causal=bool(causal),
                      window_right=int(wr), attention_chunk=int(chunk))
    if qv is not None:
        # v_cache is [v | zeros], so the desired P·V occupies the first headdim_v columns.
        out = out[..., :qv.shape[-1]].contiguous()
    elif dv_cut is not None:
        out = out[..., :dv_cut].contiguous()
    if return_softmax_lse:
        # Kernel LSE is (batch, seqlen_q, nheads); flash-attn returns (batch, nheads, seqlen_q).
        return out, (lse.transpose(0, 1) if packed else lse.transpose(1, 2)).contiguous()
    return out
