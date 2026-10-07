"""pytest coverage for the alu_attn drop-in package against an exact torch reference.

An older version imported `_ref_attention` and `_HAS_C` from flash_attn_interface after both names
had been removed with the old silent Python fallback, so the tests failed during import and checked
nothing. This file carries its own local reference and has no dependency on package internals.

Heavier gates covering every head_dim, both dtypes, varlen, paged cache, and all newer FA-3
arguments live in lab/parity_test.py and lab/parity_new_api.py. This file covers the core API and is
intended for a quick pytest pass.

Run: python -m pytest package/tests/test_alu_attn.py -v
"""
import math
import os
import sys

import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import alu_attn                                            # noqa: E402
from alu_attn import (flash_attn_func, flash_attn_qkvpacked_func,       # noqa: E402
                        flash_attn_kvpacked_func, flash_attn_with_kvcache,
                        flash_attn_varlen_func, flash_attn_combine)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
DEV = "cuda"


def ref_bshd(q, k, v, scale, causal=False, window=None, softcap=0.0, alibi=None, chunk=0, qv=None):
    """Exact fp32 reference. q,k,v: (B,S,H,D) -> (B,S,H,D)."""
    q_, k_, v_ = (x.transpose(1, 2).float() for x in (q, k, v))
    B, H, Sq, _ = q_.shape
    Hk = k_.shape[1]
    if H != Hk:
        k_ = k_.repeat_interleave(H // Hk, dim=1)
        v_ = v_.repeat_interleave(H // Hk, dim=1)
    s = q_ @ k_.transpose(-1, -2)
    if qv is not None:
        s = s + (qv.transpose(1, 2).float() @ v_.transpose(-1, -2))
    if softcap and softcap > 0:
        s = softcap * torch.tanh(s * scale / softcap)
    else:
        s = s * scale
    Sk = k_.shape[2]
    qi = torch.arange(Sq, device=q.device).view(-1, 1)
    ki = torch.arange(Sk, device=q.device).view(1, -1)
    delta = Sk - Sq
    if alibi is not None:
        s = s + alibi.view(1, H, 1, 1).float() * (ki - qi - delta).float()
    mask = torch.zeros(Sq, Sk, device=q.device, dtype=torch.bool)
    if causal:
        mask |= ki > qi + delta
    if window:
        mask |= (ki < qi + delta - window + 1) | (ki > qi + delta)
    if chunk:
        cs = ((qi + delta).clamp_min(0) // chunk) * chunk
        mask |= (ki < cs) | (ki >= cs + chunk)
    p = torch.nan_to_num(s.masked_fill(mask, float("-inf")).softmax(-1), nan=0.0)
    return (p @ v_).transpose(1, 2)


def relerr(a, b):
    a, b = a.float(), b.float()
    return (a - b).norm().item() / (b.norm().item() + 1e-12)


def rnd(*shape, dt=torch.float16):
    return torch.randn(*shape, device=DEV, dtype=dt)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("gqa", [1, 4])
def test_forward_vs_ref(dtype, causal, D, gqa):
    torch.manual_seed(0)
    B, S, H = 2, 128, 8
    Hk = H // gqa
    q, k, v = (rnd(B, S, H, D, dt=dtype), rnd(B, S, Hk, D, dt=dtype), rnd(B, S, Hk, D, dt=dtype))
    scale = 1.0 / math.sqrt(D)
    out = flash_attn_func(q, k, v, causal=causal, softmax_scale=scale)
    tol = 3e-2 if dtype == torch.bfloat16 else 1.5e-2
    assert relerr(out, ref_bshd(q, k, v, scale, causal)) < tol


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_backward_vs_ref(dtype, causal):
    torch.manual_seed(1)
    B, S, H, D = 2, 96, 4, 64
    scale = 1.0 / math.sqrt(D)
    q, k, v = (rnd(B, S, H, D, dt=dtype).requires_grad_() for _ in range(3))
    qr, kr, vr = (x.detach().float().requires_grad_(True) for x in (q, k, v))
    g = rnd(B, S, H, D, dt=dtype)
    flash_attn_func(q, k, v, causal=causal, softmax_scale=scale).backward(g)
    ref_bshd(qr, kr, vr, scale, causal).backward(g.float())
    tol = 4e-2 if dtype == torch.bfloat16 else 2e-2
    for name, a, b in [("dq", q.grad, qr.grad), ("dk", k.grad, kr.grad), ("dv", v.grad, vr.grad)]:
        assert relerr(a, b) < tol, f"{name} rel {relerr(a, b):.4f}"


def test_softcap_alibi_window():
    torch.manual_seed(2)
    B, S, H, D = 1, 128, 4, 64
    scale = 1.0 / math.sqrt(D)
    q, k, v = rnd(B, S, H, D), rnd(B, S, H, D), rnd(B, S, H, D)
    alibi = (-0.1 * torch.arange(1, H + 1, device=DEV)).float()
    assert relerr(flash_attn_func(q, k, v, causal=True, softmax_scale=scale, softcap=30.0),
                  ref_bshd(q, k, v, scale, True, softcap=30.0)) < 2e-2
    assert relerr(flash_attn_func(q, k, v, causal=True, softmax_scale=scale, alibi_slopes=alibi),
                  ref_bshd(q, k, v, scale, True, alibi=alibi)) < 2e-2
    assert relerr(flash_attn_func(q, k, v, causal=True, softmax_scale=scale, window_size=(32, 0)),
                  ref_bshd(q, k, v, scale, True, window=33)) < 2e-2


@pytest.mark.parametrize("chunk", [64, 100])
@pytest.mark.parametrize("causal", [False, True])
def test_attention_chunk(chunk, causal):
    """FA-3 attention_chunk: each query sees only keys from its own chunk."""
    torch.manual_seed(3)
    B, S, H, D = 1, 256, 4, 64
    scale = 1.0 / math.sqrt(D)
    q, k, v = rnd(B, S, H, D), rnd(B, S, H, D), rnd(B, S, H, D)
    out = flash_attn_func(q, k, v, causal=causal, attention_chunk=chunk)
    assert relerr(out, ref_bshd(q, k, v, scale, causal, chunk=chunk)) < 2e-2


def test_qv():
    """FA-3 qv scoring is q·kᵀ + qv·vᵀ."""
    torch.manual_seed(4)
    B, S, H, D = 1, 128, 4, 128
    q, k, v = rnd(B, S, H, D), rnd(B, S, H, D), rnd(B, S, H, D)
    qv = rnd(B, S, H, D)
    scale = (D + D) ** -0.5
    assert relerr(flash_attn_func(q, k, v, causal=True, qv=qv),
                  ref_bshd(q, k, v, scale, True, qv=qv)) < 3e-2


def test_output_is_contiguous():
    """Real flash-attn returns contiguous (B,S,H,D). The older wrapper returned a transposed view of
    contiguous [B,H,S,D], making `out.view(B, S, H*D)` fail in HuggingFace attention blocks. The
    result must also be independent of the internal K/V layout choice."""
    torch.manual_seed(14)
    B, S, H, D = 2, 256, 8, 128
    q, k, v = (rnd(B, S, H, D) for _ in range(3))
    out = flash_attn_func(q, k, v, causal=True)
    assert out.is_contiguous()
    out.view(B, S, H * D)
    assert torch.equal(out, flash_attn_func(q, k.contiguous(), v.contiguous(), causal=True))
    og = flash_attn_func(q.clone().requires_grad_(), k, v, causal=True)
    assert og.is_contiguous() and torch.equal(og.detach(), out)


def test_empty_rows_are_zero():
    """With causal SQ>SK, early rows have no visible keys and must return zeros like flash-attn."""
    torch.manual_seed(5)
    B, H, D = 1, 4, 64
    q, k, v = rnd(B, 256, H, D), rnd(B, 64, H, D), rnd(B, 64, H, D)
    out = flash_attn_func(q, k, v, causal=True)
    assert (out[:, :192] == 0).all()
    assert relerr(out, ref_bshd(q, k, v, 1 / math.sqrt(D), True)) < 2e-2


def test_return_attn_probs():
    torch.manual_seed(6)
    B, S, H, D = 1, 128, 4, 64
    q, k, v = rnd(B, S, H, D), rnd(B, S, H, D), rnd(B, S, H, D)
    out, lse, probs = flash_attn_func(q, k, v, causal=True, return_attn_probs=True)
    assert tuple(lse.shape) == (B, H, S) and tuple(probs.shape) == (B, H, S, S)
    assert (probs.sum(-1) - 1).abs().max() < 3e-3
    assert relerr((probs @ v.transpose(1, 2).float()).transpose(1, 2), out.float()) < 2e-2


def test_varlen_seqused():
    torch.manual_seed(7)
    import itertools
    H, D = 4, 64
    lens, used_q, used_k = [128, 64], [70, 64], [90, 30]
    tot = sum(lens)
    cu = torch.tensor([0] + list(itertools.accumulate(lens)), device=DEV, dtype=torch.int32)
    uq = torch.tensor(used_q, device=DEV, dtype=torch.int32)
    uk = torch.tensor(used_k, device=DEV, dtype=torch.int32)
    q, k, v = rnd(tot, H, D), rnd(tot, H, D), rnd(tot, H, D)
    out = flash_attn_varlen_func(q, k, v, cu, cu, max(lens), max(lens), causal=True,
                                 seqused_q=uq, seqused_k=uk)
    exp = torch.zeros_like(out)
    for i in range(len(lens)):
        o0 = int(cu[i])
        exp[o0:o0 + used_q[i]] = ref_bshd(
            q[o0:o0 + used_q[i]].unsqueeze(0), k[o0:o0 + used_k[i]].unsqueeze(0),
            v[o0:o0 + used_k[i]].unsqueeze(0), 1 / math.sqrt(D), True).squeeze(0)
    assert relerr(out, exp) < 2e-2


def test_qkv_kv_packed():
    torch.manual_seed(8)
    B, S, H, D = 2, 64, 4, 64
    qkv = rnd(B, S, 3, H, D)
    assert relerr(flash_attn_qkvpacked_func(qkv, causal=True),
                  ref_bshd(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], 1 / math.sqrt(D), True)) < 2e-2
    q, kv = rnd(B, S, H, D), rnd(B, S, 2, H, D)
    assert relerr(flash_attn_kvpacked_func(q, kv, causal=True),
                  ref_bshd(q, kv[:, :, 0], kv[:, :, 1], 1 / math.sqrt(D), True)) < 2e-2


def test_kvcache_decode_and_extras():
    torch.manual_seed(9)
    B, H, D, Smax, L = 2, 4, 64, 256, 100
    q = rnd(B, 1, H, D)
    kc, vc = rnd(B, Smax, H, D), rnd(B, Smax, H, D)
    sc = 1 / math.sqrt(D)
    assert relerr(flash_attn_with_kvcache(q, kc, vc, cache_seqlens=L),
                  ref_bshd(q, kc[:, :L], vc[:, :L], sc)) < 2e-2
    # cache_leftpad
    lp = torch.full((B,), 30, device=DEV, dtype=torch.int32)
    assert relerr(flash_attn_with_kvcache(q, kc, vc, cache_seqlens=L, cache_leftpad=lp),
                  ref_bshd(q, kc[:, 30:L], vc[:, 30:L], sc)) < 2e-2
    # cache_batch_idx
    idx = torch.tensor([1, 0], device=DEV, dtype=torch.int32)
    o = flash_attn_with_kvcache(q, kc, vc, cache_seqlens=L, cache_batch_idx=idx)
    exp = torch.cat([ref_bshd(q[b:b + 1], kc[idx[b]:idx[b] + 1, :L], vc[idx[b]:idx[b] + 1, :L], sc)
                     for b in range(B)], dim=0)
    assert relerr(o, exp) < 2e-2
    # return_softmax_lse
    o, lse = flash_attn_with_kvcache(q, kc, vc, cache_seqlens=L, return_softmax_lse=True)
    assert tuple(lse.shape) == (B, H, 1)


def test_combine():
    torch.manual_seed(10)
    B, S, H, D = 1, 128, 4, 64
    q, k, v = rnd(B, S, H, D), rnd(B, S, H, D), rnd(B, S, H, D)
    half = S // 2
    o1, l1, _ = flash_attn_func(q, k[:, :half], v[:, :half], return_attn_probs=True)
    o2, l2, _ = flash_attn_func(q, k[:, half:], v[:, half:], return_attn_probs=True)
    got = flash_attn_combine(torch.stack([o1, o2]), torch.stack([l1, l2]))
    assert relerr(got, ref_bshd(q, k, v, 1 / math.sqrt(D))) < 2e-2


def test_fp8_descale_rejected_loudly():
    """FP8 descales must fail explicitly rather than silently computing the wrong operation."""
    q = rnd(1, 32, 2, 64)
    with pytest.raises(NotImplementedError, match="FP8"):
        flash_attn_func(q, q, q, q_descale=torch.ones(1, device=DEV))


def test_scheduler_hints_do_not_change_output():
    torch.manual_seed(11)
    q, k, v = (rnd(1, 128, 4, 64) for _ in range(3))
    base = flash_attn_func(q, k, v, causal=True)
    for kw in (dict(num_splits=4), dict(pack_gqa=True), dict(sm_margin=8)):
        assert torch.equal(flash_attn_func(q, k, v, causal=True, **kw), base)


def test_determinism():
    torch.manual_seed(12)
    q, k, v = (rnd(1, 256, 8, 128) for _ in range(3))
    assert torch.equal(flash_attn_func(q, k, v, causal=True),
                       flash_attn_func(q, k, v, causal=True))


def test_install_as_flash_attn():
    alu_attn.install_as_flash_attn()
    import flash_attn
    import flash_attn_interface
    from flash_attn.bert_padding import pad_input, unpad_input      # noqa: F401
    assert flash_attn.__version__.startswith("2.")
    q, k, v = (rnd(1, 64, 2, 64) for _ in range(3))
    # FA-3 ordering: the fourth positional argument is softmax_scale, not dropout_p.
    o3 = flash_attn_interface.flash_attn_func(q, k, v, 1 / math.sqrt(64), True)
    assert relerr(o3, ref_bshd(q, k, v, 1 / math.sqrt(64), True)) < 2e-2


def test_vram_no_s_by_s():
    """Verify low VRAM usage: peak memory must remain far below materialized S×S attention."""
    torch.manual_seed(13)
    B, S, H, D = 2, 4096, 16, 128
    q, k, v = (rnd(B, S, H, D) for _ in range(3))
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    o = flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()
    peak_kernel = torch.cuda.max_memory_allocated() - base
    del o
    torch.cuda.reset_peak_memory_stats(); base2 = torch.cuda.memory_allocated()
    s = torch.matmul(q.transpose(1, 2).float(), k.transpose(1, 2).float().transpose(-1, -2))
    _ = torch.softmax(s, dim=-1)
    torch.cuda.synchronize()
    peak_ref = torch.cuda.max_memory_allocated() - base2
    assert peak_kernel < peak_ref * 0.5
