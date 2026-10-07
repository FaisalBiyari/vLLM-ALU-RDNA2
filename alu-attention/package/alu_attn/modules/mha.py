"""Name- and behavior-compatible `flash_attn.modules.mha` wrappers over ALU Attention kernels.

Upstream provides ready-to-use attention layers that call flash_attn_* functions internally. This
module provides the same public layer names backed by ALU Attention, including `MHA` and
`FlashSelfAttention`, so drop-in compatibility does not fail at import time.
"""
import math
from functools import partial

import torch
from torch import nn

from ..flash_attn_interface import (flash_attn_func, flash_attn_kvpacked_func,
                                    flash_attn_qkvpacked_func, flash_attn_varlen_func,
                                    flash_attn_varlen_kvpacked_func,
                                    flash_attn_varlen_qkvpacked_func,
                                    flash_attn_with_kvcache)
from ..layers.rotary import RotaryEmbedding

__all__ = ["FlashSelfAttention", "FlashCrossAttention", "SelfAttention", "CrossAttention",
           "MHA", "LinearResidual"]


class FlashSelfAttention(nn.Module):
    """qkv: (batch, seqlen, 3, nheads, headdim), or varlen (total, 3, nheads, headdim)."""

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 window_size=(-1, -1), alibi_slopes=None, deterministic=False):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop_p = attention_dropout
        self.window_size = window_size
        self.alibi_slopes = alibi_slopes
        self.deterministic = deterministic

    def forward(self, qkv, causal=None, cu_seqlens=None, max_seqlen=None):
        causal = self.causal if causal is None else causal
        p = self.drop_p if self.training else 0.0
        if cu_seqlens is not None:
            return flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, max_seqlen, dropout_p=p, softmax_scale=self.softmax_scale,
                causal=causal, window_size=self.window_size, alibi_slopes=self.alibi_slopes,
                deterministic=self.deterministic)
        return flash_attn_qkvpacked_func(
            qkv, dropout_p=p, softmax_scale=self.softmax_scale, causal=causal,
            window_size=self.window_size, alibi_slopes=self.alibi_slopes,
            deterministic=self.deterministic)


class FlashCrossAttention(nn.Module):
    """q: (batch, sq, nheads, d); kv: (batch, sk, 2, nheads_k, d)."""

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 window_size=(-1, -1), alibi_slopes=None, deterministic=False):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop_p = attention_dropout
        self.window_size = window_size
        self.alibi_slopes = alibi_slopes
        self.deterministic = deterministic

    def forward(self, q, kv, causal=None, cu_seqlens=None, max_seqlen=None,
                cu_seqlens_k=None, max_seqlen_k=None):
        causal = self.causal if causal is None else causal
        p = self.drop_p if self.training else 0.0
        if cu_seqlens is not None:
            return flash_attn_varlen_kvpacked_func(
                q, kv, cu_seqlens, cu_seqlens_k, max_seqlen, max_seqlen_k, dropout_p=p,
                softmax_scale=self.softmax_scale, causal=causal,
                window_size=self.window_size, alibi_slopes=self.alibi_slopes,
                deterministic=self.deterministic)
        return flash_attn_kvpacked_func(
            q, kv, dropout_p=p, softmax_scale=self.softmax_scale, causal=causal,
            window_size=self.window_size, alibi_slopes=self.alibi_slopes,
            deterministic=self.deterministic)


class SelfAttention(nn.Module):
    """Reference non-Flash implementation, matching upstream and materializing the S×S matrix in torch."""

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(self, qkv, causal=None, key_padding_mask=None):
        b, s = qkv.shape[0], qkv.shape[1]
        causal = self.causal if causal is None else causal
        q, k, v = qkv.unbind(dim=2)
        scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * scale)
        if key_padding_mask is not None:
            pad = torch.full((b, s), float("-inf"), dtype=scores.dtype, device=scores.device)
            pad.masked_fill_(key_padding_mask, 0.0)
            scores = scores + pad.unsqueeze(1).unsqueeze(2)
        if causal:
            mask = torch.triu(torch.full((s, s), float("-inf"), device=scores.device), 1)
            scores = scores + mask.to(scores.dtype)
        attn = self.drop(torch.softmax(scores, dim=-1).to(v.dtype))
        return torch.einsum("bhts,bshd->bthd", attn, v)


class CrossAttention(nn.Module):
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(self, q, kv, causal=None, key_padding_mask=None):
        b, sq = q.shape[0], q.shape[1]
        sk = kv.shape[1]
        causal = self.causal if causal is None else causal
        k, v = kv.unbind(dim=2)
        if k.shape[2] != q.shape[2]:
            r = q.shape[2] // k.shape[2]
            k = k.repeat_interleave(r, dim=2)
            v = v.repeat_interleave(r, dim=2)
        scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * scale)
        if key_padding_mask is not None:
            pad = torch.full((b, sk), float("-inf"), dtype=scores.dtype, device=scores.device)
            pad.masked_fill_(key_padding_mask, 0.0)
            scores = scores + pad.unsqueeze(1).unsqueeze(2)
        if causal:
            idx_q = torch.arange(sq, device=q.device).view(-1, 1)
            idx_k = torch.arange(sk, device=q.device).view(1, -1)
            mask = (idx_k > idx_q + sk - sq)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = self.drop(torch.softmax(scores, dim=-1).to(v.dtype))
        return torch.einsum("bhts,bshd->bthd", attn, v)


class LinearResidual(nn.Linear):
    """Returns (output, input), matching upstream."""

    def forward(self, x):
        return super().forward(x), x


class MHA(nn.Module):
    """Multi-head attention compatible with flash_attn.modules.mha.MHA.

    Supports MHA/MQA/GQA (num_heads_kv), causal attention, rotary embeddings, sliding-window attention,
    inference KV cache (inference_params), and cross-attention. Fused QKV kernels (fused_bias_fc)
    are not provided; ordinary nn.Linear is used with equivalent numerical behavior.
    """

    def __init__(self, embed_dim, num_heads, num_heads_kv=None, cross_attn=False,
                 qkv_proj_bias=True, out_proj_bias=True, dropout=0.0, softmax_scale=None,
                 causal=False, layer_idx=None, dwconv=False, rotary_emb_dim=0,
                 rotary_emb_base=10000.0, rotary_emb_scale_base=None,
                 rotary_emb_interleaved=False, use_alibi=False, window_size=(-1, -1),
                 fused_bias_fc=False, use_flash_attn=True, return_residual=False,
                 checkpointing=False, device=None, dtype=None):
        factory = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attn = cross_attn
        self.causal = causal
        self.layer_idx = layer_idx
        self.dwconv = dwconv
        self.rotary_emb_dim = rotary_emb_dim
        self.return_residual = return_residual
        self.checkpointing = checkpointing
        if dwconv:
            raise NotImplementedError("alu_attn: dwconv in MHA is not implemented")
        self.alibi_slopes = None
        if use_alibi:
            self.alibi_slopes = torch.tensor(_alibi_slopes(num_heads), device=device)
        self.window_size = window_size

        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        assert self.num_heads % self.num_heads_kv == 0, "num_heads must be divisible by num_heads_kv"
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        kv_dim = 2 * self.head_dim * self.num_heads_kv

        if self.rotary_emb_dim > 0:
            assert not cross_attn, "rotary is incompatible with cross-attention"
            self.rotary_emb = RotaryEmbedding(self.rotary_emb_dim, base=rotary_emb_base,
                                              scale_base=rotary_emb_scale_base,
                                              interleaved=rotary_emb_interleaved,
                                              device=device)
        lin = LinearResidual if return_residual else nn.Linear
        if not cross_attn:
            self.Wqkv = lin(embed_dim, qkv_dim, bias=qkv_proj_bias, **factory)
        else:
            self.Wq = lin(embed_dim, embed_dim, bias=qkv_proj_bias, **factory)
            self.Wkv = lin(embed_dim, kv_dim, bias=qkv_proj_bias, **factory)
        inner = (FlashSelfAttention if use_flash_attn else SelfAttention)
        inner_cross = (FlashCrossAttention if use_flash_attn else CrossAttention)
        kw = dict(causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout)
        if use_flash_attn:
            kw.update(window_size=window_size, alibi_slopes=self.alibi_slopes)
        self.inner_attn = inner(**kw)
        self.inner_cross_attn = inner_cross(**kw)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias, **factory)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        dt = self.out_proj.weight.dtype if dtype is None else dtype
        return torch.empty(batch_size, max_seqlen, 2, self.num_heads_kv, self.head_dim,
                           dtype=dt, device=self.out_proj.weight.device)

    def _update_kv_cache(self, kv, inference_params):
        cache, _ = inference_params.key_value_memory_dict[self.layer_idx]
        b = kv.shape[0]
        off = inference_params.seqlen_offset
        cache[:b, off:off + kv.shape[1]] = kv
        return cache[:b, :off + kv.shape[1]]

    def forward(self, x, x_kv=None, key_padding_mask=None, cu_seqlens=None, max_seqlen=None,
                mixer_subset=None, inference_params=None, **kwargs):
        if cu_seqlens is not None and inference_params is not None:
            raise NotImplementedError("alu_attn: varlen with KV cache is not supported")
        seq_offset = 0 if inference_params is None else inference_params.seqlen_offset

        if not self.cross_attn:
            qkv = self.Wqkv(x)
            res = None
            if self.return_residual:
                qkv, res = qkv
            if self.num_heads_kv == self.num_heads:
                qkv = qkv.reshape(*qkv.shape[:-1], 3, self.num_heads, self.head_dim)
                if self.rotary_emb_dim > 0:
                    qkv = self.rotary_emb(qkv, seqlen_offset=seq_offset)
                if inference_params is None:
                    out = self.inner_attn(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
                else:
                    q = qkv[:, :, 0]
                    kv = self._update_kv_cache(qkv[:, :, 1:], inference_params)
                    out = self.inner_cross_attn(q, kv, causal=(seq_offset == 0) and self.causal)
            else:
                nq = self.num_heads * self.head_dim
                q = qkv[..., :nq].reshape(*qkv.shape[:-1], self.num_heads, self.head_dim)
                kv = qkv[..., nq:].reshape(*qkv.shape[:-1], 2, self.num_heads_kv, self.head_dim)
                if self.rotary_emb_dim > 0:
                    q, kv = self.rotary_emb(q, kv, seqlen_offset=seq_offset)
                if inference_params is not None:
                    kv = self._update_kv_cache(kv, inference_params)
                out = self.inner_cross_attn(q, kv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                                            cu_seqlens_k=cu_seqlens, max_seqlen_k=max_seqlen)
        else:
            q = self.Wq(x if mixer_subset is None else x[:, mixer_subset])
            kv = self.Wkv(x_kv if x_kv is not None else x)
            res = None
            if self.return_residual:
                q, res = q
                kv = kv[0] if isinstance(kv, tuple) else kv
            q = q.reshape(*q.shape[:-1], self.num_heads, self.head_dim)
            kv = kv.reshape(*kv.shape[:-1], 2, self.num_heads_kv, self.head_dim)
            if inference_params is not None:
                kv = self._update_kv_cache(kv, inference_params)
            out = self.inner_cross_attn(q, kv)

        out = self.out_proj(out.reshape(*out.shape[:-2], self.embed_dim))
        return (out, res) if self.return_residual else out


def _alibi_slopes(nheads):
    """ALiBi slopes following the paper recipe (powers of two plus extension for non-powers of two)."""
    def pow2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]
    if math.log2(nheads).is_integer():
        return pow2(nheads)
    closest = 2 ** math.floor(math.log2(nheads))
    return pow2(closest) + _alibi_slopes(2 * closest)[0::2][: nheads - closest]
