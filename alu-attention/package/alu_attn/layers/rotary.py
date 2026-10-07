"""Pure-torch compatibility with `flash_attn.layers.rotary`.

Upstream uses a Triton kernel; this module implements the same behavior with ordinary torch
operations so imports such as `apply_rotary_emb` and `RotaryEmbedding` work unchanged. Both
upstream layout conventions are supported:
  interleaved=False (GPT-NeoX): pairs are (i, i + rotary_dim/2);
  interleaved=True  (GPT-J):    pairs are (2i, 2i+1).
"""
import math
from typing import Optional, Tuple, Union

import torch
from torch import nn

__all__ = ["apply_rotary_emb", "apply_rotary_emb_func", "apply_rotary_emb_qkv_",
           "apply_rotary_emb_kv_", "RotaryEmbedding"]


def _rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply(x, cos, sin, interleaved):
    """x: (..., rotary_dim); cos/sin: (..., rotary_dim/2), already aligned to positions."""
    ro = cos.shape[-1] * 2
    if not interleaved:
        c = torch.cat((cos, cos), dim=-1)
        s = torch.cat((sin, sin), dim=-1)
    else:
        c = torch.stack((cos, cos), dim=-1).flatten(-2)
        s = torch.stack((sin, sin), dim=-1).flatten(-2)
    x_ro, x_pass = x[..., :ro], x[..., ro:]
    out = x_ro * c + _rotate_half(x_ro, interleaved) * s
    return out if x_pass.shape[-1] == 0 else torch.cat((out, x_pass), dim=-1)


def apply_rotary_emb(x, cos, sin, interleaved=False, inplace=False, seqlen_offsets=0,
                     cu_seqlens=None, max_seqlen=None):
    """x: (batch, seqlen, nheads, headdim), or (total, nheads, headdim) with cu_seqlens.
    cos/sin: (seqlen_ro, rotary_dim/2). Returns a tensor of the same shape.

    Only the first rotary_dim = 2*cos.shape[-1] dimensions are rotated; the remaining dimensions
    pass through unchanged, matching upstream.
    """
    if cu_seqlens is None:
        assert x.dim() == 4, f"expected (batch, seqlen, nheads, headdim), got {tuple(x.shape)}"
        b, s = x.shape[0], x.shape[1]
        if isinstance(seqlen_offsets, int):
            idx = torch.arange(s, device=x.device) + seqlen_offsets
            c = cos[idx].unsqueeze(0).unsqueeze(2)          # (1,s,1,ro/2)
            sn = sin[idx].unsqueeze(0).unsqueeze(2)
        else:
            off = seqlen_offsets.view(b, 1)
            idx = torch.arange(s, device=x.device).view(1, s) + off
            c = cos[idx].unsqueeze(2)                        # (b,s,1,ro/2)
            sn = sin[idx].unsqueeze(2)
        out = _apply(x, c.to(x.dtype), sn.to(x.dtype), interleaved)
    else:
        assert x.dim() == 3, "with cu_seqlens, expected (total, nheads, headdim)"
        out = torch.empty_like(x)
        for i in range(cu_seqlens.numel() - 1):
            a, b_ = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
            off = seqlen_offsets if isinstance(seqlen_offsets, int) else int(seqlen_offsets[i])
            idx = torch.arange(b_ - a, device=x.device) + off
            c = cos[idx].unsqueeze(1)
            sn = sin[idx].unsqueeze(1)
            out[a:b_] = _apply(x[a:b_], c.to(x.dtype), sn.to(x.dtype), interleaved)
        return out
    if inplace:
        x.copy_(out)
        return x
    return out


# Upstream exposes the same operation under this additional name.
apply_rotary_emb_func = apply_rotary_emb


def apply_rotary_emb_qkv_(qkv, cos, sin, cos_k=None, sin_k=None, interleaved=False,
                          seqlen_offsets=0, cu_seqlens=None, max_seqlen=None,
                          num_heads_q=None):
    """qkv: (batch, seqlen, 3, nheads, headdim). Rotates q and k in place; leaves v unchanged."""
    q = apply_rotary_emb(qkv[:, :, 0], cos, sin, interleaved, False, seqlen_offsets)
    k = apply_rotary_emb(qkv[:, :, 1], cos if cos_k is None else cos_k,
                         sin if sin_k is None else sin_k, interleaved, False, seqlen_offsets)
    qkv[:, :, 0].copy_(q)
    qkv[:, :, 1].copy_(k)
    return qkv


def apply_rotary_emb_kv_(kv, cos, sin, interleaved=False, seqlen_offsets=0,
                         cu_seqlens=None, max_seqlen=None):
    """kv: (batch, seqlen, 2, nheads, headdim). Rotates k in place."""
    k = apply_rotary_emb(kv[:, :, 0], cos, sin, interleaved, False, seqlen_offsets)
    kv[:, :, 0].copy_(k)
    return kv


class RotaryEmbedding(nn.Module):
    """Compatibility implementation of flash_attn.layers.rotary.RotaryEmbedding.

    Maintains a cos/sin cache and can be applied to qkv, kv, or separate q/k tensors. Supports
    scale_base (XPos) and interleaved layouts. Computation uses fp32 and is cast back to the input
    dtype, matching upstream.
    """

    def __init__(self, dim: int, base=10000.0, interleaved=False, scale_base=None,
                 pos_idx_in_fp32=True, device=None):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        scale = None
        if scale_base is not None:
            scale = ((torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim)
                     / (1.4 * dim))
        self.register_buffer("scale", scale, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = self._sin_cached = None
        self._cos_k_cached = self._sin_k_cached = None

    def _compute_inv_freq(self, device=None):
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device,
                                                 dtype=torch.float32) / self.dim))

    def _update_cache(self, seqlen, device, dtype):
        if (self._cos_cached is not None and seqlen <= self._seq_len_cached
                and self._cos_cached.device == device and self._cos_cached.dtype == dtype):
            return
        self._seq_len_cached = seqlen
        t = torch.arange(seqlen, device=device,
                         dtype=torch.float32 if self.pos_idx_in_fp32 else dtype)
        inv_freq = self._compute_inv_freq(device) if self.pos_idx_in_fp32 else self.inv_freq
        freqs = torch.outer(t, inv_freq.to(device=t.device, dtype=t.dtype))
        if self.scale is None:
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)
            self._cos_k_cached = self._sin_k_cached = None
        else:
            power = ((torch.arange(seqlen, dtype=self.scale.dtype, device=device)
                      - seqlen // 2) / self.scale_base)
            scale = self.scale.to(device=power.device) ** power.unsqueeze(-1)
            self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
            self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
            self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
            self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(self, qkv: torch.Tensor, kv: Optional[torch.Tensor] = None,
                seqlen_offset: Union[int, torch.Tensor] = 0,
                max_seqlen: Optional[int] = None):
        seqlen = qkv.shape[1]
        need = (max_seqlen if max_seqlen is not None
                else seqlen + (seqlen_offset if isinstance(seqlen_offset, int) else 0))
        self._update_cache(need, qkv.device, qkv.dtype)
        if kv is None:
            if qkv.dim() == 5:      # (b, s, 3, h, d)
                return apply_rotary_emb_qkv_(qkv, self._cos_cached, self._sin_cached,
                                             self._cos_k_cached, self._sin_k_cached,
                                             self.interleaved, seqlen_offset)
            return apply_rotary_emb(qkv, self._cos_cached, self._sin_cached,
                                    self.interleaved, False, seqlen_offset)
        q = apply_rotary_emb(qkv, self._cos_cached, self._sin_cached, self.interleaved,
                             False, seqlen_offset)
        kv = apply_rotary_emb_kv_(kv, self._cos_cached if self._cos_k_cached is None
                                  else self._cos_k_cached,
                                  self._sin_cached if self._sin_k_cached is None
                                  else self._sin_k_cached,
                                  self.interleaved, seqlen_offset)
        return q, kv
