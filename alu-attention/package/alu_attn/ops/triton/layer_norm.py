"""Pure-torch compatibility with `flash_attn.ops.triton.layer_norm`.

Upstream uses Triton kernels for fused LayerNorm/RMSNorm with residuals. Triton is not available
for ROCm gfx1030 in this project, so the same operations are implemented with ordinary torch.
Results are equivalent but not fused. This is a compatibility path, not a performance accelerator.
"""
import torch
from torch import nn

__all__ = ["layer_norm_fn", "rms_norm_fn", "layer_norm_ref", "rms_norm_ref",
           "RMSNorm", "LayerNorm"]


def _norm(x, weight, bias, eps, is_rms, upcast=True):
    dt = x.dtype
    if upcast:
        x = x.float()
        weight = weight.float() if weight is not None else None
        bias = bias.float() if bias is not None else None
    if is_rms:
        out = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    else:
        out = (x - x.mean(dim=-1, keepdim=True))
        out = out * torch.rsqrt(out.square().mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out.to(dt)


def layer_norm_fn(x, weight, bias, residual=None, x1=None, weight1=None, bias1=None,
                  eps=1e-6, dropout_p=0.0, rowscale=None, prenorm=False,
                  residual_in_fp32=False, is_rms_norm=False, return_dropout_mask=False,
                  out=None, residual_out=None):
    """LayerNorm/RMSNorm with residual semantics compatible with flash-attn.

    Returns out, or (out, residual_out) when prenorm=True. If weight1/bias1 are supplied, a second
    normalization of the same input is computed, matching upstream.
    """
    if dropout_p and dropout_p > 0:
        raise NotImplementedError(
            "alu_attn: dropout inside layer_norm_fn is not implemented; apply torch.dropout "
            "outside the function for equivalent behavior")
    if rowscale is not None:
        x = x * rowscale.unsqueeze(-1)
    res = x if residual is None else x + residual
    if x1 is not None:
        res = res + x1
    res_out = res.float() if residual_in_fp32 else res
    y = _norm(res, weight, bias, eps, is_rms_norm)
    if out is not None:
        out.copy_(y); y = out
    if weight1 is not None:
        y1 = _norm(res, weight1, bias1, eps, is_rms_norm)
        return ((y, y1, res_out) if prenorm else (y, y1))
    if prenorm:
        if residual_out is not None:
            residual_out.copy_(res_out); res_out = residual_out
        return y, res_out
    return y


def rms_norm_fn(x, weight, bias=None, residual=None, x1=None, weight1=None, bias1=None,
                eps=1e-6, dropout_p=0.0, rowscale=None, prenorm=False,
                residual_in_fp32=False, return_dropout_mask=False, out=None,
                residual_out=None):
    return layer_norm_fn(x, weight, bias, residual, x1, weight1, bias1, eps, dropout_p,
                         rowscale, prenorm, residual_in_fp32, True, return_dropout_mask,
                         out, residual_out)


def layer_norm_ref(x, weight, bias, residual=None, eps=1e-6, prenorm=False,
                   upcast=False, **kw):
    res = x if residual is None else x + residual
    y = _norm(res, weight, bias, eps, False, upcast)
    return (y, res) if prenorm else y


def rms_norm_ref(x, weight, bias=None, residual=None, eps=1e-6, prenorm=False,
                 upcast=False, **kw):
    res = x if residual is None else x + residual
    y = _norm(res, weight, bias, eps, True, upcast)
    return (y, res) if prenorm else y


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, dropout_p=0.0, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.drop = nn.Dropout(dropout_p) if dropout_p > 0 else None
        self.weight = nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        return rms_norm_fn(x, self.weight, self.bias, residual=residual, eps=self.eps,
                           dropout_p=self.drop.p if self.drop is not None else 0.0,
                           prenorm=prenorm, residual_in_fp32=residual_in_fp32)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, dropout_p=0.0, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.drop = nn.Dropout(dropout_p) if dropout_p > 0 else None
        self.weight = nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        return layer_norm_fn(x, self.weight, self.bias, residual=residual, eps=self.eps,
                             dropout_p=self.drop.p if self.drop is not None else 0.0,
                             prenorm=prenorm, residual_in_fp32=residual_in_fp32)
