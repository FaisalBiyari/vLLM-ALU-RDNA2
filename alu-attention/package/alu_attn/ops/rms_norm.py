"""flash_attn.ops.rms_norm — wrappers around ops.triton.layer_norm matching flash-attn."""
import torch
from .triton.layer_norm import RMSNorm, rms_norm_fn

__all__ = ["RMSNorm", "rms_norm", "dropout_add_rms_norm"]


def rms_norm(x, weight, epsilon):
    return rms_norm_fn(x, weight, None, eps=epsilon)


def dropout_add_rms_norm(x0, residual, weight, bias, dropout_p, epsilon,
                         rowscale=None, layerscale=None, prenorm=False,
                         residual_in_fp32=False, return_dropout_mask=False):
    if dropout_p and dropout_p > 0:
        x0 = torch.nn.functional.dropout(x0, dropout_p)
    if layerscale is not None:
        x0 = x0 * layerscale
    return rms_norm_fn(x0, weight, bias, residual=residual, eps=epsilon, rowscale=rowscale,
                       prenorm=prenorm, residual_in_fp32=residual_in_fp32)
