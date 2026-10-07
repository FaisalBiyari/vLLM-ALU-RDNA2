"""flash_attn.ops.layer_norm — wrappers around ops.triton.layer_norm matching flash-attn."""
import torch
from .triton.layer_norm import LayerNorm, layer_norm_fn

__all__ = ["LayerNorm", "dropout_add_layer_norm", "layer_norm"]


def layer_norm(x, weight, bias, epsilon):
    return layer_norm_fn(x, weight, bias, eps=epsilon)


def dropout_add_layer_norm(x0, residual, weight, bias, dropout_p, epsilon,
                           rowscale=None, layerscale=None, prenorm=False,
                           residual_in_fp32=False, return_dropout_mask=False):
    if dropout_p and dropout_p > 0:
        x0 = torch.nn.functional.dropout(x0, dropout_p)
    if layerscale is not None:
        x0 = x0 * layerscale
    return layer_norm_fn(x0, weight, bias, residual=residual, eps=epsilon, rowscale=rowscale,
                         prenorm=prenorm, residual_in_fp32=residual_in_fp32)
