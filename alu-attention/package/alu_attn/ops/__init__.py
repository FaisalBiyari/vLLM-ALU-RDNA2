"""Name-compatible `flash_attn.ops` implemented in pure torch without Triton.

Triton is not available for ROCm gfx1030 in this project, so these functions use ordinary torch
operations. Numerical behavior is preserved, but kernel fusion is not. The modules exist so
third-party code does not fail at import time.
"""
from . import triton
from .rms_norm import RMSNorm, rms_norm, dropout_add_rms_norm
from .layer_norm import LayerNorm, dropout_add_layer_norm

__all__ = ["triton", "RMSNorm", "rms_norm", "dropout_add_rms_norm",
           "LayerNorm", "dropout_add_layer_norm"]
