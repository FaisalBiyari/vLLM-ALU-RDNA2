"""Register the rank-local `torch.ops.alu_attn.forward` compatibility operator.

Framework integrations can call this operator without depending on the pybind API.
Inputs and output use [B, H, S, D]. Unsupported configurations return an empty tensor
so the caller can choose its normal fallback. Operator registration is idempotent.
"""
import torch

LAST_ERROR = None      # Reason registration failed, for manual diagnostics.
REGISTERED = False

_LIB = None


def _forward(q, k, v, scale, causal):
    from . import _C
    empty = q.new_empty(0)
    # Kernel constraints: fp16/bf16, contiguous last dimension, and divisible GQA. Unsupported
    # inputs return an empty tensor rather than raising, as required by the fallback contract.
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return empty
    if q.dtype not in (torch.float16, torch.bfloat16) or not q.is_cuda:
        return empty
    if q.shape[-1] != k.shape[-1] or q.shape[-1] > 512:   # kernel-instantiated head dims (concurrency.py)
        return empty
    if q.shape[1] % k.shape[1]:
        return empty
    qc = q if q.stride(-1) == 1 else q.contiguous()
    kc = k if k.stride(-1) == 1 else k.contiguous()
    vc = v if v.stride(-1) == 1 else v.contiguous()
    try:
        out, _ = _C.fwd(qc, kc, vc, float(scale), bool(causal), -1, 0 if causal else -1,
                        0.0, None, 0.0, 0, 0, None)
    except Exception:
        return empty
    return out


def register():
    """Idempotent and non-throwing: registration failure must not break `import alu_attn`."""
    global _LIB, REGISTERED, LAST_ERROR
    if REGISTERED:
        return True
    try:
        ns = getattr(torch.ops, "alu_attn", None)
        if ns is not None and "forward" in dir(ns):
            REGISTERED = True          # Already declared, for example by the old build; do not replace it.
            return True
    except Exception:
        pass
    try:
        _LIB = torch.library.Library("alu_attn", "FRAGMENT")
        _LIB.define("forward(Tensor q, Tensor k, Tensor v, float scale, bool causal) -> Tensor")
        _LIB.impl("forward", _forward, "CUDA")
        REGISTERED = True
        return True
    except Exception as e:     # pragma: no cover
        LAST_ERROR = f"{type(e).__name__}: {e}"
        return False
