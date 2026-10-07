"""Packed, paged attention for continuous batching and linear MTP verification.

This is an attention primitive, not a request scheduler or a speculative sampler.
It never updates seq_lens. The caller owns page allocation, prefix copy-on-write,
accepted lengths, and separate target/draft state. No per-request Python loop runs
on this path. Cache tensors use logical [blocks,page,Hkv,content] axes; arbitrary
positive, nonoverlapping strides support token-major and head-major storage without copying.
16-bit cache rows must retain 16-byte alignment for native vector loads.
"""
from dataclasses import dataclass
from typing import Optional

import torch

from .concurrency import DecodePlan, plan_decode


@dataclass
class DecodeWorkspace:
    """Caller-owned buffers for ONE stream and ONE in-flight attention operation.

    Create before graph capture. Reusing this object overwrites previous outputs.
    Never share it between simultaneously executing graphs, streams, or TP ranks.
    """
    plan: DecodePlan
    output: torch.Tensor
    scratch: torch.Tensor
    lse: Optional[torch.Tensor]


def create_decode_workspace(q: torch.Tensor, k_cache: torch.Tensor,
                            block_table: torch.Tensor, *, num_splits: int = 0,
                            return_lse: bool = False) -> DecodeWorkspace:
    """Allocate fixed-shape buffers outside capture; metadata VALUES may change on replay."""
    if q.dim() not in (3, 4) or not q.is_cuda:
        raise ValueError("workspace requires a HIP/GPU q with packed or rectangular shape")
    if k_cache.dim() != 4 or block_table.dim() != 2:
        raise ValueError("invalid cache or block-table shape")
    rows = q.numel() // (q.shape[-2] * q.shape[-1])
    cu = torch.cuda.get_device_properties(q.device).multi_processor_count
    plan = plan_decode(rows, q.shape[-2], k_cache.shape[2], q.shape[-1],
                       block_table.shape[1] * k_cache.shape[1], cu, num_splits)
    with torch.cuda.device(q.device):
        return DecodeWorkspace(plan, torch.empty_like(q, memory_format=torch.contiguous_format),
                               torch.empty(plan.scratch_elements, device=q.device, dtype=torch.float32),
                               torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
                               if return_lse else None)


def paged_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                    block_table: torch.Tensor, seq_lens: torch.Tensor,
                    query_start_loc: torch.Tensor, *,
                    new_k: Optional[torch.Tensor] = None,
                    new_v: Optional[torch.Tensor] = None,
                    softmax_scale: Optional[float] = None, causal: bool = True,
                    window_size=(-1, -1), softcap: float = 0.0,
                    alibi_slopes: Optional[torch.Tensor] = None,
                    cache_leftpad: Optional[torch.Tensor] = None,
                    attention_chunk: int = 0, num_splits: int = 0,
                    kv_row: int = 0, v_off: int = 0,
                    rotary_cos: Optional[torch.Tensor] = None,
                    rotary_sin: Optional[torch.Tensor] = None,
                    rotary_interleaved: bool = True,
                    rotary_seqlens: Optional[torch.Tensor] = None,
                    workspace: Optional[DecodeWorkspace] = None,
                    return_lse: bool = False):
    """Compute one packed batch, including ragged decode/MTP/chunked-prefill queries.

    q: contiguous [allocated_tokens,Hq,D]. GPU int32 query_start_loc[B+1]
    starts at zero and is nondecreasing; its last value is <= allocated_tokens.
    Empty requests and trailing graph padding are supported (zero output, -inf LSE).
    seq_lens[B] and block_table[B,max_blocks] must be GPU int32 on the query device.

    Read-only mode (new_k/new_v omitted): seq_lens includes all query K/V already
    written to cache. Query j of request b has position seq_lens[b]-q_len[b]+j.
    With causal=True it cannot see later speculative tokens, even if those tokens
    are physically present in the cache. Stale rejected tokens beyond seq_lens
    are not read. The caller must recompute/update these lengths every iteration.

    Append mode: seq_lens is the length BEFORE append. new K/V have [T,Hkv,D]
    with the SAME query offsets as q. After writing them, attention uses the same
    bottom-right mask. This function does not commit speculative lengths or
    implement rollback, copy-on-write, or tree-attention masks.

    Metadata values and writable-page ownership are trusted hot-path inputs.
    Use lab/concurrency_canary.py on the target GPU before deployment. Unlike the
    compatibility API, this path does not silently copy/cast query or cache tensors.
    """
    from . import _C
    if getattr(_C, "DECODE_ABI_VERSION", 0) < 3:
        raise RuntimeError("ALU Attention decode ABI 3 is required; rebuild the native extension")
    if q.dim() != 3:
        raise ValueError("paged_attention expects packed q [T,Hq,D]")
    if len(window_size) != 2:
        raise ValueError("window_size must contain left and right radii")
    wl, wr = window_size
    wl = -1 if wl is None or wl < 0 else int(wl)
    wr = -1 if wr is None or wr < 0 else int(wr)
    output = scratch = lse = None
    if workspace is not None:
        if workspace.plan.kv_heads != k_cache.shape[2]:
            raise ValueError("workspace was created for a different KV-head geometry")
        if num_splits not in (0, workspace.plan.num_splits):
            raise ValueError("num_splits disagrees with workspace plan")
        if return_lse and workspace.lse is None:
            raise ValueError("workspace was not allocated with return_lse=True")
        num_splits = workspace.plan.num_splits
        output, scratch = workspace.output, workspace.scratch
        lse = workspace.lse if return_lse else None
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    rdim = rotary_cos.shape[-1] * 2 if rotary_cos is not None else 0
    out, logsumexp = _C.dec(
        q, k_cache, v_cache, block_table, seq_lens, new_k, new_v,
        rotary_cos, rotary_sin, bool(rotary_interleaved), scale,
        wl + 1 if wl >= 0 else -1, int(num_splits),
        cache_leftpad, bool(return_lse), float(softcap), alibi_slopes, rdim,
        rotary_seqlens, int(kv_row), int(v_off), query_start_loc=query_start_loc,
        causal=bool(causal), window_right=wr, attention_chunk=int(attention_chunk),
        output=output, workspace=scratch, lse_buffer=lse)
    return (out, logsumexp) if return_lse else out
