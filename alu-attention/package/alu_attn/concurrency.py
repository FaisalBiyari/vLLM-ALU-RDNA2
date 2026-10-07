"""Pure host-side capacity and scheduling math for ALU Attention.

64 is the initial qualification target, not an architectural batch-size limit.
Memory estimates cover full-attention KV tensors only, not model weights, MTP
weights/state, linear-attention state, graphs, collectives, or activations.
"""
from dataclasses import dataclass
from typing import Iterable

TARGET_CONCURRENCY = 64
SUPPORTED_GQA = (1, 2, 3, 4, 6, 8)
# D512 added for Gemma4 full-attention layers: fa_forward_d512_k (D-split kernel, BR=32/BC=32,
# LDS 50KB) + fa_decode D=512 specializations; validated by the fa_forward/fa_decode gates.
SUPPORTED_HEAD_DIMS = (16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 512)


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class DecodePlan:
    query_rows: int
    query_heads: int
    kv_heads: int
    head_dim: int
    num_splits: int
    scratch_elements: int

    @property
    def scratch_bytes(self) -> int:
        return 4 * self.scratch_elements


def plan_decode(query_rows: int, query_heads: int, kv_heads: int, head_dim: int,
                context_bound: int, cu_count: int, num_splits: int = 0) -> DecodePlan:
    """Match kernels/fa_decode_schedule.hpp without reading device metadata.

    query_rows is allocated token capacity, including graph padding. It is NOT
    request count: B requests with K speculative tokens may have B*(K+1) rows.
    context_bound comes from block-table width times page size, not a GPU reduction.
    """
    for name, value, minimum in (("query_rows", query_rows, 0), ("query_heads", query_heads, 1),
                                 ("kv_heads", kv_heads, 1), ("head_dim", head_dim, 1),
                                 ("context_bound", context_bound, 1), ("cu_count", cu_count, 0),
                                 ("num_splits", num_splits, 0)):
        _integer(value, name, minimum)
    if query_heads % kv_heads or query_heads // kv_heads not in SUPPORTED_GQA:
        raise ValueError("unsupported rank-local GQA geometry")
    if head_dim not in SUPPORTED_HEAD_DIMS:
        raise ValueError("head_dim needs a native decode specialization")
    if num_splits > 512:
        raise ValueError("num_splits cannot exceed 512")
    splits = num_splits
    if splits == 0:
        if query_rows == 0:
            splits = 1
        else:
            groups = query_rows * kv_heads
            target = 4 * (cu_count or 72)
            splits = min(512, max(1, (context_bound + 255) // 256),
                         max(1, (target + groups - 1) // groups))
    elements = 0 if splits == 1 else query_rows * query_heads * splits * (head_dim + 2)
    return DecodePlan(query_rows, query_heads, kv_heads, head_dim, splits, elements)


def local_head_geometry(query_heads: int, kv_heads: int, tensor_parallel_size: int):
    """Return per-rank heads with KV sharding or replication, not TP collectives."""
    _integer(query_heads, "query_heads", 1)
    _integer(kv_heads, "kv_heads", 1)
    _integer(tensor_parallel_size, "tensor_parallel_size", 1)
    if query_heads % tensor_parallel_size:
        raise ValueError("query heads must divide evenly across TP ranks")
    if kv_heads >= tensor_parallel_size:
        if kv_heads % tensor_parallel_size:
            raise ValueError("KV heads must divide evenly across TP ranks")
        local_kv = kv_heads // tensor_parallel_size
    else:
        if tensor_parallel_size % kv_heads:
            raise ValueError("KV replication requires TP to be a multiple of KV heads")
        local_kv = 1
    local_q = query_heads // tensor_parallel_size
    if local_q % local_kv or local_q // local_kv not in SUPPORTED_GQA:
        raise ValueError("unsupported rank-local GQA geometry")
    return local_q, local_kv


def full_attention_kv_bytes(sequence_lengths: Iterable[int], *, attention_layers: int,
                            local_kv_heads: int, head_dim: int, element_bytes: int = 2,
                            page_size: int = 1024, lookahead_tokens: int = 0) -> int:
    """Conservative rank-local KV budget with no prefix sharing or cache eviction.

    sequence_lengths include prompt and generated tokens. Each request reserves
    its own rounded-up pages; speculative lookahead can require an additional page.
    Layer count must be supplied from the actual model, especially for hybrid models.
    """
    for name, value, minimum in (("attention_layers", attention_layers, 1),
                                 ("local_kv_heads", local_kv_heads, 1),
                                 ("head_dim", head_dim, 1), ("element_bytes", element_bytes, 1),
                                 ("page_size", page_size, 1), ("lookahead_tokens", lookahead_tokens, 0)):
        _integer(value, name, minimum)
    if page_size & (page_size - 1):
        raise ValueError("page_size must be a power of two")
    pages = 0
    for length in sequence_lengths:
        _integer(length, "sequence length", 0)
        pages += (length + lookahead_tokens + page_size - 1) // page_size
    return pages * page_size * attention_layers * local_kv_heads * head_dim * element_bytes * 2
