"""FA-3 argument ordering for the top-level `flash_attn_interface` package used by hopper/.

FA-2 and FA-3 use DIFFERENT positional argument orders:

    FA-2: flash_attn_func(q, k, v, dropout_p, softmax_scale, ...)
    FA-3: flash_attn_func(q, k, v, softmax_scale, causal, ...)

The fourth positional argument therefore means dropout probability in FA-2 but softmax scale in
FA-3. One function cannot safely infer the convention from the value. The interfaces are kept
separate: `import flash_attn` uses FA-2 ordering and `import flash_attn_interface` uses this FA-3
ordering. Both call the same kernels. FA-3 also returns `(out, softmax_lse)` when
`return_attn_probs=True`, while FA-2 returns a triple including the probability matrix.
Registration is handled by `alu_attn.install_as_flash_attn()`.
"""
from . import flash_attn_interface as _fa2

__all__ = ["flash_attn_func", "flash_attn_qkvpacked_func", "flash_attn_varlen_func",
           "flash_attn_with_kvcache", "flash_attn_combine", "get_scheduler_metadata"]


def flash_attn_func(q, k, v, softmax_scale=None, causal=False, qv=None,
                    q_descale=None, k_descale=None, v_descale=None,
                    window_size=(-1, -1), attention_chunk=0, softcap=0.0,
                    num_splits=1, pack_gqa=None, deterministic=False, sm_margin=0,
                    return_attn_probs=False):
    r = _fa2.flash_attn_func(q, k, v, 0.0, softmax_scale, causal, window_size, softcap,
                             None, deterministic, return_attn_probs,
                             qv=qv, attention_chunk=attention_chunk,
                             q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
                             num_splits=num_splits, pack_gqa=pack_gqa, sm_margin=sm_margin)
    return (r[0], r[1]) if return_attn_probs else r   # FA-3 returns a pair, without materializing the S×S matrix.


def flash_attn_qkvpacked_func(qkv, softmax_scale=None, causal=False,
                              q_descale=None, k_descale=None, v_descale=None,
                              window_size=(-1, -1), attention_chunk=0, softcap=0.0,
                              num_splits=1, pack_gqa=None, deterministic=False, sm_margin=0,
                              return_attn_probs=False):
    r = _fa2.flash_attn_qkvpacked_func(qkv, 0.0, softmax_scale, causal, window_size, softcap,
                                       None, deterministic, return_attn_probs,
                                       attention_chunk=attention_chunk,
                                       q_descale=q_descale, k_descale=k_descale,
                                       v_descale=v_descale, num_splits=num_splits,
                                       pack_gqa=pack_gqa, sm_margin=sm_margin)
    return (r[0], r[1]) if return_attn_probs else r


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                           seqused_q=None, seqused_k=None, softmax_scale=None, causal=False,
                           qv=None, q_descale=None, k_descale=None, v_descale=None,
                           window_size=(-1, -1), attention_chunk=0, softcap=0.0,
                           num_splits=1, pack_gqa=None, deterministic=False, sm_margin=0,
                           return_attn_probs=False):
    r = _fa2.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                                    max_seqlen_k, 0.0, softmax_scale, causal, window_size,
                                    softcap, None, deterministic, return_attn_probs,
                                    seqused_q=seqused_q, seqused_k=seqused_k, qv=qv,
                                    attention_chunk=attention_chunk,
                                    q_descale=q_descale, k_descale=k_descale,
                                    v_descale=v_descale, num_splits=num_splits,
                                    pack_gqa=pack_gqa, sm_margin=sm_margin)
    return (r[0], r[1]) if return_attn_probs else r


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, qv=None,
                            rotary_cos=None, rotary_sin=None, cache_seqlens=None,
                            cache_batch_idx=None, cache_leftpad=None, page_table=None,
                            cu_seqlens_q=None, cu_seqlens_k_new=None, max_seqlen_q=None,
                            rotary_seqlens=None, q_descale=None, k_descale=None,
                            v_descale=None, softmax_scale=None, causal=False,
                            window_size=(-1, -1), attention_chunk=0, softcap=0.0,
                            rotary_interleaved=True, scheduler_metadata=None,
                            num_splits=0, pack_gqa=None, sm_margin=0,
                            return_softmax_lse=False):
    return _fa2.flash_attn_with_kvcache(
        q, k_cache, v_cache, k=k, v=v, qv=qv, rotary_cos=rotary_cos, rotary_sin=rotary_sin,
        cache_seqlens=cache_seqlens, cache_batch_idx=cache_batch_idx,
        cache_leftpad=cache_leftpad, block_table=None, softmax_scale=softmax_scale,
        causal=causal, window_size=window_size, softcap=softcap,
        rotary_interleaved=rotary_interleaved, alibi_slopes=None, num_splits=num_splits,
        return_softmax_lse=return_softmax_lse, page_table=page_table,
        attention_chunk=attention_chunk, q_descale=q_descale, k_descale=k_descale,
        v_descale=v_descale, pack_gqa=pack_gqa, sm_margin=sm_margin,
        scheduler_metadata=scheduler_metadata, cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k_new, max_seqlen_q=max_seqlen_q,
        rotary_seqlens=rotary_seqlens)


flash_attn_combine = _fa2.flash_attn_combine


def get_scheduler_metadata(*args, **kw):
    """FA-3 can precompute a schedule for a specific shape and pass it back as scheduler_metadata.
    Here the launcher computes the schedule on every invocation in only a few microseconds, so
    there is nothing useful to cache. Returning None tells FA-3-compatible callers to schedule
    normally and preserves compatibility without source changes."""
    return None
