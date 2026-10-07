# SPDX-License-Identifier: Apache-2.0
"""ALU Attention backend for AMD RDNA2 (gfx1030).

ALU Attention is the default MHA backend for gfx1030. It runs standard causal
decoder attention for prompt processing, decode and MTP verification using
native ALU HIP kernels.

The Triton backend classes are inherited only for vLLM metadata construction
and the existing KV-cache update contract; Triton attention computation is
not used by this backend's forward path.
"""

import torch

from alu_attn import _C
from alu_attn.concurrency import SUPPORTED_GQA, SUPPORTED_HEAD_DIMS

from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadataBuilder,
)


class AluAttentionBackend(TritonAttentionBackend):

    @staticmethod
    def get_name():
        return "ALU_ATTN"

    @classmethod
    def supports_batch_invariance(cls):
        # Split scheduling may change reduction order with batch shape.
        return False

    @staticmethod
    def get_impl_cls():
        return AluAttentionImpl

    @staticmethod
    def get_builder_cls():
        return AluAttentionMetadataBuilder


class AluAttentionMetadataBuilder(TritonAttentionMetadataBuilder):
    """Add CPU-derived ALU dispatch state to Triton-compatible metadata."""

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        fast_build=False,
    ):
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build,
        )

        # This classification is performed entirely from vLLM's CPU-side
        # scheduler metadata. It introduces no device synchronization.
        #
        # seq_lens_cpu_upper_bound is exact for prefill rows. A request is a
        # fresh prefill iff its sequence length equals its current query
        # length. We require the entire batch to be fresh before using the
        # dense ALU forward kernel; mixed decode/extend/prefill batches remain
        # on the native paged path.
        all_fresh_prefill = False

        # Single-request prompt metadata used by the native ALU
        # paged-KV gather -> fwd_varlen path.
        #
        # These values come from vLLM's CPU-side scheduler metadata.
        # No GPU-resident scalar is read and no device synchronization
        # is introduced here.
        single_prefill = False
        single_q_len = 0
        single_seq_len = 0

        if (
            common_attn_metadata.num_reqs > 0
            and common_attn_metadata.max_query_len > 16
            and common_attn_metadata.seq_lens_cpu_upper_bound is not None
        ):
            query_start_loc_cpu = (
                common_attn_metadata.query_start_loc_cpu
            )

            query_lens = (
                query_start_loc_cpu[1:]
                - query_start_loc_cpu[:-1]
            )

            seq_lens = (
                common_attn_metadata.seq_lens_cpu_upper_bound[
                    : common_attn_metadata.num_reqs
                ]
            )

            all_fresh_prefill = bool(
                (seq_lens == query_lens).all().item()
            )

            if common_attn_metadata.num_reqs == 1:
                # .item() here reads CPU scheduler tensors, not GPU tensors.
                single_q_len = int(query_lens[0].item())
                single_seq_len = int(seq_lens[0].item())

                single_prefill = (
                    single_q_len > 16
                    and single_seq_len >= single_q_len
                    and single_q_len
                    == int(common_attn_metadata.num_actual_tokens)
                )

        metadata.alu_all_fresh_prefill = (
            all_fresh_prefill
        )

        metadata.alu_single_prefill = single_prefill
        metadata.alu_single_q_len = single_q_len
        metadata.alu_single_seq_len = single_seq_len

        # ALU_DISPATCH_DEBUG (0.6.3 concurrency dev): CPU-side dispatch
        # counter. The metadata builder runs in Python for EVERY step
        # (prefill + decode), including CUDA-graph decode replays (only
        # the model forward is captured, the builder is not), so this is
        # the runtime route proof for graph mode. Classification mirrors
        # forward(): FRESH = all-new-prefill batch (_C.fwd_varlen),
        # SPP = single-request paged prefill (_C.fwd_varlen),
        # DEC = native paged decode (_C.dec). All three are native ALU
        # kernels; Triton is inherited for metadata only.
        if __import__("os").environ.get("ALU_DISPATCH_DEBUG") == "1":
            _sw = getattr(self, "sliding_window", (-1, -1))
            if isinstance(_sw, (tuple, list)):
                _sw0, _sw1 = _sw[0], _sw[1]
            else:
                _sw0 = _sw1 = _sw
            _b_fresh = bool(all_fresh_prefill) and _sw0 < 0 and _sw1 < 0
            _b_spp = bool(single_prefill) and not _b_fresh and _sw0 < 0 and _sw1 < 0
            _route = "FRESH" if _b_fresh else ("SPP" if _b_spp else "DEC")
            _bst = getattr(self, "_alu_bdisp", None)
            if _bst is None:
                _bst = self._alu_bdisp = {"FRESH": 0, "SPP": 0, "DEC": 0,
                                          "n": 0, "np_hi": 0}
            _bst[_route] += 1
            try:
                _bst["n"] += 1
                _bst["np_hi"] = max(_bst["np_hi"],
                                    int(common_attn_metadata.num_reqs))
                if _bst["n"] % 200 == 0:
                    import torch as _t
                    try:
                        _rank = (_t.distributed.get_rank()
                                 if _t.distributed.is_initialized() else 0)
                    except Exception:
                        _rank = "?"
                    import sys as _s
                    _bn, _bf, _bs, _bd, _bh = (_bst["n"], _bst["FRESH"],
                                               _bst["SPP"], _bst["DEC"],
                                               _bst["np_hi"])
                    print(f"[ALU_BUILD] rank={_rank} steps={_bn} "
                          f"FRESH={_bf} SPP={_bs} DEC={_bd} "
                          f"num_reqs_hi={_bh}",
                          file=_s.stderr, flush=True)
            except Exception:
                pass

        return metadata


class AluAttentionImpl(TritonAttentionImpl):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if getattr(_C, "DECODE_ABI_VERSION", 0) < 3:
            raise RuntimeError(
                "ALU Attention requires native decode ABI 3 or newer"
            )

        if not torch.version.hip:
            raise RuntimeError(
                "ALU Attention requires ROCm PyTorch"
            )

        props = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        )

        arch = getattr(
            props,
            "gcnArchName",
            "",
        ).split(":", 1)[0]

        if arch != "gfx1030":
            raise RuntimeError(
                f"ALU_ATTN currently supports gfx1030 only; got {arch!r}"
            )

    def _unsupported_reason(
        self,
        layer,
        query,
        kv_cache,
        meta,
        output,
        output_scale,
        output_block_scale,
    ):
        if self.attn_type != AttentionType.DECODER:
            return "only decoder self-attention is supported"

        if meta.use_cascade:
            return "cascade attention is not supported"

        if not isinstance(meta.causal, bool) or not meta.causal:
            return "standard causal decoder attention is required"

        if self.kv_cache_dtype not in (
            "auto",
            "float16",
            "bfloat16",
            "float32",
        ):
            return (
                "unsupported KV cache dtype: "
                f"{self.kv_cache_dtype}"
            )

        if getattr(self, "_is_per_token_head_quant", False):
            return "per-token/head KV quantization is not supported"

        if output_scale is not None:
            return "fused output scaling is not supported"

        if output_block_scale is not None:
            return "fused output block scaling is not supported"

        if self.alibi_slopes is not None:
            return "ALiBi is not supported"

        if self.sinks is not None:
            return "attention sinks are not supported"

        if self.use_alibi_sqrt:
            return "ALiBi sqrt mode is not supported"

        if self.chunk_lookback != -1:
            return "chunk lookback is not supported"

        for name in (
            "mm_prefix_range_tensor",
            "rswa_prefix_lens",
            "rswa_window",
        ):
            if getattr(meta, name, None) is not None:
                return (
                    "unsupported attention metadata: "
                    f"{name}"
                )

        # D512_ENABLEMENT_MARKER (tree-local, d512 validation stack):
        # Gemma4 sets mm_prefix_clamp_sliding_window statically on its sliding
        # layers (gemma4.py:531) even in pure-text --language-model-only
        # serving. The in-kernel clamp only acts on bidirectional mm ranges,
        # which exist only when the mm metadata tensors below are non-None —
        # already rejected by the loop above. Co-firing on the static flag
        # alone false-positives for all text requests (server crash during
        # cudagraph profiling). Co-firing with actual mm metadata is a
        # provable no-op (the loop above guarantees all tensors are None when
        # it passes) and keeps the guard intact for any future path where the
        # flag could matter.
        if getattr(
            layer,
            "mm_prefix_clamp_sliding_window",
            False,
        ) and any(
            getattr(meta, n, None) is not None
            for n in (
                "mm_prefix_range_tensor",
                "rswa_prefix_lens",
                "rswa_window",
            )
        ):
            return (
                "multimodal prefix window adjustment "
                "is not supported"
            )

        if query.dim() != 3:
            return (
                "packed query [T,H,D] is required"
            )

        if not query.is_contiguous():
            return "query must be contiguous"

        if query.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            return (
                "unsupported query dtype: "
                f"{query.dtype}"
            )

        if self.head_size not in SUPPORTED_HEAD_DIMS:
            return (
                "unsupported ALU head size: "
                f"{self.head_size}"
            )

        if self.num_heads % self.num_kv_heads:
            return (
                "query heads must be divisible "
                "by KV heads"
            )

        gqa = self.num_heads // self.num_kv_heads

        if gqa not in SUPPORTED_GQA:
            return (
                "unsupported rank-local GQA ratio: "
                f"{gqa}"
            )

        if kv_cache.dim() != 4:
            return "expected 4-D paged KV cache"

        if kv_cache.shape[1] != self.num_kv_heads:
            return (
                "KV cache head count does not match "
                "local KV heads"
            )

        if kv_cache.shape[-1] != 2 * self.head_size:
            return (
                "ALU requires interleaved [K|V] "
                "cache rows"
            )

        if kv_cache.dtype != query.dtype:
            return "query/cache dtype mismatch"

        page = int(kv_cache.shape[2])

        if page < 1 or (page & (page - 1)):
            return (
                "ALU requires a power-of-two "
                "cache page size"
            )

        if meta.query_start_loc is None:
            return "query_start_loc is required"

        if meta.seq_lens is None:
            return "seq_lens is required"

        if meta.block_table is None:
            return "block_table is required"

        if not output.is_contiguous():
            return "output must be contiguous"

        if output.numel() != query.numel():
            return (
                "output buffer does not match "
                "query size"
            )

        return None

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if attn_metadata is None:
            return output.fill_(0)

        reason = self._unsupported_reason(
            layer,
            query,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )

        if reason is not None:
            raise RuntimeError(
                "ALU_ATTN configuration is unsupported: "
                f"{reason}"
            )

        # ALU_DISPATCH_DEBUG (0.6.3 concurrency dev): per-step dispatch
        # counter, printed to stderr every 1000 forwards per worker.
        # Inert unless the env var is set to "1". It resolves the actual
        # route exactly as forward() does below (FRESH/SPP = _C.fwd_varlen,
        # DEC = _C.dec; all native ALU, no Triton) including the
        # fresh-prefill demotions (key/value None, sliding-window layers,
        # non-fp16, capture).
        if __import__("os").environ.get("ALU_DISPATCH_DEBUG") == "1":
            _sw = self.sliding_window
            _fresh_r = (
                bool(getattr(attn_metadata, "alu_all_fresh_prefill", False))
                and key is not None and value is not None
                and _sw[0] < 0 and _sw[1] < 0
            )
            _spp_r = (
                bool(getattr(attn_metadata, "alu_single_prefill", False))
                and not _fresh_r
                and _sw[0] < 0 and _sw[1] < 0
                and query.dtype == torch.float16
                and not torch.cuda.is_current_stream_capturing()
            )
            _route = "FRESH" if _fresh_r else ("SPP" if _spp_r else "DEC")
            _st = getattr(self, "_alu_disp", None)
            if _st is None:
                _st = self._alu_disp = {
                    "FRESH": 0, "SPP": 0, "DEC": 0, "n": 0,
                    "np_lo": 10 ** 9, "np_hi": 0, "nq_lo": 10 ** 9, "nq_hi": 0,
                }
            _st[_route] += 1
            _np = int(getattr(attn_metadata, "num_reqs", -1))
            if _np < 0 and attn_metadata.query_start_loc is not None:
                _np = int(attn_metadata.query_start_loc.shape[0] - 1)
            _nq = int(attn_metadata.num_actual_tokens)
            _st["np_lo"] = min(_st["np_lo"], _np)
            _st["np_hi"] = max(_st["np_hi"], _np)
            _st["nq_lo"] = min(_st["nq_lo"], _nq)
            _st["nq_hi"] = max(_st["nq_hi"], _nq)
            _st["n"] += 1
            if _st["n"] % 100 == 0:
                try:
                    _rank = (torch.distributed.get_rank()
                             if torch.distributed.is_initialized() else 0)
                except Exception:
                    _rank = "?"
                import sys as _s
                print(
                    f"[ALU_DISPATCH] rank={_rank} steps={_st['n']} "
                    f"FRESH={_st['FRESH']} SPP={_st['SPP']} DEC={_st['DEC']} "
                    f"num_reqs=[{_st['np_lo']},{_st['np_hi']}] "
                    f"num_tokens=[{_st['nq_lo']},{_st['nq_hi']}]",
                    file=_s.stderr, flush=True,
                )

        # Fresh prompt processing uses ALU's tiled Flash-Attention-style
        # forward kernel. This avoids using the decode-oriented paged kernel
        # for thousands of query rows.
        #
        # Only pure fresh-prefill batches use this path. Any batch containing
        # decode or cached/extended prompt rows remains on _C.dec until ALU
        # gains a dedicated native paged-prefill kernel.
        use_fresh_prefill = bool(
            getattr(
                attn_metadata,
                "alu_all_fresh_prefill",
                False,
            )
        )

        # The currently qualified Qwen path has no sliding window. Keep
        # non-default sliding-window configurations on the already-qualified
        # native paged path until their fwd_varlen semantics are validated.
        no_sliding_window = (
            self.sliding_window[0] < 0
            and self.sliding_window[1] < 0
        )

        # KV-shared layers (e.g. Gemma4 MTP / assistant draft layers) do not
        # compute their own K/V; they read the target's K/V from the shared
        # paged KV cache (is_kv_shared_layer=True, kv_sharing_target_layer_name
        # set by the proposer). The fresh-prefill fast path is an optimization
        # that needs the layer's *current* K/V tensors, which are absent here
        # (key/value are None). Falling through lets the native paged path
        # (_C.dec / single-paged-prefill) read K/V from the shared cache — the
        # documented "MTP verification" path. Previously this raised and
        # crashed EngineCore the first time a KV-shared draft layer hit a
        # fresh-prefill batch.
        if use_fresh_prefill and (key is None or value is None):
            use_fresh_prefill = False

        if use_fresh_prefill and no_sliding_window:

            num_actual_tokens = (
                attn_metadata.num_actual_tokens
            )

            q = query[:num_actual_tokens]
            k = key[:num_actual_tokens]
            v = value[:num_actual_tokens]

            if not q.is_contiguous():
                q = q.contiguous()

            if not k.is_contiguous():
                k = k.contiguous()

            if not v.is_contiguous():
                v = v.contiguous()

            result = _C.fwd_varlen(
                q,
                k,
                v,
                attn_metadata.query_start_loc,
                attn_metadata.query_start_loc,
                attn_metadata.max_query_len,
                attn_metadata.max_seq_len,
                self.scale,
                True,
                -1,
                0,
                (
                    0.0
                    if self.logits_soft_cap is None
                    else self.logits_soft_cap
                ),
                None,
                0.0,
                0,
            )

            if isinstance(result, tuple):
                result = result[0]

            output_view = output.view_as(query)

            output_view[:num_actual_tokens].copy_(
                result
            )

            if num_actual_tokens < output_view.shape[0]:
                output_view[
                    num_actual_tokens:
                ].zero_()

            return output

        # Native ALU paged-prefill architecture (single request):
        #
        #   paged vLLM KV cache
        #       -> gather the one request's physical pages
        #       -> dense K/V through the current sequence length
        #       -> native ALU fwd_varlen
        #
        # q_len and seq_len are
        # supplied by CPU scheduler metadata. No GPU .item() synchronization
        # is needed.
        #
        # This path is deliberately restricted to:
        #   * one request,
        #   * real prompt processing (q_len > 16),
        #   * FP16, which is the qualified Qwen3.8 deployment,
        #   * eager/non-captured execution,
        #   * no sliding window.
        #
        # Multi-request/mixed batches remain on ALU's native paged _C.dec
        # path and are not changed by this optimization.
        use_single_paged_prefill = bool(
            getattr(
                attn_metadata,
                "alu_single_prefill",
                False,
            )
        )

        if (
            use_single_paged_prefill
            and not use_fresh_prefill
            and no_sliding_window
            and query.dtype == torch.float16
            and not torch.cuda.is_current_stream_capturing()
        ):
            q_len = int(
                getattr(
                    attn_metadata,
                    "alu_single_q_len",
                    0,
                )
            )
            seq_len = int(
                getattr(
                    attn_metadata,
                    "alu_single_seq_len",
                    0,
                )
            )

            if q_len <= 16 or seq_len < q_len:
                raise RuntimeError(
                    "invalid ALU single-request prefill metadata: "
                    f"q_len={q_len}, seq_len={seq_len}"
                )

            if q_len != attn_metadata.num_actual_tokens:
                raise RuntimeError(
                    "ALU single-request prefill token-count mismatch: "
                    f"q_len={q_len}, "
                    f"num_actual_tokens="
                    f"{attn_metadata.num_actual_tokens}"
                )

            hs = self.head_size

            # vLLM:
            #   [blocks, kv_heads, page, 2*head_size]
            #
            # ALU dense forward wants:
            #   [sequence, kv_heads, head_size]
            paged = kv_cache.transpose(1, 2)
            key_cache, value_cache = paged.split(
                hs,
                dim=-1,
            )

            page = int(key_cache.shape[1])
            kv_heads = int(key_cache.shape[2])
            num_pages = int(key_cache.shape[0])

            needed_pages = (
                seq_len + page - 1
            ) // page

            table_width = int(
                attn_metadata.block_table.shape[1]
            )

            needed_pages = min(
                needed_pages,
                num_pages,
                table_width,
            )

            if needed_pages <= 0:
                raise RuntimeError(
                    "ALU paged-prefill resolved zero KV pages"
                )

            # Physical page IDs stay GPU-resident.
            block_ids = (
                attn_metadata.block_table[
                    0,
                    :needed_pages,
                ]
                .to(dtype=torch.long)
                .contiguous()
            )

            # index_select materializes only the pages belonging to this
            # request. Flatten page-major storage and trim padding after
            # the final partially-filled page.
            k_dense = (
                key_cache
                .index_select(0, block_ids)
                .reshape(
                    -1,
                    kv_heads,
                    hs,
                )[:seq_len]
                .contiguous()
            )

            v_dense = (
                value_cache
                .index_select(0, block_ids)
                .reshape(
                    -1,
                    kv_heads,
                    hs,
                )[:seq_len]
                .contiguous()
            )

            q = (
                query[:q_len]
                .reshape(
                    q_len,
                    self.num_heads,
                    hs,
                )
                .contiguous()
            )

            cu_q = (
                attn_metadata.query_start_loc[:2]
                .to(dtype=torch.int32)
                .contiguous()
            )

            cu_k = torch.tensor(
                [0, seq_len],
                dtype=torch.int32,
                device=q.device,
            )

            result = _C.fwd_varlen(
                q,
                k_dense,
                v_dense,
                cu_q,
                cu_k,
                q_len,
                seq_len,
                self.scale,
                True,
                -1,
                0,
                (
                    0.0
                    if self.logits_soft_cap is None
                    else self.logits_soft_cap
                ),
                None,
                0.0,
                0,
            )

            if isinstance(result, tuple):
                result = result[0]

            output_view = output.view_as(query)

            output_view[:q_len].copy_(result)

            if q_len < output_view.shape[0]:
                output_view[q_len:].zero_()

            return output

        # Decode, MTP verification, multi-request cached prompt processing,
        # mixed batches, and other non-C1 prompt extensions consume vLLM's
        # paged KV cache directly through native ALU decode.
        cache = kv_cache.transpose(1, 2)

        _C.dec(
            query,
            cache,
            cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            None,
            None,
            None,
            None,
            False,
            self.scale,
            (
                self.sliding_window[0] + 1
                if self.sliding_window[0] >= 0
                else -1
            ),
            0,
            kv_row=0,
            v_off=self.head_size,
            query_start_loc=attn_metadata.query_start_loc,
            causal=True,
            window_right=self.sliding_window[1],
            softcap=self.logits_soft_cap,
            output=output.view_as(query),
        )

        return output
