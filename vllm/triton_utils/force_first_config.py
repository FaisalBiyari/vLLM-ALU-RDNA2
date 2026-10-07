# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skip Triton autotuning under VLLM_TRITON_FORCE_FIRST_CONFIG."""

import os

from vllm.logger import init_logger
from vllm.triton_utils.importing import HAS_TRITON

logger = init_logger(__name__)

_installed: bool = False

# BEGIN gfx1030 GDN static winners (vLLM 0.30 Module 04)
# Proven on the RDNA2/gfx1030 Qwen GDN prefill path.  Match by config identity
# rather than historical list index because vLLM 0.30 contains multiple
# implementations of some kernel names with differently sized config lists.
_GFX1030_GDN_STATIC_WINNERS: dict[str, dict[str, object]] = {
    "chunk_local_cumsum_scalar_kernel": {
        "kwargs": {},
        "num_warps": 1,
        "num_stages": 3,
    },
    "chunk_scaled_dot_kkt_fwd_kernel": {
        "kwargs": {"BK": 32},
        "num_warps": 4,
        "num_stages": 4,
    },
    "merge_16x16_to_64x64_inverse_kernel": {
        "kwargs": {},
        "num_warps": 2,
        "num_stages": 4,
    },
    "recompute_w_u_fwd_kernel": {
        "kwargs": {},
        "num_warps": 8,
        "num_stages": 2,
    },
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": {
        "kwargs": {"BV": 32},
        "num_warps": 4,
        "num_stages": 2,
    },
    "chunk_fwd_kernel_o": {
        "kwargs": {"BK": 32, "BV": 64},
        "num_warps": 8,
        "num_stages": 2,
    },
}


def _gfx1030_gdn_static_winners_enabled() -> bool:
    return os.environ.get(
        "VLLM_ROCM_GFX1030_GDN_STATIC_WINNERS", "0"
    ).strip().lower() in ("1", "true")


def _gfx1030_gdn_static_winner_indices(kernel_name: str, configs) -> list[int]:
    spec = _GFX1030_GDN_STATIC_WINNERS.get(kernel_name)
    if spec is None:
        return []

    expected_kwargs = spec["kwargs"]
    expected_warps = spec["num_warps"]
    expected_stages = spec["num_stages"]

    matches: list[int] = []
    for idx, config in enumerate(configs):
        if (
            dict(getattr(config, "kwargs", {})) == expected_kwargs
            and getattr(config, "num_warps", None) == expected_warps
            and getattr(config, "num_stages", None) == expected_stages
        ):
            matches.append(idx)
    return matches
# END gfx1030 GDN static winners (vLLM 0.30 Module 04)


def is_installed() -> bool:
    """Return whether the first-valid-config patch is currently installed."""
    return _installed


def install() -> None:
    """Install the Autotuner.run replacement."""
    global _installed
    if _installed:
        return
    if not HAS_TRITON:
        return

    import importlib

    autotuner_mod = importlib.import_module("triton.runtime.autotuner")
    Autotuner = autotuner_mod.Autotuner
    from triton.compiler.errors import CompileTimeAssertionFailure
    from triton.runtime.errors import OutOfResources, PTXASError

    _invalid_config_errors = (OutOfResources, CompileTimeAssertionFailure, PTXASError)
    _picked_cache: dict[tuple, int] = {}
    seen_kernels: set[str] = set()
    seen_static_winners: set[str] = set()

    def _run_first_valid_config(self, *args, **kwargs):
        if not self.configs:
            return self.fn(*args, **kwargs)

        key_vals = tuple(kwargs[name] for name in self.keys if name in kwargs)
        cache_key = (id(self), key_vals)
        kernel_name = getattr(self.base_fn, "__name__", repr(self.fn))

        cached_idx = _picked_cache.get(cache_key)
        if cached_idx is not None:
            candidate_indices = [cached_idx]
        else:
            candidate_indices = list(range(len(self.configs)))
            if _gfx1030_gdn_static_winners_enabled():
                preferred_indices = _gfx1030_gdn_static_winner_indices(
                    kernel_name, self.configs
                )
                if len(preferred_indices) == 1:
                    preferred_idx = preferred_indices[0]
                    candidate_indices = [preferred_idx] + [
                        idx for idx in candidate_indices if idx != preferred_idx
                    ]
                    if kernel_name not in seen_static_winners:
                        seen_static_winners.add(kernel_name)
                        logger.info(
                            "[triton-static-winner] kernel=%s "
                            "preferred_index=%d preferred=%s",
                            kernel_name,
                            preferred_idx,
                            self.configs[preferred_idx],
                        )
                elif kernel_name in _GFX1030_GDN_STATIC_WINNERS:
                    if kernel_name not in seen_static_winners:
                        seen_static_winners.add(kernel_name)
                        logger.warning(
                            "[triton-static-winner-skip] kernel=%s "
                            "signature_matches=%d; preserving stock force-first order",
                            kernel_name,
                            len(preferred_indices),
                        )

        last_exc: Exception | None = None
        for idx in candidate_indices:
            config = self.configs[idx]
            if config.pre_hook is not None:
                full_nargs = {
                    **dict(zip(self.arg_names, args)),
                    **kwargs,
                    **config.all_kwargs(),
                }
                config.pre_hook(full_nargs)
            # Prefer self.fn.run(...) — the kernel-launch entrypoint for both
            # JITFunction and Heuristics. Calling JITFunction(...) directly
            # raises "Cannot call @triton.jit'd outside of the scope of a
            # kernel". Fall back to plain call only if .run is missing.
            launch = getattr(self.fn, "run", self.fn)
            try:
                result = launch(*args, **kwargs, **config.all_kwargs())
            except _invalid_config_errors as e:
                last_exc = e
                continue

            if cached_idx is None:
                _picked_cache[cache_key] = idx
                self.best_config = config
                if kernel_name not in seen_kernels:
                    seen_kernels.add(kernel_name)
                    logger.info(
                        "[triton-autotune-disabled] kernel=%s configs=%d "
                        "picked_index=%d picked=%s",
                        kernel_name,
                        len(self.configs),
                        idx,
                        config,
                    )
            return result

        raise RuntimeError(
            f"No valid config for kernel "
            f"{kernel_name} key={key_vals} (tried {len(self.configs)} configs)"
        ) from last_exc

    Autotuner.run = _run_first_valid_config
    _installed = True
