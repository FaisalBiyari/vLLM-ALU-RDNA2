# Submitting to upstream vLLM

Upstream accepts **small, isolated, arch-gated** changes. This fork's 17
files + 1 new file split into ~7 PRs, smallest-merge-first. Every change is
a **no-op on CUDA** (arch-gated on `gfx1030` / ROCm), which is what makes
them reviewable upstream.

**Order matters.** Land the platform helpers first; later PRs build on them.

## PR map

| # | PR | Files | Risk |
|---|---|---|---|
| 1 | `on_gfx1030()` / `on_rdna2()` platform helpers | `vllm/platforms/rocm.py` | trivial — pure queries, cached at import |
| 2 | bf16x3 JIT warmup: skip on non-CUDA | `vllm/model_executor/warmup/kernel_warmup.py` | trivial — 1 guard |
| 3 | pip-ROCm userspace lib preload | `vllm/env_override.py` | low — fixes symbol preemption for pip-installed ROCm |
| 4 | gfx1030 skinny-GEMM routing + M=1 path | `vllm/model_executor/layers/utils.py` | low — routes to existing kernels |
| 5 | TunableOp explicit DB loader | `vllm/v1/executor/multiproc_executor.py` | low — deterministic kernel config |
| 6 | **wvSplitK small-M decode GEMM** | `csrc/rocm/skinny_gemms.cu` + `CMakeLists.txt` | the big one — see below |
| 7 | ALU_ATTN attention backend | `alu_attn.py` + `registry.py` + `platforms/rocm.py` selection | needs the `alu_attn` engine dependency (see below) |
| 8 | (optional) XGMI allreduce router | `cuda_communicator.py`, `custom_all_reduce.py`, `quick_all_reduce.py` | XGMI/multi-GPU only |
| 9 | (optional) Triton tuning: wide-UA + split-KV + GDN winners | `triton_attn.py`, `triton_unified_attention.py`, `chunked_prefill_paged_decode.py`, `force_first_config.py` | perf-only, env-gated |

## Rules that get these merged

- **Keep the gates.** Every new path checks `on_gfx1030()` (or ROCm +
  arch). CUDA never sees the new code. That is the whole review argument.
- **Env-gate the perf tuning.** `VLLM_ROCM_GFX1030_*` vars default OFF.
  Upstream hates behavior changes that need no opt-in.
- **One concern per PR.** wvSplitK is its own PR with its own bench numbers.
- **PR-6 needs a bench table.** Small-M decode GEMM: M=1..5, N/K shapes,
  vs `F.linear`, on gfx1030, with the "finite-but-wrong" correctness check
  (max-abs-err vs fp32 reference). Numbers in the PR description.
- **DCO sign-off required.** `git commit -s`.

## PR-7 (ALU_ATTN) — the dependency question

The attention backend imports `alu_attn` (the engine in `alu-attention/`).
Upstream vLLM will not vendor a new kernel engine. Options:

- **Best:** publish `alu_attn` as a standalone PyPI/ROCm package first
  (it is API-compatible with flash-attn). The vLLM PR then just adds the
  backend adapter + registry entry, with `alu_attn` an **optional**
  dependency (import guarded — exactly as `vllm/v1/attention/backends/
  alu_attn.py` already does). No `alu_attn` → backend unavailable → stock
  ROCm ordering. That is the pattern upstream accepts for optional
  accelerators.
- The adapter's `try: import alu_attn / except ImportError` guard is the
  load-bearing line. Keep it.

## What does NOT go upstream

- Lab run scripts, model paths, tuning DBs, internal benchmarks.
- The `_rocm_C.abi3.so` binary (upstream builds its own from source).
- Env vars that encode lab-specific tuning history — rename to neutral
  upstream naming if they go in (e.g. `VLLM_ROCM_GFX1030_WIDE_UA` →
  `VLLM_ROCM_TRITON_UA_WIDE_TILE`).

## Checklist before opening any PR

- [ ] Fork `vllm-project/vllm`, branch from `main` (not this fork).
- [ ] Cherry-pick ONLY the files for that PR.
- [ ] `git commit -s` (DCO).
- [ ] Pre-commit hooks pass: `pre-commit run --all-files`.
- [ ] ROCm CI (`.buildkite/`) — at minimum confirm it does not regress
      non-gfx1030 ROCm (the gates guarantee it, but show the gate in the
      PR diff so reviewers see it).
- [ ] Bench numbers in the description for perf PRs.
