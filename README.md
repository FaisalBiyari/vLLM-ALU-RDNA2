# vLLM-ALU-RDNA2

An unofficial RDNA2-focused distribution of **vLLM 0.30.0** with
**ALU Attention** and optimized **gfx1030 / RDNA2** serving paths.

vLLM **0.30.0** source with a **gfx1030 / RDNA2** attention backend and decode
kernels built in. Upstream vLLM does not ship working attention or
small-M decode GEMMs for RDNA2 (W6800 / W6900X / V620 class GPUs).
This repo does. Build it, run it, serve on it.

**Validated:** Dual AMD Radeon PRO W6800X Duo MPX GPU Modules, 
logically 4× Radeon Pro W6800X dies (gfx1030), with Infinity Fabric Link Bridge (IFLB),
TP4, Ubuntu 26.04, Python 3.13.15, torch `2.13.0+rocm10.0.0`.
Qwen3.8-27B fp16, 262,144 context, CUDA graphs, MTP, 40-way concurrency — all serving.

## What is different from upstream vLLM 0.30.0

Everything below is **additive**. No upstream code path is removed.
On CUDA / other ROCm arches these changes are no-ops (arch-gated).

### New files (1)

- `vllm/v1/attention/backends/alu_attn.py` — ALU_ATTN backend adapter (prefill + paged decode + MTP verify). Auto-selected on gfx1030 — no flag.

### Modified files (17)

| File | Change |
|---|---|
| `vllm/platforms/rocm.py` | `on_gfx1030()` / `on_rdna2()` helpers; ALU_ATTN ranked first MHA backend on gfx1030 |
| `vllm/v1/attention/backends/registry.py` | register `ALU_ATTN` |
| `vllm/v1/attention/backends/triton_attn.py` | gfx1030 wide-tile tuning for explicit Triton use |
| `vllm/v1/attention/ops/triton_unified_attention.py` | gfx1030 wide-UA long-prefill tuning (env-gated) |
| `vllm/v1/attention/ops/chunked_prefill_paged_decode.py` | gfx1030 split-KV paged decode (env-gated) |
| `csrc/rocm/skinny_gemms.cu` | **wvSplitK** weight-stationary small-M decode GEMM (compile-gated) |
| `CMakeLists.txt` | compile flag `VLLM_ROCM_GFX1030_WVSPLITK=1` opt-in |
| `vllm/envs.py` | the 4 `VLLM_ROCM_GFX1030_*` env vars |
| `vllm/env_override.py` | pip-ROCm userspace lib preload fix |
| `vllm/model_executor/models/config.py` | Gemma4 head-dim-512 → ALU (when engine supports D512) |
| `vllm/model_executor/layers/utils.py` | gfx1030 skinny-GEMM routing + M=1 path |
| `vllm/model_executor/warmup/kernel_warmup.py` | skip bf16x3 JIT warmup on non-CUDA |
| `vllm/triton_utils/force_first_config.py` | validated GDN static winners (env-gated) |
| `vllm/v1/executor/multiproc_executor.py` | explicit TunableOp DB loader |
| `vllm/distributed/device_communicators/cuda_communicator.py` | gfx1030 allreduce router |
| `vllm/distributed/device_communicators/custom_all_reduce.py` | gfx1030 graph-staging fix |
| `vllm/distributed/device_communicators/quick_all_reduce.py` | external `.so` loader for quick reduce |

### Not in this repo

- Compiled `.so` files — **you build them** (that is the point). Build is
  self-contained: the ROCm compiler comes from the pip wheels, not the host.
- Internal tuning DBs, benchmarks, lab scripts, archive dirs.

## The one flag that matters

```
VLLM_ROCM_GFX1030_WVSPLITK=1
```

- **At build time** → compiles the wvSplitK decode GEMM into `_rocm_C`.
- **At serve time** → enables it (M=2..5 decode batches).
- Off = stock vLLM GEMM path, clean error if mismatched. Never a GPU hang.

Other env vars (all default OFF, see `vllm/envs.py`):
`VLLM_ROCM_GFX1030_WIDE_UA`, `VLLM_ROCM_GFX1030_SPLITKV`,
`VLLM_ROCM_GFX1030_GDN_STATIC_WINNERS`.

## What's in this folder

```
.                    vLLM 0.30.0 source (+ the 17 files above)
alu-attention/       ALU Attention engine source (package/ + kernels/), v0.6.3
build-venv.sh        one-shot build: venv + alu_attn + vllm (with PASS/FAIL gates)
serve.sh             generic serve launcher (self-locating, no hardcoded paths)
requirements/        upstream requirement files (unmodified)
```

## Build & run

- **Build:** `./build-venv.sh` — see `BUILD.md` for what it does, step by step.
- **Serve:** `./serve.sh /path/to/model` — see the script header for flags.
- **Upstream PRs:** see `UPSTREAM-PR.md` for how the 17 files split into
  PRs against vllm-project/vllm.

## Requirements (exact)

- Linux x86_64, gfx1030 GPU(s) visible via `/dev/kfd`
- `uv` (or python 3.13), `cmake`, `ninja`, rustup with Rust **1.95**
- Internet access to `pypi.org` + `stable.repo.amd.com`
- ~20 GB disk for the venv, ~30-60 min for a first full build

## License

vLLM: Apache-2.0 (upstream). ALU Attention: see `alu-attention/LICENSE`.
