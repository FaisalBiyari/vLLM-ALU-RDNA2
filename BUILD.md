# BUILD

One command builds the whole stack:

```bash
cd <this-folder>
./build-venv.sh /path/to/your/venv        # e.g. ./venv
```

~30-60 min. Every phase ends in a `PASS:` line. Any `FAIL:` line = stop,
read the log above it.

## What the script does (3 phases + 1)

### [1/3] Base venv — torch + triton + ROCm userspace

```bash
uv venv --seed --python 3.13.15 ./venv
./venv/bin/pip install \
  torch==2.13.0+rocm10.0.0 \
  torchvision==0.28.0+rocm10.0.0 \
  torchaudio==2.11.0.2+rocm10.0.0 \
  triton==3.8.0+git4cff872c.rocm10.0.0 \
  rocm-sdk-core==10.0.0 rocm-sdk-libraries==10.0.0 \
  rocm-sdk-devel==10.0.0 rocm-sdk-device-gfx1030==10.0.0 \
  rocm-bootstrap==0.1.0 \
  amd-torch-device-gfx1030==2.13.0+rocm10.0.0 \
  amd-torchvision-device-gfx1030==0.28.0+rocm10.0.0 \
  transformers==5.17.0 \
  --extra-index-url https://stable.repo.amd.com/rocm/pytorch/whl-next/ \
  --extra-index-url https://stable.repo.amd.com/rocm/core/whl-next/
```

- The two AMD index URLs are **public and verified** (2026-10-06).
- The ROCm **compiler (hipcc) ships inside** `rocm-sdk-devel` — no system
  ROCm install needed. That is why the build is self-contained.
- `rocm-sdk-device-gfx1030` is a files-only wheel (drops `.so` into
  `_rocm_sdk_libraries/`), not an importable module.

### [1c] rocm-sdk init — **expand the ROCm devel toolchain** (mandatory)

```bash
./venv/bin/rocm-sdk init
```

- `pip install rocm-sdk-devel` only drops a compressed `_devel.tar`.
  `rocm-sdk init` expands it into `site-packages/_rocm_sdk_devel/` — the real
  headers (`include/hip/…`), the `bin/hipcc` compiler, and hard-links the
  **gfx1030** device files from `rocm-sdk-device-gfx1030` into that tree.
- Without this step there are no compiler headers and `hipcc` falls back to
  the wrong arch → the ALU/vLLM compiles fail. **The script runs it and gates
  on `include/hip/hip_runtime.h`, `bin/hipcc`, and a `*gfx1030*` file existing.**

### [1b] amdsmi — GPU enumeration

Copied from inside the already-installed `rocm-sdk-core` wheel:

```bash
SP=./venv/lib/python3.13/site-packages
cp -a $SP/_rocm_sdk_core/share/amd_smi/amdsmi  $SP/amdsmi
mkdir -p $SP/lib
cp -a $SP/_rocm_sdk_core/lib/libamd_smi.so.27  $SP/lib/libamd_smi.so.27
cp -a $SP/_rocm_sdk_core/lib/libamd_smi.so.27  $SP/amdsmi/libamd_smi.so
```

Why: vLLM detects the ROCm platform via `amdsmi` on bare Linux (the torch
fallback is WSL-only). Same version as the lab: `27.0.0+6b0e43f3`.

### [2/3] alu_attn — the ALU Attention engine

```bash
./venv/bin/pip install ./alu-attention/package --no-build-isolation
```

- **`--no-build-isolation` is mandatory.** The ALU `pyproject.toml` lists
  `torch` as a build dep; an isolated build would pull stock CUDA torch from
  PyPI and `setup.py`'s ROCm check would reject it. `--no-build-isolation`
  makes it use the venv's ROCm torch + hipcc.
- Built with `PYTORCH_ROCM_ARCH=gfx1030 HSA_OVERRIDE_GFX_VERSION=10.3.0
  ROCM_HOME=$SP/_rocm_sdk_devel` (the script sets all of these).
- Checks: `alu_attn.__version__ == 0.6.3` and `512 in SUPPORTED_HEAD_DIMS`
  (D512 = Gemma4 head-dim-512 support).

### [3/3] vllm — editable install, compiles `_rocm_C`

```bash
cd <this-folder>
VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=gfx1030 HSA_OVERRIDE_GFX_VERSION=10.3.0 \
ROCM_HOME=$SP/_rocm_sdk_devel \
LD_LIBRARY_PATH=$SP/_rocm_sdk_devel/lib:$SP/_rocm_sdk_core/lib \
VLLM_ROCM_GFX1030_WVSPLITK=1 VCS_VERSIONING_PRETEND_VERSION=0.30.0 \
./venv/bin/pip install -e . --no-build-isolation
```

- Builds: Rust frontend (`vllm-rs`, toolchain 1.95) + all C++/HIP extensions.
- `VLLM_ROCM_GFX1030_WVSPLITK=1` **at build time** = wvSplitK decode GEMM
  kernels compiled into `_rocm_C.abi3.so` (~1600+ `wvSplitK*` symbols).
- `VCS_VERSIONING_PRETEND_VERSION=0.30.0` pins this downstream distribution
  to its vLLM 0.30.0 base during local builds and keeps version detection
  deterministic.
- First run fetches `triton_kernels` (pinned commit) via CMake. To skip the
  ~15 min fetch, point `TRITON_KERNELS_SRC_DIR` at an existing checkout of
  the same tag.

## Verify the build (after the script finishes)

```bash
# 1. wvSplitK kernels are IN the built .so
nm -C vllm/_rocm_C.abi3.so | grep -ci "wvSplitK"    # > 0

# 2. import + arch detection
./venv/bin/python -c "
import vllm
from vllm.platforms.rocm import on_gfx1030
print(vllm.__version__, 'on_gfx1030 =', on_gfx1030())"   # True on this node
```

## Serve (proof ALU is actually selected)

```bash
VLLM_ROCM_GFX1030_WVSPLITK=1 ./venv/bin/vllm serve /path/to/model \
  --tensor-parallel-size 4 --dtype float16 --kv-cache-dtype float16 \
  --block-size 1024 --gpu-memory-utilization 0.92 --max-num-seqs 8 \
  --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 8192 \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --language-model-only --skip-mm-profiling --host 0.0.0.0 --port 8000
```

**The line that proves it** (per TP worker in the log):

```
Overriding with ALU_ATTN out of potential backends: [...]
```

A clean `/health` 200 is NOT proof — Triton returns 200 too. Grep the log
for the `Overriding with ALU_ATTN` line on every rank.

## Known-good pin table (what the venv must contain)

| package | version |
|---|---|
| torch | 2.13.0+rocm10.0.0 |
| triton | 3.8.0+git4cff872c.rocm10.0.0 |
| rocm-sdk-core / -devel / -libraries / -device-gfx1030 | 10.0.0 |
| amd-torch-device-gfx1030 | 2.13.0+rocm10.0.0 |
| transformers | 5.17.0 |
| amdsmi | 27.0.0+6b0e43f3 |
| alu_attn | 0.6.3 (built from `alu-attention/`) |
| vllm | 0.30.0 (editable, from this folder) |

## Troubleshooting (the 3 that actually happen)

- **`Failed to import from amdsmi`** → phase [1b] didn't run; re-run it.
- **`No available kernel` / attention crash at boot** → not gfx1030, or
  `rocm-sdk-device-gfx1030` wheel not installed.
- **wvSplitK "refuses gfx1030" at serve time** → the `.so` was built
  WITHOUT the flag. Rebuild phase [3/3] with `VLLM_ROCM_GFX1030_WVSPLITK=1`.
