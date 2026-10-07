#!/usr/bin/env bash
###############################################################################
# build-venv.sh  —  build the vLLM 0.30.0 + ALU Attention (gfx1030/RDNA2) stack
#
#   Reproduces, from this folder alone, the exact source tree + venv that is
#   running in the home lab (4x W6800X gfx1030, TP4).
#
#   Three phases, each ending in an explicit PASS/FAIL gate:
#     [1/3] base venv  : ROCm 10 userspace + torch 2.13 + triton (AMD indexes)
#     [1b]  amdsmi     : GPU enumeration (bundled in the rocm-sdk-core wheel)
#     [2/3] alu_attn   : the ALU Attention HIP-kernel engine (embedded source)
#     [3/3] vllm       : editable install that COMPILES _rocm_C with the
#                        gfx1030 wvSplitK decode kernel (the 0.6.3 headline)
#
#   Run:   ./build-venv.sh [venv-dir] [extra flags]     (default venv-dir: ./venv)
#   Flags: --force   wipe a non-empty target venv first
#          --keep    keep a partial venv if a phase fails (for debugging)
#   Needs: Linux x86_64, uv (or python3.13), cmake, ninja, a Rust toolchain
#          (rustup, channel 1.95), and an internet connection.
#
#   Every block below is copy/paste-able into a plain shell. Nothing prompts.
###############################################################################
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR=""
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1;;
    --keep)  ;;  # reserved
    -*)      echo "unknown flag: $arg (use --force to allow wiping a non-empty venv)"; exit 2;;
    *)       VENV_DIR="$arg";;
  esac
done
VENV_DIR="${VENV_DIR:-$HERE/venv}"
PYVER="3.13.15"

# --- AMD public wheel indexes (verified live 2026-10-06) --------------------
IDX_PYTORCH="https://stable.repo.amd.com/rocm/pytorch/whl-next/"
IDX_CORE="https://stable.repo.amd.com/rocm/core/whl-next/"

# --- version pins (byte-for-byte the lab venv) ------------------------------
Torch="torch==2.13.0+rocm10.0.0"
TorchVision="torchvision==0.28.0+rocm10.0.0"
TorchAudio="torchaudio==2.11.0.2+rocm10.0.0"
Triton="triton==3.8.0+git4cff872c.rocm10.0.0"
ROCmCore="rocm-sdk-core==10.0.0"
ROCmLibs="rocm-sdk-libraries==10.0.0"
ROCmDevel="rocm-sdk-devel==10.0.0"
ROCmDev1030="rocm-sdk-device-gfx1030==10.0.0"
ROCmBoot="rocm-bootstrap==0.1.0"
AmdTorchDev="amd-torch-device-gfx1030==2.13.0+rocm10.0.0"
AmdTvDev="amd-torchvision-device-gfx1030==0.28.0+rocm10.0.0"
Transformers="transformers==5.17.0"

# Optional: point at a pre-cloned ROCm triton_kernels to skip the ~15 min
# CMake FetchContent clone on the first build. Leave unset for a clean build.
: "${TRITON_KERNELS_SRC_DIR:=}"

fail() { echo; echo "FAIL: $*"; echo "  (see log above)"; exit 1; }
pass() { echo "PASS: $*"; }

echo "==================================================================="
echo " vLLM 0.30.0 + ALU Attention  gfx1030  —  build-venv"
echo "   venv dir : $VENV_DIR"
echo "   python   : $PYVER"
echo "==================================================================="

# ---------------------------------------------------------------------------
echo; echo "[0] prerequisites"
command -v cmake >/dev/null || fail "cmake not found (apt install cmake / brew install cmake)"
command -v ninja >/dev/null || fail "ninja not found (apt install ninja-build)"
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
[ -x "$UV" ] || fail "uv not found (curl -LsSf https://astral.sh/uv/install.sh | sh)"
command -v rustc >/dev/null 2>&1 || { source "$HOME/.cargo/env" 2>/dev/null; command -v rustc >/dev/null || fail "Rust toolchain not found (rustup)"; }
rustc --version
pass "cmake $(cmake --version | head -1 | grep -oE '[0-9.]+'), ninja $(ninja --version), rustc $(rustc --version | grep -oE 'rustc [0-9.]+' | awk '{print $2}')"

# ---------------------------------------------------------------------------
echo; echo "[1/3] base venv  (ROCm 10 userspace + torch 2.13 + triton)"
if [ -e "$VENV_DIR" ]; then
  if [ "$FORCE" = "1" ]; then
    ARCHIVE="${VENV_DIR}.old-$(date +%Y%m%d-%H%M%S)"
    echo "  moving existing $VENV_DIR -> $ARCHIVE (nothing deleted)"
    mv "$VENV_DIR" "$ARCHIVE"
  else
    echo
    echo "FAIL: $VENV_DIR already exists and is not empty."
    echo "  Re-run with --force to archive it to a timestamped sibling dir"
    echo "  (it is MOVED, never deleted), or point the script at a fresh dir."
    exit 1
  fi
fi
"$UV" venv --seed --python "$PYVER" "$VENV_DIR" || fail "uv venv failed"
PIP="$VENV_DIR/bin/pip"
PY="$VENV_DIR/bin/python"
SP="$VENV_DIR/lib/python3.13/site-packages"

# ~12 GB of wheels land in the venv; refuse to start mid-quota
KAVAIL=$(df -Pk "$(dirname "$VENV_DIR")" | awk 'NR==2 {print $4}')
[ "$KAVAIL" -gt 31457280 ] || fail "need >30 GB free on $(df -h "$(dirname "$VENV_DIR")" | tail -1 | awk '{print $6}') (have $((KAVAIL/1024/1024)) GB) — point the venv at a bigger volume"

echo "  installing torch/triton/rocm-sdk from AMD indexes (large download ~10 GB)..."
"$PIP" install --upgrade pip setuptools wheel >/dev/null
"$PIP" install \
  "$Torch" "$TorchVision" "$TorchAudio" "$Triton" \
  "$ROCmCore" "$ROCmLibs" "$ROCmDevel" "$ROCmDev1030" "$ROCmBoot" \
  "$AmdTorchDev" "$AmdTvDev" "$Transformers" \
  --extra-index-url "$IDX_PYTORCH" \
  --extra-index-url "$IDX_CORE" \
  || fail "torch/rocm install failed (check index URLs + network)"

"$PY" - <<'PY' || fail "torch/rocm import check"
import torch
assert torch.version.hip is not None, "torch is not a ROCm build"
print("  torch", torch.__version__, "hip", torch.version.hip)
import rocm_sdk_core, rocm_sdk_devel, rocm_sdk_libraries
from importlib.metadata import version as _v
# rocm-sdk-device-gfx1030 is a files-only dist (drops .so into _rocm_sdk_libraries)
assert _v("rocm-sdk-device-gfx1030") == "10.0.0", _v("rocm-sdk-device-gfx1030")
print("  rocm-sdk userspace present (core/devel/libraries/device-gfx1030)")
PY
pass "base venv: torch $( "$PY" -c 'import torch;print(torch.__version__)' )"

# ---------------------------------------------------------------------------
echo; echo "  [1c] rocm-sdk init  (expand the ROCm devel toolchain into the venv)"
# pip only drops a compressed _devel.tar; this step expands the real
# headers + hipcc into site-packages/_rocm_sdk_devel/ and hardlinks the
# per-arch (gfx1030) device files into that tree.
"$VENV_DIR/bin/rocm-sdk" init || fail "rocm-sdk init failed"
[ -f "$SP/_rocm_sdk_devel/include/hip/hip_runtime.h" ] || fail "devel tree missing hip_runtime.h (init did not expand)"
[ -x "$SP/_rocm_sdk_devel/bin/hipcc" ] || fail "devel tree missing bin/hipcc"
find "$SP/_rocm_sdk_devel" -name "*gfx1030*" >/dev/null 2>&1 || fail "devel tree missing gfx1030 device files (rocm-sdk-device-gfx1030 not linked)"
pass "rocm-sdk devel toolchain expanded (headers + hipcc + gfx1030 device files)"

# ---------------------------------------------------------------------------
echo; echo "[1b] amdsmi  (GPU enumeration — required for vLLM to pick the ROCm platform)"
# The pip rocm-sdk-core wheel bundles BOTH the amdsmi python package
# (_rocm_sdk_core/share/amd_smi/amdsmi/) AND the native lib (_rocm_sdk_core/lib/
# libamd_smi.so.27). Install = copy the package into site-packages and put the
# .so where amdsmi_wrapper's loader finds it. No compiler, no system ROCm.
CORE="$SP/_rocm_sdk_core"
AMDSMI_PKG_SRC="$CORE/share/amd_smi/amdsmi"
AMDSMI_LIB="$CORE/lib/libamd_smi.so.27"
[ -f "$AMDSMI_PKG_SRC/__init__.py" ] || fail "rocm-sdk-core wheel missing amdsmi package (unexpected)"
[ -f "$AMDSMI_LIB" ] || fail "rocm-sdk-core wheel missing libamd_smi.so.27 (unexpected)"
# idempotent re-run: archive any prior amdsmi copy this script made (never delete)
[ -d "$SP/amdsmi" ] && mv "$SP/amdsmi" "$SP/amdsmi.prev-$(date +%H%M%S)" 2>/dev/null || true
cp -a "$AMDSMI_PKG_SRC" "$SP/amdsmi"
mkdir -p "$SP/lib"
cp -a "$AMDSMI_LIB" "$SP/lib/libamd_smi.so.27"          # loader path 1: site-packages/lib/
cp -a "$AMDSMI_LIB" "$SP/amdsmi/libamd_smi.so"            # loader path 2: package dir (belt+braces)
"$PY" - <<'PY' || fail "amdsmi import check"
import amdsmi
amdsmi.amdsmi_init()
n = len(amdsmi.amdsmi_get_processor_handles())
amdsmi.amdsmi_shut_down()
print("  amdsmi", amdsmi.__version__, "| GPU handles visible:", n)
assert n > 0, "amdsmi sees 0 GPUs — is this a gfx1030 node with /dev/kfd?"
PY
pass "amdsmi (ROCm platform detection)"

# ---------------------------------------------------------------------------
echo; echo "[2/3] alu_attn  (ALU Attention HIP-kernel engine, from embedded source)"
ALU_PKG="$HERE/alu-attention/package"
[ -f "$ALU_PKG/setup.py" ] || fail "embedded ALU source missing: $ALU_PKG"
# --no-build-isolation is MANDATORY: ALU's pyproject requires 'torch' at build
# time; isolated build would pull stock CUDA torch from PyPI and setup.py's
# ROCm check would reject it.
env PYTORCH_ROCM_ARCH=gfx1030 HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    ROCM_HOME="$SP/_rocm_sdk_devel" \
    LD_LIBRARY_PATH="$SP/_rocm_sdk_devel/lib:$SP/_rocm_sdk_core/lib" \
    HIP_VISIBLE_DEVICES= \
  "$PIP" install "$ALU_PKG" --no-build-isolation || fail "alu_attn build failed (needs the venv's ROCm toolchain; see log)"
"$PY" - <<'PY' || fail "alu_attn import check"
import alu_attn
assert alu_attn.__version__ == "0.6.3", alu_attn.__version__
from alu_attn.concurrency import SUPPORTED_HEAD_DIMS
assert 512 in SUPPORTED_HEAD_DIMS, "D512 (Gemma4) support missing"
print("  alu_attn", alu_attn.__version__, "head_dims", sorted(SUPPORTED_HEAD_DIMS))
PY
pass "alu_attn 0.6.3 (D512 + D256)"

# ---------------------------------------------------------------------------
echo; echo "[3/3] vllm  (editable install — compiles _rocm_C incl. wvSplitK)"
[ -x "$SP/_rocm_sdk_devel/bin/hipcc" ] || fail "venv missing the ROCm toolchain — re-run (the rocm-sdk init step must have passed)"

# build deps vLLM's setup.py needs at configure time
"$PIP" install "setuptools>=77.0.3,<80.0.0" "setuptools-scm>=8" "setuptools-rust>=1.9.0" \
  "packaging" cmake ninja || fail "vLLM build-deps install failed"

echo "  compiling extensions (Rust frontend + _rocm_C wvSplitK) ... this is the long step (~30-60 min)"
source "$HOME/.cargo/env" 2>/dev/null || true
( cd "$HERE" && \
  VLLM_TARGET_DEVICE=rocm \
  PYTORCH_ROCM_ARCH=gfx1030 \
  HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  ROCM_HOME="$SP/_rocm_sdk_devel" \
  LD_LIBRARY_PATH="$SP/_rocm_sdk_devel/lib:$SP/_rocm_sdk_core/lib" \
  VLLM_ROCM_GFX1030_WVSPLITK=1 \
  VCS_VERSIONING_PRETEND_VERSION=0.30.0 \
  HIP_VISIBLE_DEVICES="" \
  ${TRITON_KERNELS_SRC_DIR:+TRITON_KERNELS_SRC_DIR="$TRITON_KERNELS_SRC_DIR"} \
  "$PY" -m pip install -e . --no-build-isolation ) \
  || fail "vLLM editable build failed (full log above; usually a compiler or network fetch)"

# --- verify the wvSplitK kernels actually compiled in ------------------------
# NOTE: use `grep -c` (not `grep -q`) — under `set -o pipefail`, `grep -q`
# exits on the first match and SIGPIPEs `nm` (exit 141), which the pipeline
# reports as a FAILURE even when the symbols are present.
SO="$(find "$HERE/vllm" -name '_rocm_C.abi3.so' | head -1)"
[ -n "$SO" ] || fail "built _rocm_C.abi3.so not found in $HERE/vllm"
WSYMS="$(nm -C "$SO" 2>/dev/null | grep -ciE 'wvSplitK|wave32')"
if [ "$WSYMS" -gt 0 ]; then
  pass "wvSplitK/wave32 kernels present in _rocm_C.abi3.so ($WSYMS symbols)"
else
  fail "wvSplitK symbols NOT in _rocm_C.abi3.so (build ran without VLLM_ROCM_GFX1030_WVSPLITK?)"
fi

"$PY" - <<'PY' || fail "vllm import check"
import vllm
from vllm.platforms.rocm import on_gfx1030
print("  vllm", vllm.__version__)
print("  on_gfx1030() ->", on_gfx1030(), "(True on this gfx1030 node)")
PY

echo
echo "==================================================================="
echo " BUILD COMPLETE"
echo "   venv : $VENV_DIR"
echo "   tree : $HERE  (compiled .so in place)"
echo "   run  : cd $HERE && VLLM_ROCM_GFX1030_WVSPLITK=1 \\"
echo "             $VENV_DIR/bin/vllm serve <model> --tensor-parallel-size 4 ..."
echo "   (or:  ./serve.sh <model-path>  — see serve.sh)"
echo "==================================================================="
