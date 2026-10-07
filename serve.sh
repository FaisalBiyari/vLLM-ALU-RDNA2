#!/usr/bin/env bash
###############################################################################
# serve.sh — serve a model on this vLLM + ALU Attention build (gfx1030)
#
# Usage:
#   ./serve.sh /path/to/model                     # 4 GPUs, port 8000
#   ./serve.sh /path/to/model --tp 1 --port 8100  # overrides
#   ./serve.sh /path/to/model -- --enforce-eager  # anything after -- goes
#                                                # verbatim to `vllm serve`
#
# Self-locating: uses ./venv (next to this script) and the source tree it
# lives in. Build first:  ./build-venv.sh
#
# Env overrides: TP, PORT, HOST, SERVED_NAME, MAXLEN, MAX_SEQS,
# VLLM_ROCM_GFX1030_WVSPLITK (default 1), VLLM_LOG_FILE
###############################################################################
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HERE/venv}"
PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "no venv at $VENV — run ./build-venv.sh first"; exit 1; }
SP="$VENV/lib/python3.13/site-packages"
[ -d "$SP/_rocm_sdk_devel" ] || { echo "venv missing _rocm_sdk_devel: $SP — rerun ./build-venv.sh"; exit 1; }

TP="${TP:-4}"; PORT="${PORT:-8000}"; HOST="${HOST:-0.0.0.0}"
MAX_SEQS="${MAX_SEQS:-8}"; SERVED_NAME="${SERVED_NAME:-model}"
WVSPLITK="${VLLM_ROCM_GFX1030_WVSPLITK:-1}"
LOG="${VLLM_LOG_FILE:-$HERE/serve.log}"
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --tp)       TP="$2"; shift 2;;
    --port)     PORT="$2"; shift 2;;
    --host)     HOST="$2"; shift 2;;
    --max-seqs) MAX_SEQS="$2"; shift 2;;
    --name)     SERVED_NAME="$2"; shift 2;;
    --)         shift; while [ $# -gt 0 ]; do EXTRA+=("$1"); shift; done;;
    *)          EXTRA+=("$1"); shift;;
  esac
done
[ ${#EXTRA[@]} -ge 1 ] || { echo "usage: $0 /path/to/model [--tp N] [--port N] [-- <extra vllm flags>]"; exit 2; }
MODEL="${EXTRA[0]}"

echo "serving: $MODEL"
echo "  tp=$TP port=$PORT host=$HOST wvsplitk=$WVSPLITK"
echo "  log  : $LOG"

# Proven gfx1030 env (matches the lab serving configuration 1:1)
exec env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="$HERE" \
  ROCM_HOME="$SP/_rocm_sdk_devel" \
  LD_LIBRARY_PATH="$SP/_rocm_sdk_devel/lib:$SP/_rocm_sdk_core/lib" \
  VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=gfx1030 HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  HIP_FORCE_DEV_KERNARG=1 TORCH_BLAS_PREFER_HIPBLASLT=0 \
  PYTORCH_TUNABLEOP_ENABLED=0 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_RECORD_UNTUNED=0 \
  PYTORCH_TUNABLEOP_EXPLICIT_LOAD=0 \
  VLLM_ROCM_USE_SKINNY_GEMM=1 VLLM_ROCM_GFX1030_WIDE_UA=1 VLLM_ROCM_GFX1030_SPLITKV=1 \
  VLLM_ROCM_GFX1030_WVSPLITK="$WVSPLITK" \
  VLLM_ROCM_GFX1030_GDN_STATIC_WINNERS=1 VLLM_TRITON_FORCE_FIRST_CONFIG=1 \
  VLLM_USE_NCCL_SYMM_MEM=0 VLLM_USE_DEEP_GEMM=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
  OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
  TORCHINDUCTOR_BUNDLE_TRITON_INTO_FX_GRAPH_CACHE=0 \
  "$PY" -m vllm.entrypoints.cli.main serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --block-size 1024 --dtype float16 --kv-cache-dtype float16 \
  --tensor-parallel-size "$TP" --gpu-memory-utilization 0.92 \
  --max-num-seqs "$MAX_SEQS" \
  --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 8192 \
  --trust-remote-code \
  --host "$HOST" --port "$PORT" \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --disable-custom-all-reduce \
  "${EXTRA[@]:1}" 2>&1 | tee "$LOG"
