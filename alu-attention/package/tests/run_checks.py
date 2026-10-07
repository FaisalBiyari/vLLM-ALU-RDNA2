"""run_checks.py — one entry point for all package checks, without requiring pytest.

Runs the standalone gates in sequence and returns a nonzero exit code if any gate fails:
  1. lab/parity_test.py    — previously supported behavior (all head_dim values, masks, GQA, varlen,
                             paged cache, backward, determinism, install_as_flash_attn);
  2. lab/parity_new_api.py — PLAN.md section 3 additions (attention_chunk, qv, seqused_*,
                             return_softmax_lse, combine, bert_padding, explicit FP8 rejection);
  3. lab/tp_attention_canary.py — TP1/2/4/8 rank-local prefill/decode plus interleaved paged KV;
  4. package/tests/test_alu_attn.py under pytest, if pytest is available.

An older version imported `_ref_attention` and `_HAS_C` from flash_attn_interface even though those
names had already been removed, so it failed during import and checked nothing. The reference logic
already lives in the two lab gates and is not duplicated here.

Run: python package/tests/run_checks.py
"""
import os
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GATES = [
    ("parity: previously supported behavior", os.path.join(ROOT, "lab", "parity_test.py")),
    ("parity: newly covered API surface",       os.path.join(ROOT, "lab", "parity_new_api.py")),
    ("attention: TP1/2/4/8 + interleaved vLLM KV", os.path.join(ROOT, "lab", "tp_attention_canary.py")),
]


def main():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    failed = []
    for name, path in GATES:
        print(f"\n{'=' * 70}\n== {name}\n{'=' * 70}")
        r = subprocess.run([sys.executable, path], cwd=ROOT, env=env)
        if r.returncode != 0:
            failed.append(name)

    print(f"\n{'=' * 70}\n== pytest (if installed)\n{'=' * 70}")
    have_pytest = subprocess.run([sys.executable, "-c", "import pytest"],
                                 capture_output=True).returncode == 0
    if not have_pytest:
        # Missing pytest is not a failed gate; the standalone checks above already ran.
        print("pytest is not installed in this environment; skipping "
              "(coverage is provided by lab/parity_test.py + lab/parity_new_api.py)")
    else:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q",
                            os.path.join(ROOT, "package", "tests", "test_alu_attn.py")],
                           cwd=ROOT, env=env)
        if r.returncode not in (0, 5):   # 5 = no tests collected
            failed.append("pytest test_alu_attn.py")

    print()
    if failed:
        print(f"*** FAILED GATES: {', '.join(failed)}")
        return 1
    print("ALL PACKAGE GATES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
