"""ALU Attention compiler policy. Loaded by setup.py without importing the GPU package.

No finite-only arithmetic, unsafe reassociation, or blanket fast-math is allowed.
Explicit HIP math intrinsics already present in the kernels are not rewritten.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import shlex
from typing import Mapping, Sequence

POLICY_ID = "alu-linux-cxx20-precise-v2"
UNSAFE = frozenset({
    "-ffast-math", "-Ofast", "-ffinite-math-only", "-fno-honor-nans",
    "-fno-honor-infinities", "-funsafe-math-optimizations", "-fassociative-math",
    "-freciprocal-math", "-fno-signed-zeros", "-fapprox-func", "-fcx-limited-range",
    "--use_fast_math", "-use_fast_math", "-cl-fast-relaxed-math",
    "-cl-finite-math-only", "-ffp-contract=fast", "-ffp-contract=fast-honor-pragmas",
    "-ffp-model=fast", "-ffp-model=aggressive", "-fdenormal-fp-math=positive-zero",
    "-fdenormal-fp-math=preserve-sign", "-fdenormal-fp-math-f32=positive-zero",
    "-fdenormal-fp-math-f32=preserve-sign", "-mllvm", "--ffast-math",
})
ENV_FLAGS = (
    "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "HIPFLAGS", "HIPCC_FLAGS",
    "HIPCC_COMPILE_FLAGS_APPEND", "HIPCC_LINK_FLAGS_APPEND", "NVCC_PREPEND_FLAGS",
    "NVCC_APPEND_FLAGS", "AMD_COMGR_FLAGS",
)
PRECISE_FLAGS = ("-O3", "-std=c++20", "-fno-fast-math", "-fno-finite-math-only",
                 "-ffp-contract=on")


def validate_environment(env: Mapping[str, str] | None = None) -> None:
    env = os.environ if env is None else env
    for name in ENV_FLAGS:
        value = env.get(name, "")
        # shlex is only used to inspect flags, never to execute a command.
        try:
            parts = shlex.split(value, posix=True)
        except ValueError as exc:
            raise RuntimeError(f"ALU: malformed {name}: {exc}") from exc
        for item in parts:
            standard_conflict = (item.startswith("-std=")
                                 and item not in ("-std=c++20", "-std=gnu++20"))
            if (item in UNSAFE or standard_conflict or item.startswith("-Wno-")
                    or item == "-w" or item.startswith("-Wno-error")):
                raise RuntimeError(f"ALU: conflicting {name} flag {item!r}; remove it before building")
    if env.get("CCC_OVERRIDE_OPTIONS", "").strip():
        raise RuntimeError("ALU: unset CCC_OVERRIDE_OPTIONS; it can rewrite validated compiler flags")
    if env.get("ALU_WARNINGS_AS_ERRORS", "0") not in ("0", "1"):
        raise RuntimeError("ALU_WARNINGS_AS_ERRORS must be 0 or 1")


def _clean(flags: Sequence[str]) -> list[str]:
    if isinstance(flags, (str, bytes)):
        raise TypeError("ALU: expected a list of compiler arguments, not one string")
    result = []
    for flag in flags:
        if not isinstance(flag, str):
            raise TypeError(f"ALU: compiler flag must be a string, got {type(flag).__name__}")
        if flag in UNSAFE:
            raise RuntimeError(f"ALU: unsafe explicit compiler flag {flag!r}; update the build entry point")
        if flag.startswith(("-std=", "-ffp-contract=", "-ffp-model=")):
            continue
        if flag in ("-fno-fast-math", "-fno-finite-math-only", "-O3", "-O2", "-O0", "-O1"):
            continue
        if flag.startswith("-Wno-") or flag == "-w":
            raise RuntimeError(f"ALU: warning suppression is not permitted in the supported build: {flag}")
        result.append(flag)
    return result


def harden_extension_args(existing: Mapping[str, Sequence[str]], setup_file: str) -> dict[str, list[str]]:
    """Preserve include/ABI/architecture arguments and append an explicit math policy."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError("ALU Attention supports Linux only")
    validate_environment()
    if not isinstance(existing, Mapping) or set(existing) - {"cxx", "nvcc"}:
        raise RuntimeError("ALU: expected extra_compile_args with cxx/nvcc keys only")
    header = Path(setup_file).resolve().parent / "csrc" / "alu_compiler_policy.h"
    if not header.is_file():
        raise RuntimeError(f"ALU: compiler policy header is missing: {header}")
    cxx = _clean(existing.get("cxx", []))
    hip = _clean(existing.get("nvcc", []))
    cxx += [*PRECISE_FLAGS, "-Werror=unknown-pragmas", "-Werror=c++20-extensions",
            "-include", str(header)]
    hip += [*PRECISE_FLAGS, "-Werror=unknown-pragmas", "-Werror=c++20-extensions",
            "-include", str(header)]
    if os.environ.get("ALU_WARNINGS_AS_ERRORS") == "1":
        cxx += ["-Werror"]
        hip += ["-Werror"]
    return {"cxx": cxx, "nvcc": hip}
