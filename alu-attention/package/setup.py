"""Build the Linux/ROCm extension from this complete source checkout."""
from pathlib import Path
import runpy
import sys

if not sys.platform.startswith("linux"):
    raise RuntimeError("ALU Attention supports Linux only")

from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if not torch.version.hip:
    raise RuntimeError("ALU Attention requires a ROCm-enabled PyTorch build")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
policy = runpy.run_path(str(HERE / "alu_build_policy.py"))
policy["validate_environment"]()
# PyTorch's ROCm extension helper hipifies this CUDA-spelling stream source at build time.
# Do not track generated stream.hip or a second generated host dispatcher in the repository.
sources = ["csrc/dispatch.cpp", "csrc/stream.cu", "csrc/_fwd_shim.cu",
           "csrc/_bwd_shim.cu", "csrc/_dec_shim.cu"]
if not (ROOT / "kernels/fa_decode.hip").is_file():
    raise RuntimeError("Build from the complete ALU source tree; the kernels directory is required")
dependencies = [str(p) for base in (ROOT / "kernels", HERE / "csrc")
                for p in sorted(base.iterdir()) if p.suffix in (".h", ".hpp", ".hip", ".inc")]

setup(
    name="alu_attn",
    version="0.6.3",
    description="Linux/ROCm FlashAttention HIP kernels for RDNA2/gfx1030",
    packages=find_packages(),
    ext_modules=[CUDAExtension(
        "alu_attn._C", sources, depends=dependencies,
        extra_compile_args=policy["harden_extension_args"]({"cxx": [], "nvcc": []}, __file__),
    )],
    cmdclass={"build_ext": BuildExtension},
)
