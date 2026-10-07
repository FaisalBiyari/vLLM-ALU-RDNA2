#include "alu_compiler_policy.h" // ALU_COMPILER_HARDENING_POLICY
// HIP translation unit for the native PyTorch extension; no standalone main.
#include <torch/extension.h>
#include "../../kernels/fa_decode.hip"
