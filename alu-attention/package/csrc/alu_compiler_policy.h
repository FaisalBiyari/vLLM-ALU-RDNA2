// ALU Attention Linux compiler contract. Included before project and PyTorch code.
#pragma once
#define ALU_COMPILER_POLICY_ID "alu-linux-cxx20-precise-v2"

#if !defined(__linux__) && !(defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__)
# error "ALU Attention requires Linux."
#endif
#if !defined(__cplusplus) || __cplusplus < 202002L
# error "ALU Attention requires -std=c++20."
#endif
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
# error "ALU Attention requires NaN/infinity semantics. Use -fno-fast-math -fno-finite-math-only."
#endif
#if defined(__clang__)
# if __has_warning("-Wnan-infinity-disabled")
#  pragma clang diagnostic error "-Wnan-infinity-disabled"
# endif
# if __has_warning("-Wc++20-extensions")
#  pragma clang diagnostic error "-Wc++20-extensions"
# endif
#endif
