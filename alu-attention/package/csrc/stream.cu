#include "alu_compiler_policy.h" // ALU_COMPILER_HARDENING_POLICY
// stream.cu — obtain the current PyTorch stream and device CU count in a translation unit
// compiled by the ROCm/HIP toolchain. PyTorch's ROCm build intentionally exposes the CUDA-spelling
// compatibility API; using it keeps stream selection identical to the caller and avoids default-stream
// races or device-wide synchronization.
#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

extern "C" void* alu_current_stream(long long dev){
    return (void*)at::cuda::getCurrentCUDAStream((c10::DeviceIndex)dev).stream();
}

extern "C" int alu_device_cu_count(long long dev){
    hipDeviceProp_t prop{};
    if(hipGetDeviceProperties(&prop, (int)dev) != hipSuccess) return 0;
    return prop.multiProcessorCount;
}

extern "C" int alu_last_launch_error(){
    return (int)hipGetLastError();
}
