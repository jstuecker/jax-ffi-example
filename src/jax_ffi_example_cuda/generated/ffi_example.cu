// This file was automatically generated
// You can modify it, but I recommend automatically regenerating this code whenever you adapt 
// one of the kernels. The FFI Bindings are very tedious in jax and they involve a lot of 
// boilerplate code that is easy to mess up.

#include <map>
#include <tuple>
#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"

// A wrapper to encapsulate an FFI call
template <typename T>
nanobind::capsule EncapsulateFfiCall(T *fn) {
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be and XLA FFI handler");
    return nanobind::capsule(reinterpret_cast<void *>(fn));
}
#include "../example.cuh"

namespace nb = nanobind;
namespace ffi = xla::ffi;

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: Multiply                                  */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error MultiplyFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer x,
    ffi::AnyBuffer y,
    ffi::Result<ffi::AnyBuffer> output,
    size_t num,
    size_t grid_size,
    size_t block_size
) {
    dim3 blockDim(block_size);
    dim3 gridDim(grid_size);
    size_t smem = 0;
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    float* x_val = reinterpret_cast<float*>(x.untyped_data());
    float* y_val = reinterpret_cast<float*>(y.untyped_data());
    float* output_val = reinterpret_cast<float*>(output->untyped_data());

    void* args[] = {
        &x_val,
        &y_val,
        &output_val,
        &num
    };
    cudaLaunchKernel((const void*)Multiply, gridDim, blockDim, args, smem, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MultiplyFFI, MultiplyFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // x
        .Arg<ffi::AnyBuffer>() // y
        .Ret<ffi::AnyBuffer>() // output
        .Attr<size_t>("num")
        .Attr<size_t>("grid_size")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: SimpleDirectSummationForce                */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error SimpleDirectSummationForceFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer xm,
    ffi::Result<ffi::AnyBuffer> force_out,
    float epsilon,
    int gap,
    size_t block_size
) {
    int n = xm.element_count()/4;
    dim3 blockDim(block_size);
    dim3 gridDim((n + blockDim.x - 1)/blockDim.x);
    size_t smem = blockDim.x * sizeof(float3);
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    PosMass* xm_val = reinterpret_cast<PosMass*>(xm.untyped_data());
    float3* force_out_val = reinterpret_cast<float3*>(force_out->untyped_data());

    void* args[] = {
        &xm_val,
        &force_out_val,
        &n,
        &epsilon,
        &gap
    };
    cudaLaunchKernel((const void*)SimpleDirectSummationForce, gridDim, blockDim, args, smem, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SimpleDirectSummationForceFFI, SimpleDirectSummationForceFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // xm
        .Ret<ffi::AnyBuffer>() // force_out
        .Attr<float>("epsilon")
        .Attr<int>("gap")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                             FFI call to CUDA kernel: DirectSummationForce                      */
/* ---------------------------------------------------------------------------------------------- */

ffi::Error DirectSummationForceFFIHost(
    cudaStream_t stream,
    ffi::AnyBuffer xm,
    ffi::Result<ffi::AnyBuffer> force_out,
    float epsilon,
    size_t block_size
) {
    int n = xm.element_count()/4;
    dim3 blockDim(block_size);
    dim3 gridDim((n + blockDim.x - 1)/blockDim.x);
    size_t smem = blockDim.x * sizeof(float3);
    
    // Build a bundled argument list for cudaLaunchKernel
    // For pointers we need to create a pointer to the pointer
    PosMass* xm_val = reinterpret_cast<PosMass*>(xm.untyped_data());
    float3* force_out_val = reinterpret_cast<float3*>(force_out->untyped_data());

    void* args[] = {
        &xm_val,
        &force_out_val,
        &n,
        &epsilon
    };
    cudaLaunchKernel((const void*)DirectSummationForce, gridDim, blockDim, args, smem, stream);

    cudaError_t last_error = cudaGetLastError();
    if (last_error != cudaSuccess) {
        return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
    }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    DirectSummationForceFFI, DirectSummationForceFFIHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>() // xm
        .Ret<ffi::AnyBuffer>() // force_out
        .Attr<float>("epsilon")
        .Attr<size_t>("block_size"),
    {xla::ffi::Traits::kCmdBufferCompatible}
);

/* ---------------------------------------------------------------------------------------------- */
/*                               Module declaration through nanobind                              */
/* ---------------------------------------------------------------------------------------------- */

NB_MODULE(ffi_example, m) {
    m.def("Multiply", []() { return EncapsulateFfiCall(&MultiplyFFI); });
    m.def("SimpleDirectSummationForce", []() { return EncapsulateFfiCall(&SimpleDirectSummationForceFFI); });
    m.def("DirectSummationForce", []() { return EncapsulateFfiCall(&DirectSummationForceFFI); });
}