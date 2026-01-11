from pathlib import Path
import os

from jax_ffi_gen import parse, generator as gen

HERE = Path(__file__).resolve().parent

kernels = parse.get_functions_from_file(
    str(HERE / "example.cuh"),
    names=["Multiply", "SimpleDirectSummationForce", "DirectSummationForce"],
)

kernels["SimpleDirectSummationForce"].par["n"].expression = "xm.element_count()/4"
kernels["SimpleDirectSummationForce"].grid_size_expression = "(n + blockDim.x - 1)/blockDim.x"
kernels["SimpleDirectSummationForce"].smem_size_expression = "blockDim.x * sizeof(float3)"

kernels["DirectSummationForce"].par["n"].expression = "xm.element_count()/4"
kernels["DirectSummationForce"].grid_size_expression = "(n + blockDim.x - 1)/blockDim.x"
kernels["DirectSummationForce"].smem_size_expression = "blockDim.x * sizeof(float3)"

gen.generate_ffi_module_file(
    str(HERE / "generated/ffi_example.cu"), kernels,
    includes=["../example.cuh"]
)