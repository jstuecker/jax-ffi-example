from pathlib import Path
import os

from jax_ffi_gen import parse, generator as gen

HERE = Path(__file__).resolve().parent

kernels = parse.get_functions_from_file(
    str(HERE / "example.cuh"),
    names=["Multiply", "DirectSummationForce"],
)

gen.generate_ffi_module_file(
    str(HERE / "generated/ffi_example.cu"), kernels,
    includes=["../example.cuh"]
)