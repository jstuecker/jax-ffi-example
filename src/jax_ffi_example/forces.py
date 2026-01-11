import numpy as np
import jax
import jax.numpy as jnp
from jax_ffi_example_cuda import ffi_example

jax.ffi.register_ffi_target("Multiply", ffi_example.Multiply(), platform="CUDA")

def div_ceil(a, b):
    return (a + b - 1) // b

def multiply(a: jax.Array, b: jax.Array, block_size=64):
    assert a.shape == b.shape
    assert a.dtype == b.dtype == jnp.float32

    outputs = (jax.ShapeDtypeStruct(a.shape, a.dtype),)

    grid_size = div_ceil(a.size, block_size)
    
    out = jax.ffi.ffi_call("Multiply", outputs)(
        a, b, 
        num=np.int32(a.size),
        block_size=np.uint64(block_size), 
        grid_size=np.uint64(grid_size),
    )[0]

    return out