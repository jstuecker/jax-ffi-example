import numpy as np
import jax
import jax.numpy as jnp
from jax_ffi_example_cuda import ffi_example

# ------------------------------------------------------------------------------------------------ #
#                                             FFI Calls                                            #
# ------------------------------------------------------------------------------------------------ #

jax.ffi.register_ffi_target("Multiply", ffi_example.Multiply(), platform="CUDA")
jax.ffi.register_ffi_target("SimpleDirectSummationForce", ffi_example.SimpleDirectSummationForce(), platform="CUDA")
jax.ffi.register_ffi_target("DirectSummationForce", ffi_example.DirectSummationForce(), platform="CUDA")

def div_ceil(a, b):
    return (a + b - 1) // b

def multiply(a: jax.Array, b: jax.Array, block_size=64):
    assert a.shape == b.shape
    assert a.dtype == b.dtype == jnp.float32

    outputs = (jax.ShapeDtypeStruct(a.shape, a.dtype),)

    grid_size = div_ceil(a.size, block_size)
    
    out = jax.ffi.ffi_call("Multiply", outputs)(
        a, b, 
        num=np.uint64(a.size),
        block_size=np.uint64(block_size), 
        grid_size=np.uint64(grid_size),
    )[0]

    return out
multiply.jit = jax.jit(multiply, static_argnames=("block_size",))

def simple_cuda_force(
        pos: jax.Array, mass: jax.Array, softening: float, block_size=64, gap=0
    ) -> jax.Array:
    assert pos.dtype == mass.dtype == jnp.float32
    assert (pos.shape[:-1] == mass.shape) and (pos.shape[-1] == 3)

    posmass = jnp.concatenate([pos, mass[..., None]], axis=-1)

    outputs = (jax.ShapeDtypeStruct(pos.shape, mass.dtype),)

    return jax.ffi.ffi_call("SimpleDirectSummationForce", outputs)(
        posmass, epsilon=np.float32(softening), block_size=np.uint64(block_size), gap=np.int32(gap)
    )[0]
simple_cuda_force.jit = jax.jit(simple_cuda_force, static_argnames=("softening", "block_size", "gap"))

def cuda_force(
        pos: jax.Array, mass: jax.Array, softening: float, block_size=64
    ) -> jax.Array:
    assert pos.dtype == mass.dtype == jnp.float32
    assert (pos.shape[:-1] == mass.shape) and (pos.shape[-1] == 3)

    posmass = jnp.concatenate([pos, mass[..., None]], axis=-1)

    outputs = (jax.ShapeDtypeStruct(pos.shape, mass.dtype),)

    return jax.ffi.ffi_call("DirectSummationForce", outputs)(
        posmass, epsilon=np.float32(softening), block_size=np.uint64(block_size),
    )[0]
cuda_force.jit = jax.jit(cuda_force, static_argnames=("softening", "block_size"))

# ------------------------------------------------------------------------------------------------ #
#                                        Jax reference Code                                        #
# ------------------------------------------------------------------------------------------------ #

def jax_force(
        pos: jax.Array, mass: jax.Array, n2lim: int = 1e8, softening: float = 1e-5
    ) -> jax.Array:
    N = pos.shape[0]

    nmax = int(np.ceil(n2lim / len(pos)))
    nev = int(np.ceil(pos.shape[0] / nmax))

    def force_over_range(i1, i2):
        xi = pos[jnp.arange(nmax, dtype=jnp.int32) + i1]
        dx = pos - xi[:,None]
        r_ij2 = dx[...,0]**2 + dx[...,1]**2 + dx[...,2]**2
        distinv = 1./jnp.sqrt(r_ij2 + softening**2)

        fij = dx * (mass * distinv**3)[...,None]

        return jnp.sum(fij, axis=1)

    def handle_interval(_, i):
        return None, force_over_range(nmax * i, nmax * (i + 1))

    _, forces = jax.lax.scan(handle_interval, None, jnp.arange(nev, dtype=jnp.int32))

    return jnp.concatenate(forces)[0:N]
jax_force.jit = jax.jit(jax_force, static_argnames=("n2lim",))
