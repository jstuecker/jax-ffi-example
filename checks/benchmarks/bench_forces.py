from jax_ffi_example import forces
import jax
import jax.numpy as jnp
import pytest

@pytest.mark.parametrize("N", (int(1e3), int(3e3), int(1e4), int(3e4), int(1e5)))
def bench_direct_summation_force(jax_bench, N):
    pos = jax.random.normal(jax.random.key(0), (N,3))
    mass = jnp.ones_like(pos[...,0]) 

    jb = jax_bench(jit_rounds=10, jit_warmup=1)

    jb.measure(
        fn_jit=forces.direct_summation_force.jit, tag="CUDA",
        pos=pos, mass=mass, softening=0.1,
    )

    jb.measure(
        fn_jit=forces.direct_summation_force_jax.jit, tag="jax",
        pos=pos, mass=mass, softening=0.1,
    )