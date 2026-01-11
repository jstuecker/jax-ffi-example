from jax_ffi_example import forces
import jax
import jax.numpy as jnp
import pytest

@pytest.mark.parametrize("gap", (0, 1, 2, 4, 8, 16))
def bench_gap(jax_bench, gap):
    pos = jax.random.normal(jax.random.key(0), (int(1e5),3))
    mass = jnp.ones_like(pos[...,0]) 

    jb = jax_bench(jit_rounds=10, jit_warmup=1)

    jb.measure(
        fn_jit=forces.simple_cuda_force.jit, tag="CUDA-bad",
        pos=pos, mass=mass, softening=0.1, gap=gap
    )

@pytest.mark.parametrize("N", (int(1e3), int(3e3), int(1e4), int(3e4), int(1e5)))
def bench_force(jax_bench, N):
    pos = jax.random.normal(jax.random.key(0), (N,3))
    mass = jnp.ones_like(pos[...,0])

    jb = jax_bench(jit_rounds=10, jit_warmup=1)

    jb.measure(
        fn_jit=forces.simple_cuda_force.jit, tag="CUDA-coalesced",
        pos=pos, mass=mass, softening=0.1,
    )
    jb.measure(
        fn_jit=forces.simple_cuda_force.jit, tag="CUDA-uncoalesced",
        pos=pos, mass=mass, softening=0.1, gap=10,
    )

    jb.measure(
        fn_jit=forces.cuda_force.jit, tag="CUDA-smem",
        pos=pos, mass=mass, softening=0.1,
    )

    jb.measure(
        fn_jit=forces.jax_force.jit, tag="jax",
        pos=pos, mass=mass, softening=0.1,
    )