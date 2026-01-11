#ifndef FORCES_CUH
#define FORCES_CUH

/* ---------------------------------------------------------------------------------------------- */
/*                              Some helpful structures and functions                             */
/* ---------------------------------------------------------------------------------------------- */

// Calculates integer division in round-up mode
__host__ __device__ __forceinline__ int div_ceil(int a, int b) {
    return (a + b - 1) / b;
}

struct __align__(16) PosMass {
    float3 pos;
    float mass;
};

/* ---------------------------------------------------------------------------------------------- */
/*                                     A simple example Kernel                                    */
/* ---------------------------------------------------------------------------------------------- */

__global__ void Multiply(
    const float* x,
    const float* y,
    float* output,
    size_t num
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx > num)
        return;
    output[idx] = x[idx] * y[idx];
}

/* ---------------------------------------------------------------------------------------------- */
/*                                       Simple Force Kernel                                      */
/* ---------------------------------------------------------------------------------------------- */

__device__ __forceinline__ float3 get_force(PosMass p1, PosMass p2, float eps2){
    float3 dx = make_float3(p1.pos.x - p2.pos.x, p1.pos.y - p2.pos.y, p1.pos.z - p2.pos.z);
    float r2 = dx.x*dx.x + dx.y*dx.y + dx.z*dx.z + eps2;
    float rinv = rsqrtf(r2); // inverse square-root
    float mrinv3 = p1.mass*rinv*rinv*rinv;

    return make_float3(dx.x*mrinv3, dx.y*mrinv3, dx.z*mrinv3);
}

__global__ void DirectSummationForce(
    const PosMass *xm,
    float3 *force_out,
    int n,
    float epsilon
) {
    const int steps = div_ceil(n, blockDim.x);
    float epsilon2 = epsilon * epsilon;

    int ipart = blockIdx.x * blockDim.x + threadIdx.x;
    PosMass xmi;
    if(ipart < n)
        xmi = xm[ipart];

    extern __shared__ PosMass xmj_shared[];

    float3 force = {0.f, 0.f, 0.f};

    for (int jblock = 0; jblock < steps; jblock += 1) {
        int num = min(blockDim.x, n - blockDim.x * jblock);

        __syncthreads();
        if(threadIdx.x < num)
            xmj_shared[threadIdx.x] = xm[jblock * blockDim.x + threadIdx.x];
        __syncthreads();

        for (int j = 0; j < num; j++) {
            float3 fij = get_force(xmj_shared[j], xmi, epsilon2);
            force = make_float3(force.x + fij.x, force.y + fij.y, force.z + fij.z);
        }
    }

    if(ipart < n)
        force_out[blockIdx.x * blockDim.x + threadIdx.x] = force;
}

#endif