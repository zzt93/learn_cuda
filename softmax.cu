%%writefile cnn.cu

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset = offset >> 1) {
        val += __shlf_down_sync(0xffffffff, val, offset)
    }
    return val;
}

__device__ void block_sum(float val) {
    unsigned int idx = blockIdx.x;
    __shared__ float m_s[32];

    val = warp_reduce_sum(val);
    if (idx % warpSize == 0) {
        m_s[idx / warpSize] = val;
    }
    __syncthreads();
    if (idx < 32) {
        val = (threadIdx.x < blockDim.x/warpSize) ? m_s[idx / warpSize] : 0;
        val = warp_reduce_sum(val);
    }
    return val;
}

__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float block_max(float val) {
    unsigned int idx = blockIdx.x;
    __shared__ float m_s[32];

    val = warp_reduce_max(val);
    if (idx % warpSize == 0) {
        m_s[idx / warpSize] = val;
    }
    __syncthreads();
    if (idx < 32) {
        val = (threadIdx.x < blockDim.x/warpSize) ? m_s[idx / warpSize] : -FLT_MAX;
        val = warp_reduce_max(val);
    }
    return val;
}

__global__ void softmax(float *in, float *out, int n) {
    unsigned int idx = blockIdx.x;
    int stride = blockDim.x;
    __shared__ float maxRes;
    __shared__ float sum;

    // every block get max
    float maxIn = -FLT_MAX;
    for (int i = idx; i < n; i+=stride) {
        maxIn = fmaxf(maxIn, in[i]);
    }
    // block之间其实是重复计算了的
    maxIn = block_max(maxIn);
    if (idx == 0) {
        maxRes = maxIn;
    }
    __syncthreads();

    float val = 0;
    for (int i = idx; i < n; i+=stride) {
        val += expf(in[i]);
    }
    // block之间其实是重复计算了的
    val = block_sum(val);
    if (idx == 0) {
        sum = val;
    }
    __syncthreads();

    if (idx < n)
        out[idx] = expf(in[idx] - maxRes) / sum;
}