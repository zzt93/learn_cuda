%%writefile reduce.cu

#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}

// ncu: On average, each warp of this kernel spends 10.7 cycles
// being stalled waiting for sibling warps at a CTA barrier
__global__ void reduce(float *A, float *out, int N) {
    unsigned int offset = blockDim.x * blockIdx.x;
    unsigned int l_idx = threadIdx.x;
    if (offset + l_idx >= N) return;
    float *l_a = A + offset;

    for (int stride = blockDim.x/2; stride > 0; stride/=2) {
        if (l_idx < stride) {
            l_a[l_idx] += l_a[l_idx + stride];
        }
        __syncthreads();
    }

    if (l_idx == 0) {
        out[blockIdx.x] = l_a[l_idx];
//        printf("%d sum: %f\n", blockIdx.x, l_a[l_idx]);
    }
}

// On average, each warp of this kernel spends 7.7 cycles being stalled waiting for a
// scoreboard dependency on a L1TEX (local, global, surface, texture) operation.
// one block handle 4 * blockDim.x data
// blockCnt/4
// when element <= 32, not use __syncthreads
__global__ void reduceUnroll(float *A, float *out, int N) {
    unsigned int offsetStart = blockDim.x * blockIdx.x;
    unsigned int offsetEnd = blockDim.x * (blockIdx.x + 3);
    unsigned int l_idx = threadIdx.x;
    if (offsetEnd + l_idx >= N) return;

    float *l_a = A + offsetStart;
    {
        float a = l_a[l_idx];
        float b = l_a[l_idx + blockDim.x];
        float c = l_a[l_idx + blockDim.x * 2];
        float d = l_a[l_idx + blockDim.x * 3];
        l_a[l_idx] = a + b + c + d;
        __syncthreads();
    }

    for (int stride = blockDim.x / 2; stride > 32; stride/=2) {
        if (l_idx < stride) {
            l_a[l_idx] += l_a[l_idx + stride];
        }
        __syncthreads();
    }

    volatile float *t = l_a;
    if (l_idx < 32) {
        t[l_idx] += t[l_idx + 32];
        t[l_idx] += t[l_idx + 16];
        t[l_idx] += t[l_idx + 8];
        t[l_idx] += t[l_idx + 4];
        t[l_idx] += t[l_idx + 2];
        t[l_idx] += t[l_idx + 1];
    }

    if (l_idx == 0) {
        out[blockIdx.x] = t[l_idx];
    }
}

__global__ void reduceUnrollShared(float *A, float *out, int N) {
    unsigned int offsetStart = blockDim.x * blockIdx.x;
    unsigned int offsetEnd = blockDim.x * (blockIdx.x + 3);
    unsigned int l_idx = threadIdx.x;
    if (offsetEnd + l_idx >= N) return;

    float *l_a = A + offsetStart;
    {
        float a = l_a[l_idx];
        float b = l_a[l_idx + blockDim.x];
        float c = l_a[l_idx + blockDim.x * 2];
        float d = l_a[l_idx + blockDim.x * 3];
        l_a[l_idx] = a + b + c + d;
        __syncthreads();
    }

    for (int stride = blockDim.x / 2; stride > 32; stride/=2) {
        if (l_idx < stride) {
            l_a[l_idx] += l_a[l_idx + stride];
        }
        __syncthreads();
    }

    __shared__ volatile float t[64];
    if (l_idx < 32) {
        t[l_idx] = l_a[l_idx] + l_a[l_idx + 32];
        t[l_idx] += t[l_idx + 16];
        t[l_idx] += t[l_idx + 8];
        t[l_idx] += t[l_idx + 4];
        t[l_idx] += t[l_idx + 2];
        t[l_idx] += t[l_idx + 1];
    }

    if (l_idx == 0) {
        out[blockIdx.x] = t[l_idx];
    }
}

__global__ void reduceUnrollShlf(float *A, float *out, int N) {
    unsigned int offsetStart = blockDim.x * blockIdx.x;
    unsigned int offsetEnd = blockDim.x * (blockIdx.x + 3);
    unsigned int l_idx = threadIdx.x;
    if (offsetEnd + l_idx >= N) return;

    float *l_a = A + offsetStart;
    {
        float a = l_a[l_idx];
        float b = l_a[l_idx + blockDim.x];
        float c = l_a[l_idx + blockDim.x * 2];
        float d = l_a[l_idx + blockDim.x * 3];
        l_a[l_idx] = a + b + c + d;
        __syncthreads();
    }

    for (int stride = blockDim.x / 2; stride >= 32; stride/=2) {
        if (l_idx < stride) {
            l_a[l_idx] += l_a[l_idx + stride];
        }
        __syncthreads();
    }

    float t = l_a[l_idx];
    if (l_idx < 32) {
        t += __shfl_down_sync(0xFFFFFFFF, t, 16);
        t += __shfl_down_sync(0x0000FFFF, t, 8);
        t += __shfl_down_sync(0x000000FF, t, 4);
        t += __shfl_down_sync(0x0000000F, t, 2);
        t += __shfl_down_sync(0x00000003, t, 1);
    }

    if (l_idx == 0) {
        out[blockIdx.x] = t;
    }
}

//__global__ void reduceUnrollWarpReduce(float *A, float *out, int N) {
//    unsigned int offsetStart = blockDim.x * blockIdx.x;
//    unsigned int offsetEnd = blockDim.x * (blockIdx.x + 3);
//    unsigned int l_idx = threadIdx.x;
//    if (offsetEnd + l_idx >= N) return;
//
//    float *l_a = A + offsetStart;
//    {
//        float a = l_a[l_idx];
//        float b = l_a[l_idx + blockDim.x];
//        float c = l_a[l_idx + blockDim.x * 2];
//        float d = l_a[l_idx + blockDim.x * 3];
//        l_a[l_idx] = a + b + c + d;
//        __syncthreads();
//    }
//
//    for (int stride = blockDim.x / 2; stride >= 32; stride/=2) {
//        if (l_idx < stride) {
//            l_a[l_idx] += l_a[l_idx + stride];
//        }
//        __syncthreads();
//    }
//
//    float t = l_a[l_idx];
//    if (l_idx < 32) {
//        t = __reduce_add_sync(0xFFFFFFFF, t);
//    }
//
//    if (l_idx == 0) {
//        out[blockIdx.x] = t;
//    }
//}

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg=cooperative_groups;

/// The following example accepts input in *A and outputs a result into *sum
/// It spreads the data equally within the block
__device__ void block_reduce(const float * A, int count, cuda::atomic<float, cuda::thread_scope_block>& total_sum) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    float thread_sum = 0;

    // Stride loop over all values, each thread accumulates its part of the array.
    for (int i = block.thread_rank(); i < count; i += block.size()) {
        thread_sum += A[i];
    }

    // reduce thread sums in the tile, add the result to the atomic by all tiles
    // cg::plus<float> allows cg::reduce() to know it can use hardware acceleration for addition
    cg::reduce_update_async(tile, total_sum, thread_sum, cg::plus<float>());

    // synchronize the block, to ensure all async reductions are ready
    block.sync();
}

__global__ void reduceUnrollCG(float *A, float *out, int N) {
    __shared__ cuda::atomic<float, cuda::thread_scope_block> sum;
    if (threadIdx.x == 0) {
        sum.store(0.0f, cuda::memory_order_relaxed);
    }
    __syncthreads();
    block_reduce(A + blockIdx.x * 4 * blockDim.x, blockDim.x * 4, sum);
    if (threadIdx.x == 0) {
        out[blockIdx.x] = sum.load();
    }
}

/// The following example accepts input in *A and outputs a result into *sum
/// It spreads the data equally within the block
__device__ void block_reduce_sync(const float * A, int count, float *out) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    float thread_sum = 0;

    // Stride loop over all values, each thread accumulates its part of the array.
    for (int i = block.thread_rank(); i < count; i += block.size()) {
        thread_sum += A[i];
    }

    // reduce thread sums across the tile, add the result to the atomic
    // cg::plus<float> allows cg::reduce() to know it can use hardware acceleration for addition
    float tile_sum = cg::reduce(tile, thread_sum, cg::plus<float>());

    if (threadIdx.x == 0) {
        out[blockIdx.x] = sum.load();
    }
    // synchronize the block, to ensure all async reductions are ready
    block.sync();
}

__global__ void reduceUnrollCGSync(float *A, float *out, int N) {
    block_reduce(A + blockIdx.x * 4 * blockDim.x, blockDim.x * 4, out);
}

__global__ void reduceUnrollSharedAll(float *A, float *out, int N) {
    unsigned int offsetStart = blockDim.x * blockIdx.x;
    unsigned int offsetEnd = blockDim.x * (blockIdx.x + 3);
    unsigned int l_idx = threadIdx.x;
    if (offsetEnd + l_idx >= N) return;

    float *l_a = A + offsetStart;
    extern __shared__ float s_a[]; // 动态共享内存声明
    {
        float a = l_a[l_idx];
        float b = l_a[l_idx + blockDim.x];
        float c = l_a[l_idx + blockDim.x * 2];
        float d = l_a[l_idx + blockDim.x * 3];
        s_a[l_idx] = a + b + c + d;
        __syncthreads();
    }

    for (int stride = blockDim.x / 2; stride > 32; stride/=2) {
        if (l_idx < stride) {
            s_a[l_idx] += s_a[l_idx + stride];
        }
        __syncthreads();
    }
    __shared__ volatile float t[64];
    if (l_idx < 32) {
        t[l_idx] = s_a[l_idx] + s_a[l_idx + 32];
        t[l_idx] += t[l_idx + 16];
        t[l_idx] += t[l_idx + 8];
        t[l_idx] += t[l_idx + 4];
        t[l_idx] += t[l_idx + 2];
        t[l_idx] += t[l_idx + 1];
    }

    if (l_idx == 0) {
        out[blockIdx.x] = t[l_idx];
    }
}

int main(int argc,char** argv) {
    int N = 1024 * 128;
    float A[N];
    float sum = 0;
    for (int i = 0; i < N; i++) {
        A[i] = 1;
        sum += A[i];
    }

    int mode = 0;
    if (argc > 1) {
        mode = atoi(argv[1]);
    }
    int blockCnt = 128;
    if (argc > 2) {
        blockCnt = atoi(argv[2]);
    }
    int all_threads = N;
    if (mode > 0) {
        all_threads = N / 4;
    }

    dim3 grid(blockCnt, 1);
    dim3 block((all_threads-1) / blockCnt + 1, 1);

    float *d_sum, *d_in;
    float *h_sum = (float *)malloc(sizeof(float) * blockCnt);
    CHECK(cudaMalloc(&d_sum, sizeof(float) * blockCnt));
    CHECK(cudaMalloc(&d_in, sizeof(float) * N));
    CHECK(cudaMemcpy(d_in, A, sizeof(float) * N, cudaMemcpyHostToDevice));
    switch (mode) {
        case 0:
            reduce<<<grid, block>>>(d_in, d_sum, N);
            break;
        case 1:
            reduceUnroll<<<grid, block>>>(d_in, d_sum, N);
            break;
        case 2:
            reduceUnrollShared<<<grid, block>>>(d_in, d_sum, N);
            break;
        case 3:
            reduceUnrollShlf<<<grid, block>>>(d_in, d_sum, N);
            break;
        case 4:
            reduceUnrollSharedAll<<<grid, block, block.x * sizeof(float)>>>(d_in, d_sum, N);
            break;
        case 5:
            reduceUnrollCG<<<grid, block, block.x * sizeof(float)>>>(d_in, d_sum, N);
            break;
        case 6:
            reduceUnrollCGSync<<<grid, block, block.x * sizeof(float)>>>(d_in, d_sum, N);
            break;
    }

    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_sum, d_sum, sizeof(float) * blockCnt, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_sum));
    CHECK(cudaFree(d_in));

    float result = 0;
    for (int i = 0; i < blockCnt; i++) {
        result += h_sum[i];
    }
    free(h_sum);
    printf("gpu result: %f, cpu sum: %f\n", result, sum);

//    std::cout << "gpu result: " << result << " cpu sum: " << sum << std::endl;
    return 0;
}