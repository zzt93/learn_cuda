%%writefile mm.cu

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

__global__ void gemm(float *l, int l_nx, int l_ny, float *r, int r_nx, float *out) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int out_nx = r_nx;
    float sum = 0;
    for (int i = 0; i < l_nx; i++) {
        sum += l[iy * l_nx + i] * r[i * r_nx + ix];
    }
    out[ix + iy * out_nx] = sum;
}

// 假设 A 为 MxK, B 为 KxN, C 为 MxN
template<int TILE_SIZE>
__global__ void matmul_shared(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;  // 块内线程坐标
    int row = blockIdx.y * blockDim.y + ty;  // 全局行索引
    int col = blockIdx.x * blockDim.x + tx;  // 全局列索引

    float sum = 0.0;
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 协作加载数据到共享内存
        int tiledCol = t * TILE_SIZE + tx;
        int tiledRow = t * TILE_SIZE + ty;

        // 边界检查（防止越界）
        tileA[ty][tx] = (row < M && tiledCol < K) ? A[row * K + tiledCol] : 0.0;
        tileB[ty][tx] = (tiledRow < K && col < N) ? B[tiledRow * N + col] : 0.0;

        __syncthreads();  // 同步块内所有线程

        // 用共享内存计算子块乘积
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads();  // 确保计算完成再加载下一块
    }

// 写入结果
    if (row < M && col < N) C[row * N + col] = sum;
}


// 假设 A 为 MxK, B 为 KxN, C 为 MxN
// 并行范围：block
// (x, y) --- write c[y][x]
// (x, y) --- read a[y][], b[][x]
// (tid.x, tid.y) --- write s_a[tid.y][tid.x], b[tid.y][tid.x]
// (tid.x, tid.y) --- read s_a[tid.y][], b[][tid.x]
#define tile_size 32
__global__ void mm_shared(float *a, float *b, float *c, int m, int n, int k) {
    __shared__ float s_a[tile_size][tile_size];
    __shared__ float s_b[tile_size][tile_size];

    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int l_x = threadIdx.x;
    unsigned int l_y = threadIdx.y;

    float sum = 0;
    for (int ki = 0; ki < (k-1)/tile_size + 1; ki++) {
        unsigned int tile_x = l_x + ki * tile_size;
        unsigned int tile_y = l_y + ki * tile_size;
        // init shared mem
        s_a[l_y][l_x] = a[y * k + tile_x];
        s_b[l_y][l_x] = b[tile_y * n + x];

        __syncthreads();

        // sum in shared mem
        for (int s = 0; s < tile_size; s++) {
            sum += s_a[l_y][s] * s_b[s][l_x];
        }
        __syncthreads();  // 确保计算完成再加载下一块

    }
    c[y * n + x] = sum;
}

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

__global__ void mm_shared_async(float *a, float *b, float *c, int m, int n, int k) {
    __shared__ float s_a[tile_size][tile_size];
    __shared__ float s_b[tile_size][tile_size];
    auto block = cooperative_groups::this_thread_block();

    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int l_x = threadIdx.x;
    unsigned int l_y = threadIdx.y;

    float sum = 0;
    for (int ki = 0; ki < (k-1)/tile_size + 1; ki++) {
        unsigned int a_start_x = ki * tile_size, b_start_y = ki * tile_size;
        unsigned int a_start_y = y - l_y, b_start_x = x - l_x;
        // init shared mem
        for (int j = 0; j < tile_size; j++) {
            cooperative_groups::memcpy_async(block, (float *)s_a + j * tile_size, a + a_start_y * k + a_start_x, sizeof(float) * tile_size);
            cooperative_groups::memcpy_async(block, (float *)s_b + j * tile_size, b + b_start_y * n + b_start_x, sizeof(float) * tile_size);
        }
        cooperative_groups::wait(block); // Joins all threads, waits for all copies to complete

        // sum in shared mem
        for (int s = 0; s < tile_size; s++) {
            sum += s_a[l_y][s] * s_b[s][l_x];
        }
        __syncthreads();  // 确保计算完成再加载下一块

    }
    c[y * n + x] = sum;
}


// 算法优化
// shared mem 缓存left，如果left小于64k（低精度就支持更多元素）
// 尽量都放到同一个block，1024个thread（output多于1024，可以考虑+stride）
// 矩阵分块乘法
int main(int argc, char ** argv) {
    int m = 1024, k = 512, n = 1024;

    int mode = 0;
    if (argc > 1) {
        mode = atoi(argv[1]);
    }

    // alloc
    float *h_a = (float *)malloc(sizeof(float) * m * k);
    float *h_b = (float *)malloc(sizeof(float) * n * k);
    float *h_c = (float *)malloc(sizeof(float) * n * n);
    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, sizeof(float )*m * k));
    CHECK(cudaMalloc(&d_b, sizeof(float )*n * k));
    CHECK(cudaMalloc(&d_c, sizeof(float )*n * m));
    // init
    for (int y = 0; y < m; y++) {
        int v = 2;
        if (y % 2 == 0) {
            v = 1;
        }
        for (int x = 0; x < k; x++) {
            h_a[y * k + x] = v;
        }
    }
    for (int y = 0; y < k; y++) {
        for (int x = 0; x < n; x++) {
            int v = 2;
            if (x % 2 == 0) {
                v = 1;
            }
            h_b[y * n + x] = v;
        }
    }
    CHECK(cudaMemcpy(d_a, h_a, sizeof(float) * m * k, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, sizeof(float) * n * k, cudaMemcpyHostToDevice));
    // call
    dim3 blockDim(tile_size, tile_size);
    dim3 gridDim((m-1)/tile_size+1, (n-1)/tile_size+1);
    switch (mode) {
        case 0:
            gemm<<<gridDim, blockDim>>>(d_a, k, m, d_b, n, d_c);
            break;
        case 1:
            matmul_shared<tile_size><<<gridDim, blockDim>>>(d_a, d_b, d_c, m, n, k);
            break;
        case 2:
            mm_shared<<<gridDim, blockDim>>>(d_a, d_b, d_c, m, n, k);
            break;
        case 3:
            mm_shared_async<<<gridDim, blockDim>>>(d_a, d_b, d_c, m ,n, k);
            break;
    }
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_c, d_c, sizeof(float) * n * m, cudaMemcpyDeviceToHost));
    // check
    bool wrong = false;
    for (int y = 0; y < m; y++) {
        for (int x = 0; x < n; x++) {
            int v = 2;
            if (y % 2 == 0) {
                v = 1;
            }
            if (x % 2 == 1) {
                v *= 2;
            }
            if (h_c[y * n + x] != v * k) {
                if (y < 2 && x < 4) {
                    printf("%f == %d ", h_c[y * n + x], v*k);
                }
                wrong = true;
            }
        }
    }
    if (wrong) {
        printf("Wrong\n");
    } else {
        printf("Correct\n");
    }
    // free
}