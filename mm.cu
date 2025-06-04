//%%writefile mm.cu

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
    unsigned int out_ny = l_ny;
    float sum = 0;
    for (int i = 0; i < l_nx; i++) {
        sum += l[iy * l_nx + i] * r[i * r_nx + ix];
    }
    out[ix + iy * out_nx] = sum;
}

// // 假设 A 为 MxK, B 为 KxN, C 为 MxN
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

// 算法优化
// shared mem 缓存left，如果left小于64k（低精度就支持更多元素）
// 尽量都放到同一个block，1024个thread（output多于1024，可以考虑+stride）
// 矩阵分块乘法
int main(int argc, char ** argv) {

}