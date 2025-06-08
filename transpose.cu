%%writefile transpose.cu

#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

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

// simple impl
// global row read, global col write
__global__ void transpose(float *in, float *out, int nx, int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    out[ix * ny + iy] = in[iy * nx + ix];
}

// use shared memory
// global row read, shared memory row write
// global row write, shared memory col read
// This kernel has uncoalesced shared accesses resulting in a total of 61440 excessive wavefronts (88% of the total 69632 wavefronts).
__global__ void transposeShared(float *in, float *out, int nx, int ny) {
    extern __shared__ float s[];
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int offset_in_block = threadIdx.y * blockDim.x + threadIdx.x;
    s[offset_in_block] = in[ix + iy * nx];
    __syncthreads();

    unsigned int new_block_ix = offset_in_block % blockDim.y;
    unsigned int new_block_iy = offset_in_block / blockDim.y;
    unsigned int new_ix = blockDim.y * blockIdx.y + new_block_ix;
    unsigned int new_iy = blockDim.x * blockIdx.x + new_block_iy;
    out[new_ix + new_iy * ny] = s[new_block_ix * blockDim.x + new_block_iy];
}

// shared memory without bank conflict
// bank_offset=1: This kernel has uncoalesced shared accesses resulting in a total of 4096 excessive wavefronts (33% of the total 12288 wavefronts).
__global__ void transposeShared2(float *in, float *out, int nx, int ny, int bank_offset) {
    extern __shared__ float s[];
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int offset_in_block = threadIdx.y * blockDim.x + threadIdx.x;
    s[threadIdx.y * (blockDim.x + bank_offset) + threadIdx.x] = in[ix + iy * nx];
    __syncthreads();

    unsigned int new_block_ix = offset_in_block % blockDim.y;
    unsigned int new_block_iy = offset_in_block / blockDim.y;
    unsigned int new_ix = blockDim.y * blockIdx.y + new_block_ix;
    unsigned int new_iy = blockDim.x * blockIdx.x + new_block_iy;
    out[new_ix + new_iy * ny] = s[new_block_ix * (blockDim.x + bank_offset) + new_block_iy];
}

#define N_CHUNKS    4

int main(int argc, char **argv) {
    int nx = 512, ny = 256;
    int mode = 0;
    int blockCnt = 16;
    if (argc >= 2) {
        mode = atoi(argv[1]);
    }
    if (argc >= 4) {
        nx = atoi(argv[2]);
        ny = atoi(argv[3]);
    }
    if (argc >= 5) {
        blockCnt = atoi(argv[4]);
    }

    // malloc
    float *h_in = (float*)malloc(nx * ny * sizeof(float));
    float *h_out = (float*)malloc(ny * nx * sizeof(float));
    if (mode >= 3) {
        CHECK(cudaMallocHost((void **) &h_out, nx * ny * sizeof(float)));
        CHECK(cudaMallocHost((void **) &h_in, nx * ny * sizeof(float)));
    }
    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, nx * ny * sizeof(float)));
    CHECK(cudaMalloc(&d_out, nx * ny * sizeof(float)));

    // init
    int c = 0;
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            h_in[i * nx + j] = c;
            c++;
        }
    }
    if (mode < 3) {
        CHECK(cudaMemcpy(d_in, h_in, nx * ny * sizeof(float), cudaMemcpyHostToDevice));
    }

    dim3 gridDim(blockCnt, blockCnt);
    int block_x = (nx-1)/blockCnt+1;
    int block_y = (ny-1)/blockCnt+1;
    dim3 blockDim(block_x, block_y);
    printf("grid(%d,%d) block(%d, %d)", blockCnt, blockCnt, (nx-1)/blockCnt+1, (ny-1)/blockCnt+1);
    // call
    int bank_offset = 1;
    switch (mode) {
        case 0:
            transpose<<<gridDim, blockDim>>>(d_in, d_out, nx, ny);
            break;
        case 1:
            transposeShared<<<gridDim, blockDim, block_x * block_y * sizeof(float)>>>(d_in, d_out, nx, ny);
            break;
        case 2:
            transposeShared2<<<gridDim, blockDim, (block_x+bank_offset) * block_y * sizeof(float)>>>(d_in, d_out, nx, ny, bank_offset);
            break;
        case 3: {
            cudaStream_t streams[N_CHUNKS];
            float *d_sub_out[N_CHUNKS];
            for (int i = 0; i < N_CHUNKS; ++i) {
                cudaStreamCreate(&streams[i]);
                CHECK(cudaMalloc(&d_sub_out[i], nx * ny / N_CHUNKS * sizeof(float)));
            }
            int chunk_size = nx * ny / N_CHUNKS;
            int chunk_bytes = chunk_size * sizeof(float);
            dim3 subGridDim(blockCnt, blockCnt/N_CHUNKS);
            int col_size = blockCnt/N_CHUNKS * blockDim.y;
            for (int i = 0; i < N_CHUNKS; ++i) {
                cudaMemcpyAsync(d_in + i * chunk_size, h_in + i * chunk_size, chunk_bytes, cudaMemcpyHostToDevice, streams[i]);
                transposeShared2<<<subGridDim, blockDim, (block_x+bank_offset) * block_y * sizeof(float), streams[i]>>>(d_in + i * chunk_size, d_sub_out[i], nx, ny/N_CHUNKS, bank_offset);
                for (int j = 0; j < blockDim.x * blockCnt; ++j) {
                    cudaMemcpyAsync(h_out + j * ny + i * col_size, d_sub_out[i] + j * col_size, col_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
                }
            }
            for (int i = 0; i < N_CHUNKS; ++i) {
                CHECK(cudaStreamSynchronize(streams[i]));
            }
            break;
        }
        case 4: {
            cudaStream_t streams[N_CHUNKS];
            float *d_sub_out[N_CHUNKS];
            for (int i = 0; i < N_CHUNKS; ++i) {
                cudaStreamCreate(&streams[i]);
                CHECK(cudaMalloc(&d_sub_out[i], nx * ny / N_CHUNKS * sizeof(float)));
            }
            int chunk_size = nx * ny / N_CHUNKS;
            int chunk_bytes = chunk_size * sizeof(float);
            dim3 subGridDim(blockCnt, blockCnt/N_CHUNKS);
            int col_size = blockCnt/N_CHUNKS * blockDim.y;
            for (int i = 0; i < N_CHUNKS; ++i) {
                cudaMemcpyAsync(d_in + i * chunk_size, h_in + i * chunk_size, chunk_bytes, cudaMemcpyHostToDevice, streams[i]);
            }
            for (int i = 0; i < N_CHUNKS; ++i) {
                transposeShared2<<<subGridDim, blockDim, (block_x + bank_offset) * block_y * sizeof(float), streams[i]>>>(d_in + i * chunk_size,d_sub_out[i], nx, ny / N_CHUNKS, bank_offset);
            }
            for (int i = 0; i < N_CHUNKS; ++i) {
                for (int j = 0; j < blockDim.x * blockCnt; ++j) {
                    cudaMemcpyAsync(h_out + j * ny + i * col_size, d_sub_out[i] + j * col_size, col_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
                }
            }
            for (int i = 0; i < N_CHUNKS; ++i) {
                CHECK(cudaStreamSynchronize(streams[i]));
            }
            break;
        }

    }
    if (mode < 3) {
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(h_out, d_out, nx * ny * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // validate
    bool wrong = false;
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            if (h_in[i * nx + j] != h_out[j * ny + i]) {
                wrong = true;
            }
        }
    }
    if (wrong) {
        printf("wrong \n");
    }

    // free
    if (mode < 3) {
        free(h_in);
        free(h_out);
    } else {
        cudaFreeHost(h_in);
        cudaFreeHost(h_out);
    }
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
}