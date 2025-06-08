%%writefile cnn.cu

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

//This kernel has uncoalesced global accesses resulting in a total of 657649 excessive sectors (13% of the total 4875185 sectors).
//    Section: Source Counters
//    ------------------------- ----------- ------------
//    Metric Name               Metric Unit Metric Value
//    ------------------------- ----------- ------------
//    Branch Instructions Ratio           %         0.06
//    Branch Instructions              inst      915,776
//    Branch Efficiency                   %          100
//    Avg. Divergent Branches                          0
__global__ void convolve2d(float *img, int img_nx, int img_ny, float *kernel, int k_nx, int k_ny, float *out, int out_nx, int out_ny, int stride, int padding, int padding_value) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= out_nx || iy >= out_ny) return;

    int img_x_s = -padding + ix * stride;
    int img_y_s = -padding + iy * stride;
    float sum = 0;
    for (int i = 0; i < k_ny; i++) {
        for (int j = 0; j < k_nx; j++) {
            int img_y = img_y_s + i;
            int img_x = img_x_s + j;
            if (img_x < 0 || img_y < 0 || img_x >= img_nx || img_y >= img_ny) {
                sum += kernel[i * k_nx + j] * padding_value;
                continue;
            }
            sum += kernel[i * k_nx + j] * img[img_x + (img_y) * img_nx];
        }
    }
    out[iy * out_nx + ix] = sum;
}

//This kernel has uncoalesced global accesses resulting in a total of 657649 excessive sectors (13% of the total 4875185 sectors)
//    Metric Name               Metric Unit Metric Value
//      ------------------------- ----------- ------------
//      Branch Instructions Ratio           %         0.12
//      Branch Instructions              inst    1,896,896
//      Branch Efficiency                   %          100
//      Avg. Divergent Branches                          0
__global__ void convolve2dOpt(float *img, int img_nx, int img_ny, float *kernel, int k_nx, int k_ny, float *out, int out_nx, int out_ny, int stride, int padding, int padding_value) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= out_nx || iy >= out_ny) return;

    int img_x_s = -padding + ix * stride;
    int img_y_s = -padding + iy * stride;
    float sum = 0;
    for (int i = 0; i < k_ny; i++) {
        int img_y = img_y_s + i;
        if (img_y < 0 || img_y >= img_ny) {
            for (int j = 0; j < k_nx; j++) {
                sum += kernel[i * k_nx + j] * padding_value;
            }
            continue;
        }
        for (int j = 0; j < k_nx; j++) {
            int img_x = img_x_s + j;
            if (img_x < 0 || img_x >= img_nx) {
                sum += kernel[i * k_nx + j] * padding_value;
                continue;
            }
            sum += kernel[i * k_nx + j] * img[img_x + (img_y) * img_nx];
        }
    }
    out[iy * out_nx + ix] = sum;
}

//This kernel has uncoalesced global accesses resulting in a total of 668117 excessive sectors (14% of the total 4886933 sectors).
__global__ void convolve2dBranch(float *img, int img_nx, int img_ny, float *kernel, int k_nx, int k_ny, float *out, int out_nx, int out_ny, int stride, int padding, int padding_value) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= out_nx || iy >= out_ny) return;

    int img_x_s = -padding + ix * stride;
    int img_y_s = -padding + iy * stride;
    float sum = 0;
    for (int i = 0; i < k_ny; i++) {
        for (int j = 0; j < k_nx; j++) {
            int img_x = img_x_s + j;
            int img_y = img_y_s + i;
            int cond = img_x < 0 || img_y < 0 || img_x >= img_nx || img_y >= img_ny;
            float v = cond * padding_value + (!cond) * img[img_x + (img_y) * img_nx];
            sum += kernel[i * k_nx + j] * v;
        }
    }
    out[iy * out_nx + ix] = sum;
}

__global__ void convolve2dShared(float *img, int img_nx, int img_ny, float *kernel, int k_nx, int k_ny, float *out, int out_nx, int out_ny, int stride, int padding, int padding_value, int s_nx, int s_ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= out_nx || iy >= out_ny) return;

    extern __shared__ float s_block_img[];
    int img_x_start = -padding + ix * stride;
    int img_y_start = -padding + iy * stride;
    int cond_x = threadIdx.x == blockDim.x - 1, cond_y = threadIdx.y == blockDim.y - 1;
    int y_range = cond_y * k_ny + (!cond_y) * stride;
    int x_range = cond_x * k_nx + (!cond_x) * stride;
    for (int y = 0; y < y_range; y++) {
        int s_y = y + threadIdx.y;
        int img_y = img_y_start + y;
        for (int x = 0; x < x_range; x++) {
            int s_x = x + threadIdx.x;
            int img_x = img_x_start + x;
            if (img_y < 0 || img_y >= img_ny || img_x < 0 || img_x >= img_nx) {
                s_block_img[s_x + s_y * s_nx] = padding_value;
                continue;
            }
            s_block_img[s_x + s_y * s_nx] = img[img_x + img_y * img_nx];
        }
    }
    __syncthreads();

    int shared_start_x = threadIdx.x * stride;
    int shared_start_y = threadIdx.y * stride;
    float sum = 0;
    for (int y = 0; y < k_ny; y++) {
        for (int x = 0; x < k_nx; x++) {
            int img_x = shared_start_x + x;
            int img_y = shared_start_y + y;
            sum += kernel[y * k_nx + x] * s_block_img[img_x + img_y * s_nx];
        }
    }
    out[iy * out_nx + ix] = sum;
}



int main(int argc, char **argv) {
    int mode = 0;
    if (argc >= 2) {
        mode = atoi(argv[1]);
    }
    int nx = 1024, ny = 1024;
    int stride = 1, padding = 1, padding_value=0;
    int kx = 5, ky = 5;
    int out_nx = ((nx + padding * 2) - kx) /stride + 1;
    int out_ny = ((ny + padding * 2) - ky) /stride + 1;

    int block_nx = 32, block_ny = 32;
    // alloc
    float *h_in = (float*)malloc(sizeof(float) * nx * ny);
    float *h_out = (float*)malloc(sizeof(float) * out_nx * out_ny);
    float *h_kernel = (float*)malloc(sizeof(float) * kx * ky);
    float *d_in, *d_out, *d_kernel;
    CHECK(cudaMalloc(&d_in, sizeof(float) * nx * ny));
    CHECK(cudaMalloc(&d_out, sizeof(float) * out_nx * out_ny));
    CHECK(cudaMalloc(&d_kernel, sizeof(float) * kx * ky));
    // init
    dim3 blockDim(block_nx, block_ny);
    dim3 gridDim((out_nx-1)/block_nx+1, (out_ny-1)/block_ny+1);
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            h_in[y * nx + x] = 1;
        }
    }
    for (int y = 0; y < ky; y++) {
        for (int x = 0; x < kx; x++) {
            h_kernel[y * kx + x] = 1;
        }
    }
    CHECK(cudaMemcpy(d_in, h_in, sizeof(float) * nx * ny, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_kernel, h_kernel, sizeof(float) * kx * ky, cudaMemcpyHostToDevice));

    // call
    switch (mode) {
        case 0:
            convolve2d<<<gridDim, blockDim>>>(d_in, nx, ny, d_kernel, kx, ky, d_out, out_nx, out_ny, stride, padding, padding_value);
            break;
        case 1:
            convolve2dBranch<<<gridDim, blockDim>>>(d_in, nx, ny, d_kernel, kx, ky, d_out, out_nx, out_ny, stride, padding, padding_value);
            break;
        case 2:
            convolve2dOpt<<<gridDim, blockDim>>>(d_in, nx, ny, d_kernel, kx, ky, d_out, out_nx, out_ny, stride, padding, padding_value);
            break;
        case 3:
            if (kx < stride) {
                printf("not support now");
                break;
            }
            int block_shared_nx = (blockDim.x - 1)* stride + kx;
            int block_shared_ny = (blockDim.y - 1)* stride + ky;
            convolve2dShared<<<gridDim, blockDim, sizeof(float)*block_shared_nx*block_shared_ny>>>(d_in, nx, ny, d_kernel, kx, ky, d_out, out_nx, out_ny, stride, padding, padding_value, block_shared_nx, block_shared_ny);
            break;
    }
    CHECK(cudaGetLastError());
    // check
    CHECK(cudaMemcpy(h_out, d_out, sizeof(float) * out_nx * out_ny, cudaMemcpyDeviceToHost));
    bool wrong = false;
    for (int y = 0; y < out_ny; y++) {
        for (int x = 0; x < out_nx; x++) {
            if (x == 0 && y == 0 || x == out_nx-1 && y == out_ny-1 || x == 0 && y == out_ny-1 || x == out_nx-1 && y == 0) {
                if (h_out[y * out_nx + x] != 16) {
                    wrong = true;
                }
            } else if (x == 0 || y == 0 || x == out_nx-1 || y == out_ny-1) {
                if (h_out[y * out_nx + x] != 20) {
                    wrong = true;
                }
            } else {
                if (h_out[y * out_nx + x] != 25) {
                    wrong = true;
                }
            }
        }
    }
    if (wrong) {
        printf("wrong\n");
    }
    // free

}