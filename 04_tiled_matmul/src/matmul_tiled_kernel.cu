#include <torch/extension.h>
#include <cuda_runtime.h>

const int TILE_WIDTH = 16;
__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K){
    __shared__ float a_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float b_tile[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    float value = 0.0f;

    for (int phase = 0; phase < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++phase){
        int a_global_row = row;
        int a_global_col = phase * TILE_WIDTH + tx;
        int b_global_row = phase * TILE_WIDTH + ty;
        int b_global_col = col;

        a_tile[ty][tx] = (a_global_row < M && a_global_col < K) ? A[a_global_row * K + a_global_col] : 0.0f;
        b_tile[ty][tx] = (b_global_row < K && b_global_col < N) ? B[b_global_row * N + b_global_col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k){
            value += a_tile[ty][k] * b_tile[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N){
        C[row * N + col] = value;
    }
}

void matmul_tiled_launcher(const float* A, const float* B, float* C, int M, int N, int K){
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 BlocksPerGrid(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (M + TILE_WIDTH - 1) / TILE_WIDTH
    );

    matmul_tiled_kernel<<<BlocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
}
