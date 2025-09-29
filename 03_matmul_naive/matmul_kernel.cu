#include <cuda_runtime.h>


__global__ void matmul_kernel(const float* A,const float* B,float* C,int M,int N,int K){

    // 2D Thread indexing 
    int row =  blockIdx.y * blockDim.y + threadIdx.y;
    int col =  blockIdx.x * blockDim.x + threadIdx.x;

    //Ensures threads that fall outside the matrix dimensions don’t write to memory (common when grid size > matrix size).
    if (row < M && col < N) {
        float value = 0.0f;

        for (int k = 0; k < K; ++k) {
    value += A[row * K + k] * B[k * N + col];
}
        C[row*N + col] = value;
    }
}

void matmul_launcher(const float* A, const float* B, float* C, int M, int N, int K) {
    
    dim3 threadsPerBlock(16, 16);

    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
}