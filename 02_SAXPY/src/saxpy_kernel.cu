#include <torch/extension.h> 

__global__ void saxpy_kernel(float a , const float* x,float* y,int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        y[i] = a * x[i] + y[i];
    }
}

void saxpy_launcher(float a , const float* x,float* y,int n){
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    saxpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, x, y, n);
}