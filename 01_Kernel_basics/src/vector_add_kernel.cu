#include <torch/extension.h> 

__global__ void vector_add_kernel(const float* a,const float* b,float* c,int n ){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n){
        c[index] = a[index] + b[index];
    }
}


void vector_add_launcher(const float* a,const float* b, float* c,int n){

    const int threadsPerBlock = 256;

    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, n);
}