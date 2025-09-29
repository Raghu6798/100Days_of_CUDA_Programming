#include <torch/extension.h> 
#include <cuda_runtime.h>


void saxpy_launcher(float a , const float* x,float* y,int n);

torch::Tensor saxpy(float a, const torch::Tensor& x, torch::Tensor& y){
    TORCH_CHECK(x.device().is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(y.device().is_cuda(), "Input y must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "Input y must be contiguous");
    TORCH_CHECK(x.sizes() == y.sizes(), "Input tensors must have the same shape");

    int n = x.numel();
    saxpy_launcher(
        a,
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &saxpy, "SAXPY (a * x + y) kernel (CUDA)");
}