#include <torch/extension.h> 
#include <cuda_runtime.h>
//forward decleration 
void vector_add_launcher(const float* a,const float* b,float* c,int n);

torch::Tensor vector_add(const torch::Tensor& a,const torch::Tensor& b){
   TORCH_CHECK(a.device().is_cuda(),"Input a must be a CUDA Tensor");
   TORCH_CHECK(b.device().is_cuda(),"Input b must be a CUDA Tensor");
   TORCH_CHECK(a.is_contiguous(), "Input a must be contiguous");
   TORCH_CHECK(b.is_contiguous(), "Input b must be contiguous");

   TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same shape");

   auto c = torch::empty_like(a);
   int n = a.numel();

   vector_add_launcher(
    a.data_ptr<float>(),
    b.data_ptr<float>(),
    c.data_ptr<float>(),
    n
   );

   return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &vector_add, "Vector Addition (CUDA)");
}

