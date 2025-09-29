#include <torch/extension.h>
#include <cuda_runtime.h>

// Forward declaration of CUDA launcher (this is defined in matmul_kernel.cu)
void matmul_launcher(const float* A, const float* B, float* C, int M, int N, int K);

// C++ interface (called from Python)
torch::Tensor matmul(const torch::Tensor& A, const torch::Tensor& B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D matrices");
    TORCH_CHECK(A.size(1) == B.size(0),
                "Inner dimension of the two matrices must match for matmul");

    int M = A.size(0);
    int K = A.size(1);  
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    matmul_launcher(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}  

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul, "Naive Matrix Multiplication (CUDA)");
} 