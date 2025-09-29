
import torch
import vector_add_cpp

def main():
    print("--- Day 1: Vector Addition ---")
    tensor_size = 20_000_000
    a = torch.randn(tensor_size, device='cuda')
    b = torch.randn(tensor_size, device='cuda')
    print(f"Tensors created with size: {tensor_size} on device: {a.device}\n")

    c_pytorch = a + b
    print("PyTorch native addition complete.")

    c_cuda = vector_add_cpp.forward(a, b)
    print("Custom C++/CUDA kernel complete.")

    print("\n--- Verifying results ---")
    is_cuda_close = torch.allclose(c_pytorch, c_cuda)
    print(f"Verification (PyTorch vs C++/CUDA): {'SUCCESS' if is_cuda_close else 'FAILURE'}")

if __name__ == "__main__":
    main()