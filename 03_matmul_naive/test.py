import torch
import matmul_cpp # Import the new, simple name

def main():
    print("--- Day 3: Naive Matrix Multiplication ---")
    M, K, N = 512, 1024, 256
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    print(f"Matrices: A=({M}x{K}), B=({K}x{N}) on device: {A.device}\n")

    C_pytorch = torch.matmul(A, B)
    print("PyTorch native matmul complete.")

    C_custom = matmul_cpp.forward(A, B) # Call the new module
    print("Custom C++/CUDA matmul kernel complete.\n")
    
    print("--- Verifying results ---")
    is_close = torch.allclose(C_pytorch, C_custom, atol=1e-3)
    print(f"Verification (results are close): {'SUCCESS' if is_close else 'FAILURE'}")

if __name__ == "__main__":
    main()