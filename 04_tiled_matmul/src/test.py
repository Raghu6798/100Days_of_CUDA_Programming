import torch
import tiled_matmul


def main():
    print("--- Day 2: Tiled Matrix Multiplication (Shared Memory) ---")
    
    M, K, N = 1024, 1024, 1024
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    print(f"Matrices: A=({M}x{K}), B=({K}x{N})\n")

    print("--- Verifying Correctness ---")
    C_pytorch = torch.matmul(A, B)
    C_tiled = tiled_matmul.forward(A, B)
    is_close = torch.allclose(C_pytorch, C_tiled, atol=1e-2)
    print(f"Tiled kernel is correct: {'SUCCESS' if is_close else 'FAILURE'}\n")

    if not is_close:
        print("Verification failed. Aborting benchmark.")
        return



if __name__ == "__main__":
    main()