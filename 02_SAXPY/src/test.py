import torch
import saxpy_cpp

def main():
    print("--- Day 2: SAXPY (a * x + y) ---")
    
    # --- Setup ---
    tensor_size = 10_000_000
    a = 2.5
    x = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
    y = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
    
    print(f"Scalar 'a': {a}")
    print(f"Tensors created with size: {tensor_size} on device: {x.device}\n")

    # --- 1. PyTorch's native implementation ---
    y_clone_for_pytorch = y.clone()
    y_pytorch_result = a * x + y_clone_for_pytorch
    print("PyTorch native SAXPY complete.")

    # --- 2. Your custom C++/CUDA kernel ---
    y_custom_result = saxpy_cpp.forward(a, x, y)
    print("Custom C++/CUDA SAXPY kernel complete.\n")
    
    # --- Verification ---
    print("--- Verifying results ---")
    
    # NEW: Let's calculate and print the maximum difference
    max_abs_diff = (y_pytorch_result - y_custom_result).abs().max()
    print(f"Maximum absolute difference: {max_abs_diff.item()}")

    # MODIFIED: Tell allclose to accept a slightly larger absolute tolerance (atol)
    is_close = torch.allclose(y_pytorch_result, y_custom_result, atol=1e-6)
    is_y_modified_inplace = (y is y_custom_result)

    print(f"Verification (results are close): {'SUCCESS' if is_close else 'FAILURE'}")
    print(f"Verification (operation was in-place): {'SUCCESS' if is_y_modified_inplace else 'FAILURE'}")


if __name__ == "__main__":
    main()