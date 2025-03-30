import torch
import numpy as np
from triton_rmsnorm import rmsnorm_triton, rmsnorm_torch


def test_rmsnorm():
    # Test configurations
    batch_dims = [1, 8, 32, 256]
    seq_dims = [32, 64, 128, 256, 512, 1024]
    hidden_dims = [64, 128, 256, 512, 1024, 2048]
    dimensions = [batch_dims, seq_dims, hidden_dims]

    def input_generator(batch_size, seq_len, hidden_dim, device='cuda', dtype=torch.float16):
        # Generate random input and weight
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)
        weight = torch.randn(hidden_dim, device=device, dtype=dtype)
        return x, weight

    from benchmark import verify_correctness_func, benchmark_performance_func

    print("Running correctness test...")
    correctness = verify_correctness_func(
        rmsnorm_triton,
        rmsnorm_torch,
        input_generator,
        dimensions,
        device='cuda',
        dtype=torch.float16,
        atol=1e-2,
        rtol=1e-2
    )
    print(f"Correctness tests: {correctness['passed']}/{correctness['total']} passed")

    print("\nRunning performance benchmark...")
    triton_time, torch_time, speedup, perf = benchmark_performance_func(
        rmsnorm_triton,
        rmsnorm_torch,
        input_generator,
        dimensions,
        device='cuda',
        dtype=torch.float16
    )
    print(f"\nMedian times:\n  Triton: {triton_time:.2f} us\n  PyTorch: {torch_time:.2f} us\n  Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    test_rmsnorm()
