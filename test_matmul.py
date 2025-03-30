import torch
import torch.nn.functional as F
import numpy as np

from benchmark import verify_correctness_func, benchmark_performance_func
from triton_matmul import triton_matmul


def torch_matmul(a, b):
    return torch.matmul(a, b)


def input_generator(batch_size, input_dim, output_dim, device='cuda', dtype=torch.float16):
    a = torch.randn(batch_size, input_dim, device=device, dtype=dtype)
    a = F.normalize(a, dim=-1)
    b = torch.randn(input_dim, output_dim, device=device, dtype=dtype)
    b = F.normalize(b, dim=-1)
    return a, b


if __name__ == '__main__':
    print("Testing matrix multiplication implementation...")

    # Define dimensions to test
    batch_sizes = [32, 128, 512, 2048]
    input_dims = [32, 128, 512, 2048]
    output_dims = [32, 128, 512, 2048]
    dimensions = [batch_sizes, input_dims, output_dims]

    # Verify correctness
    print("Verifying correctness...")
    correctness = verify_correctness_func(
        triton_matmul,
        torch_matmul,
        input_generator,
        dimensions,
        device='cuda',
        dtype=torch.bfloat16,
        atol=1e-2,
        rtol=1e-2
    )
    print(f"Correctness tests: {correctness['passed']}/{correctness['total']} passed")

    # Benchmark performance
    print("\nBenchmarking performance...")
    triton_time, torch_time, speedup, perf = benchmark_performance_func(
        triton_matmul,
        torch_matmul,
        input_generator,
        dimensions,
        device='cuda',
        dtype=torch.bfloat16
    )
    print(f"\nMedian times:\n  Triton: {triton_time:.2f} us\n  PyTorch: {torch_time:.2f} us\n  Speedup: {speedup:.2f}x")
