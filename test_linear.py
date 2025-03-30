import torch
import torch.nn as nn
import numpy as np
from triton_linear import TritonLinear
from benchmark import verify_correctness_forward, benchmark_performance_forward



def module_generator(batch_size, in_features, out_features, device='cuda', dtype=torch.float16):
    """
    Generate Triton and PyTorch modules for comparison
    """
    triton_module = TritonLinear(in_features, out_features, device=device, dtype=dtype)
    torch_module = nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)
    return triton_module, torch_module


def input_generator(batch_size, in_features, out_features, device='cuda', dtype=torch.float16):
    """
    Generate input tensors for the modules
    """
    result = torch.randn((batch_size, in_features), device=device, dtype=dtype)
    result = torch.nn.functional.normalize(result, dim=-1)
    return result


def test_linear_forward():
    """
    Test the forward pass of TritonLinear
    """
    # Example test configuration
    batch_dims = [16, 64, 256, 1024]
    input_dims = [128, 512, 2048]
    output_dims = [128, 512, 2048]
    dimensions = [batch_dims, input_dims, output_dims]

    print("Running correctness test (forward)...")
    correctness = verify_correctness_forward(module_generator, input_generator, dimensions)
    print(f"Correctness tests: {correctness['passed']}/{correctness['total']} passed")

    print("Running performance benchmark (forward)...")
    triton_time, torch_time, speedup, perf = benchmark_performance_forward(module_generator, input_generator, dimensions)
    print(f"\nMedian times:\n  Triton: {triton_time:.2f} us\n  PyTorch: {torch_time:.2f} us\n  Speedup: {speedup:.2f}x")


if __name__ == '__main__':
    test_linear_forward()
