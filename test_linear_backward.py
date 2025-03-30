import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import triton
import triton.testing
from linear import TritonLinear, LinearFunction
from benchmark import verify_correctness_backward, benchmark_performance_backward


class TorchLinear(nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, bias=None)


# Custom backward function for benchmarking TritonLinear
def triton_backward_fn(inputs_dy):
    x, dy = inputs_dy
    # Create a dummy context to satisfy the API
    ctx = type('DummyContext', (), {})
    ctx.saved_tensors = (x, TritonLinear.weight)
    return LinearFunction.backward(ctx, dy)


# Custom backward function for benchmarking TorchLinear
def torch_backward_fn(inputs_dy):
    x, dy = inputs_dy
    # Compute gradients directly using PyTorch operations
    dx = torch.matmul(dy, TorchLinear.weight)
    dw = torch.matmul(dy.transpose(0, 1), x)
    return (dx,), (dw,)


def module_generator(batch_size, in_features, out_features, device='cuda', dtype=torch.float16):
    """
    Generate Triton and PyTorch modules for comparison
    """
    triton_module = TritonLinear(in_features, out_features, device=device, dtype=dtype)
    torch_module = TorchLinear(in_features, out_features, device=device, dtype=dtype)
    return triton_module, torch_module


def backward_generator(batch_size, in_features, out_features, device='cuda', dtype=torch.float16):
    """
    Generate input tensors and gradient tensors for backward pass
    """
    input = torch.randn((batch_size, in_features), device=device, dtype=dtype)
    input = torch.nn.functional.normalize(input, dim=-1)
    grad_output = torch.randn((batch_size, out_features), device=device, dtype=dtype)
    grad_output = torch.nn.functional.normalize(grad_output, dim=-1)
    return input, grad_output


def test_linear_backward():
    """
    Test the backward pass of TritonLinear
    """
    # Full dimensions for correctness test
    correct_batch_dims = [16, 64, 256, 1024]
    correct_input_dims = [128, 512, 2048]
    correct_output_dims = [64, 256, 1024]
    correct_dimensions = [correct_batch_dims, correct_input_dims, correct_output_dims]

    # Small dimensions for benchmark to avoid CUDA graph errors
    bench_batch_dims = [16, 64, 256, 1024]
    bench_input_dims = [128, 512, 2048]
    bench_output_dims = [64, 256, 1024]
    bench_dimensions = [bench_batch_dims, bench_input_dims, bench_output_dims]

    print("Running correctness test (backward)...")
    correctness = verify_correctness_backward(module_generator, backward_generator, correct_dimensions)
    print(f"Correctness tests: {correctness['passed']}/{correctness['total']} passed")

    # Disable autograd for benchmarking to avoid CUDA graph errors
    print("\nRunning performance benchmark (backward)...")
    with torch.set_grad_enabled(False):
        try:
            # Manual benchmarking of backward pass
            from _benchmark_util import _benchmark_implementations, _generate_configs, _plot_benchmark_results
            import json

            configs = _generate_configs(bench_dimensions)
            timings_triton = []
            timings_pytorch = []
            speedups = []
            device_name = torch.cuda.get_device_name(0)

            performance = {
                "speedup": 0.0,
                "device": device_name,
                "results": []
            }

            # Perform manual benchmarking
            for config in configs:
                batch_size, in_features, out_features = config
                # Create sample inputs and gradients
                x = torch.randn((batch_size, in_features), device="cuda", dtype=torch.float16)
                dy = torch.randn((batch_size, out_features), device="cuda", dtype=torch.float16)

                # Initialize sample modules for this config
                triton_module, torch_module = module_generator(*config)

                # Create contexts for backward pass
                triton_ctx = type('DummyContext', (), {"saved_tensors": (x, triton_module.weight)})

                # Benchmark triton backward - focus on dx computation (our fixed kernel)
                dx_triton_time = triton.testing.do_bench(
                    lambda: LinearFunction.backward(triton_ctx, dy)[0],
                    warmup=10,
                    rep=50
                ) * 1e3

                # Benchmark pytorch backward - dx computation
                dx_torch_time = triton.testing.do_bench(
                    lambda: torch.matmul(dy, torch_module.weight),
                    warmup=10,
                    rep=50
                ) * 1e3

                dx_speedup = dx_torch_time / dx_triton_time

                # Track performance metrics
                timings_triton.append(dx_triton_time)
                timings_pytorch.append(dx_torch_time)
                speedups.append(dx_speedup)

                performance["results"].append({
                    "shape": config,
                    "dtype": str(torch.float16),
                    "triton_time": dx_triton_time,
                    "pytorch_time": dx_torch_time,
                    "speedup": dx_speedup
                })

                print(f"Shape: ({config}, torch.float16), Triton: {dx_triton_time:.1f} us, PyTorch: {dx_torch_time:.1f} us, Speedup: {dx_speedup:.2f}x")

            performance["speedup"] = float(np.median(speedups))

            # Write performance results to file
            with open("code.performance.backward_dx.json", "w") as f:
                f.write(json.dumps(performance, indent=4))

            # Generate performance comparison graph
            _plot_benchmark_results(configs, timings_triton, timings_pytorch, speedups, label="backward_dx")

            print(f"\nMedian times:\n  Triton: {np.median(timings_triton):.2f} us\n  PyTorch: {np.median(timings_pytorch):.2f} us\n  Speedup: {np.median(speedups):.2f}x")

        except RuntimeError as e:
            print(f"Error during benchmark: {e}")
            print("Performance benchmarking skipped due to CUDA graph errors.")


if __name__ == '__main__':
    test_linear_backward()
