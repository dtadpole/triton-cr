import torch
import triton
from triton_linear import TritonLinear
import time

def main():
    # Test with large batch sizes
    batch_sizes = [512, 1024, 2048]
    feature_dims = [64, 256, 1024]

    print('Testing forward pass with large batch sizes')
    print('Batch Size | Features | Output | Triton (ms) | PyTorch (ms) | Speedup')
    print('-' * 70)

    total_speedup = 0
    test_count = 0

    try:
        for batch_size in batch_sizes:
            for in_features in feature_dims:
                for out_features in feature_dims:
                    try:
                        # Create inputs and models
                        x = torch.randn((batch_size, in_features), device='cuda', dtype=torch.float16)

                        # Create models
                        triton_linear = TritonLinear(in_features, out_features, device='cuda', dtype=torch.float16)
                        torch_linear = torch.nn.Linear(in_features, out_features, bias=False, device='cuda', dtype=torch.float16)
                        torch_linear.weight.data.copy_(triton_linear.weight.data)

                        # Warm-up
                        for _ in range(10):
                            triton_out = triton_linear(x)
                            torch_out = torch_linear(x)

                        # Check correctness
                        assert torch.allclose(triton_out, torch_out, rtol=1e-2, atol=1e-2), "Outputs don't match"

                        # Benchmark Triton
                        torch.cuda.synchronize()
                        start = time.time()
                        for _ in range(100):
                            triton_out = triton_linear(x)
                        torch.cuda.synchronize()
                        triton_time = (time.time() - start) * 1000 / 100  # ms

                        # Benchmark PyTorch
                        torch.cuda.synchronize()
                        start = time.time()
                        for _ in range(100):
                            torch_out = torch_linear(x)
                        torch.cuda.synchronize()
                        torch_time = (time.time() - start) * 1000 / 100  # ms

                        speedup = torch_time / triton_time
                        total_speedup += speedup
                        test_count += 1

                        print(f'{batch_size:9d} | {in_features:8d} | {out_features:6d} | {triton_time:10.4f} | {torch_time:11.4f} | {speedup:7.4f}')
                    except Exception as e:
                        print(f"Error testing batch={batch_size}, in={in_features}, out={out_features}: {str(e)}")

        if test_count > 0:
            avg_speedup = total_speedup / test_count
            print(f"\nAverage speedup across all tests: {avg_speedup:.4f}x")

    except Exception as e:
        print(f"Error during testing: {str(e)}")

if __name__ == "__main__":
    main()
