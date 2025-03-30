import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from linear import TritonLinear, LinearFunction
import triton

def test_large_batch_backward():
    """
    Test the backward pass of TritonLinear with large batch sizes
    """
    print("Testing backward pass with large batch sizes...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16

    # Test configuration with large batch sizes
    batch_sizes = [256, 512, 1024, 2048]
    input_dims = [128, 256, 512]
    output_dims = [64, 128, 256]

    # For each test configuration
    for batch_size in batch_sizes:
        for in_features in input_dims:
            for out_features in output_dims:
                print(f"\nTesting shape: batch_size={batch_size}, in_features={in_features}, out_features={out_features}")

                # Create inputs and modules
                x = torch.randn((batch_size, in_features), device=device, dtype=dtype)
                triton_linear = TritonLinear(in_features, out_features, device=device, dtype=dtype)
                torch_linear = nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)

                # Copy weights for fair comparison
                torch_linear.weight.data.copy_(triton_linear.weight.data)

                # Create random gradients
                dy = torch.randn((batch_size, out_features), device=device, dtype=dtype)

                # Forward pass
                triton_output = LinearFunction.apply(x, triton_linear.weight)
                torch_output = torch_linear(x)

                # Check forward pass
                forward_match = torch.allclose(triton_output, torch_output, atol=1e-2, rtol=1e-2)
                print(f"  Forward pass {'✅ passed' if forward_match else '❌ failed'}")

                # Backward pass (using Triton's custom backward)
                ctx = type('DummyContext', (), {"saved_tensors": (x, triton_linear.weight)})
                triton_dx, triton_dw = LinearFunction.backward(ctx, dy)

                # Backward pass (using PyTorch's autograd)
                x_torch = x.clone().detach().requires_grad_()
                torch_out = torch_linear(x_torch)
                torch_out.backward(dy)
                torch_dx = x_torch.grad
                torch_dw = torch_linear.weight.grad

                # Check grad with respect to input
                dx_match = torch.allclose(triton_dx, torch_dx, atol=1e-2, rtol=1e-2)
                print(f"  Backward dx {'✅ passed' if dx_match else '❌ failed'}")

                # Check grad with respect to weight
                dw_match = torch.allclose(triton_dw, torch_dw, atol=1e-2, rtol=1e-2)
                print(f"  Backward dw {'✅ passed' if dw_match else '❌ failed'}")

                # Additional diagnostics if there's a mismatch
                if not dx_match:
                    max_diff = torch.max(torch.abs(triton_dx - torch_dx))
                    relative_diff = torch.norm(triton_dx - torch_dx) / torch.norm(torch_dx)
                    print(f"  dx max diff: {max_diff:.6f}, relative diff: {relative_diff:.6f}")

                if not dw_match:
                    max_diff = torch.max(torch.abs(triton_dw - torch_dw))
                    relative_diff = torch.norm(triton_dw - torch_dw) / torch.norm(torch_dw)
                    print(f"  dw max diff: {max_diff:.6f}, relative diff: {relative_diff:.6f}")

if __name__ == '__main__':
    test_large_batch_backward()
