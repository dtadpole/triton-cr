import torch
import torch.nn as nn
import triton
import triton.language as tl
import math
from typing import Optional


# 1. Triton kernel for the forward pass (matrix multiplication)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _linear_forward_kernel(
    # Pointers to matrices
    x_ptr, w_ptr, out_ptr,
    # Matrix dimensions
    M, N, K,
    # Matrix strides
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    # Group sizes
    GROUP_SIZE_M: tl.constexpr,
    # Precision
    ACC_TYPE: tl.constexpr,
):
    """
    Linear layer forward pass: out = x @ w.t()

    Parameters:
        x_ptr: pointer to the input tensor (M, K)
        w_ptr: pointer to the weight tensor (N, K) - transposed during computation
        out_ptr: pointer to the output tensor (M, N)
        M: batch dimension of x
        N: output dimension
        K: input dimension
        strides: strides for each tensor dimension
    """
    # Program ID and compute grid indices
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block starting indices
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N

    # Create ranges for indices
    offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = n_start + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Compute pointers for this block
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    # Transpose the weights matrix for linear layer: transpose(w) => K * N
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    # Initialize accumulator with zeros
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACC_TYPE)

    # Iterate to compute the matrix multiplication
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute mask for boundary checks
        k_offset = k * BLOCK_SIZE_K

        # Load data with masks for boundary checks
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] + k_offset < K)
        w_mask = (offs_k[:, None] + k_offset < K) & (offs_n[None, :] < N)

        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Compute matrix multiplication for this block
        acc += tl.dot(x_block, w_block)

        # Move pointers to the next k block
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk

    # Write result with boundary checks
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc, mask=out_mask)


# 2. Triton kernel for the backward pass with respect to input (dL/dx = dL/dy @ w)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _linear_backward_input_kernel(
    # Pointers to matrices
    grad_y_ptr, w_ptr, grad_x_ptr,
    # Matrix dimensions
    M, K, N,  # Note: K and N are swapped compared to forward (K is output dim, N is input dim)
    # Matrix strides
    stride_ym, stride_yk,
    stride_wk, stride_wn,
    stride_xm, stride_xn,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    # Group sizes
    GROUP_SIZE_M: tl.constexpr,
    # Precision
    ACC_TYPE: tl.constexpr,
):
    """
    Backward pass for linear layer with respect to input (dL/dx = dL/dy @ w)

    Parameters:
        grad_y_ptr: pointer to the gradient tensor from output (M, K)
        w_ptr: pointer to the weight tensor (K, N)
        grad_x_ptr: pointer to the output gradient tensor (M, N)
        M: batch dimension
        K: output dimension
        N: input dimension
    """
    # Program ID and compute grid indices
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block starting indices
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N

    # Create ranges for indices
    offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = n_start + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator with zeros
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACC_TYPE)

    # Iterate to compute the matrix multiplication
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute offset for k dimension
        k_offset = k * BLOCK_SIZE_K

        # Load grad_y (M, K)
        grad_y_ptrs = grad_y_ptr + (offs_m[:, None] * stride_ym + (k_offset + offs_k[None, :]) * stride_yk)
        grad_y_mask = (offs_m[:, None] < M) & ((k_offset + offs_k[None, :]) < K)
        grad_y = tl.load(grad_y_ptrs, mask=grad_y_mask, other=0.0)

        # Load w (K, N) - need to transpose when multiplying
        w_ptrs = w_ptr + ((k_offset + offs_k)[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        w_mask = ((k_offset + offs_k)[:, None] < K) & (offs_n[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Compute matrix multiplication for this block
        acc += tl.dot(grad_y, w)

        # Move pointers to the next k block
        # Already handled by using k_offset in the pointer calculations

    # Write result with boundary checks
    x_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    grad_x_ptrs = grad_x_ptr + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
    tl.store(grad_x_ptrs, acc, mask=x_mask)


# 3. Triton kernel for the backward pass with respect to weights (dL/dw = x.t() @ dL/dy)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _linear_backward_weight_kernel(
    # Pointers to matrices
    x_ptr, grad_y_ptr, grad_w_ptr,
    # Matrix dimensions
    N, K, M,  # In backward: N=in_features, K=out_features, M=batch_size
    # Matrix strides
    stride_xm, stride_xn,
    stride_ym, stride_yk,
    stride_wk, stride_wn,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    # Group sizes
    GROUP_SIZE_M: tl.constexpr,
    # Precision
    ACC_TYPE: tl.constexpr,
):
    """
    Backward pass for linear layer with respect to weights (dL/dw = x.t() @ dL/dy)

    Parameters:
        x_ptr: pointer to the input tensor (M, N)
        grad_y_ptr: pointer to the gradient tensor from output (M, K)
        grad_w_ptr: pointer to the output gradient tensor for weights (K, N)
        N: input dimension
        K: output dimension
        M: batch dimension
    """
    # Program ID and compute grid indices
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(K, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block starting indices
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N

    # Create ranges for indices
    offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)  # output dimension (K)
    offs_n = n_start + tl.arange(0, BLOCK_SIZE_N)  # input dimension (N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)            # batch dimension (M)

    # Initialize accumulator with zeros - for grad_w (K, N)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACC_TYPE)

    # Iterate over batch dimension to compute the matrix multiplication
    for k in range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        # Compute offset for k dimension
        k_offset = k * BLOCK_SIZE_K

        # Load grad_y (M, K) - we need rows for batch dimension, cols for output dimension
        # grad_y[batch_idx, out_idx] - need slice of (BLOCK_SIZE_K, BLOCK_SIZE_M)
        grad_y_ptrs = grad_y_ptr + ((k_offset + offs_k)[:, None] * stride_ym + offs_m[None, :] * stride_yk)
        grad_y_mask = ((k_offset + offs_k)[:, None] < M) & (offs_m[None, :] < K)
        grad_y = tl.load(grad_y_ptrs, mask=grad_y_mask, other=0.0)

        # Load x (M, N) - we need rows for batch dimension, cols for input dimension
        # x[batch_idx, in_idx] - need slice of (BLOCK_SIZE_K, BLOCK_SIZE_N)
        x_ptrs = x_ptr + ((k_offset + offs_k)[:, None] * stride_xm + offs_n[None, :] * stride_xn)
        x_mask = ((k_offset + offs_k)[:, None] < M) & (offs_n[None, :] < N)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Compute grad_w += grad_y.t() @ x
        # Need: (BLOCK_SIZE_M, BLOCK_SIZE_K) @ (BLOCK_SIZE_K, BLOCK_SIZE_N) -> (BLOCK_SIZE_M, BLOCK_SIZE_N)
        acc += tl.dot(tl.trans(grad_y), x)

    # Write result with boundary checks for grad_w (K, N)
    grad_w_mask = (offs_m[:, None] < K) & (offs_n[None, :] < N)
    grad_w_ptrs = grad_w_ptr + (offs_m[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    tl.store(grad_w_ptrs, acc, mask=grad_w_mask)


# Autograd function for the Triton Linear layer
class TritonLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)

        # Get dimensions
        batch_dim = x.shape[:-1]  # Handle arbitrary batch dimensions
        batch_size = x.numel() // x.size(-1) if x.dim() > 1 else 1
        in_features = x.size(-1)
        out_features = weight.size(0)

        # Reshape x to 2D (batch_size, in_features) if needed
        x_2d = x.view(-1, in_features)

        # Allocate output tensor
        output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)

        # Choose accumulation precision
        acc_type = tl.float32 if x.dtype == torch.float16 else tl.float32

        # Launch the kernel
        grid = lambda META: (triton.cdiv(batch_size, META['BLOCK_SIZE_M']) * triton.cdiv(out_features, META['BLOCK_SIZE_N']),)
        _linear_forward_kernel[grid](
            x_2d, weight, output,
            batch_size, out_features, in_features,
            x_2d.stride(0), x_2d.stride(1),
            weight.stride(1), weight.stride(0),  # Transposed access for weights
            output.stride(0), output.stride(1),
            ACC_TYPE=acc_type,
        )

        # Reshape output back to match input's batch dimensions
        if len(batch_dim) > 0:
            output = output.view(*batch_dim, out_features)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors

        # Get dimensions
        batch_dim = x.shape[:-1]  # Batch dimensions
        batch_size = x.numel() // x.size(-1) if x.dim() > 1 else 1
        in_features = x.size(-1)
        out_features = weight.size(0)

        # Reshape to 2D
        x_2d = x.reshape(-1, in_features)
        grad_output_2d = grad_output.reshape(-1, out_features)

        # Allocate memory for gradients
        grad_x = torch.empty_like(x_2d)
        grad_weight = torch.zeros_like(weight)

        # Choose accumulation precision
        acc_type = tl.float32 if x.dtype == torch.float16 else tl.float32

        # 1. Compute gradient w.r.t. input (dL/dx = dL/dy @ w)
        grid_dx = lambda META: (triton.cdiv(batch_size, META['BLOCK_SIZE_M']) * triton.cdiv(in_features, META['BLOCK_SIZE_N']),)
        _linear_backward_input_kernel[grid_dx](
            grad_output_2d, weight, grad_x,
            batch_size, out_features, in_features,
            grad_output_2d.stride(0), grad_output_2d.stride(1),
            weight.stride(0), weight.stride(1),
            grad_x.stride(0), grad_x.stride(1),
            ACC_TYPE=acc_type,
        )

        # 2. Compute gradient w.r.t. weights (dL/dw = x.t() @ dL/dy)
        grid_dw = lambda META: (triton.cdiv(out_features, META['BLOCK_SIZE_M']) * triton.cdiv(in_features, META['BLOCK_SIZE_N']),)
        _linear_backward_weight_kernel[grid_dw](
            x_2d, grad_output_2d, grad_weight,
            in_features, out_features, batch_size,
            x_2d.stride(0), x_2d.stride(1),
            grad_output_2d.stride(0), grad_output_2d.stride(1),
            grad_weight.stride(0), grad_weight.stride(1),
            ACC_TYPE=acc_type,
        )

        # Reshape grad_x back to match input's batch dimensions
        if len(batch_dim) > 0:
            grad_x = grad_x.view(*batch_dim, in_features)

        return grad_x, grad_weight


# Updated Linear layer using Triton kernel
class TritonLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights - similar to how nn.Linear initializes
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        # No bias support as requested
        self.bias = None

        # Initialize weights using the same method as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return TritonLinearFunction.apply(x, self.weight)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
