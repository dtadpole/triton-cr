import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        # Configs for batch size 32
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        # Configs for batch size 64
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        # Configs for batch size 128
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
    ],
    key=['batch_size', 'in_features', 'out_features'],
)
@triton.jit
def linear_forward_kernel(
    # Pointers to matrices
    input_ptr, weight_ptr, output_ptr,
    # Matrix dimensions
    batch_size, in_features, out_features,
    # Strides
    input_batch_stride, input_feature_stride,
    weight_out_stride, weight_in_stride,
    output_batch_stride, output_feature_stride,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute linear layer without bias: output = input @ weight.T

    Parameters:
        input_ptr: pointer to the input tensor (batch_size, in_features)
        weight_ptr: pointer to the weight tensor (out_features, in_features)
        output_ptr: pointer to the output tensor (batch_size, out_features)
        batch_size: number of rows in the input
        in_features: number of columns in the input / weight
        out_features: number of rows in the weight / columns in the output
    """
    # Matrix multiplication (M, K) @ (K, N) -> (M, N)
    # Input: (batch_size, in_features) [M, K]
    # Weight: (out_features, in_features) [N, K]
    # Output: (batch_size, out_features) [M, N]

    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(batch_size, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(out_features, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block start indices
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    # Bounds checking
    tm = start_m + tl.arange(0, BLOCK_SIZE_M)
    tn = start_n + tl.arange(0, BLOCK_SIZE_N)

    # Create row and column masks to handle out-of-bounds accesses
    mask_m = tm < batch_size
    mask_n = tn < out_features

    # Pointers to output blocks
    output_ptrs = output_ptr + tm[:, None] * output_batch_stride + tn[None, :] * output_feature_stride

    # Initialize accumulator with higher precision for numerical stability
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Compute base pointers for input and weight blocks
    input_block_ptr = input_ptr + tm[:, None] * input_batch_stride
    weight_block_ptr = weight_ptr + tn[:, None] * weight_out_stride

    # Process first k block (prologue)
    k = 0
    mask_k = tl.arange(0, BLOCK_SIZE_K) < in_features

    # Load first input and weight blocks
    a = tl.load(
        input_block_ptr + tl.arange(0, BLOCK_SIZE_K)[None, :] * input_feature_stride,
        mask=mask_m[:, None] & mask_k[None, :],
        other=0.0
    )
    b = tl.load(
        weight_block_ptr + tl.arange(0, BLOCK_SIZE_K)[None, :] * weight_in_stride,
        mask=mask_n[:, None] & mask_k[None, :],
        other=0.0
    )

    # Main loop with software pipelining
    for k in range(BLOCK_SIZE_K, in_features, BLOCK_SIZE_K):
        # Update mask for the next K block
        mask_k = tl.arange(0, BLOCK_SIZE_K) + k < in_features

        # Compute dot product with current blocks while loading next blocks
        acc += tl.dot(a, tl.trans(b))

        # Load next blocks (prefetching)
        a = tl.load(
            input_block_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[None, :]) * input_feature_stride,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        )
        b = tl.load(
            weight_block_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[None, :]) * weight_in_stride,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0
        )

    # Process final blocks (epilogue)
    acc += tl.dot(a, tl.trans(b))

    # Store the result
    tl.store(output_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# Backward pass kernel for computing gradients with respect to input (dx)
@triton.autotune(
    configs=[
        # Configs for batch size 32
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        # Configs for batch size 64
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        # Configs for batch size 128
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
    ],
    key=['batch_size', 'in_features', 'out_features'],
)
@triton.jit
def linear_backward_dx_kernel(
    # Pointers to matrices
    grad_output_ptr, weight_ptr, grad_input_ptr,
    # Matrix dimensions
    batch_size, out_features, in_features,
    # Strides
    grad_output_batch_stride, grad_output_feature_stride,
    weight_out_stride, weight_in_stride,
    grad_input_batch_stride, grad_input_feature_stride,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute gradient with respect to input: dx = dy @ weight
    """
    # Program ID
    pid = tl.program_id(axis=0)

    # Number of programs needed for each dimension
    num_pid_m = tl.cdiv(batch_size, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(in_features, BLOCK_SIZE_N)

    # Compute group information for better L2 cache efficiency
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    # Compute the program ID within the group
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block start indices
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    # Create block indices
    tm = start_m + tl.arange(0, BLOCK_SIZE_M)
    tn = start_n + tl.arange(0, BLOCK_SIZE_N)

    # Create masks for boundary checking
    mask_m = tm < batch_size
    mask_n = tn < in_features

    # Initialize accumulator with higher precision for numerical stability
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Pointer to output (grad_input)
    grad_input_ptrs = grad_input_ptr + tm[:, None] * grad_input_batch_stride + tn[None, :] * grad_input_feature_stride

    # Loop over k dimension (output_features) in blocks
    for k in range(0, out_features, BLOCK_SIZE_K):
        # Create block indices and mask for k dimension
        tk = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = tk < out_features

        # Load grad_output block (M, K)
        grad_output_ptrs = grad_output_ptr + tm[:, None] * grad_output_batch_stride + tk[None, :] * grad_output_feature_stride
        a = tl.load(grad_output_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load weight block (K, N)
        weight_ptrs = weight_ptr + tk[:, None] * weight_out_stride + tn[None, :] * weight_in_stride
        b = tl.load(weight_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Matrix multiplication
        acc += tl.dot(a, b)

    # Store result
    tl.store(grad_input_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# Backward pass kernel for computing gradients with respect to weights (dw)
@triton.autotune(
    configs=[
        # Configs for batch size 32
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        # Configs for batch size 64
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        # Configs for batch size 128
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
    ],
    key=['out_features', 'in_features', 'batch_size'],
)
@triton.jit
def linear_backward_dw_kernel(
    # Pointers to matrices
    grad_output_ptr, input_ptr, grad_weight_ptr,
    # Matrix dimensions
    out_features, in_features, batch_size,
    # Strides
    grad_output_batch_stride, grad_output_feature_stride,
    input_batch_stride, input_feature_stride,
    grad_weight_out_stride, grad_weight_in_stride,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute gradient with respect to weights: dw = dy^T @ x
    """
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(out_features, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(in_features, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block start indices
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    # Bounds checking
    tm = start_m + tl.arange(0, BLOCK_SIZE_M)
    tn = start_n + tl.arange(0, BLOCK_SIZE_N)

    # Create row and column masks to handle out-of-bounds accesses
    mask_m = tm < out_features
    mask_n = tn < in_features

    # Pointers to output
    grad_weight_ptrs = grad_weight_ptr + tm[:, None] * grad_weight_out_stride + tn[None, :] * grad_weight_in_stride

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over k dimension (batch size)
    for k in range(0, batch_size, BLOCK_SIZE_K):
        # Calculate the batch indices for this block
        tk = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = tk < batch_size

        # Load blocks for this batch segment
        # Load grad_output for these batch indices and all the output features we're computing
        a_ptrs = grad_output_ptr + tk[:, None] * grad_output_batch_stride + tm[None, :] * grad_output_feature_stride
        a = tl.load(a_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)

        # Load inputs for these batch indices and all the input features we're computing
        b_ptrs = input_ptr + tk[:, None] * input_batch_stride + tn[None, :] * input_feature_stride
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Compute the contribution to the weight gradients from this batch segment
        # We need a transposed version of grad_output so we get the right dimensions for matrix multiply
        acc += tl.dot(tl.trans(a), b)

    # Store the result
    tl.store(grad_weight_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        batch_size = x.shape[0]
        in_features = x.shape[1]
        out_features = weight.shape[0]

        # For smaller batch sizes, continue using our optimized Triton implementation
        # Prepare output tensor
        output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)

        # Get tensor strides for input, weights and output
        input_batch_stride = x.stride(0)
        input_feature_stride = x.stride(1)

        weight_out_stride = weight.stride(0)
        weight_in_stride = weight.stride(1)

        output_batch_stride = output.stride(0)
        output_feature_stride = output.stride(1)

        grid = lambda META: (triton.cdiv(batch_size, META['BLOCK_SIZE_M']) * triton.cdiv(out_features, META['BLOCK_SIZE_N']), )

        # Launch kernel with calculated grid size
        linear_forward_kernel[grid](
            x, weight, output,
            batch_size, in_features, out_features,
            input_batch_stride, input_feature_stride,
            weight_out_stride, weight_in_stride,
            output_batch_stride, output_feature_stride,
        )

        # Save input and weight for backward pass
        ctx.save_for_backward(x, weight)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        x, weight = ctx.saved_tensors

        # Get tensor dimensions
        batch_size = x.shape[0]
        in_features = x.shape[1]
        out_features = weight.shape[0]

        # For very large batch sizes, use PyTorch's native implementation
        # which can be more efficient due to CUBLAS optimizations
        if batch_size >= 2048:
            grad_input = torch.matmul(grad_output, weight)
            grad_weight = torch.matmul(grad_output.transpose(0, 1), x)
            return grad_input, grad_weight

        # For large batch sizes, use a hybrid approach for backward dx
        if batch_size >= 512:
            # Use PyTorch's implementation for dx
            grad_input = torch.matmul(grad_output, weight)

            # Use Triton for dw calculation for large output dimensions
            if out_features >= 512:
                # Use our optimized Triton kernel for dw
                grad_weight = torch.zeros((out_features, in_features), device=x.device, dtype=weight.dtype)

                # Get tensor strides
                grad_output_batch_stride = grad_output.stride(0)
                grad_output_feature_stride = grad_output.stride(1)

                input_batch_stride = x.stride(0)
                input_feature_stride = x.stride(1)

                grad_weight_out_stride = grad_weight.stride(0)
                grad_weight_in_stride = grad_weight.stride(1)

                # Efficient grid size for dw computation
                grid_size = (triton.cdiv(out_features, 64) * triton.cdiv(in_features, 64),)

                # Calculate gradient with respect to weights using Triton
                linear_backward_dw_kernel[grid_size](
                    grad_output, x, grad_weight,
                    out_features, in_features, batch_size,
                    grad_output_batch_stride, grad_output_feature_stride,
                    input_batch_stride, input_feature_stride,
                    grad_weight_out_stride, grad_weight_in_stride,
                )
            else:
                # Use PyTorch for smaller output dimensions
                grad_weight = torch.matmul(grad_output.transpose(0, 1), x)

            return grad_input, grad_weight

        # For smaller batch sizes, use our Triton kernel for dx
        # Prepare gradient tensors with correct shapes
        grad_input = torch.zeros((batch_size, in_features), device=x.device, dtype=x.dtype)

        # Get tensor strides
        grad_output_batch_stride = grad_output.stride(0)
        grad_output_feature_stride = grad_output.stride(1)

        weight_out_stride = weight.stride(0)
        weight_in_stride = weight.stride(1)

        grad_input_batch_stride = grad_input.stride(0)
        grad_input_feature_stride = grad_input.stride(1)

        # Calculate grid size
        grid_size = (triton.cdiv(batch_size, 64) * triton.cdiv(in_features, 64),)

        # Calculate gradient with respect to input using Triton kernel
        linear_backward_dx_kernel[grid_size](
            grad_output, weight, grad_input,
            batch_size, out_features, in_features,
            grad_output_batch_stride, grad_output_feature_stride,
            weight_out_stride, weight_in_stride,
            grad_input_batch_stride, grad_input_feature_stride,
        )

        # Calculate gradient with respect to weights using PyTorch's matmul
        grad_weight = torch.matmul(grad_output.transpose(0, 1), x)

        return grad_input, grad_weight


class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, device='cuda', dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return LinearFunction.apply(x, self.weight)
