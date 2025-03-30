import torch
import triton
import triton.language as tl
import math


def rmsnorm_torch(x, weight, eps=1e-6):
    """
    Compute RMSNorm using PyTorch.

    Args:
        x: Input tensor of shape [batch_size, seq_len, hidden_dim]
        weight: Scale parameter of shape [hidden_dim]
        eps: Epsilon for numerical stability

    Returns:
        RMSNorm result of shape [batch_size, seq_len, hidden_dim]
    """
    variance = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    return weight * x_norm


@triton.jit
def _rmsnorm_kernel(
    x_ptr, weight_ptr, output_ptr,
    stride_b, stride_s, stride_h,
    stride_weight,
    batch_size, seq_len, hidden_dim,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    """
    Compute RMSNorm using Triton.

    Args:
        x_ptr: Pointer to input tensor of shape [batch_size, seq_len, hidden_dim]
        weight_ptr: Pointer to weight tensor of shape [hidden_dim]
        output_ptr: Pointer to output tensor of shape [batch_size, seq_len, hidden_dim]
        stride_b: Stride for batch dimension of input tensor
        stride_s: Stride for sequence dimension of input tensor
        stride_h: Stride for hidden dimension of input tensor
        stride_weight: Stride for weight tensor
        batch_size: Batch size
        seq_len: Sequence length
        hidden_dim: Hidden dimension size
        eps: Epsilon for numerical stability
        BLOCK_SIZE: Size of hidden dimension block for processing
    """
    # Get program ID for batch and sequence dimensions
    batch_id = tl.program_id(0)
    seq_id = tl.program_id(1)

    # Check if we're within bounds
    if batch_id >= batch_size or seq_id >= seq_len:
        return

    # Compute the pointer to the start of the row for the current batch and sequence
    x_row_ptr = x_ptr + batch_id * stride_b + seq_id * stride_s

    # Compute mean square
    mean_square = 0.0
    for h_idx in range(0, hidden_dim, BLOCK_SIZE):
        # Create a block mask to handle boundary conditions
        h_mask = tl.arange(0, BLOCK_SIZE) < (hidden_dim - h_idx)

        # Load input data for this block
        offsets = tl.arange(0, BLOCK_SIZE) * stride_h
        block_data = tl.load(x_row_ptr + h_idx * stride_h + offsets, mask=h_mask, other=0.0)

        # Add squared values to mean_square
        mean_square += tl.sum(block_data * block_data, axis=0)

    # Compute mean square by dividing by hidden_dim
    mean_square = mean_square / hidden_dim

    # Compute scaling factor
    scaling_factor = 1.0 / tl.sqrt(mean_square + eps)

    # Compute output row pointer
    output_row_ptr = output_ptr + batch_id * stride_b + seq_id * stride_s

    # Normalize and scale with weight
    for h_idx in range(0, hidden_dim, BLOCK_SIZE):
        # Create a block mask to handle boundary conditions
        h_mask = tl.arange(0, BLOCK_SIZE) < (hidden_dim - h_idx)

        # Load input data for this block
        x_offsets = tl.arange(0, BLOCK_SIZE) * stride_h
        block_data = tl.load(x_row_ptr + h_idx * stride_h + x_offsets, mask=h_mask, other=0.0)

        # Load weights for this block
        weight_offsets = tl.arange(0, BLOCK_SIZE) * stride_weight
        block_weight = tl.load(weight_ptr + h_idx * stride_weight + weight_offsets, mask=h_mask, other=0.0)

        # Compute normalized and scaled output
        block_output = block_data * scaling_factor * block_weight

        # Store the result
        output_offsets = tl.arange(0, BLOCK_SIZE) * stride_h
        tl.store(output_row_ptr + h_idx * stride_h + output_offsets, block_output, mask=h_mask)


def rmsnorm_triton(x, weight, eps=1e-6):
    """
    Compute RMSNorm using Triton.

    Args:
        x: Input tensor of shape [batch_size, seq_len, hidden_dim]
        weight: Scale parameter of shape [hidden_dim]
        eps: Epsilon for numerical stability

    Returns:
        RMSNorm result of shape [batch_size, seq_len, hidden_dim]
    """
    # Extract tensor dimensions
    batch_size, seq_len, hidden_dim = x.shape

    # Allocate output tensor
    output = torch.empty_like(x)

    # Get strides
    stride_b, stride_s, stride_h = x.stride()
    stride_weight = weight.stride(0)

    # Determine block size for hidden dimension - this will be auto-tuned
    # For smaller hidden dims, use a smaller block size
    if hidden_dim <= 128:
        block_size = 128
        num_warps = 4
    elif hidden_dim <= 256:
        block_size = 256
        num_warps = 8
    elif hidden_dim <= 512:
        block_size = 512
        num_warps = 16
    else:
        block_size = 1024
        num_warps = 16

    # Launch kernel
    grid = (batch_size, seq_len)
    _rmsnorm_kernel[grid](
        x, weight, output,
        stride_b, stride_s, stride_h,
        stride_weight,
        batch_size, seq_len, hidden_dim,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps
    )

    return output
