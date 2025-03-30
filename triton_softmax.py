import torch
import triton
import triton.language as tl
import math


def softmax_torch(x):
    """
    Compute softmax using PyTorch.

    Args:
        x: Input tensor of shape [batch_size, seq_len]

    Returns:
        Softmax result of shape [batch_size, seq_len]
    """
    return torch.softmax(x, dim=-1)


@triton.jit
def _softmax_kernel(
    x_ptr, output_ptr,
    stride_x_batch, stride_x_seq,
    stride_out_batch, stride_out_seq,
    batch_size, seq_len,
    BLOCK_SIZE: tl.constexpr
):
    """
    Compute softmax using Triton.

    Args:
        x_ptr: Pointer to input tensor of shape [batch_size, seq_len]
        output_ptr: Pointer to output tensor of shape [batch_size, seq_len]
        stride_x_batch: Stride for batch dimension of input tensor
        stride_x_seq: Stride for sequence dimension of input tensor
        stride_out_batch: Stride for batch dimension of output tensor
        stride_out_seq: Stride for sequence dimension of output tensor
        batch_size: Number of sequences in batch
        seq_len: Length of sequence
        BLOCK_SIZE: Size of block for processing
    """
    # Get program ID
    pid = tl.program_id(0)

    # Check if we're still within batch size
    if pid >= batch_size:
        return

    # Compute the pointer to the row for the current batch
    x_row_ptr = x_ptr + pid * stride_x_batch

    # Compute max value for numerical stability
    row_max = -float('inf')
    for seq_idx in range(0, seq_len, BLOCK_SIZE):
        # Create a block mask to handle boundary conditions
        seq_mask = tl.arange(0, BLOCK_SIZE) < (seq_len - seq_idx)

        # Load input data for this block
        offsets = tl.arange(0, BLOCK_SIZE) * stride_x_seq
        block_data = tl.load(x_row_ptr + seq_idx * stride_x_seq + offsets, mask=seq_mask, other=-float('inf'))

        # Update row_max
        row_max = tl.maximum(row_max, tl.max(block_data, axis=0))

    # Compute sum of exponentials (shifted by row_max for stability)
    row_sum = 0.0
    for seq_idx in range(0, seq_len, BLOCK_SIZE):
        # Create a block mask to handle boundary conditions
        seq_mask = tl.arange(0, BLOCK_SIZE) < (seq_len - seq_idx)

        # Load input data for this block
        offsets = tl.arange(0, BLOCK_SIZE) * stride_x_seq
        block_data = tl.load(x_row_ptr + seq_idx * stride_x_seq + offsets, mask=seq_mask, other=-float('inf'))

        # Compute exponentials (shifted by row_max)
        block_exp = tl.exp(block_data - row_max)

        # Add to sum
        row_sum += tl.sum(block_exp, axis=0)

    # Write normalized exponents to output
    output_row_ptr = output_ptr + pid * stride_out_batch
    for seq_idx in range(0, seq_len, BLOCK_SIZE):
        # Create a block mask to handle boundary conditions
        seq_mask = tl.arange(0, BLOCK_SIZE) < (seq_len - seq_idx)

        # Load input data for this block
        offsets = tl.arange(0, BLOCK_SIZE) * stride_x_seq
        block_data = tl.load(x_row_ptr + seq_idx * stride_x_seq + offsets, mask=seq_mask, other=-float('inf'))

        # Compute softmax
        block_exp = tl.exp(block_data - row_max)
        block_softmax = block_exp / row_sum

        # Write to output
        out_offsets = tl.arange(0, BLOCK_SIZE) * stride_out_seq
        tl.store(output_row_ptr + seq_idx * stride_out_seq + out_offsets, block_softmax, mask=seq_mask)


def softmax_triton(x):
    """
    Compute softmax using Triton.

    Args:
        x: Input tensor of shape [batch_size, seq_len]

    Returns:
        Softmax result of shape [batch_size, seq_len]
    """
    # Extract tensor dimensions
    batch_size, seq_len = x.shape

    # Allocate output tensor
    output = torch.empty_like(x)

    # Calculate strides
    stride_x_batch, stride_x_seq = x.stride()
    stride_out_batch, stride_out_seq = output.stride()

    # Determine block size
    BLOCK_SIZE = min(seq_len, 1024)  # Adjust based on sequence length

    # Launch kernel
    grid = (batch_size,)
    _softmax_kernel[grid](
        x, output,
        stride_x_batch, stride_x_seq,
        stride_out_batch, stride_out_seq,
        batch_size, seq_len,
        BLOCK_SIZE
    )

    return output
