import torch
import triton
import triton.language as tl


# Create auto-tuned configurations
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
def _matmul_kernel_autotune(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Matrix strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    # Group sizes
    GROUP_SIZE_M: tl.constexpr,
    # Precision
    ACC_TYPE: tl.constexpr,
):
    """
    Auto-tuned version of the matrix multiplication kernel
    """
    # Program ID
    # pid = tl.program_id(axis=0)
    # num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    # num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # group_id = pid // num_pid_in_group
    # first_pid_m = group_id * GROUP_SIZE_M
    # group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # # pid_m = first_pid_m + (pid % group_size_m)
    # pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    # pid_n = (pid % num_pid_in_group) // group_size_m

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
    offs_am = m_start + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = n_start + tl.arange(0, BLOCK_SIZE_N)

    # Compute offset for k dimension
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Compute pointers for loading blocks from A and B
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator with zeros
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACC_TYPE)

    # Iterate to compute the matmul blockwise
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):

        # Compute mask conditions to handle boundaries
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_bn[None, :] < N)

        # Load data with boundary condition masks
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate using dot product
        acc += tl.dot(a, b)

        # Increment pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Write back result with boundary checks
    offs_cm = m_start + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_start + tl.arange(0, BLOCK_SIZE_N)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=c_mask)


# Define a PyTorch function that uses the triton kernel
def matmul(a, b):
    # Check constraints
    assert a.device.type == 'cuda' and b.device.type == 'cuda'
    assert a.dim() == 2 and b.dim() == 2, "Only 2D matrices are supported"
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, f"Incompatible dimensions: {a.shape} and {b.shape}"

    # Convert inputs to the same dtype if needed
    if a.dtype != b.dtype:
        b = b.to(a.dtype)

    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Choose accumulation precision based on input
    acc_type = tl.float32 if a.dtype == torch.float16 else tl.float32

    # Launch the kernel with autotuned parameters
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    _matmul_kernel_autotune[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACC_TYPE=acc_type,
    )
    return c
