import torch
import triton
import triton.language as tl


# Create auto-tuned configurations
@triton.autotune(
    configs=[
        # Configs for batch size 32
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": gm,
            },
            num_stages=ns,
            num_warps=nw,
        )
        for bm in [16, 32, 64, 128]
        for bn in [16, 32, 64, 128]
        for bk in [32, 64, 128]
        for gm in [8]
        for ns in [4, 6]
        for nw in [4]
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_kernel_by_m(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # Matrix strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # Group sizes
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Auto-tuned version of the matrix multiplication kernel
    """
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    pid = tl.program_id(axis=0)
    # pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    # pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)
    # pid_n = pid // tl.cdiv(M, BLOCK_SIZE_M)
    # pid_m = pid % tl.cdiv(M, BLOCK_SIZE_M)

    # Create ranges for indices
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Compute offset for k dimension
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Compute pointers for loading blocks from A and B
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Initialize accumulator with zeros
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate to compute the matmul blockwise
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):

        # Compute mask conditions to handle boundaries
        # a_mask = (offs_m[:, None] < M) & (offs_k[None, :] + k * BLOCK_SIZE_K < K)
        # b_mask = (offs_k[:, None] + k * BLOCK_SIZE_K < K) & (offs_n[None, :] < N)
        a_mask = offs_k[None, :] + k * BLOCK_SIZE_K < K
        b_mask = offs_k[:, None] + k * BLOCK_SIZE_K < K

        # Load data with boundary condition masks
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate using dot product
        acc += tl.dot(a, b)

        # Increment pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    acc = acc.to(a_ptr.type.element_ty)

    # Write back result with boundary checks
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=c_mask)


# Define a PyTorch function that uses the triton kernel
def triton_matmul(a, b):
    assert a.device == b.device, f"Incompatible devices: {a.device} and {b.device}"
    assert a.dtype == b.dtype, f"Incompatible dtypes: {a.dtype} and {b.dtype}"
    assert a.shape[1] == b.shape[0], f"Incompatible dimensions: {a.shape} and {b.shape}"
    M, K = a.shape
    _, N = b.shape

    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Launch the kernel with autotuned parameters
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _matmul_kernel_by_m[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    # if M >= N:
    #     _matmul_kernel_by_m[grid](
    #         a, b, c,
    #         M, N, K,
    #         a.stride(0), a.stride(1),
    #         b.stride(0), b.stride(1),
    #         c.stride(0), c.stride(1),
    #     )
    # else:
    #    _matmul_kernel_by_m[grid](
    #        b, a, c,
    #        N, M, K,
    #        b.stride(1), b.stride(0),
    #        a.stride(1), a.stride(0),
    #        c.stride(1), c.stride(0),
    #    )
    return c
