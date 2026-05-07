#!/usr/bin/env python3
"""Large GPU matmul smoke test for LithOS using Triton."""

import importlib.util
import os
import sys
import time


def main() -> int:
    if importlib.util.find_spec("triton") is None:
        print("[triton-mm] SKIP: triton not installed")
        return 0
    if importlib.util.find_spec("torch") is None:
        print("[triton-mm] SKIP: torch not installed (required by triton)")
        return 0

    import torch
    import triton
    import triton.language as tl

    print("[triton-mm] version", triton.__version__)
    if not torch.cuda.is_available():
        print("[triton-mm] FAIL: CUDA unavailable")
        return 2

    n = int(os.environ.get("LITHOS_MM_TRITON_N", "4096"))
    warmup = int(os.environ.get("LITHOS_MM_TRITON_WARMUP", "2"))
    iters = int(os.environ.get("LITHOS_MM_TRITON_ITERS", "4"))

    block_m = 128
    block_n = 128
    block_k = 32
    num_warps = 8
    num_stages = 4

    @triton.jit
    def matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a_ptrs = a_ptr + (offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak)
            b_ptrs = b_ptr + ((k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn)

            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(a, b)

        c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    def triton_matmul(x: "torch.Tensor", y: "torch.Tensor") -> "torch.Tensor":
        m, k = x.shape
        k2, n2 = y.shape
        if k != k2:
            raise ValueError("incompatible shapes for matmul")

        c = torch.empty((m, n2), device=x.device, dtype=torch.float32)
        grid = (triton.cdiv(m, block_m), triton.cdiv(n2, block_n))
        matmul_kernel[grid](
            x,
            y,
            c,
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            c.stride(0),
            c.stride(1),
            M=m,
            N=n2,
            K=k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return c

    a = torch.randn((n, n), device="cuda", dtype=torch.float16)
    b = torch.randn((n, n), device="cuda", dtype=torch.float16)

    for _ in range(warmup):
        _ = triton_matmul(a, b)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        c = triton_matmul(a, b)
    torch.cuda.synchronize()
    end = time.perf_counter()

    checksum = float(c.sum().item())
    elapsed = end - start
    flops = 2.0 * (n ** 3) * iters
    tflops = flops / elapsed / 1e12

    print(
        f"[triton-mm] PASS n={n} iters={iters} elapsed_s={elapsed:.4f} "
        f"tflops={tflops:.3f} checksum={checksum:.3e}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
