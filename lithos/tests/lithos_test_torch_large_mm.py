#!/usr/bin/env python3
"""Large CUDA matmul smoke test for LithOS using PyTorch."""

import importlib.util
import os
import sys
import time


def main() -> int:
    if importlib.util.find_spec("torch") is None:
        print("[torch-mm] SKIP: torch not installed")
        return 0

    import torch

    print("[torch-mm] version", torch.__version__)
    if not torch.cuda.is_available():
        print("[torch-mm] FAIL: CUDA unavailable")
        return 2

    n = int(os.environ.get("LITHOS_MM_TORCH_N", "4096"))
    warmup = int(os.environ.get("LITHOS_MM_TORCH_WARMUP", "2"))
    iters = int(os.environ.get("LITHOS_MM_TORCH_ITERS", "4"))

    a = torch.randn((n, n), device="cuda", dtype=torch.float32)
    b = torch.randn((n, n), device="cuda", dtype=torch.float32)

    for _ in range(warmup):
        _ = a @ b
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    end = time.perf_counter()

    checksum = float(c.sum().item())
    elapsed = end - start
    flops = 2.0 * (n ** 3) * iters
    tflops = flops / elapsed / 1e12

    print(
        f"[torch-mm] PASS n={n} iters={iters} elapsed_s={elapsed:.4f} "
        f"tflops={tflops:.3f} checksum={checksum:.3e}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
