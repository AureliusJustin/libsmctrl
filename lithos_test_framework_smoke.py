#!/usr/bin/env python3
"""
Torch + JAX framework smoke tests under the LithOS wrapper.

This script is intentionally lightweight:
- verifies import and CUDA availability
- launches at least one CUDA kernel in each framework
- returns non-zero only on real failures
"""

import importlib.util
import subprocess
import sys


def run_case(name: str, snippet: str) -> int:
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    print(f"[{name}] exit={proc.returncode}")
    return proc.returncode


def torch_case() -> int:
    if importlib.util.find_spec("torch") is None:
        print("[torch] SKIP: not installed")
        return 0
    code = r'''
import sys
import torch
print("[torch] version", torch.__version__)
if not torch.cuda.is_available():
    print("[torch] FAIL: CUDA unavailable")
    sys.exit(2)
x = torch.randn(1_000_000, device="cuda")
y = x * x
_ = y.sum().item()
torch.cuda.synchronize()
print("[torch] PASS")
'''
    return run_case("torch", code)


def jax_case() -> int:
    if importlib.util.find_spec("jax") is None:
        print("[jax] SKIP: not installed")
        return 0
    code = r'''
import sys
import jax
import jax.numpy as jnp
print("[jax] version", jax.__version__)
backend = jax.default_backend()
print("[jax] backend", backend)
if backend != "gpu":
    print("[jax] FAIL: GPU backend unavailable")
    sys.exit(2)
x = jnp.arange(1_000_000, dtype=jnp.float32)
y = jnp.sin(x)
y.block_until_ready()
print("[jax] PASS")
'''
    return run_case("jax", code)


def main() -> int:
    failures = 0
    for fn in (torch_case, jax_case):
        rc = fn()
        if rc != 0:
            failures += 1
    if failures:
        print(f"framework smoke FAILED ({failures} case(s))")
        return 1
    print("framework smoke PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
