#!/usr/bin/env python3
"""Large GPU matmul smoke test for LithOS using JAX."""

import importlib.util
import os
import sys
import time


def main() -> int:
    if importlib.util.find_spec("jax") is None:
        print("[jax-mm] SKIP: jax not installed")
        return 0

    import jax
    import jax.numpy as jnp

    print("[jax-mm] version", jax.__version__)
    backend = jax.default_backend()
    print("[jax-mm] backend", backend)
    if backend != "gpu":
        print("[jax-mm] FAIL: GPU backend unavailable")
        return 2

    n = int(os.environ.get("LITHOS_MM_JAX_N", "4096"))
    warmup = int(os.environ.get("LITHOS_MM_JAX_WARMUP", "2"))
    iters = int(os.environ.get("LITHOS_MM_JAX_ITERS", "4"))

    key = jax.random.key(0)
    key_a, key_b = jax.random.split(key)
    a = jax.random.normal(key_a, (n, n), dtype=jnp.float32)
    b = jax.random.normal(key_b, (n, n), dtype=jnp.float32)

    mm = jax.jit(lambda x, y: x @ y)

    for _ in range(warmup):
        _ = mm(a, b).block_until_ready()

    start = time.perf_counter()
    c = None
    for _ in range(iters):
        c = mm(a, b)
    assert c is not None
    c.block_until_ready()
    end = time.perf_counter()

    checksum = float(jnp.sum(c).item())
    elapsed = end - start
    flops = 2.0 * (n ** 3) * iters
    tflops = flops / elapsed / 1e12

    print(
        f"[jax-mm] PASS n={n} iters={iters} elapsed_s={elapsed:.4f} "
        f"tflops={tflops:.3f} checksum={checksum:.3e}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
