#!/usr/bin/env python3
"""Large GPU matmul smoke test for LithOS using TensorFlow."""

import importlib.util
import os
import sys
import time


def main() -> int:
    if importlib.util.find_spec("tensorflow") is None:
        print("[tf-mm] SKIP: tensorflow not installed")
        return 0

    import tensorflow as tf

    print("[tf-mm] version", tf.__version__)
    if not tf.config.list_physical_devices("GPU"):
        print("[tf-mm] FAIL: GPU backend unavailable")
        return 2

    n = int(os.environ.get("LITHOS_MM_TF_N", "4096"))
    warmup = int(os.environ.get("LITHOS_MM_TF_WARMUP", "2"))
    iters = int(os.environ.get("LITHOS_MM_TF_ITERS", "4"))

    with tf.device("/GPU:0"):
        a = tf.random.normal((n, n), dtype=tf.float32)
        b = tf.random.normal((n, n), dtype=tf.float32)

        @tf.function
        def mm(x, y):
            return tf.matmul(x, y)

        for _ in range(warmup):
            _ = mm(a, b)
        _ = tf.reduce_sum(mm(a, b)).numpy()

        start = time.perf_counter()
        for _ in range(iters):
            c = mm(a, b)
        checksum = float(tf.reduce_sum(c).numpy())
        end = time.perf_counter()

    elapsed = end - start
    flops = 2.0 * (n ** 3) * iters
    tflops = flops / elapsed / 1e12

    print(
        f"[tf-mm] PASS n={n} iters={iters} elapsed_s={elapsed:.4f} "
        f"tflops={tflops:.3f} checksum={checksum:.3e}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
