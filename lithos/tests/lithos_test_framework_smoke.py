#!/usr/bin/env python3
"""Run Torch, JAX, Triton, and TensorFlow large-matmul LithOS tests."""

from pathlib import Path
import subprocess
import sys


def run_case(name: str, script_path: Path) -> int:
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    print(f"[{name}] exit={proc.returncode}")
    return proc.returncode


def torch_case() -> int:
    return run_case("torch", Path(__file__).with_name("lithos_test_torch_large_mm.py"))


def jax_case() -> int:
    return run_case("jax", Path(__file__).with_name("lithos_test_jax_large_mm.py"))


def triton_case() -> int:
    return run_case("triton", Path(__file__).with_name("lithos_test_triton_large_mm.py"))


def tensorflow_case() -> int:
    return run_case("tensorflow", Path(__file__).with_name("lithos_test_tf_large_mm.py"))


def main() -> int:
    failures = 0
    for fn in (torch_case, jax_case, triton_case, tensorflow_case):
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
