"""Benchmark the fast Metal scan against the pure-MLX reference.

Run on Apple Silicon:

    python benchmark_reuse_scan.py

The shapes are representative for RE-USE after the convolutional encoder. Exact
lengths vary with sample rate/chunk size.
"""

from __future__ import annotations

import time

import mlx.core as mx

from mlx_reuse_fast_scan import selective_scan_metal, selective_scan_reference


def _case(
    name: str, batch: int, dim: int, length: int, dstate: int, iters: int = 100
) -> None:
    mx.random.seed(0)
    u = mx.random.normal((batch, dim, length), dtype=mx.float32)
    delta = mx.random.normal((batch, dim, length), dtype=mx.float32) * 0.2
    A = -mx.exp(mx.random.normal((dim, dstate), dtype=mx.float32) * 0.1)
    B = mx.random.normal((batch, dstate, length), dtype=mx.float32) * 0.1
    C = mx.random.normal((batch, dstate, length), dtype=mx.float32) * 0.1
    D = mx.random.normal((dim,), dtype=mx.float32) * 0.1
    z = mx.random.normal((batch, dim, length), dtype=mx.float32)
    delta_bias = mx.random.normal((dim,), dtype=mx.float32) * 0.1

    y_ref = selective_scan_reference(u, delta, A, B, C, D, z, delta_bias, True)
    y_fast = selective_scan_metal(u, delta, A, B, C, D, z, delta_bias, True)
    mx.eval(y_ref, y_fast)
    max_err = mx.max(mx.abs(y_ref - y_fast))
    print(f"{name}: max error = {max_err}")

    # Warmup/JIT.
    for _ in range(5):
        mx.eval(selective_scan_metal(u, delta, A, B, C, D, z, delta_bias, True))

    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(selective_scan_metal(u, delta, A, B, C, D, z, delta_bias, True))
    t1 = time.perf_counter()

    # Reference is much slower; fewer iterations.
    ref_iters = max(5, iters // 10)
    t2 = time.perf_counter()
    for _ in range(ref_iters):
        mx.eval(selective_scan_reference(u, delta, A, B, C, D, z, delta_bias, True))
    t3 = time.perf_counter()

    fast_ms = (t1 - t0) / iters * 1e3
    ref_ms = (t3 - t2) / ref_iters * 1e3
    print(
        f"{name}: metal={fast_ms:.3f} ms, reference={ref_ms:.3f} ms, speedup={ref_ms / fast_ms:.1f}x"
    )


def main() -> None:
    # RE-USE default hidden size = 64, expand = 4 => d_inner = 256, d_state = 16.
    _case("time-like", batch=82, dim=256, length=51, dstate=16)
    _case("freq-like", batch=51, dim=256, length=82, dstate=16)


if __name__ == "__main__":
    main()
