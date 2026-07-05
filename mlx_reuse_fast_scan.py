"""
Fast Metal selective-scan patch for the MLX RE-USE / SEMamba runtime.

Target runtime:
    appautomaton/mlx-speech
    mlx_speech.models.reuse.mamba.scan.selective_scan

This replaces the pure-MLX Python-loop selective scan used by that runtime with a
single custom Metal kernel specialized for the real-valued Mamba scan used in
RE-USE / SEMamba:

    u, delta, z: [B, d_inner, L]
    A:           [d_inner, d_state]        # already negative
    B, C:        [B, d_state, L]           # variable/input-dependent
    D:           [d_inner]
    delta_bias:  [d_inner]

The kernel is a fused per-(batch, d_inner) scan. It keeps the d_state vector in
registers and loops over L inside one GPU thread. For RE-USE this is a practical
fast path because d_state is small (16) and the encoded sequence lengths are
moderate; it avoids the reference implementation's Python loop and its large
[B, d_inner, L, d_state] intermediates.

It is intentionally inference-only. It is not a differentiable MLX custom op.
"""

from __future__ import annotations

import importlib
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn


# Filled by install_fast_reuse_scan(). Used only for fallback/debugging.
_ORIGINAL_SELECTIVE_SCAN: Optional[Callable[..., mx.array]] = None


_METAL_SRC = r"""
    uint lane = thread_position_in_grid.x;

    uint dim = uint(DIM);
    uint length = uint(LENGTH);
    uint dstate = uint(DSTATE);

    uint batch = lane / dim;
    uint d = lane - batch * dim;

    float state[DSTATE];
    for (uint n = 0; n < dstate; ++n) {
        state[n] = 0.0f;
    }

    for (uint t = 0; t < length; ++t) {
        uint udt_idx = (batch * dim + d) * length + t;

        float dt = delta[udt_idx];
        if (HAS_DELTA_BIAS) {
            dt += delta_bias[d];
        }

        if (DELTA_SOFTPLUS) {
            // Stable softplus. Matches the usual log(1 + exp(x)) while avoiding
            // overflow for large positive dt and underflow noise for very negative dt.
            if (dt > 20.0f) {
                // softplus(x) ~= x
            } else if (dt < -20.0f) {
                dt = metal::exp(dt);
            } else {
                dt = metal::log(1.0f + metal::exp(dt));
            }
        }

        float u_t = u[udt_idx];
        float acc = 0.0f;

        for (uint n = 0; n < dstate; ++n) {
            uint an_idx = d * dstate + n;
            uint bcn_idx = (batch * dstate + n) * length + t;

            float a_dn = A[an_idx];
            float b_tn = Bvar[bcn_idx];
            float c_tn = Cvar[bcn_idx];

            // Upstream reference:
            //   deltaA  = exp(delta * A)
            //   deltaBu = delta * B * u
            //   x_t     = deltaA * x_{t-1} + deltaBu
            float dA = metal::exp(dt * a_dn);
            float dBu = dt * b_tn * u_t;

            state[n] = dA * state[n] + dBu;
            acc += state[n] * c_tn;
        }

        if (HAS_D) {
            acc += u_t * Dvec[d];
        }

        if (HAS_Z) {
            float z_t = z[udt_idx];
            float gate = z_t / (1.0f + metal::exp(-z_t)); // silu(z)
            acc *= gate;
        }

        out[udt_idx] = acc;
    }
"""


_selective_scan_kernel = mx.fast.metal_kernel(
    name="reuse_semamba_selective_scan_fused",
    input_names=["u", "delta", "A", "Bvar", "Cvar", "Dvec", "z", "delta_bias"],
    output_names=["out"],
    source=_METAL_SRC,
)


def _f32_contiguous(x: mx.array) -> mx.array:
    return mx.contiguous(x.astype(mx.float32))


def selective_scan_reference(
    u: mx.array,
    delta: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array | None = None,
    z: mx.array | None = None,
    delta_bias: mx.array | None = None,
    delta_softplus: bool = True,
) -> mx.array:
    """Pure-MLX fallback matching mlx-speech's reference implementation."""
    u = _f32_contiguous(u)
    delta = _f32_contiguous(delta)
    A = _f32_contiguous(A)
    B = _f32_contiguous(B)
    C = _f32_contiguous(C)

    if delta_bias is not None:
        delta = delta + delta_bias.astype(mx.float32)[None, :, None]
    if delta_softplus:
        delta = nn.softplus(delta)

    batch, dim, length = u.shape
    dstate = A.shape[1]

    deltaA = mx.exp(delta[..., None] * A[None, :, None, :])
    B_t = mx.transpose(B, (0, 2, 1))[:, None, :, :]
    deltaB_u = delta[..., None] * B_t * u[..., None]
    C_t = mx.transpose(C, (0, 2, 1))

    x = mx.zeros((batch, dim, dstate), dtype=mx.float32)
    ys = []
    for i in range(length):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        y = mx.sum(x * C_t[:, None, i, :], axis=-1)
        ys.append(y)
    y = mx.stack(ys, axis=2)

    if D is not None:
        y = y + u * D.astype(mx.float32)[None, :, None]
    if z is not None:
        y = y * nn.silu(_f32_contiguous(z))
    return y


def selective_scan_metal(
    u: mx.array,
    delta: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array | None = None,
    z: mx.array | None = None,
    delta_bias: mx.array | None = None,
    delta_softplus: bool = True,
    *,
    max_register_state: int = 64,
) -> mx.array:
    """Fused Metal selective scan for RE-USE / SEMamba.

    The signature intentionally matches
    `mlx_speech.models.reuse.mamba.scan.selective_scan`, so this function can be
    monkey-patched into the existing runtime.
    """
    if u.ndim != 3 or delta.ndim != 3 or B.ndim != 3 or C.ndim != 3 or A.ndim != 2:
        return selective_scan_reference(
            u, delta, A, B, C, D, z, delta_bias, delta_softplus
        )

    batch, dim, length = u.shape
    if delta.shape != (batch, dim, length):
        raise ValueError(
            f"delta must have shape {(batch, dim, length)}, got {delta.shape}"
        )
    dstate = A.shape[1]
    if A.shape != (dim, dstate):
        raise ValueError(f"A must have shape [d_inner, d_state], got {A.shape}")
    if B.shape != (batch, dstate, length):
        raise ValueError(f"B must have shape {(batch, dstate, length)}, got {B.shape}")
    if C.shape != (batch, dstate, length):
        raise ValueError(f"C must have shape {(batch, dstate, length)}, got {C.shape}")

    # The register-array implementation is intended for small d_state. RE-USE
    # uses d_state=16. Fall back rather than generating a huge per-thread array.
    if dstate <= 0 or dstate > max_register_state:
        return selective_scan_reference(
            u, delta, A, B, C, D, z, delta_bias, delta_softplus
        )

    u = _f32_contiguous(u)
    delta = _f32_contiguous(delta)
    A = _f32_contiguous(A)
    B = _f32_contiguous(B)
    C = _f32_contiguous(C)

    has_d = D is not None
    has_z = z is not None
    has_delta_bias = delta_bias is not None

    Dvec = _f32_contiguous(D) if D is not None else mx.zeros((dim,), dtype=mx.float32)
    zbuf = _f32_contiguous(z) if z is not None else mx.zeros_like(u)
    dbuf = (
        _f32_contiguous(delta_bias)
        if delta_bias is not None
        else mx.zeros((dim,), dtype=mx.float32)
    )

    if Dvec.shape != (dim,):
        raise ValueError(f"D must have shape {(dim,)}, got {Dvec.shape}")
    if zbuf.shape != (batch, dim, length):
        raise ValueError(f"z must have shape {(batch, dim, length)}, got {zbuf.shape}")
    if dbuf.shape != (dim,):
        raise ValueError(f"delta_bias must have shape {(dim,)}, got {dbuf.shape}")

    lanes = int(batch * dim)
    if lanes == 0 or length == 0:
        return mx.zeros_like(u)

    out = _selective_scan_kernel(
        inputs=[u, delta, A, B, C, Dvec, zbuf, dbuf],
        template=[
            ("DIM", int(dim)),
            ("LENGTH", int(length)),
            ("DSTATE", int(dstate)),
            ("HAS_D", bool(has_d)),
            ("HAS_Z", bool(has_z)),
            ("HAS_DELTA_BIAS", bool(has_delta_bias)),
            ("DELTA_SOFTPLUS", bool(delta_softplus)),
        ],
        grid=(lanes, 1, 1),
        threadgroup=(min(256, lanes), 1, 1),
        output_shapes=[u.shape],
        output_dtypes=[mx.float32],
    )[0]
    return out


def install_fast_reuse_scan(*, verbose: bool = True) -> list[str]:
    """Patch mlx-speech's RE-USE Mamba modules to use the Metal scan.

    Call this before constructing/loading the REUSEEnhancer. It patches both the
    scan module and the block module, because block.py imports `selective_scan`
    into its module globals.

    Returns:
        List of patched module attributes.
    """
    global _ORIGINAL_SELECTIVE_SCAN

    patched: list[str] = []

    scan_mod = importlib.import_module("mlx_speech.models.reuse.mamba.scan")
    block_mod = importlib.import_module("mlx_speech.models.reuse.mamba.block")

    if _ORIGINAL_SELECTIVE_SCAN is None:
        _ORIGINAL_SELECTIVE_SCAN = getattr(scan_mod, "selective_scan", None)

    scan_mod.selective_scan = selective_scan_metal
    block_mod.selective_scan = selective_scan_metal
    patched.extend(
        [
            "mlx_speech.models.reuse.mamba.scan.selective_scan",
            "mlx_speech.models.reuse.mamba.block.selective_scan",
        ]
    )

    if verbose:
        print("Installed fast RE-USE selective scan:")
        for name in patched:
            print(f"  - {name}")

    return patched


def uninstall_fast_reuse_scan(*, verbose: bool = True) -> None:
    """Restore the original pure-MLX scan when available."""
    if _ORIGINAL_SELECTIVE_SCAN is None:
        return
    scan_mod = importlib.import_module("mlx_speech.models.reuse.mamba.scan")
    block_mod = importlib.import_module("mlx_speech.models.reuse.mamba.block")
    scan_mod.selective_scan = _ORIGINAL_SELECTIVE_SCAN
    block_mod.selective_scan = _ORIGINAL_SELECTIVE_SCAN
    if verbose:
        print("Restored original RE-USE selective scan")


__all__ = [
    "install_fast_reuse_scan",
    "uninstall_fast_reuse_scan",
    "selective_scan_metal",
    "selective_scan_reference",
]
