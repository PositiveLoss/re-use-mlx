# Fast RE-USE / SEMamba inference on Apple Silicon with MLX + Metal

This package patches the pure-MLX RE-USE / SEMamba runtime from `mlx-speech` so
its Mamba selective scan runs in one custom Metal kernel instead of a Python loop.

The patch targets the scan used by RE-USE:

```text
u, delta, z: [B, d_inner, L]
A:           [d_inner, d_state]
B, C:        [B, d_state, L]
D:           [d_inner]
delta_bias:  [d_inner]
```

It is a fused per-`(batch, d_inner)` scan. It keeps the `d_state` vector in
registers and loops over `L` inside the GPU thread. This is usually the right
trade-off for RE-USE because `d_state=16` and the encoded time/frequency lengths
are moderate.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install mlx mlx-speech soundfile numpy 'huggingface_hub[hf_xet]'
```

## Download weights

Faraday weights:

```bash
huggingface-cli download --local-dir re-use-mlx faraday/re-use-mlx
```

or App Automaton's Python MLX runtime weights:

```bash
huggingface-cli download --local-dir reuse-semamba-mlx appautomaton/re-use-semamba-mlx
```

## Run inference

```bash
python fast_reuse_inference.py noisy.wav clean.wav --weights re-use-mlx
```

The script accepts either:

- a directory containing `model.safetensors`
- a directory containing Faraday's `model_mlx.safetensors`
- a direct `.safetensors` file path

Automatic download:

```bash
python fast_reuse_inference.py noisy.wav clean.wav --repo faraday/re-use-mlx
```

## Benchmark only the scan

```bash
python benchmark_reuse_scan.py
```

## Use inside your own code

Patch before you instantiate/load the model:

```python
from mlx_reuse_fast_scan import install_fast_reuse_scan
install_fast_reuse_scan()

from mlx_speech.generation.reuse import REUSEEnhancer
from mlx_speech.models.reuse import load_mlx_semamba

model = load_mlx_semamba("re-use-mlx/model_mlx.safetensors")
enhancer = REUSEEnhancer(model)
```

## Important limits

- This is inference-only, not an autograd custom op.
- It was written for the RE-USE/SEMamba variable-B/C scan with `d_state=16`.
- It has to be run on Apple Silicon with MLX's Metal backend.
- If Faraday's `model_mlx.safetensors` key layout does not match the Python
  `mlx-speech` runtime, use `appautomaton/re-use-semamba-mlx` or convert the
  keys once. The scan patch itself is independent of the weight source.
