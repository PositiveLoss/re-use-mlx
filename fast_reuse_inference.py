"""Fast local RE-USE / SEMamba inference with MLX + Metal selective scan.

Examples:

    # Using the Faraday repo downloaded from Hugging Face:
    huggingface-cli download --local-dir re-use-mlx faraday/re-use-mlx
    python fast_reuse_inference.py noisy.wav clean.wav --weights re-use-mlx

    # Using appautomaton's Python MLX runtime weights:
    huggingface-cli download --local-dir reuse-semamba-mlx appautomaton/re-use-semamba-mlx
    python fast_reuse_inference.py noisy.wav clean.wav --weights reuse-semamba-mlx

    # Download automatically:
    python fast_reuse_inference.py noisy.wav clean.wav --repo faraday/re-use-mlx

Notes:
  - This script requires Apple Silicon + MLX.
  - It patches mlx-speech's RE-USE Mamba scan before model construction.
  - For a Faraday directory, it will use model_mlx.safetensors if present.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np
import soundfile as sf

from mlx_reuse_fast_scan import install_fast_reuse_scan


def _resolve_weights_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_file():
        return p
    if not p.exists():
        raise FileNotFoundError(p)

    # appautomaton/mlx-speech convention
    model = p / "model.safetensors"
    if model.exists():
        return p

    # faraday/re-use-mlx convention
    faraday_model = p / "model_mlx.safetensors"
    if faraday_model.exists():
        return faraday_model

    raise FileNotFoundError(
        f"Could not find model.safetensors or model_mlx.safetensors under {p}"
    )


def _maybe_download(repo_id: Optional[str]) -> Optional[Path]:
    if not repo_id:
        return None
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "Install huggingface_hub first: pip install 'huggingface_hub[hf_xet]'"
        ) from e

    local_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=["*.safetensors", "*.json", "LICENSE", "NOTICE", "README.md"],
    )
    return Path(local_dir)


def _read_mono_audio(path: str | Path) -> tuple[mx.array, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # soundfile returns [samples, channels]. RE-USE wrapper expects mono [T].
    mono = np.mean(audio, axis=1).astype(np.float32)
    return mx.array(mono), int(sr)


def load_legacy_faraday_weights(model_path: Path):
    from safetensors import safe_open
    import re
    from mlx_speech.models.reuse import SEMamba
    from mlx.utils import tree_unflatten

    with safe_open(str(model_path), "numpy") as f:
        keys = set(f.keys())

    if any(k.startswith("TSMamba") for k in keys):
        # Already the new format
        model = SEMamba()
        model.load_weights(str(model_path))
        mx.eval(model.parameters())
        return model

    print(
        f"Warning: {model_path.name} uses an outdated key format. Converting on the fly...",
        flush=True,
    )
    state = {}
    with safe_open(str(model_path), "numpy") as f:
        for k in f.keys():
            new_k = k
            new_k = new_k.replace("denseEncoder", "dense_encoder")
            new_k = new_k.replace("maskDecoder", "mask_decoder")
            new_k = new_k.replace("phaseDecoder", "phase_decoder")

            if ".denseBlock.layers." in new_k:
                new_k = new_k.replace(
                    ".denseBlock.layers.", ".dense_block.dense_block."
                )

            new_k = new_k.replace("upConv1", "up_conv1")
            new_k = new_k.replace("upConv2", "up_conv2")
            new_k = new_k.replace("finalConv", "final_conv")

            new_k = new_k.replace(".conv1.", ".dense_conv_1.")
            new_k = new_k.replace(".conv2.", ".dense_conv_2.")

            new_k = new_k.replace("phaseConvR", "phase_conv_r")
            new_k = new_k.replace("phaseConvI", "phase_conv_i")

            new_k = new_k.replace("tfMamba", "TSMamba")
            new_k = new_k.replace("freqMamba", "freq_mamba")
            new_k = new_k.replace("timeMamba", "time_mamba")
            new_k = new_k.replace("inProj", "in_proj")
            new_k = new_k.replace("outProj", "out_proj")
            new_k = new_k.replace("dtProj", "dt_proj")
            new_k = new_k.replace("xProj", "x_proj")
            new_k = new_k.replace("outputProj", "output_proj")

            new_k = new_k.replace("forward.conv1d", "forward_blocks.conv1d")
            new_k = new_k.replace("forward.dt_proj", "forward_blocks.dt_proj")
            new_k = new_k.replace("forward.in_proj", "forward_blocks.in_proj")
            new_k = new_k.replace("forward.out_proj", "forward_blocks.out_proj")
            new_k = new_k.replace("forward.x_proj", "forward_blocks.x_proj")
            new_k = new_k.replace("forward.A_log", "forward_blocks.A_log")
            new_k = new_k.replace("forward.D", "forward_blocks.D")

            new_k = new_k.replace("backward.conv1d", "backward_blocks.conv1d")
            new_k = new_k.replace("backward.dt_proj", "backward_blocks.dt_proj")
            new_k = new_k.replace("backward.in_proj", "backward_blocks.in_proj")
            new_k = new_k.replace("backward.out_proj", "backward_blocks.out_proj")
            new_k = new_k.replace("backward.x_proj", "backward_blocks.x_proj")
            new_k = new_k.replace("backward.A_log", "backward_blocks.A_log")
            new_k = new_k.replace("backward.D", "backward_blocks.D")

            new_k = re.sub(r"\.layers\.0\.", ".conv.", new_k)
            new_k = re.sub(r"\.layers\.1\.", ".norm.", new_k)
            new_k = re.sub(r"\.layers\.2\.", ".act.", new_k)

            state[new_k] = mx.array(f.get_tensor(k))

    model = SEMamba()
    model.update(tree_unflatten(list(state.items())))
    mx.eval(model.parameters())
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Noisy input WAV/FLAC/etc.")
    parser.add_argument("output", type=Path, help="Enhanced output WAV")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Directory/file with model.safetensors or model_mlx.safetensors",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="Optional Hugging Face repo id, e.g. faraday/re-use-mlx",
    )
    parser.add_argument("--chunk-size-s", type=float, default=1.0)
    parser.add_argument("--no-fast-scan", action="store_true")
    args = parser.parse_args()

    if args.weights is None:
        downloaded = _maybe_download(args.repo or "faraday/re-use-mlx")
        if downloaded is None:
            raise SystemExit("Provide --weights or --repo")
        weights = _resolve_weights_path(downloaded)
    else:
        weights = _resolve_weights_path(args.weights)

    if not args.no_fast_scan:
        install_fast_reuse_scan(verbose=True)

    # Import after patching. REUSEEnhancer uses mlx_speech.models.reuse SEMamba.
    from mlx_speech.generation.reuse import REUSEEnhancer
    from mlx_speech.models.reuse import load_mlx_semamba

    waveform, sr = _read_mono_audio(args.input)

    # load_mlx_semamba accepts either a directory with model.safetensors or a file.
    if weights.name == "model_mlx.safetensors":
        model = load_legacy_faraday_weights(weights)
    else:
        model = load_mlx_semamba(weights)
    enhancer = REUSEEnhancer(model)

    enhanced = enhancer.enhance(waveform, sr, chunk_size_s=args.chunk_size_s)
    mx.eval(enhanced)

    out_np = np.asarray(enhanced, dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), out_np, sr)
    print(f"wrote {args.output} at {sr} Hz")


if __name__ == "__main__":
    main()
