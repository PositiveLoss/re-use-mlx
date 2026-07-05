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
