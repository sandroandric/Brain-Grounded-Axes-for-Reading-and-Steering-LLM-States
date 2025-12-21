"""
End-to-end TinyLlama steering pipeline:
1) Extract word-level hidden states
2) Train word-axis adapter
3) Run steering across multiple layers
4) Evaluate steering significance
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
from typing import List

from transformers import AutoConfig


def choose_layers(n_layers: int) -> List[int]:
    picks = {
        max(1, n_layers // 6),
        max(1, n_layers // 4),
        max(1, n_layers // 2),
        max(1, (3 * n_layers) // 4),
        n_layers,
    }
    return sorted(picks)


def run(cmd: List[str]):
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description="TinyLlama steering pipeline (multi-layer).")
    p.add_argument("--dataset-root", type=pathlib.Path, default=pathlib.Path("."))
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    p.add_argument("--axes", nargs="+", type=int, default=[13, 19, 15, 2])
    p.add_argument("--layers", nargs="+", type=int, default=None)
    p.add_argument("--time-align-root", type=pathlib.Path, default=None)
    p.add_argument("--prompt-file", type=pathlib.Path, default=pathlib.Path("analysis/prompts_steering_zh.txt"))
    p.add_argument("--strengths", nargs="+", type=float, default=[-5, -2, -1, 0, 1, 2, 5])
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--n-perms", type=int, default=1000)
    p.add_argument("--skip-hidden", action="store_true")
    p.add_argument("--skip-adapter", action="store_true")
    p.add_argument("--skip-steer", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()

    dataset_root = args.dataset_root.resolve()
    if args.time_align_root:
        time_align_root = args.time_align_root
    else:
        default_word = dataset_root / "derivatives" / "annotations" / "time_align" / "word-level"
        default_stim = dataset_root / "stimuli" / "time_align"
        time_align_root = default_word if default_word.exists() else default_stim
    hidden_root = dataset_root / "outputs" / "llm_hidden_tinyllama"
    adapter_dir = dataset_root / "outputs" / "llm_adapter_word_axes_tinyllama"
    steer_root = dataset_root / "outputs" / "llm_steering_tinyllama_layers"

    if not args.skip_hidden:
        run([
            "python", "analysis/extract_llm_word_hidden.py",
            "--time-align-root", str(time_align_root),
            "--out-dir", str(hidden_root),
            "--model", args.model,
            "--device", args.device,
        ])

    if not args.skip_adapter:
        run([
            "python", "analysis/train_llm_word_axes_adapter.py",
            "--word-axes-root", str(dataset_root / "outputs" / "word_axes_ica"),
            "--hidden-root", str(hidden_root),
            "--time-align-root", str(time_align_root),
            "--axes", *[str(a) for a in args.axes],
            "--out-dir", str(adapter_dir),
            "--standardize",
        ])

    if args.layers:
        layers = args.layers
    else:
        config = AutoConfig.from_pretrained(args.model)
        layers = choose_layers(config.num_hidden_layers)

    for layer in layers:
        layer_dir = steer_root / f"layer_{layer}"
        if not args.skip_steer:
            run([
                "python", "analysis/llm_word_axes_steer_batch.py",
                "--adapter-dir", str(adapter_dir),
                "--model", args.model,
                "--device", args.device,
                "--axes", *[str(a) for a in args.axes],
                "--prompt-file", str(args.prompt_file),
                "--strengths", *[str(s) for s in args.strengths],
                "--samples", str(args.samples),
                "--layer", str(layer),
                "--max-new-tokens", str(args.max_new_tokens),
                "--temperature", str(args.temperature),
                "--top-p", str(args.top_p),
                "--out-dir", str(layer_dir),
            ])

        if not args.skip_eval:
            run([
                "python", "analysis/llm_word_axes_steer_eval.py",
                "--adapter-dir", str(adapter_dir),
                "--input-glob", str(layer_dir / "axis_*.json"),
                "--model", args.model,
                "--device", args.device,
                "--out-json", str(layer_dir / "steer_eval.json"),
                "--n-perms", str(args.n_perms),
            ])


if __name__ == "__main__":
    main()
