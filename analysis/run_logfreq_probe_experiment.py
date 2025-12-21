"""
Brain-axis vs text-probe comparison for log-frequency steering.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess
from typing import Dict, List, Tuple

import numpy as np


def run(cmd: List[str]):
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_axis_vector(adapter_dir: pathlib.Path, axis_id: int) -> np.ndarray:
    W = np.load(adapter_dir / "adapter_W.npy")
    sidecar_path = adapter_dir / "adapter_sidecar.json"
    axes = None
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text())
        axes = sidecar.get("axes")
    if axes and axis_id in axes:
        idx = axes.index(axis_id)
    else:
        idx = axis_id
    if W.ndim == 1:
        vec = W
    else:
        vec = W[idx]
    return vec.astype(np.float64)


def read_metric(path: pathlib.Path) -> Tuple[float, float]:
    data = json.loads(path.read_text())
    files = list(data.get("files", {}).values())
    if not files:
        return float("nan"), float("nan")
    metrics = files[0]["metrics"]["pos_neg_ttest"]
    return float(metrics.get("cohen_d", float("nan"))), float(metrics.get("perm_p", float("nan")))


def main():
    p = argparse.ArgumentParser(description="Run logfreq probe vs brain-axis steering comparison.")
    p.add_argument("--dataset-root", type=pathlib.Path, default=pathlib.Path("."))
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    p.add_argument("--layer", type=int, default=11)
    p.add_argument("--run-tag", type=str, default=None)
    p.add_argument("--brain-adapter-dir", type=pathlib.Path, default=pathlib.Path("outputs/llm_adapter_word_axes_tinyllama"))
    p.add_argument("--brain-axis", type=int, default=15)
    p.add_argument("--hidden-root", type=pathlib.Path, default=pathlib.Path("outputs/llm_hidden_tinyllama"))
    p.add_argument("--time-align-root", type=pathlib.Path, default=None)
    p.add_argument("--confounds-json", type=pathlib.Path, default=pathlib.Path("metadata/lexica/confounds.json"))
    p.add_argument("--prompt-file", type=pathlib.Path, default=pathlib.Path("analysis/prompts_steering_zh.txt"))
    p.add_argument("--strengths", nargs="+", type=float, default=[-5, -2, -1, 0, 1, 2, 5])
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--padding-side", choices=["left", "right"], default=None)
    p.add_argument("--n-perms", type=int, default=1000)
    p.add_argument("--skip-hidden", action="store_true")
    p.add_argument("--skip-probe-train", action="store_true")
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

    run_tag = args.run_tag or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = dataset_root / "outputs" / "llm_probe_efficiency" / f"run_{run_tag}"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "prompts_used.txt").write_text(args.prompt_file.read_text())

    probe_adapter_dir = run_root / "logfreq_probe_adapter"
    brain_dir = run_root / "brain_axis"
    probe_dir = run_root / "text_probe"
    eval_dir = run_root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_hidden:
        run([
            "python", "analysis/extract_llm_word_hidden.py",
            "--time-align-root", str(time_align_root),
            "--out-dir", str(args.hidden_root),
            "--model", args.model,
            "--device", args.device,
            "--layer", str(args.layer),
        ])

    if not args.skip_probe_train:
        run([
            "python", "analysis/train_llm_logfreq_probe.py",
            "--hidden-root", str(args.hidden_root),
            "--time-align-root", str(time_align_root),
            "--confounds-json", str(args.confounds_json),
            "--out-dir", str(probe_adapter_dir),
            "--standardize",
        ])

    if not args.skip_steer:
        run([
            "python", "analysis/llm_word_axes_steer_batch.py",
            "--adapter-dir", str(args.brain_adapter_dir),
            "--model", args.model,
            "--device", args.device,
            "--axes", str(args.brain_axis),
            "--prompt-file", str(args.prompt_file),
            "--strengths", *[str(s) for s in args.strengths],
            "--samples", str(args.samples),
            "--layer", str(args.layer),
            "--max-new-tokens", str(args.max_new_tokens),
            "--temperature", str(args.temperature),
            "--top-p", str(args.top_p),
            "--batch-size", str(args.batch_size),
            *([] if args.padding_side is None else ["--padding-side", args.padding_side]),
            "--out-dir", str(brain_dir),
        ])
        run([
            "python", "analysis/llm_word_axes_steer_batch.py",
            "--adapter-dir", str(probe_adapter_dir),
            "--model", args.model,
            "--device", args.device,
            "--axes", "0",
            "--prompt-file", str(args.prompt_file),
            "--strengths", *[str(s) for s in args.strengths],
            "--samples", str(args.samples),
            "--layer", str(args.layer),
            "--max-new-tokens", str(args.max_new_tokens),
            "--temperature", str(args.temperature),
            "--top-p", str(args.top_p),
            "--batch-size", str(args.batch_size),
            *([] if args.padding_side is None else ["--padding-side", args.padding_side]),
            "--out-dir", str(probe_dir),
        ])

    if not args.skip_eval:
        run([
            "python", "analysis/llm_steering_logfreq_eval.py",
            "--input-glob", str(brain_dir / "axis_*.json"),
            "--confounds-json", str(args.confounds_json),
            "--out-json", str(eval_dir / "brain_logfreq_eval.json"),
            "--n-perms", str(args.n_perms),
        ])
        run([
            "python", "analysis/llm_steering_logfreq_eval.py",
            "--input-glob", str(probe_dir / "axis_*.json"),
            "--confounds-json", str(args.confounds_json),
            "--out-json", str(eval_dir / "probe_logfreq_eval.json"),
            "--n-perms", str(args.n_perms),
        ])
        run([
            "python", "analysis/llm_steering_ppl_eval.py",
            "--input-glob", str(brain_dir / "axis_*.json"),
            "--model", args.model,
            "--device", args.device,
            "--out-json", str(eval_dir / "brain_ppl_eval.json"),
            "--n-perms", str(args.n_perms),
        ])
        run([
            "python", "analysis/llm_steering_ppl_eval.py",
            "--input-glob", str(probe_dir / "axis_*.json"),
            "--model", args.model,
            "--device", args.device,
            "--out-json", str(eval_dir / "probe_ppl_eval.json"),
            "--n-perms", str(args.n_perms),
        ])

        brain_logfreq_d, brain_logfreq_p = read_metric(eval_dir / "brain_logfreq_eval.json")
        brain_ppl_d, brain_ppl_p = read_metric(eval_dir / "brain_ppl_eval.json")
        probe_logfreq_d, probe_logfreq_p = read_metric(eval_dir / "probe_logfreq_eval.json")
        probe_ppl_d, probe_ppl_p = read_metric(eval_dir / "probe_ppl_eval.json")

        brain_vec = load_axis_vector(args.brain_adapter_dir, args.brain_axis)
        probe_vec = load_axis_vector(probe_adapter_dir, 0)
        cos = float(np.dot(brain_vec, probe_vec) / (np.linalg.norm(brain_vec) * np.linalg.norm(probe_vec) + 1e-12))

        summary = {
            "run_tag": run_tag,
            "model": args.model,
            "layer": args.layer,
            "brain_axis": args.brain_axis,
            "text_probe": "logfreq",
            "cosine_similarity": cos,
            "brain_axis_metrics": {
                "logfreq_d": brain_logfreq_d,
                "logfreq_perm_p": brain_logfreq_p,
                "ppl_d": brain_ppl_d,
                "ppl_perm_p": brain_ppl_p,
            },
            "text_probe_metrics": {
                "logfreq_d": probe_logfreq_d,
                "logfreq_perm_p": probe_logfreq_p,
                "ppl_d": probe_ppl_d,
                "ppl_perm_p": probe_ppl_p,
            },
        }
        (run_root / "efficiency_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        (run_root / "efficiency_summary.csv").write_text(
            "method,logfreq_d,ppl_d\n"
            f"brain_axis,{brain_logfreq_d},{brain_ppl_d}\n"
            f"text_probe,{probe_logfreq_d},{probe_ppl_d}\n"
        )
        print(f"[saved] {run_root / 'efficiency_summary.json'}")


if __name__ == "__main__":
    main()
