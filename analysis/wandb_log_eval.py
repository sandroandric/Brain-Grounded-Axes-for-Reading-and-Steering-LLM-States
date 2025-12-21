"""
Log steering evaluation JSONs to Weights & Biases.

Supports both llm_word_axes_steer_eval.py and llm_steering_ppl_eval.py outputs.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, Any

import wandb


def flatten_metrics(prefix: str, metrics: Dict[str, Any], out: Dict[str, Any]):
    for k, v in metrics.items():
        if isinstance(v, dict):
            flatten_metrics(f"{prefix}{k}/", v, out)
        else:
            out[f"{prefix}{k}"] = v


def main():
    p = argparse.ArgumentParser(description="Log eval JSONs to W&B.")
    p.add_argument("--eval-json", type=pathlib.Path, required=False)
    p.add_argument("--input", dest="eval_json", type=pathlib.Path, required=False)
    p.add_argument("--project", required=True)
    p.add_argument("--entity", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--group", default=None)
    p.add_argument("--tags", nargs="*", default=None)
    p.add_argument("--prefix", default="")
    args = p.parse_args()

    if args.eval_json is None:
        raise SystemExit("error: --eval-json (or --input) is required")

    data = json.loads(args.eval_json.read_text())
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        group=args.group,
        tags=args.tags,
        config={"source": str(args.eval_json)},
        reinit=True,
    )

    if "model" in data:
        wandb.config.update({"model": data.get("model")}, allow_val_change=True)
    if "layer" in data:
        wandb.config.update({"layer": data.get("layer")}, allow_val_change=True)

    files = data.get("files", {})
    for key, entry in files.items():
        axis_id = entry.get("axis_id")
        metrics = entry.get("metrics", entry)
        flat: Dict[str, Any] = {}
        flatten_metrics("", metrics, flat)
        prefix = args.prefix
        if axis_id is not None:
            prefix = f"{prefix}axis_{axis_id}/"
        elif key:
            prefix = f"{prefix}{key.replace('/', '_')}/"
        out = {f"{prefix}{k}": v for k, v in flat.items()}
        wandb.log(out)

    wandb.finish()


if __name__ == "__main__":
    main()
