"""
Compute bootstrap CI and prompt-level consistency for steering outputs.

Scores each generation with the adapter, then reports:
- delta mean (pos - neg) with bootstrap CI
- per-prompt sign consistency (fraction of prompts with pos > neg)
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def load_adapter(adapter_dir: pathlib.Path):
    W = np.load(adapter_dir / "adapter_W.npy")
    b = np.load(adapter_dir / "adapter_b.npy")
    sidecar_path = adapter_dir / "adapter_sidecar.json"
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
    axes = sidecar.get("axes")
    mean_path = adapter_dir / "adapter_scaler_mean.npy"
    scale_path = adapter_dir / "adapter_scaler_scale.npy"
    mean = np.load(mean_path) if mean_path.exists() else None
    scale = np.load(scale_path) if scale_path.exists() else None
    return W, b, axes, mean, scale


def ensure_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def resolve_axis_index(axis_id: int, axes: List[int] | None, n_axes: int) -> int:
    if axes and axis_id in axes:
        return axes.index(axis_id)
    if 0 <= axis_id < n_axes:
        return axis_id
    raise ValueError(f"Axis {axis_id} not found. Available axes: {axes}")


def model_layer_index(model, layer: int) -> int:
    n_layers = model.config.num_hidden_layers
    layer_idx = layer if layer >= 0 else n_layers + layer + 1
    if layer_idx < 0 or layer_idx > n_layers:
        return n_layers
    return layer_idx


def find_subsequence(haystack: List[int], needle: List[int]) -> int | None:
    if not needle:
        return 0
    for i in range(0, len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return None


def score_text(
    model,
    tokenizer,
    text: str,
    prompt_ids: List[int] | None,
    layer_idx: int,
    W: np.ndarray,
    b: np.ndarray,
    mean: np.ndarray | None,
    scale: np.ndarray | None,
    axis_idx: int,
    device: str,
) -> float | None:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"][0].tolist()
    if not input_ids:
        return None

    start_idx = 0
    if prompt_ids:
        idx = find_subsequence(input_ids, prompt_ids)
        if idx is not None:
            start_idx = idx + len(prompt_ids)

    if device == "mps":
        model_inputs = {k: v.to("mps") for k, v in enc.items()}
    elif device == "cuda":
        model_inputs = {k: v.to("cuda") for k, v in enc.items()}
    else:
        model_inputs = enc

    with torch.no_grad():
        outputs = model(**model_inputs, output_hidden_states=True)
    hs = outputs.hidden_states[layer_idx][0].detach().cpu().numpy()

    if start_idx >= hs.shape[0]:
        start_idx = 0

    hs = hs[start_idx:]
    if hs.shape[0] == 0:
        return None

    if mean is not None and scale is not None:
        hs = (hs - mean) / scale

    scores = hs @ W.T + b
    axis_scores = scores[:, axis_idx]
    return float(axis_scores.mean())


def bootstrap_ci(pos: np.ndarray, neg: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    delta = float(np.mean(pos) - np.mean(neg))
    boot = []
    for _ in range(n_boot):
        bpos = rng.choice(pos, size=len(pos), replace=True)
        bneg = rng.choice(neg, size=len(neg), replace=True)
        boot.append(float(np.mean(bpos) - np.mean(bneg)))
    ci_low = float(np.percentile(boot, 2.5))
    ci_high = float(np.percentile(boot, 97.5))
    return delta, ci_low, ci_high


def main():
    p = argparse.ArgumentParser(description="Bootstrap CI + prompt consistency for steering outputs.")
    p.add_argument("--adapter-dir", type=pathlib.Path, required=True)
    p.add_argument("--input-json", type=pathlib.Path, required=True)
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    args = p.parse_args()

    data = json.loads(args.input_json.read_text())
    axis_id = int(data.get("axis_id", data.get("axis", 0)))

    W, b, axes, mean, scale = load_adapter(args.adapter_dir)
    axis_idx = resolve_axis_index(axis_id, axes, W.shape[0])

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    ensure_pad_token(tokenizer)
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device != "cpu" else torch.float32,
    )
    if args.device == "mps":
        model = model.to("mps")
    elif args.device == "cuda":
        model = model.to("cuda")
    model.eval()

    layer_idx = model_layer_index(model, args.layer)

    prompt_cache: Dict[str, List[int]] = {}
    scored = []
    for gen in data.get("generations", []):
        prompt = gen.get("prompt", "")
        if prompt not in prompt_cache:
            prompt_cache[prompt] = tokenizer(prompt, add_special_tokens=False).input_ids
        score = score_text(
            model,
            tokenizer,
            gen.get("text", ""),
            prompt_cache.get(prompt),
            layer_idx,
            W,
            b,
            mean,
            scale,
            axis_idx,
            args.device,
        )
        if score is None:
            continue
        scored.append(
            {
                "prompt": prompt,
                "strength": float(gen.get("strength", 0.0)),
                "score": score,
            }
        )

    pos = np.array([g["score"] for g in scored if g["strength"] > 0], dtype=float)
    neg = np.array([g["score"] for g in scored if g["strength"] < 0], dtype=float)
    if len(pos) < 2 or len(neg) < 2:
        raise SystemExit("Not enough pos/neg samples for bootstrap.")

    delta, ci_low, ci_high = bootstrap_ci(pos, neg, args.n_boot, args.seed)

    # Prompt-level consistency
    by_prompt: Dict[str, Dict[str, List[float]]] = {}
    for g in scored:
        by_prompt.setdefault(g["prompt"], {}).setdefault(
            "pos" if g["strength"] > 0 else "neg",
            [],
        ).append(g["score"])

    prompt_signs = []
    for prompt, vals in by_prompt.items():
        if "pos" not in vals or "neg" not in vals:
            continue
        sign = np.sign(np.mean(vals["pos"]) - np.mean(vals["neg"]))
        if sign != 0:
            prompt_signs.append(sign)

    n_prompts = len(prompt_signs)
    n_pos = int(np.sum(np.array(prompt_signs) > 0))
    n_neg = int(np.sum(np.array(prompt_signs) < 0))

    out = {
        "input_json": str(args.input_json),
        "axis_id": axis_id,
        "layer": layer_idx,
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
        "delta_mean": delta,
        "delta_ci_95": [ci_low, ci_high],
        "prompt_consistency": {
            "n_prompts": n_prompts,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "frac_pos": (n_pos / n_prompts) if n_prompts else None,
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
