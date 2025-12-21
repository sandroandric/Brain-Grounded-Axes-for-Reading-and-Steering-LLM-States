"""
Evaluate steering significance for word-axis steering runs.

Scores each generated continuation along the trained word-axis adapter and
tests whether scores change with steering strength.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import stats
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
) -> Tuple[float | None, Dict]:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"][0].tolist()
    if not input_ids:
        return None, {"n_tokens": 0, "prompt_match": False}

    prompt_match = False
    start_idx = 0
    if prompt_ids:
        idx = find_subsequence(input_ids, prompt_ids)
        if idx is not None:
            prompt_match = True
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
        prompt_match = False

    hs = hs[start_idx:]
    if hs.shape[0] == 0:
        return None, {"n_tokens": 0, "prompt_match": prompt_match}

    if mean is not None and scale is not None:
        hs = (hs - mean) / scale

    scores = hs @ W.T + b
    if scores.ndim == 1:
        scores = scores[:, None]
    axis_scores = scores[:, axis_idx]
    return float(axis_scores.mean()), {"n_tokens": int(axis_scores.shape[0]), "prompt_match": prompt_match}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size < 2 or b.size < 2:
        return float("nan")
    v1 = np.var(a, ddof=1)
    v2 = np.var(b, ddof=1)
    pooled = np.sqrt(((a.size - 1) * v1 + (b.size - 1) * v2) / (a.size + b.size - 2))
    return float((np.mean(a) - np.mean(b)) / (pooled + 1e-10))


def perm_pvalue(a: np.ndarray, b: np.ndarray, n_perm: int, seed: int) -> float:
    if n_perm <= 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    combined = np.concatenate([a, b])
    n_a = a.size
    obs = float(np.mean(a) - np.mean(b))
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        diff = float(np.mean(combined[:n_a]) - np.mean(combined[n_a:]))
        if abs(diff) >= abs(obs):
            count += 1
    return (count + 1) / (n_perm + 1)


def main():
    p = argparse.ArgumentParser(description="Evaluate word-axis steering significance.")
    p.add_argument("--adapter-dir", type=pathlib.Path, required=True)
    p.add_argument("--inputs", nargs="+", default=None)
    p.add_argument("--input-glob", type=str, default=None)
    p.add_argument("--model", default="uer/gpt2-chinese-cluecorpussmall")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--prompt", type=str, default=None, help="Prompt used in steering runs.")
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    p.add_argument("--n-perms", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not args.inputs and not args.input_glob:
        raise ValueError("Provide --inputs or --input-glob")

    paths = []
    if args.inputs:
        paths.extend([pathlib.Path(p) for p in args.inputs])
    if args.input_glob:
        paths.extend([pathlib.Path(p) for p in glob.glob(args.input_glob)])
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise FileNotFoundError("No input steering JSON files found.")

    W, b, axes, mean, scale = load_adapter(args.adapter_dir)

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
    if args.prompt:
        prompt_cache[args.prompt] = tokenizer(args.prompt, add_special_tokens=False).input_ids

    results = {"model": args.model, "layer": layer_idx, "files": {}}

    for path in sorted(paths):
        data = json.loads(path.read_text())
        axis_id = int(data.get("axis_id", data.get("axis", 0)))
        axis_idx = resolve_axis_index(axis_id, axes, W.shape[0])

        scores_by_strength: Dict[float, List[float]] = {}
        meta = {"prompt_match_count": 0, "n_scored": 0}

        for gen in data.get("generations", []):
            strength = float(gen.get("strength", 0.0))
            text = gen.get("text", "")
            prompt_text = gen.get("prompt") or args.prompt
            prompt_ids = None
            if prompt_text:
                if prompt_text not in prompt_cache:
                    prompt_cache[prompt_text] = tokenizer(prompt_text, add_special_tokens=False).input_ids
                prompt_ids = prompt_cache[prompt_text]
            score, info = score_text(
                model,
                tokenizer,
                text,
                prompt_ids,
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
            scores_by_strength.setdefault(strength, []).append(score)
            meta["n_scored"] += 1
            if info.get("prompt_match"):
                meta["prompt_match_count"] += 1

        strengths = sorted(scores_by_strength.keys())
        all_strengths = []
        all_scores = []
        for s in strengths:
            vals = scores_by_strength[s]
            all_strengths.extend([s] * len(vals))
            all_scores.extend(vals)

        pos = np.array([v for s, vals in scores_by_strength.items() if s > 0 for v in vals])
        neg = np.array([v for s, vals in scores_by_strength.items() if s < 0 for v in vals])

        t_stat = p_val = d_val = perm_p = float("nan")
        if pos.size >= 2 and neg.size >= 2:
            t_stat, p_val = stats.ttest_ind(pos, neg)
            d_val = cohens_d(pos, neg)
            perm_p = perm_pvalue(pos, neg, args.n_perms, args.seed)

        r_val = r_p = float("nan")
        if len(all_scores) >= 3 and len(set(all_strengths)) > 1:
            r_val, r_p = stats.pearsonr(all_strengths, all_scores)

        results["files"][path.name] = {
            "axis_id": axis_id,
            "axis_index": axis_idx,
            "scores_by_strength": {str(k): v for k, v in scores_by_strength.items()},
            "mean_by_strength": {str(k): float(np.mean(v)) for k, v in scores_by_strength.items()},
            "pos_neg_ttest": {"t": float(t_stat), "p": float(p_val), "cohen_d": float(d_val), "perm_p": float(perm_p)},
            "strength_corr": {"r": float(r_val), "p": float(r_p)},
            "meta": meta,
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
