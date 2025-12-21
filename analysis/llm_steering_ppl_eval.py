"""
Compute perplexity of steered generations and test for fluency shifts.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_subsequence(haystack: List[int], needle: List[int]) -> int | None:
    if not needle:
        return 0
    for i in range(0, len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return None


def cohen_d(x, y):
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2 + 1e-8)
    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled + 1e-8)


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


def eval_metric(values_by_strength: Dict[float, List[float]], n_perm: int, seed: int):
    strengths = sorted(values_by_strength.keys())
    all_strengths = []
    all_vals = []
    for s in strengths:
        vals = values_by_strength[s]
        all_strengths.extend([s] * len(vals))
        all_vals.extend(vals)

    pos = np.array([v for s, vals in values_by_strength.items() if s > 0 for v in vals])
    neg = np.array([v for s, vals in values_by_strength.items() if s < 0 for v in vals])
    t_stat = p_val = d_val = perm_p = float("nan")
    if pos.size >= 2 and neg.size >= 2:
        t_stat, p_val = stats.ttest_ind(pos, neg)
        d_val = cohen_d(pos, neg)
        perm_p = perm_pvalue(pos, neg, n_perm, seed)

    r_val = r_p = float("nan")
    if len(all_vals) >= 3 and len(set(all_strengths)) > 1:
        r_val, r_p = stats.pearsonr(all_strengths, all_vals)

    return {
        "mean_by_strength": {str(k): float(np.mean(v)) for k, v in values_by_strength.items()},
        "pos_neg_ttest": {"t": float(t_stat), "p": float(p_val), "cohen_d": float(d_val), "perm_p": float(perm_p)},
        "strength_corr": {"r": float(r_val), "p": float(r_p)},
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate steering outputs with perplexity.")
    p.add_argument("--input-glob", type=str, required=True)
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--n-perms", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    args = p.parse_args()

    paths = [pathlib.Path(p) for p in glob.glob(args.input_glob)]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise FileNotFoundError("No steering JSON files found.")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device != "cpu" else torch.float32,
    )
    if args.device == "mps":
        mdl = mdl.to("mps")
    elif args.device == "cuda":
        mdl = mdl.to("cuda")
    mdl.eval()

    results = {"model": args.model, "files": {}}

    for path in sorted(paths):
        data = json.loads(path.read_text())
        axis_id = int(data.get("axis_id", data.get("axis", 0)))
        values_by_strength = defaultdict(list)
        n_scored = 0

        for gen in data.get("generations", []):
            strength = float(gen.get("strength", 0.0))
            text = gen.get("text", "")
            prompt = gen.get("prompt")

            full_ids = tok(text, add_special_tokens=False).input_ids
            if not full_ids:
                continue
            start_idx = 0
            if prompt:
                prompt_ids = tok(prompt, add_special_tokens=False).input_ids
                idx = find_subsequence(full_ids, prompt_ids)
                if idx is not None:
                    start_idx = idx + len(prompt_ids)

            if args.max_tokens and len(full_ids) > args.max_tokens:
                trim = len(full_ids) - args.max_tokens
                full_ids = full_ids[trim:]
                start_idx = max(0, start_idx - trim)

            if len(full_ids) < 2 or start_idx >= len(full_ids):
                continue

            input_ids = torch.tensor([full_ids], device=mdl.device)
            with torch.no_grad():
                outputs = mdl(input_ids)
            logits = outputs.logits[0, :-1]
            targets = input_ids[0, 1:]
            start = max(start_idx, 1)
            lp = torch.log_softmax(logits[start - 1 :], dim=-1)
            tgt = targets[start - 1 :]
            nll = -lp.gather(1, tgt.unsqueeze(1)).mean().item()
            ppl = float(np.exp(nll))
            values_by_strength[strength].append(ppl)
            n_scored += 1

        key = f"{path.parent.name}/{path.name}"
        results["files"][key] = {
            "axis_id": axis_id,
            "layer": path.parent.name,
            "metrics": eval_metric(values_by_strength, args.n_perms, args.seed),
            "meta": {"n_scored": n_scored},
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
