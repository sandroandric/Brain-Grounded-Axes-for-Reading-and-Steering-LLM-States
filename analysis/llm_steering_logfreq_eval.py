"""
Evaluate steering outputs with log-frequency mean (no POS/NER models).
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

try:
    import jieba  # type: ignore
except Exception:
    jieba = None


def is_punct(tok: str) -> bool:
    return re.fullmatch(r"[\W_]+", tok) is not None


def normalize_token(t: str) -> str:
    return re.sub(r"\s+", "", str(t)).strip()


def strip_prompt(text: str, prompt: str | None) -> Tuple[str, bool]:
    if not prompt:
        return text, False
    idx = text.find(prompt)
    if idx == -1:
        return text, False
    return text[idx + len(prompt) :], True


def segment(text: str, segmenter: str) -> List[str]:
    if segmenter == "jieba" and jieba is not None:
        toks = jieba.lcut(text)
    elif segmenter == "whitespace":
        toks = text.split()
    else:
        toks = list(text)
    out = []
    for t in toks:
        t = t.strip()
        if not t:
            continue
        if is_punct(t):
            continue
        out.append(t)
    return out


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


def load_logfreq(confounds_path: pathlib.Path) -> Dict[str, float]:
    if not confounds_path.exists():
        return {}
    data = json.loads(confounds_path.read_text())
    out = {}
    for word, vals in data.items():
        lf = vals.get("logfreq")
        if lf is None:
            continue
        out[normalize_token(word)] = float(lf)
    return out


def main():
    p = argparse.ArgumentParser(description="Evaluate steering outputs with logfreq mean.")
    p.add_argument("--input-glob", type=str, required=True)
    p.add_argument("--confounds-json", type=pathlib.Path, default=pathlib.Path("metadata/lexica/confounds.json"))
    p.add_argument("--segmenter", choices=["jieba", "char", "whitespace"], default="jieba")
    p.add_argument("--n-perms", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    args = p.parse_args()

    paths = [pathlib.Path(p) for p in glob.glob(args.input_glob)]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise FileNotFoundError("No steering JSON files found.")

    if args.segmenter == "jieba" and jieba is None:
        args.segmenter = "char"

    logfreq_map = load_logfreq(args.confounds_json)
    results = {"files": {}, "segmenter": args.segmenter}

    for path in sorted(paths):
        data = json.loads(path.read_text())
        axis_id = int(data.get("axis_id", data.get("axis", 0)))
        values_by_strength = defaultdict(list)
        coverages = []
        prompt_match = 0

        for gen in data.get("generations", []):
            strength = float(gen.get("strength", 0.0))
            text = gen.get("text", "")
            prompt = gen.get("prompt")
            text, matched = strip_prompt(text, prompt)
            if matched:
                prompt_match += 1
            tokens = segment(text, args.segmenter)
            if not tokens:
                continue
            vals = [logfreq_map.get(normalize_token(t)) for t in tokens]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            values_by_strength[strength].append(float(np.mean(vals)))
            coverages.append(len(vals) / len(tokens))

        key = f"{path.parent.name}/{path.name}"
        results["files"][key] = {
            "axis_id": axis_id,
            "metrics": eval_metric(values_by_strength, args.n_perms, args.seed),
            "meta": {
                "n_scored": int(sum(len(v) for v in values_by_strength.values())),
                "mean_logfreq_coverage": float(np.mean(coverages)) if coverages else float("nan"),
                "prompt_match_count": prompt_match,
            },
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
