"""
Permutation test (max-stat) for lexicon-based word-axis associations.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import pearsonr

from analysis.word_axis_eval_lexica import load_vocab_scores, load_concreteness, load_vad, match_lexicon


def max_abs_r(scores: np.ndarray, values: np.ndarray) -> Tuple[int, float]:
    valid = ~np.isnan(values)
    if valid.sum() < 10:
        return -1, np.nan
    vals = values[valid]
    sc = scores[valid]
    best_idx = -1
    best_r = 0.0
    for k in range(sc.shape[1]):
        r, _ = pearsonr(sc[:, k], vals)
        if abs(r) > abs(best_r):
            best_r = float(r)
            best_idx = k
    return best_idx, best_r


def perm_test(scores: np.ndarray, values: np.ndarray, n_perms: int, rng: np.random.Generator) -> Tuple[int, float, float]:
    idx, obs_r = max_abs_r(scores, values)
    if idx < 0 or np.isnan(obs_r):
        return -1, np.nan, np.nan
    null = np.zeros(n_perms, dtype=np.float32)
    valid = ~np.isnan(values)
    vals = values[valid]
    sc = scores[valid]
    for i in range(n_perms):
        perm = rng.permutation(vals)
        best = 0.0
        for k in range(sc.shape[1]):
            r, _ = pearsonr(sc[:, k], perm)
            if abs(r) > abs(best):
                best = float(r)
        null[i] = abs(best)
    pval = float((1 + np.sum(null >= abs(obs_r))) / (1 + len(null)))
    return idx, float(obs_r), pval


def main():
    p = argparse.ArgumentParser(description="Permutation test for lexicon-based word axes.")
    p.add_argument("--word-axes-root", type=pathlib.Path, required=True)
    p.add_argument("--lexica-root", type=pathlib.Path, default=pathlib.Path("metadata/lexica"))
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    p.add_argument("--n-perms", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    vocab, scores = load_vocab_scores(args.word_axes_root)
    conc = load_concreteness(args.lexica_root)
    val, aro = load_vad(args.lexica_root)

    conc_arr, n_conc = match_lexicon(vocab, conc)
    val_arr, n_val = match_lexicon(vocab, val)
    aro_arr, n_aro = match_lexicon(vocab, aro)

    rng = np.random.default_rng(args.seed)
    results = {}
    for name, arr, n_match in [
        ("concreteness", conc_arr, n_conc),
        ("valence", val_arr, n_val),
        ("arousal", aro_arr, n_aro),
    ]:
        axis, r, pval = perm_test(scores, arr, args.n_perms, rng)
        results[name] = {
            "best_axis": axis,
            "best_pearson_r": r,
            "p_value": pval,
            "n_matched": n_match,
        }

    payload = {
        "n_perms": args.n_perms,
        "seed": args.seed,
        "results": results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
