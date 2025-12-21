"""
Evaluate word axes against Chinese lexicon norms (concreteness, valence, arousal).

Uses actual Chinese psycholinguistic norms:
- MELD-SCH concreteness: 9,877 two-character words (Xu & Li 2020)
- Chinese VAD: 11,310 words valence/arousal (Xu et al. 2022)

Run example:
PYTHONPATH=. python analysis/word_axis_eval_lexica.py \
  --word-axes-root outputs/word_axes_ica \
  --lexica-root metadata/lexica \
  --out-json outputs/word_axes_ica/axis_eval_lexica.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def load_vocab_scores(root: pathlib.Path) -> Tuple[List[str], np.ndarray]:
    """Load vocab and axis scores from word atlas."""
    vocab = np.load(root / "vocab.npy", allow_pickle=True).tolist()
    scores_path = root / "pca_scores.npy"
    if not scores_path.exists():
        scores_path = root / "ica_scores.npy"
    scores = np.load(scores_path)
    return vocab, scores


def clean_word(w: str) -> str:
    """Remove whitespace and punctuation from word."""
    return re.sub(r'\s+', '', str(w).strip())


def load_concreteness(lexica_root: pathlib.Path) -> Dict[str, float]:
    """Load MELD-SCH concreteness norms."""
    path = lexica_root / "meld_sch_concreteness.csv"
    if not path.exists():
        print(f"[warn] Concreteness lexicon not found: {path}")
        return {}
    df = pd.read_csv(path)
    # Column: Word, Mean of Valid Ratings
    scores = {}
    for _, row in df.iterrows():
        w = clean_word(row['Word'])
        if w:
            scores[w] = float(row['Mean of Valid Ratings'])
    print(f"[load] Concreteness: {len(scores)} words")
    return scores


def load_vad(lexica_root: pathlib.Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Load Chinese VAD norms (valence, arousal)."""
    path = lexica_root / "chinese_vad_11310.csv"
    if not path.exists():
        print(f"[warn] VAD lexicon not found: {path}")
        return {}, {}
    df = pd.read_csv(path)
    # Columns: Word, Valence_Mean, Arousal_Mean
    valence = {}
    arousal = {}
    for _, row in df.iterrows():
        w = clean_word(row['Word'])
        if w:
            valence[w] = float(row['Valence_Mean'])
            arousal[w] = float(row['Arousal_Mean'])
    print(f"[load] VAD: {len(valence)} words")
    return valence, arousal


def match_lexicon(vocab: List[str], lexicon: Dict[str, float]) -> Tuple[np.ndarray, int]:
    """Match vocab against lexicon, return array and match count."""
    arr = np.full(len(vocab), np.nan)
    matched = 0
    for i, w in enumerate(vocab):
        w_clean = clean_word(w)
        if w_clean in lexicon:
            arr[i] = lexicon[w_clean]
            matched += 1
    return arr, matched


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2 + 1e-8)
    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled + 1e-8)


def eval_axis_continuous(axis_scores: np.ndarray, values: np.ndarray, name: str) -> Dict:
    """Evaluate axis against continuous variable (Pearson, Spearman, effect size)."""
    valid = ~np.isnan(values)
    if valid.sum() < 10:
        return {"n_matched": int(valid.sum()), "pearson_r": np.nan, "spearman_r": np.nan}

    a = axis_scores[valid]
    v = values[valid]

    r_pearson, p_pearson = pearsonr(a, v)
    r_spearman, p_spearman = spearmanr(a, v)

    # Effect size: high vs low quantiles
    q20, q80 = np.percentile(v, [20, 80])
    hi_mask = v >= q80
    lo_mask = v <= q20
    d = cohen_d(a[hi_mask], a[lo_mask])

    return {
        "n_matched": int(valid.sum()),
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "spearman_r": float(r_spearman),
        "spearman_p": float(p_spearman),
        "cohen_d_hi_vs_lo": float(d),
        "mean_hi_q80": float(np.mean(a[hi_mask])) if hi_mask.any() else None,
        "mean_lo_q20": float(np.mean(a[lo_mask])) if lo_mask.any() else None,
    }


def eval_all_axes(scores: np.ndarray, values: np.ndarray, name: str, top_k: int = 10) -> List[Dict]:
    """Evaluate all axes against a variable, return sorted by abs(r)."""
    results = []
    valid = ~np.isnan(values)
    if valid.sum() < 10:
        return [{"axis": k, "pearson_r": np.nan, "n_matched": int(valid.sum())} for k in range(scores.shape[1])]

    for k in range(scores.shape[1]):
        res = eval_axis_continuous(scores[:, k], values, name)
        res["axis"] = k
        results.append(res)

    # Sort by absolute Pearson r
    results.sort(key=lambda x: abs(x.get("pearson_r", 0) or 0), reverse=True)
    return results[:top_k]


def print_summary(name: str, results: List[Dict], values: np.ndarray):
    """Print summary of evaluation results."""
    valid = ~np.isnan(values)
    print(f"\n=== {name} ===")
    print(f"Matched words: {valid.sum()} / {len(values)}")
    if results and results[0].get("pearson_r") is not None:
        print(f"Top axes by |r|:")
        for r in results[:5]:
            axis = r["axis"]
            pr = r.get("pearson_r", np.nan)
            d = r.get("cohen_d_hi_vs_lo", np.nan)
            print(f"  Axis {axis:2d}: r={pr:+.3f}, d={d:+.3f}")


def main():
    p = argparse.ArgumentParser(description="Evaluate word axes against Chinese lexicon norms.")
    p.add_argument("--word-axes-root", type=pathlib.Path, required=True,
                   help="Path to word axes (e.g., outputs/word_axes_ica)")
    p.add_argument("--lexica-root", type=pathlib.Path, default=pathlib.Path("metadata/lexica"),
                   help="Path to lexica directory")
    p.add_argument("--out-json", type=pathlib.Path, required=True,
                   help="Output JSON path")
    p.add_argument("--top-k", type=int, default=10,
                   help="Number of top axes to report per variable")
    args = p.parse_args()

    # Load word atlas
    vocab, scores = load_vocab_scores(args.word_axes_root)
    print(f"[load] Word atlas: {len(vocab)} words, {scores.shape[1]} axes")

    # Load lexica
    concreteness = load_concreteness(args.lexica_root)
    valence, arousal = load_vad(args.lexica_root)

    # Match against vocab
    conc_arr, n_conc = match_lexicon(vocab, concreteness)
    val_arr, n_val = match_lexicon(vocab, valence)
    aro_arr, n_aro = match_lexicon(vocab, arousal)

    print(f"\n[match] Concreteness: {n_conc} / {len(vocab)} ({100*n_conc/len(vocab):.1f}%)")
    print(f"[match] Valence: {n_val} / {len(vocab)} ({100*n_val/len(vocab):.1f}%)")
    print(f"[match] Arousal: {n_aro} / {len(vocab)} ({100*n_aro/len(vocab):.1f}%)")

    # Evaluate axes
    results = {
        "meta": {
            "vocab_size": len(vocab),
            "n_axes": scores.shape[1],
            "concreteness_matched": n_conc,
            "valence_matched": n_val,
            "arousal_matched": n_aro,
            "lexica": {
                "concreteness": "MELD-SCH (Xu & Li 2020)",
                "vad": "Chinese VAD 11,310 (Xu et al. 2022)",
            },
        },
        "concreteness": eval_all_axes(scores, conc_arr, "concreteness", args.top_k),
        "valence": eval_all_axes(scores, val_arr, "valence", args.top_k),
        "arousal": eval_all_axes(scores, aro_arr, "arousal", args.top_k),
    }

    # Print summaries
    print_summary("Concreteness", results["concreteness"], conc_arr)
    print_summary("Valence", results["valence"], val_arr)
    print_summary("Arousal", results["arousal"], aro_arr)

    # Correlation matrix between lexicon variables (where all are available)
    all_valid = ~np.isnan(conc_arr) & ~np.isnan(val_arr) & ~np.isnan(aro_arr)
    if all_valid.sum() > 10:
        c, v, a = conc_arr[all_valid], val_arr[all_valid], aro_arr[all_valid]
        corr_cv, _ = pearsonr(c, v)
        corr_ca, _ = pearsonr(c, a)
        corr_va, _ = pearsonr(v, a)
        results["lexicon_correlations"] = {
            "n_all_matched": int(all_valid.sum()),
            "concreteness_valence": float(corr_cv),
            "concreteness_arousal": float(corr_ca),
            "valence_arousal": float(corr_va),
        }
        print(f"\n=== Lexicon Inter-correlations (n={all_valid.sum()}) ===")
        print(f"  Concreteness-Valence: r={corr_cv:.3f}")
        print(f"  Concreteness-Arousal: r={corr_ca:.3f}")
        print(f"  Valence-Arousal: r={corr_va:.3f}")

    # Save
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {args.out_json}")


if __name__ == "__main__":
    main()
