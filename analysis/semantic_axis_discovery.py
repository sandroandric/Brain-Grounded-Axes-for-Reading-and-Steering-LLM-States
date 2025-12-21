"""
Rigorous analysis of whether MEG-derived word axes capture genuine neurolinguistic semantic dimensions.

Tests:
1. Permutation tests: Are axis-lexicon correlations above chance?
2. Axis specificity: Do different axes specialize for different semantic dimensions?
3. Exemplar analysis: What words define each axis?
4. Dissociation analysis: Are concreteness/valence/arousal captured by distinct axes?

Run:
PYTHONPATH=. python analysis/semantic_axis_discovery.py \
  --word-axes-root outputs/word_axes_ica \
  --lexica-root metadata/lexica \
  --n-perm 1000 \
  --out-dir outputs/semantic_analysis
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
from collections import defaultdict


def load_vocab_scores(root: pathlib.Path) -> Tuple[List[str], np.ndarray]:
    """Load vocab and axis scores."""
    vocab = np.load(root / "vocab.npy", allow_pickle=True).tolist()
    scores_path = root / "ica_scores.npy" if (root / "ica_scores.npy").exists() else root / "pca_scores.npy"
    scores = np.load(scores_path)
    return vocab, scores


def clean_word(w: str) -> str:
    return re.sub(r'\s+', '', str(w).strip())


def load_lexica(lexica_root: pathlib.Path) -> Dict[str, Dict[str, float]]:
    """Load all lexica."""
    lexica = {}

    # Concreteness
    conc_path = lexica_root / "meld_sch_concreteness.csv"
    if conc_path.exists():
        df = pd.read_csv(conc_path)
        lexica["concreteness"] = {clean_word(r["Word"]): float(r["Mean of Valid Ratings"])
                                   for _, r in df.iterrows() if clean_word(r["Word"])}

    # VAD
    vad_path = lexica_root / "chinese_vad_11310.csv"
    if vad_path.exists():
        df = pd.read_csv(vad_path)
        lexica["valence"] = {clean_word(r["Word"]): float(r["Valence_Mean"])
                             for _, r in df.iterrows() if clean_word(r["Word"])}
        lexica["arousal"] = {clean_word(r["Word"]): float(r["Arousal_Mean"])
                             for _, r in df.iterrows() if clean_word(r["Word"])}

    return lexica


def match_lexicon(vocab: List[str], lexicon: Dict[str, float]) -> np.ndarray:
    """Match vocab to lexicon values."""
    arr = np.full(len(vocab), np.nan)
    for i, w in enumerate(vocab):
        w_clean = clean_word(w)
        if w_clean in lexicon:
            arr[i] = lexicon[w_clean]
    return arr


def permutation_test(axis_scores: np.ndarray, lex_values: np.ndarray, n_perm: int = 1000) -> Dict:
    """Permutation test for correlation significance."""
    valid = ~np.isnan(lex_values)
    if valid.sum() < 20:
        return {"observed_r": np.nan, "p_perm": np.nan, "null_mean": np.nan, "null_std": np.nan}

    a = axis_scores[valid]
    v = lex_values[valid]

    observed_r, _ = pearsonr(a, v)

    # Permutation null distribution
    null_rs = []
    rng = np.random.default_rng(42)
    for _ in range(n_perm):
        v_perm = rng.permutation(v)
        r_perm, _ = pearsonr(a, v_perm)
        null_rs.append(r_perm)
    null_rs = np.array(null_rs)

    # Two-tailed p-value
    p_perm = (np.abs(null_rs) >= np.abs(observed_r)).mean()

    return {
        "observed_r": float(observed_r),
        "p_perm": float(p_perm),
        "null_mean": float(null_rs.mean()),
        "null_std": float(null_rs.std()),
        "percentile": float((null_rs < observed_r).mean() * 100),
        "n_matched": int(valid.sum()),
    }


def get_axis_exemplars(vocab: List[str], axis_scores: np.ndarray, lex_values: np.ndarray,
                       lexicon: Dict[str, float], n_top: int = 15) -> Dict:
    """Get exemplar words at axis extremes, with lexicon values."""
    valid = ~np.isnan(lex_values)
    valid_idx = np.where(valid)[0]

    if len(valid_idx) < 20:
        return {"high": [], "low": []}

    # Sort by axis score among matched words
    scores_valid = axis_scores[valid]
    order = np.argsort(scores_valid)

    low_idx = valid_idx[order[:n_top]]
    high_idx = valid_idx[order[-n_top:][::-1]]

    def make_entry(idx):
        w = vocab[idx]
        w_clean = clean_word(w)
        return {
            "word": w_clean,
            "axis_score": float(axis_scores[idx]),
            "lex_value": float(lexicon.get(w_clean, np.nan)),
        }

    return {
        "high": [make_entry(i) for i in high_idx],
        "low": [make_entry(i) for i in low_idx],
    }


def compute_axis_profile(scores: np.ndarray, axis_idx: int, lex_arrays: Dict[str, np.ndarray]) -> Dict:
    """Compute semantic profile for a single axis across all dimensions."""
    profile = {}
    for dim, arr in lex_arrays.items():
        valid = ~np.isnan(arr)
        if valid.sum() < 20:
            profile[dim] = np.nan
        else:
            r, _ = pearsonr(scores[:, axis_idx][valid], arr[valid])
            profile[dim] = float(r)
    return profile


def compute_axis_specificity(profiles: Dict[int, Dict[str, float]], dimensions: List[str]) -> Dict:
    """Analyze whether axes show specificity for different dimensions."""
    # For each dimension, find the axis with strongest absolute correlation
    best_axis = {}
    for dim in dimensions:
        rs = [(ax, abs(prof.get(dim, 0) or 0)) for ax, prof in profiles.items()]
        rs.sort(key=lambda x: x[1], reverse=True)
        best_axis[dim] = {"axis": rs[0][0], "abs_r": rs[0][1], "top3": rs[:3]}

    # Check dissociation: do different dimensions have different best axes?
    unique_best = len(set(best_axis[d]["axis"] for d in dimensions))
    total_dims = len(dimensions)

    return {
        "best_axis_per_dimension": best_axis,
        "n_unique_best_axes": unique_best,
        "n_dimensions": total_dims,
        "dissociation_ratio": unique_best / total_dims,
    }


def compute_overall_semantic_score(perm_results: Dict[int, Dict[str, Dict]]) -> Dict:
    """Compute overall evidence for semantic structure."""
    # Count significant axes per dimension
    sig_counts = defaultdict(int)
    total_r = defaultdict(float)

    for axis_idx, dim_results in perm_results.items():
        for dim, res in dim_results.items():
            if res.get("p_perm", 1) < 0.05:
                sig_counts[dim] += 1
            total_r[dim] += abs(res.get("observed_r", 0) or 0)

    n_axes = len(perm_results)

    return {
        "significant_axes_per_dimension": dict(sig_counts),
        "mean_abs_r_per_dimension": {d: v / n_axes for d, v in total_r.items()},
        "total_significant_axis_dimension_pairs": sum(sig_counts.values()),
        "n_axes": n_axes,
    }


def main():
    p = argparse.ArgumentParser(description="Rigorous semantic axis discovery analysis.")
    p.add_argument("--word-axes-root", type=pathlib.Path, required=True)
    p.add_argument("--lexica-root", type=pathlib.Path, default=pathlib.Path("metadata/lexica"))
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--top-exemplars", type=int, default=15)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("[load] Loading word atlas...")
    vocab, scores = load_vocab_scores(args.word_axes_root)
    n_words, n_axes = scores.shape
    print(f"  {n_words} words, {n_axes} axes")

    print("[load] Loading lexica...")
    lexica = load_lexica(args.lexica_root)
    dimensions = list(lexica.keys())
    print(f"  Dimensions: {dimensions}")

    # Match lexica to vocab
    lex_arrays = {dim: match_lexicon(vocab, lex) for dim, lex in lexica.items()}
    for dim, arr in lex_arrays.items():
        n_match = (~np.isnan(arr)).sum()
        print(f"  {dim}: {n_match} matched ({100*n_match/n_words:.1f}%)")

    # === 1. Permutation tests ===
    print(f"\n[perm] Running permutation tests (n={args.n_perm})...")
    perm_results = {}
    for ax in range(n_axes):
        perm_results[ax] = {}
        for dim in dimensions:
            res = permutation_test(scores[:, ax], lex_arrays[dim], args.n_perm)
            perm_results[ax][dim] = res
        if (ax + 1) % 5 == 0:
            print(f"  Completed axis {ax + 1}/{n_axes}")

    # Save permutation results
    with open(args.out_dir / "permutation_tests.json", "w") as f:
        json.dump(perm_results, f, indent=2)

    # === 2. Axis profiles ===
    print("\n[profile] Computing axis semantic profiles...")
    profiles = {ax: compute_axis_profile(scores, ax, lex_arrays) for ax in range(n_axes)}

    with open(args.out_dir / "axis_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2)

    # === 3. Axis specificity ===
    print("\n[specificity] Analyzing axis specificity...")
    specificity = compute_axis_specificity(profiles, dimensions)

    with open(args.out_dir / "axis_specificity.json", "w") as f:
        json.dump(specificity, f, indent=2)

    # === 4. Exemplar words ===
    print("\n[exemplars] Extracting exemplar words...")
    exemplars = {}
    for ax in range(n_axes):
        exemplars[ax] = {}
        for dim in dimensions:
            exemplars[ax][dim] = get_axis_exemplars(
                vocab, scores[:, ax], lex_arrays[dim], lexica[dim], args.top_exemplars
            )

    with open(args.out_dir / "axis_exemplars.json", "w") as f:
        json.dump(exemplars, f, indent=2, ensure_ascii=False)

    # === 5. Overall semantic score ===
    print("\n[summary] Computing overall semantic structure evidence...")
    overall = compute_overall_semantic_score(perm_results)

    with open(args.out_dir / "overall_evidence.json", "w") as f:
        json.dump(overall, f, indent=2)

    # === Print summary ===
    print("\n" + "=" * 60)
    print("SEMANTIC AXIS DISCOVERY SUMMARY")
    print("=" * 60)

    print(f"\n1. PERMUTATION TEST RESULTS (α=0.05, n_perm={args.n_perm})")
    print("-" * 40)
    for dim in dimensions:
        sig_axes = [ax for ax in range(n_axes) if perm_results[ax][dim].get("p_perm", 1) < 0.05]
        print(f"  {dim}: {len(sig_axes)}/{n_axes} axes significant")
        if sig_axes:
            top_sig = sorted(sig_axes, key=lambda x: abs(perm_results[x][dim]["observed_r"]), reverse=True)[:3]
            for ax in top_sig:
                r = perm_results[ax][dim]["observed_r"]
                p = perm_results[ax][dim]["p_perm"]
                print(f"    Axis {ax:2d}: r={r:+.3f}, p_perm={p:.3f}")

    print(f"\n2. AXIS SPECIFICITY")
    print("-" * 40)
    print(f"  Dissociation ratio: {specificity['dissociation_ratio']:.2f}")
    print(f"  (1.0 = each dimension has unique best axis)")
    for dim in dimensions:
        best = specificity["best_axis_per_dimension"][dim]
        print(f"  {dim}: best axis = {best['axis']} (|r|={best['abs_r']:.3f})")

    print(f"\n3. OVERALL SEMANTIC STRUCTURE")
    print("-" * 40)
    total_sig = overall["total_significant_axis_dimension_pairs"]
    max_possible = n_axes * len(dimensions)
    print(f"  Significant axis-dimension pairs: {total_sig}/{max_possible}")
    print(f"  Expected by chance (α=0.05): {max_possible * 0.05:.1f}")
    print(f"  Enrichment ratio: {total_sig / (max_possible * 0.05):.1f}x")

    print(f"\n4. TOP AXES BY DIMENSION")
    print("-" * 40)
    for dim in dimensions:
        # Sort axes by absolute correlation for this dimension
        ax_rs = [(ax, profiles[ax][dim]) for ax in range(n_axes)]
        ax_rs.sort(key=lambda x: abs(x[1] or 0), reverse=True)
        top3 = ax_rs[:3]
        print(f"\n  {dim.upper()}:")
        for ax, r in top3:
            p = perm_results[ax][dim].get("p_perm", 1)
            sig = "*" if p < 0.05 else ""
            # Get top exemplar words
            hi_words = [e["word"] for e in exemplars[ax][dim]["high"][:5]]
            lo_words = [e["word"] for e in exemplars[ax][dim]["low"][:5]]
            print(f"    Axis {ax:2d}: r={r:+.3f}{sig}")
            print(f"      High: {', '.join(hi_words)}")
            print(f"      Low:  {', '.join(lo_words)}")

    # === Interpretation ===
    print(f"\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    enrichment = total_sig / (max_possible * 0.05)
    if enrichment > 3:
        print("✓ STRONG evidence for semantic structure in MEG axes")
        print(f"  {enrichment:.1f}x more significant correlations than expected by chance")
    elif enrichment > 1.5:
        print("◐ MODERATE evidence for semantic structure")
        print(f"  {enrichment:.1f}x enrichment over chance")
    else:
        print("✗ WEAK evidence for semantic structure")
        print(f"  Only {enrichment:.1f}x enrichment over chance")

    if specificity["dissociation_ratio"] >= 0.67:
        print("✓ Axes show DISSOCIABLE semantic dimensions")
    else:
        print("◐ Limited axis specialization across dimensions")

    print(f"\nResults saved to: {args.out_dir}/")
    print("  - permutation_tests.json")
    print("  - axis_profiles.json")
    print("  - axis_specificity.json")
    print("  - axis_exemplars.json")
    print("  - overall_evidence.json")


if __name__ == "__main__":
    main()
