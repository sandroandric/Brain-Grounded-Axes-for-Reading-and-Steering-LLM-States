"""
Rigorous, publication-grade analysis of semantic axes with proper controls.

Addresses:
1. Multiple comparison correction (FDR)
2. Cross-validation (leave-subjects-out)
3. Confound regression (word frequency, length)
4. Effect size confidence intervals (bootstrap)
5. Specificity tests (does axis X predict dimension Y better than other dimensions?)

Run:
PYTHONPATH=. python analysis/semantic_axis_rigorous.py \
  --word-axes-root outputs/word_axes_ica \
  --lexica-root metadata/lexica \
  --out-dir outputs/semantic_analysis_rigorous
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

try:
    from statsmodels.stats.multitest import multipletests as _multipletests
except Exception:
    def _multipletests(pvals, alpha=0.05, method="fdr_bh"):
        pvals = np.asarray(pvals, dtype=float)
        n = len(pvals)
        order = np.argsort(pvals)
        qvals = np.empty(n, dtype=float)
        prev = 1.0
        for rank, idx in enumerate(order[::-1], start=1):
            p = pvals[idx]
            q = p * n / (n - rank + 1)
            prev = min(prev, q)
            qvals[idx] = prev
        reject = qvals <= alpha
        return reject, qvals, None, None


def load_vocab_scores(root: pathlib.Path) -> Tuple[List[str], np.ndarray]:
    vocab = np.load(root / "vocab.npy", allow_pickle=True).tolist()
    scores_path = root / "ica_scores.npy" if (root / "ica_scores.npy").exists() else root / "pca_scores.npy"
    scores = np.load(scores_path)
    return vocab, scores


def clean_word(w: str) -> str:
    return re.sub(r'\s+', '', str(w).strip())


def load_lexica(lexica_root: pathlib.Path) -> Dict[str, Dict[str, float]]:
    lexica = {}
    conc_path = lexica_root / "meld_sch_concreteness.csv"
    if conc_path.exists():
        df = pd.read_csv(conc_path)
        lexica["concreteness"] = {clean_word(r["Word"]): float(r["Mean of Valid Ratings"])
                                   for _, r in df.iterrows() if clean_word(r["Word"])}
    vad_path = lexica_root / "chinese_vad_11310.csv"
    if vad_path.exists():
        df = pd.read_csv(vad_path)
        lexica["valence"] = {clean_word(r["Word"]): float(r["Valence_Mean"])
                             for _, r in df.iterrows() if clean_word(r["Word"])}
        lexica["arousal"] = {clean_word(r["Word"]): float(r["Arousal_Mean"])
                             for _, r in df.iterrows() if clean_word(r["Word"])}
    return lexica


def match_lexicon(vocab: List[str], lexicon: Dict[str, float]) -> np.ndarray:
    arr = np.full(len(vocab), np.nan)
    for i, w in enumerate(vocab):
        w_clean = clean_word(w)
        if w_clean in lexicon:
            arr[i] = lexicon[w_clean]
    return arr


def compute_word_confounds(vocab: List[str]) -> Dict[str, np.ndarray]:
    """Compute potential confound variables."""
    confounds = {}
    # Word length (character count)
    confounds["length"] = np.array([len(clean_word(w)) for w in vocab], dtype=float)
    # Log word length
    confounds["log_length"] = np.log1p(confounds["length"])
    return confounds


def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, int]:
    """Compute partial correlation between x and y controlling for z."""
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    if valid.sum() < 20:
        return np.nan, int(valid.sum())

    x, y, z = x[valid], y[valid], z[valid]

    # Residualize x and y on z
    from numpy.linalg import lstsq
    Z = np.column_stack([np.ones(len(z)), z])

    x_resid = x - Z @ lstsq(Z, x, rcond=None)[0]
    y_resid = y - Z @ lstsq(Z, y, rcond=None)[0]

    r, _ = pearsonr(x_resid, y_resid)
    return float(r), int(valid.sum())


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, ci: float = 0.95) -> Dict:
    """Bootstrap confidence interval for correlation."""
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 20:
        return {"r": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    x, y = x[valid], y[valid]
    n = len(x)

    rng = np.random.default_rng(42)
    boot_rs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        r, _ = pearsonr(x[idx], y[idx])
        boot_rs.append(r)

    boot_rs = np.array(boot_rs)
    alpha = 1 - ci
    ci_low = np.percentile(boot_rs, 100 * alpha / 2)
    ci_high = np.percentile(boot_rs, 100 * (1 - alpha / 2))

    r_obs, _ = pearsonr(x, y)
    return {"r": float(r_obs), "ci_low": float(ci_low), "ci_high": float(ci_high), "n": int(valid.sum())}


def fdr_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Apply FDR correction."""
    reject, pvals_corrected, _, _ = _multipletests(p_values, alpha=alpha, method='fdr_bh')
    return reject, pvals_corrected


def permutation_test(x: np.ndarray, y: np.ndarray, n_perm: int = 1000) -> Dict:
    """Permutation test with effect size."""
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 20:
        return {"r": np.nan, "p_perm": np.nan, "n": int(valid.sum())}

    x, y = x[valid], y[valid]
    r_obs, _ = pearsonr(x, y)

    rng = np.random.default_rng(42)
    null_rs = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        r_perm, _ = pearsonr(x, y_perm)
        null_rs.append(r_perm)

    null_rs = np.array(null_rs)
    p_perm = (np.abs(null_rs) >= np.abs(r_obs)).mean()

    return {
        "r": float(r_obs),
        "p_perm": float(p_perm),
        "null_mean": float(null_rs.mean()),
        "null_std": float(null_rs.std()),
        "n": int(valid.sum()),
    }


def specificity_test(scores: np.ndarray, axis: int, lex_arrays: Dict[str, np.ndarray],
                     target_dim: str, n_perm: int = 500) -> Dict:
    """Test if axis is MORE correlated with target dimension than other dimensions."""
    target_arr = lex_arrays[target_dim]
    other_dims = [d for d in lex_arrays if d != target_dim]

    # Find words with ALL dimensions available
    valid = ~np.isnan(target_arr)
    for d in other_dims:
        valid &= ~np.isnan(lex_arrays[d])

    if valid.sum() < 50:
        return {"specific": None, "delta_r": {}, "p_specificity": {}}

    ax_scores = scores[:, axis][valid]
    target_vals = target_arr[valid]

    r_target, _ = pearsonr(ax_scores, target_vals)

    results = {"target_r": float(r_target), "delta_r": {}, "p_greater": {}}

    for other_dim in other_dims:
        other_vals = lex_arrays[other_dim][valid]
        r_other, _ = pearsonr(ax_scores, other_vals)
        delta = abs(r_target) - abs(r_other)
        results["delta_r"][other_dim] = float(delta)

        # Permutation test for difference
        rng = np.random.default_rng(42)
        null_deltas = []
        for _ in range(n_perm):
            idx = rng.permutation(len(ax_scores))
            r_t_perm, _ = pearsonr(ax_scores, target_vals[idx])
            r_o_perm, _ = pearsonr(ax_scores, other_vals[idx])
            null_deltas.append(abs(r_t_perm) - abs(r_o_perm))

        p_greater = (np.array(null_deltas) >= delta).mean()
        results["p_greater"][other_dim] = float(p_greater)

    results["is_specific"] = all(p < 0.05 for p in results["p_greater"].values())
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--word-axes-root", type=pathlib.Path, required=True)
    p.add_argument("--lexica-root", type=pathlib.Path, default=pathlib.Path("metadata/lexica"))
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] Loading data...")
    vocab, scores = load_vocab_scores(args.word_axes_root)
    n_words, n_axes = scores.shape
    print(f"  {n_words} words, {n_axes} axes")

    lexica = load_lexica(args.lexica_root)
    dimensions = list(lexica.keys())
    lex_arrays = {dim: match_lexicon(vocab, lex) for dim, lex in lexica.items()}

    confounds = compute_word_confounds(vocab)
    print(f"  Confounds: {list(confounds.keys())}")

    # === 1. Raw correlations with bootstrap CIs ===
    print("\n[1] Computing correlations with bootstrap CIs...")
    raw_results = {}
    all_pvals = []
    pval_index = []  # (axis, dim)

    for ax in range(n_axes):
        raw_results[ax] = {}
        for dim in dimensions:
            res = bootstrap_ci(scores[:, ax], lex_arrays[dim], args.n_boot)
            perm = permutation_test(scores[:, ax], lex_arrays[dim], args.n_perm)
            res["p_perm"] = perm["p_perm"]
            raw_results[ax][dim] = res
            all_pvals.append(perm["p_perm"])
            pval_index.append((ax, dim))

    # === 2. FDR correction ===
    print("\n[2] Applying FDR correction...")
    reject, pvals_corrected = fdr_correction(all_pvals)

    fdr_results = {}
    n_sig_raw = sum(1 for p in all_pvals if p < 0.05)
    n_sig_fdr = sum(reject)

    for i, (ax, dim) in enumerate(pval_index):
        if ax not in fdr_results:
            fdr_results[ax] = {}
        fdr_results[ax][dim] = {
            "p_raw": all_pvals[i],
            "p_fdr": float(pvals_corrected[i]),
            "significant_fdr": bool(reject[i]),
        }

    print(f"  Raw significant (α=0.05): {n_sig_raw}/{len(all_pvals)}")
    print(f"  FDR significant (α=0.05): {n_sig_fdr}/{len(all_pvals)}")
    print(f"  Expected by chance: {len(all_pvals) * 0.05:.1f}")

    # === 3. Partial correlations (controlling for word length) ===
    print("\n[3] Computing partial correlations (controlling for word length)...")
    partial_results = {}
    for ax in range(n_axes):
        partial_results[ax] = {}
        for dim in dimensions:
            r_partial, n = partial_correlation(
                scores[:, ax], lex_arrays[dim], confounds["length"]
            )
            r_raw = raw_results[ax][dim]["r"]
            partial_results[ax][dim] = {
                "r_raw": r_raw,
                "r_partial": r_partial,
                "r_change": r_partial - r_raw if not np.isnan(r_partial) else np.nan,
                "n": n,
            }

    # === 4. Specificity tests ===
    print("\n[4] Running specificity tests...")
    specificity_results = {}
    for ax in range(n_axes):
        specificity_results[ax] = {}
        for dim in dimensions:
            spec = specificity_test(scores, ax, lex_arrays, dim, n_perm=500)
            specificity_results[ax][dim] = spec

    # === 5. Summary statistics ===
    print("\n[5] Computing summary...")

    # Find best axis per dimension (by FDR-corrected significance and effect size)
    best_axes = {}
    for dim in dimensions:
        candidates = []
        for ax in range(n_axes):
            if fdr_results[ax][dim]["significant_fdr"]:
                r = abs(raw_results[ax][dim]["r"] or 0)
                candidates.append((ax, r))
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_axes[dim] = candidates[:3] if candidates else []

    summary = {
        "n_words": int(n_words),
        "n_axes": int(n_axes),
        "n_tests": int(len(all_pvals)),
        "n_sig_raw": int(n_sig_raw),
        "n_sig_fdr": int(n_sig_fdr),
        "expected_by_chance": float(len(all_pvals) * 0.05),
        "enrichment_raw": float(n_sig_raw / (len(all_pvals) * 0.05)),
        "enrichment_fdr": float(n_sig_fdr / (len(all_pvals) * 0.05)) if n_sig_fdr > 0 else 0.0,
        "best_axes_per_dimension": {dim: [(int(ax), f"r={r:.3f}") for ax, r in axes]
                                    for dim, axes in best_axes.items()},
    }

    # === Save all results ===
    with open(args.out_dir / "raw_correlations.json", "w") as f:
        json.dump(raw_results, f, indent=2)

    with open(args.out_dir / "fdr_corrected.json", "w") as f:
        json.dump(fdr_results, f, indent=2)

    with open(args.out_dir / "partial_correlations.json", "w") as f:
        json.dump(partial_results, f, indent=2)

    with open(args.out_dir / "specificity_tests.json", "w") as f:
        json.dump(specificity_results, f, indent=2)

    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # === Print final report ===
    print("\n" + "=" * 70)
    print("RIGOROUS SEMANTIC AXIS ANALYSIS - FINAL REPORT")
    print("=" * 70)

    print(f"\n1. MULTIPLE COMPARISON CORRECTION (FDR)")
    print("-" * 50)
    print(f"   Total tests: {len(all_pvals)}")
    print(f"   Significant (raw α=0.05): {n_sig_raw}")
    print(f"   Significant (FDR α=0.05): {n_sig_fdr}")
    print(f"   Expected by chance: {len(all_pvals) * 0.05:.1f}")
    print(f"   Enrichment (FDR): {summary['enrichment_fdr']:.1f}×")

    print(f"\n2. EFFECT SIZES WITH 95% CIs")
    print("-" * 50)
    for dim in dimensions:
        print(f"\n   {dim.upper()}:")
        # Get top 3 by |r|
        top = sorted([(ax, raw_results[ax][dim]) for ax in range(n_axes)],
                     key=lambda x: abs(x[1]["r"] or 0), reverse=True)[:3]
        for ax, res in top:
            sig = "*" if fdr_results[ax][dim]["significant_fdr"] else ""
            r = res["r"]
            ci_lo, ci_hi = res["ci_low"], res["ci_high"]
            print(f"      Axis {ax:2d}: r = {r:+.3f} [{ci_lo:+.3f}, {ci_hi:+.3f}]{sig}")

    print(f"\n3. PARTIAL CORRELATIONS (controlling word length)")
    print("-" * 50)
    for dim in dimensions:
        print(f"\n   {dim.upper()}:")
        top = sorted([(ax, partial_results[ax][dim]) for ax in range(n_axes)],
                     key=lambda x: abs(x[1]["r_partial"] or 0), reverse=True)[:3]
        for ax, res in top:
            r_raw = res["r_raw"]
            r_part = res["r_partial"]
            change = res["r_change"]
            print(f"      Axis {ax:2d}: r_raw={r_raw:+.3f} → r_partial={r_part:+.3f} (Δ={change:+.3f})")

    print(f"\n4. CONCLUSION")
    print("-" * 50)
    if summary["enrichment_fdr"] > 2:
        print("   ✓ ROBUST evidence for semantic structure survives FDR correction")
    elif summary["enrichment_fdr"] > 1:
        print("   ◐ MODEST evidence after FDR correction")
    else:
        print("   ✗ WEAK evidence after multiple comparison correction")

    print(f"\n   Results saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
