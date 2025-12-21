"""
Full confound control analysis for semantic axes.

Controls for:
1. Word length (character count)
2. Log word frequency (from corpus)
3. Surprisal proxy (GPT embedding change)

Tests whether semantic axis correlations survive after partialling out ALL confounds.

Run:
python analysis/semantic_axis_full_confounds.py \
  --word-axes-root outputs/word_axes_ica \
  --lexica-root metadata/lexica \
  --confounds-json metadata/lexica/confounds.json \
  --out-dir outputs/semantic_full_confounds
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests


def clean_word(w: str) -> str:
    return re.sub(r'\s+', '', str(w).strip())


def load_vocab_scores(root: pathlib.Path) -> Tuple[List[str], np.ndarray]:
    vocab = np.load(root / "vocab.npy", allow_pickle=True).tolist()
    scores_path = root / "ica_scores.npy" if (root / "ica_scores.npy").exists() else root / "pca_scores.npy"
    scores = np.load(scores_path)
    return vocab, scores


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


def load_confounds(confounds_path: pathlib.Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Load word frequency and surprisal confounds."""
    with open(confounds_path) as f:
        data = json.load(f)

    logfreq = {}
    surprisal = {}

    for word, vals in data.items():
        w = clean_word(word)
        if vals.get("logfreq") is not None:
            logfreq[w] = vals["logfreq"]
        if vals.get("surprisal") is not None:
            surprisal[w] = vals["surprisal"]

    return logfreq, surprisal


def match_to_vocab(vocab: List[str], lexicon: Dict[str, float]) -> np.ndarray:
    arr = np.full(len(vocab), np.nan)
    for i, w in enumerate(vocab):
        w_clean = clean_word(w)
        if w_clean in lexicon:
            arr[i] = lexicon[w_clean]
    return arr


def partial_correlation(x: np.ndarray, y: np.ndarray, confounds: np.ndarray) -> Tuple[float, float, int]:
    """Partial correlation controlling for confounds."""
    if confounds.ndim == 1:
        confounds = confounds.reshape(-1, 1)

    valid = ~np.isnan(x) & ~np.isnan(y)
    for i in range(confounds.shape[1]):
        valid &= ~np.isnan(confounds[:, i])

    if valid.sum() < 30:
        return np.nan, np.nan, int(valid.sum())

    x_v = x[valid]
    y_v = y[valid]
    c_v = confounds[valid]

    C = np.column_stack([np.ones(len(x_v)), c_v])
    from numpy.linalg import lstsq
    x_resid = x_v - C @ lstsq(C, x_v, rcond=None)[0]
    y_resid = y_v - C @ lstsq(C, y_v, rcond=None)[0]

    r, p = pearsonr(x_resid, y_resid)
    return float(r), float(p), int(valid.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--word-axes-root", type=pathlib.Path, required=True)
    p.add_argument("--lexica-root", type=pathlib.Path, default=pathlib.Path("metadata/lexica"))
    p.add_argument("--confounds-json", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
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
    lex_arrays = {dim: match_to_vocab(vocab, lex) for dim, lex in lexica.items()}

    print("[load] Loading confounds...")
    logfreq_dict, surprisal_dict = load_confounds(args.confounds_json)
    print(f"  Logfreq: {len(logfreq_dict)} words")
    print(f"  Surprisal: {len(surprisal_dict)} words")

    # Create confound arrays
    word_length = np.array([len(clean_word(w)) for w in vocab], dtype=float)
    logfreq = match_to_vocab(vocab, logfreq_dict)
    surprisal = match_to_vocab(vocab, surprisal_dict)

    print(f"\n[match] Confound coverage in vocab:")
    print(f"  Word length: {(~np.isnan(word_length)).sum()} / {n_words}")
    print(f"  Logfreq: {(~np.isnan(logfreq)).sum()} / {n_words}")
    print(f"  Surprisal: {(~np.isnan(surprisal)).sum()} / {n_words}")

    # Check confound-lexicon correlations
    print("\n[check] Confound-Lexicon correlations:")
    for dim in dimensions:
        valid = ~np.isnan(lex_arrays[dim]) & ~np.isnan(logfreq) & ~np.isnan(surprisal)
        if valid.sum() > 30:
            r_len, _ = pearsonr(word_length[valid], lex_arrays[dim][valid])
            r_freq, _ = pearsonr(logfreq[valid], lex_arrays[dim][valid])
            r_surp, _ = pearsonr(surprisal[valid], lex_arrays[dim][valid])
            print(f"  {dim}:")
            print(f"    ~ word_length: r={r_len:+.3f}")
            print(f"    ~ logfreq:     r={r_freq:+.3f}")
            print(f"    ~ surprisal:   r={r_surp:+.3f}")

    # Check confound inter-correlations
    print("\n[check] Confound inter-correlations:")
    valid_all = ~np.isnan(logfreq) & ~np.isnan(surprisal)
    if valid_all.sum() > 30:
        r_lf, _ = pearsonr(word_length[valid_all], logfreq[valid_all])
        r_ls, _ = pearsonr(word_length[valid_all], surprisal[valid_all])
        r_fs, _ = pearsonr(logfreq[valid_all], surprisal[valid_all])
        print(f"  word_length ~ logfreq:   r={r_lf:+.3f}")
        print(f"  word_length ~ surprisal: r={r_ls:+.3f}")
        print(f"  logfreq ~ surprisal:     r={r_fs:+.3f}")

    # Build confound matrix
    confounds = np.column_stack([word_length, logfreq, surprisal])
    confound_names = ["word_length", "logfreq", "surprisal"]

    # Run partial correlation analysis
    print(f"\n[analysis] Partial correlations controlling for: {confound_names}")

    results = {"raw": {}, "partial": {}, "comparison": {}}

    for dim in dimensions:
        print(f"\n  {dim.upper()}")
        print(f"  {'Axis':>5} {'Raw r':>10} {'Partial r':>12} {'Δr':>10} {'%change':>10} {'p_partial':>12} {'n':>6}")

        dim_results = []

        for ax in range(n_axes):
            # Raw correlation
            valid_raw = ~np.isnan(lex_arrays[dim])
            if valid_raw.sum() > 30:
                r_raw, _ = pearsonr(scores[valid_raw, ax], lex_arrays[dim][valid_raw])
            else:
                r_raw = np.nan

            # Partial correlation (all confounds)
            r_partial, p_partial, n = partial_correlation(
                scores[:, ax], lex_arrays[dim], confounds
            )

            delta_r = r_partial - r_raw if not (np.isnan(r_partial) or np.isnan(r_raw)) else np.nan
            pct_change = (delta_r / abs(r_raw) * 100) if (r_raw != 0 and not np.isnan(delta_r)) else np.nan

            dim_results.append({
                "axis": int(ax),
                "r_raw": float(r_raw) if not np.isnan(r_raw) else None,
                "r_partial": float(r_partial) if not np.isnan(r_partial) else None,
                "delta_r": float(delta_r) if not np.isnan(delta_r) else None,
                "pct_change": float(pct_change) if not np.isnan(pct_change) else None,
                "p_partial": float(p_partial) if not np.isnan(p_partial) else None,
                "n": n,
            })

        # Sort by absolute partial r
        dim_results.sort(key=lambda x: abs(x["r_partial"] or 0), reverse=True)

        # Print top 5
        for res in dim_results[:5]:
            ax = res["axis"]
            rr = res["r_raw"] or 0
            rp = res["r_partial"] or 0
            dr = res["delta_r"]
            pc = res["pct_change"]
            pp = res["p_partial"] or 1
            n = res["n"]

            dr_str = f"{dr:+.3f}" if dr is not None else "N/A"
            pc_str = f"{pc:+.1f}%" if pc is not None else "N/A"

            print(f"  {ax:5d} {rr:+10.3f} {rp:+12.3f} {dr_str:>10} {pc_str:>10} {pp:12.2e} {n:6d}")

        results["partial"][dim] = dim_results

    # Summary
    print("\n" + "=" * 80)
    print("FULL CONFOUND CONTROL SUMMARY")
    print("=" * 80)

    print(f"\nConfounds controlled: {confound_names}")

    # Count survivors
    total_raw_sig = 0
    total_partial_sig = 0

    for dim in dimensions:
        raw_sig = sum(1 for r in results["partial"][dim]
                      if r["r_raw"] is not None and abs(r["r_raw"]) > 0.05)
        partial_sig = sum(1 for r in results["partial"][dim]
                          if r["r_partial"] is not None and r["p_partial"] is not None
                          and abs(r["r_partial"]) > 0.05 and r["p_partial"] < 0.05)

        # Compute mean absolute change
        changes = [abs(r["delta_r"]) for r in results["partial"][dim]
                   if r["delta_r"] is not None and abs(r["r_raw"] or 0) > 0.05]
        mean_change = np.mean(changes) if changes else 0

        survival = partial_sig / max(raw_sig, 1)
        print(f"\n{dim}:")
        print(f"  Raw |r| > 0.05: {raw_sig}")
        print(f"  Partial |r| > 0.05 & p < 0.05: {partial_sig}")
        print(f"  Survival rate: {survival*100:.0f}%")
        print(f"  Mean |Δr|: {mean_change:.4f}")

        total_raw_sig += raw_sig
        total_partial_sig += partial_sig

        results["comparison"][dim] = {
            "n_raw_sig": raw_sig,
            "n_partial_sig": partial_sig,
            "survival_rate": survival,
            "mean_abs_delta_r": mean_change,
        }

    # FDR correction
    print("\n[FDR] Applying FDR correction to partial p-values...")
    all_p = []
    for dim in dimensions:
        for r in results["partial"][dim]:
            if r["p_partial"] is not None:
                all_p.append(r["p_partial"])

    reject, p_fdr, _, _ = multipletests(all_p, method='fdr_bh')
    n_sig_fdr = sum(reject)

    print(f"  Total tests: {len(all_p)}")
    print(f"  Significant after FDR: {n_sig_fdr}")
    print(f"  Expected by chance: {len(all_p) * 0.05:.1f}")
    print(f"  Enrichment: {n_sig_fdr / (len(all_p) * 0.05):.1f}x")

    results["fdr"] = {
        "n_tests": len(all_p),
        "n_sig": int(n_sig_fdr),
        "expected": float(len(all_p) * 0.05),
        "enrichment": float(n_sig_fdr / (len(all_p) * 0.05)) if n_sig_fdr > 0 else 0,
    }

    # Final verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if n_sig_fdr >= 5 and results["fdr"]["enrichment"] > 2:
        print("✓ Semantic effects SURVIVE full confound control")
        print(f"  {n_sig_fdr} axis-dimension pairs significant after controlling for")
        print(f"  word length, frequency, and surprisal")
    elif n_sig_fdr >= 3:
        print("◐ Semantic effects PARTIALLY survive confound control")
    else:
        print("✗ Semantic effects largely explained by confounds")

    # Save
    with open(args.out_dir / "full_confound_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
