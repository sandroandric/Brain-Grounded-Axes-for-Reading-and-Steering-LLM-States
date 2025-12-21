"""
Rigorous cross-validation of semantic axes.

Method:
1. Split 12 subjects into train (6) and test (6)
2. Compute averaged word atlas from training subjects only
3. Fit ICA on training atlas
4. Project test subjects' atlases onto training ICA components
5. Evaluate correlations with lexica in held-out subjects
6. Repeat with swapped train/test for full cross-validation

This tests whether semantic structure generalizes across subjects.

Run:
PYTHONPATH=. python analysis/semantic_axis_crossval.py \
  --encoding-root outputs/encoding/plv_glm \
  --lexica-root metadata/lexica \
  --out-dir outputs/semantic_crossval
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import FastICA


def clean_word(w: str) -> str:
    return re.sub(r'\s+', '', str(w).strip())


def load_subject_atlas(encoding_root: pathlib.Path, sub: str) -> Tuple[List[str], np.ndarray]:
    """Load word atlas for a single subject."""
    sub_dir = encoding_root / sub
    atlas = np.load(sub_dir / "word_atlas.npy")
    with open(sub_dir / "word_vocab.json") as f:
        data = json.load(f)
    vocab = data["vocab"]
    return vocab, atlas


def load_lexica(lexica_root: pathlib.Path) -> Dict[str, Dict[str, float]]:
    """Load all lexica."""
    import pandas as pd
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


def align_vocabs(vocabs: List[List[str]], atlases: List[np.ndarray]) -> Tuple[List[str], List[np.ndarray]]:
    """Align multiple subject atlases to common vocabulary."""
    # Find common vocabulary
    vocab_sets = [set(v) for v in vocabs]
    common_vocab = vocab_sets[0]
    for vs in vocab_sets[1:]:
        common_vocab = common_vocab.intersection(vs)
    common_vocab = sorted(list(common_vocab))

    # Build index mappings and aligned atlases
    aligned_atlases = []
    for vocab, atlas in zip(vocabs, atlases):
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        indices = [word_to_idx[w] for w in common_vocab]
        aligned_atlases.append(atlas[indices])

    return common_vocab, aligned_atlases


def compute_ica_axes(atlas: np.ndarray, n_components: int = 20, random_state: int = 42) -> Tuple[np.ndarray, FastICA]:
    """Compute ICA axes from word atlas."""
    ica = FastICA(n_components=n_components, random_state=random_state, max_iter=1000)
    scores = ica.fit_transform(atlas)
    return scores, ica


def project_to_axes(atlas: np.ndarray, ica: FastICA) -> np.ndarray:
    """Project atlas onto pre-computed ICA axes."""
    return ica.transform(atlas)


def match_lexicon(vocab: List[str], lexicon: Dict[str, float]) -> np.ndarray:
    """Match vocabulary to lexicon values."""
    arr = np.full(len(vocab), np.nan)
    for i, w in enumerate(vocab):
        w_clean = clean_word(w)
        if w_clean in lexicon:
            arr[i] = lexicon[w_clean]
    return arr


def evaluate_axes(scores: np.ndarray, lex_values: np.ndarray) -> List[Dict]:
    """Evaluate all axes against lexicon values."""
    valid = ~np.isnan(lex_values)
    if valid.sum() < 20:
        return [{"axis": k, "r": np.nan, "p": np.nan} for k in range(scores.shape[1])]

    results = []
    for k in range(scores.shape[1]):
        r, p = pearsonr(scores[valid, k], lex_values[valid])
        results.append({"axis": k, "r": float(r), "p": float(p), "n": int(valid.sum())})
    return results


def run_crossval_fold(train_subs: List[str], test_subs: List[str],
                      encoding_root: pathlib.Path, lexica: Dict[str, Dict[str, float]],
                      n_components: int = 20) -> Dict:
    """Run one fold of cross-validation."""
    print(f"  Loading train subjects: {train_subs}")

    # Load train subjects
    train_vocabs = []
    train_atlases = []
    for sub in train_subs:
        vocab, atlas = load_subject_atlas(encoding_root, sub)
        train_vocabs.append(vocab)
        train_atlases.append(atlas)

    # Align train vocabs
    train_vocab, train_aligned = align_vocabs(train_vocabs, train_atlases)
    print(f"    Train common vocab: {len(train_vocab)} words")

    # Average train atlases
    train_avg = np.mean(train_aligned, axis=0)

    # Fit ICA on train
    train_scores, ica = compute_ica_axes(train_avg, n_components)
    print(f"    ICA fitted on train ({n_components} components)")

    # Load test subjects
    print(f"  Loading test subjects: {test_subs}")
    test_vocabs = []
    test_atlases = []
    for sub in test_subs:
        vocab, atlas = load_subject_atlas(encoding_root, sub)
        test_vocabs.append(vocab)
        test_atlases.append(atlas)

    # Align test to same vocabulary as train
    # First align test subjects among themselves
    test_vocab, test_aligned = align_vocabs(test_vocabs, test_atlases)

    # Find intersection with train vocab
    common_vocab = sorted(list(set(train_vocab).intersection(set(test_vocab))))
    print(f"    Test-Train common vocab: {len(common_vocab)} words")

    # Re-index train and test to common vocab
    train_word_to_idx = {w: i for i, w in enumerate(train_vocab)}
    test_word_to_idx = {w: i for i, w in enumerate(test_vocab)}

    train_indices = [train_word_to_idx[w] for w in common_vocab]
    test_indices = [test_word_to_idx[w] for w in common_vocab]

    train_avg_common = train_avg[train_indices]
    test_avg_common = np.mean([a[test_indices] for a in test_aligned], axis=0)

    # Project train and test onto ICA space
    # We need to re-fit ICA on common vocab portion of train
    train_scores_common, ica_common = compute_ica_axes(train_avg_common, n_components)
    test_scores = project_to_axes(test_avg_common, ica_common)

    # Match lexica to common vocab
    lex_arrays = {dim: match_lexicon(common_vocab, lex) for dim, lex in lexica.items()}

    # Evaluate on train
    train_results = {}
    for dim in lexica:
        train_results[dim] = evaluate_axes(train_scores_common, lex_arrays[dim])

    # Evaluate on test (the critical test)
    test_results = {}
    for dim in lexica:
        test_results[dim] = evaluate_axes(test_scores, lex_arrays[dim])

    return {
        "train_subjects": train_subs,
        "test_subjects": test_subs,
        "common_vocab_size": len(common_vocab),
        "train_results": train_results,
        "test_results": test_results,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoding-root", type=pathlib.Path, required=True)
    p.add_argument("--lexica-root", type=pathlib.Path, default=pathlib.Path("metadata/lexica"))
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--n-components", type=int, default=20)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Get all subjects
    subjects = sorted([d.name for d in args.encoding_root.iterdir() if d.is_dir() and d.name.startswith("sub-")])
    print(f"Found {len(subjects)} subjects: {subjects}")

    # Load lexica
    lexica = load_lexica(args.lexica_root)
    dimensions = list(lexica.keys())
    print(f"Loaded lexica: {dimensions}")

    # Define cross-validation folds
    # Split 1: odd subjects train, even test
    # Split 2: even subjects train, odd test
    odd_subs = [s for s in subjects if int(s.split("-")[1]) % 2 == 1]
    even_subs = [s for s in subjects if int(s.split("-")[1]) % 2 == 0]

    print(f"\nFold 1: Train={odd_subs}, Test={even_subs}")
    print(f"Fold 2: Train={even_subs}, Test={odd_subs}")

    # Run cross-validation
    results = []

    print("\n=== Fold 1 ===")
    fold1 = run_crossval_fold(odd_subs, even_subs, args.encoding_root, lexica, args.n_components)
    results.append(fold1)

    print("\n=== Fold 2 ===")
    fold2 = run_crossval_fold(even_subs, odd_subs, args.encoding_root, lexica, args.n_components)
    results.append(fold2)

    # Aggregate results across folds
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 70)

    aggregated = {"by_dimension": {}, "summary": {}}

    for dim in dimensions:
        print(f"\n{dim.upper()}")
        print("-" * 40)

        # Collect test correlations across folds
        all_test_rs = defaultdict(list)
        all_train_rs = defaultdict(list)

        for fold in results:
            for axis_res in fold["test_results"][dim]:
                ax = axis_res["axis"]
                all_test_rs[ax].append(axis_res["r"])
            for axis_res in fold["train_results"][dim]:
                ax = axis_res["axis"]
                all_train_rs[ax].append(axis_res["r"])

        # Average across folds
        dim_results = []
        for ax in range(args.n_components):
            mean_test = np.nanmean(all_test_rs[ax])
            mean_train = np.nanmean(all_train_rs[ax])
            std_test = np.nanstd(all_test_rs[ax])

            # Check if effect replicates (same sign in both folds)
            signs = [np.sign(r) for r in all_test_rs[ax] if not np.isnan(r)]
            replicates = len(signs) == 2 and signs[0] == signs[1]

            dim_results.append({
                "axis": int(ax),
                "train_r_mean": float(mean_train),
                "test_r_mean": float(mean_test),
                "test_r_std": float(std_test),
                "test_rs": [float(r) for r in all_test_rs[ax]],
                "replicates": bool(replicates),
            })

        # Sort by absolute test r
        dim_results.sort(key=lambda x: abs(x["test_r_mean"]) if not np.isnan(x["test_r_mean"]) else 0, reverse=True)

        # Print top axes
        print(f"  {'Axis':>5} {'Train r':>10} {'Test r':>10} {'Test std':>10} {'Replicates':>12}")
        for res in dim_results[:5]:
            ax = res["axis"]
            tr = res["train_r_mean"]
            te = res["test_r_mean"]
            ts = res["test_r_std"]
            rep = "✓" if res["replicates"] else "✗"
            print(f"  {ax:5d} {tr:+10.3f} {te:+10.3f} {ts:10.3f} {rep:>12}")

        aggregated["by_dimension"][dim] = dim_results

    # Summary: count replicating axes
    print("\n" + "=" * 70)
    print("SUMMARY: AXES THAT REPLICATE ACROSS SUBJECTS")
    print("=" * 70)

    for dim in dimensions:
        replicating = [r for r in aggregated["by_dimension"][dim] if r["replicates"] and abs(r["test_r_mean"]) > 0.05]
        n_rep = len(replicating)
        print(f"\n{dim}: {n_rep} axes replicate with |r| > 0.05")
        if replicating:
            for r in replicating[:3]:
                print(f"  Axis {r['axis']}: test r = {r['test_r_mean']:+.3f} ± {r['test_r_std']:.3f}")

        aggregated["summary"][dim] = {
            "n_replicating": n_rep,
            "top_replicating": replicating[:5] if replicating else [],
        }

    # Overall assessment
    total_replicating = sum(aggregated["summary"][d]["n_replicating"] for d in dimensions)
    print(f"\nTotal replicating axis-dimension pairs: {total_replicating}")

    if total_replicating >= 3:
        print("✓ Semantic structure GENERALIZES across subjects")
    else:
        print("✗ Limited cross-subject generalization")

    # Save results
    with open(args.out_dir / "crossval_folds.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(args.out_dir / "crossval_aggregated.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"\nResults saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
