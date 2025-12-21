"""
Box 5: Clean LLM Projection Adapter for TinyLlama (Y-Leakage Free)

Same methodology as word_llm_adapter_clean.py but using TinyLlama embeddings.
TinyLlama: 2048 dims, 23 layers vs GPT-2: 1024 dims, 25 layers

Run:
python analysis/word_llm_adapter_tinyllama.py \
  --atlas-root outputs/encoding/plv_glm \
  --tinyllama-root derivatives/annotations/embeddings/tinyllama/word-level \
  --time-align-root derivatives/annotations/time_align/word-level \
  --out-dir outputs/llm_adapter_tinyllama \
  --n-components 20
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import scipy.io
from sklearn.decomposition import FastICA
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests


# Genuine axes from confound analysis
GENUINE_AXES_INDICES = [2, 10, 12, 13, 15, 16, 19]


def clean_word(w: str) -> str:
    return re.sub(r'\s+', '', str(w).strip())


def load_sub_atlas(sub_dir: pathlib.Path):
    """Load per-subject word atlas."""
    atlas = np.load(sub_dir / "word_atlas.npy")
    meta = json.load((sub_dir / "word_vocab.json").open())
    vocab = meta["vocab"]
    return vocab, atlas


def build_split_atlas(atlas_root: pathlib.Path, subjects: List[str]) -> Tuple[List[str], np.ndarray]:
    """Build word atlas from a subset of subjects."""
    union = set()
    per_sub = {}

    for sub in subjects:
        sub_dir = atlas_root / f"sub-{sub}"
        if not sub_dir.exists():
            print(f"[warn] Subject {sub} not found, skipping")
            continue
        vocab, atlas = load_sub_atlas(sub_dir)
        per_sub[sub] = (vocab, atlas)
        union.update(vocab)

    vocab_union = sorted(list(union))
    word_to_idx = {w: i for i, w in enumerate(vocab_union)}

    # Average across subjects
    sum_mat = None
    counts = np.zeros(len(vocab_union), dtype=np.int32)

    for sub, (vocab, atlas) in per_sub.items():
        mat = np.zeros((len(vocab_union), atlas.shape[1]), dtype=np.float32)
        for i, w in enumerate(vocab):
            j = word_to_idx[w]
            mat[j] = atlas[i]
        if sum_mat is None:
            sum_mat = mat
        else:
            sum_mat += mat
        counts += np.array([w in vocab for w in vocab_union], dtype=np.int32)

    counts = np.maximum(counts, 1)
    avg_mat = sum_mat / counts[:, None]

    return vocab_union, avg_mat


def run_ica(atlas: np.ndarray, n_components: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Run ICA on word atlas."""
    model = FastICA(n_components=n_components, random_state=42, max_iter=500)
    scores = model.fit_transform(atlas)  # [n_words, n_components]
    components = model.mixing_.T  # [n_components, feature_dim]
    return scores, components


def load_story_tinyllama_embeddings(npy_path: pathlib.Path, layer: int = -1) -> np.ndarray:
    """Load TinyLlama embeddings for a story.

    Format: (n_layers, n_words, hidden_dim) = (23, n_words, 2048)
    """
    data = np.load(str(npy_path))
    # Use specified layer (default: last layer)
    return data[layer]


def load_story_words(time_align_path: pathlib.Path) -> List[str]:
    mat = scipy.io.loadmat(str(time_align_path))
    words = mat['word'].flatten()
    return [clean_word(str(w)) for w in words]


def build_word_type_embeddings(
    tinyllama_root: pathlib.Path,
    time_align_root: pathlib.Path,
    hidden_dim: int = 2048,
) -> Dict[str, np.ndarray]:
    """Build word-type-level TinyLlama embeddings by averaging across all positions."""
    word_emb_sums = defaultdict(lambda: np.zeros(hidden_dim))
    word_counts = defaultdict(int)

    for story_id in range(1, 61):
        # TinyLlama files: story_X_word_tinyllama_0-22_2048.npy
        tinyllama_files = list(tinyllama_root.glob(f"story_{story_id}_word_tinyllama*.npy"))
        if not tinyllama_files:
            continue
        tinyllama_path = tinyllama_files[0]

        time_path = time_align_root / f"story_{story_id}_word_time.mat"
        if not time_path.exists():
            continue

        embeddings = load_story_tinyllama_embeddings(tinyllama_path, layer=-1)
        words = load_story_words(time_path)

        min_len = min(len(words), len(embeddings))
        for word, emb in zip(words[:min_len], embeddings[:min_len]):
            if word:
                word_emb_sums[word] += emb
                word_counts[word] += 1

    result = {}
    for word in word_emb_sums:
        result[word] = word_emb_sums[word] / word_counts[word]

    return result


def build_matched_data(
    word_embeddings: Dict[str, np.ndarray],
    vocab: List[str],
    ica_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Match TinyLlama embeddings to word atlas vocabulary."""
    X_list = []
    Y_list = []
    matched_words = []

    vocab_clean = {clean_word(w): i for i, w in enumerate(vocab)}

    for word, emb in word_embeddings.items():
        word_clean = clean_word(word)
        if word_clean in vocab_clean:
            idx = vocab_clean[word_clean]
            X_list.append(emb)
            Y_list.append(ica_scores[idx])
            matched_words.append(word)

    return np.array(X_list), np.array(Y_list), matched_words


def align_ica_signs(scores_train: np.ndarray, scores_test: np.ndarray, vocab_train: List[str], vocab_test: List[str]) -> np.ndarray:
    """Align ICA component signs between train and test based on shared words."""
    train_idx = {clean_word(w): i for i, w in enumerate(vocab_train)}
    test_idx = {clean_word(w): i for i, w in enumerate(vocab_test)}

    shared = set(train_idx.keys()) & set(test_idx.keys())
    if len(shared) < 100:
        print(f"[warn] Only {len(shared)} shared words for sign alignment")
        return scores_test

    train_shared_idx = [train_idx[w] for w in shared]
    test_shared_idx = [test_idx[w] for w in shared]

    n_components = scores_train.shape[1]
    aligned = scores_test.copy()

    for c in range(n_components):
        r, _ = pearsonr(scores_train[train_shared_idx, c], scores_test[test_shared_idx, c])
        if r < 0:
            aligned[:, c] *= -1

    return aligned


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas-root", type=pathlib.Path, required=True,
                   help="Per-subject word atlas root (outputs/encoding/plv_glm)")
    p.add_argument("--tinyllama-root", type=pathlib.Path, required=True)
    p.add_argument("--time-align-root", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--n-components", type=int, default=20)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Define subject splits
    odd_subjects = ["01", "03", "05", "07", "09", "11"]
    even_subjects = ["02", "04", "06", "08", "10", "12"]

    print("=" * 70, flush=True)
    print("TINYLLAMA CLEAN ADAPTER (Y-LEAKAGE FREE)", flush=True)
    print("=" * 70, flush=True)
    print("\nModel: TinyLlama 1.1B (2048 dims, 23 layers)", flush=True)
    print("Comparison: GPT-2 (1024 dims, 25 layers)", flush=True)

    # Build separate atlases
    print(f"\n[build] Building ODD subjects atlas ({odd_subjects})...", flush=True)
    vocab_odd, atlas_odd = build_split_atlas(args.atlas_root, odd_subjects)
    print(f"  {len(vocab_odd)} words, {atlas_odd.shape[1]} features", flush=True)

    print(f"\n[build] Building EVEN subjects atlas ({even_subjects})...", flush=True)
    vocab_even, atlas_even = build_split_atlas(args.atlas_root, even_subjects)
    print(f"  {len(vocab_even)} words, {atlas_even.shape[1]} features", flush=True)

    # Run ICA on each
    print(f"\n[ica] Running ICA with {args.n_components} components...", flush=True)
    scores_odd, _ = run_ica(atlas_odd, args.n_components)
    scores_even, _ = run_ica(atlas_even, args.n_components)

    # Align ICA signs based on shared vocabulary
    print("[align] Aligning ICA component signs...", flush=True)
    scores_even_aligned = align_ica_signs(scores_odd, scores_even, vocab_odd, vocab_even)

    # Load TinyLlama embeddings
    print("\n[load] Loading TinyLlama embeddings...", flush=True)
    tinyllama_embeddings = build_word_type_embeddings(args.tinyllama_root, args.time_align_root)
    print(f"  {len(tinyllama_embeddings)} word types", flush=True)

    # Check embedding dimension
    sample_emb = next(iter(tinyllama_embeddings.values()))
    print(f"  Embedding dim: {len(sample_emb)}", flush=True)

    # Build matched data for each split
    print("\n[match] Matching to atlases...", flush=True)
    X_odd, Y_odd, words_odd = build_matched_data(tinyllama_embeddings, vocab_odd, scores_odd)
    X_even, Y_even, words_even = build_matched_data(tinyllama_embeddings, vocab_even, scores_even_aligned)

    print(f"  ODD atlas: {len(words_odd)} matched words", flush=True)
    print(f"  EVEN atlas: {len(words_even)} matched words", flush=True)

    # Find shared words between splits
    odd_set = set(words_odd)
    even_set = set(words_even)
    shared_words = odd_set & even_set
    print(f"  Shared words: {len(shared_words)}", flush=True)

    # Create matched arrays for shared words only
    odd_word_idx = {w: i for i, w in enumerate(words_odd)}
    even_word_idx = {w: i for i, w in enumerate(words_even)}

    shared_list = sorted(list(shared_words))
    X_shared = np.array([X_odd[odd_word_idx[w]] for w in shared_list])
    Y_odd_shared = np.array([Y_odd[odd_word_idx[w]] for w in shared_list])
    Y_even_shared = np.array([Y_even[even_word_idx[w]] for w in shared_list])

    print(f"  Shared for training: {len(shared_list)} words")

    # Train adapter: ODD → EVEN and EVEN → ODD
    print("\n" + "=" * 70)
    print("TRAINING TINYLLAMA CLEAN ADAPTER")
    print("=" * 70)

    # Fewer alphas for faster training (2048 dim is much larger than GPT-2's 1024)
    alphas = np.logspace(-1, 5, 20)
    results = {"model": "TinyLlama-1.1B", "fold_1": {}, "fold_2": {}, "combined": {}}

    n_axes = args.n_components

    # Fold 1: Train on ODD, test on EVEN
    print("\n[fold1] Train on ODD subjects, test on EVEN subjects", flush=True)
    print(f"{'Axis':>6} {'r_train':>10} {'r_test':>10} {'R²_test':>10} {'p_test':>12}", flush=True)

    fold1_results = []
    for ax in range(n_axes):
        model = RidgeCV(alphas=alphas, cv=3)  # Fewer folds for speed
        model.fit(X_shared, Y_odd_shared[:, ax])

        y_pred_train = model.predict(X_shared)
        y_pred_test = model.predict(X_shared)

        r_train, _ = pearsonr(y_pred_train, Y_odd_shared[:, ax])
        r_test, p_test = pearsonr(y_pred_test, Y_even_shared[:, ax])
        r2_test = r_test ** 2 if r_test > 0 else -(r_test ** 2)

        fold1_results.append({
            "axis": ax,
            "r_train": float(r_train),
            "r_test": float(r_test),
            "r2_test": float(r2_test),
            "p_test": float(p_test),
        })

        if ax in GENUINE_AXES_INDICES:
            print(f"{ax:6d} {r_train:+10.4f} {r_test:+10.4f} {r2_test:10.4f} {p_test:12.2e}")

    results["fold_1"]["per_axis"] = fold1_results

    # Fold 2: Train on EVEN, test on ODD
    print("\n[fold2] Train on EVEN subjects, test on ODD subjects", flush=True)
    print(f"{'Axis':>6} {'r_train':>10} {'r_test':>10} {'R²_test':>10} {'p_test':>12}", flush=True)

    fold2_results = []
    for ax in range(n_axes):
        model = RidgeCV(alphas=alphas, cv=3)  # Fewer folds for speed
        model.fit(X_shared, Y_even_shared[:, ax])

        y_pred_train = model.predict(X_shared)
        y_pred_test = model.predict(X_shared)

        r_train, _ = pearsonr(y_pred_train, Y_even_shared[:, ax])
        r_test, p_test = pearsonr(y_pred_test, Y_odd_shared[:, ax])
        r2_test = r_test ** 2 if r_test > 0 else -(r_test ** 2)

        fold2_results.append({
            "axis": ax,
            "r_train": float(r_train),
            "r_test": float(r_test),
            "r2_test": float(r2_test),
            "p_test": float(p_test),
        })

        if ax in GENUINE_AXES_INDICES:
            print(f"{ax:6d} {r_train:+10.4f} {r_test:+10.4f} {r2_test:10.4f} {p_test:12.2e}")

    results["fold_2"]["per_axis"] = fold2_results

    # Combined results
    print("\n" + "=" * 70)
    print("COMBINED RESULTS (AVERAGE OF BOTH FOLDS)")
    print("=" * 70)
    print(f"\n{'Axis':>6} {'r_test_avg':>12} {'R²_test_avg':>12} {'p_min':>12} {'Genuine':>10}")

    combined = []
    all_p = []
    for ax in range(n_axes):
        r1 = fold1_results[ax]["r_test"]
        r2 = fold2_results[ax]["r_test"]
        p1 = fold1_results[ax]["p_test"]
        p2 = fold2_results[ax]["p_test"]

        r_avg = (r1 + r2) / 2
        r2_avg = r_avg ** 2 if r_avg > 0 else -(r_avg ** 2)
        p_min = min(p1, p2)

        is_genuine = ax in GENUINE_AXES_INDICES

        combined.append({
            "axis": ax,
            "r_fold1": float(r1),
            "r_fold2": float(r2),
            "r_avg": float(r_avg),
            "r2_avg": float(r2_avg),
            "p_fold1": float(p1),
            "p_fold2": float(p2),
            "p_min": float(p_min),
            "is_genuine": is_genuine,
        })

        all_p.append(p_min)

        marker = "✓" if is_genuine else ""
        print(f"{ax:6d} {r_avg:+12.4f} {r2_avg:12.4f} {p_min:12.2e} {marker:>10}")

    results["combined"]["per_axis"] = combined

    # FDR correction
    reject, p_fdr, _, _ = multipletests(all_p, method='fdr_bh')
    n_sig = sum(reject)

    print(f"\n[fdr] Significant after FDR: {n_sig}/{n_axes}")

    # Summary for genuine axes only
    print("\n" + "=" * 70)
    print("SUMMARY: GENUINE AXES ONLY")
    print("=" * 70)

    genuine_r = [combined[ax]["r_avg"] for ax in GENUINE_AXES_INDICES]
    genuine_r2 = [combined[ax]["r2_avg"] for ax in GENUINE_AXES_INDICES]
    genuine_p = [combined[ax]["p_min"] for ax in GENUINE_AXES_INDICES]
    genuine_sig = sum(1 for p in genuine_p if p < 0.05)

    mean_r = np.mean(np.abs(genuine_r))
    mean_r2 = np.mean([r2 for r2 in genuine_r2 if r2 > 0])

    print(f"\nGenuine axes: {GENUINE_AXES_INDICES}")
    print(f"Mean |r| (cross-subject): {mean_r:.4f}")
    print(f"Mean R² (positive only): {mean_r2:.4f}")
    print(f"Axes with p < 0.05: {genuine_sig}/{len(GENUINE_AXES_INDICES)}")

    results["summary"] = {
        "n_shared_words": len(shared_list),
        "n_axes_total": n_axes,
        "n_axes_fdr_sig": int(n_sig),
        "genuine_axes": GENUINE_AXES_INDICES,
        "genuine_mean_r": float(mean_r),
        "genuine_mean_r2": float(mean_r2) if mean_r2 > 0 else 0.0,
        "genuine_n_sig": genuine_sig,
    }

    # Save
    with open(args.out_dir / "tinyllama_adapter_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {args.out_dir}/")

    return results


if __name__ == "__main__":
    main()
