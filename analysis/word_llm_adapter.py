"""
Box 5: Word-Level LLM Projection Adapter

Train a linear adapter mapping GPT-2 embeddings to MEG-derived brain axes (word atlas).

Architecture:
- Input: GPT-2 last layer embedding (1024 dims) per word type
- Output: Predicted brain axis scores (ICA axes from word atlas)
- Method: Ridge regression with story-level cross-validation

Key insight: We map word-type-level representations.
- GPT embeddings are contextual (per position), so we average per word type
- ICA scores are already per word type (from word atlas)

Genuine axes (survive confound control):
- Concreteness: 2, 10, 12, 15, 19
- Arousal: 13, 15, 16, 19

Run:
python analysis/word_llm_adapter.py \
  --gpt-root derivatives/annotations/embeddings/gpt/word-level \
  --time-align-root derivatives/annotations/time_align/word-level \
  --word-axes-root outputs/word_axes_ica \
  --out-dir outputs/llm_adapter \
  --n-test-stories 12
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
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


# Genuine axes from confound analysis
GENUINE_AXES = {
    "concreteness": [2, 10, 12, 15, 19],
    "arousal": [13, 15, 16, 19],
}
ALL_GENUINE = sorted(set(ax for axes in GENUINE_AXES.values() for ax in axes))


def clean_word(w: str) -> str:
    """Clean word string for matching."""
    return re.sub(r'\s+', '', str(w).strip())


def load_word_axes(root: pathlib.Path) -> Tuple[List[str], np.ndarray]:
    """Load word atlas vocabulary and ICA scores."""
    vocab = np.load(root / "vocab.npy", allow_pickle=True).tolist()
    scores = np.load(root / "ica_scores.npy")
    return vocab, scores


def load_story_gpt_embeddings(gpt_path: pathlib.Path, layer: int = -1) -> np.ndarray:
    """
    Load GPT embeddings for a story.

    Args:
        gpt_path: Path to GPT .mat file
        layer: Which layer to use (-1 = last layer)

    Returns:
        embeddings: shape (n_words, 1024)
    """
    mat = scipy.io.loadmat(str(gpt_path))
    data = mat['data']  # shape (25, n_words, 1024)
    return data[layer]  # shape (n_words, 1024)


def load_story_words(time_align_path: pathlib.Path) -> List[str]:
    """Load word list for a story."""
    mat = scipy.io.loadmat(str(time_align_path))
    words = mat['word'].flatten()
    return [clean_word(str(w)) for w in words]


def build_word_type_embeddings(
    gpt_root: pathlib.Path,
    time_align_root: pathlib.Path,
    story_ids: List[int],
) -> Dict[str, Tuple[np.ndarray, int]]:
    """
    Build word-type-level GPT embeddings by averaging across positions.

    Returns:
        word_embeddings: {word: (mean_embedding, count)}
    """
    word_emb_sums = defaultdict(lambda: np.zeros(1024))
    word_counts = defaultdict(int)

    for story_id in story_ids:
        # Find GPT file
        gpt_files = list(gpt_root.glob(f"story_{story_id}_word_gpt*.mat"))
        if not gpt_files:
            print(f"[warn] No GPT file for story {story_id}")
            continue
        gpt_path = gpt_files[0]

        # Find time align file
        time_path = time_align_root / f"story_{story_id}_word_time.mat"
        if not time_path.exists():
            print(f"[warn] No time align for story {story_id}")
            continue

        # Load data
        embeddings = load_story_gpt_embeddings(gpt_path, layer=-1)
        words = load_story_words(time_path)

        if len(words) != len(embeddings):
            print(f"[warn] Story {story_id}: {len(words)} words vs {len(embeddings)} embeddings")
            min_len = min(len(words), len(embeddings))
            words = words[:min_len]
            embeddings = embeddings[:min_len]

        # Aggregate per word type
        for word, emb in zip(words, embeddings):
            if word:  # Skip empty
                word_emb_sums[word] += emb
                word_counts[word] += 1

    # Compute means
    result = {}
    for word in word_emb_sums:
        result[word] = (word_emb_sums[word] / word_counts[word], word_counts[word])

    return result


def build_training_data(
    word_embeddings: Dict[str, Tuple[np.ndarray, int]],
    vocab: List[str],
    ica_scores: np.ndarray,
    min_count: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build training data by matching GPT embeddings to word atlas.

    Returns:
        X: GPT embeddings (n_matched, 1024)
        Y: ICA scores (n_matched, n_axes)
        matched_words: list of matched words
    """
    X_list = []
    Y_list = []
    matched_words = []

    vocab_clean = {clean_word(w): i for i, w in enumerate(vocab)}

    for word, (emb, count) in word_embeddings.items():
        if count < min_count:
            continue

        word_clean = clean_word(word)
        if word_clean in vocab_clean:
            idx = vocab_clean[word_clean]
            X_list.append(emb)
            Y_list.append(ica_scores[idx])
            matched_words.append(word)

    return np.array(X_list), np.array(Y_list), matched_words


def train_adapter(
    X: np.ndarray,
    Y: np.ndarray,
    axes: List[int],
    n_folds: int = 5,
    alphas: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Train ridge regression adapter with cross-validation.

    Args:
        X: GPT embeddings (n_samples, 1024)
        Y: ICA scores (n_samples, n_axes)
        axes: Which axes to train on
        n_folds: Number of CV folds
        alphas: Ridge alpha values to try

    Returns:
        W: Weight matrix (n_target_axes, 1024)
        b: Bias vector (n_target_axes,)
        metrics: Per-axis metrics
    """
    if alphas is None:
        alphas = np.logspace(-2, 6, 50)

    Y_target = Y[:, axes]
    n_samples, n_axes = Y_target.shape

    # Train adapter
    print(f"[train] Training adapter on {n_samples} words, {n_axes} axes")

    W = np.zeros((n_axes, X.shape[1]))
    b = np.zeros(n_axes)

    metrics = {"axes": axes, "per_axis": {}}

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for ax_idx, ax in enumerate(axes):
        y = Y_target[:, ax_idx]

        # Cross-validated ridge
        model = RidgeCV(alphas=alphas, cv=n_folds)
        model.fit(X, y)

        W[ax_idx] = model.coef_
        b[ax_idx] = model.intercept_

        # CV predictions
        y_pred_cv = np.zeros(n_samples)
        for train_idx, test_idx in kf.split(X):
            model_fold = RidgeCV(alphas=alphas, cv=3)
            model_fold.fit(X[train_idx], y[train_idx])
            y_pred_cv[test_idx] = model_fold.predict(X[test_idx])

        # Metrics
        r2 = r2_score(y, y_pred_cv)
        r, p = pearsonr(y, y_pred_cv)

        metrics["per_axis"][int(ax)] = {
            "r2_cv": float(r2),
            "r_cv": float(r),
            "p_cv": float(p),
            "best_alpha": float(model.alpha_),
        }

        print(f"  Axis {ax}: R²={r2:.4f}, r={r:.4f}, p={p:.2e}, α={model.alpha_:.1e}")

    return W, b, metrics


def evaluate_generalization(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    axes: List[int],
    alphas: np.ndarray = None,
) -> Dict:
    """
    Evaluate adapter generalization on held-out stories.
    """
    if alphas is None:
        alphas = np.logspace(-2, 6, 50)

    Y_train_target = Y_train[:, axes]
    Y_test_target = Y_test[:, axes]

    results = {"axes": axes, "per_axis": {}}

    for ax_idx, ax in enumerate(axes):
        y_train = Y_train_target[:, ax_idx]
        y_test = Y_test_target[:, ax_idx]

        # Train on all training data
        model = RidgeCV(alphas=alphas, cv=5)
        model.fit(X_train, y_train)

        # Predict on test
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Metrics
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        r_train, _ = pearsonr(y_train, y_pred_train)
        r_test, p_test = pearsonr(y_test, y_pred_test)

        results["per_axis"][int(ax)] = {
            "r2_train": float(r2_train),
            "r2_test": float(r2_test),
            "r_train": float(r_train),
            "r_test": float(r_test),
            "p_test": float(p_test),
            "n_train": len(y_train),
            "n_test": len(y_test),
        }

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpt-root", type=pathlib.Path, required=True,
                   help="GPT embeddings directory")
    p.add_argument("--time-align-root", type=pathlib.Path, required=True,
                   help="Time alignment directory")
    p.add_argument("--word-axes-root", type=pathlib.Path, required=True,
                   help="Word axes directory")
    p.add_argument("--out-dir", type=pathlib.Path, required=True,
                   help="Output directory")
    p.add_argument("--n-test-stories", type=int, default=12,
                   help="Number of stories for test set")
    p.add_argument("--min-word-count", type=int, default=2,
                   help="Minimum word occurrences for training")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load word atlas
    print("[load] Loading word atlas...")
    vocab, ica_scores = load_word_axes(args.word_axes_root)
    print(f"  {len(vocab)} words, {ica_scores.shape[1]} axes")

    # Define story splits
    all_stories = list(range(1, 61))  # Stories 1-60
    np.random.seed(42)
    np.random.shuffle(all_stories)

    test_stories = all_stories[:args.n_test_stories]
    train_stories = all_stories[args.n_test_stories:]

    print(f"\n[split] Train: {len(train_stories)} stories, Test: {len(test_stories)} stories")
    print(f"  Train stories: {sorted(train_stories)[:10]}...")
    print(f"  Test stories: {sorted(test_stories)}")

    # Build word embeddings from training stories
    print("\n[build] Building word-type embeddings from training stories...")
    train_word_emb = build_word_type_embeddings(
        args.gpt_root, args.time_align_root, train_stories
    )
    print(f"  {len(train_word_emb)} unique word types from training")

    # Build word embeddings from test stories
    print("[build] Building word-type embeddings from test stories...")
    test_word_emb = build_word_type_embeddings(
        args.gpt_root, args.time_align_root, test_stories
    )
    print(f"  {len(test_word_emb)} unique word types from test")

    # Build training data
    print("\n[match] Matching to word atlas...")
    X_train, Y_train, train_words = build_training_data(
        train_word_emb, vocab, ica_scores, min_count=args.min_word_count
    )
    print(f"  Training: {len(train_words)} matched words")

    X_test, Y_test, test_words = build_training_data(
        test_word_emb, vocab, ica_scores, min_count=1  # Lower threshold for test
    )
    print(f"  Test: {len(test_words)} matched words")

    # Find words unique to test (true generalization)
    train_word_set = set(train_words)
    test_unique_mask = np.array([w not in train_word_set for w in test_words])
    n_unique = test_unique_mask.sum()
    print(f"  Test words NOT in training: {n_unique} ({n_unique/len(test_words)*100:.1f}%)")

    # Train adapter on genuine axes
    print(f"\n[train] Training adapter on GENUINE axes: {ALL_GENUINE}")
    W, b, train_metrics = train_adapter(X_train, Y_train, ALL_GENUINE)

    # Evaluate generalization
    print("\n[eval] Evaluating generalization on test stories...")
    gen_results = evaluate_generalization(
        X_train, Y_train, X_test, Y_test, ALL_GENUINE
    )

    print("\n" + "=" * 70)
    print("GENERALIZATION RESULTS (Test Stories)")
    print("=" * 70)
    print(f"{'Axis':>6} {'R²_train':>10} {'R²_test':>10} {'r_test':>10} {'p_test':>12}")

    for ax in ALL_GENUINE:
        m = gen_results["per_axis"][ax]
        print(f"{ax:6d} {m['r2_train']:10.4f} {m['r2_test']:10.4f} {m['r_test']:+10.4f} {m['p_test']:12.2e}")

    # Evaluate on truly novel words (not seen in training)
    if n_unique > 50:
        print(f"\n[eval] Evaluating on {n_unique} words NEVER seen in training...")
        X_novel = X_test[test_unique_mask]
        Y_novel = Y_test[test_unique_mask]
        novel_results = evaluate_generalization(
            X_train, Y_train, X_novel, Y_novel, ALL_GENUINE
        )

        print("\n" + "=" * 70)
        print("NOVEL WORD GENERALIZATION (words not in training stories)")
        print("=" * 70)
        print(f"{'Axis':>6} {'R²_novel':>10} {'r_novel':>10} {'p_novel':>12}")

        for ax in ALL_GENUINE:
            m = novel_results["per_axis"][ax]
            print(f"{ax:6d} {m['r2_test']:10.4f} {m['r_test']:+10.4f} {m['p_test']:12.2e}")
    else:
        novel_results = None

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    mean_r2_test = np.mean([gen_results["per_axis"][ax]["r2_test"] for ax in ALL_GENUINE])
    mean_r_test = np.mean([gen_results["per_axis"][ax]["r_test"] for ax in ALL_GENUINE])
    n_sig = sum(1 for ax in ALL_GENUINE if gen_results["per_axis"][ax]["p_test"] < 0.05)

    print(f"Mean R² (test): {mean_r2_test:.4f}")
    print(f"Mean r (test): {mean_r_test:.4f}")
    print(f"Axes with p < 0.05: {n_sig}/{len(ALL_GENUINE)}")

    if mean_r2_test > 0.01:
        print("✓ LLM adapter shows meaningful predictive power")
    elif mean_r2_test > 0:
        print("◐ LLM adapter shows weak but positive predictive power")
    else:
        print("✗ LLM adapter does not predict brain axes")

    # Save outputs
    print(f"\n[save] Saving to {args.out_dir}/")

    np.save(args.out_dir / "adapter_W.npy", W)
    np.save(args.out_dir / "adapter_b.npy", b)

    results = {
        "genuine_axes": ALL_GENUINE,
        "axis_semantics": GENUINE_AXES,
        "train_stories": sorted(train_stories),
        "test_stories": sorted(test_stories),
        "n_train_words": len(train_words),
        "n_test_words": len(test_words),
        "train_metrics": train_metrics,
        "generalization": gen_results,
        "novel_word_generalization": novel_results,
        "summary": {
            "mean_r2_test": float(mean_r2_test),
            "mean_r_test": float(mean_r_test),
            "n_axes_significant": n_sig,
        }
    }

    with open(args.out_dir / "adapter_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save word lists for inspection
    with open(args.out_dir / "train_words.txt", "w") as f:
        f.write("\n".join(sorted(train_words)))
    with open(args.out_dir / "test_words.txt", "w") as f:
        f.write("\n".join(sorted(test_words)))

    print("Done!")


if __name__ == "__main__":
    main()
