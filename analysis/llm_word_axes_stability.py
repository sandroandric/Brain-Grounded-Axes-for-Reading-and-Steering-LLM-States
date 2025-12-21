"""
Stability check for LLM -> MEG word-axis adapter across random splits.

Loads word-level hidden states and MEG word-axis scores, then runs multiple
train/test splits and reports mean ± std test performance per axis.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


def clean_word(w: str) -> str:
    return re.sub(r"\s+", "", str(w).strip())


def load_vocab_scores(root: pathlib.Path, axes: List[int] | None = None) -> Tuple[List[str], np.ndarray]:
    vocab = np.load(root / "vocab.npy", allow_pickle=True).tolist()
    scores_path = root / "ica_scores.npy"
    if not scores_path.exists():
        scores_path = root / "pca_scores.npy"
    scores = np.load(scores_path)
    if axes:
        scores = scores[:, axes]
    return vocab, scores


def load_story_words(time_align_root: pathlib.Path, story_id: int) -> List[str]:
    path = time_align_root / f"story_{story_id}_word_time.mat"
    mat = loadmat(path)
    words = [str(w).strip() for w in mat["word"].flatten()]
    return words


def build_word_hidden(
    hidden_root: pathlib.Path,
    time_align_root: pathlib.Path,
    stories: List[int],
    min_count: int = 1,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    sums: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = defaultdict(int)
    for story_id in stories:
        hidden_path = hidden_root / f"story_{story_id}_hidden.npy"
        time_path = time_align_root / f"story_{story_id}_word_time.mat"
        if not hidden_path.exists() or not time_path.exists():
            continue
        hidden = np.load(hidden_path)
        words = load_story_words(time_align_root, story_id)
        n = min(len(words), hidden.shape[0])
        for w, vec in zip(words[:n], hidden[:n]):
            w_clean = clean_word(w)
            if not w_clean:
                continue
            if w_clean not in sums:
                sums[w_clean] = vec.astype(np.float32).copy()
            else:
                sums[w_clean] += vec.astype(np.float32)
            counts[w_clean] += 1

    avg = {}
    for w, total in sums.items():
        if counts[w] >= min_count:
            avg[w] = total / counts[w]
    return avg, counts


def match_vocab(
    word_embeds: Dict[str, np.ndarray],
    vocab: List[str],
    scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    vocab_clean = {clean_word(w): i for i, w in enumerate(vocab)}
    X_list = []
    Y_list = []
    words = []
    for w, emb in word_embeds.items():
        if w in vocab_clean:
            idx = vocab_clean[w]
            X_list.append(emb)
            Y_list.append(scores[idx])
            words.append(w)
    return np.array(X_list), np.array(Y_list), words


def train_test_split(n: int, train_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(np.floor(train_frac * n))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    return train_idx, test_idx


def corr_safe(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0, 1.0
    r, p = pearsonr(a, b)
    return float(r), float(p)


def main():
    p = argparse.ArgumentParser(description="Stability check for LLM->word axes adapter.")
    p.add_argument("--word-axes-root", type=pathlib.Path, required=True)
    p.add_argument("--hidden-root", type=pathlib.Path, required=True)
    p.add_argument("--time-align-root", type=pathlib.Path, required=True)
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    p.add_argument("--axes", nargs="+", type=int, default=None)
    p.add_argument("--stories", nargs="+", type=int, default=None)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--min-count", type=int, default=1)
    p.add_argument("--standardize", action="store_true")
    p.add_argument("--alphas", nargs="+", type=float, default=[0.1, 1, 10, 100, 1000])
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    stories = args.stories if args.stories else list(range(1, 61))
    if args.seeds is None:
        args.seeds = [args.seed + i for i in range(args.n_splits)]

    vocab, scores = load_vocab_scores(args.word_axes_root, args.axes)
    word_embeds, counts = build_word_hidden(args.hidden_root, args.time_align_root, stories, min_count=args.min_count)
    X, Y, matched_words = match_vocab(word_embeds, vocab, scores)
    if len(matched_words) < 50:
        raise SystemExit("Too few matched words for stability test.")

    per_split = []
    for seed in args.seeds:
        train_idx, test_idx = train_test_split(len(matched_words), args.train_frac, seed)
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

        scaler = None
        if args.standardize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model = RidgeCV(alphas=args.alphas, fit_intercept=True)
        model.fit(X_train, Y_train)
        Y_test_pred = model.predict(X_test)

        per_axis = []
        for k in range(Y.shape[1]):
            r, p = corr_safe(Y_test[:, k], Y_test_pred[:, k])
            per_axis.append({"r": r, "p": p, "r2": float(r ** 2)})

        per_split.append({
            "seed": seed,
            "best_alpha": float(model.alpha_),
            "test": per_axis,
        })

    # Summaries
    summary = {}
    for k in range(Y.shape[1]):
        rs = np.array([s["test"][k]["r"] for s in per_split], dtype=float)
        r2s = np.array([s["test"][k]["r2"] for s in per_split], dtype=float)
        summary[str(k)] = {
            "mean_r": float(rs.mean()),
            "std_r": float(rs.std()),
            "mean_r2": float(r2s.mean()),
            "std_r2": float(r2s.std()),
            "min_r": float(rs.min()),
            "max_r": float(rs.max()),
        }

    payload = {
        "axes": args.axes,
        "n_axes": int(Y.shape[1]),
        "n_words_matched": len(matched_words),
        "train_frac": args.train_frac,
        "min_count": args.min_count,
        "standardize": bool(args.standardize),
        "alphas": args.alphas,
        "seeds": args.seeds,
        "per_split": per_split,
        "summary": summary,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
