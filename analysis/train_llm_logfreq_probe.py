"""
Train a text-derived probe to predict log-frequency from word-level hidden states.
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


def load_story_words(time_align_root: pathlib.Path, story_id: int) -> List[str]:
    path = time_align_root / f"story_{story_id}_word_time.mat"
    mat = loadmat(path)
    return [str(w).strip() for w in mat["word"].flatten()]


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


def load_logfreq(confounds_path: pathlib.Path) -> Dict[str, float]:
    if not confounds_path.exists():
        return {}
    data = json.loads(confounds_path.read_text())
    out: Dict[str, float] = {}
    for word, vals in data.items():
        lf = vals.get("logfreq")
        if lf is None:
            continue
        out[clean_word(word)] = float(lf)
    return out


def match_word_embeds(
    word_embeds: Dict[str, np.ndarray],
    logfreq: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X_list = []
    Y_list = []
    words = []
    for w, emb in word_embeds.items():
        if w in logfreq:
            X_list.append(emb)
            Y_list.append(logfreq[w])
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


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    per_axis = {}
    for k in range(y_true.shape[1]):
        r, p = pearsonr(y_true[:, k], y_pred[:, k])
        r2 = float(r ** 2)
        per_axis[str(k)] = {"r": float(r), "p": float(p), "r2": r2}
    return per_axis


def main():
    p = argparse.ArgumentParser(description="Train logfreq probe from word-level hidden states.")
    p.add_argument("--hidden-root", type=pathlib.Path, required=True)
    p.add_argument("--time-align-root", type=pathlib.Path, required=True)
    p.add_argument("--confounds-json", type=pathlib.Path, default=pathlib.Path("metadata/lexica/confounds.json"))
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--stories", nargs="+", type=int, default=None)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--min-count", type=int, default=1)
    p.add_argument("--standardize", action="store_true")
    p.add_argument("--alphas", nargs="+", type=float, default=[0.1, 1, 10, 100, 1000])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stories = args.stories if args.stories else list(range(1, 61))

    logfreq = load_logfreq(args.confounds_json)
    word_embeds, counts = build_word_hidden(args.hidden_root, args.time_align_root, stories, min_count=args.min_count)
    X, Y, matched_words = match_word_embeds(word_embeds, logfreq)
    if len(matched_words) < 50:
        raise SystemExit("Too few matched words for training.")

    train_idx, test_idx = train_test_split(len(matched_words), args.train_frac, args.seed)
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    scaler = None
    if args.standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    Y_train = Y_train[:, None]
    Y_test = Y_test[:, None]

    model = RidgeCV(alphas=args.alphas, fit_intercept=True)
    model.fit(X_train, Y_train)
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)

    metrics = {
        "train": eval_metrics(Y_train, Y_train_pred),
        "test": eval_metrics(Y_test, Y_test_pred),
    }

    np.save(args.out_dir / "adapter_W.npy", model.coef_.astype(np.float32))
    np.save(args.out_dir / "adapter_b.npy", model.intercept_.astype(np.float32))
    if scaler is not None:
        np.save(args.out_dir / "adapter_scaler_mean.npy", scaler.mean_.astype(np.float32))
        np.save(args.out_dir / "adapter_scaler_scale.npy", scaler.scale_.astype(np.float32))

    (args.out_dir / "train_words.txt").write_text("\n".join([matched_words[i] for i in train_idx]))
    (args.out_dir / "test_words.txt").write_text("\n".join([matched_words[i] for i in test_idx]))

    sidecar = {
        "axes": [0],
        "target": "logfreq",
        "n_axes": 1,
        "n_words_matched": len(matched_words),
        "train_frac": args.train_frac,
        "min_count": args.min_count,
        "standardize": bool(args.standardize),
        "alphas": args.alphas,
        "best_alpha": float(model.alpha_),
        "seed": args.seed,
        "metrics": metrics,
    }
    with (args.out_dir / "adapter_sidecar.json").open("w") as f:
        json.dump(sidecar, f, indent=2, ensure_ascii=False)

    print(f"[saved] logfreq probe to {args.out_dir} (matched {len(matched_words)} words)")


if __name__ == "__main__":
    main()
