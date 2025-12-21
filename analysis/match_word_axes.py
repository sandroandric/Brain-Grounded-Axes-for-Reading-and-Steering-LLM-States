"""
Align word-axis spaces by maximizing correlation across shared vocabulary.

Outputs a JSON mapping from base axes to target axes with correlation and sign.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_vocab_scores(root: pathlib.Path) -> Tuple[List[str], np.ndarray]:
    vocab = np.load(root / "vocab.npy", allow_pickle=True).tolist()
    scores_path = root / "ica_scores.npy"
    if not scores_path.exists():
        scores_path = root / "pca_scores.npy"
    scores = np.load(scores_path)
    return vocab, scores


def zscore(x: np.ndarray) -> np.ndarray:
    mu = np.mean(x, axis=0, keepdims=True)
    sd = np.std(x, axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (x - mu) / sd


def main() -> None:
    p = argparse.ArgumentParser(description="Match word axes between two roots via correlation.")
    p.add_argument("--base-root", type=pathlib.Path, required=True)
    p.add_argument("--target-root", type=pathlib.Path, required=True)
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    args = p.parse_args()

    base_vocab, base_scores = load_vocab_scores(args.base_root)
    tgt_vocab, tgt_scores = load_vocab_scores(args.target_root)

    base_idx = {w: i for i, w in enumerate(base_vocab)}
    tgt_idx = {w: i for i, w in enumerate(tgt_vocab)}
    common = sorted(set(base_idx) & set(tgt_idx))
    if len(common) < 50:
        raise ValueError(f"Too few shared words to match axes ({len(common)}).")

    b = np.stack([base_scores[base_idx[w]] for w in common], axis=0)
    t = np.stack([tgt_scores[tgt_idx[w]] for w in common], axis=0)
    b = zscore(b)
    t = zscore(t)

    n_base = b.shape[1]
    n_tgt = t.shape[1]
    corr = np.zeros((n_base, n_tgt), dtype=float)
    for i in range(n_base):
        for j in range(n_tgt):
            corr[i, j] = float(np.corrcoef(b[:, i], t[:, j])[0, 1])

    cost = 1.0 - np.abs(corr)
    row_ind, col_ind = linear_sum_assignment(cost)

    mapping = []
    for i, j in zip(row_ind, col_ind):
        mapping.append(
            {
                "base_axis": int(i),
                "target_axis": int(j),
                "corr": float(corr[i, j]),
                "sign": 1 if corr[i, j] >= 0 else -1,
            }
        )

    out = {
        "base_root": str(args.base_root),
        "target_root": str(args.target_root),
        "n_common": len(common),
        "mapping": mapping,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
