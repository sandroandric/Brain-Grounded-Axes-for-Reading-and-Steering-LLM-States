"""
Unsupervised axes from word-level atlases (averaged across subjects).

Steps:
- Load per-subject word_atlas.npy and vocab (word_vocab.json) from outputs/encoding/plv_glm/sub-XX
- Align vocab across subjects; average vectors per word
- Run PCA (or ICA if chosen) to get axes
- Save:
  - averaged atlas (word order and matrix)
  - axes components
  - top/bottom words per axis (for quick inspection)
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List

import numpy as np
from sklearn.decomposition import PCA, FastICA


def load_sub_atlas(sub_dir: pathlib.Path):
    atlas = np.load(sub_dir / "word_atlas.npy")
    meta = json.load((sub_dir / "word_vocab.json").open())
    vocab = meta["vocab"]
    return vocab, atlas


def align_and_average(root: pathlib.Path, subjects: List[str]):
    # collect vocab union
    union = set()
    per_sub = {}
    for sub in subjects:
        sub_dir = root / f"sub-{sub}"
        vocab, atlas = load_sub_atlas(sub_dir)
        per_sub[sub] = (vocab, atlas)
        union.update(vocab)
    vocab_union = sorted(list(union))
    word_to_idx = {w: i for i, w in enumerate(vocab_union)}
    # sum and count
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
        counts += (np.array([w in vocab for w in vocab_union], dtype=np.int32))
    counts = np.maximum(counts, 1)
    avg_mat = sum_mat / counts[:, None]
    return vocab_union, avg_mat


def top_words_for_axis(vocab: List[str], comps: np.ndarray, k: int = 20):
    tops = []
    for i in range(comps.shape[0]):
        axis = comps[i]
        order = np.argsort(axis)
        low = [(vocab[j], float(axis[j])) for j in order[:k]]
        high = [(vocab[j], float(axis[j])) for j in order[-k:][::-1]]
        tops.append({"axis": i, "low": low, "high": high})
    return tops


def main():
    p = argparse.ArgumentParser(description="Unsupervised axes from averaged word atlas.")
    p.add_argument("--atlas-root", type=pathlib.Path, required=True, help="outputs/encoding/plv_glm")
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--method", choices=["pca", "ica"], default="pca")
    p.add_argument("--components", type=int, default=20)
    args = p.parse_args()

    vocab, avg_mat = align_and_average(args.atlas_root, args.subjects)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "pca":
        model = PCA(n_components=args.components)
        scores = model.fit_transform(avg_mat)  # shape [n_words, n_comp]
        axes = model.components_  # shape [n_comp, feature_dim]
        explained = model.explained_variance_ratio_.tolist()
    else:
        model = FastICA(n_components=args.components, random_state=0)
        scores = model.fit_transform(avg_mat)  # shape [n_words, n_comp]
        axes = model.mixing_.T  # shape [n_comp, feature_dim]
        explained = None

    np.save(out_dir / "vocab.npy", np.array(vocab))
    np.save(out_dir / "avg_atlas.npy", avg_mat.astype(np.float32))
    np.save(out_dir / f"{args.method}_axes.npy", axes.astype(np.float32))
    np.save(out_dir / f"{args.method}_scores.npy", scores.astype(np.float32))  # per-word scores
    if args.method == "pca":
        np.save(out_dir / "explained_variance.npy", np.array(explained, dtype=np.float32))

    # Use scores (word loadings) to pick top/bottom words
    tops = top_words_for_axis(vocab, scores.T, k=20)
    with (out_dir / "tops.json").open("w") as f:
        json.dump(tops, f, indent=2)
    print(f"[saved] averaged atlas + {args.method} axes to {out_dir}")


if __name__ == "__main__":
    main()
