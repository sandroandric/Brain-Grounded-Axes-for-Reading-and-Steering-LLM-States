"""
Cluster the averaged word atlas with K-means and report top words per cluster.

Inputs:
- vocab.npy, avg_atlas.npy from outputs/word_axes or word_axes_ica

Outputs:
- cluster_assignments.npy (per word)
- clusters.json (top words per cluster)
- optional silhouette scores for K sweep
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import re

def is_numeric_token(w: str) -> bool:
    w = w.strip()
    # drop tokens containing any digit or percent
    return any(ch.isdigit() for ch in w) or "%" in w

def top_words_for_cluster(vocab: List[str], mat: np.ndarray, labels: np.ndarray, k: int, top_n: int = 20):
    tops = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            tops.append({"cluster": c, "size": 0, "words": []})
            continue
        centroid = mat[idx].mean(axis=0, keepdims=True)
        dists = ((mat[idx] - centroid) ** 2).sum(axis=1)
        order = np.argsort(dists)[:top_n]
        words = [(vocab[idx[i]], float(dists[order[i]])) for i in range(len(order))]
        tops.append({"cluster": c, "size": int(idx.size), "words": words})
    return tops


def main():
    p = argparse.ArgumentParser(description="K-means clustering of word atlas.")
    p.add_argument("--word-axes-root", type=pathlib.Path, required=True, help="outputs/word_axes or word_axes_ica")
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--silhouette", action="store_true", help="Compute silhouette score for this K.")
    p.add_argument("--sweep", nargs="+", type=int, help="Optional sweep of Ks to compute silhouette scores.")
    args = p.parse_args()

    vocab_all = np.load(args.word_axes_root / "vocab.npy", allow_pickle=True).tolist()
    mat_all = np.load(args.word_axes_root / "avg_atlas.npy")
    keep_idx = [i for i, w in enumerate(vocab_all) if not is_numeric_token(w)]
    vocab = [vocab_all[i] for i in keep_idx]
    mat = mat_all[keep_idx]

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        sil = {}
        for k in args.sweep:
            km = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(mat)
            score = silhouette_score(mat, km.labels_)
            sil[k] = score
        with (out_dir / "silhouette.json").open("w") as f:
            json.dump(sil, f, indent=2)
        print(f"[saved] silhouette scores to {out_dir/'silhouette.json'}")

    km = KMeans(n_clusters=args.k, random_state=0, n_init="auto").fit(mat)
    labels = km.labels_
    np.save(out_dir / "cluster_assignments.npy", labels)
    tops = top_words_for_cluster(vocab, mat, labels, args.k, top_n=20)
    with (out_dir / "clusters.json").open("w") as f:
        json.dump(tops, f, indent=2, ensure_ascii=False)
    print(f"[saved] clusters to {out_dir/'clusters.json'} (K={args.k})")


if __name__ == "__main__":
    main()
