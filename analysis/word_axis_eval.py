"""
Evaluate unsupervised word axes against known labels (POS/function/content, noun/verb, frequency).

Inputs:
- vocab.npy, avg_atlas.npy, pca_scores.npy from outputs/word_axes (or another dir)
- POS tags from derivatives/annotations/syntactic_annotations/part_of_speech/story_XX_pos.txt
- logfreq from derivatives/annotations/frequency/word-level/story_XX_word_logfreq.mat

Outputs:
- metrics JSON summarizing per-axis effects for each label
- optional CSV of top axes per label
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split


FUNCTION_POS = {"DET", "PRON", "ADP", "CCONJ", "PART", "AUX"}
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}


def load_vocab_scores(root: pathlib.Path):
    vocab = np.load(root / "vocab.npy", allow_pickle=True).tolist()
    # support both PCA and ICA outputs
    if (root / "pca_scores.npy").exists():
        scores = np.load(root / "pca_scores.npy")
    else:
        scores = np.load(root / "ica_scores.npy")
    return vocab, scores


def gather_word_stats(dataset_root: pathlib.Path) -> Dict[str, Dict]:
    stats: Dict[str, Dict] = {}
    pos_dir = dataset_root / "derivatives" / "annotations" / "syntactic_annotations" / "part_of_speech"
    freq_dir = dataset_root / "derivatives" / "annotations" / "frequency" / "word-level"
    for story_file in pos_dir.glob("*_pos.txt"):
        story_id = story_file.name.split("_")[0]  # story_XX
        freq_path = freq_dir / f"{story_id}_word_logfreq.mat"
        if not freq_path.exists():
            continue
        # load POS tokens
        tokens = []
        pos_tags = []
        with story_file.open() as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    tokens.append(parts[0])
                    pos_tags.append(parts[2])
        wf = loadmat(freq_path)["wf"].squeeze()
        T = min(len(tokens), len(wf))
        tokens = tokens[:T]
        pos_tags = pos_tags[:T]
        wf = wf[:T]
        for tok, pos, freq in zip(tokens, pos_tags, wf):
            rec = stats.setdefault(tok, {"count": 0, "pos_counts": {}, "logfreq_sum": 0.0, "logfreq_count": 0})
            rec["count"] += 1
            rec["pos_counts"][pos] = rec["pos_counts"].get(pos, 0) + 1
            rec["logfreq_sum"] += float(freq)
            rec["logfreq_count"] += 1
    # finalize
    for tok, rec in stats.items():
        if rec["logfreq_count"] > 0:
            rec["logfreq_mean"] = rec["logfreq_sum"] / rec["logfreq_count"]
        else:
            rec["logfreq_mean"] = None
        if rec["pos_counts"]:
            rec["major_pos"] = max(rec["pos_counts"].items(), key=lambda kv: kv[1])[0]
        else:
            rec["major_pos"] = None
    return stats


def build_labels(vocab: List[str], stats: Dict[str, Dict]) -> Dict[str, np.ndarray]:
    n = len(vocab)
    labels: Dict[str, np.ndarray] = {}
    # POS-based
    func = np.full(n, np.nan)
    noun = np.full(n, np.nan)
    verb = np.full(n, np.nan)
    freq = np.full(n, np.nan)
    for i, w in enumerate(vocab):
        rec = stats.get(w)
        if not rec:
            continue
        pos = rec.get("major_pos")
        if pos:
            func[i] = 1.0 if pos in FUNCTION_POS else (0.0 if pos in CONTENT_POS else np.nan)
            noun[i] = 1.0 if pos == "NOUN" else 0.0 if pos is not None else np.nan
            verb[i] = 1.0 if pos == "VERB" else 0.0 if pos is not None else np.nan
        if rec.get("logfreq_mean") is not None:
            freq[i] = rec["logfreq_mean"]
    labels["function_word"] = func
    labels["noun"] = noun
    labels["verb"] = verb
    labels["logfreq"] = freq
    # High/low frequency labels (quantiles)
    freq_valid = freq[~np.isnan(freq)]
    if freq_valid.size > 0:
        q20, q80 = np.percentile(freq_valid, [20, 80])
        freq_hi = np.where(freq >= q80, 1.0, np.where(freq <= q20, 0.0, np.nan))
        labels["freq_high"] = freq_hi
    return labels


def cohen_d(x, y):
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled + 1e-8)


def eval_axes(scores: np.ndarray, labels: Dict[str, np.ndarray]):
    """
    scores: [n_words, n_axes]
    labels: dict name -> array [n_words] (binary with nans or continuous with nans)
    """
    out = {}
    for name, lab in labels.items():
        valid = ~np.isnan(lab)
        lab_valid = lab[valid]
        score_valid = scores[valid]
        res_axes = []
        for k in range(score_valid.shape[1]):
            s = score_valid[:, k]
            if set(np.unique(lab_valid)).issubset({0.0, 1.0}):
                hi = s[lab_valid == 1.0]
                lo = s[lab_valid == 0.0]
                d = cohen_d(hi, lo)
                res_axes.append({"axis": k, "cohen_d": float(d), "mean_hi": float(np.mean(hi)) if len(hi) else None, "mean_lo": float(np.mean(lo)) if len(lo) else None})
            else:
                # continuous
                r, _ = pearsonr(s, lab_valid)
                res_axes.append({"axis": k, "pearson_r": float(r)})
        # sort by absolute effect
        if set(np.unique(lab_valid)).issubset({0.0, 1.0}):
            res_axes.sort(key=lambda x: abs(x["cohen_d"]), reverse=True)
        else:
            res_axes.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)
        out[name] = res_axes
    return out


def main():
    p = argparse.ArgumentParser(description="Evaluate unsupervised word axes against POS/frequency labels.")
    p.add_argument("--dataset-root", type=pathlib.Path, required=True)
    p.add_argument("--word-axes-root", type=pathlib.Path, required=True, help="e.g., outputs/word_axes")
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    p.add_argument("--top-k", type=int, default=5, help="Store top-k axes per label.")
    args = p.parse_args()

    vocab, scores = load_vocab_scores(args.word_axes_root)
    stats = gather_word_stats(args.dataset_root)
    labels = build_labels(vocab, stats)

    metrics = eval_axes(scores, labels)
    # trim to top-k per label
    for name, arr in metrics.items():
        metrics[name] = arr[: args.top_k]
    with args.out_json.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
