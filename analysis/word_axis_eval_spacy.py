"""
Evaluate unsupervised word axes using spaCy zh POS/NER (no predefined target axes).

Steps:
- Load vocab and axis scores (PCA or ICA) from word_axes_root.
- Gather tokens from time_align/word-level stories.
- Tag with spaCy zh (pos_ and ent_type_).
- Build word-level labels:
  - major POS per word; function vs content; noun/verb flags
  - frequency (from frequency mats)
  - animacy proxy: PERSON=1, ORG/GPE/LOC=0 (others ignored)
- Compute per-axis effects (Cohen's d or Pearson r) for each label; report top-k axes.
Outputs:
- JSON with top axes per label.

Run in the spaCy venv (zh_core_web_sm installed):
PYTHONPATH=. python analysis/word_axis_eval_spacy.py \
  --dataset-root . \
  --word-axes-root outputs/word_axes_ica \
  --out-json outputs/word_axes_ica/axis_eval_spacy.json \
  --top-k 5
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict
from typing import Dict, List

import numpy as np
import spacy
from scipy.io import loadmat
from scipy.stats import pearsonr

FUNCTION_POS = {"PART", "ADP", "CCONJ", "SCONJ", "PRON", "DET", "AUX"}
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM"}


def load_vocab_scores(root: pathlib.Path):
    vocab = np.load(root / "vocab.npy", allow_pickle=True).tolist()
    scores_path = root / "pca_scores.npy"
    if not scores_path.exists():
        scores_path = root / "ica_scores.npy"
    scores = np.load(scores_path)
    return vocab, scores


def gather_tokens(dataset_root: pathlib.Path) -> List[str]:
    time_dir = dataset_root / "derivatives" / "annotations" / "time_align" / "word-level"
    tokens_all = []
    for mat_path in sorted(time_dir.glob("story_*_word_time.mat")):
        mat = loadmat(mat_path)
        toks = mat["word"].squeeze().tolist()
        tokens_all.extend([str(t) for t in toks])
    return tokens_all


def tag_corpus(tokens: List[str], model_name: str = "zh_core_web_sm"):
    nlp = spacy.load(model_name)
    text = " ".join(tokens)
    doc = nlp(text)
    stats = defaultdict(lambda: {"pos_counts": defaultdict(int), "ent_counts": defaultdict(int)})
    for tok in doc:
        w = tok.text
        stats[w]["pos_counts"][tok.pos_] += 1
        if tok.ent_type_:
            stats[w]["ent_counts"][tok.ent_type_] += 1
    # finalize
    word_labels = {}
    for w, rec in stats.items():
        pos_counts = rec["pos_counts"]
        ent_counts = rec["ent_counts"]
        major_pos = max(pos_counts.items(), key=lambda kv: kv[1])[0] if pos_counts else None
        major_ent = max(ent_counts.items(), key=lambda kv: kv[1])[0] if ent_counts else None
        word_labels[w] = {"major_pos": major_pos, "major_ent": major_ent}
    return word_labels


def load_freq_stats(dataset_root: pathlib.Path) -> Dict[str, float]:
    freq_dir = dataset_root / "derivatives" / "annotations" / "frequency" / "word-level"
    time_dir = dataset_root / "derivatives" / "annotations" / "time_align" / "word-level"
    freq_stats = defaultdict(lambda: {"sum": 0.0, "count": 0})
    for fpath in freq_dir.glob("story_*_word_logfreq.mat"):
        mat = loadmat(fpath)
        wf = mat["wf"].squeeze()
        story_id = fpath.name.split("_")[1]
        time_path = time_dir / f"story_{story_id}_word_time.mat"
        if not time_path.exists():
            continue
        time_mat = loadmat(time_path)
        toks = time_mat["word"].squeeze().tolist()
        T = min(len(toks), len(wf))
        toks = toks[:T]
        wf = wf[:T]
        for t, v in zip(toks, wf):
            rec = freq_stats[str(t)]
            rec["sum"] += float(v)
            rec["count"] += 1
    # finalize
    out = {}
    for w, rec in freq_stats.items():
        out[w] = rec["sum"] / rec["count"] if rec["count"] > 0 else None
    return out


def build_labels(vocab: List[str], word_labels: Dict[str, Dict], freq_stats: Dict[str, float]) -> Dict[str, np.ndarray]:
    n = len(vocab)
    labels = {}
    func = np.full(n, np.nan)
    noun = np.full(n, np.nan)
    verb = np.full(n, np.nan)
    freq = np.full(n, np.nan)
    animate = np.full(n, np.nan)
    for i, w in enumerate(vocab):
        lbl = word_labels.get(w)
        if lbl:
            pos = lbl.get("major_pos")
            ent = lbl.get("major_ent")
            if pos:
                func[i] = 1.0 if pos in FUNCTION_POS else (0.0 if pos in CONTENT_POS else np.nan)
                noun[i] = 1.0 if pos == "NOUN" else 0.0 if pos is not None else np.nan
                verb[i] = 1.0 if pos == "VERB" else 0.0 if pos is not None else np.nan
            if ent:
                if ent == "PERSON":
                    animate[i] = 1.0
                elif ent in {"ORG", "GPE", "LOC", "PRODUCT"}:
                    animate[i] = 0.0
        if w in freq_stats and freq_stats[w] is not None:
            freq[i] = freq_stats[w]
    labels["function_word"] = func
    labels["noun"] = noun
    labels["verb"] = verb
    labels["logfreq"] = freq
    labels["animate"] = animate
    # high/low freq quantiles
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
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2 + 1e-8)
    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled + 1e-8)


def eval_axes(scores: np.ndarray, labels: Dict[str, np.ndarray], top_k: int = 5):
    out = {}
    for name, lab in labels.items():
        valid = ~np.isnan(lab)
        lab_valid = lab[valid]
        if lab_valid.size == 0:
            out[name] = []
            continue
        score_valid = scores[valid]
        res_axes = []
        binary = set(np.unique(lab_valid)).issubset({0.0, 1.0})
        for k in range(score_valid.shape[1]):
            s = score_valid[:, k]
            if binary:
                hi = s[lab_valid == 1.0]
                lo = s[lab_valid == 0.0]
                d = cohen_d(hi, lo)
                res_axes.append({"axis": k, "cohen_d": float(d), "mean_hi": float(np.mean(hi)) if len(hi) else None, "mean_lo": float(np.mean(lo)) if len(lo) else None})
            else:
                r, _ = pearsonr(s, lab_valid)
                res_axes.append({"axis": k, "pearson_r": float(r)})
        res_axes.sort(key=lambda x: abs(x.get("cohen_d", x.get("pearson_r", 0))), reverse=True)
        out[name] = res_axes[:top_k]
    return out


def main():
    p = argparse.ArgumentParser(description="Evaluate word axes using spaCy zh POS/NER labels.")
    p.add_argument("--dataset-root", type=pathlib.Path, required=True)
    p.add_argument("--word-axes-root", type=pathlib.Path, required=True)
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()

    vocab, scores = load_vocab_scores(args.word_axes_root)
    tokens = gather_tokens(args.dataset_root)
    word_labels = tag_corpus(tokens)
    freq_stats = load_freq_stats(args.dataset_root)
    labels = build_labels(vocab, word_labels, freq_stats)
    metrics = eval_axes(scores, labels, top_k=args.top_k)
    with args.out_json.open("w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
