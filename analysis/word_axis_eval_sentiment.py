"""
Evaluate unsupervised word axes against sentiment/valence (HF classifier) and a simple temporal heuristic.

Inputs:
- vocab.npy and pca_scores.npy or ica_scores.npy from word_axes_root
- HuggingFace sentiment model (Chinese), e.g., uer/roberta-base-finetuned-dianping-chinese

Outputs:
- JSON with top-k axes for valence (correlation) and polarity (effect size hi vs lo quantiles)
- Temporal heuristic labels (past/present/future) with correlations/effect sizes

Run example:
PYTHONPATH=. python analysis/word_axis_eval_sentiment.py \
  --word-axes-root outputs/word_axes_ica \
  --model uer/roberta-base-finetuned-dianping-chinese \
  --device mps \
  --out-json outputs/word_axes_ica/axis_eval_sentiment.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Dict, List

import numpy as np
from scipy.stats import pearsonr
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch


def load_vocab_scores(root: pathlib.Path):
    vocab = np.load(root / "vocab.npy", allow_pickle=True).tolist()
    scores_path = root / "pca_scores.npy"
    if not scores_path.exists():
        scores_path = root / "ica_scores.npy"
    scores = np.load(scores_path)
    return vocab, scores


def build_sentiment_pipe(model_name: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    if device == "cuda":
        device_id = 0
    elif device == "mps":
        device_id = torch.device("mps")
    else:
        device_id = -1
    return pipeline("sentiment-analysis", model=mdl, tokenizer=tok, device=device_id)


def score_sentiment(vocab: List[str], pipe) -> Dict[str, Dict[str, float]]:
    scores = {}
    for w in vocab:
        try:
            out = pipe(w[:128])[0]  # truncate long tokens
            lbl = out.get("label", "").lower()
            score = out.get("score", 0.0)
            # map to valence; if model outputs classes like very_positive/very_negative, handle accordingly
            if any(k in lbl for k in ["very_negative", "neg"]):
                val = -score
            elif any(k in lbl for k in ["very_positive", "pos"]):
                val = score
            elif "neutral" in lbl:
                val = 0.0
            else:
                val = score if score >= 0.5 else -score
            scores[w] = {"valence": float(val), "raw_label": lbl, "raw_score": float(score)}
        except Exception:
            continue
    return scores


PAST_MARKERS = {"过去", "曾经", "当时", "以前", "去年", "前年", "昨天", "曾"}
FUTURE_MARKERS = {"将来", "未来", "以后", "明天", "明年", "将", "会", "即将", "将要"}
PRESENT_MARKERS = {"现在", "如今", "目前", "正在", "当下", "此刻", "今天"}


def temporal_label(w: str) -> float:
    if any(m in w for m in PAST_MARKERS):
        return -1.0
    if any(m in w for m in FUTURE_MARKERS):
        return 1.0
    if any(m in w for m in PRESENT_MARKERS):
        return 0.0
    if re.search(r"\d{4}年", w):
        return -1.0
    return np.nan


def build_labels(vocab: List[str], sent_scores: Dict[str, Dict[str, float]]) -> Dict[str, np.ndarray]:
    n = len(vocab)
    valence = np.full(n, np.nan)
    temporal = np.full(n, np.nan)
    for i, w in enumerate(vocab):
        if w in sent_scores:
            valence[i] = sent_scores[w]["valence"]
        t = temporal_label(w)
        temporal[i] = t
    labels = {"valence": valence, "temporal": temporal}
    # polarity hi/lo quantiles for valence
    val_valid = valence[~np.isnan(valence)]
    if val_valid.size > 0:
        q20, q80 = np.percentile(val_valid, [20, 80])
        pol = np.where(valence >= q80, 1.0, np.where(valence <= q20, 0.0, np.nan))
        labels["polarity"] = pol
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
    p = argparse.ArgumentParser(description="Evaluate word axes with sentiment (HF) and temporal heuristic.")
    p.add_argument("--word-axes-root", type=pathlib.Path, required=True)
    p.add_argument("--model", default="uer/roberta-base-finetuned-dianping-chinese")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()

    vocab, scores = load_vocab_scores(args.word_axes_root)
    pipe = build_sentiment_pipe(args.model, args.device)
    sent_scores = score_sentiment(vocab, pipe)
    labels = build_labels(vocab, sent_scores)
    metrics = eval_axes(scores, labels, top_k=args.top_k)
    with args.out_json.open("w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
