"""
Approximate concreteness/abstractness using a multilingual sentence-transformer and seed words.

Method:
- Use a Chinese-friendly encoder (default: paraphrase-multilingual-MiniLM-L12-v2) to embed words.
- Compute concreteness score = mean sim to concrete seeds - mean sim to abstract seeds.
- Evaluate unsupervised axes by correlating with concreteness and by effect size between high/low quantiles.

Seeds (Chinese):
- Concrete: 苹果, 桌子, 眼睛, 手, 车, 房子, 水, 食物, 动物, 衣服
- Abstract: 自由, 爱, 思想, 政治, 经济, 感情, 文化, 意义, 概念, 责任

Run example:
PYTHONPATH=. python analysis/word_axis_eval_concrete.py \
  --word-axes-root outputs/word_axes_ica \
  --model paraphrase-multilingual-MiniLM-L12-v2 \
  --device mps \
  --out-json outputs/word_axes_ica/axis_eval_concrete.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import pearsonr
from transformers import AutoModel, AutoTokenizer

CONCRETE_SEEDS = ["苹果", "桌子", "眼睛", "手", "车", "房子", "水", "食物", "动物", "衣服"]
ABSTRACT_SEEDS = ["自由", "爱", "思想", "政治", "经济", "感情", "文化", "意义", "概念", "责任"]


def load_vocab_scores(root: pathlib.Path):
    vocab = np.load(root / "vocab.npy", allow_pickle=True).tolist()
    scores_path = root / "pca_scores.npy"
    if not scores_path.exists():
        scores_path = root / "ica_scores.npy"
    scores = np.load(scores_path)
    return vocab, scores


def build_encoder(model_name: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    if device == "cuda":
        dev = torch.device("cuda")
    elif device == "mps":
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    mdl.to(dev)
    mdl.eval()
    return tok, mdl, dev


def encode_texts(texts: List[str], tok, mdl, dev):
    with torch.no_grad():
        batch = tok(texts, padding=True, truncation=True, return_tensors="pt").to(dev)
        out = mdl(**batch)
        # mean pooling
        emb = out.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        emb = (emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return emb.cpu().numpy()


def compute_concreteness(vocab: List[str], tok, mdl, dev) -> Dict[str, float]:
    # encode seeds
    conc_emb = encode_texts(CONCRETE_SEEDS, tok, mdl, dev)
    abs_emb = encode_texts(ABSTRACT_SEEDS, tok, mdl, dev)
    conc_mean = conc_emb.mean(axis=0, keepdims=True)
    abs_mean = abs_emb.mean(axis=0, keepdims=True)
    scores = {}
    # batch encode vocab to be efficient
    B = 256
    for i in range(0, len(vocab), B):
        batch_words = vocab[i : i + B]
        emb = encode_texts(batch_words, tok, mdl, dev)
        # cosine similarity
        def cos(a, b):
            a_norm = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-9)
            b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
            return (a_norm * b_norm).sum(axis=1)

        sim_conc = cos(emb, conc_mean)
        sim_abs = cos(emb, abs_mean)
        vals = sim_conc - sim_abs
        for w, v in zip(batch_words, vals):
            scores[w] = float(v)
    return scores


def cohen_d(x, y):
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2 + 1e-8)
    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled + 1e-8)


def eval_axes(scores: np.ndarray, concreteness: np.ndarray, top_k: int = 5):
    valid = ~np.isnan(concreteness)
    out = {}
    if not valid.any():
        out["concreteness"] = []
        return out
    c = concreteness[valid]
    s_valid = scores[valid]
    # polarity hi/lo
    q20, q80 = np.percentile(c, [20, 80])
    pol = np.where(c >= q80, 1.0, np.where(c <= q20, 0.0, np.nan))
    pol_valid = ~np.isnan(pol)
    res_corr = []
    res_pol = []
    for k in range(s_valid.shape[1]):
        r, _ = pearsonr(s_valid[:, k], c)
        res_corr.append({"axis": k, "pearson_r": float(r)})
        if pol_valid.any():
            hi = s_valid[pol_valid][:, k][pol[pol_valid] == 1.0]
            lo = s_valid[pol_valid][:, k][pol[pol_valid] == 0.0]
            d = cohen_d(hi, lo)
            res_pol.append({"axis": k, "cohen_d": float(d)})
    res_corr.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)
    res_pol.sort(key=lambda x: abs(x["cohen_d"]), reverse=True)
    out["concreteness_corr"] = res_corr[:top_k]
    out["concreteness_pol"] = res_pol[:top_k]
    return out


def main():
    p = argparse.ArgumentParser(description="Evaluate word axes against concreteness (seed-based embedding).")
    p.add_argument("--word-axes-root", type=pathlib.Path, required=True)
    p.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()

    vocab, scores = load_vocab_scores(args.word_axes_root)
    tok, mdl, dev = build_encoder(args.model, args.device)
    conc_scores = compute_concreteness(vocab, tok, mdl, dev)
    conc_arr = np.full(len(vocab), np.nan)
    for i, w in enumerate(vocab):
        if w in conc_scores:
            conc_arr[i] = conc_scores[w]
    metrics = eval_axes(scores, conc_arr, top_k=args.top_k)
    with args.out_json.open("w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
