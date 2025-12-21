"""
Evaluate word axes using HuggingFace Chinese POS + NER taggers (unsupervised axes; labels only for interpretation).

Inputs:
- vocab.npy + pca_scores.npy or ica_scores.npy from word_axes_root
- tokens from time_align/word-level
- HF token-classification models for POS and NER

Outputs:
- JSON with top-k axes per label (function_word, noun, verb, animate, freq_high)

Run (example with ckiplab models, MPS):
  PYTHONPATH=. python analysis/word_axis_eval_hf.py \
    --dataset-root . \
    --word-axes-root outputs/word_axes_ica \
    --pos-model ckiplab/bert-base-chinese-pos \
    --ner-model ckiplab/bert-base-chinese-ner \
    --device mps \
    --out-json outputs/word_axes_ica/axis_eval_hf.json \
    --top-k 5
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict
from typing import Dict, List

import numpy as np
from scipy.io import loadmat
from scipy.stats import pearsonr
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

import re

def normalize_token(t: str) -> str:
    return re.sub(r"\s+", "", t)
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
        tokens_all.extend([normalize_token(str(t)) for t in toks])
    return tokens_all


def build_pipeline(model_name: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForTokenClassification.from_pretrained(model_name)
    if device == "cuda":
        device_id = 0
    elif device == "mps":
        device_id = torch.device("mps")
    else:
        device_id = -1
    return pipeline("token-classification", model=mdl, tokenizer=tok, aggregation_strategy="simple", device=device_id)


def tag_tokens(tokens: List[str], pipe) -> Dict[str, str]:
    text = " ".join(tokens)
    ents = pipe(text)
    token_tags = defaultdict(list)
    for ent in ents:
        tag = ent.get("entity_group") or ent.get("entity")
        if not tag:
            continue
        token_text = ent["word"].strip()
        if token_text:
            token_tags[token_text].append(tag)
    token_dom = {}
    for t, tags in token_tags.items():
        token_dom[normalize_token(t)] = max(set(tags), key=tags.count)
    return token_dom


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
            rec = freq_stats[normalize_token(str(t))]
            rec["sum"] += float(v)
            rec["count"] += 1
    out = {}
    for w, rec in freq_stats.items():
        out[w] = rec["sum"] / rec["count"] if rec["count"] > 0 else None
    return out


def build_labels(vocab: List[str], pos_tags: Dict[str, str], ner_tags: Dict[str, str], freq_stats: Dict[str, float]) -> Dict[str, np.ndarray]:
    n = len(vocab)
    labels = {}
    func = np.full(n, np.nan)
    noun = np.full(n, np.nan)
    verb = np.full(n, np.nan)
    freq = np.full(n, np.nan)
    animate = np.full(n, np.nan)
    for i, w in enumerate(vocab):
        pos = pos_tags.get(w)
        if pos:
            func[i] = 1.0 if pos in FUNCTION_POS else (0.0 if pos in CONTENT_POS else np.nan)
            noun[i] = 1.0 if pos.startswith("N") else 0.0 if pos is not None else np.nan
            verb[i] = 1.0 if pos.startswith("V") else 0.0 if pos is not None else np.nan
        ent = ner_tags.get(w)
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
    # freq high/low
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
    p = argparse.ArgumentParser(description="Evaluate word axes with HF POS/NER tags (zh).")
    p.add_argument("--dataset-root", type=pathlib.Path, required=True)
    p.add_argument("--word-axes-root", type=pathlib.Path, required=True)
    p.add_argument("--pos-model", default="ckiplab/bert-base-chinese-pos")
    p.add_argument("--ner-model", default="ckiplab/bert-base-chinese-ner")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()

    vocab, scores = load_vocab_scores(args.word_axes_root)
    tokens = gather_tokens(args.dataset_root)
    print("[info] loading POS model:", args.pos_model)
    pos_pipe = build_pipeline(args.pos_model, args.device)
    print("[info] loading NER model:", args.ner_model)
    ner_pipe = build_pipeline(args.ner_model, args.device)

    pos_tags = tag_tokens(tokens, pos_pipe)
    ner_tags = tag_tokens(tokens, ner_pipe)
    freq_stats = load_freq_stats(args.dataset_root)
    labels = build_labels(vocab, pos_tags, ner_tags, freq_stats)
    metrics = eval_axes(scores, labels, top_k=args.top_k)

    with args.out_json.open("w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
