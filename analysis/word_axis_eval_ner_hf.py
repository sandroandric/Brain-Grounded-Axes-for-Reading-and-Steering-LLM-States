"""
Auto-tag tokens with a HuggingFace Chinese NER model and report tag enrichment per axis.

Workflow:
- Load word vocab and axis scores (PCA/ICA) from outputs/word_axes or word_axes_ica.
- Tag tokens from time_align word-level files with a HF NER model.
- Build word -> tag counts, derive dominant tag per word.
- For each axis: compute mean scores per tag and effect sizes between tags.
- Output JSON with top tags per axis (no predefined target axes).

Requires: transformers, torch
Optional: use MPS if available (--device mps).
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from scipy.io import loadmat
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


def load_vocab_scores(root: pathlib.Path):
    vocab = np.load(root / "vocab.npy", allow_pickle=True).tolist()
    # support pca or ica scores
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


def tag_tokens(tokens: List[str], model_name: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    m = AutoModelForTokenClassification.from_pretrained(model_name)
    device_id = 0 if device == "cuda" else -1
    if device == "mps":
        device_id = torch.device("mps")
    ner = pipeline("ner", model=m, tokenizer=tok, aggregation_strategy="simple", device=device_id)
    text = " ".join(tokens)
    ents = ner(text)
    # build token->tag (last tag wins)
    token_tags = defaultdict(list)
    for ent in ents:
        tag = ent.get("entity_group") or ent.get("entity")
        if not tag:
            continue
        # crude split back to token text; entities may span tokens
        token_text = ent["word"].strip()
        if token_text:
            token_tags[token_text].append(tag)
    # choose dominant tag per token
    token_dom = {}
    for t, tags in token_tags.items():
        token_dom[t] = max(set(tags), key=tags.count)
    return token_dom


def enrich_axes(vocab: List[str], scores: np.ndarray, token_tags: Dict[str, str], top_k: int = 5):
    # map vocab -> tag
    tags_arr = np.array([token_tags.get(w, None) for w in vocab], dtype=object)
    unique_tags = sorted({t for t in tags_arr if t is not None})
    res = []
    for ax in range(scores.shape[1]):
        s = scores[:, ax]
        tag_means = []
        for t in unique_tags:
            mask = tags_arr == t
            if mask.sum() == 0:
                continue
            tag_means.append({"tag": t, "mean": float(s[mask].mean()), "count": int(mask.sum())})
        tag_means.sort(key=lambda x: x["mean"], reverse=True)
        res.append({"axis": ax, "tags": tag_means[:top_k]})
    return res


def main():
    p = argparse.ArgumentParser(description="HF NER tag enrichment per word axis (unsupervised).")
    p.add_argument("--dataset-root", type=pathlib.Path, required=True)
    p.add_argument("--word-axes-root", type=pathlib.Path, required=True)
    p.add_argument("--model-name", default="ckiplab/bert-base-chinese-ner")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    args = p.parse_args()

    vocab, scores = load_vocab_scores(args.word_axes_root)
    tokens = gather_tokens(args.dataset_root)
    token_tags = tag_tokens(tokens, args.model_name, args.device)

    res = enrich_axes(vocab, scores, token_tags, top_k=5)
    with args.out_json.open("w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
