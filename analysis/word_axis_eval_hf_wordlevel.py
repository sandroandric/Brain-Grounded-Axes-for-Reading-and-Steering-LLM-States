"""
Evaluate word axes using HF token-classification models (word-level mapping).

This avoids concatenating all tokens into one string and instead feeds
pre-tokenized word lists (from time_align) with is_split_into_words=True.

Outputs:
- JSON with top-k axes per label + coverage counts.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.io import loadmat
from scipy.stats import pearsonr
from transformers import AutoModelForTokenClassification, AutoTokenizer

import re


UD_FUNCTION = {"PART", "ADP", "CCONJ", "SCONJ", "PRON", "DET", "AUX"}
UD_CONTENT = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM"}


def normalize_token(t: str) -> str:
    return re.sub(r"\s+", "", t).strip()


def load_vocab_scores(root: pathlib.Path):
    vocab = np.load(root / "vocab.npy", allow_pickle=True).tolist()
    scores_path = root / "pca_scores.npy"
    if not scores_path.exists():
        scores_path = root / "ica_scores.npy"
    scores = np.load(scores_path)
    return vocab, scores


def iter_wordlists(dataset_root: pathlib.Path) -> List[List[str]]:
    time_dir = dataset_root / "derivatives" / "annotations" / "time_align" / "word-level"
    for mat_path in sorted(time_dir.glob("story_*_word_time.mat")):
        mat = loadmat(mat_path)
        toks = mat["word"].squeeze().tolist()
        words = [str(t) for t in toks]
        yield words


def chunk_words(words: List[str], max_words: int) -> List[List[str]]:
    return [words[i : i + max_words] for i in range(0, len(words), max_words)]


def build_model(model_name: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForTokenClassification.from_pretrained(model_name)
    if device == "mps":
        mdl = mdl.to(torch.device("mps"))
    elif device == "cuda":
        mdl = mdl.to(torch.device("cuda"))
    mdl.eval()
    return tok, mdl


def tag_wordlists(
    wordlists: List[List[str]],
    model_name: str,
    device: str,
    max_words: int = 128,
    batch_size: int = 1,
) -> Dict[str, str]:
    tok, mdl = build_model(model_name, device)
    id2label = mdl.config.id2label
    counts = defaultdict(lambda: defaultdict(int))

    for words in wordlists:
        for chunk in chunk_words(words, max_words=max_words):
            enc = tok(
                chunk,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                padding=True,
            )
            word_ids = enc.word_ids(batch_index=0)
            if device == "mps":
                model_inputs = {k: v.to("mps") for k, v in enc.items()}
            elif device == "cuda":
                model_inputs = {k: v.to("cuda") for k, v in enc.items()}
            else:
                model_inputs = enc
            with torch.no_grad():
                logits = mdl(**model_inputs).logits
            preds = logits.argmax(-1).cpu().numpy()[0]
            for idx, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                label = id2label.get(int(preds[idx]), None)
                if label is None:
                    continue
                w = normalize_token(chunk[word_id])
                if not w:
                    continue
                counts[w][label] += 1

    out = {}
    for w, c in counts.items():
        out[w] = max(c.items(), key=lambda kv: kv[1])[0]
    return out


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


def map_pos(tag: str):
    if tag in UD_CONTENT or (tag and tag[0] in {"N", "V", "A", "D"}):
        return "content"
    if tag in UD_FUNCTION or (tag and tag[0] in {"C", "P", "T", "M", "I", "S", "F"}):
        return "function"
    return None


def build_labels(vocab: List[str], pos_tags: Dict[str, str], ner_tags: Dict[str, str], freq_stats: Dict[str, float]) -> Dict[str, np.ndarray]:
    n = len(vocab)
    labels = {}
    func = np.full(n, np.nan)
    noun = np.full(n, np.nan)
    verb = np.full(n, np.nan)
    animate = np.full(n, np.nan)
    temporal = np.full(n, np.nan)
    freq = np.full(n, np.nan)
    for i, w in enumerate(vocab):
        norm_w = normalize_token(w)
        pos = pos_tags.get(norm_w)
        if pos:
            pos_class = map_pos(pos)
            if pos_class == "function":
                func[i] = 1.0
            elif pos_class == "content":
                func[i] = 0.0
            noun[i] = 1.0 if pos.startswith("N") or pos == "NOUN" else 0.0
            verb[i] = 1.0 if pos.startswith("V") or pos in {"VERB", "AUX"} else 0.0
        ent = ner_tags.get(norm_w)
        if ent:
            ent_clean = ent.replace("B-", "").replace("I-", "")
            if ent_clean in {"PER", "PERSON"}:
                animate[i] = 1.0
            elif ent_clean in {"ORG", "GPE", "LOC", "PRODUCT", "FAC", "NORP"}:
                animate[i] = 0.0
            if ent_clean in {"DATE", "TIME"}:
                temporal[i] = 1.0
            elif ent_clean in {"PER", "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "FAC", "NORP", "EVENT", "WORK_OF_ART", "LAW"}:
                temporal[i] = 0.0
        if norm_w in freq_stats and freq_stats[norm_w] is not None:
            freq[i] = freq_stats[norm_w]
    labels["function_word"] = func
    labels["noun"] = noun
    labels["verb"] = verb
    labels["animate"] = animate
    labels["temporal"] = temporal
    labels["logfreq"] = freq
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


def cohen_d_axis(scores: np.ndarray, lab: np.ndarray) -> np.ndarray:
    hi = lab == 1.0
    lo = lab == 0.0
    n_hi = np.sum(hi)
    n_lo = np.sum(lo)
    if n_hi < 2 or n_lo < 2:
        return np.full(scores.shape[1], np.nan)
    hi_mean = scores[hi].mean(axis=0)
    lo_mean = scores[lo].mean(axis=0)
    hi_var = scores[hi].var(axis=0, ddof=1)
    lo_var = scores[lo].var(axis=0, ddof=1)
    pooled = ((n_hi - 1) * hi_var + (n_lo - 1) * lo_var) / (n_hi + n_lo - 2 + 1e-8)
    return (hi_mean - lo_mean) / np.sqrt(pooled + 1e-8)


def perm_test(scores: np.ndarray, lab: np.ndarray, n_perms: int, rng: np.random.Generator):
    obs = cohen_d_axis(scores, lab)
    if np.all(np.isnan(obs)):
        return None
    obs_idx = int(np.nanargmax(np.abs(obs)))
    obs_best = float(obs[obs_idx])
    null = np.zeros(n_perms, dtype=np.float32)
    for i in range(n_perms):
        perm = rng.permutation(lab)
        d = cohen_d_axis(scores, perm)
        null[i] = float(np.nanmax(np.abs(d)))
    pval = float((1 + np.sum(null >= abs(obs_best))) / (1 + len(null)))
    return obs_idx, obs_best, pval


def main():
    p = argparse.ArgumentParser(description="Evaluate word axes with HF POS/NER (word-level mapping).")
    p.add_argument("--dataset-root", type=pathlib.Path, required=True)
    p.add_argument("--word-axes-root", type=pathlib.Path, required=True)
    p.add_argument("--pos-model", default="ckiplab/bert-base-chinese-pos")
    p.add_argument("--ner-model", default="ckiplab/bert-base-chinese-ner")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-words", type=int, default=128)
    p.add_argument("--n-perms", type=int, default=0, help="Permutation count for max-stat p-values (0 to skip).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    vocab, scores = load_vocab_scores(args.word_axes_root)
    wordlists = list(iter_wordlists(args.dataset_root))
    print(f"[info] wordlists={len(wordlists)} stories")

    print("[info] loading POS model:", args.pos_model)
    pos_tags = tag_wordlists(wordlists, args.pos_model, args.device, max_words=args.max_words)
    print("[info] loading NER model:", args.ner_model)
    ner_tags = tag_wordlists(wordlists, args.ner_model, args.device, max_words=args.max_words)
    freq_stats = load_freq_stats(args.dataset_root)

    labels = build_labels(vocab, pos_tags, ner_tags, freq_stats)
    metrics = eval_axes(scores, labels, top_k=args.top_k)
    perm = {}
    if args.n_perms > 0:
        rng = np.random.default_rng(args.seed)
        for name, lab in labels.items():
            valid = ~np.isnan(lab)
            lab_valid = lab[valid]
            if lab_valid.size == 0:
                continue
            scores_valid = scores[valid]
            out = perm_test(scores_valid, lab_valid, args.n_perms, rng)
            if out is None:
                continue
            axis_idx, best_d, pval = out
            perm[name] = {
                "best_axis": int(axis_idx),
                "best_cohen_d": float(best_d),
                "p_value": float(pval),
                "n_valid": int(lab_valid.size),
            }
    payload = {
        "metrics": metrics,
        "perm": perm,
        "coverage": {k: int(np.sum(~np.isnan(v))) for k, v in labels.items()},
        "pos_model": args.pos_model,
        "ner_model": args.ner_model,
        "device": args.device,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
