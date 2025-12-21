"""
Permutation test for word-axis associations using spaCy zh POS/NER labels.

For each binary label (function_word, noun, verb, animate, freq_high),
find the axis with the largest |Cohen's d| and compute a max-stat permutation p-value.
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
    animate = np.full(n, np.nan)
    freq = np.full(n, np.nan)
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
    labels["animate"] = animate
    freq_valid = freq[~np.isnan(freq)]
    if freq_valid.size > 0:
        q20, q80 = np.percentile(freq_valid, [20, 80])
        freq_hi = np.where(freq >= q80, 1.0, np.where(freq <= q20, 0.0, np.nan))
        labels["freq_high"] = freq_hi
    return labels


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
    p = argparse.ArgumentParser(description="Permutation test for spaCy-based labels on word axes.")
    p.add_argument("--dataset-root", type=pathlib.Path, required=True)
    p.add_argument("--word-axes-root", type=pathlib.Path, required=True)
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    p.add_argument("--n-perms", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", type=str, default="zh_core_web_sm")
    args = p.parse_args()

    vocab, scores = load_vocab_scores(args.word_axes_root)
    tokens = gather_tokens(args.dataset_root)
    word_labels = tag_corpus(tokens, model_name=args.model)
    freq_stats = load_freq_stats(args.dataset_root)
    labels = build_labels(vocab, word_labels, freq_stats)

    rng = np.random.default_rng(args.seed)
    results = {}
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
        results[name] = {
            "best_axis": int(axis_idx),
            "best_cohen_d": float(best_d),
            "p_value": float(pval),
            "n_valid": int(lab_valid.size),
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
