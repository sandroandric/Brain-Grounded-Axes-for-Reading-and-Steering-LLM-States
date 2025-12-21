"""
Evaluate unsupervised word axes against NER-derived animate/inanimate labels using spaCy zh.

Heuristic:
- PERSON -> animate = 1
- ORG/PRODUCT/WORK_OF_ART -> inanimate = 0
- GPE/LOC -> inanimate (0)
- Else unknown/ignored

Inputs:
- vocab.npy, pca_scores.npy from outputs/word_axes
- stimuli/text transcripts: derivatives/annotations/time_align/word-level/story_XX_word_time.mat (tokens)
- spaCy zh_core_web_sm model installed in the active environment

Outputs:
- JSON with top axes correlated with animate label
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List

import numpy as np
import spacy
from scipy.stats import pearsonr
from scipy.io import loadmat


NER_ANIMATE = {
    "PERSON": 1.0,
}
NER_INANIMATE = {
    "ORG": 0.0,
    "PRODUCT": 0.0,
    "WORK_OF_ART": 0.0,
    "GPE": 0.0,
    "LOC": 0.0,
}


def load_vocab_scores(root: pathlib.Path):
    vocab = np.load(root / "vocab.npy", allow_pickle=True).tolist()
    scores = np.load(root / "pca_scores.npy")  # [n_words, n_axes]
    return vocab, scores


def gather_tokens(dataset_root: pathlib.Path) -> List[str]:
    time_dir = dataset_root / "derivatives" / "annotations" / "time_align" / "word-level"
    tokens_all = []
    for mat_path in sorted(time_dir.glob("story_*_word_time.mat")):
        mat = loadmat(mat_path)
        toks = mat["word"].squeeze().tolist()
        tokens_all.extend([str(t) for t in toks])
    return tokens_all


def build_ner_labels(tokens: List[str], nlp) -> Dict[str, float]:
    labels: Dict[str, float] = {}
    # spaCy needs text; join tokens
    text = " ".join(tokens)
    doc = nlp(text)
    for ent in doc.ents:
        label = ent.label_
        if label in NER_ANIMATE:
            labels[ent.text] = 1.0
        elif label in NER_INANIMATE:
            labels[ent.text] = 0.0
    return labels


def main():
    p = argparse.ArgumentParser(description="Evaluate word axes against NER-derived animate labels (spaCy zh).")
    p.add_argument("--dataset-root", type=pathlib.Path, required=True)
    p.add_argument("--word-axes-root", type=pathlib.Path, required=True, help="e.g., outputs/word_axes")
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    args = p.parse_args()

    vocab, scores = load_vocab_scores(args.word_axes_root)
    tokens = gather_tokens(args.dataset_root)

    print("[info] loading spaCy zh_core_web_sm ...")
    nlp = spacy.load("zh_core_web_sm")

    ner_labels = build_ner_labels(tokens, nlp)
    animate = np.full(len(vocab), np.nan)
    for i, w in enumerate(vocab):
        if w in ner_labels:
            animate[i] = ner_labels[w]

    valid = ~np.isnan(animate)
    lab = animate[valid]
    score_valid = scores[valid]

    res = []
    for k in range(score_valid.shape[1]):
        s = score_valid[:, k]
        # binary label → use d' or correlation; here use difference + corr
        hi = s[lab == 1.0]
        lo = s[lab == 0.0]
        d = (np.mean(hi) - np.mean(lo)) / (np.std(s) + 1e-8)
        r, _ = pearsonr(s, lab)
        res.append({"axis": k, "mean_hi": float(np.mean(hi)), "mean_lo": float(np.mean(lo)), "d_prime": float(d), "pearson_r": float(r)})
    res.sort(key=lambda x: abs(x["d_prime"]), reverse=True)
    with args.out_json.open("w") as f:
        json.dump(res, f, indent=2)
    print(f"[saved] {args.out_json} (axes ranked by |d'| for animate label)")


if __name__ == "__main__":
    main()
