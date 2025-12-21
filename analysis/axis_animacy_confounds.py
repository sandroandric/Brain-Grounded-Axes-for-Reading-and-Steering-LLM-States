"""
Assess animacy axis effects while controlling for lexical confounds.

Computes:
1) Raw Cohen's d (animate vs inanimate).
2) Residualized d (axis scores after regressing confounds).
3) OLS coefficient for animacy controlling for confounds.
4) Matched-bin bootstrap d across logfreq/valence/arousal bins.

Animacy labels come from HF NER over story wordlists. Labels are cached to JSON.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from scipy.stats import t, ttest_ind
from transformers import AutoModelForTokenClassification, AutoTokenizer


ANIMATE_POS = {"PER", "PERSON"}
ANIMATE_NEG = {"ORG", "GPE", "LOC", "PRODUCT", "FAC", "NORP", "EVENT", "WORK_OF_ART", "LAW"}


def normalize_token(t: str) -> str:
    return re.sub(r"\s+", "", str(t)).strip()


def load_vocab_scores(root: pathlib.Path) -> Tuple[List[str], np.ndarray]:
    vocab = np.load(root / "vocab.npy", allow_pickle=True).tolist()
    scores_path = root / "ica_scores.npy" if (root / "ica_scores.npy").exists() else root / "pca_scores.npy"
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


def build_ner_model(model_name: str, device: str):
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
    tok, mdl = build_ner_model(model_name, device)
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


def load_ner_labels(
    dataset_root: pathlib.Path,
    model_name: str,
    device: str,
    cache_path: pathlib.Path | None,
) -> Dict[str, str]:
    if cache_path is not None and cache_path.exists():
        return json.loads(cache_path.read_text())
    wordlists = list(iter_wordlists(dataset_root))
    labels = tag_wordlists(wordlists, model_name=model_name, device=device)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2))
    return labels


def build_animacy_labels(vocab: List[str], ner_labels: Dict[str, str]) -> np.ndarray:
    anim = np.full(len(vocab), np.nan, dtype=float)
    for i, w in enumerate(vocab):
        norm = normalize_token(w)
        tag = ner_labels.get(norm)
        if not tag:
            continue
        tag = tag.replace("B-", "").replace("I-", "")
        if tag in ANIMATE_POS:
            anim[i] = 1.0
        elif tag in ANIMATE_NEG:
            anim[i] = 0.0
    return anim


def load_valence_arousal(lexica_root: pathlib.Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    vad_path = lexica_root / "chinese_vad_11310.csv"
    valence = {}
    arousal = {}
    if vad_path.exists():
        df = pd.read_csv(vad_path)
        for _, r in df.iterrows():
            w = normalize_token(r["Word"])
            if not w:
                continue
            valence[w] = float(r["Valence_Mean"])
            arousal[w] = float(r["Arousal_Mean"])
    return valence, arousal


def load_confounds(confounds_path: pathlib.Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    data = json.loads(confounds_path.read_text())
    logfreq = {}
    surprisal = {}
    for word, vals in data.items():
        w = normalize_token(word)
        if vals.get("logfreq") is not None:
            logfreq[w] = float(vals["logfreq"])
        if vals.get("surprisal") is not None:
            surprisal[w] = float(vals["surprisal"])
    return logfreq, surprisal


def match_to_vocab(vocab: List[str], lex: Dict[str, float]) -> np.ndarray:
    arr = np.full(len(vocab), np.nan)
    for i, w in enumerate(vocab):
        key = normalize_token(w)
        if key in lex:
            arr[i] = lex[key]
    return arr


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2 + 1e-8)
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled + 1e-8))


def ols_with_animacy(y: np.ndarray, anim: np.ndarray, confounds: np.ndarray) -> Dict[str, float]:
    X = np.column_stack([np.ones(len(y)), anim, confounds])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, p = X.shape
    dof = max(n - p, 1)
    s2 = (resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t_stat = beta[1] / se[1]
    p_val = 2 * (1 - t.cdf(abs(t_stat), dof))
    return {"beta": float(beta[1]), "t": float(t_stat), "p": float(p_val), "dof": int(dof)}


def residualize(y: np.ndarray, confounds: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(y)), confounds])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def quantile_bins(arr: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.nanquantile(arr, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    return np.digitize(arr, edges[1:-1], right=True)


def matched_bootstrap(
    scores: np.ndarray,
    anim: np.ndarray,
    confounds: np.ndarray,
    n_bins: int,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[float, Tuple[float, float]]:
    if confounds.ndim == 1:
        confounds = confounds[:, None]
    bins = np.column_stack([quantile_bins(confounds[:, i], n_bins) for i in range(confounds.shape[1])])
    bins_key = [tuple(b) for b in bins]
    bin_to_idx = defaultdict(list)
    for i, key in enumerate(bins_key):
        bin_to_idx[key].append(i)

    boot_vals = []
    for _ in range(n_boot):
        idx_sel = []
        for key, idxs in bin_to_idx.items():
            idxs = np.array(idxs)
            a_idx = idxs[anim[idxs] == 1.0]
            i_idx = idxs[anim[idxs] == 0.0]
            if len(a_idx) < 2 or len(i_idx) < 2:
                continue
            m = min(len(a_idx), len(i_idx))
            a_s = rng.choice(a_idx, size=m, replace=False)
            i_s = rng.choice(i_idx, size=m, replace=False)
            idx_sel.extend(a_s.tolist())
            idx_sel.extend(i_s.tolist())
        if len(idx_sel) < 10:
            continue
        idx_sel = np.array(idx_sel)
        d = cohen_d(scores[idx_sel][anim[idx_sel] == 1.0], scores[idx_sel][anim[idx_sel] == 0.0])
        boot_vals.append(d)
    if not boot_vals:
        return float("nan"), (float("nan"), float("nan"))
    boot_vals = np.array(boot_vals, dtype=float)
    return float(np.mean(boot_vals)), (float(np.percentile(boot_vals, 2.5)), float(np.percentile(boot_vals, 97.5)))


def main():
    p = argparse.ArgumentParser(description="Animacy confound control for axis scores.")
    p.add_argument("--dataset-root", type=pathlib.Path, required=True)
    p.add_argument("--word-axes-root", type=pathlib.Path, required=True)
    p.add_argument("--axes", nargs="+", type=int, default=[15])
    p.add_argument("--lexica-root", type=pathlib.Path, default=pathlib.Path("metadata/lexica"))
    p.add_argument("--confounds-json", type=pathlib.Path, default=pathlib.Path("metadata/lexica/confounds.json"))
    p.add_argument("--ner-model", default="ckiplab/bert-base-chinese-ner")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    p.add_argument("--labels-cache", type=pathlib.Path, default=pathlib.Path("metadata/lexica/ner_labels.json"))
    p.add_argument(
        "--confounds",
        nargs="+",
        default=["logfreq", "surprisal", "valence", "arousal", "length"],
        choices=["logfreq", "surprisal", "valence", "arousal", "length"],
        help="Confounds to control for (order used in regression and matching).",
    )
    p.add_argument("--n-bins", type=int, default=4)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    vocab, scores = load_vocab_scores(args.word_axes_root)

    ner_labels = load_ner_labels(args.dataset_root, args.ner_model, args.device, args.labels_cache)
    anim = build_animacy_labels(vocab, ner_labels)

    valence_dict, arousal_dict = load_valence_arousal(args.lexica_root)
    logfreq_dict, surprisal_dict = load_confounds(args.confounds_json)

    logfreq = match_to_vocab(vocab, logfreq_dict)
    surprisal = match_to_vocab(vocab, surprisal_dict)
    valence = match_to_vocab(vocab, valence_dict)
    arousal = match_to_vocab(vocab, arousal_dict)
    length = np.array([len(normalize_token(w)) for w in vocab], dtype=float)
    confound_map = {
        "logfreq": logfreq,
        "surprisal": surprisal,
        "valence": valence,
        "arousal": arousal,
        "length": length,
    }

    out = {
        "axes": [],
        "coverage": {
            "animate": int(np.sum(~np.isnan(anim))),
            "logfreq": int(np.sum(~np.isnan(logfreq))),
            "valence": int(np.sum(~np.isnan(valence))),
            "arousal": int(np.sum(~np.isnan(arousal))),
            "surprisal": int(np.sum(~np.isnan(surprisal))),
        },
        "confounds": args.confounds,
        "ner_model": args.ner_model,
    }

    for axis_id in args.axes:
        axis_scores = scores[:, axis_id]
        mask = ~np.isnan(anim)
        for name in args.confounds:
            mask &= ~np.isnan(confound_map[name])
        y = axis_scores[mask]
        a = anim[mask]
        confounds = np.column_stack([confound_map[name][mask] for name in args.confounds])

        raw_hi = y[a == 1.0]
        raw_lo = y[a == 0.0]
        raw_d = cohen_d(raw_hi, raw_lo)
        raw_t = ttest_ind(raw_hi, raw_lo, equal_var=False)

        resid = residualize(y, confounds)
        res_hi = resid[a == 1.0]
        res_lo = resid[a == 0.0]
        res_d = cohen_d(res_hi, res_lo)
        res_t = ttest_ind(res_hi, res_lo, equal_var=False)

        ols = ols_with_animacy(y, a, confounds)

        match_mean, match_ci = matched_bootstrap(
            y,
            a,
            confounds,
            n_bins=args.n_bins,
            n_boot=args.n_boot,
            rng=rng,
        )

        out["axes"].append(
            {
                "axis": int(axis_id),
                "n_total": int(len(y)),
                "n_animate": int(np.sum(a == 1.0)),
                "n_inanimate": int(np.sum(a == 0.0)),
                "raw": {"cohen_d": float(raw_d), "p": float(raw_t.pvalue)},
                "residualized": {"cohen_d": float(res_d), "p": float(res_t.pvalue)},
                "ols": ols,
                "matched": {
                    "cohen_d_mean": float(match_mean),
                    "cohen_d_ci95": [float(match_ci[0]), float(match_ci[1])],
                    "n_bins": int(args.n_bins),
                    "n_boot": int(args.n_boot),
                },
            }
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
