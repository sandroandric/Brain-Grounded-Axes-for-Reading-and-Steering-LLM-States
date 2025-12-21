"""
Evaluate steering outputs with external text-level metrics (POS/NER).

Computes function/content ratio, noun/verb ratio, animate rate, temporal rate,
logfreq mean, and tests whether these metrics shift with steering strength.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForTokenClassification, AutoTokenizer

try:
    import jieba  # type: ignore
except Exception:
    jieba = None


UD_FUNCTION = {"PART", "ADP", "CCONJ", "SCONJ", "PRON", "DET", "AUX"}
UD_CONTENT = {"NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM"}

ANIMATE_ENTS = {"PER", "PERSON", "PSN"}
INANIMATE_ENTS = {"ORG", "GPE", "LOC", "PRODUCT", "FAC", "NORP", "EVENT", "WORK_OF_ART", "LAW"}
TEMPORAL_ENTS = {"DATE", "TIME"}

AXIS_METRICS = {
    13: "function_ratio",
    15: "logfreq_mean",
    19: "noun_ratio",
    2: "animate_rate",
}


def is_punct(tok: str) -> bool:
    return re.fullmatch(r"[\W_]+", tok) is not None


def normalize_token(t: str) -> str:
    return re.sub(r"\s+", "", str(t)).strip()


def strip_prompt(text: str, prompt: str | None) -> Tuple[str, bool]:
    if not prompt:
        return text, False
    idx = text.find(prompt)
    if idx == -1:
        return text, False
    return text[idx + len(prompt) :], True


def segment(text: str, segmenter: str) -> List[str]:
    if segmenter == "jieba" and jieba is not None:
        toks = jieba.lcut(text)
    elif segmenter == "whitespace":
        toks = text.split()
    else:
        toks = list(text)
    out = []
    for t in toks:
        t = t.strip()
        if not t:
            continue
        if is_punct(t):
            continue
        out.append(t)
    return out


def build_model(model_name: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForTokenClassification.from_pretrained(model_name)
    if device == "mps":
        mdl = mdl.to(torch.device("mps"))
    elif device == "cuda":
        mdl = mdl.to(torch.device("cuda"))
    mdl.eval()
    return tok, mdl


def chunk_words(words: List[str], max_words: int) -> List[List[str]]:
    return [words[i : i + max_words] for i in range(0, len(words), max_words)]


def tag_tokens(
    tokens: List[str],
    tok,
    mdl,
    device: str,
    max_words: int = 128,
) -> List[str | None]:
    labels = [None] * len(tokens)
    id2label = mdl.config.id2label
    for offset, chunk in enumerate(chunk_words(tokens, max_words=max_words)):
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
        counts = [defaultdict(int) for _ in range(len(chunk))]
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            label = id2label.get(int(preds[idx]))
            if not label:
                continue
            if 0 <= word_id < len(chunk):
                counts[word_id][label] += 1
        for i, c in enumerate(counts):
            if not c:
                continue
            label = max(c.items(), key=lambda kv: kv[1])[0]
            labels[offset + i] = label
    return labels


def map_pos(tag: str | None):
    if not tag:
        return None
    if tag in UD_CONTENT or tag[0] in {"N", "V", "A", "D"}:
        return "content"
    if tag in UD_FUNCTION or tag[0] in {"C", "P", "T", "M", "I", "S", "F"}:
        return "function"
    return None


def normalize_ent(tag: str | None) -> str | None:
    if not tag:
        return None
    return tag.replace("B-", "").replace("I-", "")


def compute_metrics(
    tokens: List[str],
    pos_tags: List[str | None],
    ner_tags: List[str | None],
    logfreq_map: Dict[str, float],
):
    token_count = len(tokens)
    func = content = noun = verb = 0
    animate = inanimate = temporal = 0
    logfreq_vals: List[float] = []
    for pos, ent in zip(pos_tags, ner_tags):
        pos_class = map_pos(pos)
        if pos_class == "function":
            func += 1
        elif pos_class == "content":
            content += 1
        if pos:
            if pos.startswith("N") or pos in {"NOUN", "PROPN"}:
                noun += 1
            if pos.startswith("V") or pos in {"VERB", "AUX"}:
                verb += 1
        ent_clean = normalize_ent(ent)
        if ent_clean:
            if ent_clean in ANIMATE_ENTS:
                animate += 1
            elif ent_clean in INANIMATE_ENTS:
                inanimate += 1
            if ent_clean in TEMPORAL_ENTS:
                temporal += 1

    for tok in tokens:
        key = normalize_token(tok)
        if key in logfreq_map:
            logfreq_vals.append(float(logfreq_map[key]))

    func_denom = func + content
    noun_denom = noun + verb
    animate_denom = animate + inanimate

    return {
        "token_count": token_count,
        "function_ratio": func / func_denom if func_denom > 0 else np.nan,
        "noun_ratio": noun / noun_denom if noun_denom > 0 else np.nan,
        "verb_ratio": verb / noun_denom if noun_denom > 0 else np.nan,
        "animate_rate": animate / token_count if token_count > 0 else np.nan,
        "animate_share": animate / animate_denom if animate_denom > 0 else np.nan,
        "temporal_rate": temporal / token_count if token_count > 0 else np.nan,
        "logfreq_mean": float(np.mean(logfreq_vals)) if logfreq_vals else np.nan,
        "logfreq_coverage": (len(logfreq_vals) / token_count) if token_count > 0 else np.nan,
    }


def cohen_d(x, y):
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2 + 1e-8)
    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled + 1e-8)


def perm_pvalue(a: np.ndarray, b: np.ndarray, n_perm: int, seed: int) -> float:
    if n_perm <= 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    combined = np.concatenate([a, b])
    n_a = a.size
    obs = float(np.mean(a) - np.mean(b))
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        diff = float(np.mean(combined[:n_a]) - np.mean(combined[n_a:]))
        if abs(diff) >= abs(obs):
            count += 1
    return (count + 1) / (n_perm + 1)


def load_logfreq(confounds_path: pathlib.Path) -> Dict[str, float]:
    if not confounds_path.exists():
        return {}
    data = json.loads(confounds_path.read_text())
    out = {}
    for word, vals in data.items():
        lf = vals.get("logfreq")
        if lf is None:
            continue
        key = normalize_token(word)
        if key:
            out[key] = float(lf)
    return out


def eval_metric(values_by_strength: Dict[float, List[float]], n_perm: int, seed: int):
    strengths = sorted(values_by_strength.keys())
    all_strengths = []
    all_vals = []
    for s in strengths:
        vals = values_by_strength[s]
        all_strengths.extend([s] * len(vals))
        all_vals.extend(vals)

    pos = np.array([v for s, vals in values_by_strength.items() if s > 0 for v in vals])
    neg = np.array([v for s, vals in values_by_strength.items() if s < 0 for v in vals])
    t_stat = p_val = d_val = perm_p = float("nan")
    if pos.size >= 2 and neg.size >= 2:
        t_stat, p_val = stats.ttest_ind(pos, neg)
        d_val = cohen_d(pos, neg)
        perm_p = perm_pvalue(pos, neg, n_perm, seed)

    r_val = r_p = float("nan")
    if len(all_vals) >= 3 and len(set(all_strengths)) > 1:
        r_val, r_p = stats.pearsonr(all_strengths, all_vals)

    return {
        "mean_by_strength": {str(k): float(np.mean(v)) for k, v in values_by_strength.items()},
        "pos_neg_ttest": {"t": float(t_stat), "p": float(p_val), "cohen_d": float(d_val), "perm_p": float(perm_p)},
        "strength_corr": {"r": float(r_val), "p": float(r_p)},
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate steering outputs with POS/NER metrics.")
    p.add_argument("--input-glob", type=str, required=True)
    p.add_argument("--pos-model", default="ckiplab/bert-base-chinese-pos")
    p.add_argument("--ner-model", default="ckiplab/bert-base-chinese-ner")
    p.add_argument("--confounds-json", type=pathlib.Path, default=pathlib.Path("metadata/lexica/confounds.json"))
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    p.add_argument("--segmenter", choices=["jieba", "char", "whitespace"], default="jieba")
    p.add_argument("--max-words", type=int, default=128)
    p.add_argument("--all-metrics", action="store_true")
    p.add_argument("--n-perms", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    args = p.parse_args()

    paths = [pathlib.Path(p) for p in glob.glob(args.input_glob)]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise FileNotFoundError("No steering JSON files found.")

    if args.segmenter == "jieba" and jieba is None:
        args.segmenter = "char"

    pos_tok, pos_mdl = build_model(args.pos_model, args.device)
    ner_tok, ner_mdl = build_model(args.ner_model, args.device)
    logfreq_map = load_logfreq(args.confounds_json)

    results = {
        "pos_model": args.pos_model,
        "ner_model": args.ner_model,
        "segmenter": args.segmenter,
        "files": {},
    }

    for path in sorted(paths):
        data = json.loads(path.read_text())
        axis_id = int(data.get("axis_id", data.get("axis", 0)))
        target_metrics = list(AXIS_METRICS.values()) if args.all_metrics else [AXIS_METRICS.get(axis_id, "function_ratio")]

        metrics_by_strength = {m: defaultdict(list) for m in target_metrics}
        n_scored = 0
        prompt_stripped = 0

        for gen in data.get("generations", []):
            strength = float(gen.get("strength", 0.0))
            text = gen.get("text", "")
            prompt = gen.get("prompt")
            text, stripped = strip_prompt(text, prompt)
            if stripped:
                prompt_stripped += 1
            tokens = segment(text, args.segmenter)
            if not tokens:
                continue
            pos_tags = tag_tokens(tokens, pos_tok, pos_mdl, args.device, max_words=args.max_words)
            ner_tags = tag_tokens(tokens, ner_tok, ner_mdl, args.device, max_words=args.max_words)
            feats = compute_metrics(tokens, pos_tags, ner_tags, logfreq_map)
            for metric in target_metrics:
                val = feats.get(metric, np.nan)
                if np.isnan(val):
                    continue
                metrics_by_strength[metric][strength].append(float(val))
            n_scored += 1

        metrics_out = {}
        for metric, by_strength in metrics_by_strength.items():
            if not by_strength:
                continue
            metrics_out[metric] = eval_metric(by_strength, args.n_perms, args.seed)

        key = f"{path.parent.name}/{path.name}"
        results["files"][key] = {
            "axis_id": axis_id,
            "layer": path.parent.name,
            "axis_metric_map": AXIS_METRICS,
            "metrics": metrics_out,
            "meta": {
                "n_scored": n_scored,
                "prompt_stripped": prompt_stripped,
            },
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
