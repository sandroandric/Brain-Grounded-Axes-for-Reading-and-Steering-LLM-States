"""
PPL-matched control: filter steering generations so pos/neg groups are PPL-matched.

Outputs a filtered steering JSON you can re-evaluate with llm_word_axes_steer_eval.py.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_subsequence(haystack: List[int], needle: List[int]) -> int | None:
    if not needle:
        return 0
    for i in range(0, len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return None


def compute_ppl(model, tokenizer, text: str, prompt: str | None):
    full_ids = tokenizer(text, add_special_tokens=False).input_ids
    if not full_ids or len(full_ids) < 2:
        return None
    start_idx = 0
    if prompt:
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        idx = find_subsequence(full_ids, prompt_ids)
        if idx is not None:
            start_idx = idx + len(prompt_ids)
    if start_idx >= len(full_ids):
        return None
    input_ids = torch.tensor([full_ids], device=model.device)
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits[0, :-1]
    targets = input_ids[0, 1:]
    start = max(start_idx, 1)
    lp = torch.log_softmax(logits[start - 1 :], dim=-1)
    tgt = targets[start - 1 :]
    nll = -lp.gather(1, tgt.unsqueeze(1)).mean().item()
    return float(np.exp(nll))


def match_by_ppl(pos, neg):
    neg_sorted = sorted(neg, key=lambda x: x["ppl"])
    matched = []
    used = set()
    for p in sorted(pos, key=lambda x: x["ppl"]):
        # find nearest unused neg
        best_idx = None
        best_diff = None
        for i, n in enumerate(neg_sorted):
            if i in used:
                continue
            diff = abs(p["ppl"] - n["ppl"])
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_idx = i
        if best_idx is not None:
            used.add(best_idx)
            matched.append((p, neg_sorted[best_idx]))
    return matched


def main():
    p = argparse.ArgumentParser(description="PPL-matched control for steering outputs.")
    p.add_argument("--input-json", type=pathlib.Path, required=True)
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    p.add_argument("--keep-zero", action="store_true")
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    args = p.parse_args()

    data = json.loads(args.input_json.read_text())
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device != "cpu" else torch.float32,
    )
    if args.device == "mps":
        mdl = mdl.to("mps")
    elif args.device == "cuda":
        mdl = mdl.to("cuda")
    mdl.eval()

    scored = []
    for gen in data.get("generations", []):
        strength = float(gen.get("strength", 0.0))
        text = gen.get("text", "")
        prompt = gen.get("prompt")
        ppl = compute_ppl(mdl, tok, text, prompt)
        if ppl is None:
            continue
        rec = dict(gen)
        rec["ppl"] = ppl
        scored.append(rec)

    pos = [g for g in scored if g["strength"] > 0]
    neg = [g for g in scored if g["strength"] < 0]
    zero = [g for g in scored if g["strength"] == 0]

    matched = match_by_ppl(pos, neg)
    kept = []
    for p_rec, n_rec in matched:
        kept.append(p_rec)
        kept.append(n_rec)
    if args.keep_zero:
        kept.extend(zero)

    out = dict(data)
    out["generations"] = kept
    out["ppl_match"] = {
        "n_pos": len(pos),
        "n_neg": len(neg),
        "n_zero": len(zero),
        "n_matched_pairs": len(matched),
        "n_kept": len(kept),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
