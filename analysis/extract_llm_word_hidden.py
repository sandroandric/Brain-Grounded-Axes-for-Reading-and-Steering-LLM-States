"""
Extract word-level hidden states from a transformer model for each story.

Outputs:
- story_{id}_hidden.npy with shape (n_words, hidden_dim)
  where n_words matches time_align/word-level tokens.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import List

import numpy as np
import torch
from scipy.io import loadmat
from transformers import AutoModel, AutoTokenizer


def load_story_words(time_align_path: pathlib.Path) -> List[str]:
    mat = loadmat(time_align_path)
    words = [str(w).strip() for w in mat["word"].flatten()]
    return words


def ensure_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def chunk_indices(n_words: int, max_words: int, overlap: int):
    step = max(1, max_words - overlap)
    start = 0
    while start < n_words:
        end = min(start + max_words, n_words)
        yield start, end
        if end == n_words:
            break
        start += step


def extract_hidden_for_story(
    model,
    tokenizer,
    words: List[str],
    layer: int,
    device: str,
    max_words: int,
    overlap: int,
):
    n_words = len(words)
    hidden_dim = model.config.hidden_size
    sums = np.zeros((n_words, hidden_dim), dtype=np.float32)
    counts = np.zeros(n_words, dtype=np.int32)

    for start, end in chunk_indices(n_words, max_words, overlap):
        chunk = words[start:end]
        enc = tokenizer(
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
            outputs = model(**model_inputs, output_hidden_states=True)
        hs = outputs.hidden_states[layer][0].detach().cpu().numpy()

        word_sums = np.zeros((len(chunk), hidden_dim), dtype=np.float32)
        word_counts = np.zeros(len(chunk), dtype=np.int32)
        for tok_idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id < 0 or word_id >= len(chunk):
                continue
            word_sums[word_id] += hs[tok_idx]
            word_counts[word_id] += 1

        for i in range(len(chunk)):
            if word_counts[i] == 0:
                continue
            idx = start + i
            sums[idx] += word_sums[i] / word_counts[i]
            counts[idx] += 1

    counts = np.maximum(counts, 1)
    return sums / counts[:, None]


def main():
    p = argparse.ArgumentParser(description="Extract word-level hidden states for LLM adapter.")
    p.add_argument("--time-align-root", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--model", default="uer/gpt2-chinese-cluecorpussmall")
    p.add_argument("--layer", type=int, default=-1, help="Layer index (default: last).")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    p.add_argument("--stories", nargs="+", type=int, default=None)
    p.add_argument("--max-words", type=int, default=256, help="Words per chunk.")
    p.add_argument("--overlap", type=int, default=64, help="Overlap words between chunks.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] model={args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    ensure_pad_token(tokenizer)
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device != "cpu" else torch.float32,
    )
    if args.device == "mps":
        model = model.to("mps")
    elif args.device == "cuda":
        model = model.to("cuda")
    model.eval()

    stories = args.stories if args.stories else list(range(1, 61))
    for story_id in stories:
        time_path = args.time_align_root / f"story_{story_id}_word_time.mat"
        if not time_path.exists():
            print(f"[warn] missing {time_path}, skipping")
            continue
        words = load_story_words(time_path)
        if not words:
            print(f"[warn] empty story {story_id}, skipping")
            continue
        hidden = extract_hidden_for_story(
            model,
            tokenizer,
            words,
            layer=args.layer,
            device=args.device,
            max_words=args.max_words,
            overlap=args.overlap,
        )
        out_path = args.out_dir / f"story_{story_id}_hidden.npy"
        np.save(out_path, hidden.astype(np.float32))
        print(f"[saved] {out_path} shape={hidden.shape}")


if __name__ == "__main__":
    main()
