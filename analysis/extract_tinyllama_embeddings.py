"""
Extract TinyLlama 1.1B embeddings for SMN4Lang stories.

Matches the format of existing GPT-2 embeddings:
- Per-story .npy files with shape (n_layers, n_words, hidden_dim)
- Word-level alignment using time_align annotations

Run:
python analysis/extract_tinyllama_embeddings.py \
  --time-align-root derivatives/annotations/time_align/word-level \
  --out-dir derivatives/annotations/embeddings/tinyllama/word-level \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

from __future__ import annotations

import argparse
import pathlib
import re
from typing import List, Tuple

import numpy as np
import scipy.io
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def clean_word(w: str) -> str:
    return re.sub(r'\s+', '', str(w).strip())


def load_story_words(time_align_path: pathlib.Path) -> Tuple[List[str], List[float], List[float]]:
    """Load words and timing from time_align file."""
    mat = scipy.io.loadmat(str(time_align_path))
    words = [str(w).strip() for w in mat['word'].flatten()]
    # onset/offset may be named differently
    if 'start' in mat:
        onsets = mat['start'].flatten().tolist()
        offsets = mat['end'].flatten().tolist()
    elif 'onset' in mat:
        onsets = mat['onset'].flatten().tolist()
        offsets = mat['offset'].flatten().tolist()
    else:
        onsets = [0.0] * len(words)
        offsets = [0.0] * len(words)
    return words, onsets, offsets


def get_word_embeddings(
    model,
    tokenizer,
    words: List[str],
    device: str = "mps",
    max_length: int = 2048,
    stride: int = 512,
) -> np.ndarray:
    """
    Get embeddings for each word using TinyLlama with sliding window for long sequences.

    For each word, we get the embedding by:
    1. Tokenizing the text in overlapping windows
    2. Getting hidden states from all layers
    3. Averaging over subword tokens and windows

    Returns: shape (n_layers, n_words, hidden_dim)
    """
    model.eval()

    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    hidden_dim = model.config.hidden_size
    n_words = len(words)

    all_embeddings = np.zeros((n_layers, n_words, hidden_dim), dtype=np.float32)
    word_counts = np.zeros(n_words, dtype=np.int32)

    # Build full text and character-to-word mapping
    full_text = " ".join(words)
    char_to_word = {}
    char_pos = 0
    for word_idx, word in enumerate(words):
        for _ in word:
            char_to_word[char_pos] = word_idx
            char_pos += 1
        char_to_word[char_pos] = word_idx  # space after word
        char_pos += 1

    # Tokenize full text to get total length
    full_tokens = tokenizer(full_text, return_offsets_mapping=True)
    full_offset_mapping = full_tokens["offset_mapping"]
    total_tokens = len(full_tokens["input_ids"])

    # Process in sliding windows if sequence is too long
    if total_tokens <= max_length:
        windows = [(0, total_tokens)]
    else:
        windows = []
        start = 0
        while start < total_tokens:
            end = min(start + max_length, total_tokens)
            windows.append((start, end))
            if end >= total_tokens:
                break
            start += max_length - stride

    for win_start, win_end in windows:
        # Get tokens for this window
        win_input_ids = torch.tensor([full_tokens["input_ids"][win_start:win_end]]).to(device)
        win_offset_mapping = full_offset_mapping[win_start:win_end]

        # Get hidden states
        with torch.no_grad():
            outputs = model(win_input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states

        # Map tokens to words for this window
        for tok_idx, (start, end) in enumerate(win_offset_mapping):
            if start == end:  # special token
                continue

            mid_char = (start + end) // 2
            word_idx = char_to_word.get(mid_char, -1)

            if word_idx >= 0 and word_idx < n_words:
                for layer_idx in range(n_layers):
                    layer_hidden = hidden_states[layer_idx][0, tok_idx].cpu().numpy()
                    all_embeddings[layer_idx, word_idx] += layer_hidden
                word_counts[word_idx] += 1

    # Average over tokens/windows
    word_counts = np.maximum(word_counts, 1)
    for layer_idx in range(n_layers):
        all_embeddings[layer_idx] /= word_counts[:, None]

    return all_embeddings


def process_story(
    story_id: int,
    time_align_root: pathlib.Path,
    model,
    tokenizer,
    device: str,
) -> Tuple[np.ndarray, List[str]]:
    """Process a single story."""
    time_path = time_align_root / f"story_{story_id}_word_time.mat"
    if not time_path.exists():
        return None, None

    words, onsets, offsets = load_story_words(time_path)

    if len(words) == 0:
        return None, None

    # Filter out empty words
    valid_indices = [i for i, w in enumerate(words) if w.strip()]
    words = [words[i] for i in valid_indices]

    if len(words) == 0:
        return None, None

    # Get embeddings
    embeddings = get_word_embeddings(model, tokenizer, words, device)

    return embeddings, words


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--time-align-root", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--device", default="mps", help="Device: mps, cuda, cpu")
    p.add_argument("--stories", nargs="+", type=int, default=None,
                   help="Specific stories to process (default: all 1-60)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"[load] Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device != "cpu" else torch.float32,
    )
    model = model.to(args.device)
    print(f"  Model loaded on {args.device}")
    print(f"  Hidden dim: {model.config.hidden_size}")
    print(f"  Layers: {model.config.num_hidden_layers}")

    # Process stories
    stories = args.stories if args.stories else list(range(1, 61))

    for story_id in stories:
        print(f"[story {story_id}] Processing...")

        embeddings, words = process_story(
            story_id, args.time_align_root, model, tokenizer, args.device
        )

        if embeddings is None:
            print(f"  Skipped (no data)")
            continue

        # Save in same format as GPT: story_X_word_tinyllama_0-22_2048.mat style name
        # But we'll use .npy for simplicity
        n_layers = embeddings.shape[0]
        hidden_dim = embeddings.shape[2]

        out_path = args.out_dir / f"story_{story_id}_word_tinyllama_0-{n_layers-1}_{hidden_dim}.npy"
        np.save(out_path, embeddings.astype(np.float32))

        print(f"  Saved: {embeddings.shape} -> {out_path.name}")

    print("\nDone!")


if __name__ == "__main__":
    main()
