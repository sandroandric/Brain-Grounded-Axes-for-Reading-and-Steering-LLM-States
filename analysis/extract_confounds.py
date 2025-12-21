"""
Extract word-level confounds (frequency, surprisal) from SMN4Lang annotations.

Aggregates per-position values to per-word-type by averaging across all occurrences.

Outputs:
- confounds.json: {word: {logfreq: float, surprisal: float, count: int}}

Run:
python analysis/extract_confounds.py \
  --time-align-root derivatives/annotations/time_align/word-level \
  --freq-root derivatives/annotations/frequency/word-level \
  --gpt-root derivatives/annotations/embeddings/gpt/word-level \
  --out-json metadata/lexica/confounds.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import scipy.io


def clean_word(w: str) -> str:
    """Clean word string."""
    return re.sub(r'\s+', '', str(w).strip())


def load_story_words_and_freq(time_align_path: pathlib.Path, freq_path: pathlib.Path) -> List[Tuple[str, float]]:
    """Load words and their frequencies from a single story."""
    time_mat = scipy.io.loadmat(str(time_align_path))
    freq_mat = scipy.io.loadmat(str(freq_path))

    # Extract words
    words_raw = time_mat['word'].flatten()
    words = [clean_word(str(w)) for w in words_raw]

    # Extract frequencies
    freqs = freq_mat['wf'].flatten()

    if len(words) != len(freqs):
        print(f"[warn] Length mismatch: {len(words)} words vs {len(freqs)} freqs in {time_align_path.name}")
        min_len = min(len(words), len(freqs))
        words = words[:min_len]
        freqs = freqs[:min_len]

    return list(zip(words, freqs))


def load_story_gpt_embeddings(gpt_path: pathlib.Path) -> np.ndarray:
    """Load GPT embeddings for a story. Returns shape (n_words, dim) using last layer."""
    mat = scipy.io.loadmat(str(gpt_path))
    # Shape is (n_layers, n_words, dim) - use last layer
    emb = mat['data']
    if emb.ndim == 3:
        return emb[-1]  # Last layer, shape (n_words, dim)
    return emb


def compute_surprisal_proxy(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute surprisal proxy as embedding change magnitude.

    surprisal[i] = ||emb[i] - emb[i-1]|| for i > 0
    surprisal[0] = ||emb[0]|| (no context)
    """
    n_words = embeddings.shape[0]
    surprisal = np.zeros(n_words)

    # First word: norm of embedding (no context)
    surprisal[0] = np.linalg.norm(embeddings[0])

    # Subsequent words: distance from previous
    for i in range(1, n_words):
        surprisal[i] = np.linalg.norm(embeddings[i] - embeddings[i - 1])

    return surprisal


def aggregate_confounds(
    word_freq_pairs: List[Tuple[str, float]],
    word_surprisal_pairs: List[Tuple[str, float]]
) -> Dict[str, Dict[str, float]]:
    """Aggregate confounds per word-type."""
    freq_sums = defaultdict(float)
    freq_counts = defaultdict(int)
    surp_sums = defaultdict(float)
    surp_counts = defaultdict(int)

    for word, freq in word_freq_pairs:
        if word:  # Skip empty words
            freq_sums[word] += freq
            freq_counts[word] += 1

    for word, surp in word_surprisal_pairs:
        if word:
            surp_sums[word] += surp
            surp_counts[word] += 1

    # Combine
    all_words = set(freq_sums.keys()) | set(surp_sums.keys())
    result = {}

    for word in all_words:
        result[word] = {
            "logfreq": freq_sums[word] / freq_counts[word] if freq_counts[word] > 0 else None,
            "logfreq_count": freq_counts[word],
            "surprisal": surp_sums[word] / surp_counts[word] if surp_counts[word] > 0 else None,
            "surprisal_count": surp_counts[word],
        }

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--time-align-root", type=pathlib.Path, required=True)
    p.add_argument("--freq-root", type=pathlib.Path, required=True)
    p.add_argument("--gpt-root", type=pathlib.Path, required=True)
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    args = p.parse_args()

    # Find all stories
    time_files = sorted(args.time_align_root.glob("*.mat"))
    print(f"Found {len(time_files)} time_align files")

    all_word_freq = []
    all_word_surp = []

    for time_file in time_files:
        story_id = time_file.stem.replace("_word_time", "")

        # Find matching frequency file
        freq_file = args.freq_root / f"{story_id}_word_logfreq.mat"
        if not freq_file.exists():
            print(f"[skip] No frequency file for {story_id}")
            continue

        # Find matching GPT file (pattern: story_XX_word_gpt_0-24_1024.mat)
        gpt_files = list(args.gpt_root.glob(f"{story_id}_word_gpt*.mat"))
        if not gpt_files:
            print(f"[skip] No GPT file for {story_id}")
            continue
        gpt_file = gpt_files[0]

        print(f"[load] {story_id}")

        # Load words and frequencies
        word_freq = load_story_words_and_freq(time_file, freq_file)
        all_word_freq.extend(word_freq)

        # Load GPT and compute surprisal
        try:
            gpt_emb = load_story_gpt_embeddings(gpt_file)
            surprisal = compute_surprisal_proxy(gpt_emb)

            # Get words again for pairing
            time_mat = scipy.io.loadmat(str(time_file))
            words = [clean_word(str(w)) for w in time_mat['word'].flatten()]

            if len(words) != len(surprisal):
                print(f"  [warn] GPT length mismatch: {len(words)} words vs {len(surprisal)} surprisal")
                min_len = min(len(words), len(surprisal))
                words = words[:min_len]
                surprisal = surprisal[:min_len]

            all_word_surp.extend(zip(words, surprisal))
        except Exception as e:
            print(f"  [error] GPT processing failed: {e}")

    print(f"\nTotal word-frequency pairs: {len(all_word_freq)}")
    print(f"Total word-surprisal pairs: {len(all_word_surp)}")

    # Aggregate
    confounds = aggregate_confounds(all_word_freq, all_word_surp)
    print(f"Unique word types with confounds: {len(confounds)}")

    # Summary stats
    logfreqs = [v["logfreq"] for v in confounds.values() if v["logfreq"] is not None]
    surprisals = [v["surprisal"] for v in confounds.values() if v["surprisal"] is not None]

    print(f"\nLogfreq: mean={np.mean(logfreqs):.2f}, std={np.std(logfreqs):.2f}, range=[{np.min(logfreqs):.2f}, {np.max(logfreqs):.2f}]")
    print(f"Surprisal: mean={np.mean(surprisals):.2f}, std={np.std(surprisals):.2f}, range=[{np.min(surprisals):.2f}, {np.max(surprisals):.2f}]")

    # Save
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(confounds, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {args.out_json}")


if __name__ == "__main__":
    main()
