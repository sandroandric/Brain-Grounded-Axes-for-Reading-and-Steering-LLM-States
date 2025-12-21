#!/usr/bin/env python3
"""
Gamma-Band Pilot Study

A lightweight version that processes only 5 runs per subject to quickly test
whether gamma band provides better brain-LLM alignment than theta.

Key optimizations:
1. Only 5 runs per subject (vs 60)
2. Magnetometers only (102 channels vs 306)
3. Larger step size (0.2s vs 0.1s)
4. Uses existing theta results for comparison

Run:
python analysis/gamma_pilot.py --out-dir outputs/gamma_pilot
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import scipy.io
import mne
from scipy import signal
from scipy.stats import pearsonr, ttest_ind
from sklearn.decomposition import PCA
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================================
# CONFIGURATION
# ============================================================================

GAMMA_BAND = (30.0, 45.0)
NOTCH_FREQS = [50.0, 100.0, 150.0]
WINDOW_SEC = 0.5
STEP_SEC = 0.2  # Larger step = fewer windows

RUNS_PER_SUBJECT = 5  # Only process 5 runs per subject

ODD_SUBJECTS = ["01", "03", "05", "07", "09", "11"]
EVEN_SUBJECTS = ["02", "04", "06", "08", "10", "12"]


def clean_word(w: str) -> str:
    return re.sub(r'\s+', '', str(w).strip())


def load_story_words(time_align_root: pathlib.Path, story_id: int) -> List[Tuple[float, str]]:
    path = time_align_root / f"story_{story_id}_word_time.mat"
    if not path.exists():
        return []

    mat = scipy.io.loadmat(str(path))
    words = mat['word'].flatten()

    if 'start' in mat:
        onsets = mat['start'].flatten()
    elif 'onset' in mat:
        onsets = mat['onset'].flatten()
    else:
        raise KeyError(f"No 'start' or 'onset' key in {path}")

    return [(float(t), clean_word(str(w))) for w, t in zip(words, onsets) if clean_word(str(w))]


def compute_gamma_wpli_fast(fif_path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    """Compute gamma wPLI - optimized for speed."""

    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
    sfreq = raw.info['sfreq']

    # Use only magnetometers (102 channels instead of 306)
    raw.pick_types(meg='mag', eeg=False, eog=False, ecg=False)

    data = raw.get_data()
    n_ch, n_time = data.shape

    # Notch filter
    for freq in NOTCH_FREQS:
        if freq < sfreq / 2:
            data = mne.filter.notch_filter(data, sfreq, freqs=freq, method='fir', verbose=False)

    # Bandpass
    data_filt = mne.filter.filter_data(data, sfreq, GAMMA_BAND[0], GAMMA_BAND[1], method='fir', verbose=False)

    # Hilbert
    analytic = signal.hilbert(data_filt, axis=-1)

    window_samples = int(WINDOW_SEC * sfreq)
    step_samples = int(STEP_SEC * sfreq)
    n_windows = (n_time - window_samples) // step_samples + 1

    triu_idx = np.triu_indices(n_ch, k=1)
    n_edges = len(triu_idx[0])

    conn_ts = np.zeros((n_windows, n_edges), dtype=np.float32)
    centers = np.zeros(n_windows)

    for i in range(n_windows):
        start = i * step_samples
        end = start + window_samples
        win = analytic[:, start:end]

        cross = win[:, None, :] * win[None, :, :].conj()
        im = np.imag(cross)
        num = np.abs(im.mean(axis=-1))
        den = np.mean(np.abs(im), axis=-1) + 1e-12
        wpli = num / den
        np.fill_diagonal(wpli, 0.0)

        conn_ts[i] = wpli[triu_idx]
        centers[i] = (start + end) / 2 / sfreq

    return conn_ts, centers


def build_gamma_atlas(
    preproc_root: pathlib.Path,
    time_align_root: pathlib.Path,
    subjects: List[str],
    n_runs: int = RUNS_PER_SUBJECT,
) -> Tuple[List[str], np.ndarray]:
    """Build word atlas from gamma connectivity."""

    word_conn = defaultdict(list)

    for sub in subjects:
        sub_dir = preproc_root / f"sub-{sub}" / "MEG"
        if not sub_dir.exists():
            continue

        fif_files = sorted(list(sub_dir.glob("*_meg.fif")))[:n_runs]
        print(f"[{sub}] Processing {len(fif_files)} files...")

        for fif_path in tqdm(fif_files, desc=f"  Subject {sub}"):
            # Extract run ID
            match = re.search(r'run-(\d+)', fif_path.name)
            if not match:
                continue
            run_id = int(match.group(1))

            try:
                conn_ts, centers = compute_gamma_wpli_fast(fif_path)
            except Exception as e:
                print(f"    [error] {fif_path.name}: {e}")
                continue

            # Load word times
            story_id = run_id
            word_times = load_story_words(time_align_root, story_id)

            for onset, word in word_times:
                idx = np.argmin(np.abs(centers - onset))
                if idx < len(conn_ts) and np.abs(centers[idx] - onset) < WINDOW_SEC:
                    word_conn[word].append(conn_ts[idx])

    vocab = sorted([w for w in word_conn if len(word_conn[w]) >= 2])
    atlas = np.array([np.mean(word_conn[w], axis=0) for w in vocab])

    return vocab, atlas


class LowRankAdapter(nn.Module):
    def __init__(self, hidden_dim: int, brain_dim: int, rank: int = 32):
        super().__init__()
        self.U = nn.Linear(hidden_dim, rank, bias=False)
        self.V = nn.Linear(rank, brain_dim, bias=True)
        nn.init.xavier_uniform_(self.U.weight, gain=0.1)
        nn.init.xavier_uniform_(self.V.weight, gain=0.1)

    def forward(self, x):
        return self.V(torch.relu(self.U(x)))


def train_adapter(X, Y_train, Y_val, rank=32, epochs=100, device="cpu"):
    """Train adapter with cross-subject validation."""

    hidden_dim = X.shape[1]
    brain_dim = Y_train.shape[1]

    adapter = LowRankAdapter(hidden_dim, brain_dim, rank=rank).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)

    best_r = -1
    best_state = None

    for epoch in range(epochs):
        adapter.train()
        optimizer.zero_grad()
        pred = adapter(X_t)
        loss = ((pred - Y_train_t) ** 2).mean()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            adapter.eval()
            with torch.no_grad():
                val_pred = adapter(X_t).cpu().numpy()
                corrs = [pearsonr(val_pred[:, i], Y_val[:, i])[0] for i in range(Y_val.shape[1])]
                val_r = np.nanmean(corrs)

            if val_r > best_r:
                best_r = val_r
                best_state = {k: v.clone() for k, v in adapter.state_dict().items()}

            print(f"  Epoch {epoch}: loss={loss.item():.4f}, cross-val r={val_r:.4f}")

    if best_state:
        adapter.load_state_dict(best_state)

    return adapter, best_r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--preproc-root", type=pathlib.Path, default=pathlib.Path("derivatives/preprocessed_data"))
    p.add_argument("--embed-root", type=pathlib.Path,
                   default=pathlib.Path("derivatives/annotations/embeddings/tinyllama/word-level"))
    p.add_argument("--time-align-root", type=pathlib.Path,
                   default=pathlib.Path("derivatives/annotations/time_align/word-level"))
    p.add_argument("--n-components", type=int, default=10)
    p.add_argument("--runs-per-subject", type=int, default=5)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GAMMA-BAND PILOT STUDY")
    print("=" * 70)
    print(f"\nTesting if gamma (30-45 Hz) provides better brain-LLM alignment than theta.")
    print(f"Processing {args.runs_per_subject} runs per subject (pilot study).")

    # Build atlases
    print("\n" + "=" * 70)
    print("BUILDING GAMMA WORD ATLASES")
    print("=" * 70)

    print("\n[ODD subjects]")
    vocab_odd, atlas_odd = build_gamma_atlas(
        args.preproc_root, args.time_align_root, ODD_SUBJECTS, args.runs_per_subject
    )
    print(f"  Vocabulary: {len(vocab_odd)} words, {atlas_odd.shape[1]} features")

    print("\n[EVEN subjects]")
    vocab_even, atlas_even = build_gamma_atlas(
        args.preproc_root, args.time_align_root, EVEN_SUBJECTS, args.runs_per_subject
    )
    print(f"  Vocabulary: {len(vocab_even)} words, {atlas_even.shape[1]} features")

    # Shared vocabulary
    shared_vocab = sorted(set(vocab_odd) & set(vocab_even))
    print(f"\n  Shared vocabulary: {len(shared_vocab)} words")

    if len(shared_vocab) < 100:
        print("\n[error] Too few shared words. Increase runs_per_subject.")
        return

    # PCA
    odd_idx = {w: i for i, w in enumerate(vocab_odd)}
    even_idx = {w: i for i, w in enumerate(vocab_even)}

    shared_odd_idx = [odd_idx[w] for w in shared_vocab]
    shared_even_idx = [even_idx[w] for w in shared_vocab]

    atlas_shared_odd = atlas_odd[shared_odd_idx]
    atlas_shared_even = atlas_even[shared_even_idx]

    atlas_z_odd = (atlas_shared_odd - atlas_shared_odd.mean(0)) / (atlas_shared_odd.std(0) + 1e-8)
    atlas_z_even = (atlas_shared_even - atlas_shared_even.mean(0)) / (atlas_shared_even.std(0) + 1e-8)

    pca_odd = PCA(n_components=args.n_components, random_state=42)
    pca_even = PCA(n_components=args.n_components, random_state=42)

    scores_odd = pca_odd.fit_transform(atlas_z_odd)
    scores_even = pca_even.fit_transform(atlas_z_even)

    print(f"\n  PCA variance (ODD): {pca_odd.explained_variance_ratio_[:5].sum():.1%}")
    print(f"  PCA variance (EVEN): {pca_even.explained_variance_ratio_[:5].sum():.1%}")

    # Load embeddings
    print("\n" + "=" * 70)
    print("LOADING LLM EMBEDDINGS")
    print("=" * 70)

    embeddings = {}
    for story_id in range(1, 61):
        emb_files = list(args.embed_root.glob(f"story_{story_id}_word_tinyllama*.npy"))
        if not emb_files:
            continue
        emb_data = np.load(emb_files[0])[-1]
        words = load_story_words(args.time_align_root, story_id)
        for i, (_, word) in enumerate(words):
            if i < len(emb_data) and word not in embeddings:
                embeddings[word] = emb_data[i]

    print(f"  Loaded {len(embeddings)} embeddings")

    # Match
    matched_words = [w for w in shared_vocab if w in embeddings]
    print(f"  Matched: {len(matched_words)} words")

    matched_idx = [shared_vocab.index(w) for w in matched_words]
    X = np.array([embeddings[w] for w in matched_words])
    Y_odd = scores_odd[matched_idx]
    Y_even = scores_even[matched_idx]

    # Train adapter
    print("\n" + "=" * 70)
    print("TRAINING ADAPTER (CROSS-SUBJECT VALIDATION)")
    print("=" * 70)

    adapter, cross_r = train_adapter(X, Y_odd, Y_even, rank=32, epochs=100)

    print(f"\n  GAMMA cross-subject r: {cross_r:.4f}")

    # Compare with theta baseline
    # We know from previous runs that theta gave r ~ 0.003
    theta_r = 0.003

    print("\n" + "=" * 70)
    print("COMPARISON: GAMMA vs THETA")
    print("=" * 70)
    print(f"\n  Theta (previous): r = {theta_r:.4f}")
    print(f"  Gamma (pilot):    r = {cross_r:.4f}")

    improvement = (cross_r - theta_r) / abs(theta_r) * 100 if theta_r != 0 else float('inf')

    if cross_r > theta_r + 0.01:
        print(f"\n  ✓ GAMMA shows improvement (+{improvement:.0f}%)")
        print(f"    Worth running full gamma pipeline.")
    elif cross_r > theta_r:
        print(f"\n  ~ Gamma slightly better but marginal (+{improvement:.0f}%)")
        print(f"    May not be worth the computational cost.")
    else:
        print(f"\n  ✗ Gamma NOT better than theta ({improvement:.0f}%)")
        print(f"    Band choice is not the issue.")
        print(f"    Problem is fundamental: MEG→LLM alignment too weak.")

    # Save results
    results = {
        "gamma_cross_subject_r": float(cross_r),
        "theta_cross_subject_r": float(theta_r),
        "shared_vocab_size": len(shared_vocab),
        "matched_words": len(matched_words),
        "runs_per_subject": args.runs_per_subject,
        "pca_variance_odd": pca_odd.explained_variance_ratio_.tolist(),
        "pca_variance_even": pca_even.explained_variance_ratio_.tolist(),
    }

    with open(args.out_dir / "gamma_pilot_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {args.out_dir}/gamma_pilot_results.json")


if __name__ == "__main__":
    main()
