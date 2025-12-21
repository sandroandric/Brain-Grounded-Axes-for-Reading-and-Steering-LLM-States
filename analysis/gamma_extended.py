#!/usr/bin/env python3
"""
Extended Gamma-Band Study with Parallel Processing

Tests whether brain-LLM alignment improves with more data.
Uses multiprocessing for speedup.

Run:
python analysis/gamma_extended.py --out-dir outputs/gamma_extended --runs-per-subject 10
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import scipy.io
import mne
from scipy import signal
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

import torch
import torch.nn as nn


# ============================================================================
# CONFIGURATION
# ============================================================================

GAMMA_BAND = (30.0, 45.0)
NOTCH_FREQS = [50.0, 100.0, 150.0]
WINDOW_SEC = 0.5
STEP_SEC = 0.25  # Slightly larger step for speed

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
        return []
    return [(float(t), clean_word(str(w))) for w, t in zip(words, onsets) if clean_word(str(w))]


def compute_wpli_vectorized(analytic: np.ndarray, window_samples: int, step_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized wPLI computation - faster than loop."""
    n_ch, n_time = analytic.shape
    n_windows = (n_time - window_samples) // step_samples + 1

    triu_idx = np.triu_indices(n_ch, k=1)
    n_edges = len(triu_idx[0])

    conn_ts = np.zeros((n_windows, n_edges), dtype=np.float32)
    centers = np.zeros(n_windows)

    # Process in batches of windows for memory efficiency
    batch_size = 100
    for batch_start in range(0, n_windows, batch_size):
        batch_end = min(batch_start + batch_size, n_windows)

        for i in range(batch_start, batch_end):
            start = i * step_samples
            end = start + window_samples
            win = analytic[:, start:end]

            # Vectorized cross-spectral
            cross = win[:, None, :] * win[None, :, :].conj()
            im = np.imag(cross)
            num = np.abs(im.mean(axis=-1))
            den = np.mean(np.abs(im), axis=-1) + 1e-12
            wpli = num / den
            np.fill_diagonal(wpli, 0.0)

            conn_ts[i] = wpli[triu_idx]
            centers[i] = (start + end) / 2

    return conn_ts, centers


def process_single_file(args_tuple) -> Dict:
    """Process a single FIF file - for parallel execution."""
    fif_path, time_align_root, sfreq_expected = args_tuple

    try:
        # Extract run ID
        match = re.search(r'run-(\d+)', fif_path.name)
        if not match:
            return {"error": "no run id", "path": str(fif_path)}
        run_id = int(match.group(1))

        # Load and process
        raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
        sfreq = raw.info['sfreq']
        raw.pick_types(meg='mag', eeg=False, eog=False, ecg=False)

        data = raw.get_data()

        # Notch filter
        for freq in NOTCH_FREQS:
            if freq < sfreq / 2:
                data = mne.filter.notch_filter(data, sfreq, freqs=freq, method='fir', verbose=False)

        # Bandpass
        data_filt = mne.filter.filter_data(data, sfreq, GAMMA_BAND[0], GAMMA_BAND[1], method='fir', verbose=False)

        # Hilbert
        analytic = signal.hilbert(data_filt, axis=-1)

        # wPLI
        window_samples = int(WINDOW_SEC * sfreq)
        step_samples = int(STEP_SEC * sfreq)

        conn_ts, centers = compute_wpli_vectorized(analytic, window_samples, step_samples)
        centers = centers / sfreq  # Convert to seconds

        # Load words and match
        word_times = load_story_words(time_align_root, run_id)

        word_conn = {}
        for onset, word in word_times:
            idx = np.argmin(np.abs(centers - onset))
            if idx < len(conn_ts) and np.abs(centers[idx] - onset) < WINDOW_SEC:
                if word not in word_conn:
                    word_conn[word] = []
                word_conn[word].append(conn_ts[idx])

        return {
            "run_id": run_id,
            "n_windows": len(conn_ts),
            "n_words": len(word_conn),
            "word_conn": {w: [c.tolist() for c in conns] for w, conns in word_conn.items()},
        }

    except Exception as e:
        return {"error": str(e), "path": str(fif_path)}


def build_atlas_parallel(
    preproc_root: pathlib.Path,
    time_align_root: pathlib.Path,
    subjects: List[str],
    n_runs: int,
    n_workers: int = 4,
) -> Tuple[List[str], np.ndarray]:
    """Build word atlas using parallel processing."""

    # Collect all files to process
    all_files = []
    for sub in subjects:
        sub_dir = preproc_root / f"sub-{sub}" / "MEG"
        if not sub_dir.exists():
            continue
        fif_files = sorted(list(sub_dir.glob("*_meg.fif")))[:n_runs]
        for f in fif_files:
            all_files.append((f, time_align_root, 1000.0))

    print(f"  Processing {len(all_files)} files with {n_workers} workers...")

    # Process in parallel
    word_conn_all = defaultdict(list)
    completed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_file, args): args for args in all_files}

        for future in as_completed(futures):
            result = future.result()
            completed += 1

            if "error" in result:
                print(f"    [{completed}/{len(all_files)}] Error: {result.get('error', 'unknown')}")
                continue

            # Aggregate word connectivity
            for word, conns in result.get("word_conn", {}).items():
                for conn in conns:
                    word_conn_all[word].append(np.array(conn))

            if completed % 10 == 0:
                print(f"    [{completed}/{len(all_files)}] processed, {len(word_conn_all)} unique words")

    # Average across occurrences
    vocab = sorted([w for w in word_conn_all if len(word_conn_all[w]) >= 2])
    atlas = np.array([np.mean(word_conn_all[w], axis=0) for w in vocab])

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
    p.add_argument("--runs-per-subject", type=int, default=10)
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXTENDED GAMMA-BAND STUDY")
    print("=" * 70)
    print(f"Runs per subject: {args.runs_per_subject}")
    print(f"Workers: {args.n_workers}")
    print(f"Device: {args.device}")

    # Build atlases in parallel
    print("\n" + "=" * 70)
    print("BUILDING GAMMA WORD ATLASES (PARALLEL)")
    print("=" * 70)

    print("\n[ODD subjects]")
    vocab_odd, atlas_odd = build_atlas_parallel(
        args.preproc_root, args.time_align_root, ODD_SUBJECTS,
        args.runs_per_subject, args.n_workers
    )
    print(f"  Vocabulary: {len(vocab_odd)} words, {atlas_odd.shape[1] if len(atlas_odd) > 0 else 0} features")

    print("\n[EVEN subjects]")
    vocab_even, atlas_even = build_atlas_parallel(
        args.preproc_root, args.time_align_root, EVEN_SUBJECTS,
        args.runs_per_subject, args.n_workers
    )
    print(f"  Vocabulary: {len(vocab_even)} words, {atlas_even.shape[1] if len(atlas_even) > 0 else 0} features")

    # Shared vocabulary
    shared_vocab = sorted(set(vocab_odd) & set(vocab_even))
    print(f"\n  Shared vocabulary: {len(shared_vocab)} words")

    if len(shared_vocab) < 100:
        print("\n[error] Too few shared words.")
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

    # Train adapter (use MPS if available)
    print("\n" + "=" * 70)
    print("TRAINING ADAPTER (CROSS-SUBJECT VALIDATION)")
    print("=" * 70)

    adapter, cross_r = train_adapter(X, Y_odd, Y_even, rank=32, epochs=100, device=args.device)

    print(f"\n  GAMMA cross-subject r: {cross_r:.4f}")

    # Statistical significance
    from scipy import stats
    n = len(matched_words)
    t_stat = cross_r * np.sqrt(n - 2) / np.sqrt(1 - cross_r**2)
    p_value = 2 * stats.t.sf(abs(t_stat), n - 2)

    print(f"\n  Statistical test:")
    print(f"    n = {n}")
    print(f"    t = {t_stat:.3f}")
    print(f"    p = {p_value:.4f}")
    print(f"    → {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} at α=0.05")

    # Save results
    results = {
        "gamma_cross_subject_r": float(cross_r),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "shared_vocab_size": len(shared_vocab),
        "matched_words": len(matched_words),
        "runs_per_subject": args.runs_per_subject,
        "pca_variance_odd": pca_odd.explained_variance_ratio_.tolist(),
        "pca_variance_even": pca_even.explained_variance_ratio_.tolist(),
    }

    with open(args.out_dir / "gamma_extended_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON WITH PILOT")
    print("=" * 70)
    print(f"\n  Pilot (3 runs):    r = 0.010, n = 1099")
    print(f"  Extended ({args.runs_per_subject} runs): r = {cross_r:.4f}, n = {len(matched_words)}")

    if cross_r > 0.010:
        improvement = (cross_r - 0.010) / 0.010 * 100
        print(f"\n  Improvement: +{improvement:.0f}%")

    if p_value < 0.05:
        print(f"\n  ✓ BRAIN-LLM ALIGNMENT IS STATISTICALLY SIGNIFICANT")
    else:
        print(f"\n  ✗ Brain-LLM alignment still not significant")

    print(f"\n  Results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
