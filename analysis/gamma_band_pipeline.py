#!/usr/bin/env python3
"""
Gamma-Band Pipeline with Proper Notch Filtering

This script computes wPLI connectivity in gamma band (30-45 Hz) from raw MEG data
with proper notch filtering at 50 Hz + harmonics (critical for gamma).

The hypothesis is that gamma band may carry stronger semantic information than theta,
based on literature showing gamma's role in semantic binding and memory.

Run:
python analysis/gamma_band_pipeline.py --out-dir outputs/gamma_rigorous --device cpu
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

# Gamma band parameters - stay BELOW 50 Hz to avoid line noise
GAMMA_BAND = (30.0, 45.0)

# Notch filter frequencies (China uses 50 Hz power line)
NOTCH_FREQS = [50.0, 100.0, 150.0]

# wPLI window parameters
WINDOW_SEC = 0.5
STEP_SEC = 0.1

# Subject split for cross-validation
ODD_SUBJECTS = ["01", "03", "05", "07", "09", "11"]
EVEN_SUBJECTS = ["02", "04", "06", "08", "10", "12"]
ALL_SUBJECTS = sorted(ODD_SUBJECTS + EVEN_SUBJECTS)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clean_word(w: str) -> str:
    """Clean word string."""
    return re.sub(r'\s+', '', str(w).strip())


def load_story_words(time_align_root: pathlib.Path, story_id: int) -> List[Tuple[float, str]]:
    """Load word onset times for a story."""
    path = time_align_root / f"story_{story_id}_word_time.mat"
    if not path.exists():
        return []

    mat = scipy.io.loadmat(str(path))
    words = mat['word'].flatten()

    # Handle both 'start'/'end' and 'onset'/'offset' formats
    if 'start' in mat:
        onsets = mat['start'].flatten()
    elif 'onset' in mat:
        onsets = mat['onset'].flatten()
    else:
        raise KeyError(f"No 'start' or 'onset' key in {path}")

    result = []
    for w, t in zip(words, onsets):
        word = clean_word(str(w))
        if word:
            result.append((float(t), word))

    return result


# ============================================================================
# PHASE 1: GAMMA wPLI WITH NOTCH FILTER
# ============================================================================

def compute_gamma_wpli_for_run(
    fif_path: pathlib.Path,
    output_dir: pathlib.Path,
    sub_id: str,
    run_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gamma-band wPLI connectivity from MEG FIF file.

    CRITICAL: Apply notch filter BEFORE gamma band analysis.
    """
    print(f"    Loading {fif_path.name}...")

    # Load raw data
    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
    sfreq = raw.info['sfreq']

    # Pick MEG channels only (magnetometers + gradiometers)
    raw.pick_types(meg=True, eeg=False, eog=False, ecg=False)

    data = raw.get_data()
    n_ch, n_time = data.shape

    print(f"    Data shape: {n_ch} channels x {n_time} samples ({n_time/sfreq:.1f}s)")
    print(f"    Sampling rate: {sfreq} Hz")

    # Apply notch filter (CRITICAL for gamma)
    print(f"    Applying notch filter at {NOTCH_FREQS} Hz...")
    for freq in NOTCH_FREQS:
        if freq < sfreq / 2:  # Below Nyquist
            data = mne.filter.notch_filter(
                data, sfreq, freqs=freq,
                method='fir',
                verbose=False
            )

    # Bandpass to gamma
    print(f"    Bandpassing to gamma ({GAMMA_BAND[0]}-{GAMMA_BAND[1]} Hz)...")
    data_filt = mne.filter.filter_data(
        data, sfreq=sfreq,
        l_freq=GAMMA_BAND[0], h_freq=GAMMA_BAND[1],
        method='fir', verbose=False
    )

    # Hilbert transform
    print(f"    Computing Hilbert transform...")
    analytic = signal.hilbert(data_filt, axis=-1)

    # Window parameters
    window_samples = int(WINDOW_SEC * sfreq)
    step_samples = int(STEP_SEC * sfreq)
    n_windows = max(0, (n_time - window_samples) // step_samples + 1)

    # Upper triangular indices for edges
    triu_idx = np.triu_indices(n_ch, k=1)
    n_edges = len(triu_idx[0])

    print(f"    Computing wPLI for {n_windows} windows, {n_edges} edges...")

    conn_ts = np.zeros((n_windows, n_edges), dtype=np.float32)
    centers = np.zeros(n_windows)

    for i in tqdm(range(n_windows), desc="    wPLI", leave=False):
        start = i * step_samples
        end = start + window_samples
        win = analytic[:, start:end]

        # wPLI computation
        # Cross-spectral density: z_i * conj(z_j)
        cross = win[:, None, :] * win[None, :, :].conj()

        # Imaginary part captures phase lag (ignores zero-lag from volume conduction)
        im = np.imag(cross)

        # wPLI = |mean(sign(Im))| weighted by |Im|
        # Numerator: absolute value of mean imaginary part
        num = np.abs(im.mean(axis=-1))
        # Denominator: mean of absolute imaginary part
        den = np.mean(np.abs(im), axis=-1) + 1e-12

        wpli = num / den
        np.fill_diagonal(wpli, 0.0)

        conn_ts[i] = wpli[triu_idx]
        centers[i] = (start + end) / 2 / sfreq

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"sub-{sub_id}_run-{run_id}_gamma_wpli"
    np.save(output_dir / f"{base_name}.npy", conn_ts)
    np.save(output_dir / f"{base_name}_window_centers_audio_s.npy", centers)

    # Save metadata
    meta = {
        "band": "gamma",
        "band_range": list(GAMMA_BAND),
        "notch_freqs": NOTCH_FREQS,
        "window_sec": WINDOW_SEC,
        "step_sec": STEP_SEC,
        "n_channels": n_ch,
        "n_edges": n_edges,
        "n_windows": n_windows,
        "sfreq": sfreq,
        "duration_sec": float(n_time / sfreq),
    }
    with open(output_dir / f"{base_name}.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"    ✓ Saved {base_name}.npy ({conn_ts.shape})")

    return conn_ts, centers


def compute_all_gamma_wpli(
    preproc_root: pathlib.Path,
    output_root: pathlib.Path,
    subjects: List[str] = None,
):
    """Compute gamma wPLI for all subjects and runs."""

    if subjects is None:
        subjects = ALL_SUBJECTS

    print("\n" + "=" * 70)
    print("PHASE 1: COMPUTE GAMMA-BAND wPLI WITH NOTCH FILTER")
    print("=" * 70)
    print(f"  Band: {GAMMA_BAND[0]}-{GAMMA_BAND[1]} Hz")
    print(f"  Notch filters: {NOTCH_FREQS} Hz")
    print(f"  Subjects: {subjects}")

    all_results = {}

    for sub in subjects:
        sub_meg_dir = preproc_root / f"sub-{sub}" / "MEG"
        if not sub_meg_dir.exists():
            print(f"\n[warn] Subject {sub} MEG dir not found")
            continue

        fif_files = list(sub_meg_dir.glob("*_meg.fif"))
        print(f"\n[{sub}] Found {len(fif_files)} FIF files")

        sub_output = output_root / f"sub-{sub}"

        for fif_path in fif_files:
            # Extract run ID
            match = re.search(r'run-(\d+)', fif_path.name)
            if not match:
                continue
            run_id = int(match.group(1))

            # Check if already computed
            out_file = sub_output / f"sub-{sub}_run-{run_id}_gamma_wpli.npy"
            if out_file.exists():
                print(f"  [skip] Run {run_id} already computed")
                continue

            try:
                conn_ts, centers = compute_gamma_wpli_for_run(
                    fif_path, sub_output, sub, run_id
                )
                all_results[f"{sub}_{run_id}"] = {
                    "n_windows": len(centers),
                    "n_edges": conn_ts.shape[1],
                }
            except Exception as e:
                print(f"  [error] Run {run_id}: {e}")

    return all_results


# ============================================================================
# PHASE 2: BUILD WORD ATLAS FROM GAMMA
# ============================================================================

def build_gamma_word_atlas(
    gamma_root: pathlib.Path,
    time_align_root: pathlib.Path,
    subjects: List[str],
) -> Tuple[List[str], np.ndarray]:
    """Build word atlas from gamma wPLI connectivity."""

    print("\n" + "=" * 70)
    print("PHASE 2: BUILD WORD ATLAS FROM GAMMA wPLI")
    print("=" * 70)

    word_conn = defaultdict(list)
    n_matched = 0
    n_total_words = 0

    for sub in subjects:
        sub_dir = gamma_root / f"sub-{sub}"
        if not sub_dir.exists():
            continue

        npy_files = list(sub_dir.glob("*_gamma_wpli.npy"))
        print(f"\n[{sub}] Found {len(npy_files)} gamma wPLI files")

        for npy_file in npy_files:
            # Extract run ID
            match = re.search(r'run-(\d+)', npy_file.name)
            if not match:
                continue
            run_id = int(match.group(1))

            # Load connectivity timeseries
            conn_ts = np.load(npy_file)

            # Load window centers
            centers_file = npy_file.parent / npy_file.name.replace('.npy', '_window_centers_audio_s.npy')
            if not centers_file.exists():
                continue
            centers = np.load(centers_file)

            # Load word times (story_id = run_id for this dataset)
            story_id = run_id
            word_times = load_story_words(time_align_root, story_id)
            n_total_words += len(word_times)

            for onset, word in word_times:
                # Find closest window
                idx = np.argmin(np.abs(centers - onset))
                if idx < len(conn_ts) and np.abs(centers[idx] - onset) < WINDOW_SEC:
                    word_conn[word].append(conn_ts[idx])
                    n_matched += 1

    print(f"\n  Total words processed: {n_total_words}")
    print(f"  Words matched to windows: {n_matched}")
    print(f"  Unique words: {len(word_conn)}")

    # Average across occurrences (min 3 occurrences for reliability)
    vocab = sorted([w for w in word_conn if len(word_conn[w]) >= 3])
    atlas = np.array([np.mean(word_conn[w], axis=0) for w in vocab])

    print(f"  Vocabulary (min 3 occurrences): {len(vocab)} words")
    print(f"  Atlas shape: {atlas.shape}")

    return vocab, atlas


# ============================================================================
# PHASE 3: LOW-RANK ADAPTER
# ============================================================================

class LowRankAdapter(nn.Module):
    """Low-rank adapter: LLM embedding -> brain PCA scores."""

    def __init__(self, hidden_dim: int, brain_dim: int, rank: int = 64):
        super().__init__()
        self.U = nn.Linear(hidden_dim, rank, bias=False)
        self.V = nn.Linear(rank, brain_dim, bias=True)
        self.norm = nn.LayerNorm(rank)
        self.dropout = nn.Dropout(0.3)

        nn.init.xavier_uniform_(self.U.weight, gain=0.1)
        nn.init.xavier_uniform_(self.V.weight, gain=0.1)

    def forward(self, x):
        h = self.U(x)
        h = self.norm(h)
        h = torch.relu(h)
        h = self.dropout(h)
        return self.V(h)


def train_adapter_cross_subject(
    X: np.ndarray,
    Y_odd: np.ndarray,
    Y_even: np.ndarray,
    rank: int = 64,
    epochs: int = 150,
    device: str = "cpu",
) -> Tuple[LowRankAdapter, Dict]:
    """
    Train adapter on ODD subjects, validate on EVEN subjects.

    This is the critical cross-subject generalization test.
    """
    print("\n" + "=" * 70)
    print("PHASE 3: LOW-RANK ADAPTER (CROSS-SUBJECT VALIDATION)")
    print("=" * 70)

    hidden_dim = X.shape[1]
    brain_dim = Y_odd.shape[1]

    adapter = LowRankAdapter(hidden_dim, brain_dim, rank=rank).to(device)
    n_params = sum(p.numel() for p in adapter.parameters())

    print(f"\n  Adapter architecture:")
    print(f"    Input: {hidden_dim} (LLM hidden dim)")
    print(f"    Bottleneck: {rank}")
    print(f"    Output: {brain_dim} (brain PCA components)")
    print(f"    Total params: {n_params:,}")
    print(f"    Params/sample ratio: {n_params / len(X):.2f}")

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    Y_odd_t = torch.tensor(Y_odd, dtype=torch.float32).to(device)
    Y_even_t = torch.tensor(Y_even, dtype=torch.float32).to(device)

    best_val_r = -1
    best_state = None
    patience = 30
    no_improve = 0

    history = {"train_loss": [], "val_r": []}

    print(f"\n  Training (ODD subjects) -> Validating (EVEN subjects)...")

    for epoch in range(epochs):
        adapter.train()
        optimizer.zero_grad()
        pred = adapter(X_t)
        loss = criterion(pred, Y_odd_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()

        # Evaluate on EVEN subjects
        adapter.eval()
        with torch.no_grad():
            val_pred = adapter(X_t).cpu().numpy()
            # Correlation per component, then average
            correlations = []
            for i in range(brain_dim):
                r, _ = pearsonr(val_pred[:, i], Y_even[:, i])
                if not np.isnan(r):
                    correlations.append(r)
            val_r = np.mean(correlations) if correlations else 0

        scheduler.step(-val_r)  # Minimize negative correlation

        history["train_loss"].append(loss.item())
        history["val_r"].append(val_r)

        if val_r > best_val_r:
            best_val_r = val_r
            best_state = {k: v.clone() for k, v in adapter.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0:
            print(f"    Epoch {epoch:3d}: loss={loss.item():.4f}, cross-subject r={val_r:.4f}")

        if no_improve >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break

    adapter.load_state_dict(best_state)

    print(f"\n  ✓ Best cross-subject correlation: {best_val_r:.4f}")

    return adapter, {"best_val_r": best_val_r, "n_params": n_params, "history": history}


# ============================================================================
# PHASE 4: GRADIENT-BASED STEERING
# ============================================================================

def gradient_steer_generation(
    model,
    tokenizer,
    adapter: LowRankAdapter,
    prompt: str,
    axis_dir: np.ndarray,
    alpha: float,
    layer_idx: int,
    device: str,
) -> Tuple[str, float]:
    """Generate with gradient-based steering along brain axis."""

    model.eval()
    adapter.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    axis_t = torch.tensor(axis_dir, dtype=torch.float32).to(device)

    # Get model layers
    if hasattr(model, 'model'):
        layers = model.model.layers
    else:
        layers = model.transformer.h

    captured = [None]
    modified = [None]

    def capture(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured[0] = h.detach().clone()
        return out

    def steer(module, inp, out):
        if modified[0] is None:
            return out
        if isinstance(out, tuple):
            return (modified[0],) + out[1:]
        return modified[0]

    # Capture and compute gradient
    handle1 = layers[layer_idx].register_forward_hook(capture)
    with torch.enable_grad():
        _ = model(inputs.input_ids)
        h = captured[0].clone().requires_grad_(True)
        h_mean = h.mean(dim=1)  # Average over sequence
        proj = adapter(h_mean)
        score = (proj * axis_t).sum()
        grad = torch.autograd.grad(score, h)[0]
    handle1.remove()

    # Apply steering: h' = h + alpha * grad
    modified[0] = captured[0] + alpha * grad

    # Generate with steered hidden states
    handle2 = layers[layer_idx].register_forward_hook(steer)
    with torch.no_grad():
        out = model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    handle2.remove()

    text = tokenizer.decode(out[0], skip_special_tokens=True)

    # Compute shift in axis score
    with torch.no_grad():
        orig_score = (adapter(captured[0].mean(dim=1)) * axis_t).sum().item()
        new_score = (adapter(modified[0].mean(dim=1)) * axis_t).sum().item()

    return text, new_score - orig_score


# ============================================================================
# MAIN
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="Gamma-band brain-to-LLM steering pipeline")
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--preproc-root", type=pathlib.Path,
                   default=pathlib.Path("derivatives/preprocessed_data"))
    p.add_argument("--embed-root", type=pathlib.Path,
                   default=pathlib.Path("derivatives/annotations/embeddings/tinyllama/word-level"))
    p.add_argument("--time-align-root", type=pathlib.Path,
                   default=pathlib.Path("derivatives/annotations/time_align/word-level"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-components", type=int, default=20)
    p.add_argument("--adapter-rank", type=int, default=64)
    p.add_argument("--skip-wpli", action="store_true", help="Skip wPLI computation if already done")
    p.add_argument("--subjects", nargs="+", default=None, help="Subjects to process")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gamma_wpli_dir = args.out_dir / "gamma_wpli"

    print("=" * 70)
    print("GAMMA-BAND BRAIN-TO-LLM STEERING PIPELINE")
    print("=" * 70)
    print(f"\nHypothesis: Gamma band (30-45 Hz) may carry stronger semantic")
    print(f"information than theta, based on gamma's role in semantic binding.")
    print(f"\nCRITICAL: Apply notch filter at 50/100/150 Hz before gamma analysis.")

    subjects = args.subjects if args.subjects else ALL_SUBJECTS

    # ===== PHASE 1: COMPUTE GAMMA wPLI =====
    if not args.skip_wpli:
        compute_all_gamma_wpli(args.preproc_root, gamma_wpli_dir, subjects)
    else:
        print("\n[skip] Using existing gamma wPLI outputs")

    # ===== PHASE 2: BUILD WORD ATLASES =====
    print("\n[atlas] Building from ODD subjects...")
    vocab_odd, atlas_odd = build_gamma_word_atlas(
        gamma_wpli_dir, args.time_align_root, ODD_SUBJECTS
    )

    print("\n[atlas] Building from EVEN subjects...")
    vocab_even, atlas_even = build_gamma_word_atlas(
        gamma_wpli_dir, args.time_align_root, EVEN_SUBJECTS
    )

    if len(vocab_odd) < 100 or len(vocab_even) < 100:
        print("\n[error] Not enough words in atlas. Ensure wPLI computation completed.")
        return

    # Find shared vocabulary
    shared_vocab = sorted(set(vocab_odd) & set(vocab_even))
    print(f"\n  Shared vocabulary: {len(shared_vocab)} words")

    # Save atlas info
    with open(args.out_dir / "atlas_info.json", "w") as f:
        json.dump({
            "vocab_odd": len(vocab_odd),
            "vocab_even": len(vocab_even),
            "shared": len(shared_vocab),
            "features": atlas_odd.shape[1] if len(atlas_odd) > 0 else 0,
        }, f, indent=2)

    # ===== PHASE 3: PCA ON ATLAS =====
    print("\n" + "=" * 70)
    print("PCA DIMENSIONALITY REDUCTION")
    print("=" * 70)

    # Get indices for shared vocab
    odd_idx = {w: i for i, w in enumerate(vocab_odd)}
    even_idx = {w: i for i, w in enumerate(vocab_even)}

    shared_odd_idx = [odd_idx[w] for w in shared_vocab]
    shared_even_idx = [even_idx[w] for w in shared_vocab]

    atlas_shared_odd = atlas_odd[shared_odd_idx]
    atlas_shared_even = atlas_even[shared_even_idx]

    # Z-score and PCA
    atlas_z_odd = (atlas_shared_odd - atlas_shared_odd.mean(0)) / (atlas_shared_odd.std(0) + 1e-8)
    atlas_z_even = (atlas_shared_even - atlas_shared_even.mean(0)) / (atlas_shared_even.std(0) + 1e-8)

    pca_odd = PCA(n_components=args.n_components, random_state=42)
    pca_even = PCA(n_components=args.n_components, random_state=42)

    scores_odd = pca_odd.fit_transform(atlas_z_odd)
    scores_even = pca_even.fit_transform(atlas_z_even)

    print(f"\n  ODD PCA variance explained: {pca_odd.explained_variance_ratio_[:5].sum():.1%} (first 5)")
    print(f"  EVEN PCA variance explained: {pca_even.explained_variance_ratio_[:5].sum():.1%} (first 5)")

    # ===== PHASE 4: LOAD LLM EMBEDDINGS =====
    print("\n" + "=" * 70)
    print("LOADING LLM EMBEDDINGS")
    print("=" * 70)

    embeddings = {}
    for story_id in range(1, 61):
        emb_files = list(args.embed_root.glob(f"story_{story_id}_word_tinyllama*.npy"))
        if not emb_files:
            continue

        emb_data = np.load(emb_files[0])[-1]  # Last layer
        words = load_story_words(args.time_align_root, story_id)

        for i, (_, word) in enumerate(words):
            if i < len(emb_data) and word not in embeddings:
                embeddings[word] = emb_data[i]

    print(f"  Loaded embeddings for {len(embeddings)} unique words")

    # Match with shared vocabulary
    matched_words = [w for w in shared_vocab if w in embeddings]
    print(f"  Matched with shared vocab: {len(matched_words)} words")

    if len(matched_words) < 500:
        print(f"\n[error] Too few matched words ({len(matched_words)}). Need at least 500.")
        return

    # Prepare training data
    matched_idx_odd = [shared_vocab.index(w) for w in matched_words]
    matched_idx_even = [shared_vocab.index(w) for w in matched_words]

    X = np.array([embeddings[w] for w in matched_words])
    Y_odd = scores_odd[matched_idx_odd]
    Y_even = scores_even[matched_idx_even]

    print(f"  X (LLM embeddings): {X.shape}")
    print(f"  Y_odd (brain PCA): {Y_odd.shape}")
    print(f"  Y_even (brain PCA): {Y_even.shape}")

    # ===== PHASE 5: TRAIN ADAPTER =====
    adapter, train_info = train_adapter_cross_subject(
        X, Y_odd, Y_even,
        rank=args.adapter_rank,
        device=args.device
    )

    # Save adapter
    torch.save(adapter.state_dict(), args.out_dir / "gamma_adapter.pt")

    # Save training info
    with open(args.out_dir / "adapter_info.json", "w") as f:
        json.dump({
            "best_cross_subject_r": float(train_info["best_val_r"]),
            "n_params": train_info["n_params"],
            "n_training_samples": len(X),
            "llm_hidden_dim": X.shape[1],
            "brain_pca_dim": Y_odd.shape[1],
            "adapter_rank": args.adapter_rank,
        }, f, indent=2)

    # ===== CRITICAL CHECK =====
    print("\n" + "=" * 70)
    print("CROSS-SUBJECT GENERALIZATION CHECK")
    print("=" * 70)

    cross_r = train_info["best_val_r"]
    if cross_r < 0.05:
        print(f"\n  ⚠ WARNING: Cross-subject r = {cross_r:.4f}")
        print(f"  Brain-LLM alignment does not generalize across subjects.")
        print(f"  Steering is unlikely to produce meaningful semantic effects.")
        print(f"\n  This is a NEGATIVE RESULT - gamma band doesn't improve alignment.")
    elif cross_r < 0.15:
        print(f"\n  ⚠ CAUTION: Cross-subject r = {cross_r:.4f} (weak)")
        print(f"  Some generalization, but effects may be noisy.")
    else:
        print(f"\n  ✓ Cross-subject r = {cross_r:.4f}")
        print(f"  Reasonable brain-LLM alignment for steering.")

    # ===== PHASE 6: STEERING EVALUATION =====
    print("\n" + "=" * 70)
    print("PHASE 6: GRADIENT-BASED STEERING EVALUATION")
    print("=" * 70)

    print("\n[steer] Loading TinyLlama...")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float32,  # Use float32 for CPU
    ).to(args.device)
    model.eval()

    # Test prompts
    prompts = [
        "The scientist discovered that",
        "In the quiet village, people",
        "She felt a sudden wave of",
        "The ancient temple stood",
        "After years of hard work",
    ]

    # Use first PCA axis as steering direction (unit vector in component space)
    axis_dir = np.zeros(args.n_components)
    axis_dir[0] = 1.0

    print(f"\n[steer] Testing {len(prompts)} prompts with gradient steering...")

    steering_results = []
    for prompt in prompts:
        # Baseline
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
            out = model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            baseline = tokenizer.decode(out[0], skip_special_tokens=True)

        # Positive steering
        pos_text, pos_shift = gradient_steer_generation(
            model, tokenizer, adapter, prompt, axis_dir,
            alpha=2.0, layer_idx=11, device=args.device
        )

        # Negative steering
        neg_text, neg_shift = gradient_steer_generation(
            model, tokenizer, adapter, prompt, axis_dir,
            alpha=-2.0, layer_idx=11, device=args.device
        )

        steering_results.append({
            "prompt": prompt,
            "baseline": baseline,
            "positive": pos_text,
            "negative": neg_text,
            "pos_shift": float(pos_shift),
            "neg_shift": float(neg_shift),
        })

        print(f"\n  Prompt: {prompt[:30]}...")
        print(f"    Δ pos: {pos_shift:+.3f}, Δ neg: {neg_shift:+.3f}")
        print(f"    Baseline: {baseline[len(prompt):len(prompt)+50]}...")
        print(f"    Positive: {pos_text[len(prompt):len(prompt)+50]}...")
        print(f"    Negative: {neg_text[len(prompt):len(prompt)+50]}...")

    # Statistical test
    pos_shifts = [r["pos_shift"] for r in steering_results]
    neg_shifts = [r["neg_shift"] for r in steering_results]

    t_stat, p_val = ttest_ind(pos_shifts, neg_shifts)
    pooled_std = np.std(pos_shifts + neg_shifts)
    d = (np.mean(pos_shifts) - np.mean(neg_shifts)) / pooled_std if pooled_std > 0 else 0

    print(f"\n[stats] Steering effect:")
    print(f"  Mean pos shift: {np.mean(pos_shifts):+.3f} ± {np.std(pos_shifts):.3f}")
    print(f"  Mean neg shift: {np.mean(neg_shifts):+.3f} ± {np.std(neg_shifts):.3f}")
    print(f"  t = {t_stat:.3f}, p = {p_val:.4f}")
    print(f"  Cohen's d = {d:.3f}")

    # Save results
    with open(args.out_dir / "steering_results.json", "w") as f:
        json.dump({
            "results": steering_results,
            "stats": {
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "cohens_d": float(d),
                "mean_pos_shift": float(np.mean(pos_shifts)),
                "mean_neg_shift": float(np.mean(neg_shifts)),
                "std_pos_shift": float(np.std(pos_shifts)),
                "std_neg_shift": float(np.std(neg_shifts)),
            },
            "adapter_cross_subject_r": float(train_info["best_val_r"]),
        }, f, indent=2)

    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("GAMMA-BAND PIPELINE SUMMARY")
    print("=" * 70)
    print(f"\n  Band: Gamma (30-45 Hz) with notch filter at 50/100/150 Hz")
    print(f"  Words in atlas: {len(shared_vocab)}")
    print(f"  Adapter cross-subject r: {train_info['best_val_r']:.4f}")
    print(f"  Steering Cohen's d: {d:.3f}")
    print(f"  Steering p-value: {p_val:.4f}")

    # Interpret results
    print(f"\n  INTERPRETATION:")
    if train_info['best_val_r'] < 0.05:
        print(f"    ✗ Brain-LLM alignment too weak for meaningful steering")
        print(f"    Gamma band does NOT improve alignment over theta")
    elif p_val > 0.05:
        print(f"    ✗ Steering effect not significant")
    else:
        # Check for degenerate outputs
        has_degenerate = any(
            len(set(r["positive"].split())) < 10 or
            len(set(r["negative"].split())) < 10
            for r in steering_results
        )
        if has_degenerate:
            print(f"    ⚠ WARNING: Degenerate outputs detected (repetitive text)")
            print(f"    Statistical significance is MISLEADING - this is model collapse")
        else:
            print(f"    ✓ Meaningful steering effect detected")

    print(f"\n  Results saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
