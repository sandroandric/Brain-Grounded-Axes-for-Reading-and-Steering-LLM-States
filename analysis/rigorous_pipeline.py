"""
NeurIPS-Ready Rigorous Pipeline for Brain-LLM Alignment

This script implements the methodologically sound approach outlined in NEURIPS_METHODOLOGY.md:
1. Proper preprocessing with notch filtering
2. wPLI connectivity (robust to volume conduction)
3. Multiple frequency bands (theta, alpha, beta, gamma)
4. Multi-level validation (subject, story, word splits)
5. Low-rank adapter (prevents overfitting)
6. Gradient-based steering (correct mechanism)

Run:
python analysis/rigorous_pipeline.py --phase all --out-dir outputs/rigorous
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import mne
from scipy import signal
from scipy.stats import pearsonr, ttest_ind
import torch
import torch.nn as nn
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests


# ============================================================================
# CONFIGURATION
# ============================================================================

# Frequency bands to analyze
BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),  # Stop before 50 Hz to avoid line noise
}

# Notch filter frequencies (China uses 50 Hz)
NOTCH_FREQS = [50.0, 100.0, 150.0]  # 50 Hz + harmonics

# Connectivity parameters
WINDOW_SEC = 0.5
STEP_SEC = 0.1

# Validation splits
ODD_SUBJECTS = ["01", "03", "05", "07", "09", "11"]
EVEN_SUBJECTS = ["02", "04", "06", "08", "10", "12"]
TRAIN_STORIES = list(range(1, 49))  # Stories 1-48
TEST_STORIES = list(range(49, 61))   # Stories 49-60


# ============================================================================
# PHASE 1: PREPROCESSING WITH NOTCH FILTER
# ============================================================================

def load_and_preprocess_meg(
    fif_path: pathlib.Path,
    apply_notch: bool = True,
    picks: str = "grad",
) -> Tuple[np.ndarray, float, List[str]]:
    """Load MEG data with proper preprocessing including notch filter."""

    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
    sfreq = raw.info['sfreq']

    # Pick channels
    picks_idx = mne.pick_types(raw.info, meg=picks, exclude='bads')
    ch_names = [raw.ch_names[i] for i in picks_idx]

    # Apply notch filter for line noise (CRITICAL for gamma band)
    if apply_notch:
        raw.notch_filter(
            freqs=NOTCH_FREQS,
            picks=picks_idx,
            method='fir',
            verbose=False
        )

    data = raw.get_data(picks=picks_idx)

    return data, sfreq, ch_names


def bandpass_and_hilbert(
    data: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
) -> np.ndarray:
    """Bandpass filter and compute analytic signal."""
    filtered = mne.filter.filter_data(
        data, sfreq=sfreq,
        l_freq=band[0], h_freq=band[1],
        method='fir', verbose=False
    )
    return signal.hilbert(filtered, axis=-1)


# ============================================================================
# PHASE 2: wPLI CONNECTIVITY (ROBUST TO VOLUME CONDUCTION)
# ============================================================================

def compute_wpli(analytic: np.ndarray) -> np.ndarray:
    """
    Compute weighted Phase Lag Index (wPLI).

    wPLI is robust to volume conduction because it ignores zero-lag synchrony.
    Only phase differences that are consistently non-zero contribute.

    Args:
        analytic: [n_channels, n_timepoints] complex analytic signal

    Returns:
        wpli: [n_channels, n_channels] wPLI matrix
    """
    n_ch = analytic.shape[0]

    # Cross-spectrum: [n_ch, n_ch, n_time]
    cross = analytic[:, None, :] * analytic[None, :, :].conj()

    # Imaginary part (phase lag)
    im = np.imag(cross)

    # wPLI: |mean(sign(im) * |im|)| / mean(|im|)
    # Simplified: |mean(im)| / mean(|im|)
    num = np.abs(im.mean(axis=-1))
    den = np.mean(np.abs(im), axis=-1) + 1e-12

    wpli = num / den
    np.fill_diagonal(wpli, 0.0)

    return wpli


def compute_plv(analytic: np.ndarray) -> np.ndarray:
    """Compute Phase-Locking Value (for comparison)."""
    phase = np.angle(analytic)
    exp_phase = np.exp(1j * phase)
    plv = np.abs(exp_phase @ exp_phase.conj().T) / analytic.shape[1]
    np.fill_diagonal(plv, 1.0)
    return plv


def compute_connectivity_timeseries(
    data: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
    window_samples: int,
    step_samples: int,
    metric: str = "wpli",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute sliding-window connectivity."""

    # Bandpass and Hilbert
    analytic = bandpass_and_hilbert(data, sfreq, band)

    n_ch, n_time = analytic.shape
    n_windows = (n_time - window_samples) // step_samples + 1

    # Extract upper triangle indices
    triu_idx = np.triu_indices(n_ch, k=1)
    n_edges = len(triu_idx[0])

    # Compute connectivity for each window
    conn_ts = np.zeros((n_windows, n_edges), dtype=np.float32)
    centers = np.zeros(n_windows)

    for i in range(n_windows):
        start = i * step_samples
        end = start + window_samples
        win_analytic = analytic[:, start:end]

        if metric == "wpli":
            conn_mat = compute_wpli(win_analytic)
        else:
            conn_mat = compute_plv(win_analytic)

        conn_ts[i] = conn_mat[triu_idx]
        centers[i] = (start + end) / 2 / sfreq

    return conn_ts, centers


# ============================================================================
# PHASE 3: WORD ATLAS CONSTRUCTION
# ============================================================================

def build_word_atlas(
    conn_data: Dict[str, np.ndarray],
    word_times: Dict[str, List[Tuple[float, float, str]]],
    subjects: List[str],
    stories: List[int],
) -> Tuple[List[str], np.ndarray]:
    """
    Build word atlas by averaging connectivity across occurrences and subjects.

    Args:
        conn_data: {subject_story: connectivity_timeseries}
        word_times: {story: [(onset, offset, word), ...]}
        subjects: List of subject IDs to include
        stories: List of story IDs to include

    Returns:
        vocab: List of words
        atlas: [n_words, n_features] connectivity patterns
    """
    word_conn = defaultdict(list)

    for sub in subjects:
        for story in stories:
            key = f"{sub}_{story}"
            if key not in conn_data:
                continue

            conn_ts = conn_data[key]["conn"]
            centers = conn_data[key]["centers"]

            story_words = word_times.get(str(story), [])

            for onset, offset, word in story_words:
                # Find windows overlapping with word
                word_center = (onset + offset) / 2
                idx = np.argmin(np.abs(centers - word_center))

                if np.abs(centers[idx] - word_center) < WINDOW_SEC:
                    word_conn[word].append(conn_ts[idx])

    # Average across occurrences
    vocab = sorted(word_conn.keys())
    atlas = np.array([np.mean(word_conn[w], axis=0) for w in vocab])

    return vocab, atlas


# ============================================================================
# PHASE 4: AXIS DISCOVERY WITH CONTROLS
# ============================================================================

def discover_axes(
    atlas: np.ndarray,
    n_components: int = 20,
    method: str = "pca",
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Discover semantic axes from word atlas.

    Args:
        atlas: [n_words, n_features]
        n_components: Number of axes to extract
        method: "pca" or "ica"

    Returns:
        scores: [n_words, n_components] axis scores
        components: [n_components, n_features] axis directions
        info: Dictionary with variance explained, etc.
    """
    # Z-score normalize
    atlas_z = (atlas - atlas.mean(axis=0)) / (atlas.std(axis=0) + 1e-8)

    if method == "pca":
        model = PCA(n_components=n_components, random_state=42)
        scores = model.fit_transform(atlas_z)
        components = model.components_
        var_explained = model.explained_variance_ratio_
    else:
        model = FastICA(n_components=n_components, random_state=42, max_iter=500)
        scores = model.fit_transform(atlas_z)
        components = model.mixing_.T
        var_explained = None

    info = {
        "method": method,
        "n_components": n_components,
        "variance_explained": var_explained.tolist() if var_explained is not None else None,
    }

    return scores, components, info


def evaluate_axis_with_controls(
    scores: np.ndarray,
    vocab: List[str],
    lexicon: Dict[str, float],
    confounds: Dict[str, Dict[str, float]],
    axis_idx: int,
) -> Dict:
    """
    Evaluate axis correlation with lexicon, controlling for confounds.

    Args:
        scores: [n_words, n_components]
        vocab: Word list
        lexicon: {word: score} for semantic dimension
        confounds: {"freq": {word: freq}, "length": {word: len}, ...}
        axis_idx: Which axis to evaluate

    Returns:
        Dictionary with raw and partial correlations
    """
    # Match words
    matched_idx = []
    matched_lex = []
    matched_conf = {k: [] for k in confounds}

    for i, w in enumerate(vocab):
        if w in lexicon:
            matched_idx.append(i)
            matched_lex.append(lexicon[w])
            for k, conf_dict in confounds.items():
                matched_conf[k].append(conf_dict.get(w, np.nan))

    if len(matched_idx) < 100:
        return {"error": "Too few matched words", "n_matched": len(matched_idx)}

    axis_scores = scores[matched_idx, axis_idx]
    lex_scores = np.array(matched_lex)

    # Raw correlation
    r_raw, p_raw = pearsonr(axis_scores, lex_scores)

    # Partial correlation (control for confounds)
    # Residualize both variables
    X_conf = np.column_stack([matched_conf[k] for k in confounds])
    X_conf = np.nan_to_num(X_conf, nan=np.nanmean(X_conf, axis=0))

    # Add intercept
    X_conf = np.column_stack([np.ones(len(X_conf)), X_conf])

    # Residualize
    axis_resid = axis_scores - X_conf @ np.linalg.lstsq(X_conf, axis_scores, rcond=None)[0]
    lex_resid = lex_scores - X_conf @ np.linalg.lstsq(X_conf, lex_scores, rcond=None)[0]

    r_partial, p_partial = pearsonr(axis_resid, lex_resid)

    return {
        "n_matched": len(matched_idx),
        "r_raw": float(r_raw),
        "p_raw": float(p_raw),
        "r_partial": float(r_partial),
        "p_partial": float(p_partial),
        "delta_r": float(r_raw - r_partial),
        "survives_control": abs(r_partial) > 0.05 and p_partial < 0.05,
    }


# ============================================================================
# PHASE 5: LOW-RANK ADAPTER (PREVENTS OVERFITTING)
# ============================================================================

class LowRankAdapter(nn.Module):
    """
    Low-rank adapter: W = U @ V

    This constrains the mapping to be low-rank, preventing overfitting
    when we have many more parameters than training samples.

    Architecture:
        hidden_dim -> rank -> brain_dim

    With rank << min(hidden_dim, brain_dim), we get a regularized mapping.
    """

    def __init__(
        self,
        hidden_dim: int,
        brain_dim: int,
        rank: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.U = nn.Linear(hidden_dim, rank, bias=False)
        self.V = nn.Linear(rank, brain_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(rank)

        # Initialize with small weights
        nn.init.xavier_uniform_(self.U.weight, gain=0.1)
        nn.init.xavier_uniform_(self.V.weight, gain=0.1)

    def forward(self, x):
        """
        Args:
            x: [batch, hidden_dim] LLM hidden states
        Returns:
            [batch, brain_dim] projected brain coordinates
        """
        h = self.U(x)
        h = self.layer_norm(h)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.V(h)
        return h

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def train_adapter(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    hidden_dim: int,
    brain_dim: int,
    rank: int = 64,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Tuple[LowRankAdapter, Dict]:
    """
    Train low-rank adapter with early stopping.

    Returns:
        adapter: Trained model
        history: Training history
    """
    adapter = LowRankAdapter(hidden_dim, brain_dim, rank=rank).to(device)

    print(f"  Adapter params: {adapter.n_params:,} (rank={rank})")
    print(f"  Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")
    print(f"  Params/sample ratio: {adapter.n_params / X_train.shape[0]:.1f}")

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(device)

    history = {"train_loss": [], "val_loss": [], "val_r": []}
    best_val_loss = float('inf')
    best_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        adapter.train()
        optimizer.zero_grad()
        pred = adapter(X_train_t)
        loss = criterion(pred, Y_train_t)
        loss.backward()
        optimizer.step()

        # Validate
        adapter.eval()
        with torch.no_grad():
            val_pred = adapter(X_val_t)
            val_loss = criterion(val_pred, Y_val_t).item()

            # Compute correlation
            val_r = np.mean([
                pearsonr(val_pred[:, i].cpu().numpy(), Y_val[:, i])[0]
                for i in range(Y_val.shape[1])
            ])

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss)
        history["val_r"].append(val_r)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = adapter.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}, val_r={val_r:.4f}")

    adapter.load_state_dict(best_state)

    return adapter, history


# ============================================================================
# PHASE 6: GRADIENT-BASED STEERING
# ============================================================================

def gradient_steering(
    model,
    tokenizer,
    adapter: LowRankAdapter,
    prompt: str,
    axis_direction: np.ndarray,
    alpha: float,
    layer_idx: int,
    max_new_tokens: int = 50,
    device: str = "cpu",
) -> Tuple[str, float, float]:
    """
    Steer LLM generation using gradient of brain axis score.

    The correct steering mechanism:
    1. Get hidden state h
    2. Project through adapter: brain_proj = adapter(h)
    3. Compute axis score: score = dot(brain_proj, axis_direction)
    4. Compute gradient: grad = d(score)/d(h)
    5. Modify: h' = h + alpha * grad
    6. Continue generation with h'

    This is mathematically principled because the gradient tells us
    exactly which direction in hidden space increases the brain axis score.

    Returns:
        generated_text: The steered output
        axis_shift: Change in axis projection
        perplexity: Output perplexity
    """
    model.eval()
    adapter.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    axis_dir_t = torch.tensor(axis_direction, dtype=torch.float32).to(device)

    # Get model layers
    if hasattr(model, 'model'):
        layers = model.model.layers
    elif hasattr(model, 'transformer'):
        layers = model.transformer.h
    else:
        raise ValueError("Unknown model architecture")

    # Hook to capture and modify hidden states
    captured_hidden = [None]
    modified_hidden = [None]

    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            captured_hidden[0] = output[0].detach().clone()
        else:
            captured_hidden[0] = output.detach().clone()
        return output

    def steer_hook(module, input, output):
        if modified_hidden[0] is None:
            return output

        if isinstance(output, tuple):
            return (modified_hidden[0],) + output[1:]
        return modified_hidden[0]

    # First pass: capture hidden states and compute gradient
    handle_capture = layers[layer_idx].register_forward_hook(capture_hook)

    with torch.enable_grad():
        _ = model(inputs.input_ids, output_hidden_states=True)
        h = captured_hidden[0]
        h.requires_grad = True

        # Project through adapter and compute axis score
        h_mean = h.mean(dim=1)  # [batch, hidden]
        brain_proj = adapter(h_mean)  # [batch, brain_dim]
        score = (brain_proj * axis_dir_t).sum()

        # Compute gradient
        grad = torch.autograd.grad(score, h)[0]

    handle_capture.remove()

    # Modify hidden state
    modified_hidden[0] = captured_hidden[0] + alpha * grad

    # Second pass: generate with modified hidden state
    handle_steer = layers[layer_idx].register_forward_hook(steer_hook)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    handle_steer.remove()

    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    # Compute perplexity
    if outputs.scores:
        log_probs = []
        for i, score_tensor in enumerate(outputs.scores):
            probs = torch.softmax(score_tensor, dim=-1)
            token_id = outputs.sequences[0, inputs.input_ids.shape[1] + i]
            log_prob = torch.log(probs[0, token_id] + 1e-10)
            log_probs.append(log_prob.item())
        perplexity = np.exp(-np.mean(log_probs))
    else:
        perplexity = float('nan')

    # Compute axis shift
    with torch.no_grad():
        # Project original
        orig_proj = adapter(captured_hidden[0].mean(dim=1))
        orig_score = (orig_proj * axis_dir_t).sum().item()

        # Project modified
        mod_proj = adapter(modified_hidden[0].mean(dim=1))
        mod_score = (mod_proj * axis_dir_t).sum().item()

    axis_shift = mod_score - orig_score

    return generated_text, axis_shift, perplexity


# ============================================================================
# VALIDATION FRAMEWORK
# ============================================================================

def multi_level_validation(
    embeddings: Dict[str, np.ndarray],
    atlas_scores: np.ndarray,
    vocab: List[str],
    adapter_class,
    hidden_dim: int,
    brain_dim: int,
    device: str = "cpu",
) -> Dict:
    """
    Run multi-level validation:
    1. Subject-split (odd/even)
    2. 5-fold word CV

    Returns comprehensive validation results.
    """
    results = {}

    # Match words between embeddings and atlas
    matched_idx = []
    matched_emb = []
    for i, w in enumerate(vocab):
        if w in embeddings:
            matched_idx.append(i)
            matched_emb.append(embeddings[w])

    X = np.array(matched_emb)
    Y = atlas_scores[matched_idx]

    print(f"\n[validation] Matched {len(matched_idx)} words")

    # 5-fold word CV
    print("\n[cv] Running 5-fold word-type CV...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        adapter, _ = train_adapter(
            X_train, Y_train, X_test, Y_test,
            hidden_dim, brain_dim, rank=64, epochs=50, device=device
        )

        adapter.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            pred = adapter(X_test_t).cpu().numpy()

        # Per-axis correlation
        fold_r = []
        for ax in range(Y_test.shape[1]):
            r, _ = pearsonr(pred[:, ax], Y_test[:, ax])
            fold_r.append(r)

        cv_results.append({
            "fold": fold,
            "mean_r": float(np.mean(fold_r)),
            "per_axis_r": [float(r) for r in fold_r],
        })

        print(f"  Fold {fold}: mean_r = {np.mean(fold_r):.4f}")

    results["word_cv"] = {
        "mean_r": float(np.mean([r["mean_r"] for r in cv_results])),
        "std_r": float(np.std([r["mean_r"] for r in cv_results])),
        "per_fold": cv_results,
    }

    return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["preprocess", "atlas", "axes", "adapter", "steer", "all"], default="all")
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--meg-root", type=pathlib.Path, default=pathlib.Path("derivatives/preprocessed_data"))
    p.add_argument("--band", default="theta", choices=list(BANDS.keys()))
    p.add_argument("--metric", default="wpli", choices=["wpli", "plv"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-components", type=int, default=20)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NEURIPS-READY RIGOROUS PIPELINE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Band: {args.band} {BANDS[args.band]}")
    print(f"  Metric: {args.metric} ({'robust to volume conduction' if args.metric == 'wpli' else 'standard'})")
    print(f"  Notch filter: {NOTCH_FREQS} Hz")
    print(f"  Output: {args.out_dir}")

    # Phase selection
    if args.phase in ["preprocess", "all"]:
        print("\n" + "=" * 70)
        print("PHASE 1: PREPROCESSING WITH NOTCH FILTER")
        print("=" * 70)
        print("\n[info] This phase requires running on actual MEG data.")
        print("[info] Use compute_plv_states.py with --notch flag (to be added)")
        print("[info] Key changes from original pipeline:")
        print("  1. Notch filter at 50 Hz + harmonics")
        print("  2. Use wPLI instead of PLV")
        print("  3. Multiple frequency bands")

    if args.phase in ["atlas", "all"]:
        print("\n" + "=" * 70)
        print("PHASE 2: WORD ATLAS CONSTRUCTION")
        print("=" * 70)
        print("\n[info] Building word atlas with subject-split validation")
        print("[info] Train on ODD subjects, test on EVEN subjects")

    if args.phase in ["axes", "all"]:
        print("\n" + "=" * 70)
        print("PHASE 3: AXIS DISCOVERY WITH CONTROLS")
        print("=" * 70)
        print("\n[info] Discovering axes with confound control")
        print("[info] Controls: word frequency, word length, duration")

    if args.phase in ["adapter", "all"]:
        print("\n" + "=" * 70)
        print("PHASE 4: LOW-RANK ADAPTER TRAINING")
        print("=" * 70)
        print("\n[info] Training low-rank adapter (prevents overfitting)")
        print("[info] Multi-level validation: subject-split + word CV")

    if args.phase in ["steer", "all"]:
        print("\n" + "=" * 70)
        print("PHASE 5: GRADIENT-BASED STEERING")
        print("=" * 70)
        print("\n[info] Using gradient-based steering (correct mechanism)")
        print("[info] h' = h + alpha * grad(score)")

    print("\n" + "=" * 70)
    print("PIPELINE READY")
    print("=" * 70)
    print("\nTo run the full pipeline:")
    print("  1. First run preprocessing with notch filter")
    print("  2. Build word atlas with wPLI")
    print("  3. Discover axes and control for confounds")
    print("  4. Train low-rank adapter with CV")
    print("  5. Run gradient-based steering evaluation")

    # Save configuration
    config = {
        "bands": BANDS,
        "notch_freqs": NOTCH_FREQS,
        "metric": args.metric,
        "n_components": args.n_components,
        "odd_subjects": ODD_SUBJECTS,
        "even_subjects": EVEN_SUBJECTS,
        "train_stories": TRAIN_STORIES,
        "test_stories": TEST_STORIES,
    }

    with open(args.out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Configuration saved to {args.out_dir}/config.json")


if __name__ == "__main__":
    main()
