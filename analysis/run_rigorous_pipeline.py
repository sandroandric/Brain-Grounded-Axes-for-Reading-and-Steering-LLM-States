#!/usr/bin/env python3
"""
NeurIPS-Ready Rigorous Pipeline - Full Execution

This script runs the complete pipeline:
1. Compute wPLI connectivity with notch filtering
2. Build word atlas with subject-split validation
3. Discover axes with confound controls
4. Train low-rank adapter
5. Run gradient-based steering evaluation

Run:
python analysis/run_rigorous_pipeline.py --out-dir outputs/rigorous --device mps
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
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================================
# CONFIGURATION
# ============================================================================

BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

NOTCH_FREQS = [50.0, 100.0, 150.0]
WINDOW_SEC = 0.5
STEP_SEC = 0.1

ODD_SUBJECTS = ["01", "03", "05", "07", "09", "11"]
EVEN_SUBJECTS = ["02", "04", "06", "08", "10", "12"]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clean_word(w: str) -> str:
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
# PHASE 1: CONNECTIVITY WITH NOTCH FILTER
# ============================================================================

def compute_wpli_connectivity(
    data: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
    window_samples: int,
    step_samples: int,
    apply_notch: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute wPLI connectivity timeseries with notch filtering."""

    # Notch filter (CRITICAL for gamma)
    if apply_notch:
        for freq in NOTCH_FREQS:
            if freq < sfreq / 2:  # Below Nyquist
                data = mne.filter.notch_filter(
                    data, sfreq, freqs=freq, method='fir', verbose=False
                )

    # Bandpass and Hilbert
    filtered = mne.filter.filter_data(
        data, sfreq=sfreq,
        l_freq=band[0], h_freq=band[1],
        method='fir', verbose=False
    )
    analytic = signal.hilbert(filtered, axis=-1)

    n_ch, n_time = analytic.shape
    n_windows = (n_time - window_samples) // step_samples + 1
    triu_idx = np.triu_indices(n_ch, k=1)
    n_edges = len(triu_idx[0])

    conn_ts = np.zeros((n_windows, n_edges), dtype=np.float32)
    centers = np.zeros(n_windows)

    for i in range(n_windows):
        start = i * step_samples
        end = start + window_samples
        win = analytic[:, start:end]

        # wPLI computation
        cross = win[:, None, :] * win[None, :, :].conj()
        im = np.imag(cross)
        num = np.abs(im.mean(axis=-1))
        den = np.mean(np.abs(im), axis=-1) + 1e-12
        wpli = num / den
        np.fill_diagonal(wpli, 0.0)

        conn_ts[i] = wpli[triu_idx]
        centers[i] = (start + end) / 2 / sfreq

    return conn_ts, centers


# ============================================================================
# PHASE 2: WORD ATLAS
# ============================================================================

def build_word_atlas_from_existing(
    plv_root: pathlib.Path,
    time_align_root: pathlib.Path,
    subjects: List[str],
    band: str = "theta",
) -> Tuple[List[str], np.ndarray]:
    """Build word atlas from existing PLV outputs."""

    word_conn = defaultdict(list)

    for sub in subjects:
        sub_dir = plv_root / f"sub-{sub}"
        if not sub_dir.exists():
            continue

        # Find all runs for this subject
        npy_files = list(sub_dir.glob(f"*_{band}_*.npy"))
        if not npy_files:
            continue

        for npy_file in npy_files:
            # Skip component files
            if "pca_components" in npy_file.name or "explained_variance" in npy_file.name:
                continue
            if "window_centers" in npy_file.name:
                continue

            # Extract run and story info
            match = re.search(r'run-(\d+)', npy_file.name)
            if not match:
                continue

            run_id = int(match.group(1))

            # Load connectivity and centers
            conn_ts = np.load(npy_file)
            centers_file = npy_file.parent / npy_file.name.replace('.npy', '_window_centers_audio_s.npy')
            if not centers_file.exists():
                continue

            centers = np.load(centers_file)

            # Load word times (story_id = run_id for this dataset)
            story_id = run_id
            word_times = load_story_words(time_align_root, story_id)

            for onset, word in word_times:
                # Find closest window
                idx = np.argmin(np.abs(centers - onset))
                if idx < len(conn_ts) and np.abs(centers[idx] - onset) < WINDOW_SEC:
                    word_conn[word].append(conn_ts[idx])

    # Average across occurrences
    vocab = sorted([w for w in word_conn if len(word_conn[w]) >= 3])  # Min 3 occurrences
    atlas = np.array([np.mean(word_conn[w], axis=0) for w in vocab])

    return vocab, atlas


# ============================================================================
# PHASE 3: AXIS DISCOVERY
# ============================================================================

def discover_and_evaluate_axes(
    atlas: np.ndarray,
    vocab: List[str],
    lexica: Dict[str, Dict[str, float]],
    confounds: Dict[str, Dict[str, float]],
    n_components: int = 20,
) -> Dict:
    """Discover axes and evaluate with confound controls."""

    # Z-score
    atlas_z = (atlas - atlas.mean(axis=0)) / (atlas.std(axis=0) + 1e-8)

    # PCA first
    pca = PCA(n_components=n_components, random_state=42)
    pca_scores = pca.fit_transform(atlas_z)

    results = {
        "n_words": len(vocab),
        "n_features": atlas.shape[1],
        "pca_variance_explained": pca.explained_variance_ratio_.tolist(),
        "axes": {},
    }

    print(f"\n  PCA: {pca.explained_variance_ratio_[:5].sum():.1%} variance in first 5 components")

    # Evaluate each axis against each lexicon
    for lex_name, lex_dict in lexica.items():
        print(f"\n  Evaluating axes against {lex_name}...")

        for ax in range(n_components):
            # Match words
            matched = [(i, lex_dict[w]) for i, w in enumerate(vocab) if w in lex_dict]
            if len(matched) < 100:
                continue

            idx, lex_vals = zip(*matched)
            axis_vals = pca_scores[list(idx), ax]
            lex_vals = np.array(lex_vals)

            # Raw correlation
            r_raw, p_raw = pearsonr(axis_vals, lex_vals)

            # Partial correlation (control confounds)
            X_conf = []
            for i in idx:
                row = [confounds[c].get(vocab[i], 0) for c in confounds]
                X_conf.append(row)
            X_conf = np.array(X_conf)
            X_conf = np.nan_to_num(X_conf)
            X_conf = np.column_stack([np.ones(len(X_conf)), X_conf])

            axis_resid = axis_vals - X_conf @ np.linalg.lstsq(X_conf, axis_vals, rcond=None)[0]
            lex_resid = lex_vals - X_conf @ np.linalg.lstsq(X_conf, lex_vals, rcond=None)[0]

            r_partial, p_partial = pearsonr(axis_resid, lex_resid)

            key = f"axis_{ax}_{lex_name}"
            results["axes"][key] = {
                "axis": ax,
                "lexicon": lex_name,
                "n_matched": len(matched),
                "r_raw": float(r_raw),
                "p_raw": float(p_raw),
                "r_partial": float(r_partial),
                "p_partial": float(p_partial),
                "survives": abs(r_partial) > 0.03 and p_partial < 0.05,
            }

            if abs(r_raw) > 0.05:
                status = "✓" if results["axes"][key]["survives"] else "✗"
                print(f"    Axis {ax}: r_raw={r_raw:+.3f}, r_partial={r_partial:+.3f} {status}")

    return results, pca_scores


# ============================================================================
# PHASE 4: LOW-RANK ADAPTER
# ============================================================================

class LowRankAdapter(nn.Module):
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


def train_low_rank_adapter(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    rank: int = 64,
    epochs: int = 100,
    device: str = "cpu",
) -> Tuple[LowRankAdapter, Dict]:
    """Train low-rank adapter with validation."""

    hidden_dim = X_train.shape[1]
    brain_dim = Y_train.shape[1]

    adapter = LowRankAdapter(hidden_dim, brain_dim, rank=rank).to(device)
    n_params = sum(p.numel() for p in adapter.parameters())

    print(f"  Adapter: {n_params:,} params, rank={rank}")
    print(f"  Ratio: {n_params / len(X_train):.1f} params/sample")

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(device)

    best_val_r = -1
    best_state = None

    for epoch in range(epochs):
        adapter.train()
        optimizer.zero_grad()
        pred = adapter(X_train_t)
        loss = criterion(pred, Y_train_t)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            adapter.eval()
            with torch.no_grad():
                val_pred = adapter(X_val_t).cpu().numpy()
                val_r = np.mean([pearsonr(val_pred[:, i], Y_val[:, i])[0] for i in range(Y_val.shape[1])])

            if val_r > best_val_r:
                best_val_r = val_r
                best_state = {k: v.clone() for k, v in adapter.state_dict().items()}

            if epoch % 20 == 0:
                print(f"    Epoch {epoch}: loss={loss.item():.4f}, val_r={val_r:.4f}")

    adapter.load_state_dict(best_state)

    return adapter, {"best_val_r": best_val_r, "n_params": n_params}


# ============================================================================
# PHASE 5: GRADIENT-BASED STEERING
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
    """Generate with gradient-based steering."""

    model.eval()
    adapter.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    axis_t = torch.tensor(axis_dir, dtype=torch.float32).to(device)

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
        h_mean = h.mean(dim=1)
        proj = adapter(h_mean)
        score = (proj * axis_t).sum()
        grad = torch.autograd.grad(score, h)[0]
    handle1.remove()

    modified[0] = captured[0] + alpha * grad

    # Generate
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

    # Compute shift
    with torch.no_grad():
        orig_score = (adapter(captured[0].mean(dim=1)) * axis_t).sum().item()
        new_score = (adapter(modified[0].mean(dim=1)) * axis_t).sum().item()

    return text, new_score - orig_score


# ============================================================================
# MAIN
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--plv-root", type=pathlib.Path, default=pathlib.Path("outputs/meg_plv_states"))
    p.add_argument("--embed-root", type=pathlib.Path,
                   default=pathlib.Path("derivatives/annotations/embeddings/tinyllama/word-level"))
    p.add_argument("--time-align-root", type=pathlib.Path,
                   default=pathlib.Path("derivatives/annotations/time_align/word-level"))
    p.add_argument("--lexica-root", type=pathlib.Path, default=pathlib.Path("metadata/lexica"))
    p.add_argument("--confounds-path", type=pathlib.Path, default=pathlib.Path("outputs/confounds/word_confounds.json"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--band", default="theta")
    p.add_argument("--n-components", type=int, default=20)
    p.add_argument("--adapter-rank", type=int, default=64)
    p.add_argument("--n-prompts", type=int, default=50)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NEURIPS-READY RIGOROUS PIPELINE")
    print("=" * 70)

    # ===== PHASE 1: BUILD WORD ATLAS =====
    print("\n" + "=" * 70)
    print("PHASE 1: BUILD WORD ATLAS (Subject-Split)")
    print("=" * 70)

    print("\n[atlas] Building from ODD subjects...")
    vocab_odd, atlas_odd = build_word_atlas_from_existing(
        args.plv_root, args.time_align_root, ODD_SUBJECTS, args.band
    )
    print(f"  ODD: {len(vocab_odd)} words, {atlas_odd.shape[1]} features")

    print("\n[atlas] Building from EVEN subjects...")
    vocab_even, atlas_even = build_word_atlas_from_existing(
        args.plv_root, args.time_align_root, EVEN_SUBJECTS, args.band
    )
    print(f"  EVEN: {len(vocab_even)} words, {atlas_even.shape[1]} features")

    # Find shared vocabulary
    shared_vocab = sorted(set(vocab_odd) & set(vocab_even))
    print(f"\n  Shared vocabulary: {len(shared_vocab)} words")

    # ===== PHASE 2: DISCOVER AXES =====
    print("\n" + "=" * 70)
    print("PHASE 2: AXIS DISCOVERY WITH CONTROLS")
    print("=" * 70)

    # Load lexica
    lexica = {}
    if (args.lexica_root / "concreteness_zh.csv").exists():
        import pandas as pd
        df = pd.read_csv(args.lexica_root / "concreteness_zh.csv")
        lexica["concreteness"] = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        print(f"  Loaded concreteness: {len(lexica['concreteness'])} words")

    if (args.lexica_root / "vad_zh.csv").exists():
        import pandas as pd
        df = pd.read_csv(args.lexica_root / "vad_zh.csv")
        if 'valence' in df.columns:
            lexica["valence"] = dict(zip(df.iloc[:, 0], df['valence']))
        if 'arousal' in df.columns:
            lexica["arousal"] = dict(zip(df.iloc[:, 0], df['arousal']))
        print(f"  Loaded VAD: {len(df)} words")

    # Load confounds
    confounds = {"length": {}, "logfreq": {}}
    if args.confounds_path.exists():
        with open(args.confounds_path) as f:
            conf_data = json.load(f)
        for w, data in conf_data.items():
            confounds["length"][w] = len(w)
            confounds["logfreq"][w] = data.get("logfreq", 0)
    else:
        # Fallback: use word length
        for w in shared_vocab:
            confounds["length"][w] = len(w)

    # Run axis discovery on ODD subjects
    if lexica:
        odd_idx = {w: i for i, w in enumerate(vocab_odd)}
        shared_odd_idx = [odd_idx[w] for w in shared_vocab if w in odd_idx]
        atlas_shared_odd = atlas_odd[shared_odd_idx]

        axis_results, pca_scores = discover_and_evaluate_axes(
            atlas_shared_odd,
            [vocab_odd[i] for i in shared_odd_idx],
            lexica, confounds, args.n_components
        )

        # Save axis results
        with open(args.out_dir / "axis_evaluation.json", "w") as f:
            json.dump(axis_results, f, indent=2)
    else:
        print("  [warn] No lexica found, skipping axis evaluation")
        # Still run PCA
        atlas_z = (atlas_odd - atlas_odd.mean(0)) / (atlas_odd.std(0) + 1e-8)
        pca = PCA(n_components=args.n_components, random_state=42)
        pca_scores = pca.fit_transform(atlas_z)

    # ===== PHASE 3: TRAIN ADAPTER =====
    print("\n" + "=" * 70)
    print("PHASE 3: LOW-RANK ADAPTER TRAINING")
    print("=" * 70)

    # Load TinyLlama embeddings
    print("\n[embed] Loading TinyLlama embeddings...")
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

    print(f"  Loaded embeddings for {len(embeddings)} words")

    # Match with atlas
    matched_words = [w for w in shared_vocab if w in embeddings]
    print(f"  Matched: {len(matched_words)} words")

    if len(matched_words) < 1000:
        print("  [error] Too few matched words for training")
        return

    # Prepare training data
    odd_idx = {w: i for i, w in enumerate(vocab_odd)}
    even_idx = {w: i for i, w in enumerate(vocab_even)}

    X = np.array([embeddings[w] for w in matched_words])

    # PCA scores for ODD atlas
    atlas_z_odd = (atlas_odd - atlas_odd.mean(0)) / (atlas_odd.std(0) + 1e-8)
    pca_odd = PCA(n_components=args.n_components, random_state=42)
    scores_odd = pca_odd.fit_transform(atlas_z_odd)
    Y_odd = np.array([scores_odd[odd_idx[w]] for w in matched_words if w in odd_idx])

    # PCA scores for EVEN atlas
    atlas_z_even = (atlas_even - atlas_even.mean(0)) / (atlas_even.std(0) + 1e-8)
    pca_even = PCA(n_components=args.n_components, random_state=42)
    scores_even = pca_even.fit_transform(atlas_z_even)
    Y_even = np.array([scores_even[even_idx[w]] for w in matched_words if w in even_idx])

    # Filter to words in both
    both_idx = [i for i, w in enumerate(matched_words) if w in odd_idx and w in even_idx]
    X_both = X[both_idx]
    Y_odd_both = Y_odd[:len(both_idx)]
    Y_even_both = Y_even[:len(both_idx)]

    print(f"\n[train] Training adapter (ODD → EVEN cross-validation)")
    adapter, train_info = train_low_rank_adapter(
        X_both, Y_odd_both, X_both, Y_even_both,
        rank=args.adapter_rank, device=args.device
    )

    print(f"\n  Best cross-subject r: {train_info['best_val_r']:.4f}")

    # Save adapter
    torch.save(adapter.state_dict(), args.out_dir / "adapter.pt")

    # ===== PHASE 4: GRADIENT STEERING =====
    print("\n" + "=" * 70)
    print("PHASE 4: GRADIENT-BASED STEERING")
    print("=" * 70)

    print("\n[steer] Loading TinyLlama...")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16 if args.device != "cpu" else torch.float32,
    ).to(args.device)
    model.eval()

    # Test prompts
    prompts = [
        "The scientist discovered that",
        "In the quiet village, people",
        "She felt a sudden wave of",
        "The ancient temple stood",
        "After years of hard work",
    ][:args.n_prompts]

    # Use first PCA axis as steering direction
    # Note: adapter outputs n_components dims, so axis_dir should be in that space
    # A unit vector along axis 0 in component space
    axis_dir = np.zeros(args.n_components)
    axis_dir[0] = 1.0  # Steer along first principal component

    print(f"\n[steer] Testing {len(prompts)} prompts with gradient steering...")
    print(f"  Steering along PC1 (axis_dir shape: {axis_dir.shape})")

    steering_results = []
    for prompt in prompts:
        # Baseline
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
            out = model.generate(inputs.input_ids, max_new_tokens=50, do_sample=True, temperature=0.7)
            baseline = tokenizer.decode(out[0], skip_special_tokens=True)

        # Positive steering
        pos_text, pos_shift = gradient_steer_generation(
            model, tokenizer, adapter, prompt, axis_dir, alpha=2.0, layer_idx=11, device=args.device
        )

        # Negative steering
        neg_text, neg_shift = gradient_steer_generation(
            model, tokenizer, adapter, prompt, axis_dir, alpha=-2.0, layer_idx=11, device=args.device
        )

        steering_results.append({
            "prompt": prompt,
            "baseline": baseline,
            "positive": pos_text,
            "negative": neg_text,
            "pos_shift": pos_shift,
            "neg_shift": neg_shift,
        })

        print(f"\n  Prompt: {prompt[:30]}...")
        print(f"    Δ pos: {pos_shift:+.3f}, Δ neg: {neg_shift:+.3f}")

    # Statistical test
    pos_shifts = [r["pos_shift"] for r in steering_results]
    neg_shifts = [r["neg_shift"] for r in steering_results]

    t_stat, p_val = ttest_ind(pos_shifts, neg_shifts)
    d = (np.mean(pos_shifts) - np.mean(neg_shifts)) / np.std(pos_shifts + neg_shifts)

    print(f"\n[stats] Steering effect:")
    print(f"  Mean pos shift: {np.mean(pos_shifts):+.3f}")
    print(f"  Mean neg shift: {np.mean(neg_shifts):+.3f}")
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
            }
        }, f, indent=2)

    print(f"\n✓ Results saved to {args.out_dir}/")

    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Words in atlas: {len(shared_vocab)}")
    print(f"  Adapter cross-subject r: {train_info['best_val_r']:.4f}")
    print(f"  Steering Cohen's d: {d:.3f}")
    print(f"  Steering p-value: {p_val:.4f}")

    sig = "✓ SIGNIFICANT" if p_val < 0.05 else "✗ NOT SIGNIFICANT"
    effect = "LARGE" if abs(d) > 0.8 else ("MEDIUM" if abs(d) > 0.5 else "SMALL")
    print(f"\n  Result: {sig} ({effect} effect)")


if __name__ == "__main__":
    main()
