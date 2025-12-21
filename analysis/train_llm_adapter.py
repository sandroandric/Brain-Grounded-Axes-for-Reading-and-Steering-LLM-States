"""
Train linear adapter from LLM hidden states to MEG axes (Box 5 scaffold).

Assumptions:
- MEG axis coordinates saved with window centers (audio time) via discover_axes.py --save-coords
- Hidden states per story exist under --hidden-root as story_{id}_hidden.npy (shape [T, H])
- Token timings use stimuli/time_align (subtract 10.65 s, add 0.0395 s)
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from analysis.common import FMRIS_SHIFT_SEC, MEG_AUDIO_DELAY_SEC, load_run_to_story, load_time_alignment, git_hash, normalize_run


def load_meg_coords(meg_root: pathlib.Path, feature: str, sub: str, run: str, band: str, metric: str, state: str):
    run = normalize_run(run)
    base = meg_root / "coords" / feature / f"sub-{sub}" / f"sub-{sub}_run-{run}_{band}_{metric}_{state}"
    coords = np.load(base.with_name(f"{base.name}_coords.npy"))
    centers = np.load(base.with_name(f"{base.name}_window_centers_audio_s.npy"))
    return coords, centers


def align_hidden_to_windows(
    hidden: np.ndarray,
    token_times: np.ndarray,
    window_centers: np.ndarray,
    window_length: float,
    mode: str = "mean",
) -> np.ndarray:
    """Align token-level hiddens to window centers."""
    if mode == "nearest":
        idx = np.searchsorted(token_times, window_centers, side="right") - 1
        idx = np.clip(idx, 0, len(hidden) - 1)
        return hidden[idx]

    half = window_length / 2.0
    out = np.zeros((len(window_centers), hidden.shape[1]), dtype=hidden.dtype)
    for i, center in enumerate(window_centers):
        lo = center - half
        hi = center + half
        mask = (token_times >= lo) & (token_times <= hi)
        if not np.any(mask):
            idx = np.searchsorted(token_times, center, side="right") - 1
            idx = int(np.clip(idx, 0, len(hidden) - 1))
            out[i] = hidden[idx]
        else:
            out[i] = hidden[mask].mean(axis=0)
    return out


def load_hidden(hidden_root: pathlib.Path, story_id: str) -> np.ndarray:
    path = hidden_root / f"{story_id}_hidden.npy"
    return np.load(path)


def main():
    p = argparse.ArgumentParser(description="Train linear adapter from LLM hidden states to MEG axes.")
    p.add_argument("--dataset-root", type=pathlib.Path, required=True)
    p.add_argument("--run-to-story", type=pathlib.Path, required=True)
    p.add_argument("--meg-root", type=pathlib.Path, required=True, help="discover_axes --save-coords output root")
    p.add_argument("--hidden-root", type=pathlib.Path, required=True, help="Directory with story_{id}_hidden.npy")
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--features", nargs="+", required=True, help="Axes/features to fit (must exist in meg-root/coords).")
    p.add_argument("--band", default="theta")
    p.add_argument("--metric", default="plv")
    p.add_argument("--state-space", default="edge_pca")
    p.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization")
    p.add_argument("--train-runs", nargs="+", required=True)
    p.add_argument("--test-runs", nargs="+", required=True)
    p.add_argument("--subject", required=True, help="Subject to fit on (per-subject adapter).")
    p.add_argument("--standardize", action="store_true", help="Standardize hidden states before ridge.")
    p.add_argument("--align-mode", choices=["mean", "nearest"], default="mean")
    p.add_argument("--window-length", type=float, default=2.0, help="Window length (s) for align-mode=mean.")
    args = p.parse_args()

    lut = load_run_to_story(args.run_to_story)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    code_hash = git_hash(args.dataset_root)

    def collect(split_runs: List[str]):
        H_list = []
        Z_list = []
        story_list = []
        for run in split_runs:
            run = normalize_run(run)
            story_id = lut.get((args.subject, run))
            if story_id is None:
                continue
            try:
                hidden = load_hidden(args.hidden_root, story_id)
                _, token_times, _ = load_time_alignment(args.dataset_root, story_id, level="word")
                # token_times already audio-onset with delay applied
                axes_stack = []
                centers = None
                for feat in args.features:
                    coords, centers = load_meg_coords(args.meg_root, feat, args.subject, run, args.band, args.metric, args.state_space)
                    axes_stack.append(coords)
                if centers is None:
                    continue
                Z = np.stack(axes_stack, axis=1)  # [n_win, n_feat]
                H = align_hidden_to_windows(
                    hidden,
                    token_times,
                    centers,
                    window_length=args.window_length,
                    mode=args.align_mode,
                )
                n = min(len(H), len(Z))
                H_list.append(H[:n])
                Z_list.append(Z[:n])
                story_list.append(story_id)
            except FileNotFoundError:
                continue
        if not H_list:
            return None, None, []
        return np.concatenate(H_list, axis=0), np.concatenate(Z_list, axis=0), story_list

    H_train, Z_train, train_stories = collect(args.train_runs)
    H_test, Z_test, test_stories = collect(args.test_runs)
    if H_train is None or Z_train is None:
        raise RuntimeError("No training data found for adapter.")

    scaler = None
    if args.standardize:
        scaler = StandardScaler()
        H_train = scaler.fit_transform(H_train)
        if H_test is not None:
            H_test = scaler.transform(H_test)

    model = Ridge(alpha=args.alpha, fit_intercept=True)
    model.fit(H_train, Z_train)

    train_pred = model.predict(H_train)
    train_corr = [float(np.corrcoef(train_pred[:, i], Z_train[:, i])[0, 1]) if np.std(train_pred[:, i]) and np.std(Z_train[:, i]) else 0.0 for i in range(Z_train.shape[1])]
    test_corr = None
    if H_test is not None and Z_test is not None and len(H_test):
        test_pred = model.predict(H_test)
        test_corr = [
            float(np.corrcoef(test_pred[:, i], Z_test[:, i])[0, 1]) if np.std(test_pred[:, i]) and np.std(Z_test[:, i]) else 0.0
            for i in range(Z_test.shape[1])
        ]

    W = model.coef_.astype(np.float32)  # [n_feat, hidden_dim]
    b = model.intercept_.astype(np.float32)
    np.save(out_dir / f"adapter_W_sub-{args.subject}.npy", W)
    np.save(out_dir / f"adapter_b_sub-{args.subject}.npy", b)
    if scaler is not None:
        np.save(out_dir / f"adapter_scaler_mean_sub-{args.subject}.npy", scaler.mean_.astype(np.float32))
        np.save(out_dir / f"adapter_scaler_scale_sub-{args.subject}.npy", scaler.scale_.astype(np.float32))

    sidecar = {
        "subject": args.subject,
        "features": args.features,
        "band": args.band,
        "metric": args.metric,
        "state_space": args.state_space,
        "alpha": args.alpha,
        "train_runs": args.train_runs,
        "test_runs": args.test_runs,
        "train_stories": train_stories,
        "test_stories": test_stories,
        "train_corr_per_axis": train_corr,
        "test_corr_per_axis": test_corr,
        "standardize": args.standardize,
        "code_git_hash": code_hash,
    }
    with (out_dir / f"adapter_sidecar_sub-{args.subject}.json").open("w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"[saved] adapter for sub-{args.subject} with {len(args.features)} axes")


if __name__ == "__main__":
    main()
