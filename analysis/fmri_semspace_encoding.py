"""
Build an fMRI semantic space via encoding (ridge) on semantic features.

Uses precomputed fMRI reduced coordinates (e.g., shared PCA) as targets,
fits ridge on semantic features (embedding_change/logfreq/pos_id), then
outputs predicted fMRI coords per run.

Outputs:
- per run: *_enc_semspace.npy + *_times_fmri_s.npy
- per subject: *_enc.json metadata
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import Ridge

from analysis.common import FMRIS_SHIFT_SEC, git_hash


def gamma_pdf(t: np.ndarray, shape: float, scale: float) -> np.ndarray:
    return (t ** (shape - 1)) * np.exp(-t / scale) / (scale ** shape * math.gamma(shape))


def spm_hrf(dt: float, length: float = 32.0, peak: float = 6.0, undershoot: float = 16.0, ratio: float = 1 / 6) -> np.ndarray:
    t = np.arange(0, length, dt, dtype=np.float32)
    peak_pdf = gamma_pdf(t, peak, 1.0)
    undershoot_pdf = gamma_pdf(t, undershoot, 1.0)
    hrf = peak_pdf - ratio * undershoot_pdf
    if hrf.sum() != 0:
        hrf = hrf / hrf.sum()
    return hrf.astype(np.float32)


def apply_hrf(vals: np.ndarray, times: np.ndarray, length: float, peak: float, undershoot: float, ratio: float) -> np.ndarray:
    dt = float(np.median(np.diff(times)))
    if dt <= 0:
        return vals
    hrf = spm_hrf(dt, length=length, peak=peak, undershoot=undershoot, ratio=ratio)
    return np.convolve(vals, hrf, mode="full")[: vals.size].astype(np.float32)


def load_features(features_root: pathlib.Path, sub: str, run: str, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    base = features_root / f"sub-{sub}" / f"sub-{sub}_run-{run}_features"
    feats = np.load(base.with_suffix(".npy"))
    names = json.load((base.parent / f"{base.name}_names.json").open())
    centers = np.load(base.parent / f"{base.name}_window_centers_s.npy")
    idx = [names.index(name) for name in feature_names if name in names]
    if len(idx) != len(feature_names):
        missing = [name for name in feature_names if name not in names]
        raise ValueError(f"Missing features in {base}: {missing}")
    return feats[:, idx], centers.astype(np.float32), feature_names


def load_fmri_coords(fmri_root: pathlib.Path, sub: str, run: str, tag: str) -> Tuple[np.ndarray, np.ndarray]:
    base = fmri_root / f"sub-{sub}" / f"sub-{sub}_run-{run}_{tag}"
    coords = np.load(base.with_name(f"{base.name}_semspace.npy"))
    times = np.load(base.with_name(f"{base.name}_times_fmri_s.npy"))
    return coords, times.astype(np.float32)


def build_design_matrix(
    feats: np.ndarray,
    feat_times_audio: np.ndarray,
    fmri_times: np.ndarray,
    use_hrf: bool,
    hrf_length: float,
    hrf_peak: float,
    hrf_undershoot: float,
    hrf_ratio: float,
    zscore: bool,
) -> np.ndarray:
    feat_times_fmri = feat_times_audio + FMRIS_SHIFT_SEC
    X = np.zeros((len(fmri_times), feats.shape[1]), dtype=np.float32)
    for i in range(feats.shape[1]):
        vals = feats[:, i].astype(np.float32)
        if use_hrf:
            vals = apply_hrf(vals, feat_times_fmri, hrf_length, hrf_peak, hrf_undershoot, hrf_ratio)
        X[:, i] = np.interp(fmri_times, feat_times_fmri, vals, left=np.nan, right=np.nan)
    # drop any rows with NaNs
    valid = ~np.isnan(X).any(axis=1)
    X = X[valid]
    fmri_times = fmri_times[valid]
    if zscore:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        X = (X - mean) / std
    return X, fmri_times, valid


def main():
    p = argparse.ArgumentParser(description="Build fMRI semantic space via encoding model on semantic features.")
    p.add_argument("--features-root", type=pathlib.Path, required=True)
    p.add_argument("--fmri-root", type=pathlib.Path, required=True)
    p.add_argument("--fmri-tag", type=str, default="cifti_sharedpca")
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--train-runs", nargs="+", required=True)
    p.add_argument("--test-runs", nargs="+", required=True)
    p.add_argument("--feature-names", nargs="+", default=["embedding_change", "logfreq", "pos_id"])
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--zscore", action="store_true")
    p.add_argument("--hrf", action="store_true")
    p.add_argument("--hrf-length", type=float, default=32.0)
    p.add_argument("--hrf-peak", type=float, default=6.0)
    p.add_argument("--hrf-undershoot", type=float, default=16.0)
    p.add_argument("--hrf-ratio", type=float, default=1 / 6)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()

    if not args.execute:
        print("[dry-run] add --execute to run.")
        return

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    code_hash = git_hash(out_dir.parent if out_dir.parent.exists() else pathlib.Path("."))
    out_tag = f"{args.fmri_tag}_enc"

    for sub in args.subjects:
        X_train = []
        Y_train = []
        train_runs_used = []
        for run in args.train_runs:
            try:
                feats, feat_times, _ = load_features(args.features_root, sub, run, args.feature_names)
                fmri_vals, fmri_times = load_fmri_coords(args.fmri_root, sub, run, args.fmri_tag)
            except FileNotFoundError:
                continue
            X, _, valid = build_design_matrix(
                feats,
                feat_times,
                fmri_times,
                args.hrf,
                args.hrf_length,
                args.hrf_peak,
                args.hrf_undershoot,
                args.hrf_ratio,
                args.zscore,
            )
            fmri_vals = fmri_vals[valid]
            if X.size == 0 or fmri_vals.size == 0:
                continue
            X_train.append(X)
            Y_train.append(fmri_vals)
            train_runs_used.append(run)

        if not X_train:
            print(f"[warn] sub-{sub}: no training data.")
            continue

        X_train = np.concatenate(X_train, axis=0)
        Y_train = np.concatenate(Y_train, axis=0)
        model = Ridge(alpha=args.alpha, fit_intercept=True)
        model.fit(X_train, Y_train)

        sub_dir = out_dir / f"sub-{sub}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "subject": sub,
            "fmri_root": str(args.fmri_root),
            "fmri_tag": args.fmri_tag,
            "out_tag": out_tag,
            "feature_names": args.feature_names,
            "alpha": args.alpha,
            "zscore": bool(args.zscore),
            "hrf": bool(args.hrf),
            "hrf_length": args.hrf_length,
            "hrf_peak": args.hrf_peak,
            "hrf_undershoot": args.hrf_undershoot,
            "hrf_ratio": args.hrf_ratio,
            "train_runs": train_runs_used,
            "test_runs": args.test_runs,
            "code_git_hash": code_hash,
        }
        with (sub_dir / f"sub-{sub}_{out_tag}.json").open("w") as f:
            json.dump(meta, f, indent=2)

        all_runs = list(dict.fromkeys(train_runs_used + args.test_runs))
        for run in all_runs:
            try:
                feats, feat_times, _ = load_features(args.features_root, sub, run, args.feature_names)
                fmri_vals, fmri_times = load_fmri_coords(args.fmri_root, sub, run, args.fmri_tag)
            except FileNotFoundError:
                continue
            X, fmri_times, valid = build_design_matrix(
                feats,
                feat_times,
                fmri_times,
                args.hrf,
                args.hrf_length,
                args.hrf_peak,
                args.hrf_undershoot,
                args.hrf_ratio,
                args.zscore,
            )
            fmri_times = fmri_times.astype(np.float32)
            if X.size == 0:
                continue
            preds = model.predict(X).astype(np.float32)
            base = sub_dir / f"sub-{sub}_run-{run}_{out_tag}"
            np.save(base.with_name(f"{base.name}_semspace.npy"), preds)
            np.save(base.with_name(f"{base.name}_times_fmri_s.npy"), fmri_times)
            print(f"[saved] {base.with_name(f'{base.name}_semspace.npy')} shape={preds.shape}")


if __name__ == "__main__":
    main()
