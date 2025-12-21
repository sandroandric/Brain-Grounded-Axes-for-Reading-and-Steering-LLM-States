"""
Anchor MEG axis trajectories to fMRI semantic space (Box 4).

Inputs:
- MEG axis coordinates per run (produced by discover_axes.py with --save-coords)
  expected at: <meg-root>/coords/<feature>/sub-XX/sub-XX_run-YY_<band>_<metric>_<state_space>_coords.npy
  with window centers in audio time: ..._window_centers_audio_s.npy
- fMRI semantic space (fmri_semspace.py) per run:
  <fmri-root>/sub-XX/sub-XX_run-YY_<tag>_semspace.npy + *_times_fmri_s.npy

Outputs:
- JSON metrics per subject/feature with best Pearson corr (over optional lags) and component ID
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Dict, List, Tuple

import numpy as np

from analysis.common import FMRIS_SHIFT_SEC, git_hash


def load_meg_coords(root: pathlib.Path, feature: str, sub: str, run: str, band: str, metric: str, state: str):
    base = root / "coords" / feature / f"sub-{sub}" / f"sub-{sub}_run-{run}_{band}_{metric}_{state}"
    coords = np.load(base.with_name(f"{base.name}_coords.npy"))
    times = np.load(base.with_name(f"{base.name}_window_centers_audio_s.npy"))
    return coords, times


def load_fmri_coords(root: pathlib.Path, sub: str, run: str, tag: str):
    base = root / f"sub-{sub}" / f"sub-{sub}_run-{run}_{tag}"
    coords = np.load(base.with_name(f"{base.name}_semspace.npy"))
    times = np.load(base.with_name(f"{base.name}_times_fmri_s.npy"))
    return coords, times


def corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


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


def apply_hrf(meg_vals: np.ndarray, meg_t: np.ndarray, length: float, peak: float, undershoot: float, ratio: float) -> np.ndarray:
    dt = float(np.median(np.diff(meg_t)))
    if dt <= 0:
        return meg_vals
    hrf = spm_hrf(dt, length=length, peak=peak, undershoot=undershoot, ratio=ratio)
    convolved = np.convolve(meg_vals, hrf, mode="full")[: meg_vals.size]
    return convolved.astype(np.float32)


def best_corr(meg_t: np.ndarray, meg_vals: np.ndarray, fmri_t: np.ndarray, fmri_vals: np.ndarray, lags: List[float]):
    """
    Interpolate MEG to fMRI times (with optional lag) and return best corr over lags/components.
    """
    best = {"corr": 0.0, "lag": 0.0, "component": 0}
    for lag in lags:
        shifted = fmri_t + lag
        interp_meg = np.interp(shifted, meg_t, meg_vals, left=np.nan, right=np.nan)
        valid = ~np.isnan(interp_meg)
        if not np.any(valid):
            continue
        meg_use = interp_meg[valid]
        fmri_use = fmri_vals[valid]
        for comp in range(fmri_use.shape[1]):
            fv = fmri_use[:, comp]
            if np.std(meg_use) == 0 or np.std(fv) == 0:
                continue
            c = float(np.corrcoef(meg_use, fv)[0, 1])
            if abs(c) > abs(best["corr"]):
                best = {"corr": c, "lag": lag, "component": comp}
    return best


def corr_component(
    meg_t: np.ndarray,
    meg_vals: np.ndarray,
    fmri_t: np.ndarray,
    fmri_vals: np.ndarray,
    lag: float,
    component: int,
) -> float | None:
    shifted = fmri_t + lag
    interp_meg = np.interp(shifted, meg_t, meg_vals, left=np.nan, right=np.nan)
    valid = ~np.isnan(interp_meg)
    if not np.any(valid):
        return None
    meg_use = interp_meg[valid]
    fmri_use = fmri_vals[valid, component]
    return corr_safe(meg_use, fmri_use)


def main():
    p = argparse.ArgumentParser(description="Anchor MEG axes to fMRI semantic space.")
    p.add_argument("--meg-root", type=pathlib.Path, required=True, help="discover_axes --save-coords output root")
    p.add_argument("--fmri-root", type=pathlib.Path, required=True, help="fmri_semspace output root")
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--band", default="theta")
    p.add_argument("--metric", default="plv")
    p.add_argument("--state-space", default="edge_pca")
    p.add_argument("--fmri-tag", default="cifti_pca", help="Prefix used in fmri_semspace outputs")
    p.add_argument("--hrf", action="store_true", help="Convolve MEG axis with canonical HRF before interpolation.")
    p.add_argument("--hrf-length", type=float, default=32.0, help="HRF kernel length in seconds.")
    p.add_argument("--hrf-peak", type=float, default=6.0, help="HRF peak parameter (seconds).")
    p.add_argument("--hrf-undershoot", type=float, default=16.0, help="HRF undershoot parameter (seconds).")
    p.add_argument("--hrf-ratio", type=float, default=1 / 6, help="HRF undershoot-to-peak ratio.")
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--runs", nargs="+", help="Runs to evaluate in per-run mode (ignored in train/test mode).")
    p.add_argument("--lags", nargs="+", type=float, default=[0.0], help="Lags in seconds applied to fMRI time")
    p.add_argument("--train-runs", nargs="+", help="Optional runs for component/lag selection.")
    p.add_argument("--test-runs", nargs="+", help="Optional runs for evaluation (requires --train-runs).")
    args = p.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    code_hash = git_hash(out_dir.parent if out_dir.parent.exists() else pathlib.Path("."))

    use_split = bool(args.train_runs and args.test_runs)
    if (args.train_runs and not args.test_runs) or (args.test_runs and not args.train_runs):
        raise SystemExit("Provide both --train-runs and --test-runs, or neither.")
    if not use_split and not args.runs:
        raise SystemExit("Per-run mode requires --runs. For train/test mode, provide --train-runs and --test-runs.")
    summary: Dict[str, Dict] = {}
    for sub in args.subjects:
        summary[sub] = {}
        for feat in args.features:
            if use_split:
                train_scores: Dict[str, float] = {}
                test_scores: Dict[str, float] = {}
                comp_lag_scores: Dict[Tuple[int, float], List[float]] = {}
                n_components = None
                for run in args.train_runs:
                    try:
                        meg_vals, meg_t_audio = load_meg_coords(args.meg_root, feat, sub, run, args.band, args.metric, args.state_space)
                        fmri_vals, fmri_t = load_fmri_coords(args.fmri_root, sub, run, args.fmri_tag)
                    except FileNotFoundError:
                        continue
                    if args.hrf:
                        meg_vals = apply_hrf(meg_vals, meg_t_audio, args.hrf_length, args.hrf_peak, args.hrf_undershoot, args.hrf_ratio)
                    meg_t_fmri = meg_t_audio + FMRIS_SHIFT_SEC
                    if n_components is None:
                        n_components = fmri_vals.shape[1]
                    for comp in range(n_components):
                        for lag in args.lags:
                            c = corr_component(meg_t_fmri, meg_vals, fmri_t, fmri_vals, lag, comp)
                            if c is None:
                                continue
                            comp_lag_scores.setdefault((comp, lag), []).append(c)
                if not comp_lag_scores:
                    summary[sub][feat] = {"train": {}, "test": {}, "selected": None}
                    continue
                best_key = None
                best_mean = None
                for key, vals in comp_lag_scores.items():
                    mean_c = float(np.mean(vals))
                    if best_mean is None or abs(mean_c) > abs(best_mean):
                        best_mean = mean_c
                        best_key = key
                sel_comp, sel_lag = best_key
                # Evaluate train/test with selected component + lag
                for run in args.train_runs:
                    try:
                        meg_vals, meg_t_audio = load_meg_coords(args.meg_root, feat, sub, run, args.band, args.metric, args.state_space)
                        fmri_vals, fmri_t = load_fmri_coords(args.fmri_root, sub, run, args.fmri_tag)
                    except FileNotFoundError:
                        continue
                    if args.hrf:
                        meg_vals = apply_hrf(meg_vals, meg_t_audio, args.hrf_length, args.hrf_peak, args.hrf_undershoot, args.hrf_ratio)
                    meg_t_fmri = meg_t_audio + FMRIS_SHIFT_SEC
                    c = corr_component(meg_t_fmri, meg_vals, fmri_t, fmri_vals, sel_lag, sel_comp)
                    if c is not None:
                        train_scores[run] = float(c)
                for run in args.test_runs:
                    try:
                        meg_vals, meg_t_audio = load_meg_coords(args.meg_root, feat, sub, run, args.band, args.metric, args.state_space)
                        fmri_vals, fmri_t = load_fmri_coords(args.fmri_root, sub, run, args.fmri_tag)
                    except FileNotFoundError:
                        continue
                    if args.hrf:
                        meg_vals = apply_hrf(meg_vals, meg_t_audio, args.hrf_length, args.hrf_peak, args.hrf_undershoot, args.hrf_ratio)
                    meg_t_fmri = meg_t_audio + FMRIS_SHIFT_SEC
                    c = corr_component(meg_t_fmri, meg_vals, fmri_t, fmri_vals, sel_lag, sel_comp)
                    if c is not None:
                        test_scores[run] = float(c)
                summary[sub][feat] = {
                    "selected": {"component": sel_comp, "lag": sel_lag, "train_mean_corr": best_mean},
                    "train": train_scores,
                    "test": test_scores,
                }
            else:
                per_run = {}
                for run in args.runs:
                    try:
                        meg_vals, meg_t_audio = load_meg_coords(args.meg_root, feat, sub, run, args.band, args.metric, args.state_space)
                        fmri_vals, fmri_t = load_fmri_coords(args.fmri_root, sub, run, args.fmri_tag)
                    except FileNotFoundError:
                        continue
                    if args.hrf:
                        meg_vals = apply_hrf(meg_vals, meg_t_audio, args.hrf_length, args.hrf_peak, args.hrf_undershoot, args.hrf_ratio)
                    # convert MEG times (audio) to fMRI scan time
                    meg_t_fmri = meg_t_audio + FMRIS_SHIFT_SEC
                    best = best_corr(meg_t_fmri, meg_vals, fmri_t, fmri_vals, args.lags)
                    per_run[run] = best
                summary[sub][feat] = per_run

    mode = "train_test" if use_split else "per_run"
    out_json = out_dir / f"anchoring_metrics_{args.band}_{args.metric}_{args.state_space}_{mode}.json"
    payload = {
        "features": args.features,
        "band": args.band,
        "metric": args.metric,
        "state_space": args.state_space,
        "lags": args.lags,
        "fmri_tag": args.fmri_tag,
        "hrf": args.hrf,
        "hrf_length": args.hrf_length,
        "hrf_peak": args.hrf_peak,
        "hrf_undershoot": args.hrf_undershoot,
        "hrf_ratio": args.hrf_ratio,
        "mode": mode,
        "train_runs": args.train_runs,
        "test_runs": args.test_runs,
        "summary": summary,
        "code_git_hash": code_hash,
    }
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[saved] {out_json}")


if __name__ == "__main__":
    main()
