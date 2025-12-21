"""
Permutation test for MEG↔fMRI anchoring (train/test mode).

Uses selected component + lag per subject/feature (from anchor_meg_fmri.py output),
then computes a circular-shift null distribution to assess significance of mean |corr|.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Dict, List, Tuple

import numpy as np

from analysis.common import FMRIS_SHIFT_SEC


def corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


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


def align_series(
    meg_vals: np.ndarray,
    meg_t_audio: np.ndarray,
    fmri_vals: np.ndarray,
    fmri_t: np.ndarray,
    lag: float,
    component: int,
):
    meg_t_fmri = meg_t_audio + FMRIS_SHIFT_SEC
    shifted = fmri_t + lag
    interp_meg = np.interp(shifted, meg_t_fmri, meg_vals, left=np.nan, right=np.nan)
    valid = ~np.isnan(interp_meg)
    if not np.any(valid):
        return None, None, None
    meg_use = interp_meg[valid]
    fmri_use = fmri_vals[valid, component]
    if meg_use.size < 3:
        return None, None, None
    tr = float(np.median(np.diff(fmri_t)))
    return meg_use.astype(np.float32), fmri_use.astype(np.float32), tr


def circular_shift(vec: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return vec
    return np.roll(vec, shift)


def permute_mean_abs_corr(
    pairs: List[Tuple[np.ndarray, np.ndarray, int]],
    n_perms: int,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.zeros(n_perms, dtype=np.float32)
    for i in range(n_perms):
        vals = []
        for meg_use, fmri_use, min_shift in pairs:
            n = len(meg_use)
            if n < 3:
                continue
            if min_shift >= n:
                shift = rng.integers(1, n)
            else:
                shift = rng.integers(min_shift, n - min_shift)
            shifted = circular_shift(meg_use, int(shift))
            vals.append(abs(corr_safe(shifted, fmri_use)))
        out[i] = float(np.mean(vals)) if vals else 0.0
    return out


def main():
    p = argparse.ArgumentParser(description="Permutation test for MEG↔fMRI anchoring.")
    p.add_argument("--anchor-json", type=pathlib.Path, required=True)
    p.add_argument("--meg-root", type=pathlib.Path, required=True)
    p.add_argument("--fmri-root", type=pathlib.Path, required=True)
    p.add_argument("--n-perms", type=int, default=1000)
    p.add_argument("--min-shift-s", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    args = p.parse_args()

    anchor = json.loads(args.anchor_json.read_text())
    if anchor.get("mode") != "train_test":
        raise SystemExit("Anchor JSON must be train_test mode (with selected component/lag).")
    band = anchor["band"]
    metric = anchor["metric"]
    state = anchor["state_space"]
    fmri_tag = anchor["fmri_tag"]
    features = anchor["features"]
    summary = anchor["summary"]

    rng = np.random.default_rng(args.seed)
    results: Dict[str, Dict] = {"global": {}, "per_subject": {}}

    for sub, feats in summary.items():
        results["per_subject"][sub] = {}
        for feat in features:
            rec = feats.get(feat, {})
            selected = rec.get("selected")
            test_runs = rec.get("test", {})
            if not selected or not test_runs:
                continue
            comp = int(selected["component"])
            lag = float(selected["lag"])

            pairs = []
            obs_vals = []
            for run in test_runs.keys():
                try:
                    meg_vals, meg_t = load_meg_coords(args.meg_root, feat, sub, run, band, metric, state)
                    fmri_vals, fmri_t = load_fmri_coords(args.fmri_root, sub, run, fmri_tag)
                except FileNotFoundError:
                    continue
                aligned = align_series(meg_vals, meg_t, fmri_vals, fmri_t, lag, comp)
                if aligned[0] is None:
                    continue
                meg_use, fmri_use, tr = aligned
                min_shift = max(1, int(math.ceil(args.min_shift_s / tr)))
                pairs.append((meg_use, fmri_use, min_shift))
                obs_vals.append(abs(corr_safe(meg_use, fmri_use)))

            if not pairs:
                continue
            obs_mean = float(np.mean(obs_vals))
            null = permute_mean_abs_corr(pairs, args.n_perms, rng)
            pval = float((1 + np.sum(null >= obs_mean)) / (1 + len(null)))
            results["per_subject"][sub][feat] = {
                "observed_mean_abs": obs_mean,
                "null_mean_abs": float(np.mean(null)),
                "null_std": float(np.std(null)),
                "p_value": pval,
                "n_runs": len(obs_vals),
            }

    for feat in features:
        pairs_all = []
        obs_vals_all = []
        for sub, feats in summary.items():
            rec = feats.get(feat, {})
            selected = rec.get("selected")
            test_runs = rec.get("test", {})
            if not selected or not test_runs:
                continue
            comp = int(selected["component"])
            lag = float(selected["lag"])
            for run in test_runs.keys():
                try:
                    meg_vals, meg_t = load_meg_coords(args.meg_root, feat, sub, run, band, metric, state)
                    fmri_vals, fmri_t = load_fmri_coords(args.fmri_root, sub, run, fmri_tag)
                except FileNotFoundError:
                    continue
                aligned = align_series(meg_vals, meg_t, fmri_vals, fmri_t, lag, comp)
                if aligned[0] is None:
                    continue
                meg_use, fmri_use, tr = aligned
                min_shift = max(1, int(math.ceil(args.min_shift_s / tr)))
                pairs_all.append((meg_use, fmri_use, min_shift))
                obs_vals_all.append(abs(corr_safe(meg_use, fmri_use)))

        if not pairs_all:
            continue
        obs_mean = float(np.mean(obs_vals_all))
        null = permute_mean_abs_corr(pairs_all, args.n_perms, rng)
        pval = float((1 + np.sum(null >= obs_mean)) / (1 + len(null)))
        results["global"][feat] = {
            "observed_mean_abs": obs_mean,
            "null_mean_abs": float(np.mean(null)),
            "null_std": float(np.std(null)),
            "p_value": pval,
            "n_runs": len(obs_vals_all),
        }

    payload = {
        "anchor_json": str(args.anchor_json),
        "n_perms": args.n_perms,
        "min_shift_s": args.min_shift_s,
        "seed": args.seed,
        "results": results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
