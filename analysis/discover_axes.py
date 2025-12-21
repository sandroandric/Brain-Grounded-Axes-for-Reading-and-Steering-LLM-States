"""
Axis discovery from PLV states + semantic features.

Given:
- PLV/wPLI states per run (from compute_plv_states.py)
- Window-aligned semantic features (from compute_semantic_features.py)

For each subject and feature:
- Train axis on train runs (top/bottom quantiles of feature)
- Orient sign so corr(z, feature) > 0 on train
- Evaluate corr on train and test runs
- Optionally save per-run projections
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Tuple

import numpy as np

from analysis.common import git_hash, normalize_run


def load_states(base_dir: pathlib.Path, sub: str, run: str, band: str, metric: str, state_space: str):
    run = normalize_run(run)
    base = base_dir / f"sub-{sub}" / f"sub-{sub}_run-{run}_{band}_{metric}_{state_space}"
    states = np.load(base.with_suffix(".npy"))
    centers = np.load(base.parent / f"{base.name}_window_centers_audio_s.npy")
    sidecar_path = base.with_suffix(".json")
    sidecar = json.load(sidecar_path.open()) if sidecar_path.exists() else {}
    return states, centers, sidecar


def load_features(feat_dir: pathlib.Path, sub: str, run: str):
    run = normalize_run(run)
    base = feat_dir / f"sub-{sub}" / f"sub-{sub}_run-{run}_features"
    feats = np.load(base.with_suffix(".npy"))
    names = json.load((base.parent / f"{base.name}_names.json").open())
    centers = np.load(base.parent / f"{base.name}_window_centers_s.npy")
    return feats, names, centers


def align_lengths(state_arr: np.ndarray, feat_arr: np.ndarray):
    n = min(len(state_arr), len(feat_arr))
    return state_arr[:n], feat_arr[:n]


def derive_axis(states: np.ndarray, feature: np.ndarray, quantile: float = 0.2):
    low_th = np.quantile(feature, quantile)
    hi_th = np.quantile(feature, 1 - quantile)
    lo_idx = feature <= low_th
    hi_idx = feature >= hi_th
    axis = states[hi_idx].mean(axis=0) - states[lo_idx].mean(axis=0)
    return axis


def project(states: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return states @ axis


def corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def runs_list(default_start: int, default_end: int) -> List[str]:
    return [str(i) for i in range(default_start, default_end + 1)]


def main():
    p = argparse.ArgumentParser(description="Discover semantic axes from PLV states + features.")
    p.add_argument("--plv-root", type=pathlib.Path, required=True, help="Output root from compute_plv_states.py")
    p.add_argument("--features-root", type=pathlib.Path, required=True, help="Output root from compute_semantic_features.py")
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--band", default="theta")
    p.add_argument("--metric", default="plv")
    p.add_argument("--state-space", default="edge_pca")
    p.add_argument("--feature-names", nargs="+", required=True, help="Features to build axes for (must be in features file).")
    p.add_argument("--train-runs", nargs="+", default=runs_list(1, 40))
    p.add_argument("--test-runs", nargs="+", default=runs_list(41, 60))
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--quantile", type=float, default=0.2)
    p.add_argument("--save-coords", action="store_true", help="Save per-run projections.")
    args = p.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    code_hash = git_hash(out_dir.parent if out_dir.parent.exists() else pathlib.Path("."))

    for sub in args.subjects:
        axes_by_feature: Dict[str, np.ndarray] = {}
        metrics = {}
        for feat_name in args.feature_names:
            # Collect train data
            train_states = []
            train_feat = []
            for run in args.train_runs:
                try:
                    states, centers_s, sidecar = load_states(args.plv_root, sub, run, args.band, args.metric, args.state_space)
                    feats, names, feat_centers = load_features(args.features_root, sub, run)
                except FileNotFoundError:
                    continue
                if feat_name not in names:
                    continue
                feat_idx = names.index(feat_name)
                states, feats = align_lengths(states, feats)
                train_states.append(states)
                train_feat.append(feats[:, feat_idx])
            if not train_states:
                print(f"[warn] no train data for sub-{sub} feature {feat_name}")
                continue
            train_states_arr = np.concatenate(train_states, axis=0)
            train_feat_arr = np.concatenate(train_feat, axis=0)

            axis = derive_axis(train_states_arr, train_feat_arr, quantile=args.quantile)
            # orient sign
            sign = np.sign(corr_safe(project(train_states_arr, axis), train_feat_arr))
            if sign == 0:
                sign = 1.0
            axis = axis * sign
            axes_by_feature[feat_name] = axis.astype(np.float32)

            # Eval on train/test
            metrics[feat_name] = {"train_runs": {}, "test_runs": {}}
            metrics[feat_name]["train_corr"] = corr_safe(project(train_states_arr, axis), train_feat_arr)

            for split_name, runs in [("train_runs", args.train_runs), ("test_runs", args.test_runs)]:
                for run in runs:
                    try:
                        states, centers, _ = load_states(args.plv_root, sub, run, args.band, args.metric, args.state_space)
                        feats, names, _ = load_features(args.features_root, sub, run)
                    except FileNotFoundError:
                        continue
                    if feat_name not in names:
                        continue
                    feat_idx = names.index(feat_name)
                    states, feats = align_lengths(states, feats)
                    coords = project(states, axis)
                    c = corr_safe(coords, feats[:, feat_idx])
                    metrics[feat_name][split_name][run] = c
                    if args.save_coords:
                        coord_dir = out_dir / "coords" / feat_name / f"sub-{sub}"
                        coord_dir.mkdir(parents=True, exist_ok=True)
                        base = coord_dir / f"sub-{sub}_run-{run}_{args.band}_{args.metric}_{args.state_space}"
                        np.save(base.parent / f"{base.name}_coords.npy", coords.astype(np.float32))
                        np.save(base.parent / f"{base.name}_window_centers_audio_s.npy", centers[: len(coords)].astype(np.float32))
                print(f"[axis] sub-{sub} feature={feat_name} train_corr={metrics[feat_name]['train_corr']:.4f}")

        # save axes per subject
        for feat_name, axis in axes_by_feature.items():
            feat_dir = out_dir / feat_name
            feat_dir.mkdir(parents=True, exist_ok=True)
            axis_path = feat_dir / f"sub-{sub}_{args.band}_{args.metric}_{args.state_space}_axis.npy"
            np.save(axis_path, axis)
            sidecar = {
                "subject": sub,
                "feature": feat_name,
                "band": args.band,
                "metric": args.metric,
                "state_space": args.state_space,
                "quantile": args.quantile,
                "train_runs": args.train_runs,
                "test_runs": args.test_runs,
                "axis_shape": axis.shape,
                "code_git_hash": code_hash,
            }
            with axis_path.with_suffix(".json").open("w") as f:
                json.dump(sidecar, f, indent=2)
        # save metrics
        metrics_path = out_dir / f"metrics_sub-{sub}_{args.band}_{args.metric}_{args.state_space}.json"
        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
