"""
Build a shared fMRI semantic space via SRM on reduced per-subject features.

This uses per-subject PCA features (e.g., outputs/fmri_semspace_shared) as input,
then learns SRM weights per subject across training runs to align into a common space.

Outputs per subject/run:
- *_semspace.npy (time x k)
- *_times_fmri_s.npy
- per-subject weights: sub-XX_*_weights.npy
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Tuple

import numpy as np

from analysis.common import git_hash


def zscore_time(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (x - mean) / std


def load_run(input_root: pathlib.Path, sub: str, run: str, tag: str):
    base = input_root / f"sub-{sub}" / f"sub-{sub}_run-{run}_{tag}"
    data = np.load(base.with_name(f"{base.name}_semspace.npy"))
    times = np.load(base.with_name(f"{base.name}_times_fmri_s.npy"))
    return data, times


def orthonormalize(w: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(w, full_matrices=False)
    return u @ vt


def fit_srm(Xs: List[np.ndarray], k: int, n_iter: int, eps: float = 1e-6) -> List[np.ndarray]:
    """
    Xs: list of subject matrices (v x t)
    Returns list of W_i (v x k) orthonormal weights.
    """
    Ws = []
    for X in Xs:
        u, _, _ = np.linalg.svd(X, full_matrices=False)
        Ws.append(u[:, :k])

    for _ in range(n_iter):
        S = np.mean([W.T @ X for W, X in zip(Ws, Xs)], axis=0)  # (k x t)
        M = S @ S.T + eps * np.eye(k)
        invM = np.linalg.inv(M)
        new_Ws = []
        for X in Xs:
            W = X @ S.T @ invM
            new_Ws.append(orthonormalize(W))
        Ws = new_Ws
    return Ws


def main():
    p = argparse.ArgumentParser(description="SRM on reduced fMRI features (shared PCA inputs).")
    p.add_argument("--input-root", type=pathlib.Path, required=True, help="fmri_semspace_shared output root")
    p.add_argument("--input-tag", type=str, default="cifti_sharedpca", help="Input tag prefix")
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--components", type=int, default=50, help="SRM components (k)")
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--train-runs", nargs="+", required=True)
    p.add_argument("--test-runs", nargs="+", required=True)
    p.add_argument("--n-iter", type=int, default=10)
    p.add_argument("--zscore", action="store_true", help="Z-score features across time per run.")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()

    if not args.execute:
        print("[dry-run] add --execute to run.")
        return

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    code_hash = git_hash(out_dir.parent if out_dir.parent.exists() else pathlib.Path("."))

    train_set = set(args.train_runs)
    test_set = set(args.test_runs)
    all_runs = args.train_runs + args.test_runs

    available_by_sub: Dict[str, set] = {}
    for sub in args.subjects:
        avail = set()
        for run in all_runs:
            base = args.input_root / f"sub-{sub}" / f"sub-{sub}_run-{run}_{args.input_tag}_semspace.npy"
            if base.exists():
                avail.add(run)
        available_by_sub[sub] = avail

    shared_train = set(args.train_runs)
    shared_test = set(args.test_runs)
    for sub, avail in available_by_sub.items():
        missing_train = train_set - avail
        missing_test = test_set - avail
        if missing_train or missing_test:
            print(f"[warn] sub-{sub} missing runs; dropping from SRM sets: train={sorted(missing_train)} test={sorted(missing_test)}")
        shared_train &= avail
        shared_test &= avail

    if not shared_train or not shared_test:
        raise SystemExit("No shared runs across subjects after intersection. Check inputs.")

    shared_train_runs = sorted(shared_train)
    shared_test_runs = sorted(shared_test)
    if shared_train_runs != args.train_runs or shared_test_runs != args.test_runs:
        print(f"[warn] using shared train runs: {shared_train_runs}")
        print(f"[warn] using shared test runs: {shared_test_runs}")

    valid_subjects = [sub for sub in args.subjects if shared_train.issubset(available_by_sub[sub]) and shared_test.issubset(available_by_sub[sub])]
    if not valid_subjects:
        raise SystemExit("No subjects have the shared run set.")

    # Determine min length per run across valid subjects
    run_min_len: Dict[str, int] = {}
    all_runs = shared_train_runs + shared_test_runs
    for run in all_runs:
        lengths = []
        for sub in valid_subjects:
            data, _ = load_run(args.input_root, sub, run, args.input_tag)
            lengths.append(data.shape[0])
        run_min_len[run] = min(lengths)

    # Build training matrices
    Xs = []
    for sub in valid_subjects:
        chunks = []
        for run in shared_train_runs:
            data, _ = load_run(args.input_root, sub, run, args.input_tag)
            data = data[: run_min_len[run]]
            if args.zscore:
                data = zscore_time(data)
            chunks.append(data)
        concat = np.concatenate(chunks, axis=0)  # (t x v)
        Xs.append(concat.T)  # (v x t)

    v_dim = Xs[0].shape[0]
    k = min(args.components, v_dim)
    if k < args.components:
        print(f"[warn] reducing components to {k} (input dim {v_dim}).")

    Ws = fit_srm(Xs, k=k, n_iter=args.n_iter)

    tag = f"{args.input_tag.split('_')[0]}_srm"
    # Save weights + project all runs
    for sub, W in zip(valid_subjects, Ws):
        sub_dir = out_dir / f"sub-{sub}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        w_path = sub_dir / f"sub-{sub}_{tag}_weights.npy"
        np.save(w_path, W.astype(np.float32))
        meta = {
            "subject": sub,
            "tag": tag,
            "input_root": str(args.input_root),
            "input_tag": args.input_tag,
            "components": k,
            "train_runs": args.train_runs,
            "test_runs": args.test_runs,
            "train_runs_used": shared_train_runs,
            "test_runs_used": shared_test_runs,
            "n_iter": args.n_iter,
            "zscore": bool(args.zscore),
            "code_git_hash": code_hash,
        }
        with (sub_dir / f"sub-{sub}_{tag}.json").open("w") as f:
            json.dump(meta, f, indent=2)

        for run in all_runs:
            data, times = load_run(args.input_root, sub, run, args.input_tag)
            data = data[: run_min_len[run]]
            times = times[: run_min_len[run]]
            if args.zscore:
                data = zscore_time(data)
            coords = (W.T @ data.T).T  # (t x k)
            base = sub_dir / f"sub-{sub}_run-{run}_{tag}"
            np.save(base.with_name(f"{base.name}_semspace.npy"), coords.astype(np.float32))
            np.save(base.with_name(f"{base.name}_times_fmri_s.npy"), times.astype(np.float32))
            print(f"[saved] {base.with_name(f'{base.name}_semspace.npy')} shape={coords.shape}")


if __name__ == "__main__":
    main()
