"""
Compute shared fMRI semantic space per subject (Box 2, robust).

Fits a single PCA per subject across all runs (IncrementalPCA),
then projects each run into the same component space.

Outputs:
- per run: *_sharedpca_semspace.npy + *_times_fmri_s.npy
- per subject: *_sharedpca_components.npy + JSON metadata
"""

from __future__ import annotations

import argparse
import json
import pathlib

import nibabel as nib
import numpy as np
from nibabel.cifti2 import cifti2_axes
from sklearn.decomposition import IncrementalPCA

from analysis.common import git_hash


def cifti_time_size(path: pathlib.Path) -> int:
    img = nib.load(path)
    axis0 = img.header.get_axis(0)
    axis1 = img.header.get_axis(1)
    if isinstance(axis0, cifti2_axes.SeriesAxis):
        return int(axis0.size)
    if isinstance(axis1, cifti2_axes.SeriesAxis):
        return int(axis1.size)
    raise ValueError(f"No SeriesAxis found in CIFTI header for {path}")


def load_cifti(path: pathlib.Path):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    axis0 = img.header.get_axis(0)
    axis1 = img.header.get_axis(1)
    if isinstance(axis0, cifti2_axes.SeriesAxis):
        time_axis = axis0
        if data.shape[0] != time_axis.size and data.shape[1] == time_axis.size:
            data = data.T
    elif isinstance(axis1, cifti2_axes.SeriesAxis):
        time_axis = axis1
        if data.shape[1] == time_axis.size:
            data = data.T
    else:
        raise ValueError(f"No SeriesAxis found in CIFTI header for {path}")
    times = time_axis.start + np.arange(time_axis.size) * time_axis.step
    return data, times, float(time_axis.step)


def mni_time_size(path: pathlib.Path) -> int:
    img = nib.load(path)
    shape = img.shape
    if len(shape) != 4:
        raise ValueError(f"Unexpected MNI shape {shape}")
    return int(shape[3])


def load_mni(path: pathlib.Path, mask: np.ndarray | None = None):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    tr = img.header.get_zooms()[3]
    if data.ndim != 4:
        raise ValueError(f"Unexpected MNI shape {data.shape}")
    t_axis = data.shape[-1]
    data_2d = data.reshape((-1, t_axis)).T
    if mask is not None:
        data_2d = data_2d[:, mask.reshape(-1)]
    times = np.arange(data_2d.shape[0]) * tr
    return data_2d, times, float(tr)


def main():
    p = argparse.ArgumentParser(description="Build shared fMRI semantic space via IncrementalPCA.")
    p.add_argument("--dataset-root", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--components", type=int, default=100)
    p.add_argument("--space", choices=["CIFTI", "MNI"], default="CIFTI")
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--mask", type=pathlib.Path, help="Optional brain mask for MNI space.")
    p.add_argument("--batch-size", type=int, default=None, help="Optional IPCA batch size.")
    p.add_argument("--execute", action="store_true", help="Actually compute; otherwise dry run.")
    args = p.parse_args()

    if not args.execute:
        print("[dry-run] add --execute to run.")
        return

    mask_arr = None
    if args.space == "MNI" and args.mask is not None:
        mask_arr = nib.load(args.mask).get_fdata().astype(bool)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    code_hash = git_hash(args.dataset_root)
    tag = f"{args.space.lower()}_sharedpca"

    for sub in args.subjects:
        run_paths = []
        time_sizes = []
        for run in args.runs:
            if args.space == "CIFTI":
                fmri_path = (
                    args.dataset_root
                    / "derivatives"
                    / "preprocessed_data"
                    / f"sub-{sub}"
                    / "CIFTI"
                    / f"sub-{sub}_task-RDR_run-{run}_bold.dtseries.nii"
                )
                if fmri_path.exists():
                    run_paths.append((run, fmri_path))
                    time_sizes.append(cifti_time_size(fmri_path))
            else:
                fmri_path = (
                    args.dataset_root
                    / "derivatives"
                    / "preprocessed_data"
                    / f"sub-{sub}"
                    / "MNI"
                    / f"sub-{sub}_task-RDR_run-{run}_bold.nii.gz"
                )
                if fmri_path.exists():
                    run_paths.append((run, fmri_path))
                    time_sizes.append(mni_time_size(fmri_path))
        if not run_paths:
            print(f"[warn] no runs found for sub-{sub}")
            continue

        n_comp = min(args.components, min(time_sizes))
        if n_comp < args.components:
            print(f"[warn] sub-{sub}: reducing components to {n_comp} (min timepoints).")
        ipca = IncrementalPCA(n_components=n_comp, batch_size=args.batch_size)

        for run, path in run_paths:
            if args.space == "CIFTI":
                data, _, _ = load_cifti(path)
            else:
                data, _, _ = load_mni(path, mask_arr)
            ipca.partial_fit(data)

        sub_dir = out_dir / f"sub-{sub}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        components_path = sub_dir / f"sub-{sub}_{tag}_components.npy"
        np.save(components_path, ipca.components_.astype(np.float32))

        sidecar = {
            "subject": sub,
            "space": args.space,
            "method": "shared_pca",
            "components": n_comp,
            "runs": [r for r, _ in run_paths],
            "input_files": [str(p) for _, p in run_paths],
            "explained_variance": ipca.explained_variance_ratio_.tolist(),
            "code_git_hash": code_hash,
        }
        with (sub_dir / f"sub-{sub}_{tag}.json").open("w") as f:
            json.dump(sidecar, f, indent=2)

        for run, path in run_paths:
            if args.space == "CIFTI":
                data, times, _ = load_cifti(path)
            else:
                data, times, _ = load_mni(path, mask_arr)
            coords = ipca.transform(data)
            base = sub_dir / f"sub-{sub}_run-{run}_{tag}"
            np.save(base.with_name(f"{base.name}_semspace.npy"), coords.astype(np.float32))
            np.save(base.with_name(f"{base.name}_times_fmri_s.npy"), times.astype(np.float32))
            print(f"[saved] {base.with_name(f'{base.name}_semspace.npy')} shape={coords.shape}")


if __name__ == "__main__":
    main()
