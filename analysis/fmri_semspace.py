"""
Compute low-dimensional fMRI semantic space (Box 2 scaffold).

Default: PCA on preprocessed CIFTI dtseries (per run, per subject).
Outputs:
- *_semspace.npy (shape [n_TR, d])
- *_times_fmri_s.npy (TR times from scan start)
- sidecar JSON (method, d, TR, file provenance)
"""

from __future__ import annotations

import argparse
import json
import pathlib

import nibabel as nib
import numpy as np
from nibabel.cifti2 import cifti2_axes
from sklearn.decomposition import PCA

from analysis.common import git_hash


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


def load_mni(path: pathlib.Path, mask: np.ndarray | None = None):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    tr = img.header.get_zooms()[3]
    if data.ndim == 4:
        t_axis = data.shape[-1]
        vox = np.prod(data.shape[:-1])
        data_2d = data.reshape((-1, t_axis)).T  # (t, vox)
    else:
        raise ValueError(f"Unexpected MNI shape {data.shape}")
    if mask is not None:
        data_2d = data_2d[:, mask.reshape(-1)]
    times = np.arange(data_2d.shape[0]) * tr
    return data_2d, times, float(tr)


def main():
    p = argparse.ArgumentParser(description="Build fMRI semantic space via PCA.")
    p.add_argument("--dataset-root", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--components", type=int, default=100)
    p.add_argument("--space", choices=["CIFTI", "MNI"], default="CIFTI")
    p.add_argument("--subjects", nargs="+", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--method", choices=["pca"], default="pca")
    p.add_argument("--mask", type=pathlib.Path, help="Optional brain mask for MNI space.")
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

    for sub in args.subjects:
        for run in args.runs:
            if args.space == "CIFTI":
                fmri_path = (
                    args.dataset_root / "derivatives" / "preprocessed_data" / f"sub-{sub}" / "CIFTI" / f"sub-{sub}_task-RDR_run-{run}_bold.dtseries.nii"
                )
            else:
                fmri_path = (
                    args.dataset_root / "derivatives" / "preprocessed_data" / f"sub-{sub}" / "MNI" / f"sub-{sub}_task-RDR_run-{run}_bold.nii.gz"
                )
            if not fmri_path.exists():
                print(f"[warn] missing {fmri_path}, skipping")
                continue

            if args.space == "CIFTI":
                data, times, tr = load_cifti(fmri_path)
            else:
                data, times, tr = load_mni(fmri_path, mask_arr)

            if data.shape[0] < args.components:
                n_comp = data.shape[0]
            else:
                n_comp = args.components

            pca = PCA(n_components=n_comp, svd_solver="randomized")
            coords = pca.fit_transform(data)

            base = out_dir / f"sub-{sub}" / f"sub-{sub}_run-{run}_{args.space.lower()}_{args.method}"
            base.parent.mkdir(parents=True, exist_ok=True)
            semspace_path = base.with_name(f"{base.name}_semspace.npy")
            times_path = base.with_name(f"{base.name}_times_fmri_s.npy")
            pca_path = base.with_name(f"{base.name}_pca_components.npy")
            np.save(semspace_path, coords.astype(np.float32))
            np.save(times_path, times.astype(np.float32))
            np.save(pca_path, pca.components_.astype(np.float32))
            sidecar = {
                "subject": sub,
                "run": run,
                "space": args.space,
                "method": args.method,
                "components": n_comp,
                "tr_s": tr,
                "input_file": str(fmri_path),
                "explained_variance": pca.explained_variance_ratio_.tolist(),
                "code_git_hash": code_hash,
            }
            with base.with_name(f"{base.name}.json").open("w") as f:
                json.dump(sidecar, f, indent=2)
            print(f"[saved] {semspace_path} shape={coords.shape}")


if __name__ == "__main__":
    main()
