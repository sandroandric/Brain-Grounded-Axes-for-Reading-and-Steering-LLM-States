"""
Aggregate axis outputs per subject (simple mean across runs).

Usage:
  python analysis/axis_postprocess.py \
    --axes-dir outputs/plv_all_idmap \
    --out-dir outputs/meg_axes_mean \
    --band theta \
    --subjects 01 02 ... 12

Assumes files named sub-XX_run-YY_<band>_axis.npy under axes-dir.
Writes sub-XX_mean_<band>_axis.npy (+json sidecar).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Dict, List

import numpy as np


def collect_axes(axes_dir: pathlib.Path, band: str, subject: str) -> List[pathlib.Path]:
    pattern = f"sub-{subject}_run-*_"
    return sorted(axes_dir.glob(f"sub-{subject}/sub-{subject}_run-*_{band}_axis.npy"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--axes-dir", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--band", default="theta")
    p.add_argument("--subjects", nargs="+", required=True)
    args = p.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for sub in args.subjects:
        paths = collect_axes(args.axes_dir, args.band, sub)
        if not paths:
            print(f"[warn] no axes for sub-{sub}")
            continue
        arrs = [np.load(p) for p in paths]
        mean_axis = np.mean(arrs, axis=0)
        out_path = out_dir / f"sub-{sub}_mean_{args.band}_axis.npy"
        np.save(out_path, mean_axis.astype(np.float32))
        sidecar = {
            "subject": sub,
            "band": args.band,
            "source_dir": str(args.axes_dir),
            "n_runs": len(arrs),
            "axis_shape": mean_axis.shape,
            "files": [str(p) for p in paths],
        }
        with out_path.with_suffix(out_path.suffix + ".json").open("w") as f:
            json.dump(sidecar, f, indent=2)
        print(f"[saved] {out_path} (n_runs={len(arrs)})")


if __name__ == "__main__":
    main()
