"""
Backfill sidecar JSONs for existing axis outputs.

Usage:
  python analysis/backfill_sidecars.py \
    --dataset-root /Volumes/Neuro_Data_16122025/ds004078 \
    --run-to-story mapping/id_map.csv \
    --outputs-dir outputs/plv_all_idmap \
    --band theta

This scans for *_axis.npy under outputs-dir and writes matching .json files
using metadata/runs_index.csv for timing constants and paths.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

import numpy as np
import pandas as pd


def load_runs_index(root: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(root / "metadata" / "runs_index.csv", dtype=str)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=pathlib.Path, required=True)
    p.add_argument("--run-to-story", type=pathlib.Path, required=True)
    p.add_argument("--outputs-dir", type=pathlib.Path, required=True)
    p.add_argument("--band", default="theta")
    args = p.parse_args()

    df_idx = load_runs_index(args.dataset_root)
    df_idx["run_int"] = df_idx["run"].astype(int)
    df_idx["story_id"] = df_idx["story_id"].astype(str).str.zfill(2)
    df_idx = df_idx.set_index(["subject", "run"])

    axis_files = sorted(args.outputs_dir.rglob(f"*_{args.band}_axis.npy"))
    for axis_path in axis_files:
        m = re.search(r"sub-(\d+)_run-(\d+)_", axis_path.name)
        if not m:
            continue
        sub, run = m.group(1), str(int(m.group(2)))
        if axis_path.with_suffix(axis_path.suffix + ".json").exists():
            continue
        axis = np.load(axis_path)
        info = df_idx.loc[(sub, run)].to_dict()
        sidecar = {
            "dataset_root": str(args.dataset_root),
            "subject": sub,
            "run": run,
            "story_id": f"{int(run):02d}",
            "band": args.band,
            "window_length_s": None,
            "window_step_s": None,
            "feature": None,
            "embed_model": None,
            "decim": None,
            "picks": None,
            "sfreq": info.get("sampling_frequency_hz"),
            "n_channels": None,
            "n_windows": None,
            "axis_shape": axis.shape,
            "time_align_shift_s": info.get("time_align_shift_s"),
            "auditory_delay_s": info.get("auditory_delay_s"),
            "run_to_story_mapping": str(args.run_to_story),
            "code_git_hash": "unknown",
        }
        out_json = axis_path.with_suffix(axis_path.suffix + ".json")
        with out_json.open("w") as f:
            json.dump(sidecar, f, indent=2)
        print(f"[backfilled] {out_json}")


if __name__ == "__main__":
    main()
