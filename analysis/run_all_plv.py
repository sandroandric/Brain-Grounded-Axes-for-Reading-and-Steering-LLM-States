"""
Batch runner for PLV + semantic axes across all subjects/runs.

Usage:
  python analysis/run_all_plv.py \
    --dataset-root /Volumes/Neuro_Data_16122025/ds004078 \
    --run-to-story mapping/run_to_story.csv \
    --out-dir outputs/plv_all \
    --band theta \
    --feature embedding_change \
    --embed-model gpt \
    --window-length 2.0 \
    --window-step 0.5 \
    --subjects 01 02 03 04 05 06 07 08 09 10 11 12 \
    --execute

Notes:
- Honors the missing-start trigger adjustment (sub-08_run-16, sub-09_run-7) baked into step5_plv_axes.py.
- Sequential by default; adjust as needed for parallelization (e.g., run multiple instances).
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

ALL_SUBJECTS = [f"{i:02d}" for i in range(1, 13)]
ALL_RUNS = [str(i) for i in range(1, 61)]


def run_cmd(cmd):
    print(f"[run] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    p = argparse.ArgumentParser(description="Batch runner for PLV + semantic axes (step5_plv_axes.py).")
    p.add_argument("--dataset-root", type=pathlib.Path, required=True)
    p.add_argument("--run-to-story", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--band", default="theta")
    p.add_argument("--feature", default="embedding_change")
    p.add_argument("--embed-model", default="gpt")
    p.add_argument("--window-length", type=float, default=2.0)
    p.add_argument("--window-step", type=float, default=0.5)
    p.add_argument("--decim", type=int, default=1)
    p.add_argument("--subjects", nargs="+", default=ALL_SUBJECTS)
    p.add_argument("--runs", nargs="+", default=ALL_RUNS)
    p.add_argument("--execute", action="store_true", help="Actually run step5_plv_axes; otherwise dry-run.")
    args = p.parse_args()

    if not args.execute:
        print("[dry-run] add --execute to run.")
        return

    for sub in args.subjects:
        cmd = [
            sys.executable,
            str(args.dataset_root / "analysis" / "step5_plv_axes.py"),
            "--dataset-root",
            str(args.dataset_root),
            "--subject",
            sub,
            "--runs",
            *args.runs,
            "--run-to-story",
            str(args.run_to_story),
            "--feature",
            args.feature,
            "--embed-model",
            args.embed_model,
            "--out-dir",
            str(args.out_dir / f"sub-{sub}"),
            "--band",
            args.band,
            "--window-length",
            str(args.window_length),
            "--window-step",
            str(args.window_step),
            "--decim",
            str(args.decim),
            "--execute",
        ]
        run_cmd(cmd)


if __name__ == "__main__":
    main()
