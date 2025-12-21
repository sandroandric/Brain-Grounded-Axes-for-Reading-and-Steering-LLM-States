"""
Batch runner for compute_plv_states.py (Box 1).
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
    res = subprocess.run(cmd)
    if res.returncode != 0:
        sys.exit(res.returncode)


def main():
    p = argparse.ArgumentParser(description="Batch wrapper for compute_plv_states.py")
    p.add_argument("--dataset-root", type=pathlib.Path, required=True)
    p.add_argument("--run-to-story", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--band", default="theta")
    p.add_argument("--metric", default="plv")
    p.add_argument("--state-space", default="edge_pca")
    p.add_argument("--pca-components", type=int, default=128)
    p.add_argument("--window-length", type=float, default=2.0)
    p.add_argument("--window-step", type=float, default=0.5)
    p.add_argument("--picks", default="grad")
    p.add_argument("--decim", type=int, default=1)
    p.add_argument("--subjects", nargs="+", default=ALL_SUBJECTS)
    p.add_argument("--runs", nargs="+", default=ALL_RUNS)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()

    if not args.execute:
        print("[dry-run] add --execute to run.")
        return

    for sub in args.subjects:
        cmd = [
            sys.executable,
            str(args.dataset_root / "analysis" / "compute_plv_states.py"),
            "--dataset-root",
            str(args.dataset_root),
            "--run-to-story",
            str(args.run_to_story),
            "--out-dir",
            str(args.out_dir),
            "--band",
            args.band,
            "--metric",
            args.metric,
            "--state-space",
            args.state_space,
            "--pca-components",
            str(args.pca_components),
            "--picks",
            args.picks,
            "--window-length",
            str(args.window_length),
            "--window-step",
            str(args.window_step),
            "--decim",
            str(args.decim),
            "--subjects",
            sub,
            "--runs",
            *args.runs,
            "--execute",
        ]
        run_cmd(cmd)


if __name__ == "__main__":
    main()
