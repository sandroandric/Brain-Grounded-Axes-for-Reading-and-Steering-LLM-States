"""
Apply FDR correction to steering evaluation p-values.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
from typing import Dict, List, Tuple

import numpy as np


def get_field(d: Dict, field: str):
    parts = field.split(".")
    cur = d
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def fdr_bh(pvals: List[float]) -> List[float]:
    n = len(pvals)
    order = np.argsort(pvals)
    qvals = np.empty(n, dtype=float)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        p = pvals[idx]
        q = p * n / (n - rank + 1)
        prev = min(prev, q)
        qvals[idx] = prev
    return qvals.tolist()


def main():
    p = argparse.ArgumentParser(description="FDR correction for steering evals.")
    p.add_argument("--input-glob", type=str, required=True)
    p.add_argument("--p-field", type=str, default="pos_neg_ttest.p")
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    args = p.parse_args()

    paths = [pathlib.Path(p) for p in glob.glob(args.input_glob)]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise FileNotFoundError("No eval JSON files found.")

    entries = []
    for path in sorted(paths):
        data = json.loads(path.read_text())
        for name, rec in data.get("files", {}).items():
            pval = get_field(rec, args.p_field)
            if pval is None or np.isnan(pval):
                continue
            entries.append({
                "file": path.name,
                "layer": path.parent.name,
                "axis_file": name,
                "axis_id": rec.get("axis_id"),
                "p": float(pval),
            })

    pvals = [e["p"] for e in entries]
    qvals = fdr_bh(pvals) if pvals else []

    for e, q in zip(entries, qvals):
        e["p_fdr"] = q

    out = {"p_field": args.p_field, "n": len(entries), "results": entries}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
