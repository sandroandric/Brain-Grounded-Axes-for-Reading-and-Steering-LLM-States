"""
FDR correction for text-level steering metrics.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import List

import numpy as np


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
    p = argparse.ArgumentParser(description="FDR correction for text eval metrics.")
    p.add_argument("--input-json", type=pathlib.Path, required=True)
    p.add_argument("--metric", type=str, required=True)
    p.add_argument("--p-field", type=str, default="pos_neg_ttest.p")
    p.add_argument("--out-json", type=pathlib.Path, required=True)
    args = p.parse_args()

    data = json.loads(args.input_json.read_text())
    entries = []
    for key, rec in data.get("files", {}).items():
        metrics = rec.get("metrics", {})
        if args.metric not in metrics:
            continue
        stats = metrics[args.metric]
        pval = stats.get("pos_neg_ttest", {}).get("p") if args.p_field == "pos_neg_ttest.p" else None
        if pval is None or np.isnan(pval):
            continue
        entries.append(
            {
                "file": key,
                "layer": rec.get("layer"),
                "axis_id": rec.get("axis_id"),
                "metric": args.metric,
                "p": float(pval),
            }
        )

    pvals = [e["p"] for e in entries]
    qvals = fdr_bh(pvals) if pvals else []
    for e, q in zip(entries, qvals):
        e["p_fdr"] = q

    out = {"metric": args.metric, "n": len(entries), "results": entries}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
