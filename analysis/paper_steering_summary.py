"""
Build publication summary tables and plots for TinyLlama steering runs.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


AXIS_LABELS = {
    13: "function_content",
    15: "lexical_freq",
    19: "noun_verb",
    2: "animacy",
}

AXIS_METRIC = {
    13: "function_ratio",
    15: "logfreq_mean",
    19: "noun_ratio",
    2: "animate_rate",
}


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


def read_json(path: pathlib.Path):
    return json.loads(path.read_text())


def load_steer_eval(layer_dir: pathlib.Path):
    return read_json(layer_dir / "steer_eval.json")


def get_metric_stats(metrics: Dict, metric: str):
    return metrics.get(metric, None)


def mean_by_strength(stats: Dict[str, float]):
    strengths = sorted(float(k) for k in stats.keys())
    means = [stats[str(s)] for s in strengths]
    return strengths, means


def plot_axis(fig_path: pathlib.Path, axis_id: int, layer: int, steer, text, ppl):
    if plt is None:
        return
    metric = AXIS_METRIC.get(axis_id, "metric")
    fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)
    fig.suptitle(f"Layer {layer} axis {axis_id} ({AXIS_LABELS.get(axis_id, axis_id)})")

    # Adapter score
    s_stats = steer["mean_by_strength"]
    xs, ys = mean_by_strength(s_stats)
    axes[0].plot(xs, ys, marker="o")
    axes[0].set_ylabel("adapter score")
    axes[0].axhline(0, color="#999999", linewidth=0.6)

    # Text metric
    if text:
        t_stats = text["mean_by_strength"]
        xs, ys = mean_by_strength(t_stats)
        axes[1].plot(xs, ys, marker="o")
    axes[1].set_ylabel(metric)

    # Perplexity
    if ppl:
        p_stats = ppl["mean_by_strength"]
        xs, ys = mean_by_strength(p_stats)
        axes[2].plot(xs, ys, marker="o")
    axes[2].set_ylabel("perplexity")
    axes[2].set_xlabel("steering strength")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Build steering summary tables/plots.")
    p.add_argument("--root", type=pathlib.Path, default=pathlib.Path("outputs/llm_steering_tinyllama_layers"))
    p.add_argument("--layers", nargs="+", type=int, default=[4, 11, 20])
    p.add_argument("--axes", nargs="+", type=int, default=[13, 15, 19, 2])
    p.add_argument("--out-dir", type=pathlib.Path, default=None)
    args = p.parse_args()

    root = args.root
    out_dir = args.out_dir or (root / "paper_summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    text_path = root / "text_eval_by_layer.json"
    ppl_path = root / "ppl_eval_by_layer.json"
    text_data = read_json(text_path) if text_path.exists() else {}
    ppl_data = read_json(ppl_path) if ppl_path.exists() else {}

    rows = []
    fdr_entries = []

    for layer in args.layers:
        layer_dir = root / f"layer_{layer}"
        if not (layer_dir / "steer_eval.json").exists():
            continue
        steer = load_steer_eval(layer_dir)
        for axis in args.axes:
            axis_key = f"axis_{axis}.json"
            if axis_key not in steer["files"]:
                continue
            rec = steer["files"][axis_key]
            fdr_entries.append(
                {
                    "layer": layer,
                    "axis_id": axis,
                    "p": rec["pos_neg_ttest"]["p"],
                }
            )

            text_key = f"layer_{layer}/{axis_key}"
            text_rec = text_data.get("files", {}).get(text_key, {})
            text_metric = AXIS_METRIC.get(axis)
            text_stats = text_rec.get("metrics", {}).get(text_metric) if text_metric else None

            ppl_key = f"layer_{layer}/{axis_key}"
            ppl_rec = ppl_data.get("files", {}).get(ppl_key, {})
            ppl_stats = ppl_rec.get("metrics")

            rows.append(
                {
                    "layer": layer,
                    "axis_id": axis,
                    "axis_label": AXIS_LABELS.get(axis, str(axis)),
                    "adapter_p": rec["pos_neg_ttest"]["p"],
                    "adapter_perm_p": rec["pos_neg_ttest"]["perm_p"],
                    "adapter_d": rec["pos_neg_ttest"]["cohen_d"],
                    "adapter_r": rec["strength_corr"]["r"],
                    "text_metric": text_metric,
                    "text_p": text_stats["pos_neg_ttest"]["p"] if text_stats else None,
                    "text_perm_p": text_stats["pos_neg_ttest"]["perm_p"] if text_stats else None,
                    "text_d": text_stats["pos_neg_ttest"]["cohen_d"] if text_stats else None,
                    "ppl_p": ppl_stats["pos_neg_ttest"]["p"] if ppl_stats else None,
                    "ppl_perm_p": ppl_stats["pos_neg_ttest"]["perm_p"] if ppl_stats else None,
                    "ppl_d": ppl_stats["pos_neg_ttest"]["cohen_d"] if ppl_stats else None,
                }
            )

            # Plot for layer 4 axes 13/15
            if layer == 4 and axis in {13, 15}:
                fig_path = out_dir / f"layer_{layer}_axis_{axis}.png"
                plot_axis(fig_path, axis, layer, rec, text_stats, ppl_stats)

    # FDR across layer/axis
    pvals = [r["p"] for r in fdr_entries]
    qvals = fdr_bh(pvals) if pvals else []
    for entry, q in zip(fdr_entries, qvals):
        entry["p_fdr"] = q
        for row in rows:
            if row["layer"] == entry["layer"] and row["axis_id"] == entry["axis_id"]:
                row["adapter_fdr_q"] = q
                break

    # Save tables
    json_path = out_dir / "steering_summary.json"
    json_path.write_text(json.dumps(rows, indent=2))

    # CSV
    csv_path = out_dir / "steering_summary.csv"
    headers = [
        "layer",
        "axis_id",
        "axis_label",
        "adapter_p",
        "adapter_perm_p",
        "adapter_d",
        "adapter_r",
        "adapter_fdr_q",
        "text_metric",
        "text_p",
        "text_perm_p",
        "text_d",
        "ppl_p",
        "ppl_perm_p",
        "ppl_d",
    ]
    with csv_path.open("w") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            vals = [row.get(h, "") for h in headers]
            f.write(",".join("" if v is None else str(v) for v in vals) + "\n")

    # Markdown summary
    md_path = out_dir / "steering_summary.md"
    with md_path.open("w") as f:
        f.write("| layer | axis | label | adapter p | q | d | text metric p | ppl p |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for row in rows:
            f.write(
                f"| {row['layer']} | {row['axis_id']} | {row['axis_label']} | "
                f"{row['adapter_p']:.4g} | {row.get('adapter_fdr_q', float('nan')):.4g} | "
                f"{row['adapter_d']:.3g} | {row['text_p'] if row['text_p'] is not None else ''} | "
                f"{row['ppl_p'] if row['ppl_p'] is not None else ''} |\n"
            )

    # Selected set (FDR < 0.05)
    selected = [r for r in rows if r.get("adapter_fdr_q", 1.0) < 0.05]
    sel_path = out_dir / "steering_selected.json"
    sel_path.write_text(json.dumps(selected, indent=2))

    print(f"[saved] {json_path}")
    print(f"[saved] {csv_path}")
    print(f"[saved] {md_path}")
    print(f"[saved] {sel_path}")


if __name__ == "__main__":
    main()
