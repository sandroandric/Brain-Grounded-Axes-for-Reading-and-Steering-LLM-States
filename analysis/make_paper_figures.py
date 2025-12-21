"""
Generate publication-style figures for the paper draft.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/llm_steering_tinyllama_layers/paper_summary"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def plot_brain_vs_text_probe() -> None:
    summary_path = (
        ROOT
        / "outputs/llm_probe_efficiency/run_h100_probe_layer11f_b512_20251220_2230/efficiency_summary.json"
    )
    data = load_json(summary_path)
    brain = data["brain_axis_metrics"]
    probe = data["text_probe_metrics"]
    actadd_path = ROOT / "outputs/llm_actadd_baseline/efficiency_summary.json"
    actadd = None
    if actadd_path.exists():
        actadd_data = load_json(actadd_path)
        actadd = actadd_data.get("actadd_metrics")

    points = [
        ("Brain axis", brain["logfreq_d"], brain["ppl_d"], "#2ca02c"),
        ("Text probe", probe["logfreq_d"], probe["ppl_d"], "#d62728"),
    ]
    if actadd:
        points.append(("ActAdd", actadd["logfreq_d"], actadd["ppl_d"], "#1f77b4"))

    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    pad_x = max(0.1, 0.15 * (max(xs) - min(xs) or 1.0))
    pad_y = max(0.1, 0.15 * (max(ys) - min(ys) or 1.0))

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.axhline(0, color="#cccccc", lw=0.8, zorder=0)
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=0)

    for label, x, y, color in points:
        ax.scatter(x, y, s=70, color=color, edgecolor="#222222", linewidth=0.6)
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color=color,
        )

    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    ax.set_xlabel("logfreq shift (Cohen d)")
    ax.set_ylabel("PPL shift (Cohen d)")
    ax.set_title("Efficiency: logfreq shift vs PPL cost")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "brain_vs_text_probe.png", dpi=300)
    plt.close(fig)


def plot_best_layers_heatmap() -> None:
    df = pd.read_csv(OUT_DIR / "steering_selected_layers.csv")
    models = ["TinyLlama", "Qwen2-0.5B", "GPT-2"]
    axes = ["function_content", "lexical_freq", "noun_verb", "animacy"]

    mat = np.full((len(models), len(axes)), np.nan)
    for i, model in enumerate(models):
        for j, axis in enumerate(axes):
            row = df[(df["model"] == model) & (df["axis_label"] == axis)]
            if row.empty:
                continue
            mat[i, j] = float(row.iloc[0]["d"])

    vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
    vmax = max(vmax, 0.2)
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    im = ax.imshow(mat, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                continue
            color = "white" if abs(val) > 0.5 * vmax else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    ax.set_xticks(range(len(axes)))
    ax.set_xticklabels(axes, rotation=25, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_title("Best-layer steering effects (Cohen d)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Cohen d")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "best_layers_dotplot.png", dpi=300)
    plt.close(fig)


def _mean_by_strength(stats: dict) -> tuple[list[float], list[float]]:
    xs = sorted(float(k) for k in stats.keys())
    ys = [stats[str(x)] for x in xs]
    return xs, ys


def plot_layer11_axis15() -> None:
    data = load_json(
        ROOT
        / "outputs/llm_steering_tinyllama_h100_b512_20251220_2035/layer_11/steer_eval.json"
    )
    rec = data["files"]["axis_15.json"]
    xs, ys = _mean_by_strength(rec["mean_by_strength"])

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.plot(xs, ys, marker="o", color="#1f77b4", lw=1.5, markersize=4)
    ax.axhline(0, color="#cccccc", lw=0.8)
    ax.set_xlabel("steering strength")
    ax.set_ylabel("adapter score")
    ax.set_title("Layer 11 axis 15 (TinyLlama)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "layer_11_axis_15.png", dpi=300)
    plt.close(fig)


def plot_layer4_aligned() -> None:
    base = ROOT / "outputs/llm_steering_tinyllama_h100_b512_20251220_2035"
    layer11 = load_json(base / "layer_11/steer_eval.json")
    layer4 = load_json(base / "layer_4/steer_eval.json")

    axes = {13: "Function/content", 15: "Lexical frequency"}
    signs = {}
    for axis_id in axes:
        d = layer11["files"][f"axis_{axis_id}.json"]["pos_neg_ttest"]["cohen_d"]
        signs[axis_id] = 1.0 if d >= 0 else -1.0

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    ylims = []
    for idx, (axis_id, title) in enumerate(axes.items()):
        rec = layer4["files"][f"axis_{axis_id}.json"]
        xs, ys = _mean_by_strength(rec["mean_by_strength"])
        ys = [signs[axis_id] * y for y in ys]
        axs[idx].plot(xs, ys, marker="o", lw=1.5, markersize=4)
        axs[idx].axhline(0, color="#cccccc", lw=0.8)
        axs[idx].set_title(title)
        axs[idx].set_xlabel("strength")
        ylims.extend(ys)
    max_abs = max(abs(y) for y in ylims) if ylims else 1.0
    for ax in axs:
        ax.set_ylim(-max_abs * 1.1, max_abs * 1.1)
    axs[0].set_ylabel("adapter score (aligned)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "layer_4_axis_13_15_aligned.png", dpi=300)
    plt.close(fig)


def _selected_layers_table() -> pd.DataFrame:
    best = pd.read_csv(OUT_DIR / "steering_best_layers.csv")
    axes = [13, 15, 19, 2]

    # Override TinyLlama with the primary (layer 11) values for stability.
    layer11 = load_json(
        ROOT
        / "outputs/llm_steering_tinyllama_h100_b512_20251220_2035/layer_11/steer_eval.json"
    )
    rows = []
    for axis_id in axes:
        rec = layer11["files"][f"axis_{axis_id}.json"]["pos_neg_ttest"]
        axis_label = best.loc[best["axis_id"] == axis_id, "axis_label"].iloc[0]
        rows.append(
            {
                "model": "TinyLlama",
                "axis_id": axis_id,
                "axis_label": axis_label,
                "layer": 11,
                "d": rec["cohen_d"],
                "perm_p": rec["perm_p"],
            }
        )

    # Keep best layers for other models.
    selected = best[best["model"] != "TinyLlama"].copy()
    selected = pd.concat([selected, pd.DataFrame(rows)], ignore_index=True)

    out_csv = OUT_DIR / "steering_selected_layers.csv"
    out_md = OUT_DIR / "steering_selected_layers.md"
    selected.to_csv(out_csv, index=False)

    with out_md.open("w") as f:
        f.write("| model | axis | label | selected layer | d | perm_p |\n")
        f.write("|---|---|---|---|---|---|\n")
        for _, row in selected.iterrows():
            f.write(
                f"| {row['model']} | {int(row['axis_id'])} | {row['axis_label']} | "
                f"{int(row['layer'])} | {row['d']:.3f} | {row['perm_p']:.3g} |\n"
            )

    return selected


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    _selected_layers_table()
    plot_brain_vs_text_probe()
    plot_best_layers_heatmap()
    plot_layer11_axis15()
    plot_layer4_aligned()
    print(f"[saved] figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
