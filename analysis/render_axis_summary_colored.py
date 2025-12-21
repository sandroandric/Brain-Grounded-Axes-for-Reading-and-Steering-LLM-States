"""
Render axis summary CSV as a color-coded HTML table.
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
from typing import List


def parse_float(value: str):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def cell_class(col: str, value: str) -> str:
    val = parse_float(value)
    if val is None:
        return "na"

    if col.endswith("_p") or "perm_p" in col:
        if val < 0.01:
            return "good"
        if val < 0.05:
            return "mid"
        return "bad"

    if col.endswith("_r") or "pearson_r" in col or "spearman_r" in col:
        mag = abs(val)
        if mag >= 0.2:
            return "good"
        if mag >= 0.1:
            return "mid"
        return "bad"

    if col.endswith("_d") or "cohen_d" in col:
        mag = abs(val)
        if mag >= 0.5:
            return "good"
        if mag >= 0.2:
            return "mid"
        return "bad"

    return ""


def html_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def render_table(rows: List[dict], columns: List[str], title: str) -> str:
    header_cells = "".join(f"<th>{html_escape(col)}</th>" for col in columns)
    body_rows = []
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            cls = cell_class(col, value)
            class_attr = f' class="{cls}"' if cls else ""
            cells.append(f"<td{class_attr}>{html_escape(value)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    body_html = "\n".join(body_rows)
    return f"""
    <h2>{html_escape(title)}</h2>
    <table>
      <thead><tr>{header_cells}</tr></thead>
      <tbody>
      {body_html}
      </tbody>
    </table>
    """


def main():
    p = argparse.ArgumentParser(description="Render color-coded axis summary HTML.")
    p.add_argument("--csv", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--axes", nargs="*", type=int, default=None)
    p.add_argument("--title", default="Axis Summary")
    args = p.parse_args()

    rows = list(csv.DictReader(args.csv.open()))
    if args.axes:
        axes = set(args.axes)
        rows = [row for row in rows if int(row.get("axis", -1)) in axes]

    columns = list(rows[0].keys()) if rows else []

    css = """
    <style>
      body { font-family: Arial, sans-serif; margin: 24px; }
      table { border-collapse: collapse; width: 100%; font-size: 12px; }
      th, td { border: 1px solid #ccc; padding: 6px 8px; vertical-align: top; }
      th { background: #f4f4f4; position: sticky; top: 0; }
      .good { background: #c8f7c5; }
      .mid { background: #fff3b0; }
      .bad { background: #f6c1c1; }
      .na { background: #efefef; color: #777; }
      .legend { margin-bottom: 16px; }
      .legend span { padding: 4px 8px; border: 1px solid #ccc; margin-right: 8px; }
    </style>
    """

    legend = """
    <div class="legend">
      <strong>Color key:</strong>
      <span class="good">green = strong</span>
      <span class="mid">yellow = medium</span>
      <span class="bad">red = weak / not significant</span>
      <span class="na">gray = missing</span>
    </div>
    """

    html = f"<!doctype html><html><head><meta charset=\"utf-8\">{css}</head><body>{legend}"
    html += render_table(rows, columns, args.title)
    html += "</body></html>"

    args.out.write_text(html, encoding="utf-8")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
