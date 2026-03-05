"""Regenerate t-SNE scatter plots from metadata_tsne.csv.

Expected CSV columns:
- tsne_x
- tsne_y
- label or language
- speaker_id (optional for speaker plot)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _resolve_column(
    df: pd.DataFrame,
    explicit_name: str | None,
    fallback_candidates: list[str],
    role: str,
) -> str | None:
    if explicit_name is not None:
        if explicit_name not in df.columns:
            available = ", ".join(df.columns.astype(str))
            raise KeyError(f"{role} column '{explicit_name}' not found. Available: {available}")
        return explicit_name

    for name in fallback_candidates:
        if name in df.columns:
            return name
    return None


def _build_style_map(unique_labels: list[str]) -> dict[str, tuple[tuple[float, float, float, float], Any]]:
    marker_cycle = [
        "o", "s", "^", "v", "D", "P", "X", "*", "<", ">",
        "h", "H", "p", "8", "d",
        (3, 0, 0), (4, 0, 45), (5, 0, 0), (6, 0, 0), (7, 0, 0), (8, 0, 0), (9, 0, 0),
    ]
    color_pool: list[tuple[float, float, float, float]] = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(cmap_name)
        colors = getattr(cmap, "colors", None)
        if colors is None:
            colors = [cmap(i / 19.0) for i in range(20)]
        color_pool.extend(list(colors))

    style_map: dict[str, tuple[tuple[float, float, float, float], Any]] = {}
    for idx, label in enumerate(unique_labels):
        style_map[label] = (
            color_pool[idx % len(color_pool)],
            marker_cycle[idx % len(marker_cycle)],
        )
    return style_map


def plot_points(
    xy: np.ndarray,
    labels: list[str],
    title: str,
    out_png: Path,
    max_legend_items: int,
    show_legend: bool,
) -> None:
    plt.figure(figsize=(10, 8))

    labels_arr = np.array(labels)
    unique_labels = sorted(set(labels))
    style_map = _build_style_map(unique_labels)

    for label in unique_labels:
        idx = labels_arr == label
        color, marker = style_map[label]
        plt.scatter(
            xy[idx, 0],
            xy[idx, 1],
            s=16,
            alpha=0.8,
            c=[color],
            marker=marker,
            linewidths=0.2,
            edgecolors="black",
            label=label,
        )

    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")

    if show_legend:
        if len(unique_labels) <= max_legend_items:
            plt.legend(markerscale=2, fontsize=8, loc="best")
        else:
            plt.legend([], [], frameon=False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate t-SNE PNGs from metadata_tsne.csv")
    parser.add_argument("--csv", type=Path, required=True, help="Path to metadata_tsne.csv")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory for PNGs. Default: same directory as --csv",
    )
    parser.add_argument("--x_col", type=str, default="tsne_x", help="Column name for t-SNE x coordinate")
    parser.add_argument("--y_col", type=str, default="tsne_y", help="Column name for t-SNE y coordinate")
    parser.add_argument(
        "--label_col",
        type=str,
        default=None,
        help="Label column. If omitted, auto-detects from: label, language",
    )
    parser.add_argument(
        "--speaker_col",
        type=str,
        default=None,
        help="Speaker column. If omitted, auto-detects from: speaker_id, speaker",
    )
    parser.add_argument("--label_png", type=str, default="tsne_by_label.png", help="Output filename for label plot")
    parser.add_argument(
        "--speaker_png",
        type=str,
        default="tsne_by_speaker.png",
        help="Output filename for speaker plot",
    )
    parser.add_argument("--max_legend_items", type=int, default=40, help="Hide legend if unique groups exceed this")
    parser.add_argument("--no_legend", action="store_true", help="Disable legend rendering")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    df = pd.read_csv(args.csv)
    if args.x_col not in df.columns or args.y_col not in df.columns:
        available = ", ".join(df.columns.astype(str))
        raise KeyError(
            f"Coordinate columns '{args.x_col}' and/or '{args.y_col}' are missing. Available: {available}"
        )

    label_col = _resolve_column(df, args.label_col, ["label", "language"], "Label")
    speaker_col = _resolve_column(df, args.speaker_col, ["speaker_id", "speaker"], "Speaker")
    if label_col is None and speaker_col is None:
        available = ", ".join(df.columns.astype(str))
        raise KeyError(
            "Could not find a label/speaker column. "
            f"Provide --label_col and/or --speaker_col explicitly. Available: {available}"
        )

    out_dir = args.out_dir if args.out_dir is not None else args.csv.parent
    xy = df[[args.x_col, args.y_col]].to_numpy(dtype=np.float32)
    show_legend = not args.no_legend

    if label_col is not None:
        plot_points(
            xy=xy,
            labels=df[label_col].astype(str).tolist(),
            title="t-SNE (last layer) colored by label",
            out_png=out_dir / args.label_png,
            max_legend_items=args.max_legend_items,
            show_legend=show_legend,
        )
        print(f"Saved label plot: {(out_dir / args.label_png).resolve()}")

    if speaker_col is not None:
        plot_points(
            xy=xy,
            labels=df[speaker_col].astype(str).tolist(),
            title="t-SNE (last layer) colored by speaker",
            out_png=out_dir / args.speaker_png,
            max_legend_items=args.max_legend_items,
            show_legend=show_legend,
        )
        print(f"Saved speaker plot: {(out_dir / args.speaker_png).resolve()}")


if __name__ == "__main__":
    main()
