"""Regenerate t-SNE PNGs with language colors and per-language speaker shapes.

Inputs:
- metadata_tsne.csv with at least:
  - tsne_x
  - tsne_y
  - language/label column
  - speaker column

Outputs (by default):
- tsne_by_label.png
- tsne_by_speaker.png
"""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _resolve_column(
    df: pd.DataFrame,
    explicit_name: str | None,
    fallback_candidates: List[str],
    role: str,
) -> str:
    if explicit_name is not None:
        if explicit_name not in df.columns:
            available = ", ".join(df.columns.astype(str))
            raise KeyError(f"{role} column '{explicit_name}' not found. Available: {available}")
        return explicit_name

    for name in fallback_candidates:
        if name in df.columns:
            return name

    available = ", ".join(df.columns.astype(str))
    raise KeyError(
        f"Could not auto-detect {role} column from {fallback_candidates}. "
        f"Provide --{role.lower()}_col. Available: {available}"
    )


def _build_marker_pool() -> List[Any]:
    # Base filled markers + tuple-form markers to provide a large unique pool.
    markers: List[Any] = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h", "H", "p", "8", "d"]
    for sides in range(3, 13):
        for style in (0, 1, 2):
            for angle in (0, 15, 30, 45, 60, 75):
                markers.append((sides, style, angle))
    return markers


def _build_language_colors(languages: List[str]) -> Dict[str, Tuple[float, float, float, float]]:
    # Golden-ratio hue progression gives strong separation for adjacent classes.
    phi_conjugate = 0.618033988749895
    colors: Dict[str, Tuple[float, float, float, float]] = {}
    hue = 0.11
    for language in languages:
        hue = (hue + phi_conjugate) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
        colors[language] = (r, g, b, 1.0)
    return colors


def _assign_speaker_markers_per_language(
    df: pd.DataFrame,
    language_col: str,
    speaker_col: str,
) -> Dict[str, Dict[str, Any]]:
    marker_pool = _build_marker_pool()
    marker_map: Dict[str, Dict[str, Any]] = {}

    for language in sorted(df[language_col].astype(str).unique().tolist()):
        speakers = sorted(
            df.loc[df[language_col].astype(str) == language, speaker_col].astype(str).unique().tolist()
        )
        if len(speakers) > len(marker_pool):
            raise ValueError(
                f"Language '{language}' has {len(speakers)} speakers, "
                f"but marker pool has only {len(marker_pool)} unique markers."
            )
        marker_map[language] = {speaker: marker_pool[idx] for idx, speaker in enumerate(speakers)}

    return marker_map


def _plot_by_language(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    language_col: str,
    language_colors: Dict[str, Tuple[float, float, float, float]],
    out_png: Path,
    legend: bool,
    max_legend_items: int,
) -> None:
    plt.figure(figsize=(10, 8))
    for language in sorted(language_colors.keys()):
        idx = df[language_col].astype(str) == language
        plt.scatter(
            df.loc[idx, x_col].to_numpy(),
            df.loc[idx, y_col].to_numpy(),
            s=16,
            alpha=0.85,
            c=[language_colors[language]],
            marker="o",
            linewidths=0.2,
            edgecolors="black",
            label=language,
        )

    plt.title("t-SNE (last layer) colored by label")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    if legend:
        if len(language_colors) <= max_legend_items:
            plt.legend(markerscale=2, fontsize=8, loc="best")
        else:
            plt.legend([], [], frameon=False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _plot_by_speaker_with_language_colors(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    language_col: str,
    speaker_col: str,
    language_colors: Dict[str, Tuple[float, float, float, float]],
    speaker_markers: Dict[str, Dict[str, Any]],
    out_png: Path,
    legend: bool,
    max_legend_items: int,
) -> None:
    plt.figure(figsize=(10, 8))

    groups = df.groupby([language_col, speaker_col], sort=True, dropna=False)
    for (language, speaker), group in groups:
        language_str = str(language)
        speaker_str = str(speaker)
        plt.scatter(
            group[x_col].to_numpy(),
            group[y_col].to_numpy(),
            s=18,
            alpha=0.85,
            c=[language_colors[language_str]],
            marker=speaker_markers[language_str][speaker_str],
            linewidths=0.2,
            edgecolors="black",
            label=f"{language_str}:{speaker_str}",
        )

    plt.title("t-SNE (last layer) color=language, marker=speaker")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    if legend:
        group_count = groups.ngroups
        if group_count <= max_legend_items:
            plt.legend(markerscale=2, fontsize=7, loc="best")
        else:
            plt.legend([], [], frameon=False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replot t-SNE from metadata CSV with language colors and speaker markers."
    )
    parser.add_argument("--csv", type=Path, required=True, help="Path to metadata_tsne.csv")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output directory (default: CSV parent)")
    parser.add_argument("--x_col", type=str, default="tsne_x", help="X coordinate column")
    parser.add_argument("--y_col", type=str, default="tsne_y", help="Y coordinate column")
    parser.add_argument(
        "--language_col",
        type=str,
        default=None,
        help="Language column (auto-detect: label, language)",
    )
    parser.add_argument(
        "--speaker_col",
        type=str,
        default=None,
        help="Speaker column (auto-detect: speaker_id, speaker)",
    )
    parser.add_argument("--label_png", type=str, default="tsne_by_label.png", help="Output label-plot filename")
    parser.add_argument("--speaker_png", type=str, default="tsne_by_speaker.png", help="Output speaker-plot filename")
    parser.add_argument("--legend", action="store_true", help="Enable legends")
    parser.add_argument("--max_legend_items", type=int, default=80, help="Hide legend if item count exceeds this")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    df = pd.read_csv(args.csv)
    if args.x_col not in df.columns or args.y_col not in df.columns:
        available = ", ".join(df.columns.astype(str))
        raise KeyError(
            f"Coordinate columns '{args.x_col}' and/or '{args.y_col}' not found. Available: {available}"
        )

    language_col = _resolve_column(df, args.language_col, ["label", "language"], "Language")
    speaker_col = _resolve_column(df, args.speaker_col, ["speaker_id", "speaker"], "Speaker")
    out_dir = args.out_dir if args.out_dir is not None else args.csv.parent

    # Normalize types once for deterministic style mapping.
    df = df.copy()
    df[language_col] = df[language_col].astype(str)
    df[speaker_col] = df[speaker_col].astype(str)

    languages = sorted(df[language_col].unique().tolist())
    language_colors = _build_language_colors(languages)
    speaker_markers = _assign_speaker_markers_per_language(df, language_col, speaker_col)

    label_png = out_dir / args.label_png
    speaker_png = out_dir / args.speaker_png

    _plot_by_language(
        df=df,
        x_col=args.x_col,
        y_col=args.y_col,
        language_col=language_col,
        language_colors=language_colors,
        out_png=label_png,
        legend=args.legend,
        max_legend_items=args.max_legend_items,
    )
    _plot_by_speaker_with_language_colors(
        df=df,
        x_col=args.x_col,
        y_col=args.y_col,
        language_col=language_col,
        speaker_col=speaker_col,
        language_colors=language_colors,
        speaker_markers=speaker_markers,
        out_png=speaker_png,
        legend=args.legend,
        max_legend_items=args.max_legend_items,
    )

    print(f"Saved label plot: {label_png.resolve()}")
    print(f"Saved speaker plot: {speaker_png.resolve()}")
    print(f"Languages: {len(languages)} | speaker groups: {df.groupby([language_col, speaker_col]).ngroups}")


if __name__ == "__main__":
    main()
