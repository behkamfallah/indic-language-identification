"""Plot t-SNE with 10 random speakers highlighted.

- Randomly samples N speakers (default 10) from metadata_tsne.csv.
- Highlighted speakers get distinct colors.
- All non-selected points are plotted in grey.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Highlight random speakers on t-SNE plot.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to metadata_tsne.csv")
    parser.add_argument("--out_png", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--x_col", type=str, default="tsne_x", help="Column name for t-SNE x")
    parser.add_argument("--y_col", type=str, default="tsne_y", help="Column name for t-SNE y")
    parser.add_argument(
        "--speaker_col",
        type=str,
        default="speaker_id",
        help="Speaker column in CSV",
    )
    parser.add_argument("--num_speakers", type=int, default=10, help="How many random speakers to highlight")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
    parser.add_argument("--title", type=str, default=None, help="Plot title")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    df = pd.read_csv(args.csv)

    required_cols = [args.x_col, args.y_col, args.speaker_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        available = ", ".join(df.columns.astype(str))
        raise KeyError(f"Missing required columns: {missing}. Available: {available}")

    df = df.copy()
    df[args.speaker_col] = df[args.speaker_col].astype(str)

    unique_speakers = sorted(df[args.speaker_col].unique().tolist())
    if not unique_speakers:
        raise ValueError("No speakers found in the provided CSV.")

    n = min(args.num_speakers, len(unique_speakers))
    if n <= 0:
        raise ValueError("--num_speakers must be > 0.")

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(unique_speakers, size=n, replace=False).tolist()
    chosen_set = set(chosen)

    plt.figure(figsize=(10, 8))

    # Plot non-selected speakers first in grey.
    other = ~df[args.speaker_col].isin(chosen_set)
    plt.scatter(
        df.loc[other, args.x_col].to_numpy(),
        df.loc[other, args.y_col].to_numpy(),
        s=14,
        alpha=0.45,
        c="#b5b5b5",
        marker="o",
        linewidths=0.0,
        label="other speakers",
    )

    # Use a qualitative palette with exactly 10 distinct colors.
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(n)]

    for i, speaker in enumerate(chosen):
        idx = df[args.speaker_col] == speaker
        plt.scatter(
            df.loc[idx, args.x_col].to_numpy(),
            df.loc[idx, args.y_col].to_numpy(),
            s=24,
            alpha=0.9,
            c=[colors[i]],
            marker="o",
            edgecolors="black",
            linewidths=0.2,
            label=speaker,
        )

    title = args.title or f"t-SNE random {n} speakers highlighted (seed={args.seed})"
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(markerscale=1.5, fontsize=8, loc="best")
    plt.tight_layout()

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png, dpi=200)
    plt.close()

    print(f"Saved plot: {args.out_png.resolve()}")
    print(f"Chosen speakers ({n}): {', '.join(chosen)}")


if __name__ == "__main__":
    main()
