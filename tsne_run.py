"""analyze_tsne.py

Modular t-SNE analysis runner.

This is a thin CLI wrapper around tsne_utils.py.

It intentionally follows the same conventions as analyze_tsne_minimal.py:
- Load YAML config + merge overrides
- Resolve model_dir (explicit or auto-pick latest run under save_dir)
- Save artifacts under: <output_dir>/<run_id>/tsne_analysis/<tag>/

Extras (compared to minimal):
- Saves embeddings + t-SNE coordinates
- Writes a short kNN probe report (label and speaker decodability)

Typical usage (baseline):
  python analyze_tsne.py --config task.yaml --tag baseline

Debiased run (example):
  python analyze_tsne.py --config task.yaml --tag debiased \
    --override data.variant=debiased_v1 --model_dir ./models/<DEBIASED_RUN>

Compare (optional):
  python analyze_tsne.py --compare \
    --compare_a <.../tsne_analysis/baseline> \
    --compare_b <.../tsne_analysis/debiased>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tsne_utils import TsneExp, compare_two_runs, run_tsne_analysis


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="t-SNE analysis (modular, reusable)")

    # Normal run
    ap.add_argument("--config", type=Path, help="YAML config used for the run")
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        help='Config override "key=value". Can repeat: --override a.b=1 --override c=2',
    )
    ap.add_argument(
        "--model_dir",
        type=Path,
        default=None,
        help="Explicit HF model dir. If omitted, auto-pick latest run folder under save_dir.",
    )
    ap.add_argument("--max_items", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="default", help="Subfolder name under tsne_analysis/")
    ap.add_argument("--knn_k", type=int, default=5)
    ap.add_argument(
        "--kmeans_k",
        type=int,
        default=None,
        help="Override KMeans cluster count. If omitted, uses number of unique labels in each split.",
    )
    ap.add_argument(
        "--tsne_perplexity",
        type=int,
        default=None,
        help="Override t-SNE perplexity. If omitted, uses a safe heuristic.",
    )
    ap.add_argument("--show_plots", action="store_true", help="Display plots (useful in notebooks)")

    # Optional compare mode
    ap.add_argument("--compare", action="store_true", help="Compare two already-saved run folders")
    ap.add_argument("--compare_a", type=Path, default=None, help="Path to run A folder (tsne_analysis/<tag>)")
    ap.add_argument("--compare_b", type=Path, default=None, help="Path to run B folder (tsne_analysis/<tag>)")
    ap.add_argument("--compare_out", type=Path, default=None, help="Where to write comparison report")

    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    if args.compare:
        if args.compare_a is None or args.compare_b is None:
            raise ValueError("--compare requires --compare_a and --compare_b")
        out_dir = args.compare_out or (Path(args.compare_a).parent / "compare")
        report = compare_two_runs(Path(args.compare_a), Path(args.compare_b), out_dir=out_dir)
        print(f"Wrote: {report}")
        return

    if args.config is None:
        raise ValueError("--config is required unless --compare is used")

    exp = TsneExp(
        config_path=args.config,
        overrides=list(args.override),
        model_dir=args.model_dir,
        max_items=args.max_items,
        batch_size=args.batch_size,
        seed=args.seed,
        tag=args.tag,
        knn_k=args.knn_k,
        kmeans_k=args.kmeans_k,
        tsne_perplexity=args.tsne_perplexity,
    )

    paths = run_tsne_analysis(exp, show_plots=args.show_plots)
    print(f"Done. Saved to: {paths.out_dir.resolve()}")
    print(f"Summary report: {paths.report_txt.name}")
    print(
        "Representative KMeans-majority plots: "
        f"{Path(paths.plot_by_kmeans_majority_compatibility_alpha70_png).name}, "
        f"{Path(paths.plot_by_kmeans_majority_compatibility_png).name}, "
        f"{Path(paths.plot_by_kmeans_majority_compatibility_alpha30_png).name}"
    )
    print(
        "Representative KMeans-majority blue/red annotated plot: "
        f"{Path(paths.plot_by_kmeans_majority_blue_red_annotated_png).name}"
    )
    for split_name, split_dir in paths.split_dirs.items():
        print(f"[{split_name}] dir: {split_dir.resolve()}")
    for split_name, split_report in paths.split_reports.items():
        print(f"[{split_name}] report: {split_report.name}")


if __name__ == "__main__":
    main()
