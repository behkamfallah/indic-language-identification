"""
Reusable utilities for extracting last-layer audio representations, running t-SNE,
and producing lightweight diagnostic reports.

Design goals:
- Match the *working* flow of analyze_tsne_minimal.py (dataset/model resolution,
  batching, pooling) to avoid regressions.
- Keep the interface reusable for comparing multiple experiments (e.g. baseline
  vs debiased dataset/model) without duplicating code.

Key idea: a mutable experiment container (TsneExp) that you can tweak in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    set_seed,
)


from config_utils import load_yaml, get_nested, merge_dicts, parse_overrides
from dataset_utils import prepare_encoded_datasets
from trainer_utils import AudioDataCollator


@dataclass
class TsneExp:
    """Mutable experiment container.

    You can construct one, then modify fields in notebooks/scripts before running.
    """

    # Required-ish
    config_path: Path

    # Optional modifiers
    overrides: list[str] = field(default_factory=list)
    model_dir: Optional[Path] = None  # if None -> auto-pick latest run

    # Run controls
    max_items: int = 2000
    batch_size: int = 32
    seed: int = 42

    # Output naming
    tag: str = "default"  # used as a subfolder under tsne_analysis/

    # Analysis knobs
    knn_k: int = 5
    kmeans_k: Optional[int] = None  # if None -> number of unique labels in split
    tsne_perplexity: Optional[int] = None  # if None -> auto heuristic


@dataclass
class TsneRunPaths:
    """Where artifacts for one run were saved."""

    out_dir: Path
    embeddings_npy: Path
    tsne_npy: Path
    kmeans_npy: Path | str = "kmeans_labels.npy"
    plot_by_label_png: Path | str = "tsne_by_label.png"
    plot_by_speaker_png: Path | str = "tsne_by_speaker.png"
    plot_by_knn_label_png: Path | str = "tsne_by_knn_label.png"
    plot_by_kmeans_png: Path | str = "tsne_by_kmeans.png"
    plot_by_kmeans_majority_compatibility_png: Path | str = "tsne_by_kmeans_majority_compatibility_alpha50.png"
    plot_by_kmeans_majority_compatibility_alpha70_png: Path | str = "tsne_by_kmeans_majority_compatibility_alpha70.png"
    plot_by_kmeans_majority_compatibility_alpha30_png: Path | str = "tsne_by_kmeans_majority_compatibility_alpha30.png"
    metadata_csv: Path | str = "metadata_tsne.csv"
    report_txt: Path | str = "report.txt"
    split_dirs: dict[str, Path] = field(default_factory=dict)
    split_reports: dict[str, Path] = field(default_factory=dict)


def mean_pool_last_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean-pool per-frame representations to a single vector per example.

    last_hidden: (B, T, H)
    attention_mask: (B, T) where 1=valid, 0=padding
    returns: (B, H)
    """
    if attention_mask is None:
        return last_hidden.mean(dim=1)

    mask = attention_mask.to(last_hidden.dtype).unsqueeze(-1)  # (B, T, 1)
    summed = (last_hidden * mask).sum(dim=1)                   # (B, H)
    denom = mask.sum(dim=1).clamp_min(1.0)                     # (B, 1)
    return summed / denom


def _build_style_map(unique_labels: list[str]) -> dict[str, tuple[Any, str]]:
    """Map each label to a visually distinct (color, marker) style."""
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "H", "p", "8"]
    cmaps = ["tab20", "tab20b", "tab20c", "Set3", "Dark2", "Set1", "Set2", "Accent", "Paired"]
    colors: list[Any] = []
    for cmap_name in cmaps:
        cmap = plt.get_cmap(cmap_name)
        for idx in range(cmap.N):
            colors.append(cmap(idx))

    if not colors:
        colors = ["C0"]

    styles: dict[str, tuple[Any, str]] = {}
    n_colors = len(colors)
    for i, label in enumerate(unique_labels):
        color = colors[i % n_colors]
        marker = markers[(i // n_colors) % len(markers)]
        styles[label] = (color, marker)
    return styles


def plot_points(
    xy: np.ndarray,
    labels: list[str],
    title: str,
    out_png: Path,
    max_legend_items: int = 40,
    legend: bool = True,
    show: bool = True,
) -> None:
    """Scatter-plot 2D points and color/shape by label."""
    plt.figure(figsize=(10, 8))
    labels_arr = np.array(labels)
    unique_labels = sorted(set(labels))
    style_map = _build_style_map(unique_labels)

    for lab in unique_labels:
        idx = labels_arr == lab
        color, marker = style_map[lab]
        plt.scatter(
            xy[idx, 0],
            xy[idx, 1],
            s=16,
            alpha=0.75,
            color=color,
            marker=marker,
            linewidths=0.3,
            edgecolors="black",
            label=lab,
        )

    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")

    if legend:
        n_cols = 1
        if len(unique_labels) > max_legend_items:
            n_cols = 2
        if len(unique_labels) > max_legend_items * 2:
            n_cols = 3
        plt.legend(markerscale=1.6, fontsize=7, loc="best", ncol=n_cols)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    if show:
        plt.show()
    plt.close()


def plot_kmeans_majority_compatibility(
    xy: np.ndarray,
    cluster_named_labels: list[str],
    incompatible_mask: np.ndarray,
    title: str,
    out_png: Path,
    incompatible_alpha: float = 0.50,
    max_legend_items: int = 40,
    legend: bool = True,
    show: bool = True,
) -> None:
    """Plot cluster-majority compatibility; incompatible points are always red."""
    plt.figure(figsize=(10, 8))
    labels_arr = np.array(cluster_named_labels)
    incompatible = np.asarray(incompatible_mask, dtype=bool)

    compatible_labels = labels_arr[~incompatible].tolist()
    unique_compatible = sorted(set(compatible_labels))
    style_map = _build_style_map(unique_compatible)

    for lab in unique_compatible:
        idx = (labels_arr == lab) & (~incompatible)
        color, marker = style_map[lab]
        plt.scatter(
            xy[idx, 0],
            xy[idx, 1],
            s=16,
            alpha=0.75,
            color=color,
            marker=marker,
            linewidths=0.3,
            edgecolors="black",
            label=lab,
        )

    if incompatible.any():
        plt.scatter(
            xy[incompatible, 0],
            xy[incompatible, 1],
            s=26,
            alpha=float(incompatible_alpha),
            color="red",
            marker="X",
            linewidths=0.5,
            edgecolors="black",
            label="INCOMPATIBLE (label != cluster majority)",
        )

    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")

    if legend:
        n_cols = 1
        total_items = len(unique_compatible) + (1 if incompatible.any() else 0)
        if total_items > max_legend_items:
            n_cols = 2
        if total_items > max_legend_items * 2:
            n_cols = 3
        plt.legend(markerscale=1.6, fontsize=7, loc="best", ncol=n_cols)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    if show:
        plt.show()
    plt.close()


def knn_probe(X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 42) -> Tuple[float, np.ndarray, int]:
    """kNN probe with robust CV and per-sample predictions for visualization."""
    y = np.asarray(y)
    n_classes = len(set(y.tolist()))
    if n_classes < 2:
        return float("nan"), y, 1

    # keep k sane for small datasets / many classes
    k_eff = min(k, max(1, len(y) // n_classes))
    clf = KNeighborsClassifier(n_neighbors=k_eff)
    _, counts = np.unique(y, return_counts=True)
    n_splits = int(min(5, counts.min()))

    if n_splits < 2:
        clf.fit(X, y)
        preds = clf.predict(X)
        return float((preds == y).mean()), preds, k_eff

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = cross_val_predict(clf, X, y, cv=cv)
    return float((preds == y).mean()), preds, k_eff


def run_kmeans(X: np.ndarray, n_clusters: int, seed: int = 42) -> Tuple[np.ndarray, float]:
    """Run KMeans and return cluster IDs and inertia."""
    n_clusters_eff = max(1, min(int(n_clusters), len(X)))
    model = KMeans(n_clusters=n_clusters_eff, random_state=seed, n_init="auto")
    cluster_ids = model.fit_predict(X)
    return cluster_ids, float(model.inertia_)


def _resolve_config_and_paths(exp: TsneExp) -> Tuple[Dict[str, Any], Path, str, Path, str]:
    """Load YAML + overrides and resolve model_dir/output_dir like the minimal script.

    Returns:
      config_dict, model_dir, run_id, out_dir, model_id
    """
    config = load_yaml(exp.config_path)
    overrides = parse_overrides(exp.overrides)
    config = merge_dicts(config, overrides)

    run_name = get_nested(config, "run_name")
    if not run_name:
        raise ValueError("Missing 'run_name' in config.")
    run_name = str(run_name)

    base_output_dir = get_nested(config, "output_dir")
    if not base_output_dir:
        raise ValueError("Missing 'output_dir' in config.")
    base_output_dir = Path(str(base_output_dir))

    base_model_dir = get_nested(config, "save_dir")
    if not base_model_dir:
        raise ValueError("Missing 'save_dir' in config.")
    base_model_dir = Path(str(base_model_dir))

    model_id = str(get_nested(config, "model.id"))

    if exp.model_dir is not None:
        model_dir_input = Path(exp.model_dir)
        if model_dir_input.is_absolute() or model_dir_input.exists():
            model_dir = model_dir_input
        else:
            candidate_under_save_dir = base_model_dir / model_dir_input
            if candidate_under_save_dir.exists():
                model_dir = candidate_under_save_dir
            else:
                raise FileNotFoundError(
                    "Could not resolve --model_dir. Tried:\n"
                    f"  - {model_dir_input}\n"
                    f"  - {candidate_under_save_dir}\n"
                    "Pass an absolute path, a path relative to CWD, or a run folder name under save_dir."
                )
        run_id = model_dir.name
    else:
        candidates = [p for p in base_model_dir.glob(f"{run_name}_*") if p.is_dir()]
        if not candidates:
            raise FileNotFoundError(
                f"No saved runs found in {base_model_dir} matching {run_name}_*.\n"
                f"Pass model_dir explicitly or check your save_dir/run_name."
            )
        model_dir = max(candidates, key=lambda p: p.stat().st_mtime)
        run_id = model_dir.name

    # Store artifacts under <output_dir>/<run_id>/tsne_analysis/<tag>/
    out_dir = base_output_dir / run_id / "tsne_analysis" / exp.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # store resolved model_dir into config
    config = merge_dicts(config, {"model_dir": str(model_dir)})
    return config, model_dir, run_id, out_dir, model_id


def extract_embeddings_and_metadata(
    exp: TsneExp,
    split: str,
    show_plots: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame, Path]:
    """Extract last-layer pooled embeddings and associated metadata.

    Returns: (X, df, out_dir)
    """
    set_seed(exp.seed)
    config, model_dir, _run_id, out_dir, model_id = _resolve_config_and_paths(exp)

    # Prefer loading feature extractor from the fine-tuned directory
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            str(model_dir),
            return_attention_mask=True,
        )
    except Exception:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_id,
            return_attention_mask=True,
        )

    prepared_ds = prepare_encoded_datasets(
        config=config,
        feature_extractor=feature_extractor,
        seed=exp.seed,
    )
    if split == "train":
        ds = prepared_ds.train_dataset
    elif split == "eval":
        ds = prepared_ds.eval_dataset
    else:
        raise ValueError(f"Unsupported split '{split}'. Expected one of: train, eval.")
    n = min(len(ds), exp.max_items)
    ds = ds.select(range(n))

    collator = AudioDataCollator(
        feature_extractor=feature_extractor,
        model_input_name=prepared_ds.model_input_name,
    )

    # Load fine-tuned model with hidden states
    cfg = AutoConfig.from_pretrained(str(model_dir))
    cfg.output_hidden_states = True
    model = AutoModelForAudioClassification.from_pretrained(str(model_dir), config=cfg)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    speaker_col = str(get_nested(config, "data.speaker_column"))
    label_col = str(get_nested(config, "data.label_column"))

    embs: list[np.ndarray] = []
    speakers: list[str] = []
    labels: list[str] = []

    total_batches = (len(ds) + exp.batch_size - 1) // exp.batch_size
    for start in tqdm(
        range(0, len(ds), exp.batch_size),
        total=total_batches,
        desc="Extracting embeddings",
        unit="batch",
        disable=not sys.stderr.isatty(),
    ):
        items = [ds[i] for i in range(start, min(start + exp.batch_size, len(ds)))]

        speakers.extend([str(it.get(speaker_col, "NA")) for it in items])
        labels.extend([str(it.get(label_col, "NA")) for it in items])

        batch = collator(items) # because speech segment lengths can vary, we need to collate them into a batch with padding
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, return_dict=True)

        last_hidden = out.hidden_states[-1]
        attn = batch.get("attention_mask", None) # (B, T), where mask = 1 for real samples, 0 for padding.
        # Align attention mask length to last_hidden's time dimension (T_hidden)
        if attn is not None:
            T_hidden = last_hidden.shape[1]
            if attn.shape[1] != T_hidden:
                # Try common backbone locations for wav2vec2-style models
                backbone = None
                for attr in ["wav2vec2", "model", "hubert", "mms"]:
                    if hasattr(model, attr):
                        backbone = getattr(model, attr)
                        break

                if backbone is not None:
                    if hasattr(backbone, "_get_feature_vector_attention_mask"):
                        attn = backbone._get_feature_vector_attention_mask(T_hidden, attn)
                    elif hasattr(backbone, "get_feature_vector_attention_mask"):
                        attn = backbone.get_feature_vector_attention_mask(T_hidden, attn)
                # If none of the helpers exist, you can also just set attn=None as a fallback:
                # else:
                #     attn = None
        pooled = mean_pool_last_hidden(last_hidden, attn)
        embs.append(pooled.detach().cpu().numpy())

    X = np.concatenate(embs, axis=0)
    df = pd.DataFrame({"speaker_id": speakers, "label": labels})
    return X, df, out_dir


def run_tsne_analysis(
    exp: TsneExp,
    show_plots: bool = False,
    legend: bool = True,
) -> TsneRunPaths:
    """End-to-end for all splits: embeddings -> t-SNE -> kNN -> KMeans -> plots -> reports."""
    legend = True
    split_names = ["train", "eval"]
    out_dir: Optional[Path] = None
    split_dirs: dict[str, Path] = {}
    split_reports: dict[str, Path] = {}
    split_metrics: dict[str, dict[str, float | int]] = {}
    split_artifacts: dict[str, dict[str, Path]] = {}

    for split_name in split_names:
        X, df, resolved_out_dir = extract_embeddings_and_metadata(exp, split=split_name)
        out_dir = resolved_out_dir

        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        split_dirs[split_name] = split_dir

        embeddings_path = split_dir / "last_layer_embeddings.npy"
        metadata_path = split_dir / "metadata.csv"
        np.save(embeddings_path, X)
        df.to_csv(metadata_path, index=False)

        if len(X) < 2:
            raise ValueError(f"Split '{split_name}' has only {len(X)} item(s); need at least 2 for t-SNE.")

        # t-SNE
        if exp.tsne_perplexity is None:
            perplexity = min(30, max(5, (len(X) - 1) // 3))
        else:
            perplexity = int(exp.tsne_perplexity)
        perplexity = min(perplexity, max(1, len(X) - 1))

        tsne = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=exp.seed,
        )
        X2 = tsne.fit_transform(X)
        tsne_path = split_dir / "tsne_2d.npy"
        np.save(tsne_path, X2)

        # kNN probes + visualization labels
        label_acc, label_preds, knn_k_eff_label = knn_probe(
            X, df["label"].astype(str).values, k=exp.knn_k, seed=exp.seed
        )
        speaker_acc, speaker_preds, knn_k_eff_speaker = knn_probe(
            X, df["speaker_id"].astype(str).values, k=exp.knn_k, seed=exp.seed
        )

        # KMeans
        auto_k = int(max(1, df["label"].astype(str).nunique()))
        kmeans_k = exp.kmeans_k if exp.kmeans_k is not None else auto_k
        kmeans_cluster_ids, kmeans_inertia = run_kmeans(X, n_clusters=kmeans_k, seed=exp.seed)
        kmeans_path = split_dir / "kmeans_labels.npy"
        np.save(kmeans_path, kmeans_cluster_ids)

        # Name each cluster by its majority ground-truth label.
        labels_str = df["label"].astype(str)
        ctab_majority = pd.crosstab(pd.Series(kmeans_cluster_ids, name="cluster_id"), labels_str)
        majority_by_cluster: dict[int, str] = {}
        for cluster_id, row in ctab_majority.iterrows():
            row_sorted = row.sort_values(ascending=False)
            majority_label = str(row_sorted.index[0]) if len(row_sorted.index) > 0 else "NA"
            majority_by_cluster[int(cluster_id)] = majority_label

        cluster_majority_labels = [majority_by_cluster[int(cid)] for cid in kmeans_cluster_ids.tolist()]
        cluster_named = [
            f"{majority_by_cluster[int(cid)]} (c{int(cid):02d})"
            for cid in kmeans_cluster_ids.tolist()
        ]
        incompatible_mask = labels_str.values != np.asarray(cluster_majority_labels, dtype=str)

        # Save t-SNE coords + metadata for easy plotting/comparison
        df_tsne = df.copy()
        df_tsne["tsne_x"] = X2[:, 0]
        df_tsne["tsne_y"] = X2[:, 1]
        df_tsne["knn_pred_label"] = label_preds.astype(str)
        df_tsne["knn_pred_speaker"] = speaker_preds.astype(str)
        df_tsne["kmeans_cluster_id"] = kmeans_cluster_ids.astype(int)
        df_tsne["kmeans_cluster_majority_label"] = cluster_majority_labels
        df_tsne["kmeans_cluster"] = cluster_named
        df_tsne["kmeans_majority_compatible"] = ~incompatible_mask
        metadata_tsne_path = split_dir / "metadata_tsne.csv"
        df_tsne.to_csv(metadata_tsne_path, index=False)

        # Plots (legend always on)
        plot_by_label_png = split_dir / "tsne_by_label.png"
        plot_by_speaker_png = split_dir / "tsne_by_speaker.png"
        plot_by_knn_label_png = split_dir / "tsne_by_knn_label.png"
        plot_by_kmeans_png = split_dir / "tsne_by_kmeans.png"
        plot_by_kmeans_majority_compatibility_alpha70_png = (
            split_dir / "tsne_by_kmeans_majority_compatibility_alpha70.png"
        )
        plot_by_kmeans_majority_compatibility_png = (
            split_dir / "tsne_by_kmeans_majority_compatibility_alpha50.png"
        )
        plot_by_kmeans_majority_compatibility_alpha30_png = (
            split_dir / "tsne_by_kmeans_majority_compatibility_alpha30.png"
        )
        plot_points(
            X2,
            df["label"].astype(str).tolist(),
            title=f"t-SNE (last layer) by label [{split_name}] ({exp.tag})",
            out_png=plot_by_label_png,
            legend=legend,
            show=show_plots,
        )
        plot_points(
            X2,
            df["speaker_id"].astype(str).tolist(),
            title=f"t-SNE (last layer) by speaker [{split_name}] ({exp.tag})",
            out_png=plot_by_speaker_png,
            max_legend_items=50,
            legend=legend,
            show=show_plots,
        )
        plot_points(
            X2,
            df_tsne["knn_pred_label"].astype(str).tolist(),
            title=f"t-SNE by kNN predicted label [{split_name}] ({exp.tag})",
            out_png=plot_by_knn_label_png,
            legend=legend,
            show=show_plots,
        )
        plot_points(
            X2,
            df_tsne["kmeans_cluster"].astype(str).tolist(),
            title=f"t-SNE by KMeans cluster [{split_name}] ({exp.tag})",
            out_png=plot_by_kmeans_png,
            legend=legend,
            show=show_plots,
        )
        plot_kmeans_majority_compatibility(
            X2,
            df_tsne["kmeans_cluster"].astype(str).tolist(),
            incompatible_mask=incompatible_mask,
            title=f"t-SNE by KMeans majority compatibility (alpha=0.70) [{split_name}] ({exp.tag})",
            out_png=plot_by_kmeans_majority_compatibility_alpha70_png,
            incompatible_alpha=0.70,
            legend=legend,
            show=show_plots,
        )
        plot_kmeans_majority_compatibility(
            X2,
            df_tsne["kmeans_cluster"].astype(str).tolist(),
            incompatible_mask=incompatible_mask,
            title=f"t-SNE by KMeans majority compatibility (alpha=0.50) [{split_name}] ({exp.tag})",
            out_png=plot_by_kmeans_majority_compatibility_png,
            incompatible_alpha=0.50,
            legend=legend,
            show=show_plots,
        )
        plot_kmeans_majority_compatibility(
            X2,
            df_tsne["kmeans_cluster"].astype(str).tolist(),
            incompatible_mask=incompatible_mask,
            title=f"t-SNE by KMeans majority compatibility (alpha=0.30) [{split_name}] ({exp.tag})",
            out_png=plot_by_kmeans_majority_compatibility_alpha30_png,
            incompatible_alpha=0.30,
            legend=legend,
            show=show_plots,
        )

        cluster_counts = df_tsne["kmeans_cluster"].value_counts().sort_index()
        top_labels_lines: list[str] = []
        ctab = pd.crosstab(df_tsne["kmeans_cluster"], df_tsne["label"])
        for cluster_name in ctab.index:
            top = ctab.loc[cluster_name].sort_values(ascending=False).head(3)
            top_labels = ", ".join([f"{lab}:{int(cnt)}" for lab, cnt in top.items() if int(cnt) > 0])
            top_labels_lines.append(f"{cluster_name}: {top_labels if top_labels else 'NA'}")

        # Per-split report
        report_path = split_dir / "report.txt"
        lines = [
            f"tag: {exp.tag}",
            f"split: {split_name}",
            f"N: {len(X)}",
            f"emb_dim: {X.shape[1]}",
            f"tsne_perplexity: {perplexity}",
            f"knn_k_requested: {exp.knn_k}",
            f"knn_k_effective_label: {knn_k_eff_label}",
            f"knn_k_effective_speaker: {knn_k_eff_speaker}",
            f"knn_acc_label: {label_acc:.4f}",
            f"knn_acc_speaker: {speaker_acc:.4f}",
            f"kmeans_k_requested: {kmeans_k}",
            f"kmeans_k_effective: {len(np.unique(kmeans_cluster_ids))}",
            f"kmeans_inertia: {kmeans_inertia:.4f}",
            f"kmeans_majority_incompatible_count: {int(incompatible_mask.sum())}",
            f"kmeans_majority_incompatible_rate: {(float(incompatible_mask.mean()) * 100.0):.2f}%",
            "",
            "kmeans_cluster_counts:",
        ]
        lines.extend([f"{name}: {int(count)}" for name, count in cluster_counts.items()])
        lines.extend(["", "kmeans_top_labels_per_cluster:"])
        lines.extend(top_labels_lines)
        lines.append("")
        report_path.write_text("\n".join(lines))

        split_reports[split_name] = report_path
        split_metrics[split_name] = {
            "N": int(len(X)),
            "emb_dim": int(X.shape[1]),
            "tsne_perplexity": int(perplexity),
            "knn_acc_label": float(label_acc),
            "knn_acc_speaker": float(speaker_acc),
            "kmeans_k_effective": int(len(np.unique(kmeans_cluster_ids))),
            "kmeans_inertia": float(kmeans_inertia),
            "kmeans_majority_incompatible_count": int(incompatible_mask.sum()),
            "kmeans_majority_incompatible_rate": float(incompatible_mask.mean()),
        }
        split_artifacts[split_name] = {
            "metadata_csv": metadata_tsne_path,
            "embeddings_npy": embeddings_path,
            "tsne_npy": tsne_path,
            "kmeans_npy": kmeans_path,
            "plot_by_label_png": plot_by_label_png,
            "plot_by_speaker_png": plot_by_speaker_png,
            "plot_by_knn_label_png": plot_by_knn_label_png,
            "plot_by_kmeans_png": plot_by_kmeans_png,
            "plot_by_kmeans_majority_compatibility_png": plot_by_kmeans_majority_compatibility_png,
            "plot_by_kmeans_majority_compatibility_alpha70_png": plot_by_kmeans_majority_compatibility_alpha70_png,
            "plot_by_kmeans_majority_compatibility_alpha30_png": plot_by_kmeans_majority_compatibility_alpha30_png,
        }

    if out_dir is None:
        raise RuntimeError("Failed to resolve output directory for t-SNE analysis.")

    # Root summary report (keeps compare helper compatible via knn_acc_* lines from eval split)
    summary_report_path = out_dir / "report.txt"
    summary_lines = [
        f"tag: {exp.tag}",
        "splits: train, eval",
    ]
    if "eval" in split_metrics:
        summary_lines.extend(
            [
                f"knn_acc_label: {split_metrics['eval']['knn_acc_label']:.4f}",
                f"knn_acc_speaker: {split_metrics['eval']['knn_acc_speaker']:.4f}",
            ]
        )
    summary_lines.append("")
    for split_name in split_names:
        if split_name not in split_metrics:
            continue
        m = split_metrics[split_name]
        summary_lines.extend(
            [
                f"[{split_name}]",
                f"N: {m['N']}",
                f"emb_dim: {m['emb_dim']}",
                f"tsne_perplexity: {m['tsne_perplexity']}",
                f"knn_acc_label: {m['knn_acc_label']:.4f}",
                f"knn_acc_speaker: {m['knn_acc_speaker']:.4f}",
                f"kmeans_k_effective: {m['kmeans_k_effective']}",
                f"kmeans_inertia: {m['kmeans_inertia']:.4f}",
                f"kmeans_majority_incompatible_count: {m['kmeans_majority_incompatible_count']}",
                f"kmeans_majority_incompatible_rate: {(m['kmeans_majority_incompatible_rate'] * 100.0):.2f}%",
                "",
            ]
        )
    summary_report_path.write_text("\n".join(summary_lines))

    representative_split = "eval" if "eval" in split_artifacts else next(iter(split_artifacts))
    rep = split_artifacts[representative_split]
    return TsneRunPaths(
        out_dir=out_dir,
        metadata_csv=rep["metadata_csv"],
        embeddings_npy=rep["embeddings_npy"],
        tsne_npy=rep["tsne_npy"],
        kmeans_npy=rep["kmeans_npy"],
        plot_by_label_png=rep["plot_by_label_png"],
        plot_by_speaker_png=rep["plot_by_speaker_png"],
        plot_by_knn_label_png=rep["plot_by_knn_label_png"],
        plot_by_kmeans_png=rep["plot_by_kmeans_png"],
        plot_by_kmeans_majority_compatibility_png=rep["plot_by_kmeans_majority_compatibility_png"],
        plot_by_kmeans_majority_compatibility_alpha70_png=rep["plot_by_kmeans_majority_compatibility_alpha70_png"],
        plot_by_kmeans_majority_compatibility_alpha30_png=rep["plot_by_kmeans_majority_compatibility_alpha30_png"],
        report_txt=summary_report_path,
        split_dirs=split_dirs,
        split_reports=split_reports,
    )


def compare_two_runs(
    run_a_dir: Path,
    run_b_dir: Path,
    out_dir: Path,
) -> Path:
    """Tiny comparison helper.

    Writes compare_report.txt with deltas of kNN probe accuracies if both reports exist.
    (Keeps it intentionally simple.)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "compare_report.txt"

    def _read_knn(path: Path) -> Dict[str, float]:
        d: Dict[str, float] = {}
        if not path.exists():
            return d
        for line in path.read_text().splitlines():
            if line.startswith("knn_acc_label:"):
                d["label"] = float(line.split(":", 1)[1].strip())
            if line.startswith("knn_acc_speaker:"):
                d["speaker"] = float(line.split(":", 1)[1].strip())
        return d

    a = _read_knn(run_a_dir / "report.txt")
    b = _read_knn(run_b_dir / "report.txt")

    lines = [
        f"run_a: {run_a_dir}",
        f"run_b: {run_b_dir}",
        "",
    ]
    if a and b:
        lines += [
            f"knn_acc_label  (a): {a.get('label', float('nan')):.4f}",
            f"knn_acc_label  (b): {b.get('label', float('nan')):.4f}",
            f"delta_label (b-a): {(b.get('label', float('nan')) - a.get('label', float('nan'))):.4f}",
            "",
            f"knn_acc_speaker (a): {a.get('speaker', float('nan')):.4f}",
            f"knn_acc_speaker (b): {b.get('speaker', float('nan')):.4f}",
            f"delta_speaker(b-a): {(b.get('speaker', float('nan')) - a.get('speaker', float('nan'))):.4f}",
            "",
        ]
    else:
        lines.append("Could not read probe numbers from one or both report.txt files.")

    report.write_text("\n".join(lines))
    return report
