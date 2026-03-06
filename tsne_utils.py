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
from typing import Any, Dict, Optional, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, cross_val_score
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
    split: str = "eval"  # "train" or "eval"
    max_items: int = 2000
    batch_size: int = 32
    seed: int = 42

    # Output naming
    tag: str = "default"  # used as a subfolder under tsne_analysis/

    # Analysis knobs
    knn_k: int = 5
    tsne_perplexity: Optional[int] = None  # if None -> auto heuristic


@dataclass
class TsneRunPaths:
    """Where artifacts for one run were saved."""

    out_dir: Path
    embeddings_npy: Path
    tsne_npy: Path
    plot_by_label_png: Path | str = "tsne_by_label.png"
    plot_by_speaker_png: Path | str = "tsne_by_speaker.png"
    metadata_csv: Path | str = "metadata_tsne.csv"


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


def plot_points(
    xy: np.ndarray,
    labels: list[str],
    title: str,
    out_png: Path,
    max_legend_items: int = 40,
    legend: bool = False,
    show: bool = True,
) -> None:
    """Scatter-plot 2D points and color by label.

    If show=True, will display in notebooks (plt.show()).
    """
    plt.figure(figsize=(10, 8))
    labels_arr = np.array(labels)
    unique_labels = sorted(set(labels))

    for lab in unique_labels:
        idx = labels_arr == lab
        plt.scatter(xy[idx, 0], xy[idx, 1], s=10, alpha=0.7, label=lab)

    plt.title(title)
    # plt.xlabel("t-SNE dim 1")
    # plt.ylabel("t-SNE dim 2")

    if legend:
        if len(unique_labels) <= max_legend_items:
            plt.legend(markerscale=2, fontsize=8, loc="best")
        else:
            plt.legend([], [], frameon=False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    if show:
        plt.show()
    plt.close()


def knn_probe_accuracy(X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 42) -> float:
    """Quick decodability probe: kNN accuracy with 5-fold stratified CV."""
    y = np.asarray(y)
    n_classes = len(set(y.tolist()))
    if n_classes < 2:
        return float("nan")

    # keep k sane for small datasets / many classes
    k_eff = min(k, max(1, len(y) // n_classes))
    clf = KNeighborsClassifier(n_neighbors=k_eff)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=cv)
    return float(scores.mean())


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

    model_id = str(get_nested(config, "model.id", "facebook/mms-300m"))

    if exp.model_dir is not None:
        model_dir = Path(exp.model_dir)
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
    show_plots: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame, Path]:
    """Extract last-layer pooled embeddings and associated metadata.

    Returns: (X, df, out_dir)
    """
    set_seed(exp.seed)
    config, _model_dir, _run_id, out_dir, model_id = _resolve_config_and_paths(exp)

    # Prefer loading feature extractor from the fine-tuned directory
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            config["model_dir"],
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
    ds = prepared_ds.train_dataset if exp.split == "train" else prepared_ds.eval_dataset
    n = min(len(ds), exp.max_items)
    ds = ds.select(range(n))

    collator = AudioDataCollator(
        feature_extractor=feature_extractor,
        model_input_name=prepared_ds.model_input_name,
    )

    # Load fine-tuned model with hidden states
    cfg = AutoConfig.from_pretrained(config["model_dir"])
    cfg.output_hidden_states = True
    model = AutoModelForAudioClassification.from_pretrained(config["model_dir"], config=cfg)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    speaker_col = str(get_nested(config, "data.speaker_column", "speaker_id"))
    label_col = str(get_nested(config, "data.label_column", "language"))

    embs: list[np.ndarray] = []
    speakers: list[str] = []
    labels: list[str] = []

    for start in tqdm(range(0, len(ds), exp.batch_size), desc="Extracting embeddings"):
        items = [ds[i] for i in range(start, min(start + exp.batch_size, len(ds)))]

        speakers.extend([str(it.get(speaker_col, "NA")) for it in items])
        labels.extend([str(it.get(label_col, "NA")) for it in items])

        batch = collator(items) # because speech segment lengths can vary, we need to collate them into a batch with padding
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, return_dict=True)

        print(f"Processed {len(embs) + len(items)}/{len(ds)} items", end="\r")
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
    legend: bool = False,
) -> TsneRunPaths:
    """End-to-end: extract embeddings -> t-SNE -> plots -> report -> save artifacts."""
    X, df, out_dir = extract_embeddings_and_metadata(exp)

    # Save embeddings + metadata (these are the *reusable* artifacts)
    embeddings_path = out_dir / "last_layer_embeddings.npy"
    metadata_path = out_dir / "metadata.csv"
    np.save(embeddings_path, X)
    df.to_csv(metadata_path, index=False)

    # t-SNE
    if exp.tsne_perplexity is None:
        perplexity = min(30, max(5, (len(X) - 1) // 3))
    else:
        perplexity = int(exp.tsne_perplexity)
    perplexity = min(perplexity, max(5, len(X) - 1))

    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=exp.seed,
    )
    X2 = tsne.fit_transform(X)
    tsne_path = out_dir / "tsne_2d.npy"
    np.save(tsne_path, X2)

    # Save t-SNE coords alongside metadata for easy plotting/comparison
    df_tsne = df.copy()
    df_tsne["tsne_x"] = X2[:, 0]
    df_tsne["tsne_y"] = X2[:, 1]
    df_tsne.to_csv(out_dir / "metadata_tsne.csv", index=False)

    # Plots
    plot_by_label_png = out_dir / "tsne_by_label.png"
    plot_by_speaker_png = out_dir / "tsne_by_speaker.png"
    plot_points(
        X2,
        df["label"].astype(str).tolist(),
        title=f"t-SNE (last layer) colored by label [{exp.split}] ({exp.tag})",
        out_png=plot_by_label_png,
        legend=legend,
        show=show_plots,
    )
    plot_points(
        X2,
        df["speaker_id"].astype(str).tolist(),
        title=f"t-SNE (last layer) colored by speaker [{exp.split}] ({exp.tag})",
        out_png=plot_by_speaker_png,
        max_legend_items=50,
        legend=legend,
        show=show_plots,
    )

    # Report (kNN probes)
    speaker_acc = knn_probe_accuracy(X, df["speaker_id"].astype(str).values, k=exp.knn_k, seed=exp.seed)
    label_acc = knn_probe_accuracy(X, df["label"].astype(str).values, k=exp.knn_k, seed=exp.seed)
    report_path = out_dir / "report.txt"
    report_path.write_text(
        "\n".join(
            [
                f"tag: {exp.tag}",
                f"split: {exp.split}",
                f"N: {len(X)}",
                f"emb_dim: {X.shape[1]}",
                f"tsne_perplexity: {perplexity}",
                f"knn_k: {exp.knn_k}",
                f"knn_acc_label: {label_acc:.4f}",
                f"knn_acc_speaker: {speaker_acc:.4f}",
                "",
            ]
        )
    )

    return TsneRunPaths(
        out_dir=out_dir,
        metadata_csv=metadata_path,
        embeddings_npy=embeddings_path,
        tsne_npy=tsne_path,
        plot_by_label_png=plot_by_label_png,
        plot_by_speaker_png=plot_by_speaker_png,
        report_txt=report_path,
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
