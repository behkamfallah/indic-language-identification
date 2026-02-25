# analyze_tsne_minimal.py
"""
MINIMAL t-SNE analysis for "last-layer representations".

What this script does:
1) Load your YAML config (same as training uses).
2) Build the dataset with the existing pipeline (same encoding).
3) Load a fine-tuned model from --model_dir.
4) Extract last hidden layer, mean-pool over time -> one vector per utterance.
5) Run t-SNE -> 2D points.
6) Save:
   - tsne_by_language.png
   - tsne_by_speaker.png
   - metadata_tsne.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    set_seed,
)

# These are your repo utilities (we reuse them to keep changes minimal).
from config_utils import load_yaml, get_nested, merge_dicts, parse_overrides
from dataset_utils import prepare_encoded_datasets
from trainer_utils import AudioDataCollator


def mean_pool_last_hidden(last_hidden: torch.Tensor, attention_mask: torch.Tensor | None= None) -> torch.Tensor:
    """
    last_hidden: (B, T, H)
    attention_mask: (B, T) where 1 = valid, 0 = padding (optional)
    return: (B, H)
    """
    if attention_mask is None:
        return last_hidden.mean(dim=1)

    mask = attention_mask.to(last_hidden.dtype).unsqueeze(-1)  # (B, T, 1)
    summed = (last_hidden * mask).sum(dim=1)  # (B, H)
    denom = mask.sum(dim=1).clamp_min(1.0)    # (B, 1)
    return summed / denom


def plot_points(
    xy: np.ndarray,
    labels: list[str],
    title: str,
    out_png: Path,
    max_legend_items: int = 40,
    legend: bool = False,
) -> None:
    """
    Scatter-plot 2D points and color by label.

    Note:
    - Legends can explode if you have many speakers.
    - We cap legend size and omit it if too large.
    """
    plt.figure(figsize=(10, 8))

    labels_arr = np.array(labels)
    unique_labels = sorted(set(labels))

    # Plot each group separately so matplotlib can build a legend.
    for lab in unique_labels:
        idx = labels_arr == lab
        plt.scatter(xy[idx, 0], xy[idx, 1], s=10, alpha=0.7, label=lab)

    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")

    # Keep legend only if it won't be unreadable.
    if legend:
        if len(unique_labels) <= max_legend_items:
            plt.legend(markerscale=2, fontsize=8, loc="best")
        else:
            plt.legend([], [], frameon=False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def build_tsne_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="t-SNE analysis of audio model representations")
    ap.add_argument("--config", type=Path, required=True, help="YAML config used for the run")
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        help='Config override "key=value". Can repeat: --override a.b=1 --override c=2',
    )
    ap.add_argument("--model_dir", type=Path, default=None, help="Explicit HF model dir. If omitted, auto-pick latest run.")
    ap.add_argument("--split", choices=["train", "eval"], default="eval", help="Which dataset split to analyze")
    ap.add_argument("--max_items", type=int, default=2000, help="Cap on number of items for t-SNE (for speed)")
    ap.add_argument("--batch_size", type=int, default=8, help="Batch size for embedding extraction")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return ap


def main() -> None:
    # -----------------------
    # 1) CLI args
    # -----------------------
    ap = build_tsne_arg_parser()
    args = ap.parse_args()

    # Seed controls t-SNE randomness + any dataset shuffling done inside utilities.
    set_seed(args.seed)

    # -----------------------
    # 2) Load config + merge overrides (same style as training)
    # -----------------------
    config = load_yaml(args.config)
    overrides = parse_overrides(args.override)
    config = merge_dicts(config, overrides)

    # Required fields (training expects these to exist too)
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

    # YAML contains model.id used in training; we only need it as fallback
    model_id = str(get_nested(config, "model.id", "facebook/mms-300m"))

    # -----------------------
    # 3) Resolve model_dir consistent with training naming:
    # train_model.py uses run_id = f"{run_name}_{timestamp}" and saves under save_dir/run_id :contentReference[oaicite:1]{index=1}
    # -----------------------
    if args.model_dir is not None:
        model_dir = args.model_dir
        run_id = model_dir.name
    else:
        # auto-pick latest folder matching run_name*
        candidates = [p for p in Path(base_model_dir).glob(f"{run_name}*") if p.is_dir()]
        if not candidates:
            raise FileNotFoundError(
                f"No saved runs found in {base_model_dir} matching {run_name}*.\n"
                f"Pass --model_dir explicitly or check your save_dir/run_name."
            )
        model_dir = max(candidates, key=lambda p: p.stat().st_mtime)
        run_id = model_dir.name

    # Save output alongside training's run_id structure
    output_dir = f"{base_output_dir}/{run_id}/tsne_analysis"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Store resolved model_dir back into config (dict!)
    config["model_dir"] = str(model_dir)
    # config.merge_dicts(config, {"model_dir": str(model_dir)})

    # Prefer loading FE from the fine-tuned directory
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

    # -----------------------
    # 3) Build datasets exactly like training does
    # -----------------------
    prepared = prepare_encoded_datasets(
        config=config,
        feature_extractor=feature_extractor,
        seed=args.seed,
    )
    ds = prepared.train_dataset if args.split == "train" else prepared.eval_dataset


    # Reduce size for t-SNE speed.
    n = min(len(ds), args.max_items)
    ds = ds.select(range(n))

    # Collator from your repo: turns list of dicts -> padded batch tensors.
    collator = AudioDataCollator(
        feature_extractor=feature_extractor,
        model_input_name=prepared.model_input_name,
    )

    # -----------------------
    # 4) Load fine-tuned model with hidden states enabled
    # -----------------------
    cfg = AutoConfig.from_pretrained(str(config["model_dir"]))
    cfg.output_hidden_states = True  # critical: we need last hidden layer
    model = AutoModelForAudioClassification.from_pretrained(str(config["model_dir"]), config=cfg)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Column names (from YAML, with defaults)
    speaker_col = str(get_nested(config, "data.speaker_column", "speaker_id"))
    label_col = str(get_nested(config, "data.label_column", "language"))

    # -----------------------
    # 5) Extract embeddings (last hidden layer -> mean pooled)
    # -----------------------
    embs = []
    speakers: list[str] = []
    labels: list[str] = []


    # Manual batching keeps the code minimal and avoids Trainer usage.
    for start in range(0, len(ds), args.batch_size):
        # Grab a slice of items (each item is a dict).
        items = [ds[i] for i in range(start, min(start + args.batch_size, len(ds)))]

        # Save metadata BEFORE collator (collator may drop non-tensor fields).
        speakers.extend([str(it.get(speaker_col, "NA")) for it in items])
        labels.extend([str(it.get(label_col, "NA")) for it in items])

        # Collate into padded tensors.
        batch = collator(items)
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, return_dict=True)

        # out.hidden_states is a tuple of layers; last one is what we want.
        last_hidden = out.hidden_states[-1]  # (B, T, H)

        # If provided, attention_mask tells us which time-steps are padding.
        # attn = batch.get("attention_mask", None)

        pooled = mean_pool_last_hidden(last_hidden)  # (B, H)
        embs.append(pooled.detach().cpu().numpy())

    X = np.concatenate(embs, axis=0)  # (N, H)

    # -----------------------
    # 6) t-SNE -> 2D
    # -----------------------
    # Perplexity must be < N; keep it safe.
    # Simple heuristic: <= 30, and not too large for small N.
    perplexity = min(30, max(5, (len(X) - 1) // 3))

    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=args.seed,
    )
    X2 = tsne.fit_transform(X)  # (N, 2)

    # -----------------------
    # 7) Save plots + metadata
    # -----------------------

    df = pd.DataFrame(
        {
            "speaker_id": speakers,
            "label": labels,
            "tsne_x": X2[:, 0],
            "tsne_y": X2[:, 1],
        }
    )
    df.to_csv(output_dir / "metadata_tsne.csv", index=False)

    plot_points(
        X2,
        labels,
        title=f"t-SNE (last layer) colored by label [{args.split}]",
        out_png=output_dir / "tsne_by_label.png",
    )
    plot_points(
        X2,
        speakers,
        title=f"t-SNE (last layer) colored by speaker [{args.split}]",
        out_png=output_dir / "tsne_by_speaker.png",
        max_legend_items=50,  # speakers can be many; keep legend sane
    )

    print(f"Done. Saved to: {output_dir.resolve()}")
    print(f"N={len(X)} | emb_dim={X.shape[1]} | perplexity={perplexity}")


if __name__ == "__main__":
    main()