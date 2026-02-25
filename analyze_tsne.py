# analyze_tsne.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForAudioClassification, set_seed

from config_utils import load_yaml, get_nested
from dataset_utils import prepare_encoded_datasets
from trainer_utils import AudioDataCollator


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """
    hidden: (B, T, H)
    attention_mask: (B, T) with 1 for valid frames, 0 for padding
    returns: (B, H)
    """
    if attention_mask is None:
        return hidden.mean(dim=1)

    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)  # (B, T, 1)
    summed = (hidden * mask).sum(dim=1)                   # (B, H)
    denom = mask.sum(dim=1).clamp_min(1.0)                # (B, 1)
    return summed / denom


def plot_tsne(points_2d: np.ndarray, labels: list[str], title: str, out_png: Path, max_legend: int = 30) -> None:
    plt.figure(figsize=(10, 8))
    labels_arr = np.array(labels)

    # stable ordering for legend
    uniq = sorted(set(labels))
    for lab in uniq:
        idx = labels_arr == lab
        plt.scatter(points_2d[idx, 0], points_2d[idx, 1], s=10, alpha=0.7, label=lab)

    plt.title(title)
    # plt.xlabel("t-SNE dim 1")
    # plt.ylabel("t-SNE dim 2")

    # legends can get huge; cap it
    if len(uniq) <= max_legend:
        plt.legend(markerscale=2, fontsize=8, loc="best")
    else:
        plt.legend([], [], frameon=False)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def knn_probe_accuracy(X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 42) -> float:
    """
    Quick probe: kNN accuracy with 5-fold stratified CV.
    Interpretable for “does embedding separate speakers?”
    """
    n_classes = len(set(y.tolist()))
    if n_classes < 2:
        return float("nan")

    clf = KNeighborsClassifier(n_neighbors=min(k, max(1, len(y) // n_classes)))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=cv)
    return float(scores.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="YAML config used for the run")
    ap.add_argument("--model_dir", type=Path, required=True, help="Path to the *fine-tuned* saved model directory")
    ap.add_argument("--split", type=str, default="eval", choices=["train", "eval"], help="Which split to embed")
    ap.add_argument("--max_items", type=int, default=2000, help="Cap items for speed (t-SNE scales poorly)")
    ap.add_argument("--batch_size", type=int, default=8, help="Batch size for embedding extraction")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=Path, default=Path("./tsne_outputs"))
    args = ap.parse_args()

    config = load_yaml(args.config)
    set_seed(args.seed)

    # Load feature extractor from the fine-tuned directory if available; fallback to model.id
    model_id = str(get_nested(config, "model.id", "facebook/mms-300m"))
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(str(args.model_dir), return_attention_mask=True)
    except Exception:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, return_attention_mask=True)

    prepared = prepare_encoded_datasets(config=config, feature_extractor=feature_extractor, seed=args.seed)
    ds = prepared.train_dataset if args.split == "train" else prepared.eval_dataset

    # Cap dataset size for t-SNE speed
    n = min(len(ds), args.max_items)
    ds = ds.select(range(n))

    collator = AudioDataCollator(feature_extractor=feature_extractor, model_input_name=prepared.model_input_name)

    # Load fine-tuned model (ensure hidden states are returned)
    cfg = AutoConfig.from_pretrained(str(args.model_dir))
    cfg.output_hidden_states = True
    model = AutoModelForAudioClassification.from_pretrained(str(args.model_dir), config=cfg)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    speaker_col = str(get_nested(config, "data.speaker_column", "speaker_id"))
    lang_col = str(get_nested(config, "data.label_column", "language"))

    all_embs: list[np.ndarray] = []
    all_speakers: list[str] = []
    all_langs: list[str] = []

    # Manual batching with your existing collator (minimal coupling)
    for start in range(0, len(ds), args.batch_size):
        items = [ds[i] for i in range(start, min(start + args.batch_size, len(ds)))]

        # keep metadata before collator drops it
        all_speakers.extend([str(it.get(speaker_col, "NA")) for it in items])
        all_langs.extend([str(it.get(lang_col, "NA")) for it in items])

        batch = collator(items)
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, return_dict=True)

        hidden_last = out.hidden_states[-1]  # (B, T, H)
        attn = batch.get("attention_mask", None)
        pooled = mean_pool(hidden_last, attn)  # (B, H)

        all_embs.append(pooled.detach().cpu().numpy())

    X = np.concatenate(all_embs, axis=0)  # (N, H)

    # t-SNE
    perplexity = min(30, max(5, (len(X) - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
    )
    X2 = tsne.fit_transform(X)

    # Save outputs
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "last_layer_embeddings.npy", X)
    np.save(args.out_dir / "tsne_2d.npy", X2)

    df = pd.DataFrame(
        {
            "speaker_id": all_speakers,
            "language": all_langs,
            "tsne_x": X2[:, 0],
            "tsne_y": X2[:, 1],
        }
    )
    df.to_csv(args.out_dir / "metadata_tsne.csv", index=False)

    plot_tsne(X2, all_langs, f"t-SNE (last layer) colored by language [{args.split}]", args.out_dir / "tsne_by_language.png")
    plot_tsne(X2, all_speakers, f"t-SNE (last layer) colored by speaker [{args.split}]", args.out_dir / "tsne_by_speaker.png", max_legend=50)

    # Quick quantitative “speaker leakage” signal
    # encode labels to ints
    spk_vals = pd.Series(all_speakers).astype("category").cat.codes.to_numpy()
    lang_vals = pd.Series(all_langs).astype("category").cat.codes.to_numpy()

    spk_acc = knn_probe_accuracy(X, spk_vals, k=5, seed=args.seed)
    lang_acc = knn_probe_accuracy(X, lang_vals, k=5, seed=args.seed)

    num_spk = len(set(all_speakers))
    num_lang = len(set(all_langs))
    spk_chance = 1.0 / max(1, num_spk)
    lang_chance = 1.0 / max(1, num_lang)

    report = (
        f"Split: {args.split}\n"
        f"N: {len(X)}\n"
        f"Embedding dim: {X.shape[1]}\n"
        f"Speakers: {num_spk} | chance ~ {spk_chance:.3f} | kNN@5 acc ~ {spk_acc:.3f}\n"
        f"Languages: {num_lang} | chance ~ {lang_chance:.3f} | kNN@5 acc ~ {lang_acc:.3f}\n"
        f"Perplexity used: {perplexity}\n"
    )
    (args.out_dir / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved outputs to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()