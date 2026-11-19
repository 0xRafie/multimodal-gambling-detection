"""Train and test the six fusion strategies on SigLIP 2 + Multilingual E5-large embeddings (Table 6).

Each fusion model is trained on the train split with early stopping on validation F1
(patience 5) and evaluated once on the held-out test split.

Example:
    python train_fusion.py                       # all six strategies
    python train_fusion.py --fusions contrastive
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from gamblingdet.config import FUSION_IMAGE_ENCODER, FUSION_TEXT_ENCODER, FUSIONS, SEED
from gamblingdet.data import load_dataset, load_splits
from gamblingdet.embeddings import load_embeddings
from gamblingdet.engine import compute_metrics, make_loader, predict, seed_everything, train_fusion
from gamblingdet.models import FUSION_MODELS, FUSION_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fusions", nargs="+", choices=FUSIONS, default=FUSIONS)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--emb-dir", default="embeddings")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    splits = load_splits(load_dataset(args.data_dir, check_images=False))
    emb_dir = Path(args.emb_dir)
    img = [load_embeddings(emb_dir / f"image_{FUSION_IMAGE_ENCODER}.npz", d["id"]) for d in splits]
    txt = [load_embeddings(emb_dir / f"text_{FUSION_TEXT_ENCODER}.npz", d["id"]) for d in splits]
    y = [d["label"].values for d in splits]
    train_loader = make_loader(img[0], txt[0], labels=y[0], shuffle=True)
    val_loader = make_loader(img[1], txt[1], labels=y[1])
    test_loader = make_loader(img[2], txt[2], labels=y[2])
    img_dim, txt_dim = img[0].shape[1], txt[0].shape[1]

    out = Path(args.out_dir)
    for sub in ("checkpoints", "predictions", "results"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    rows = []
    for fusion in args.fusions:
        seed_everything(args.seed)
        model, history = train_fusion(FUSION_MODELS[fusion](img_dim, txt_dim), train_loader, val_loader, args.device)
        val_metrics = compute_metrics(*predict(model, val_loader, args.device))
        labels, probs = predict(model, test_loader, args.device)
        test_metrics = compute_metrics(labels, probs)

        name = f"fusion_{fusion}"
        torch.save(
            {
                "kind": "fusion",
                "fusion": fusion,
                "image_encoder": FUSION_IMAGE_ENCODER,
                "text_encoder": FUSION_TEXT_ENCODER,
                "img_dim": img_dim,
                "txt_dim": txt_dim,
                "state_dict": model.state_dict(),
                "test_metrics": test_metrics,
            },
            out / "checkpoints" / f"{name}.pt",
        )
        pd.DataFrame({"id": splits[2]["id"], "label": labels, "prob": probs, "pred": (probs > 0.5).astype(int)}).to_csv(
            out / "predictions" / f"{name}.csv", index=False
        )
        result = {
            "experiment": name,
            "fusion": FUSION_NAMES[fusion],
            "val": val_metrics,
            "test": test_metrics,
            "history": history,
        }
        (out / "results" / f"{name}.json").write_text(json.dumps(result, indent=2))
        rows.append({"fusion": FUSION_NAMES[fusion], **test_metrics})
        print(f"{FUSION_NAMES[fusion]:24s} " + "  ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))

    pd.DataFrame(rows).to_csv(out / "results" / "fusion_summary.csv", index=False)


if __name__ == "__main__":
    main()
