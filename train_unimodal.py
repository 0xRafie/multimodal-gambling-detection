"""Train and test the MLP head on one frozen image or text encoder (Tables 4 and 5 of the paper).

Protocol, as in the paper's runs:
  Phase 1  train on train, early stopping on validation loss  -> validation metrics
  Phase 2  retrain from scratch on train+val for 50 epochs     -> test metrics, saved checkpoint

Example:
    python train_unimodal.py --modality image --encoder siglip2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gamblingdet.config import ENCODERS, SEED
from gamblingdet.data import load_dataset, load_splits
from gamblingdet.embeddings import load_embeddings
from gamblingdet.engine import compute_metrics, make_loader, predict, seed_everything, train_unimodal
from gamblingdet.models import MLPClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modality", choices=["image", "text"], required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--emb-dir", default="embeddings")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.encoder not in ENCODERS[args.modality]:
        parser.error(f"choose --encoder from {list(ENCODERS[args.modality])}")

    name = f"{args.modality}_{args.encoder}"
    train, val, test = load_splits(load_dataset(args.data_dir, check_images=False))
    emb_path = Path(args.emb_dir) / f"{name}.npz"
    x_train, x_val, x_test = (load_embeddings(emb_path, d["id"]) for d in (train, val, test))
    y_train, y_val, y_test = (d["label"].values for d in (train, val, test))
    dim = x_train.shape[1]

    seed_everything(args.seed)
    model, history = train_unimodal(
        MLPClassifier(dim),
        make_loader(x_train, labels=y_train, shuffle=True),
        make_loader(x_val, labels=y_val),
        args.device,
    )
    val_metrics = compute_metrics(*predict(model, make_loader(x_val, labels=y_val), args.device))

    seed_everything(args.seed)
    final, _ = train_unimodal(
        MLPClassifier(dim),
        make_loader(np.vstack([x_train, x_val]), labels=np.concatenate([y_train, y_val]), shuffle=True),
        None,
        args.device,
    )
    labels, probs = predict(final, make_loader(x_test, labels=y_test), args.device)
    test_metrics = compute_metrics(labels, probs)

    out = Path(args.out_dir)
    for sub in ("checkpoints", "predictions", "results"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "unimodal",
            "modality": args.modality,
            "encoder": args.encoder,
            "embed_dim": dim,
            "state_dict": final.state_dict(),
            "test_metrics": test_metrics,
        },
        out / "checkpoints" / f"{name}.pt",
    )
    pd.DataFrame({"id": test["id"], "label": labels, "prob": probs, "pred": (probs > 0.5).astype(int)}).to_csv(
        out / "predictions" / f"{name}.csv", index=False
    )
    result = {
        "experiment": name,
        "encoder": ENCODERS[args.modality][args.encoder]["name"],
        "val": val_metrics,
        "test": test_metrics,
        "history": history,
    }
    (out / "results" / f"{name}.json").write_text(json.dumps(result, indent=2))

    print(f"{result['encoder']}")
    for split, m in (("val", val_metrics), ("test", test_metrics)):
        print(f"  {split:4s} " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))


if __name__ == "__main__":
    main()
