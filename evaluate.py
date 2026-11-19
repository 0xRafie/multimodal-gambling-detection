"""Evaluate a saved checkpoint on the test split using cached embeddings.

Example:
    python evaluate.py --checkpoint checkpoints/image_siglip2.pt
    python evaluate.py --checkpoint checkpoints/fusion_contrastive.pt
"""

import argparse
from pathlib import Path

import torch

from gamblingdet.data import load_dataset, load_splits
from gamblingdet.embeddings import load_embeddings
from gamblingdet.engine import compute_metrics, make_loader, predict
from gamblingdet.models import FUSION_MODELS, MLPClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--emb-dir", default="embeddings")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    test = load_splits(load_dataset(args.data_dir, check_images=False))[2]
    emb_dir = Path(args.emb_dir)

    if ckpt["kind"] == "unimodal":
        model = MLPClassifier(ckpt["embed_dim"])
        x = load_embeddings(emb_dir / f"{ckpt['modality']}_{ckpt['encoder']}.npz", test["id"])
        loader = make_loader(x, labels=test["label"].values)
    else:
        model = FUSION_MODELS[ckpt["fusion"]](ckpt["img_dim"], ckpt["txt_dim"])
        img = load_embeddings(emb_dir / f"image_{ckpt['image_encoder']}.npz", test["id"])
        txt = load_embeddings(emb_dir / f"text_{ckpt['text_encoder']}.npz", test["id"])
        loader = make_loader(img, txt, labels=test["label"].values)

    model.load_state_dict(ckpt["state_dict"])
    metrics = compute_metrics(*predict(model, loader, args.device))
    print(Path(args.checkpoint).stem)
    print("  test " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
