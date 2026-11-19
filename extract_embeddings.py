"""Extract frozen-encoder embeddings for every sample in dataset.csv.

Example:
    python extract_embeddings.py --modality image --encoders siglip2 clip dinov3
    python extract_embeddings.py --modality text --encoders e5 indobert qwen3
"""

import argparse
from pathlib import Path

import torch

from gamblingdet.config import ENCODERS
from gamblingdet.data import load_dataset
from gamblingdet.embeddings import extract_image_embeddings, extract_text_embeddings, save_embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modality", choices=["image", "text"], required=True)
    parser.add_argument("--encoders", nargs="+", help="default: all encoders of the modality")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--emb-dir", default="embeddings")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    df = load_dataset(args.data_dir, check_images=args.modality == "image")
    out_dir = Path(args.emb_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for key in args.encoders or list(ENCODERS[args.modality]):
        if key not in ENCODERS[args.modality]:
            parser.error(f"unknown {args.modality} encoder '{key}'; choose from {list(ENCODERS[args.modality])}")
        path = out_dir / f"{args.modality}_{key}.npz"
        if path.exists() and not args.overwrite:
            print(f"skip {path} (exists; use --overwrite)")
            continue
        if args.modality == "image":
            emb = extract_image_embeddings(key, df["image_path"].tolist(), args.device)
        else:
            emb = extract_text_embeddings(key, df["text_caption"].tolist(), args.device)
        save_embeddings(path, df["id"].values, emb)
        print(f"saved {path} {emb.shape}")


if __name__ == "__main__":
    main()
