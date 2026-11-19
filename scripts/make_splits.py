"""Regenerate splits/{train,val,test}.txt from dataset.csv (stratified 70/10/20, seed 66)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gamblingdet.data import SPLIT_DIR, SPLITS, load_dataset, split_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default=str(SPLIT_DIR))
    args = parser.parse_args()

    frames = split_dataset(load_dataset(args.data_dir, check_images=False))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(SPLITS, frames):
        (out / f"{name}.txt").write_text("\n".join(map(str, frame["id"])) + "\n")
        print(f"{name}: {len(frame)} samples, {int(frame['label'].sum())} gambling")


if __name__ == "__main__":
    main()
