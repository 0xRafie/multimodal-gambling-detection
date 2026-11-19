"""Dataset loading and the stratified 70/10/20 split."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import SEED, TEST_RATIO, TRAIN_RATIO, VAL_RATIO


def load_dataset(data_dir, check_images=True):
    """Read `dataset.csv` and resolve image paths relative to `data_dir`.

    Expected layout (as published on Hugging Face):
        data_dir/dataset.csv
        data_dir/image/image_1.jpg ...
    """
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "dataset.csv")
    df["text_caption"] = df["text_caption"].fillna("")
    df["label"] = df["label"].astype(int)
    df["image_path"] = df["image_path"].map(lambda p: str(data_dir / p))
    if not check_images:
        return df

    missing = [p for p in df["image_path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image(s) listed in dataset.csv are missing, e.g. {missing[0]}. "
            "Run scripts/download_data.sh first."
        )
    return df


SPLIT_DIR = Path(__file__).resolve().parent.parent / "splits"
SPLITS = ("train", "val", "test")


def load_splits(df, split_dir=SPLIT_DIR):
    """Return (train, val, test) frames using the sample ids stored in `splits/*.txt`.

    The id files were produced by `split_dataset` (see scripts/make_splits.py) and are the
    exact split used in the paper; shipping them guards against changes in scikit-learn.
    """
    by_id = df.set_index("id")
    frames = []
    for name in SPLITS:
        ids = [int(x) for x in (Path(split_dir) / f"{name}.txt").read_text().split()]
        frames.append(by_id.loc[ids].reset_index())
    return tuple(frames)


def split_dataset(df, seed=SEED):
    """Stratified split: hold out 20% test first, then split the rest into 70/10 of the total."""
    trainval, test = train_test_split(df, test_size=TEST_RATIO, stratify=df["label"], random_state=seed)
    train, val = train_test_split(
        trainval,
        test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
        stratify=trainval["label"],
        random_state=seed,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
