"""CPU smoke tests: split files, model shapes, and training on synthetic embeddings.

Set GD_DATA_DIR to a folder with dataset.csv to also check that the split regenerates exactly.
"""

import os

import numpy as np
import pandas as pd
import pytest
import torch

from gamblingdet.config import FUSIONS
from gamblingdet.data import SPLIT_DIR, SPLITS, load_dataset, load_splits, split_dataset
from gamblingdet.embeddings import load_embeddings, save_embeddings
from gamblingdet.engine import compute_metrics, make_loader, predict, seed_everything, train_fusion, train_unimodal
from gamblingdet.models import FUSION_MODELS, MLPClassifier


def read_ids(name):
    return [int(x) for x in (SPLIT_DIR / f"{name}.txt").read_text().split()]


def test_split_files_are_disjoint_and_complete():
    ids = {name: read_ids(name) for name in SPLITS}
    assert [len(ids[n]) for n in SPLITS] == [3519, 503, 1006]
    union = set().union(*ids.values())
    assert len(union) == 5028 and union == set(range(1, 5029))


@pytest.mark.skipif(not os.environ.get("GD_DATA_DIR"), reason="GD_DATA_DIR not set")
def test_split_regenerates_from_dataset():
    df = load_dataset(os.environ["GD_DATA_DIR"], check_images=False)
    for frame, name in zip(split_dataset(df), SPLITS):
        assert frame["id"].tolist() == read_ids(name)
    test = load_splits(df)[2]
    assert int(test["label"].sum()) == 439  # 419 TP + 20 FN in the paper's confusion matrix


@pytest.mark.parametrize("fusion", FUSIONS)
def test_fusion_output_shape(fusion):
    model = FUSION_MODELS[fusion](1152, 1024)
    assert model(torch.randn(4, 1152), torch.randn(4, 1024)).shape == (4, 1)


def test_weighted_fusion_starts_at_equal_weights():
    w = torch.softmax(FUSION_MODELS["weighted"](8, 8).weights, dim=0)
    assert torch.allclose(w, torch.tensor([0.5, 0.5]))


def test_contrastive_temperature_init():
    tau = FUSION_MODELS["contrastive"](8, 8).logit_scale.exp().item()
    assert tau == pytest.approx(1 / 0.07)


def synthetic(n, dim, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    x = rng.normal(size=(n, dim)).astype(np.float32) + y[:, None] * 0.8
    return x, y


def test_unimodal_training_learns_separable_data():
    seed_everything()
    x, y = synthetic(600, 32, 0)
    xv, yv = synthetic(200, 32, 1)
    model, history = train_unimodal(
        MLPClassifier(32), make_loader(x, labels=y, shuffle=True), make_loader(xv, labels=yv), "cpu"
    )
    assert history["val_loss"]
    assert compute_metrics(*predict(model, make_loader(xv, labels=yv), "cpu"))["f1"] > 0.9


def test_unimodal_without_validation_runs_all_epochs():
    x, y = synthetic(64, 8, 2)
    _, history = train_unimodal(MLPClassifier(8), make_loader(x, labels=y), None, "cpu", epochs=7)
    assert len(history["train_loss"]) == 7 and not history["val_loss"]


@pytest.mark.parametrize("fusion", FUSIONS)
def test_fusion_training_learns_separable_data(fusion):
    seed_everything()
    xi, y = synthetic(600, 24, 3)
    xt = xi[:, :16] + np.random.default_rng(4).normal(size=(600, 16)).astype(np.float32)
    xvi, yv = synthetic(200, 24, 5)
    xvt = xvi[:, :16] + np.random.default_rng(6).normal(size=(200, 16)).astype(np.float32)
    model, _ = train_fusion(
        FUSION_MODELS[fusion](24, 16),
        make_loader(xi, xt, labels=y, shuffle=True),
        make_loader(xvi, xvt, labels=yv),
        "cpu",
    )
    assert compute_metrics(*predict(model, make_loader(xvi, xvt, labels=yv), "cpu"))["f1"] > 0.85


def test_embedding_cache_reorders_by_id(tmp_path):
    ids = np.array([10, 20, 30])
    emb = np.arange(6, dtype=np.float32).reshape(3, 2)
    save_embeddings(tmp_path / "e.npz", ids, emb)
    assert load_embeddings(tmp_path / "e.npz", pd.Series([30, 10])).tolist() == [[4, 5], [0, 1]]
