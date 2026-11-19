"""Training and evaluation loops (BCE-with-logits, Adam 1e-3, ReduceLROnPlateau(0.5, patience 3))."""

import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import BATCH_SIZE, EPOCHS, LR, PATIENCE, SEED, THRESHOLD


def seed_everything(seed=SEED):
    """Seed all RNGs and force deterministic kernels (as in the paper's fusion runs)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_loader(*arrays, labels, shuffle=False):
    tensors = [torch.tensor(a, dtype=torch.float32) for a in arrays]
    return DataLoader(TensorDataset(*tensors, torch.tensor(labels)), batch_size=BATCH_SIZE, shuffle=shuffle)


def _step(model, batch, device):
    *inputs, y = batch
    return model(*[x.to(device) for x in inputs]), y.to(device).float().unsqueeze(1)


def train_unimodal(model, train_loader, val_loader, device, epochs=EPOCHS, patience=PATIENCE):
    """Early stopping on validation loss. With `val_loader=None` it trains for all `epochs`."""
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    best_loss, best_state, wait = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}

    for _ in range(epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            out, y = _step(model, batch, device)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        history["train_loss"].append(total / len(train_loader))

        if val_loader is None:
            continue
        model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                out, y = _step(model, batch, device)
                total += criterion(out, y).item()
        val_loss = total / len(val_loader)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss, wait = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_fusion(model, train_loader, val_loader, device, epochs=EPOCHS, patience=PATIENCE):
    """Early stopping and LR scheduling on validation F1; keeps the best-F1 weights."""
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    best_f1, best_state, wait = 0.0, None, 0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for _ in range(epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            out, y = _step(model, batch, device)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        history["train_loss"].append(total / len(train_loader))

        model.eval()
        total, preds, labels = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                out, y = _step(model, batch, device)
                total += criterion(out, y).item()
                preds.extend((torch.sigmoid(out) > THRESHOLD).int().flatten().tolist())
                labels.extend(y.int().flatten().tolist())
        val_f1 = f1_score(labels, preds)
        history["val_loss"].append(total / len(val_loader))
        history["val_f1"].append(val_f1)
        scheduler.step(1 - val_f1)
        if val_f1 > best_f1:
            best_f1, wait = val_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict(model, loader, device):
    model = model.to(device).eval()
    probs, labels = [], []
    for batch in loader:
        out, y = _step(model, batch, device)
        probs.append(torch.sigmoid(out).flatten().cpu().numpy())
        labels.append(y.flatten().cpu().numpy())
    return np.concatenate(labels).astype(int), np.concatenate(probs)


def compute_metrics(labels, probs, threshold=THRESHOLD):
    preds = (probs > threshold).astype(int)
    metrics = {
        "recall": recall_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "accuracy": accuracy_score(labels, preds),
        "auc": roc_auc_score(labels, probs),
    }
    return {k: float(v) for k, v in metrics.items()}
