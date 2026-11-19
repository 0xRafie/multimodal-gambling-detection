"""Frozen-encoder feature extraction (Table "Embedding configuration" in the paper).

Image: DINOv3-H+ (CLS / pooler), CLIP-ViT-L/14-336 (projected), SigLIP 2 So400m (MAP head).
Text:  Indonesian BERT (mean pooling), Qwen3-Embedding-0.6B (last token + L2),
       Multilingual E5-large ("query: " prefix, mean pooling + L2).
"""

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .config import ENCODERS, MAX_LENGTH


def _load_image_encoder(key, device):
    from transformers import (
        AutoImageProcessor,
        AutoModel,
        AutoProcessor,
        CLIPModel,
        CLIPProcessor,
    )

    model_id = ENCODERS["image"][key]["model_id"]
    if key == "clip":
        processor, model = CLIPProcessor.from_pretrained(model_id), CLIPModel.from_pretrained(model_id)
    elif key == "dinov3":
        processor, model = AutoImageProcessor.from_pretrained(model_id), AutoModel.from_pretrained(model_id)
    elif key == "siglip2":
        processor, model = AutoProcessor.from_pretrained(model_id), AutoModel.from_pretrained(model_id)
    else:
        raise ValueError(f"Unknown image encoder: {key}")
    return processor, model.eval().to(device)


def _image_features(key, model, inputs):
    if key == "dinov3":
        out = model(**inputs)
        if getattr(out, "pooler_output", None) is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0]
    # CLIP: projected embedding; SigLIP 2: MAP-head pooled embedding.
    return model.get_image_features(**inputs)


@torch.no_grad()
def extract_image_embeddings(key, image_paths, device):
    """One image at a time, as in the paper's runs. Unreadable images get a zero vector."""
    processor, model = _load_image_encoder(key, device)
    dim = ENCODERS["image"][key]["dim"]
    features, failed = [], 0
    for path in tqdm(image_paths, desc=f"image/{key}"):
        try:
            inputs = processor(images=Image.open(path).convert("RGB"), return_tensors="pt").to(device)
            features.append(_image_features(key, model, inputs).squeeze(0).float().cpu().numpy())
        except Exception as err:  # noqa: BLE001 - as in the paper's runs, keep alignment with a zero vector
            warnings.warn(f"Could not embed {path}: {err}; using a zero vector", stacklevel=2)
            features.append(np.zeros(dim, dtype=np.float32))
            failed += 1
    if failed:
        warnings.warn(f"{failed} image(s) replaced by zero vectors")
    return np.stack(features)


def _mean_pool(last_hidden, mask):
    mask = mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def _last_token_pool(last_hidden, mask):
    if mask[:, -1].sum() == mask.shape[0]:  # left padding
        return last_hidden[:, -1]
    lengths = mask.sum(dim=1) - 1
    return last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), lengths]


@torch.no_grad()
def extract_text_embeddings(key, texts, device):
    from transformers import AutoModel, AutoTokenizer

    model_id = ENCODERS["text"][key]["model_id"]
    remote = key == "qwen3"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=remote)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=remote).eval().to(device)

    texts = list(texts)
    if key == "e5":
        texts = ["query: " + t for t in texts]
    batch_size = 8 if key == "qwen3" else 32

    features = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"text/{key}"):
        batch = tokenizer(
            texts[i : i + batch_size],
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        hidden = model(**batch).last_hidden_state
        if key == "qwen3":
            emb = F.normalize(_last_token_pool(hidden, batch["attention_mask"]), p=2, dim=1)
        elif key == "e5":
            emb = F.normalize(_mean_pool(hidden, batch["attention_mask"]), p=2, dim=1)
        else:  # indobert: plain mean pooling, no normalisation
            emb = _mean_pool(hidden, batch["attention_mask"])
        features.append(emb.float().cpu().numpy())
    return np.concatenate(features)


def save_embeddings(path, ids, emb):
    np.savez(path, ids=np.asarray(ids), emb=emb.astype(np.float32))


def load_embeddings(path, ids):
    """Load cached embeddings and return them in the order of `ids`."""
    cache = np.load(path)
    index = {int(i): n for n, i in enumerate(cache["ids"])}
    return cache["emb"][[index[int(i)] for i in ids]]
