"""Hyperparameters and encoder registry used for every experiment in the paper."""

SEED = 66
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 5
LR = 1e-3
HIDDEN_DIM = 128
DROPOUT = 0.3
PROJ_DIM = 512
MAX_LENGTH = 512
THRESHOLD = 0.5

TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

IMAGE_ENCODERS = {
    "dinov3": {
        "name": "DINOv3-H+",
        "model_id": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        "dim": 1280,
    },
    "clip": {
        "name": "CLIP-ViT-L/14",
        "model_id": "openai/clip-vit-large-patch14-336",
        "dim": 768,
    },
    "siglip2": {
        "name": "SigLIP 2",
        "model_id": "google/siglip2-so400m-patch14-384",
        "dim": 1152,
    },
}

TEXT_ENCODERS = {
    "indobert": {
        "name": "Indonesian BERT (Cahya)",
        "model_id": "cahya/bert-base-indonesian-1.5G",
        "dim": 768,
    },
    "qwen3": {
        "name": "Qwen3-Embedding-0.6B",
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "dim": 1024,
    },
    "e5": {
        "name": "Multilingual E5-large",
        "model_id": "intfloat/multilingual-e5-large",
        "dim": 1024,
    },
}

ENCODERS = {"image": IMAGE_ENCODERS, "text": TEXT_ENCODERS}

# Fusion experiments use the best unimodal encoders (Section "Multimodal Classification").
FUSION_IMAGE_ENCODER = "siglip2"
FUSION_TEXT_ENCODER = "e5"
FUSIONS = ["concat", "average", "weighted", "stacking", "projection", "contrastive"]
