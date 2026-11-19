#!/usr/bin/env bash
# Reproduce every table of the paper: embeddings -> unimodal heads -> fusion strategies.
# DINOv3 is a gated model: accept its licence on Hugging Face and run `hf auth login` first.
set -euo pipefail

python extract_embeddings.py --modality image --encoders dinov3 clip siglip2
python extract_embeddings.py --modality text --encoders indobert qwen3 e5

for enc in dinov3 clip siglip2; do python train_unimodal.py --modality image --encoder "$enc"; done
for enc in indobert qwen3 e5; do python train_unimodal.py --modality text --encoder "$enc"; done

python train_fusion.py
