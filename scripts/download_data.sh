#!/usr/bin/env bash
# Download the dataset (dataset.csv + 5,028 images, ~0.9 GB) from Hugging Face into ./data
set -euo pipefail
hf download 0xRafie/indonesian-gambling-multimodal-dataset --repo-type dataset --local-dir "${1:-data}"
