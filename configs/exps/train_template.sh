#!/bin/bash
# Training script template for PyTorch Lightning project
# Usage: ./configs/exps/train_template.sh [config_path]

set -e
set -x

CONFIG_PATH='configs/yamls/template_config.yaml'

# Run training with poetry
poetry run python -m project_src.main \
    --yml-config "${CONFIG_PATH}" \
    --task train
