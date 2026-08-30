#!/usr/bin/env bash
# Run detection training on the synthetic mock dataset.
# Run mock_dataset.sh first if data/test/mock_det/ does not exist.
# Run from the project root.
set -e
set -x

python -m project_src.main \
    --yml-config configs/yamls/internal_train/det/tape_measure_det_mock.yaml \
    --task train
