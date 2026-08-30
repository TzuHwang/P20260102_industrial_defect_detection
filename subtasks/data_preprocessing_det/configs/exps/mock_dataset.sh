#!/usr/bin/env bash
# Generate the synthetic mock detection dataset.
# Run from the project root.
set -e
set -x

python -m subtasks.data_preprocessing_det.scripts.mock_dataset
