#!/usr/bin/env bash
# Fine-tune RF-DETR (Medium) on the converted COCO-format tape-measure front-side
# detection dataset. Run convert_to_coco.py first if the dataset dir is missing.
# Run from the project root.
set -e
set -x

DATASET_DIR='data/internal_train/rfdetr_coco_front'
OUTPUT_DIR='outputs/rfdetr_medium_front'

python -m subtasks.data_preprocessing_det.scripts.train_rfdetr \
    --dataset-dir "${DATASET_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --epochs 40 \
    --batch-size 4 \
    --grad-accum-steps 4
